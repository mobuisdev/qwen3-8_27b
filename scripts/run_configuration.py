#!/usr/bin/env python3
"""Launch one SGLang configuration and run repeatable single-user workloads.

This is deliberately self-contained so the command, startup log, server-info
snapshot, per-request raw response metadata, hardware samples, and normalized
CSV/JSON rows always share one run identifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import pynvml
import requests
import sglang
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
RESULT_FIELDS = (ROOT / "benchmark_results.csv").read_text().splitlines()[0].split(",")
REPOSITORIES = json.loads((ROOT / "models" / "repositories.json").read_text())


def model_snapshot_path(repo: dict[str, Any]) -> Path:
    """Resolve a pinned Hub snapshot without a machine-specific manifest."""
    cache_name = f"models--{repo['repo_id'].replace('/', '--')}"
    path = ROOT / "hf-home" / "hub" / cache_name / "snapshots" / repo["resolved_revision"]
    if not path.is_dir():
        raise SystemExit(
            f"Pinned model snapshot is missing: {path}\n"
            f"Run {ROOT / '.venv' / 'bin' / 'python'} "
            f"{ROOT / 'scripts' / 'download_models.py'} first."
        )
    return path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 4)


def process_tree_rss(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss + sum(
            child.memory_info().rss for child in process.children(recursive=True)
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


class HardwareMonitor:
    FIELDS = [
        "timestamp",
        "monotonic",
        "phase",
        "gpu_memory_used_bytes",
        "gpu_memory_free_bytes",
        "gpu_utilization_percent",
        "gpu_power_w",
        "gpu_temperature_c",
        "gpu_graphics_clock_mhz",
        "gpu_memory_clock_mhz",
        "pcie_tx_kib_s",
        "pcie_rx_kib_s",
        "host_used_bytes",
        "host_available_bytes",
        "swap_used_bytes",
        "swap_sin_bytes",
        "swap_sout_bytes",
        "server_tree_rss_bytes",
    ]

    def __init__(self, pid: int, path: Path, interval: float = 0.5):
        self.pid = pid
        self.path = path
        self.interval = interval
        self.phase = "startup"
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        pynvml.nvmlInit()
        self.device = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    def _sample(self) -> dict[str, Any]:
        memory = pynvml.nvmlDeviceGetMemoryInfo(self.device)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.device)
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        try:
            tx = pynvml.nvmlDeviceGetPcieThroughput(
                self.device, pynvml.NVML_PCIE_UTIL_TX_BYTES
            )
            rx = pynvml.nvmlDeviceGetPcieThroughput(
                self.device, pynvml.NVML_PCIE_UTIL_RX_BYTES
            )
        except pynvml.NVMLError:
            tx = rx = None
        with self.lock:
            phase = self.phase
        return {
            "timestamp": now_iso(),
            "monotonic": time.monotonic(),
            "phase": phase,
            "gpu_memory_used_bytes": memory.used,
            "gpu_memory_free_bytes": memory.free,
            "gpu_utilization_percent": utilization.gpu,
            "gpu_power_w": round(pynvml.nvmlDeviceGetPowerUsage(self.device) / 1000, 3),
            "gpu_temperature_c": pynvml.nvmlDeviceGetTemperature(
                self.device, pynvml.NVML_TEMPERATURE_GPU
            ),
            "gpu_graphics_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                self.device, pynvml.NVML_CLOCK_GRAPHICS
            ),
            "gpu_memory_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                self.device, pynvml.NVML_CLOCK_MEM
            ),
            "pcie_tx_kib_s": tx,
            "pcie_rx_kib_s": rx,
            "host_used_bytes": vm.used,
            "host_available_bytes": vm.available,
            "swap_used_bytes": swap.used,
            "swap_sin_bytes": swap.sin,
            "swap_sout_bytes": swap.sout,
            "server_tree_rss_bytes": process_tree_rss(self.pid),
        }

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            while not self.stop_event.is_set():
                try:
                    sample = self._sample()
                    self.samples.append(sample)
                    writer.writerow(sample)
                    handle.flush()
                except Exception as exc:  # monitoring must not kill a benchmark
                    writer.writerow({"timestamp": now_iso(), "phase": f"monitor_error:{exc}"})
                    handle.flush()
                self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        pynvml.nvmlShutdown()

    def interval_summary(self, start: float, end: float) -> dict[str, Any]:
        rows = [s for s in self.samples if start <= s["monotonic"] <= end]
        if not rows:
            rows = [self._sample()]
        return {
            "peak_vram_gb": gib(max(s["gpu_memory_used_bytes"] for s in rows)),
            "min_vram_free_gb": gib(min(s["gpu_memory_free_bytes"] for s in rows)),
            "host_ram_peak_gb": gib(max(s["host_used_bytes"] for s in rows)),
            "swap_peak_gb": gib(max(s["swap_used_bytes"] for s in rows)),
            "gpu_peak_power_w": max(s["gpu_power_w"] for s in rows),
            "gpu_peak_temperature_c": max(s["gpu_temperature_c"] for s in rows),
            "server_tree_rss_peak_gb": gib(max(s["server_tree_rss_bytes"] for s in rows)),
            "swap_in_delta_bytes": max(s["swap_sin_bytes"] for s in rows)
            - min(s["swap_sin_bytes"] for s in rows),
            "swap_out_delta_bytes": max(s["swap_sout_bytes"] for s in rows)
            - min(s["swap_sout_bytes"] for s in rows),
        }


def flush_cache(base_url: str) -> None:
    response = requests.post(f"{base_url}/flush_cache", timeout=60)
    response.raise_for_status()


def generate_ids(tokenizer: Any, target: int, nonce: int) -> list[int]:
    prefix = tokenizer.encode(
        f"Benchmark run {nonce}. Read the following technical notes carefully. ",
        add_special_tokens=False,
    )
    body = tokenizer.encode(
        "The system is a deterministic inference benchmark. Each paragraph contains "
        "ordinary prose about software, mathematics, hardware, history, and careful "
        "measurement. Preserve the sequence and continue with a clear structured answer. ",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "\nNow produce a numbered technical essay with many distinct points and no early conclusion.",
        add_special_tokens=False,
    )
    if target < len(prefix) + len(suffix):
        raise ValueError(f"Target {target} is too short")
    count = target - len(prefix) - len(suffix)
    return prefix + (body * (count // len(body) + 1))[:count] + suffix


def stream_request(
    base_url: str,
    input_ids: list[int],
    max_new_tokens: int,
    seed: int,
    ignore_eos: bool = True,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": ignore_eos,
            "sampling_seed": seed,
        },
        "stream": True,
    }
    started = time.perf_counter()
    first_token_at = None
    last = None
    response = requests.post(
        f"{base_url}/generate", json=payload, stream=True, timeout=(60, 3600)
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        if line.strip() == "data: [DONE]":
            break
        last = json.loads(line[5:].strip())
        completion = (last.get("meta_info") or {}).get("completion_tokens", 0)
        if completion and first_token_at is None:
            first_token_at = time.perf_counter()
    ended = time.perf_counter()
    if last is None:
        raise RuntimeError("Streaming response contained no JSON events")
    meta = last.get("meta_info") or {}
    prompt_tokens = int(meta.get("prompt_tokens", len(input_ids)))
    completion_tokens = int(meta.get("completion_tokens", 0))
    ttft = (first_token_at or ended) - started
    total = ended - started
    decode = max(0.0, ended - (first_token_at or ended))
    return {
        "client_started_monotonic": started,
        "client_ended_monotonic": ended,
        "actual_prompt_tokens": prompt_tokens,
        "generated_tokens": completion_tokens,
        "ttft_seconds": ttft,
        "prefill_seconds": ttft,
        "prefill_tokens_per_second": prompt_tokens / ttft if ttft else None,
        "decode_seconds": decode,
        "decode_tokens_per_second": (
            (completion_tokens - 1) / decode if completion_tokens > 1 and decode else None
        ),
        "total_seconds": total,
        "server_e2e_seconds": meta.get("e2e_latency"),
        "server_decode_tokens_per_second": meta.get("decode_throughput"),
        "speculative_tokens_proposed": meta.get("spec_num_proposed_drafts"),
        "speculative_tokens_accepted": meta.get("spec_num_correct_drafts"),
        "speculative_acceptance_rate": meta.get("spec_accept_rate"),
        "average_accepted_tokens_per_step": meta.get("spec_accept_length"),
        "spec_verify_count": meta.get("spec_verify_ct"),
        "finish_reason": meta.get("finish_reason"),
        "output_text": last.get("text"),
        "meta_info": meta,
    }


def run_quality_suite(
    base_url: str,
    tokenizer: Any,
    context_sizes: list[int],
    context_limit: int,
    monitor: HardwareMonitor,
) -> list[dict[str, Any]]:
    small_cases = [
        {
            "id": "logic",
            "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep remain? Explain briefly.",
            "expected_terms": ["9"],
        },
        {
            "id": "coding",
            "prompt": (
                "Write a Python function balanced_brackets(text: str) -> bool for (), [], and {}. "
                "Ignore non-bracket characters, reject mismatched nesting, and include three asserts."
            ),
            "expected_terms": ["def balanced_brackets", "stack", "assert"],
        },
        {
            "id": "instruction",
            "prompt": (
                'Return exactly one minified JSON object with keys "alpha" and "beta" in that order; '
                'their integer values must be 2 and 3. Do not use a Markdown fence or add commentary.'
            ),
            "expected_terms": ['{"alpha":2,"beta":3}'],
        },
        {
            "id": "factual",
            "prompt": "In two short paragraphs, explain why Earth's daytime sky is blue and sunsets are often red.",
            "expected_terms": ["Rayleigh", "wavelength"],
        },
    ]
    outcomes: list[dict[str, Any]] = []
    for index, case in enumerate(small_cases):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        # Transformers 5 may return a BatchEncoding here even without tensor
        # output. SGLang's JSON API requires a plain one-dimensional int list.
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(token_id) for token_id in ids]
        flush_cache(base_url)
        monitor.set_phase(f"quality_{case['id']}")
        result = stream_request(base_url, ids, 384, 7000 + index, ignore_eos=False)
        output = result.get("output_text") or ""
        outcomes.append(
            {
                "case": case,
                "result": result,
                "automatic_term_check": all(
                    term.lower() in output.lower() for term in case["expected_terms"]
                ),
            }
        )

    needle_text = "The cobalt falcon access code is 739184."
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    question_ids = tokenizer.encode(
        "\nQuestion: What is the cobalt falcon access code? Answer with the digits and one short sentence.",
        add_special_tokens=False,
    )
    filler_ids = tokenizer.encode(
        "Archive note: routine measurements were checked and filed without any access codes. ",
        add_special_tokens=False,
    )
    for target in context_sizes:
        if target + 128 > context_limit:
            continue
        for position in (0.10, 0.50, 0.90):
            available = target - len(needle_ids) - len(question_ids)
            filler = (filler_ids * (available // len(filler_ids) + 1))[:available]
            insertion = int(len(filler) * position)
            ids = filler[:insertion] + needle_ids + filler[insertion:] + question_ids
            flush_cache(base_url)
            monitor.set_phase(f"needle_{target}_{int(position * 100)}")
            result = stream_request(
                base_url, ids, 128, 8000 + target + int(position * 100), ignore_eos=False
            )
            output = result.get("output_text") or ""
            outcomes.append(
                {
                    "case": {
                        "id": "needle",
                        "target_tokens": target,
                        "needle_position": position,
                        "needle": needle_text,
                    },
                    "result": result,
                    "automatic_term_check": "739184" in output,
                }
            )
    return outcomes


def server_command(args: argparse.Namespace, model_path: str) -> list[str]:
    server = [
        str(ROOT / ".venv" / "bin" / "sglang"),
        "serve",
        "--model-path",
        model_path,
        "--served-model-name",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--context-length",
        str(args.context),
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--chunked-prefill-size",
        str(args.chunk),
        "--attention-backend",
        args.backend,
        "--disable-flashinfer-autotune",
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--max-running-requests",
        "1",
        "--cpu-offload-gb",
        "0",
        "--mamba-ssm-dtype",
        args.mamba_dtype,
        "--mamba-radix-cache-strategy",
        args.mamba_strategy,
        "--skip-server-warmup",
        "--watchdog-timeout",
        str(args.watchdog_timeout),
        "--enable-metrics",
    ]
    if args.max_mamba_cache_size is not None:
        server += ["--max-mamba-cache-size", str(args.max_mamba_cache_size)]
    if args.cuda_graphs == "disabled":
        server += [
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-backend-decode",
            "disabled",
        ]
    else:
        server += [
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-max-bs-decode",
            "1",
        ]
    if args.mtp:
        server += [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--enable-linear-replayssm-spec",
        ]
    if args.mamba_strategy == "no_buffer":
        # SGLang rejects no_buffer with overlap scheduling because state cannot
        # be preserved safely across overlapped batches.
        server.append("--disable-overlap-schedule")
    # Keep the server (not the desktop, terminal, or benchmark client) inside a
    # hard user cgroup. MemorySwapMax=0 ensures host swap cannot be used to make
    # a model configuration appear viable or destabilize the interactive session.
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryHigh={args.host_memory_high_gb}G",
        "-p",
        f"MemoryMax={args.host_memory_limit_gb}G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=kill",
        "--",
        *server,
    ]


def wait_ready(base_url: str, process: subprocess.Popen, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with status {process.returncode}")
        try:
            # `/health` is a real generation request in this SGLang branch and
            # therefore triggers first-forward Triton compilation. With server
            # warm-up deliberately skipped, `/model_info` is the non-mutating
            # readiness endpoint used by SGLang's own scripted-runtime tests.
            response = requests.get(f"{base_url}/model_info", timeout=2)
            if response.status_code == 200:
                info = requests.get(f"{base_url}/server_info", timeout=10)
                return info.json() if info.ok else {"server_info_error": info.text}
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"Server not ready after {timeout}s: {last_error}")


def append_results(rows: list[dict[str, Any]]) -> None:
    csv_path = ROOT / "benchmark_results.csv"
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in RESULT_FIELDS})
    json_path = ROOT / "benchmark_results.json"
    existing = json.loads(json_path.read_text())
    existing.extend(rows)
    json_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")


def normalized_row(
    args: argparse.Namespace,
    repo: dict,
    request: dict[str, Any] | None,
    run_id: str,
    startup_success: bool,
    model_load: dict[str, Any],
    hardware: dict[str, Any] | None = None,
    note: str = "",
) -> dict[str, Any]:
    request = request or {}
    hardware = hardware or {}
    before = psutil.virtual_memory()
    swap = psutil.swap_memory()
    row = {
        "timestamp": now_iso(),
        "configuration_id": args.config_id,
        "run_id": run_id,
        "workload": args.workload,
        "repetition": request.get("repetition"),
        "model": repo["repo_id"],
        "model_revision": repo["resolved_revision"],
        "runtime": "SGLang source Qwen3.8 PR + lm_head fix backport",
        "runtime_version": getattr(sglang, "__version__", "unknown"),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "driver_version": pynvml.nvmlSystemGetDriverVersion(),
        "mtp_enabled": args.mtp,
        "context_limit": args.context,
        "actual_prompt_tokens": request.get("actual_prompt_tokens"),
        "generated_tokens": request.get("generated_tokens"),
        "kv_cache_dtype": args.kv_cache_dtype,
        "attention_backend": args.backend,
        "chunked_prefill_size": args.chunk,
        "mem_fraction_static": args.mem_fraction,
        "mamba_ssm_dtype": args.mamba_dtype,
        "mamba_radix_cache_strategy": args.mamba_strategy,
        "cuda_graphs": args.cuda_graphs,
        "host_memory_limit_gb": args.host_memory_limit_gb,
        "speculative_algorithm": "EAGLE" if args.mtp else None,
        "speculative_steps": 3 if args.mtp else None,
        "speculative_draft_tokens": 4 if args.mtp else None,
        "startup_success": startup_success,
        "request_success": bool(request),
        "oom": "out of memory" in note.lower() or "cuda oom" in note.lower(),
        "host_oom": "host oom" in note.lower(),
        "model_load_vram_gb": model_load.get("gpu_memory_used_gb"),
        "peak_vram_gb": hardware.get("peak_vram_gb"),
        "vram_free_before_request_gb": request.get("vram_free_before_request_gb"),
        "host_ram_used_before_gb": request.get("host_ram_used_before_gb", gib(before.used)),
        "host_ram_available_before_gb": request.get(
            "host_ram_available_before_gb", gib(before.available)
        ),
        "host_ram_peak_gb": hardware.get("host_ram_peak_gb"),
        "swap_used_before_gb": request.get("swap_used_before_gb", gib(swap.used)),
        "swap_peak_gb": hardware.get("swap_peak_gb"),
        "cpu_offload_detected": False,
        "prefill_seconds": request.get("prefill_seconds"),
        "prefill_tokens_per_second": request.get("prefill_tokens_per_second"),
        "ttft_seconds": request.get("ttft_seconds"),
        "decode_seconds": request.get("decode_seconds"),
        "decode_tokens_per_second": request.get("decode_tokens_per_second"),
        "total_seconds": request.get("total_seconds"),
        "gpu_peak_power_w": hardware.get("gpu_peak_power_w"),
        "gpu_peak_temperature_c": hardware.get("gpu_peak_temperature_c"),
        "speculative_tokens_proposed": request.get("speculative_tokens_proposed"),
        "speculative_tokens_accepted": request.get("speculative_tokens_accepted"),
        "speculative_acceptance_rate": request.get("speculative_acceptance_rate"),
        "average_accepted_tokens_per_step": request.get(
            "average_accepted_tokens_per_step"
        ),
        "configured_context": args.context,
        "runtime_reported_kv_capacity": model_load.get("runtime_reported_kv_capacity"),
        "largest_successful_prompt_tokens": request.get("actual_prompt_tokens"),
        "largest_successful_total_tokens": (
            request.get("actual_prompt_tokens", 0) + request.get("generated_tokens", 0)
            if request
            else None
        ),
        "notes": note,
    }
    # Keep all extra raw fields in JSON; DictWriter intentionally selects its schema.
    row.update({f"raw_{key}": value for key, value in hardware.items()})
    row["server_e2e_seconds"] = request.get("server_e2e_seconds")
    row["server_decode_tokens_per_second"] = request.get(
        "server_decode_tokens_per_second"
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--model", choices=["radixark", "unsloth"], required=True)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--mem-fraction", type=float, default=0.90)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--backend", choices=["flashinfer", "triton"], default="flashinfer")
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["fp8_e4m3", "bfloat16", "auto"],
        default="fp8_e4m3",
    )
    parser.add_argument("--mamba-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--mamba-strategy",
        choices=["no_buffer", "extra_buffer", "extra_buffer_lazy"],
        default="extra_buffer_lazy",
    )
    parser.add_argument("--max-mamba-cache-size", type=int)
    parser.add_argument("--cuda-graphs", choices=["disabled", "decode"], default="disabled")
    parser.add_argument("--host-memory-high-gb", type=int, default=36)
    parser.add_argument("--host-memory-limit-gb", type=int, default=40)
    parser.add_argument(
        "--workload", choices=["startup", "prefill", "decode", "quality"], default="startup"
    )
    parser.add_argument("--prompt-sizes", default="1024")
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--watchdog-timeout", type=int, default=900)
    parser.add_argument(
        "--append-canonical-results",
        action="store_true",
        help="Append this run to tracked benchmark_results.{csv,json}.",
    )
    args = parser.parse_args()

    run_id = f"{args.config_id}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    repo = REPOSITORIES[args.model]
    model_path = str(model_snapshot_path(repo))
    command = server_command(args, model_path)
    command_path = ROOT / "raw_results" / "commands" / f"{run_id}.sh"
    command_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"{run_id}.log"
    monitor_path = ROOT / "monitoring" / f"{run_id}.csv"
    raw_path = ROOT / "raw_results" / f"{run_id}.json"
    base_url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    cuda_wheel_root = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cu13"
    )
    cuda_include = str(cuda_wheel_root / "include")
    cuda_lib = str(cuda_wheel_root / "lib")
    # The host has a complete CUDA 13.2 compiler/runtime development install
    # except for cuRAND headers. The PyTorch wheel carries CUDA 13.0 headers;
    # adding that entire directory to CPATH mixes 13.0 headers with nvcc 13.2.
    # Expose only the ABI-compatible cuRAND headers through a tiny shim so all
    # core CUDA/CCCL headers continue to come from the matching host toolkit.
    cuda_header_shim = ROOT / "cuda-header-shim"
    cuda_header_shim.mkdir(exist_ok=True)
    for source in (Path(cuda_include)).glob("curand*"):
        target = cuda_header_shim / source.name
        if not target.exists():
            target.symlink_to(source)
    env.update(
        {
            "HF_HOME": str(ROOT / "hf-home"),
            "SGLANG_CACHE_DIR": str(ROOT / "sglang-cache"),
            "FLASHINFER_WORKSPACE_BASE": str(ROOT / "sglang-cache"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            # First-forward compilation can otherwise fan out to 32 Inductor
            # workers plus an unconstrained Ninja build on this 32-thread host.
            # Keep compilation serial so a cache miss cannot exhaust host RAM
            # and take down the interactive desktop.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "MAX_JOBS": "1",
            "CPATH": os.pathsep.join(
                filter(None, [str(cuda_header_shim), env.get("CPATH")])
            ),
            "LIBRARY_PATH": os.pathsep.join(
                filter(None, [cuda_lib, env.get("LIBRARY_PATH")])
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(
                filter(None, [cuda_lib, env.get("LD_LIBRARY_PATH")])
            ),
        }
    )
    # Record the complete isolated build environment required by FlashInfer's
    # one-time SM120 JIT compilation.
    transcript_keys = (
        "HF_HOME",
        "SGLANG_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "MAX_JOBS",
        "CPATH",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    )
    transcript = ["#!/usr/bin/env bash", "set -euo pipefail"]
    transcript.extend(
        f"export {key}={shlex.quote(env[key])}" for key in transcript_keys
    )
    transcript.extend((shlex.join(command), ""))
    command_path.write_text("\n".join(transcript))
    command_path.chmod(0o755)
    process = None
    monitor = None
    log_handle = log_path.open("w")
    startup_success = False
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {
        "run_id": run_id,
        "arguments": vars(args),
        "model": repo,
        "model_path": model_path,
        "command": command,
        "started": now_iso(),
        "requests": [],
    }
    model_load: dict[str, Any] = {}
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        monitor = HardwareMonitor(process.pid, monitor_path)
        monitor.start()
        raw["server_info"] = wait_ready(base_url, process, args.startup_timeout)
        startup_success = True
        monitor.set_phase("loaded_idle")
        time.sleep(2)
        idle = monitor.samples[-1]
        model_load = {
            "gpu_memory_used_gb": gib(idle["gpu_memory_used_bytes"]),
            "gpu_memory_free_gb": gib(idle["gpu_memory_free_bytes"]),
            "host_used_gb": gib(idle["host_used_bytes"]),
            "host_available_gb": gib(idle["host_available_bytes"]),
            "swap_used_gb": gib(idle["swap_used_bytes"]),
            "server_tree_rss_gb": gib(idle["server_tree_rss_bytes"]),
            "runtime_reported_kv_capacity": (
                next(iter(raw["server_info"].get("internal_states", [])), {})
                .get("memory_usage", {})
                .get("token_capacity")
            ),
        }
        raw["model_load"] = model_load

        if args.workload != "startup":
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
            # Warm-up is deliberately not reported as a measured result.
            flush_cache(base_url)
            warm_ids = generate_ids(tokenizer, min(256, args.context - 32), 0)
            monitor.set_phase("warmup")
            raw["warmup"] = stream_request(base_url, warm_ids, 16, 1)

            prompt_sizes = [int(item) for item in args.prompt_sizes.split(",") if item]
            if args.workload == "quality":
                raw["quality_results"] = run_quality_suite(
                    base_url, tokenizer, prompt_sizes, args.context, monitor
                )
            for prompt_size in ([] if args.workload == "quality" else prompt_sizes):
                if prompt_size + args.output_tokens > args.context:
                    raise ValueError(
                        f"prompt {prompt_size} + output {args.output_tokens} exceeds context {args.context}"
                    )
                for repetition in range(1, args.repetitions + 1):
                    flush_cache(base_url)
                    time.sleep(0.25)
                    vm = psutil.virtual_memory()
                    swap = psutil.swap_memory()
                    mem = pynvml.nvmlDeviceGetMemoryInfo(monitor.device)
                    input_ids = generate_ids(
                        tokenizer, prompt_size, nonce=prompt_size * 10 + repetition
                    )
                    monitor.set_phase(f"request_{prompt_size}_{repetition}")
                    request = stream_request(
                        base_url,
                        input_ids,
                        args.output_tokens,
                        seed=1000 + repetition,
                    )
                    request.update(
                        {
                            "repetition": repetition,
                            "requested_prompt_tokens": prompt_size,
                            "vram_free_before_request_gb": gib(mem.free),
                            "host_ram_used_before_gb": gib(vm.used),
                            "host_ram_available_before_gb": gib(vm.available),
                            "swap_used_before_gb": gib(swap.used),
                        }
                    )
                    time.sleep(0.6)  # let the monitor capture the final interval
                    summary = monitor.interval_summary(
                        request["client_started_monotonic"],
                        request["client_ended_monotonic"] + 0.5,
                    )
                    request["hardware"] = summary
                    raw["requests"].append(request)
                    rows.append(
                        normalized_row(
                            args,
                            repo,
                            request,
                            run_id,
                            True,
                            model_load,
                            summary,
                            note=f"measured repetition {repetition}; client TTFT includes HTTP and scheduling",
                        )
                    )
        if args.workload == "startup":
            rows.append(
                normalized_row(
                    args,
                    repo,
                    None,
                    run_id,
                    True,
                    model_load,
                    note="startup-only context capacity probe",
                )
            )
    except Exception as exc:
        raw["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        rows.append(
            normalized_row(
                args,
                repo,
                None,
                run_id,
                startup_success,
                model_load,
                note=f"{type(exc).__name__}: {exc}",
            )
        )
    finally:
        raw["finished"] = now_iso()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if monitor is not None:
            monitor.stop()
        log_handle.close()
        # Detect explicit offload/fallback text after the log has been flushed.
        log_text = log_path.read_text(errors="replace")
        cpu_offload = any(
            marker in log_text.lower()
            for marker in ("cpu offload enabled", "offloading weights", "kv cache offload")
        )
        for row in rows:
            row["cpu_offload_detected"] = cpu_offload
        raw["cpu_offload_detected"] = cpu_offload
        raw["normalized_rows"] = rows
        raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True, default=str) + "\n")
        if args.append_canonical_results:
            append_results(rows)
    print(json.dumps({"run_id": run_id, "startup_success": startup_success, "rows": len(rows), "raw": str(raw_path)}))
    return 0 if startup_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
