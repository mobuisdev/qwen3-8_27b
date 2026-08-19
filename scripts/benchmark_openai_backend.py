#!/usr/bin/env python3
"""Measure single-request decode speed through an OpenAI-compatible streaming API."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import pynvml
import requests
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MONITOR_FIELDS = (
    "timestamp",
    "monotonic",
    "phase",
    "gpu_memory_used_bytes",
    "gpu_memory_free_bytes",
    "gpu_utilization_percent",
    "gpu_power_w",
    "gpu_temperature_c",
    "host_used_bytes",
    "host_available_bytes",
    "swap_used_bytes",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 4)


class Monitor:
    def __init__(self, path: Path, gpu_index: int) -> None:
        self.path = path
        self.samples: list[dict[str, Any]] = []
        self.phase = "idle"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        pynvml.nvmlInit()
        self.device = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    def _run(self) -> None:
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MONITOR_FIELDS)
            writer.writeheader()
            while not self.stop_event.is_set():
                memory = pynvml.nvmlDeviceGetMemoryInfo(self.device)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.device)
                vm = psutil.virtual_memory()
                swap = psutil.swap_memory()
                with self.lock:
                    phase = self.phase
                sample = {
                    "timestamp": now_iso(),
                    "monotonic": time.monotonic(),
                    "phase": phase,
                    "gpu_memory_used_bytes": memory.used,
                    "gpu_memory_free_bytes": memory.free,
                    "gpu_utilization_percent": utilization.gpu,
                    "gpu_power_w": round(
                        pynvml.nvmlDeviceGetPowerUsage(self.device) / 1000, 3
                    ),
                    "gpu_temperature_c": pynvml.nvmlDeviceGetTemperature(
                        self.device, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "host_used_bytes": vm.used,
                    "host_available_bytes": vm.available,
                    "swap_used_bytes": swap.used,
                }
                self.samples.append(sample)
                writer.writerow(sample)
                handle.flush()
                self.stop_event.wait(0.5)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        pynvml.nvmlShutdown()

    def summary(self, start: float, end: float) -> dict[str, Any]:
        rows = [row for row in self.samples if start <= row["monotonic"] <= end]
        if not rows:
            return {}
        return {
            "peak_vram_gb": gib(max(row["gpu_memory_used_bytes"] for row in rows)),
            "min_vram_free_gb": gib(
                min(row["gpu_memory_free_bytes"] for row in rows)
            ),
            "peak_gpu_utilization_percent": max(
                row["gpu_utilization_percent"] for row in rows
            ),
            "peak_gpu_power_w": max(row["gpu_power_w"] for row in rows),
            "peak_gpu_temperature_c": max(
                row["gpu_temperature_c"] for row in rows
            ),
            "peak_host_used_gb": gib(max(row["host_used_bytes"] for row in rows)),
            "min_host_available_gb": gib(
                min(row["host_available_bytes"] for row in rows)
            ),
            "peak_swap_used_gb": gib(max(row["swap_used_bytes"] for row in rows)),
        }


class NullMonitor:
    """No-op monitor for remote, CPU, AMD, or other non-NVML backends."""

    def start(self) -> None:
        pass

    def set_phase(self, phase: str) -> None:
        pass

    def stop(self) -> None:
        pass

    def summary(self, start: float, end: float) -> dict[str, Any]:
        return {}


def generate_ids(
    tokenizer: Any, target: int, nonce: str, workload: str = "essay"
) -> list[int]:
    prefix = tokenizer.encode(
        f"Unique benchmark request {nonce}. Read these technical notes carefully. ",
        add_special_tokens=False,
    )
    body = tokenizer.encode(
        "This deterministic document discusses software design, mathematics, hardware, "
        "testing, history, and precise measurement using ordinary explanatory prose. ",
        add_special_tokens=False,
    )
    if workload == "code":
        instruction = (
            "\nWrite a complete Python module implementing an asynchronous dependency-aware "
            "task scheduler. Include type hints, validation, cancellation, retries, docstrings, "
            "and unit-test examples. Output only code and use the full available token budget."
        )
    else:
        instruction = (
            "\nNow produce a numbered technical essay with many distinct points and no early "
            "conclusion. Use the full available token budget."
        )
    suffix = tokenizer.encode(instruction, add_special_tokens=False)
    remaining = target - len(prefix) - len(suffix)
    if remaining < 0:
        raise ValueError(f"Prompt target {target} is too small")
    return prefix + (body * (remaining // len(body) + 1))[:remaining] + suffix


def wait_ready(base_url: str, timeout: int, readiness_path: str) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            health = requests.get(f"{base_url}{readiness_path}", timeout=3)
            if health.ok:
                models = requests.get(f"{base_url}/v1/models", timeout=10)
                models.raise_for_status()
                return models.json()
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"Server not ready after {timeout}s: {last_error}")


def stream_completion(
    base_url: str,
    model: str,
    input_ids: list[int],
    output_tokens: int,
    tokenizer: Any,
    seed: int,
    api_style: str,
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if api_style == "chat":
        payload = {
            "model": model,
            "messages": messages or [
                {
                    "role": "user",
                    "content": tokenizer.decode(
                        input_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                }
            ],
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        endpoint = "chat/completions"
    else:
        payload = {
            "model": model,
            "prompt": input_ids,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "seed": seed,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        endpoint = "completions"
    started = time.perf_counter()
    first_content_at = None
    first_chunk_tokens = 1
    output_parts: list[str] = []
    usage: dict[str, Any] = {}
    response = requests.post(
        f"{base_url}/v1/{endpoint}",
        json=payload,
        stream=True,
        timeout=(60, 7200),
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if api_style == "chat":
            delta = choices[0].get("delta") or {} if choices else {}
            text = "".join(
                str(delta.get(key) or "")
                for key in ("reasoning", "reasoning_content", "content")
            )
        else:
            text = choices[0].get("text", "") if choices else ""
        if text:
            if first_content_at is None:
                first_content_at = time.perf_counter()
                first_chunk_tokens = max(
                    1, len(tokenizer.encode(text, add_special_tokens=False))
                )
            output_parts.append(text)
    ended = time.perf_counter()
    output_text = "".join(output_parts)
    generated = int(
        usage.get("completion_tokens")
        or len(tokenizer.encode(output_text, add_special_tokens=False))
    )
    prompt_tokens = int(usage.get("prompt_tokens") or len(input_ids))
    first = first_content_at or ended
    decode_seconds = max(0.0, ended - first)
    return {
        "client_started_monotonic": started,
        "client_ended_monotonic": ended,
        "actual_prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "first_chunk_tokens": first_chunk_tokens,
        "ttft_seconds": first - started,
        "total_seconds": ended - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (
            (generated - first_chunk_tokens) / decode_seconds
            if generated > first_chunk_tokens and decode_seconds
            else None
        ),
        "output_text": output_text,
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:32000")
    parser.add_argument("--model", default="qwen38-ninfer-nvfp4")
    parser.add_argument("--model-path", default=str(ROOT / "models" / "qwen_base"))
    parser.add_argument("--label", default="openai_backend")
    parser.add_argument("--prompt-sizes", default="1024,32768,131072,190000")
    parser.add_argument("--output-tokens", type=int, default=768)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--prefix-cache-size", type=int, default=0)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="NVML GPU index to monitor (default: 0).",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable local NVIDIA/host monitoring for a remote or non-NVIDIA server.",
    )
    parser.add_argument(
        "--readiness-path",
        default="/health",
        help="Use /model_info for the pinned 2026-08-17 SGLang runtime.",
    )
    parser.add_argument("--api-style", choices=("completion", "chat"), default="chat")
    parser.add_argument("--workload", choices=("essay", "code"), default="code")
    args = parser.parse_args()

    run_id = (
        f"{args.label}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    raw_path = ROOT / "raw_results" / f"{run_id}.json"
    csv_path = ROOT / "raw_results" / f"{run_id}.csv"
    monitor_path = ROOT / "monitoring" / f"{run_id}.csv"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    raw: dict[str, Any] = {
        "run_id": run_id,
        "started": now_iso(),
        "arguments": vars(args),
        "server_models": wait_ready(
            args.base_url, args.startup_timeout, args.readiness_path
        ),
        "requests": [],
    }
    monitor: Monitor | NullMonitor
    monitor = (
        NullMonitor()
        if args.no_monitor
        else Monitor(monitor_path, args.gpu_index)
    )
    monitor.start()
    try:
        monitor.set_phase("warmup")
        warm_ids = generate_ids(tokenizer, 256, "warmup", args.workload)
        raw["warmup"] = stream_completion(
            args.base_url, args.model, warm_ids, 32, tokenizer, 1, args.api_style
        )
        if args.prefix_cache_size:
            shared_ids = generate_ids(
                tokenizer,
                args.prefix_cache_size,
                f"{run_id}-shared-prefix",
                args.workload,
            )
            monitor.set_phase("prefix_cache_prime")
            prime = stream_completion(
                args.base_url,
                args.model,
                shared_ids,
                16,
                tokenizer,
                2,
                args.api_style,
            )
            extension = tokenizer.encode(
                "\nNew user turn: summarize the most important implications in detail.",
                add_special_tokens=False,
            )
            monitor.set_phase("prefix_cache_reuse")
            cached = stream_completion(
                args.base_url,
                args.model,
                shared_ids + extension,
                256,
                tokenizer,
                3,
                args.api_style,
                (
                    [
                        {
                            "role": "user",
                            "content": tokenizer.decode(
                                shared_ids,
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            ),
                        },
                        {"role": "assistant", "content": prime["output_text"]},
                        {
                            "role": "user",
                            "content": "Summarize the most important implications in detail.",
                        },
                    ]
                    if args.api_style == "chat"
                    else None
                ),
            )
            time.sleep(0.6)
            cached["hardware"] = monitor.summary(
                cached["client_started_monotonic"],
                cached["client_ended_monotonic"] + 0.5,
            )
            raw["prefix_cache_test"] = {
                "shared_prompt_tokens": len(shared_ids),
                "extension_tokens": len(extension),
                "prime": prime,
                "cached": cached,
            }
            print(
                json.dumps(
                    {
                        "prefix_cache_shared_tokens": len(shared_ids),
                        "prime_ttft_s": round(prime["ttft_seconds"], 3),
                        "cached_ttft_s": round(cached["ttft_seconds"], 3),
                        "cached_decode_tok_s": round(
                            cached["decode_tokens_per_second"], 2
                        ),
                    }
                ),
                flush=True,
            )
        for prompt_size in [
            int(item) for item in args.prompt_sizes.split(",") if item
        ]:
            for repetition in range(1, args.repetitions + 1):
                phase = f"request_{prompt_size}_{repetition}"
                monitor.set_phase(phase)
                request = stream_completion(
                    args.base_url,
                    args.model,
                    generate_ids(
                        tokenizer,
                        prompt_size,
                        f"{run_id}-{phase}",
                        args.workload,
                    ),
                    args.output_tokens,
                    tokenizer,
                    1000 + repetition,
                    args.api_style,
                )
                time.sleep(0.6)
                request.update(
                    {
                        "requested_prompt_tokens": prompt_size,
                        "repetition": repetition,
                        "hardware": monitor.summary(
                            request["client_started_monotonic"],
                            request["client_ended_monotonic"] + 0.5,
                        ),
                    }
                )
                raw["requests"].append(request)
                print(
                    json.dumps(
                        {
                            "prompt": prompt_size,
                            "repetition": repetition,
                            "generated": request["generated_tokens"],
                            "ttft_s": round(request["ttft_seconds"], 3),
                            "decode_tok_s": round(
                                request["decode_tokens_per_second"], 2
                            ),
                        }
                    ),
                    flush=True,
                )
        try:
            raw["metrics"] = requests.get(
                f"{args.base_url}/metrics", timeout=30
            ).text
        except requests.RequestException as exc:
            raw["metrics_error"] = str(exc)
    finally:
        monitor.stop()
        raw["finished"] = now_iso()
        raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        rows = []
        for request in raw["requests"]:
            row = {
                key: value
                for key, value in request.items()
                if key not in {"output_text", "usage", "hardware"}
            }
            row.update(request.get("hardware") or {})
            rows.append(row)
        if rows:
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({"run_id": run_id, "raw": str(raw_path), "csv": str(csv_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
