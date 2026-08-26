#!/usr/bin/env python3
"""Fail early when a benchmark host is not in a safe state."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
NINFER_SHA256 = "bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32"
NINFER_GROUPWISE_SHA256 = (
    "eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e"
)


def meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        result[key] = int(value.strip().split()[0]) * 1024
    return result


def inference_processes() -> list[str]:
    markers = ("ninfer-serve", "vllm serve", "sglang.launch_server", "sglang serve")
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(marker in command for marker in markers):
            found.append(f"pid={entry.name} {command.strip()}")
    return found


def require(path: Path, description: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing {description}: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("all", "ninfer", "vllm", "sglang"),
        default="all",
    )
    parser.add_argument(
        "--verify-model-hash",
        action="store_true",
        help="Read and hash the selected NInfer artifact.",
    )
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Check the isolated candidate matrix instead of publication pins.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    memory = meminfo()
    available = memory.get("MemAvailable", 0)
    swap_used = memory.get("SwapTotal", 0) - memory.get("SwapFree", 0)
    if available < 45 * GIB:
        errors.append(
            f"Only {available / GIB:.1f} GiB host RAM is available; require at least 45 GiB."
        )
    if swap_used > 512 * 1024**2:
        warnings.append(
            f"{swap_used / GIB:.1f} GiB swap is already used; a fresh reboot is preferable."
        )

    running = inference_processes()
    if running:
        errors.append("An inference server is already running:\n  " + "\n  ".join(running))

    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,pstate,power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        fields = [field.strip() for field in gpu.split(",")]
        if len(fields) >= 4 and float(fields[3]) < 29000:
            errors.append(
                f"Only {float(fields[3]):.0f} MiB GPU memory is free; require at least 29000 MiB."
            )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        errors.append(f"nvidia-smi failed: {exc}")
        gpu = "unavailable"

    unsloth = ROOT / "hf-home/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
    gittensor = ROOT / "hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/ec8ad26b9e3b33c7d05c0e5743b60f37f5139005"
    radixark = ROOT / "hf-home/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/554ebba9b5f1b79dc11246341960360e6ef05ef4"
    ninfer_model = ROOT / "ninfer-models/qwen3_8_27b_nvfp4.ninfer"
    candidate_gittensor = (
        ROOT
        / "hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090"
        / "snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348"
    )
    candidate_dspark = (
        ROOT
        / "hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-DSpark-NVFP4"
        / "snapshots/eba1ac5a66c74902eaa95a4000a7c5eda96d8e95"
    )
    candidate_ninfer = ROOT / "ninfer-models/groupwise/qwen3_8_27b.ninfer"

    if args.backend in ("all", "ninfer"):
        selected_ninfer = candidate_ninfer if args.candidate else ninfer_model
        expected_ninfer_hash = (
            NINFER_GROUPWISE_SHA256 if args.candidate else NINFER_SHA256
        )
        ninfer_is_optional = args.candidate and args.backend == "all"
        if ninfer_is_optional and not selected_ninfer.exists():
            warnings.append(
                "Optional NInfer groupwise candidate is not installed; "
                "use --backend ninfer to require it."
            )
        else:
            require(
                ROOT / "vendor/ninfer/build/apps/ninfer-serve",
                "NInfer binary",
                errors,
            )
            require(selected_ninfer, "NInfer artifact", errors)
            if args.verify_model_hash and selected_ninfer.is_file():
                actual = sha256(selected_ninfer)
                if actual != expected_ninfer_hash:
                    errors.append(f"NInfer artifact SHA-256 mismatch: {actual}")
    if args.backend in ("all", "vllm"):
        if args.candidate:
            require(candidate_gittensor, "candidate Gittensor snapshot", errors)
            if not (
                (ROOT / ".venv-vllm-0.27.1/bin/vllm").exists()
                or (ROOT / ".venv-vllm-qwen38-candidate/bin/vllm").exists()
            ):
                errors.append("Missing both stable and candidate vLLM environments")
        else:
            require(ROOT / ".venv-vllm-0.27.1/bin/vllm", "vLLM environment", errors)
            require(unsloth, "Unsloth snapshot", errors)
            require(gittensor, "Gittensor snapshot", errors)
    if args.backend in ("all", "sglang"):
        if args.candidate:
            require(
                ROOT / ".venv-sglang-qwen38-candidate/bin/sglang",
                "candidate SGLang environment",
                errors,
            )
            require(candidate_gittensor, "candidate Gittensor snapshot", errors)
            require(candidate_dspark, "candidate DSpark snapshot", errors)
        else:
            require(ROOT / ".venv/bin/sglang", "SGLang environment", errors)
            require(radixark, "RadixArk snapshot", errors)

    client_python = next(
        (
            path
            for path in (
                ROOT / ".venv-tools/bin/python",
                ROOT / ".venv-sglang-qwen38-candidate/bin/python",
                ROOT / ".venv-vllm-qwen38-candidate/bin/python",
                ROOT / ".venv-vllm-0.27.1/bin/python",
                ROOT / ".venv/bin/python",
            )
            if path.exists()
        ),
        None,
    )
    if client_python is None:
        errors.append("No Python environment is available for the benchmark clients.")

    print(f"GPU: {gpu}")
    print(f"Host RAM available: {available / GIB:.1f} GiB")
    print(f"Swap currently used: {swap_used / GIB:.1f} GiB")
    print(f"Benchmark client Python: {client_python or 'missing'}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("Preflight passed. Run only one backend at a time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
