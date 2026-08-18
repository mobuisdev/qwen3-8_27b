#!/usr/bin/env python3
"""Record exact Python/runtime versions and the local SGLang source delta."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import pathlib
import subprocess
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SGLANG = ROOT / "vendor" / "sglang"


def public_path(path: str) -> str:
    resolved = pathlib.Path(path).absolute()
    try:
        return f"<benchmark-root>/{resolved.relative_to(ROOT)}"
    except ValueError:
        return f"<external-python>/{resolved.name}"


def command(*args: str) -> str:
    result = subprocess.run(
        args,
        cwd=SGLANG if args and args[0] == "git" else ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=pathlib.Path, default=ROOT / "reports" / "environment.txt"
    )
    parser.add_argument(
        "--help-output",
        type=pathlib.Path,
        help="Optionally store the SGLang CLI help alongside the environment.",
    )
    args = parser.parse_args()

    packages = [
        "sglang",
        "torch",
        "flashinfer-python",
        "sglang-kernel",
        "transformers",
        "huggingface-hub",
        "compressed-tensors",
        "triton",
        "nvidia-ml-py",
    ]
    lines = [
        "Qwen3.8-27B RTX 5090 benchmark environment",
        f"captured_at={dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"python={sys.version}",
        f"executable={public_path(sys.executable)}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"cuda_available={torch.cuda.is_available()}",
        f"gpu_name={torch.cuda.get_device_name(0)}",
        f"compute_capability={torch.cuda.get_device_capability(0)}",
        f"compiled_arches={torch.cuda.get_arch_list()}",
        f"vram_bytes={torch.cuda.get_device_properties(0).total_memory}",
        "rust_extensions=disabled (SGLANG_BUILD_RUST_EXTS=none; Python frontend used)",
        "",
        "[selected packages]",
    ]
    for package in packages:
        lines.append(f"{package}={metadata.version(package)}")
    lines.extend(
        [
            "",
            "[sglang source]",
            command("git", "status", "--short", "--branch"),
            command("git", "rev-parse", "HEAD"),
            "",
            "[sglang local correctness patch]",
            command(
                "git",
                "diff",
                "--",
                "python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py",
            ),
            "",
            "[package freeze]",
            "\n".join(
                sorted(
                    f"{dist.metadata['Name']}=={dist.version}"
                    for dist in metadata.distributions()
                    if dist.metadata.get("Name")
                )
            ),
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    if args.help_output is not None:
        args.help_output.parent.mkdir(parents=True, exist_ok=True)
        help_text = command(sys.executable, "-m", "sglang.launch_server", "--help")
        args.help_output.write_text(help_text + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
