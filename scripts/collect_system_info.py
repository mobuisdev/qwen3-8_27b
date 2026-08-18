#!/usr/bin/env python3
"""Collect reproducible, read-only host and accelerator information."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import shlex
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]


def redact(value: str) -> str:
    replacements = (
        (str(ROOT), "<benchmark-root>"),
        (str(pathlib.Path.home()), "<home>"),
        (os.environ.get("USER", ""), "<user>"),
    )
    for private, public in replacements:
        if private:
            value = value.replace(private, public)
    return value


def run(command: list[str]) -> str:
    rendered = shlex.join(command)
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            cwd=ROOT,
        )
        output = redact(result.stdout.rstrip())
        return f"$ {rendered}\nexit={result.returncode}\n{output}\n"
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"$ {rendered}\nerror={exc!r}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=pathlib.Path, default=ROOT / "reports" / "system-info.txt"
    )
    args = parser.parse_args()

    commands = [
        ["uname", "-srmo"],
        ["cat", "/etc/os-release"],
        ["systemd-detect-virt"],
        ["lscpu"],
        ["lsmem", "--summary=only"],
        ["free", "-b"],
        ["swapon", "--show", "--bytes"],
        ["cat", "/proc/pressure/memory"],
        ["vmstat", "-s"],
        ["df", "-h", "--output=fstype,size,used,avail,pcent", ".", "/tmp"],
        ["findmnt", "-T", ".", "-no", "FSTYPE,OPTIONS"],
        ["python3", "--version"],
        ["nvcc", "--version"],
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,"
            "memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit,"
            "pstate,clocks.current.graphics,clocks.current.memory,"
            "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,"
            "pcie.link.width.max",
            "--format=csv",
        ],
        ["nvidia-smi", "topo", "-m"],
        ["lspci", "-vv", "-d", "10de:"],
        ["rpm", "-qa"],
    ]
    header = [
        "Qwen3.8-27B RTX 5090 benchmark system snapshot",
        f"captured_at={dt.datetime.now(dt.timezone.utc).isoformat()}",
        "benchmark_root=<repository-root>",
        "HF_HOME=<benchmark-root>/hf-home",
        "privacy=hostnames, user paths, device UUIDs, and process lists omitted",
        "WSL2=no (native Linux)",
        "",
    ]
    sections: list[str] = ["\n".join(header)]
    for command in commands:
        output = run(command)
        if command[:2] == ["rpm", "-qa"]:
            lines = sorted(
                line
                for line in output.splitlines()
                if line.startswith(("cuda", "nvidia", "akmod-nvidia", "xorg-x11-drv-nvidia"))
                or line.startswith("$ ")
                or line.startswith("exit=")
            )
            output = "\n".join(lines) + "\n"
        sections.append(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
