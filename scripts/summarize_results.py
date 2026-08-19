#!/usr/bin/env python3
"""Build publication-ready CSV and Markdown tables from one run prefix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "configuration",
    "model",
    "actual_prompt_tokens",
    "samples",
    "median_output_tokens",
    "median_ttft_s",
    "median_decode_tps",
    "min_decode_tps",
    "max_decode_tps",
    "peak_vram_gib",
    "minimum_host_available_gib",
)


def number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def configuration(data: dict[str, Any]) -> str:
    args = data.get("arguments") or {}
    return str(args.get("label") or args.get("config_id") or data.get("run_id"))


def model(data: dict[str, Any]) -> str:
    value = data.get("model")
    if isinstance(value, dict):
        return str(value.get("repo_id") or value)
    return str((data.get("arguments") or {}).get("model") or value or "unknown")


def redact_private_paths(value: object) -> str:
    """Remove machine-specific path prefixes from publication text."""
    rendered = str(value)
    replacements = {
        str(ROOT): "<REPOSITORY>",
        str(Path.home()): "<HOME>",
    }
    user = os.environ.get("USER")
    if user:
        replacements[f"/home/{user}"] = "<HOME>"
    for private, public in sorted(replacements.items(), key=lambda item: -len(item[0])):
        rendered = rendered.replace(private, public)
    return rendered


def public_error_summary(error: object) -> str:
    if isinstance(error, dict):
        kind = error.get("type", "Error")
        message = error.get("message", "")
        return redact_private_paths(f"{kind}: {message}".rstrip())
    return redact_private_paths(error)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Run-ID prefix shared by this session.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    matched: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((ROOT / "raw_results").glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if str(data.get("run_id", path.stem)).startswith(args.prefix):
            matched.append((path, data))
    if not matched:
        raise SystemExit(f"No raw result has run_id prefix {args.prefix!r}")

    rows: list[dict[str, Any]] = []
    quality: list[tuple[str, int, int]] = []
    prefix_reuse: list[dict[str, Any]] = []
    failures: list[str] = []
    for path, data in matched:
        if data.get("error"):
            failures.append(f"{path.name}: {public_error_summary(data['error'])}")
        incomplete = [
            request
            for request in data.get("requests") or []
            if request.get("decode_tokens_per_second") is None
        ]
        if incomplete:
            failures.append(
                f"{path.name}: IncompleteMeasurement: decode throughput unavailable "
                f"for {len(incomplete)} request(s)"
            )
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for request in data.get("requests") or []:
            prompt = request.get("actual_prompt_tokens")
            if prompt is not None and request.get("decode_tokens_per_second") is not None:
                grouped[int(prompt)].append(request)
        for prompt, requests in sorted(grouped.items()):
            decode = [float(item["decode_tokens_per_second"]) for item in requests]
            ttft = [float(item["ttft_seconds"]) for item in requests]
            output = [int(item["generated_tokens"]) for item in requests]
            vram = []
            available = []
            for item in requests:
                hardware = item.get("hardware") or {}
                if (value := number(hardware.get("peak_vram_gb"))) is not None:
                    vram.append(value)
                for candidate in (
                    hardware.get("min_host_available_gb"),
                    item.get("host_ram_available_before_gb"),
                ):
                    if (value := number(candidate)) is not None:
                        available.append(value)
            rows.append(
                {
                    "configuration": configuration(data),
                    "model": model(data),
                    "actual_prompt_tokens": prompt,
                    "samples": len(requests),
                    "median_output_tokens": round(statistics.median(output)),
                    "median_ttft_s": round(statistics.median(ttft), 3),
                    "median_decode_tps": round(statistics.median(decode), 2),
                    "min_decode_tps": round(min(decode), 2),
                    "max_decode_tps": round(max(decode), 2),
                    "peak_vram_gib": round(max(vram), 2) if vram else "",
                    "minimum_host_available_gib": (
                        round(min(available), 2) if available else ""
                    ),
                }
            )

        checks: list[bool] = []
        checks.extend(
            bool(case.get("automatic_check")) for case in data.get("chat_cases") or []
        )
        if data.get("tool_case"):
            checks.append(bool(data["tool_case"].get("automatic_check")))
        checks.extend(
            bool(case.get("automatic_check")) for case in data.get("needle_cases") or []
        )
        checks.extend(
            bool(case.get("automatic_term_check"))
            for case in data.get("quality_results") or []
        )
        if checks:
            quality.append((configuration(data), sum(checks), len(checks)))

        if data.get("prefix_cache_test"):
            test = data["prefix_cache_test"]
            prime = test.get("prime") or {}
            cached = test.get("cached") or {}
            prefix_reuse.append(
                {
                    "configuration": configuration(data),
                    "shared": test.get("shared_prompt_tokens", ""),
                    "prime_ttft": number(prime.get("ttft_seconds")),
                    "cached_ttft": number(cached.get("ttft_seconds")),
                    "cached_decode": number(cached.get("decode_tokens_per_second")),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.prefix}-results.csv"
    md_path = args.output_dir / f"{args.prefix}-results.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Benchmark results: {args.prefix}",
        "",
        "All throughput values are client-observed medians. Prompt counts are the actual",
        "server-reported counts, not requested target sizes.",
        "",
        "| Configuration | Model | Actual prompt | n | Output | TTFT (s) | Decode tok/s | Range | Peak VRAM GiB | Min host available GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['configuration']} | {row['model']} | {row['actual_prompt_tokens']} | "
            f"{row['samples']} | {row['median_output_tokens']} | {row['median_ttft_s']} | "
            f"{row['median_decode_tps']} | {row['min_decode_tps']}-{row['max_decode_tps']} | "
            f"{row['peak_vram_gib']} | {row['minimum_host_available_gib']} |"
        )
    if quality:
        lines.extend(["", "## Quality smoke tests", ""])
        lines.extend(f"- {name}: {passed}/{total} passed" for name, passed, total in quality)
    if prefix_reuse:
        lines.extend(
            [
                "",
                "## Prefix reuse",
                "",
                "| Configuration | Shared target tokens | Prime TTFT (s) | Cached TTFT (s) | Cached decode tok/s |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in prefix_reuse:
            lines.append(
                f"| {item['configuration']} | {item['shared']} | "
                f"{item['prime_ttft']:.3f} | {item['cached_ttft']:.3f} | "
                f"{item['cached_decode']:.2f} |"
            )
    if failures:
        lines.extend(["", "## Recorded failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.extend(["", "## Raw sources", ""])
    lines.extend(f"- `{path.name}`" for path, _ in matched)
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "rows": len(rows)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
