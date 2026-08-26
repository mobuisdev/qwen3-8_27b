#!/usr/bin/env python3
"""Run focused coding-agent checks against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from benchmark_paths import session_directory


ROOT = Path(__file__).resolve().parents[1]


def post(base_url: str, route: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}{route}", json=payload, timeout=(60, 3600)
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return {"elapsed_seconds": elapsed, "response": response.json()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:32000")
    parser.add_argument("--model", default="qwen38-ninfer-nvfp4")
    parser.add_argument("--model-path", default=str(ROOT / "models" / "qwen_base"))
    parser.add_argument("--needle-context", type=int, default=100000)
    parser.add_argument(
        "--needle-max-tokens",
        type=int,
        default=64,
        help="Generation budget for each needle case (default: 64).",
    )
    parser.add_argument("--label", default="openai_quality")
    args = parser.parse_args()

    run_id = (
        f"{args.label}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    output: dict[str, Any] = {
        "run_id": run_id,
        "started": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "chat_cases": [],
        "needle_cases": [],
    }
    cases = [
        (
            "logic",
            "A farmer has 17 sheep. All but 9 die. How many remain? Answer briefly.",
            ["9"],
        ),
        (
            "coding",
            "Write Python def balanced_brackets(text: str) -> bool for (), [], and {}. "
            "Ignore other characters, reject mismatched nesting, and include three asserts.",
            ["balanced_brackets", "stack", "assert"],
        ),
        (
            "json",
            'Return exactly {"alpha":2,"beta":3} with no Markdown or commentary.',
            ['{"alpha":2,"beta":3}'],
        ),
    ]
    for case_id, prompt, expected in cases:
        result = post(
            args.base_url,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 768,
            },
        )
        message = result["response"]["choices"][0]["message"]
        combined = " ".join(
            str(message.get(key) or "")
            for key in ("reasoning", "reasoning_content", "content")
        )
        result.update(
            {
                "case": case_id,
                "expected": expected,
                "automatic_check": all(
                    term.lower() in combined.lower() for term in expected
                ),
            }
        )
        output["chat_cases"].append(result)

    tool_result = post(
        args.base_url,
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the current weather in Paris? Use the provided tool.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
        },
    )
    tool_message = tool_result["response"]["choices"][0]["message"]
    calls = tool_message.get("tool_calls") or []
    tool_result["automatic_check"] = bool(
        calls
        and calls[0].get("function", {}).get("name") == "get_weather"
        and "paris" in calls[0].get("function", {}).get("arguments", "").lower()
    )
    output["tool_case"] = tool_result

    needle_text = "The cobalt falcon access code is 739184."
    question_text = (
        "\nQuestion: What is the cobalt falcon access code? Answer with the digits."
    )
    filler_text = (
        "Archive note: routine measurements were checked and filed without access codes. "
    )
    filler_token_count = len(
        tokenizer.encode(filler_text, add_special_tokens=False)
    )
    filler_repetitions = max(
        1, (args.needle_context - 256) // filler_token_count
    )
    for position in (0.1, 0.5, 0.9):
        insertion = int(filler_repetitions * position)
        content = (
            f"Quality nonce {run_id}-{position}.\n"
            + filler_text * insertion
            + needle_text
            + filler_text * (filler_repetitions - insertion)
            + question_text
        )
        result = post(
            args.base_url,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": args.needle_max_tokens,
            },
        )
        message = result["response"]["choices"][0]["message"]
        text = " ".join(
            str(message.get(key) or "")
            for key in ("reasoning", "reasoning_content", "content")
        )
        result.update(
            {
                "position": position,
                "prompt_tokens": result["response"]["usage"]["prompt_tokens"],
                "automatic_check": "739184" in text,
            }
        )
        output["needle_cases"].append(result)
        print(
            json.dumps(
                {
                    "needle_position": position,
                    "passed": result["automatic_check"],
                    "elapsed_s": round(result["elapsed_seconds"], 2),
                }
            ),
            flush=True,
        )

    output["finished"] = datetime.now(timezone.utc).isoformat()
    path = session_directory(ROOT, "raw_results", run_id) / f"{run_id}.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "chat_pass": [case["automatic_check"] for case in output["chat_cases"]],
                "tool_pass": output["tool_case"]["automatic_check"],
                "needle_pass": [case["automatic_check"] for case in output["needle_cases"]],
                "raw": str(path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
