#!/usr/bin/env python3
"""Compare immutable publication/candidate pins with official upstream heads."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "models" / "upstream_pins.json"
USER_AGENT = "qwen38-rtx5090-upstream-audit/1"


def fetch_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def hf_model(repo_id: str) -> dict:
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    result = fetch_json(f"https://huggingface.co/api/models/{encoded}")
    if not isinstance(result, dict):
        raise TypeError(f"Unexpected Hugging Face response for {repo_id}")
    return result


def github_latest_release(repo: str) -> dict:
    result = fetch_json(f"https://api.github.com/repos/{repo}/releases/latest")
    if not isinstance(result, dict):
        raise TypeError(f"Unexpected GitHub release response for {repo}")
    return result


def github_compare(repo: str, revision: str, branch: str) -> dict:
    result = fetch_json(
        f"https://api.github.com/repos/{repo}/compare/{revision}...{branch}"
    )
    if not isinstance(result, dict):
        raise TypeError(f"Unexpected GitHub comparison response for {repo}")
    return result


def audit(pins: dict) -> dict:
    output: dict[str, object] = {
        "schema_version": 1,
        "pin_catalog_checked_at": pins["checked_at"],
        "models": {},
        "releases": {},
        "sources": {},
    }
    model_cache: dict[str, dict] = {}
    for name, spec in pins["models"].items():
        repo_id = spec["repo_id"]
        if repo_id not in model_cache:
            model_cache[repo_id] = hf_model(repo_id)
        info = model_cache[repo_id]
        current = info["sha"]
        output["models"][name] = {
            "repo_id": repo_id,
            "track": spec["track"],
            "pinned": spec["revision"],
            "current": current,
            "updated": current != spec["revision"],
            "last_modified": info.get("lastModified"),
        }
    for name, spec in pins["releases"].items():
        info = github_latest_release(spec["repo"])
        current = info["tag_name"]
        output["releases"][name] = {
            "repo": spec["repo"],
            "pinned": spec["tag"],
            "current": current,
            "updated": current != spec["tag"],
            "published_at": info.get("published_at"),
        }
    for name, spec in pins["sources"].items():
        info = github_compare(spec["repo"], spec["revision"], spec["branch"])
        output["sources"][name] = {
            "repo": spec["repo"],
            "track": spec["track"],
            "pinned": spec["revision"],
            "branch": spec["branch"],
            "status": info.get("status"),
            "ahead_by": info.get("ahead_by"),
            "behind_by": info.get("behind_by"),
            "updated": bool(info.get("ahead_by")),
            "comparison_url": info.get("html_url"),
        }
    return output


def print_summary(result: dict) -> None:
    for category in ("models", "releases", "sources"):
        print(f"[{category}]")
        for name, item in result[category].items():
            marker = "UPDATE" if item["updated"] else "current"
            if category == "sources":
                detail = (
                    f"{item['repo']} {item['pinned'][:12]} -> {item['branch']} "
                    f"(ahead={item['ahead_by']}, status={item['status']})"
                )
            else:
                detail = f"{item['pinned']} -> {item['current']}"
            print(f"{marker:7} {name}: {detail}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check tracked model, release, and source pins against upstream."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the complete audit as JSON instead of a compact summary.",
    )
    parser.add_argument(
        "--fail-on-update",
        action="store_true",
        help="Exit 2 when any tracked pin differs from its upstream head.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    try:
        result = audit(pins)
    except (OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"upstream audit failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_summary(result)
    changed = any(
        item["updated"]
        for category in ("models", "releases", "sources")
        for item in result[category].values()
    )
    return 2 if args.fail_on_update and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
