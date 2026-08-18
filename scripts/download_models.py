#!/usr/bin/env python3
"""Download only pinned model weight files into the experiment HF cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_HOME = ROOT / "hf-home"
REPOS = ROOT / "models" / "repositories.json"
MANIFEST = ROOT / "models" / "downloaded_snapshots.json"
MODEL_NAMES = ("radixark", "unsloth")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download immutable model snapshots into the local HF cache."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_NAMES,
        dest="models",
        help="Snapshot to download; repeat for multiple models (default: both).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = str(HF_HOME)
    metadata = json.loads(REPOS.read_text())
    try:
        downloaded = json.loads(MANIFEST.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        downloaded = {}
    for name in args.models or MODEL_NAMES:
        repo = metadata[name]
        repo_id = repo["repo_id"]
        revision = repo["resolved_revision"]
        print(f"Downloading {repo_id}@{revision}", flush=True)
        # Re-running is cheap: Hub blobs are content-addressed, so existing
        # weights are reused and only missing metadata/tokenizer files transfer.
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=HF_HOME / "hub",
        )
        downloaded[name] = {
            "repo_id": repo_id,
            "revision": revision,
            "snapshot_path": str(Path(path).resolve()),
        }
        print(f"Completed {name}: {path}", flush=True)
    MANIFEST.write_text(
        json.dumps(downloaded, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
