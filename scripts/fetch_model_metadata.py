#!/usr/bin/env python3
"""Fetch immutable model metadata without downloading weight tensors."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = {
    "radixark": (
        "RadixArk/Qwen3.8-27B-NVFP4",
        "554ebba9b5f1b79dc11246341960360e6ef05ef4",
    ),
    "unsloth": (
        "unsloth/Qwen3.8-27B-NVFP4",
        "7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108",
    ),
    "qwen_base": (
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ),
}
ALLOW = [
    "*.json",
    "*.jinja",
    "*.txt",
    "README.md",
    "LICENSE",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh tracked metadata for the pinned model revisions."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Confirm the network fetch and tracked metadata replacement.",
    )
    args = parser.parse_args(argv)
    if not args.refresh:
        parser.error("refusing to replace tracked metadata without --refresh")
    return args


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    inventory: dict[str, object] = {}
    for key, (repo_id, revision) in MODELS.items():
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
        snapshot = pathlib.Path(
            snapshot_download(
                repo_id,
                revision=revision,
                allow_patterns=ALLOW,
            )
        )
        target = ROOT / "models" / key
        target.mkdir(parents=True, exist_ok=True)
        for source in snapshot.iterdir():
            if source.is_file():
                shutil.copy2(source, target / source.name)
        files = [
            {
                "name": sibling.rfilename,
                "size_bytes": sibling.size,
                "blob_id": sibling.blob_id,
                "lfs": sibling.lfs,
            }
            for sibling in info.siblings
        ]
        inventory[key] = {
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": info.sha,
            "last_modified": info.last_modified.isoformat()
            if info.last_modified
            else None,
            "private": info.private,
            "gated": info.gated,
            "pipeline_tag": info.pipeline_tag,
            "library_name": info.library_name,
            "tags": info.tags,
            "files": files,
            "total_file_bytes": sum(item["size_bytes"] or 0 for item in files),
        }
    (ROOT / "models" / "repositories.json").write_text(
        json.dumps(inventory, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(ROOT / "models" / "repositories.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
