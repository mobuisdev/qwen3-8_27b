#!/usr/bin/env python3
"""Fetch immutable model metadata without downloading weight tensors."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = {
    "radixark": {
        "repo_id": "RadixArk/Qwen3.8-27B-NVFP4",
        "revision": "554ebba9b5f1b79dc11246341960360e6ef05ef4",
        "copy_snapshot": True,
    },
    "unsloth": {
        "repo_id": "unsloth/Qwen3.8-27B-NVFP4",
        "revision": "7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108",
        "copy_snapshot": True,
    },
    "qwen_base": {
        "repo_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "copy_snapshot": True,
    },
    "gittensor_candidate": {
        "repo_id": "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090",
        "revision": "0cc27958cefbbe231782ec8511de8c4eb5233348",
        "copy_snapshot": False,
    },
    "gittensor_dspark_candidate": {
        "repo_id": "gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4",
        "revision": "eba1ac5a66c74902eaa95a4000a7c5eda96d8e95",
        "copy_snapshot": False,
    },
    "ninfer_nvfp4_current_metadata": {
        "repo_id": "neroued/Qwen3.8-27B-nvfp4-NInfer",
        "revision": "204e3d92c30d9d05f3300d2f52e443ad1edf6ddf",
        "copy_snapshot": False,
    },
    "ninfer_groupwise_candidate": {
        "repo_id": "neroued/Qwen3.8-27B-NInfer",
        "revision": "18dfc887423fa5aabf3cb56fac41490e462b3fab",
        "copy_snapshot": False,
    },
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
    for key, spec in MODELS.items():
        repo_id = spec["repo_id"]
        revision = spec["revision"]
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
        if spec["copy_snapshot"]:
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
