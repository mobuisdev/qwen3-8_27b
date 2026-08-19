"""Shared paths for session-scoped benchmark artifacts."""

from __future__ import annotations

import os
import re
from pathlib import Path


SESSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def session_name(fallback: str) -> str:
    """Return the validated publication session or a unique run fallback."""
    value = os.environ.get("BENCH_SESSION") or fallback
    if not SESSION_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "BENCH_SESSION must be one safe directory name containing only "
            "letters, digits, dots, underscores, and hyphens"
        )
    return value


def session_directory(root: Path, category: str, fallback: str) -> Path:
    """Create and return ``<root>/<category>/<session>``."""
    path = root / category / session_name(fallback)
    path.mkdir(parents=True, exist_ok=True)
    return path
