#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export QWEN_NINFER_MODEL_PATH="${QWEN_NINFER_MODEL_PATH:-$ROOT/ninfer-models/groupwise/qwen3_8_27b.ninfer}"
export QWEN_NINFER_MODEL_ID="${QWEN_NINFER_MODEL_ID:-qwen38-ninfer-groupwise}"

exec "$ROOT/scripts/launch/launch_ninfer_5090.sh"
