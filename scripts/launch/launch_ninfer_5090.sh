#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_DEFAULT="$ROOT/vendor/ninfer/build/apps/ninfer-serve"
MODEL_DEFAULT="$ROOT/ninfer-models/qwen3_8_27b_nvfp4.ninfer"

BIN="${QWEN_NINFER_BIN:-$BIN_DEFAULT}"
MODEL_PATH="${QWEN_NINFER_MODEL_PATH:-$MODEL_DEFAULT}"
PORT="${QWEN_NINFER_PORT:-32000}"
MODEL_ID="${QWEN_NINFER_MODEL_ID:-qwen38-ninfer-nvfp4}"
MAX_CONTEXT="${QWEN_NINFER_MAX_CONTEXT:-200000}"
KV_CAPACITY="${QWEN_NINFER_KV_CAPACITY:-auto}"
MAX_CONCURRENCY="${QWEN_NINFER_MAX_CONCURRENCY:-1}"
DRAFT_TOKENS="${QWEN_NINFER_DRAFT_TOKENS:-3}"

if [[ ! -x "$BIN" ]]; then
  echo "Missing NInfer server binary: $BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing NInfer model artifact: $MODEL_PATH" >&2
  exit 1
fi

SPECULATIVE_ARGS=()
if (( DRAFT_TOKENS > 0 )); then
  SPECULATIVE_ARGS=(
    --spec mtp
    --draft-tokens "$DRAFT_TOKENS"
    --lm-head-draft
  )
fi

# Keep the inference process away from the desktop's last 20+ GiB of RAM.
# MemorySwapMax=0 ensures a failed load is killed inside this scope rather than
# swapping the workstation into an unusable state.
exec systemd-run --user --scope --quiet \
  -p MemoryHigh=36G \
  -p MemoryMax=40G \
  -p MemorySwapMax=0 \
  -p OOMPolicy=kill \
  -- \
  "$BIN" "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model-id "$MODEL_ID" \
    --max-context "$MAX_CONTEXT" \
    --kv-capacity "$KV_CAPACITY" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --max-pending-requests 8 \
    --prefill-chunk 1024 \
    --kv-dtype int8 \
    --max-request-mib 32 \
    --media-cache-mib 0 \
    --no-thinking \
    --greedy \
    "${SPECULATIVE_ARGS[@]}"
