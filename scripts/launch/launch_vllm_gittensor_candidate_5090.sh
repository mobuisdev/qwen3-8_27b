#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="$ROOT/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348"

export QWEN_VLLM_MODEL_PATH="${QWEN_VLLM_MODEL_PATH:-$MODEL}"
export QWEN_VLLM_SERVED_MODEL_NAME="${QWEN_VLLM_SERVED_MODEL_NAME:-qwen38-vllm-gittensor-lmhead4}"
export QWEN_VLLM_MTP_TOKENS=0
export QWEN_VLLM_MAX_MODEL_LEN="${QWEN_VLLM_MAX_MODEL_LEN:-200000}"
export QWEN_VLLM_GPU_MEMORY_UTILIZATION="${QWEN_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

exec "$ROOT/scripts/launch/launch_vllm_5090.sh"
