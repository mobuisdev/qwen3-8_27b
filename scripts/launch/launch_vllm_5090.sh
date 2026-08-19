#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv-vllm-0.27.1"
CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
MODEL_DEFAULT="$ROOT/hf-home/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"

MODEL_PATH="${QWEN_VLLM_MODEL_PATH:-$MODEL_DEFAULT}"
PORT="${QWEN_VLLM_PORT:-31000}"
# Reserve room for FlashInfer's lazy 394 MiB workspace and first-use kernels.
# At 0.93, automatic KV sizing can leave too little free VRAM for generation.
GPU_MEMORY_UTILIZATION="${QWEN_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${QWEN_VLLM_MAX_MODEL_LEN:--1}"
MTP_TOKENS="${QWEN_VLLM_MTP_TOKENS:-3}"
SERVED_MODEL_NAME="${QWEN_VLLM_SERVED_MODEL_NAME:-qwen38-vllm-unsloth}"
MAX_NUM_SEQS="${QWEN_VLLM_MAX_NUM_SEQS:-1}"
CUDAGRAPH_DEFAULT='{"cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}'
CUDAGRAPH_CONFIG="${QWEN_VLLM_CUDAGRAPH_CONFIG:-$CUDAGRAPH_DEFAULT}"

SPECULATIVE_ARGS=()
if (( MTP_TOKENS > 0 )); then
  SPECULATIVE_ARGS=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}"
  )
fi

if [[ ! -x "$VENV/bin/vllm" ]]; then
  echo "Missing vLLM environment: $VENV" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/model.safetensors" && ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  echo "Missing model weights or index in: $MODEL_PATH" >&2
  exit 1
fi

# The server alone is constrained. MemorySwapMax=0 prevents a compile or load
# failure from pushing the interactive desktop into swap/OOM instability.
exec env \
  CUDA_HOME="$CUDA_HOME" \
  PATH="$CUDA_HOME/bin:$PATH" \
  LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  FLASHINFER_WORKSPACE_BASE="$ROOT/flashinfer-workspace" \
  HF_HOME="$ROOT/hf-home" \
  VLLM_CACHE_ROOT="$ROOT/vllm-cache" \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  MAX_JOBS=1 \
  TOKENIZERS_PARALLELISM=false \
  systemd-run --user --scope --quiet \
    -p MemoryHigh=36G \
    -p MemoryMax=40G \
    -p MemorySwapMax=0 \
    -p OOMPolicy=kill \
    -- \
    "$VENV/bin/vllm" serve "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --kv-cache-dtype fp8 \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens 2048 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_xml \
      --enable-prefix-caching \
      --language-model-only \
      --cpu-offload-gb 0 \
      --no-enable-flashinfer-autotune \
      "${SPECULATIVE_ARGS[@]}" \
      --compilation-config "$CUDAGRAPH_CONFIG"
