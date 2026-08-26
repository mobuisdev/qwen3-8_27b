#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv-sglang-qwen38-candidate"
MODEL="$ROOT/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348"
DRAFT="$ROOT/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-DSpark-NVFP4/snapshots/eba1ac5a66c74902eaa95a4000a7c5eda96d8e95"

if [[ ! -x "$VENV/bin/sglang" ]]; then
  echo "Missing candidate SGLang environment: $VENV" >&2
  exit 1
fi
if [[ ! -f "$MODEL/model.safetensors.index.json" ]]; then
  echo "Missing candidate Gittensor target snapshot: $MODEL" >&2
  exit 1
fi
if [[ ! -f "$DRAFT/model.safetensors" ]]; then
  echo "Missing candidate Gittensor DSpark snapshot: $DRAFT" >&2
  exit 1
fi

export HF_HOME="$ROOT/hf-home"
export SGLANG_CACHE_DIR="$ROOT/sglang-cache-candidate"
export FLASHINFER_WORKSPACE_BASE="$ROOT/sglang-cache-candidate"
export TORCHINDUCTOR_COMPILE_THREADS=1
export MAX_JOBS=1
export CPATH="$ROOT/cuda-header-shim"
export LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"

exec systemd-run --user --scope --quiet \
  -p MemoryHigh=36G -p MemoryMax=40G -p MemorySwapMax=0 -p OOMPolicy=kill -- \
  "$VENV/bin/sglang" serve \
  --model-path "$MODEL" --served-model-name qwen38-sglang-gittensor-dspark \
  --speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-model-quantization modelopt_fp4 \
  --speculative-dspark-block-size 7 --trust-remote-code --tp-size 1 \
  --host 127.0.0.1 --port 30000 --context-length 122880 \
  --mem-fraction-static 0.86 --chunked-prefill-size 1024 \
  --attention-backend flashinfer --disable-flashinfer-autotune \
  --kv-cache-dtype fp8_e4m3 --max-running-requests 1 --cpu-offload-gb 0 \
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size 8 --mm-feature-transport cpu \
  --skip-server-warmup --watchdog-timeout 900 \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 1 --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-metrics
