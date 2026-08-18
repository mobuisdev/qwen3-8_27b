#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="$BENCH_ROOT/hf-home"
export SGLANG_CACHE_DIR="$BENCH_ROOT/sglang-cache"
export FLASHINFER_WORKSPACE_BASE="$BENCH_ROOT/sglang-cache"
export TORCHINDUCTOR_COMPILE_THREADS=1
export MAX_JOBS=1
export CPATH="$BENCH_ROOT/cuda-header-shim"
export LIBRARY_PATH="$BENCH_ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$BENCH_ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"

MODEL="$BENCH_ROOT/hf-home/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/554ebba9b5f1b79dc11246341960360e6ef05ef4"

exec systemd-run --user --scope --quiet \
  -p MemoryHigh=36G -p MemoryMax=40G -p MemorySwapMax=0 -p OOMPolicy=kill -- \
  "$BENCH_ROOT/.venv/bin/sglang" serve \
  --model-path "$MODEL" --served-model-name qwen38-radixark \
  --host 127.0.0.1 --port 30000 --context-length 200000 \
  --mem-fraction-static 0.91 --chunked-prefill-size 2048 \
  --attention-backend flashinfer --disable-flashinfer-autotune \
  --kv-cache-dtype fp8_e4m3 --max-running-requests 1 --cpu-offload-gb 0 \
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size 4 --skip-server-warmup --watchdog-timeout 900 \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 1 --enable-metrics
