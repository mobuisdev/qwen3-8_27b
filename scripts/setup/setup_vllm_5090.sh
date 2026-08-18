#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv-vllm-0.27.1"
TOOLS_VENV="$ROOT/.venv-tools"
UNSLOTH_MODEL_REPO="unsloth/Qwen3.8-27B-NVFP4"
UNSLOTH_MODEL_REVISION="7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108"
GITTENSOR_MODEL_REPO="gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"
GITTENSOR_MODEL_REVISION="ec8ad26b9e3b33c7d05c0e5743b60f37f5139005"

if ! command -v uv >/dev/null; then
  echo "Missing prerequisite: uv (see docs/INSTALL.md)" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi
UV_CACHE_DIR="$ROOT/.uv-cache" uv pip install \
  --python "$VENV/bin/python" \
  'vllm==0.27.1' \
  'torch==2.13.0' \
  'transformers==5.15.0' \
  'flashinfer-python==0.6.16.post3'

# Keep FlashInfer's JIT compiler and CUDA headers on the same 13.0 minor
# version. A mixed 13.3 compiler/13.0 header environment failed locally.
UV_CACHE_DIR="$ROOT/.uv-cache" uv pip install \
  --python "$VENV/bin/python" \
  'nvidia-cuda-nvcc==13.0.88' \
  'nvidia-cuda-crt==13.0.88' \
  'nvidia-nvvm==13.0.88'

if [[ ! -x "$TOOLS_VENV/bin/python" ]]; then
  uv venv --python 3.12 "$TOOLS_VENV"
fi
UV_CACHE_DIR="$ROOT/.uv-cache" uv pip install \
  --python "$TOOLS_VENV/bin/python" \
  -r "$ROOT/requirements-benchmark.txt"

HF_HOME="$ROOT/hf-home" "$TOOLS_VENV/bin/hf" download \
  "$UNSLOTH_MODEL_REPO" \
  --revision "$UNSLOTH_MODEL_REVISION"

HF_HOME="$ROOT/hf-home" "$TOOLS_VENV/bin/hf" download \
  "$GITTENSOR_MODEL_REPO" \
  --revision "$GITTENSOR_MODEL_REVISION"

echo "vLLM is ready. Its first guarded launch compiles one FP4 kernel:"
echo "  $ROOT/scripts/launch/launch_vllm_5090.sh"
