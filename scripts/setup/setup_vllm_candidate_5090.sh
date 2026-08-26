#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="$ROOT/vendor/vllm-qwen38-candidate"
VENV="$ROOT/.venv-vllm-qwen38-candidate"
VLLM_COMMIT="1eab6fef01b78ec4eab6b7156bbf5f120e48d381"

for command_name in git uv systemd-run; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Missing prerequisite: $command_name (see docs/INSTALL.md)" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/vendor"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$SOURCE_DIR"
fi

current_commit="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
  echo "Refusing to use a modified candidate vLLM checkout at $SOURCE_DIR" >&2
  exit 1
fi
if [[ "$current_commit" != "$VLLM_COMMIT" ]]; then
  if ! git -C "$SOURCE_DIR" cat-file -e "${VLLM_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$SOURCE_DIR" fetch origin "$VLLM_COMMIT"
  fi
  git -C "$SOURCE_DIR" checkout --detach "$VLLM_COMMIT"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi

systemd-run --user --scope --quiet \
  -p MemoryHigh=36G -p MemoryMax=40G -p MemorySwapMax=0 -p OOMPolicy=kill \
  -- env \
    UV_CACHE_DIR="$ROOT/.uv-cache" \
    MAX_JOBS=1 NVCC_THREADS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 \
    uv pip install --python "$VENV/bin/python" --editable "$SOURCE_DIR"

UV_CACHE_DIR="$ROOT/.uv-cache" uv pip install --python "$VENV/bin/python" \
  'nvidia-cuda-nvcc==13.0.88' \
  'nvidia-cuda-crt==13.0.88' \
  'nvidia-nvvm==13.0.88'

"$ROOT/scripts/setup/setup_candidate_models_5090.sh"

"$VENV/bin/python" -c \
  'import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'

echo "Candidate vLLM source environment and Gittensor snapshots are ready."
