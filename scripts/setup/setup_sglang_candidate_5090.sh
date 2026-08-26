#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="$ROOT/vendor/sglang-qwen38-candidate"
VENV="$ROOT/.venv-sglang-qwen38-candidate"
SGLANG_COMMIT="8a1e6e4e461044246739b5a1ad579c8acc556a2d"

for command_name in git uv systemd-run; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Missing prerequisite: $command_name (see docs/INSTALL.md)" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/vendor"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/sgl-project/sglang.git "$SOURCE_DIR"
fi

current_commit="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
  echo "Refusing to use a modified candidate SGLang checkout at $SOURCE_DIR" >&2
  exit 1
fi
if [[ "$current_commit" != "$SGLANG_COMMIT" ]]; then
  if ! git -C "$SOURCE_DIR" cat-file -e "${SGLANG_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$SOURCE_DIR" fetch origin "$SGLANG_COMMIT"
  fi
  git -C "$SOURCE_DIR" checkout --detach "$SGLANG_COMMIT"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi

systemd-run --user --scope --quiet \
  -p MemoryHigh=36G -p MemoryMax=40G -p MemorySwapMax=0 -p OOMPolicy=kill \
  -- env \
    SGLANG_BUILD_RUST_EXTS=none \
    UV_CACHE_DIR="$ROOT/.uv-cache" \
    uv pip install --python "$VENV/bin/python" --editable "$SOURCE_DIR/python"

systemd-run --user --scope --quiet \
  -p MemoryHigh=36G -p MemoryMax=40G -p MemorySwapMax=0 -p OOMPolicy=kill \
  -- env UV_CACHE_DIR="$ROOT/.uv-cache" \
    uv pip install --python "$VENV/bin/python" \
      'torch==2.13.0' \
      'transformers==5.12.1' \
      'flashinfer-python[cu13]==0.6.17' \
      'sglang-kernel==0.4.6.post1'

"$VENV/bin/python" -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())'

"$ROOT/scripts/setup/setup_candidate_models_5090.sh"

echo "Candidate SGLang and Gittensor/DSpark snapshots are ready."
