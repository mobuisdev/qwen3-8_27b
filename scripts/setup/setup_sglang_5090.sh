#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="$ROOT/vendor/sglang"
VENV="$ROOT/.venv"
TOOLS_VENV="$ROOT/.venv-tools"
PATCH_FILE="$ROOT/patches/sglang-pr-34904-lm-head.patch"
SGLANG_COMMIT="374a6b24f2f2b52fc131417d8d0e4e78900f7a5d"

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
if [[ "$current_commit" != "$SGLANG_COMMIT" ]]; then
  if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
    echo "Refusing to replace a modified SGLang checkout at $SOURCE_DIR" >&2
    exit 1
  fi
  if ! git -C "$SOURCE_DIR" cat-file -e "${SGLANG_COMMIT}^{commit}" 2>/dev/null; then
    if ! git -C "$SOURCE_DIR" fetch origin "$SGLANG_COMMIT"; then
      # GitHub may reject direct SHA fetches. The tested commit belongs to this
      # pull request, so fetch its current head and then verify the pinned object.
      git -C "$SOURCE_DIR" fetch origin pull/34859/head
    fi
  fi
  if ! git -C "$SOURCE_DIR" cat-file -e "${SGLANG_COMMIT}^{commit}" 2>/dev/null; then
    echo "Pinned SGLang commit is unavailable: $SGLANG_COMMIT" >&2
    exit 1
  fi
  git -C "$SOURCE_DIR" checkout --detach "$SGLANG_COMMIT"
fi

if git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "SGLang lm_head correctness patch is already applied."
elif git -C "$SOURCE_DIR" apply --check "$PATCH_FILE"; then
  git -C "$SOURCE_DIR" apply "$PATCH_FILE"
else
  echo "SGLang checkout has an unexpected source delta; refusing to continue." >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi

systemd-run --user --scope --quiet \
  -p MemoryHigh=36G \
  -p MemoryMax=40G \
  -p MemorySwapMax=0 \
  -p OOMPolicy=kill \
  -- env \
    SGLANG_BUILD_RUST_EXTS=none \
    UV_CACHE_DIR="$ROOT/.uv-cache" \
    uv pip install --python "$VENV/bin/python" --editable "$SOURCE_DIR"

systemd-run --user --scope --quiet \
  -p MemoryHigh=36G \
  -p MemoryMax=40G \
  -p MemorySwapMax=0 \
  -p OOMPolicy=kill \
  -- env UV_CACHE_DIR="$ROOT/.uv-cache" \
    uv pip install --python "$VENV/bin/python" \
      'torch==2.13.0' \
      'transformers==5.12.1' \
      'flashinfer-python[cu13]==0.6.17' \
      'sglang-kernel==0.4.6.post1'

"$VENV/bin/python" -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())'

if [[ ! -x "$TOOLS_VENV/bin/python" ]]; then
  uv venv --python 3.12 "$TOOLS_VENV"
fi
UV_CACHE_DIR="$ROOT/.uv-cache" uv pip install \
  --python "$TOOLS_VENV/bin/python" \
  -r "$ROOT/requirements-benchmark.txt"

"$TOOLS_VENV/bin/python" "$ROOT/scripts/download_models.py" --model radixark

echo "SGLang and its pinned RadixArk checkpoint are ready."
