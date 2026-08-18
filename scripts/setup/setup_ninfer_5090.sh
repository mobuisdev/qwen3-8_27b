#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="$ROOT/vendor/ninfer"
BUILD_DIR="$SOURCE_DIR/build"
TOOLS_VENV="$ROOT/.venv-tools"
MODEL_DIR="$ROOT/ninfer-models"
MODEL_FILE="$MODEL_DIR/qwen3_8_27b_nvfp4.ninfer"

NINFER_COMMIT="b2b96bae4dd88f95b9ea8126d68fae3b88caa374"
MODEL_REPO="neroued/Qwen3.8-27B-nvfp4-NInfer"
MODEL_REVISION="d6d0b3b61a38262e57217e64e7f44cf4ce98bda1"
MODEL_SHA256="bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32"

for command_name in git cmake ninja nvcc pkg-config uv systemd-run sha256sum; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Missing prerequisite: $command_name (see docs/INSTALL.md)" >&2
    exit 1
  fi
done

for pkg_module in libavformat libavcodec libavutil libswscale libswresample libcurl; do
  if ! pkg-config --exists "$pkg_module"; then
    echo "Missing development package for pkg-config module: $pkg_module" >&2
    echo "Install the packages listed in docs/INSTALL.md, then rerun." >&2
    exit 1
  fi
done

mkdir -p "$ROOT/vendor" "$ROOT/.ccache" "$MODEL_DIR"

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/Neroued/ninfer.git "$SOURCE_DIR"
fi

if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]] && \
   [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" != "$NINFER_COMMIT" ]]; then
  echo "Refusing to replace a modified NInfer checkout at $SOURCE_DIR" >&2
  exit 1
fi

git -C "$SOURCE_DIR" fetch origin "$NINFER_COMMIT"
git -C "$SOURCE_DIR" checkout --detach "$NINFER_COMMIT"

CC_BIN="${CC:-$(command -v gcc-15 || command -v gcc)}"
CXX_BIN="${CXX:-$(command -v g++-15 || command -v g++)}"

CCACHE_DIR="$ROOT/.ccache" cmake \
  -S "$SOURCE_DIR" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC_BIN" \
  -DCMAKE_CXX_COMPILER="$CXX_BIN"

CCACHE_DIR="$ROOT/.ccache" systemd-run --user --scope --quiet \
  -p MemoryHigh=36G \
  -p MemoryMax=40G \
  -p MemorySwapMax=0 \
  -p OOMPolicy=kill \
  -- cmake --build "$BUILD_DIR" --parallel 1

if [[ ! -x "$TOOLS_VENV/bin/python" ]]; then
  uv venv --python 3.12 "$TOOLS_VENV"
fi
UV_CACHE_DIR="$ROOT/.uv-cache" uv pip install \
  --python "$TOOLS_VENV/bin/python" \
  -r "$ROOT/requirements-benchmark.txt"

HF_HOME="$ROOT/hf-home" "$TOOLS_VENV/bin/hf" download \
  "$MODEL_REPO" \
  qwen3_8_27b_nvfp4.ninfer \
  --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR"

actual_sha256="$(sha256sum "$MODEL_FILE")"
actual_sha256="${actual_sha256%% *}"
if [[ "$actual_sha256" != "$MODEL_SHA256" ]]; then
  echo "Model SHA-256 mismatch: $actual_sha256" >&2
  exit 1
fi

echo "NInfer is ready. Start it with:"
echo "  $ROOT/scripts/launch/launch_ninfer_5090.sh"
