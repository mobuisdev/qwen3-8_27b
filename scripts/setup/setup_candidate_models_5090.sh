#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_VENV="$ROOT/.venv-tools"
MODEL_DIR="$ROOT/ninfer-models/groupwise"

GITTENSOR_REPO="gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"
GITTENSOR_REVISION="0cc27958cefbbe231782ec8511de8c4eb5233348"
DSPARK_REPO="gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4"
DSPARK_REVISION="eba1ac5a66c74902eaa95a4000a7c5eda96d8e95"
NINFER_REPO="neroued/Qwen3.8-27B-NInfer"
NINFER_REVISION="18dfc887423fa5aabf3cb56fac41490e462b3fab"
NINFER_FILE="$MODEL_DIR/qwen3_8_27b.ninfer"
NINFER_SHA256="eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e"

download_ninfer=false
if [[ "${1:-}" == "--with-ninfer-groupwise" ]]; then
  download_ninfer=true
  shift
fi
if (( $# != 0 )); then
  echo "Usage: $0 [--with-ninfer-groupwise]" >&2
  exit 2
fi

for command_name in uv sha256sum; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Missing prerequisite: $command_name (see docs/INSTALL.md)" >&2
    exit 1
  fi
done

if [[ ! -x "$TOOLS_VENV/bin/python" ]]; then
  uv venv --python 3.12 "$TOOLS_VENV"
fi
UV_CACHE_DIR="$ROOT/.uv-cache" uv pip install \
  --python "$TOOLS_VENV/bin/python" \
  -r "$ROOT/requirements-benchmark.txt"

HF_HOME="$ROOT/hf-home" "$TOOLS_VENV/bin/hf" download \
  "$GITTENSOR_REPO" --revision "$GITTENSOR_REVISION"
HF_HOME="$ROOT/hf-home" "$TOOLS_VENV/bin/hf" download \
  "$DSPARK_REPO" --revision "$DSPARK_REVISION"

if [[ "$download_ninfer" == true ]]; then
  mkdir -p "$MODEL_DIR"
  HF_HOME="$ROOT/hf-home" "$TOOLS_VENV/bin/hf" download \
    "$NINFER_REPO" qwen3_8_27b.ninfer \
    --revision "$NINFER_REVISION" --local-dir "$MODEL_DIR"
  actual_sha256="$(sha256sum "$NINFER_FILE")"
  actual_sha256="${actual_sha256%% *}"
  if [[ "$actual_sha256" != "$NINFER_SHA256" ]]; then
    echo "NInfer groupwise model SHA-256 mismatch: $actual_sha256" >&2
    exit 1
  fi
fi

echo "Candidate Gittensor target and DSpark snapshots are ready."
if [[ "$download_ninfer" == true ]]; then
  echo "Candidate NInfer groupwise artifact is ready."
else
  echo "NInfer groupwise was skipped; rerun with --with-ninfer-groupwise to add it."
fi
