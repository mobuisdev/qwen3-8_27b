# Installation and fresh-OS setup

This guide reconstructs the measured deployment from the tracked repository,
without relying on an existing virtual environment, model cache, or compiled
binary. It pins source commits, model revisions, and key direct packages, but
is not a hash-locked build: transitive packages and live registry artifacts may
change. The dated environment captures record the package sets actually tested.

## 1. Get the repository

Clone the repository and enter its root:

```bash
git clone https://github.com/mobuisdev/qwen3-8_27b.git
cd qwen3-8_27b
```

The tracked files contain the pinned source revisions, model hashes, launch
settings, compact results, and setup scripts.

If migrating an existing installation, commit or back up the repository first.

The full raw benchmark record is intentionally ignored. Before wiping an
existing system, copy these directories separately if wanted:

```text
logs/
monitoring/
raw_results/
```

Model weights and caches do not need backup unless download time matters.

## 2. Restore the NVIDIA stack

Install a driver supporting the RTX 5090 and CUDA 13.1 or newer, then install a
CUDA toolkit with `nvcc`. The measured host used driver 595.91.07 and toolkit
13.2.51; do not downgrade a newer known-good driver merely to match these
numbers.

Verify:

```bash
nvidia-smi
nvcc --version
```

NInfer is specialized for `sm_120a` and rejects other CUDA architectures.

## 3. Install build prerequisites

On Nobara/Fedora, install the equivalent of:

```bash
sudo dnf install git cmake ninja-build gcc15 gcc15-c++ \
  pkgconf-pkg-config ffmpeg-devel libcurl-devel \
  cuda-cudart-static cuda-nvtx-devel
```

If the distribution exposes only Fedora's free FFmpeg development packages,
replace `ffmpeg-devel` with:

```text
libavformat-free-devel libavcodec-free-devel libavutil-free-devel
libswscale-free-devel libswresample-free-devel
```

Required `pkg-config` modules are `libavformat`, `libavcodec`, `libavutil`,
`libswscale`, and `libcurl`. Also install Python 3.12 and
[uv](https://docs.astral.sh/uv/getting-started/installation/) for the small
benchmark and Hugging Face download environment.

Package names may change on a future Fedora release; the capabilities above
are authoritative.

## 4. Rebuild and download NInfer

From the repository root:

```bash
./scripts/setup/setup_ninfer_5090.sh
```

The script:

1. checks prerequisites rather than modifying the OS;
2. checks out NInfer commit
   `b2b96bae4dd88f95b9ea8126d68fae3b88caa374`;
3. compiles serially inside the 40 GiB/no-swap cgroup;
4. downloads the 20.02 GiB Qwen3.8 NVFP4 artifact at revision
   `d6d0b3b61a38262e57217e64e7f44cf4ce98bda1`;
5. verifies SHA-256
   `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32`.

Allow roughly 25 GiB for the source build and model artifact. More temporary
space is useful while package managers operate.

## 5. Launch and verify

```bash
./scripts/launch/launch_ninfer_5090.sh
curl http://127.0.0.1:32000/health
curl http://127.0.0.1:32000/v1/models
```

Expected client settings:

```text
base URL: http://127.0.0.1:32000/v1
model:    qwen38-ninfer-nvfp4
```

The default 200k total sequence ceiling is intentional. A 190k prompt plus its
completion must fit below that total. Keep previous conversation turns
byte-for-byte stable and append new turns so `restore_turn_checkpoint` prefix
reuse can avoid cold re-prefill.

## 6. Optional vLLM fallback

To recreate the tested vLLM 0.27.1 environment and pinned Unsloth checkpoint:

```bash
./scripts/setup/setup_vllm_5090.sh
./scripts/launch/launch_vllm_5090.sh
```

The setup downloads both pinned checkpoints used by the publication matrix.
The first launch may spend several minutes compiling one FlashInfer FP4 kernel.
It remains inside the same 40 GiB/no-swap boundary. The default MTP profile
provides about 100 tok/s but only about 121.6k tokens of KV capacity.

## 7. Optional SGLang baseline reconstruction

Run `./scripts/setup/setup_sglang_5090.sh`. It checks out the pinned
source revision, applies the tracked `patches/` correctness backport, installs
the pinned direct compatibility versions inside the RAM guard, creates the
lightweight tools environment, and downloads the pinned RadixArk checkpoint.
[compatibility-notes.md](compatibility-notes.md) and the dated SGLang report
retain the full rationale.

After reconstructing all three backends, follow [RERUN.md](RERUN.md) for the
fixed publication matrix.

## 8. Optional upstream candidate matrix

Do not update the publication environments in place. The candidate setup uses
separate source directories, virtual environments, model revisions, and cache
paths:

```bash
./scripts/setup/setup_candidate_models_5090.sh
./scripts/setup/setup_sglang_candidate_5090.sh
# Optional source-build and NInfer comparisons:
./scripts/setup/setup_vllm_candidate_5090.sh
./scripts/setup/setup_candidate_models_5090.sh --with-ninfer-groupwise
```

See [CANDIDATES.md](CANDIDATES.md) for exact revisions, launch commands,
context limits, benchmark labels, and promotion criteria. These commands add
roughly 20.2 GiB for the Gittensor target and drafter, 17.0 GiB if the optional
NInfer artifact is selected, and separate source environments for engines that
are built.

## Regenerable disk usage

These paths can be removed when reclaiming space, provided no server is
running:

| Path | Approximate measured size | Recreated by |
|---|---:|---|
| `hf-home/` | 62 GiB | setup/download scripts |
| `ninfer-models/` | 21 GiB | `scripts/setup/setup_ninfer_5090.sh` |
| `.venv` + `.venv-vllm-0.27.1` | 16.1 GiB | backend setup scripts |
| `.uv-cache/` | about 0.4 GiB of cache-only blocks | uv |
| `vendor/` | 1.2 GiB | setup scripts |
| `vllm-cache/` | 0.9 GiB | first vLLM launches |

Do not remove the tracked `models/` directory: it contains compact checkpoint
metadata used by the audit scripts, not the large weights.
