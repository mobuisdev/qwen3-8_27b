# Qwen3.8-27B on one RTX 5090

This repository is a pinned local-serving reconstruction kit and benchmark
archive for Qwen3.8-27B on an RTX 5090 32 GiB workstation with 64 GiB host RAM.

The result archive and supplied backend launchers target this exact class of
machine. Anyone with an OpenAI-compatible local server can reuse the neutral
benchmark clients on another configuration; see [scripts/README.md](scripts/README.md).

The current recommendation is **NInfer + Qwen3.8-27B NVFP4 + INT8 KV + MTP3**.
It measured **155.42 tok/s median at an actual 190,003-token prompt**. Start with
[REPORT.md](REPORT.md) for the comparison and caveats.

## Use the current installation

```bash
./scripts/launch/launch_ninfer_5090.sh
```

Connect an OpenAI-compatible client to:

```text
base URL: http://127.0.0.1:32000/v1
model:    qwen38-ninfer-nvfp4
```

The launcher is localhost-only and places the server in a user cgroup with
`MemoryHigh=36G`, `MemoryMax=40G`, `MemorySwapMax=0`, and `OOMPolicy=kill`.
This protects the desktop from the host-RAM failure mode found during the first
benchmark.

For a fresh OS, follow [docs/INSTALL.md](docs/INSTALL.md), or after installing
its system prerequisites run:

```bash
./scripts/setup/setup_ninfer_5090.sh
./scripts/launch/launch_ninfer_5090.sh
```

For a clean post-reboot publication run across all five final profiles, follow
[docs/RERUN.md](docs/RERUN.md). It includes preflight checks and generates
Markdown/CSV tables from only the new session.

## What is tracked

| Path | Purpose |
|---|---|
| `REPORT.md` | Canonical result and recommendation |
| `docs/` | Installation, methodology, rerun workflow, and compatibility notes |
| `scripts/setup/` | Pinned backend provisioning |
| `scripts/launch/` | RAM-guarded measured-profile launchers |
| `reports/archive/2026-08-18-backend-follow-up/results-summary.json` | Compact machine-readable follow-up results |
| `reports/archive/2026-08-17-sglang/results.{json,csv}` | Normalized SGLang baseline results |
| `reports/archive/` | Fixed historical studies and their supporting data |
| `reports/runs/<session>/` | One self-contained publication directory per rerun |
| `scripts/*.py` | Benchmark, quality, preflight, capture, and metadata tools |
| `requirements-benchmark.txt` | Pinned direct dependencies for the neutral clients |
| `models/` | Small pinned model metadata and checkpoint audit; no weights |
| `reports/**/system-info.txt`, `reports/**/*environment.txt` | Public-safe dated hardware and software snapshots |

The three `scripts/launch/launch_sglang_*_5090.sh` files reproduce the
published SGLang profiles without including one-off exploratory command
captures.

## What is deliberately not tracked

The following are downloaded or generated and can be recreated:

- `.venv*/` and `.uv-cache*/`;
- `vendor/` source checkouts and builds;
- `hf-home/` and `ninfer-models/` model data;
- `vllm-cache/`, `sglang-cache/`, `flashinfer-workspace/`, and
  `cuda-header-shim/`;
- full `logs/`, `monitoring/`, and `raw_results/` output.

On the measured machine these occupy about 101 GiB, primarily 62 GiB of
Hugging Face data, the 20 GiB NInfer artifact, and 16.1 GiB across the separate
SGLang and vLLM environments. `du` shows the shared uv cache as roughly 8 GiB
because most package files are hard-linked into a virtual environment; the
blocks owned only by `.uv-cache/` are about 390 MiB while those environments
still exist.
They are kept locally for immediate use but excluded from Git.

The Python environments are intentionally separate because their dependency
sets are not interchangeable:

| Path | Purpose |
|---|---|
| `.venv` | Pinned SGLang source build and runtime |
| `.venv-vllm-0.27.1` | Pinned vLLM runtime |
| `.venv-tools` | Lightweight benchmark and model-download clients |

Combining the backend environments would allow one engine to replace the
other's Transformers, FlashInfer, CUDA, or kernel packages. All three share
one `.uv-cache/` for downloads without sharing installed packages.

Before wiping the OS, separately copy `logs/`, `monitoring/`, and `raw_results/`
if the full forensic record matters. The compact reported results and all
reconstruction inputs are tracked in Git.

Raw server logs can contain local paths, device identifiers, and process
details. They are ignored deliberately: redact them before publishing or
attaching them to an issue.

## Backend choices

- **NInfer:** preferred for this exact GPU/model; approximately 155 tok/s at
  190k context and working turn-prefix reuse.
- **vLLM:** useful general backend; approximately 107 tok/s with MTP but only
  121.6k KV capacity, or 44 tok/s at 190k with the smaller no-MTP checkpoint.
- **SGLang:** 2026-08-17 baseline; approximately 50.5 tok/s at 200k. Its pinned,
  source-patched setup remains available for comparison.

Do not run more than one backend at once on this GPU.

## Reproducibility boundary

The setup scripts pin source commits, model revisions, and the direct package
versions that affected compatibility. They do not provide a hash-locked
transitive dependency graph and they depend on live Git and package/model
registries. They are therefore reconstruction recipes, not hermetic or
byte-for-byte reproducible builds. The dated environment captures under
`reports/archive/` and `reports/runs/` record the complete package set used for
the published measurements.

## License

Code and documentation developed for this benchmark are available under
the [MIT License](LICENSE). Third-party model metadata and the SGLang patch
retain their upstream licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
