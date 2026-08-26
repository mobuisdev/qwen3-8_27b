# Compatibility notes

Baseline checked 2026-08-17; upstream status rechecked 2026-08-26. These notes
separate released support from the day-zero source revisions required by
Qwen3.8.

> These details specify the 2026-08-17 SGLang baseline. The current backend
> comparison and recommended NInfer deployment are in
> [REPORT.md](../REPORT.md); rebuild instructions are in
> [INSTALL.md](INSTALL.md).

## Selected stack

- GPU: NVIDIA RTX 5090, compute capability 12.0 (SM120).
- Driver: 595.91.07; CUDA driver/toolkit 13.2. The working driver was unchanged.
- Runtime source: SGLang Qwen3.8 support PR #34859 at
  `374a6b24f2f2b52fc131417d8d0e4e78900f7a5d`.
- Local compatibility patch: the `ParallelLMHead` dispatch change from SGLang
  PR #34904, applied narrowly in
  `vendor/sglang/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`.
- Python 3.12.12 in `.venv`; PyTorch 2.13.0+cu130; Transformers 5.12.1;
  FlashInfer 0.6.17; sglang-kernel 0.4.6.post1.

At baseline time, SGLang 0.5.17 was the current stable release but predated
Qwen3.8. Its package pins PyTorch 2.11.0, Transformers 5.12.1, and FlashInfer
0.6.15.post1, while the official Qwen3.8 MTP recipe says the FlashInfer
`uniform_q_len` support needed for this path arrived afterward. FlashInfer
0.6.17 is therefore used. Consequently the experiment uses the exact source
commit and pinned direct compatibility packages, not a nominally stable
release lacking the new model.

On 2026-08-19, Qwen3.8 support PR #34859 merged into SGLang `main` as
`8a1e6e4e461044246739b5a1ad579c8acc556a2d`. SGLang 0.5.18 was released on
2026-08-22 and now supersedes 0.5.17, but the isolated post-merge candidate
already contains the Qwen3.8 merge and passed the local target/DSpark matrix.
The dated publication and candidate environments remain pinned so their
measurements stay reproducible. PR #34904, used by the baseline as a narrow
compressed-tensors `lm_head` correctness backport, remains open. The isolated
evaluation is defined in [CANDIDATES.md](CANDIDATES.md).

The first isolated Python 3.13 attempt reached an `outlines-core` Rust source
build without a usable wheel, so the tested environment uses Python 3.12.
SGLang's optional Rust frontend extensions are disabled with
`SGLANG_BUILD_RUST_EXTS=none`; the Python HTTP frontend and native CUDA
inference kernels remain enabled.

## RTX 5090 execution choices

- FlashInfer is the official recipe's attention backend for RTX 5090; the
  TensorRT-LLM MHA path cited there is for SM100 rather than SM120. Triton is a
  valid comparison/fallback and is tested only after the FlashInfer baseline.
- FlashInfer autotuning is disabled for measured runs. On this exact day-zero
  branch it grew the server tree from roughly 6 GiB to 56 GiB RSS during
  startup, filled the 8 GiB zram device, and destabilized the desktop before
  serving a request. The captured log and 0.5-second monitor trace showed host
  OOM pressure rather than a
  CUDA OOM. Disabling autotune is a current CLI-supported safety setting, not
  an attention-backend fallback.
- Prefill CUDA graphs are disabled. Even with FlashInfer autotuning off, the
  default broad graph capture grew the server to 56.7 GiB RSS and again filled
  zram. After stable eager runs, decode-only graph capture restricted to batch
  size one was validated: it added about 0.09 GiB VRAM, took 1.42 seconds to
  capture from a warm cache, and raised RadixArk non-MTP decode from 34.54 to
  60.76 tok/s. The recommended profiles use only this bounded graph shape;
  broad/default multi-shape capture remains unsafe on this host.
- Every subsequent server is placed in a user systemd scope with
  `MemoryHigh=36G`, `MemoryMax=40G`, and `MemorySwapMax=0`. This is a host
  safety boundary, not CPU offloading: normal server RSS is about 6 GiB, and
  any regressing JIT/capture process is killed before it can evict the desktop.
- Compilation is serialized with `TORCHINDUCTOR_COMPILE_THREADS=1` and
  `MAX_JOBS=1`. The first FP4 build produced 17 CUDA objects and exceeded the
  default 300-second SGLang watchdog without breaching the memory boundary, so
  the guarded commands use a 900-second watchdog; subsequent runs use the
  cached artifacts.
- SGLang's automatic startup warm-up is skipped. On this branch it continued
  compiling/initializing after Uvicorn started while `/health` remained 503,
  again pushing total host RAM above 40 GiB. The harness substitutes one tiny,
  explicitly monitored warm-up only for workload runs; startup-only probes send
  no inference request.
- Readiness polling uses `/model_info`, not `/health`. In this branch `/health`
  performs a real generation-based probe, so polling it silently initiated the
  same first-forward Triton compilation that warm-up suppression was intended
  to defer. SGLang's own scripted-runtime code documents `/model_info` as the
  appropriate non-generating readiness probe in this mode.
- `fp8_e4m3` is requested for the KV cache. Startup logs are checked for the
  actual cache dtype and scaling behavior.
- No CPU weight or KV-cache offloading is permitted. Every command explicitly
  sets `--cpu-offload-gb 0`; host RSS, total RAM, swap, and logs are monitored.
- Concurrency and CUDA graph maximum batch size are both 1 for primary latency
  tests.
- The official SM120 recipe recommends a 2,048-token chunked prefill as a
  starting point. This experiment also tests 1,024 and 4,096 rather than
  assuming 2,048 wins.
- Hybrid Gated-DeltaNet state defaults to FP32. BF16 state is a documented
  memory/performance tradeoff and is tested separately. The lazy extra-buffer
  strategy saves one state slot during overlap scheduling.

## MTP

The source CLI and official recipe agree on integrated EAGLE MTP with three
steps, top-k 1, and four draft tokens. MTP runs additionally enable the Qwen
linear-attention replay path. Per-request responses expose proposal count,
correct draft count, acceptance rate, verify count, and accepted length; the
harness stores all of them.

## Checkpoint qualification

The detailed audit is in `models/model_comparison.md` and
`models/model_audit.json`.

- RadixArk is the checkpoint in the official RTX 5090 recipe. It uses ModelOpt
  mixed precision: FP8 attention, NVFP4 MLP and language head, with BF16 vision
  and MTP tensors.
- Unsloth explicitly labels its checkpoint as vLLM-only and not compatible
  with released SGLang because its FP8 language head was not treated as a
  quantized linear layer. Upstream issue #34895 and PR #34904 describe the
  failure and fix. Therefore all Unsloth SGLang numbers are clearly labeled as
  an experimental patched-runtime closest equivalent.

## 2026-08-20 candidate checkpoint status

- The official Qwen base, RadixArk, and Unsloth snapshot revisions used here
  are unchanged.
- Gittensor's RTX-5090 checkpoint moved from the publication revision
  `ec8ad26b9e3b33c7d05c0e5743b60f37f5139005` to candidate revision
  `0cc27958cefbbe231782ec8511de8c4eb5233348`. Its target `lm_head` is now
  NVFP4 rather than BF16, reducing the published checkpoint size from about
  20.59 GB to 18.77 GB.
- The matching Gittensor DSpark NVFP4 drafter is pinned at
  `eba1ac5a66c74902eaa95a4000a7c5eda96d8e95`. It is a short/medium-context
  candidate because target plus drafter does not fit a 190k speculative
  profile on one 32 GB GPU.
- NInfer's published NVFP4 artifact remains byte-identical at SHA-256
  `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32`.
  The separate 16.96 GiB groupwise-int artifact is an unbenchmarked candidate,
  not a replacement.

## 2026-08-26 maintenance review

- The active NInfer branch is seven commits ahead of the measured pin. Every
  changed path is evaluation code, evaluation configuration, a README, or a
  model card; no server, CUDA kernel, cache, loader, or model runtime changed.
  The measured binary therefore remains current for inference.
- The NInfer NVFP4 repository moved to revision
  `204e3d92c30d9d05f3300d2f52e443ad1edf6ddf`, but the served 20.02 GiB
  artifact remains byte-identical at the tracked SHA-256. The revision change
  publishes evaluation and model-card metadata and does not require a model
  download.
- RadixArk revision `319f741cce68d7914884900c138a1fbb70a42f30`
  explicitly reverts its language head to the same NVFP4 state as the pinned
  publication revision. Unsloth's current movement is metadata/model-card
  maintenance. Neither changes the active NInfer artifact.
- vLLM 0.28.0, Transformers 5.16.1, and SGLang 0.5.18 are newer than their
  publication environments. They remain isolated benchmark candidates rather
  than in-place upgrades because NInfer is the active backend and still has the
  best measured 190k-context result.
- DeepSeek Harness was upgraded from 0.1.0-rc.7 to 0.1.1-rc.2 at
  `b150a551b8d465e31e418e1b2eaf5e79bbb7d28e`. Its locked dependencies, source
  build, profile resolution, local resource monitor, live UI, deterministic
  model request, and ownership-aware shutdown all passed locally.

## Primary sources

- SGLang Qwen3.8 cookbook: <https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B>
- SGLang Qwen3.8 support PR: <https://github.com/sgl-project/sglang/pull/34859>
- SGLang Unsloth issue: <https://github.com/sgl-project/sglang/issues/34895>
- SGLang language-head fix PR: <https://github.com/sgl-project/sglang/pull/34904>
- SGLang releases: <https://github.com/sgl-project/sglang/releases>
- FlashInfer releases: <https://github.com/flashinfer-ai/flashinfer/releases>
- PyTorch install selector: <https://pytorch.org/get-started/locally/>
- RadixArk checkpoint: <https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4>
- Unsloth checkpoint: <https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4>
- Base Qwen model card: <https://huggingface.co/Qwen/Qwen3.8-27B>
