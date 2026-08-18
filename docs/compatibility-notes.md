# Compatibility notes

Checked 2026-08-17. These notes separate released support from the day-zero
source revisions required by Qwen3.8.

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

SGLang 0.5.17 is the current stable release but predates Qwen3.8. Its package
pins PyTorch 2.11.0, Transformers 5.12.1, and FlashInfer 0.6.15.post1, while
the official Qwen3.8 MTP recipe says the FlashInfer `uniform_q_len` support
needed for this path arrived afterward. FlashInfer 0.6.17 is therefore used.
Consequently the experiment uses the exact source commit and pinned direct
compatibility packages, not a nominally stable release lacking the new model.

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
