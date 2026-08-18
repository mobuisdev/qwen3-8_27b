# Qwen3.8-27B on RTX 5090: current report

Last updated: 2026-08-18
Machine: RTX 5090 32 GiB, Ryzen 9 9950X3D, 64 GiB RAM, Nobara Linux 44

## Executive summary

| Question | Current answer |
|---|---|
| Recommended backend | **NInfer** at commit `b2b96bae4dd88f95b9ea8126d68fae3b88caa374` |
| Model | Qwen3.8-27B NVFP4 registered NInfer artifact |
| Settings | INT8 KV, MTP3, optimized LM-head draft, CUDA graphs, concurrency 1 |
| Context allocation | 200,000 total tokens |
| Exact 190k result | **155.42 tok/s median** at 190,003 prompt tokens, 768 generated |
| Short-context result | 198.01 tok/s median at 997 prompt tokens |
| Cold 190k TTFT | 67.64 seconds median |
| Cached agent-turn TTFT | **0.397 seconds** with 182,095 cached tokens |
| Peak VRAM at 190k | 29.05 GiB |
| Host RAM safety | 9.54 GiB peak used; at least 52.88 GiB available |
| Focused quality checks | Logic, code, exact JSON, tool call, and 3/3 long-context needles passed |

The 2026-08-17 SGLang baseline achieved approximately 50.5 tok/s at 200k context.
The follow-up therefore found a roughly 3x generation-speed improvement while
keeping the requested long context fully GPU-resident.

## Recommendation

Run:

```bash
./scripts/launch/launch_ninfer_5090.sh
```

Connect to:

```text
base URL: http://127.0.0.1:32000/v1
model:    qwen38-ninfer-nvfp4
```

NInfer is the recommendation for this exact single-GPU deployment because it
is specialized for the RTX 5090 and the registered Qwen checkpoint. It exposes
OpenAI Chat Completions, Responses, Anthropic Messages, streaming, usage
accounting, and parsed tool calls.

Keep conversation history append-only. NInfer's compatible-prefix cache works
at turn boundaries: the test restored 182,095 tokens and recomputed only 40.
Changing the content of an existing message caused a full cache miss.

The 200k setting is a total sequence ceiling, not a 200k-input guarantee plus
unlimited output. Budget prompt and completion together below it.

## Backend comparison

| Backend/checkpoint | MTP | Long-context capacity | Short decode | Long decode | Verdict |
|---|---:|---:|---:|---:|---|
| **NInfer NVFP4** | 3 | 200,000 configured/tested pool | 198.01 tok/s at 997 | **155.42 tok/s at 190,003** | Recommended |
| vLLM 0.27.1 + Unsloth | 3 | 121,600 | 106.93 tok/s at 1k | 91.12 tok/s at 115k | Fast fallback, insufficient context |
| vLLM 0.27.1 + Gittensor ModelOpt | off | 246,176 | 53.68 tok/s at 1k | 44.40 tok/s at 190k | Context fits, speed does not |
| SGLang + RadixArk | off | about 202,763 | 60.90 tok/s at 1k | 50.48 tok/s at 200k | 2026-08-17 baseline |
| SGLang + RadixArk | on | about 19,859 | 109.74 tok/s over 768 output | Not viable at long context | Short-context baseline |

### Why vLLM is not the winner

vLLM was installed and fully tested. Missing dependencies were solvable, not a
reason to exclude it. The isolated environment uses vLLM 0.27.1, PyTorch
2.13.0, Transformers 5.15.0, FlashInfer 0.6.16.post3, and CUDA 13.0 JIT
packages pinned to a consistent minor version.

The issue is the joint VRAM operating point:

- the Unsloth MTP configuration reaches about 100 tok/s but leaves only 4.74
  GiB for KV, limiting capacity to 121,600 tokens;
- the smaller Gittensor ModelOpt checkpoint leaves enough room for 190k but
  measured only 44–54 tok/s without MTP;
- CPU offload would trade away generation speed and increase host-memory risk.

vLLM remains the more general ecosystem choice and its guarded launcher is
included. It simply did not achieve both requested properties on this GPU.

### Why NInfer was initially less obvious

NInfer is a source-built, specialized C++/CUDA runtime with a closed set of
registered models and GPUs. It has no packaged binary or install target and
requires CUDA 13.1+, CMake/Ninja, FFmpeg development headers, libcurl headers,
and an `sm_120a` build. Those requirements were resolved and the official
20.02 GiB artifact was revision-pinned and hash-verified.

That maintenance cost is real, but it is justified by the measured result for
this exact workstation. The pinned setup script and installation guide now make
the dependency path explicit.

## NInfer performance

One request was active at a time. The code workload used greedy decoding,
thinking disabled, 768 output tokens, 1,024-token chunked prefill, INT8
group-64 KV, MTP with three draft tokens, the optimized LM-head proposal path,
and CUDA graphs. Each row is the median of three independent cold prompts.

| Actual prompt | Median decode | Range | Median TTFT | Median prefill |
|---:|---:|---:|---:|---:|
| 997 | 198.01 tok/s | 188.09–199.64 | 0.128 s | 7,908 tok/s |
| 31,418 | 191.76 tok/s | 188.36–192.59 | 4.529 s | 6,992 tok/s |
| 95,849 | 162.88 tok/s | 161.59–174.29 | 21.991 s | 4,379 tok/s |
| 182,099 | 154.55 tok/s | 154.17–161.94 | 62.755 s | 2,911 tok/s |
| **190,003** | **155.42 tok/s** | **152.53–156.52** | **67.638 s** | about 2,821 tok/s |

All runs reached the full 768-token output limit. Code-workload MTP acceptance
was approximately 74–80%, explaining why this result is faster than NInfer's
mixed-corpus aggregate. Speculative speed varies with output predictability;
free-form story generation can be slower.

### Capacity and memory

At startup NInfer reported:

```text
weights uploaded       19.73 GiB
INT8 KV cache            7.24 GiB / 200,000 tokens
free after startup       2.79 GiB
sizing headroom          1.00 GiB
```

At the exact 190,003-token prompt, client telemetry recorded 29.05 GiB peak
VRAM, 607.61 W peak board power, 71 C peak temperature, 9.54 GiB peak host RAM
used, and 52.88 GiB minimum host RAM available. Existing system swap remained
essentially flat; the server scope itself had `MemorySwapMax=0`.

### Prefix reuse

| Shape | Cache hit | Computed prefill | TTFT |
|---|---:|---:|---:|
| Existing user message mutated | 0 | full prompt | about 60.3 s |
| Stable history + appended assistant/user turns | 182,095 | 40 tokens | **0.397 s** |

The cached 256-token continuation decoded at 129.44 tok/s. For coding-agent
latency this history shape matters as much as raw cold-prefill throughput.

### Focused behavior checks

The NInfer route passed simple logic, a Python implementation with assertions,
strict minified JSON, a parsed `get_weather({"city":"Paris"})` call, and needle
retrieval at 10%, 50%, and 90% of a 176,271-token prompt.

These are smoke tests, not a full coding or reasoning evaluation. Upstream has
not yet published a complete Qwen3.8 capability evaluation for both NInfer
weight profiles. Retest representative workloads after changing the artifact,
runtime commit, KV dtype, or thinking mode.

## Host safety

Every measured-profile launcher uses a user systemd scope:

```text
MemoryHigh=36G
MemoryMax=40G
MemorySwapMax=0
OOMPolicy=kill
```

Builds/JIT work are serialized. Broad default CUDA-graph capture and
FlashInfer autotuning remain prohibited for the SGLang baseline setup because
they previously pushed total system use to roughly 61–62 GiB and destabilized
the desktop. No tested production profile uses CPU weight or KV offload.

## Limitations

- NInfer supports only registered artifacts, one RTX 5090, and startup-fixed
  capacity. It is not a general multi-GPU/continuous-batching backend.
- Generation throughput depends on speculative acceptance; 155 tok/s is the
  measured coding workload, not a promise for every response style.
- Cold 190k prefill still takes about 68 seconds. Turn-prefix reuse is required
  for an interactive long-running agent to feel fast.
- The exact software snapshots are dated. Do not silently replace commits or
  model revisions and compare new numbers as if they were the same setup.
- `llama.cpp` was not rerun in either study, so no empirical claim is made
  against the user's separate llama.cpp workflow.

## Reproducibility

For a clean machine, follow [docs/INSTALL.md](docs/INSTALL.md). For a fresh
publication run, use the fixed matrix and exact workflow in
[docs/RERUN.md](docs/RERUN.md). The compact follow-up data is in
[backend_results_2026-08-18.json](backend_results_2026-08-18.json).
The normalized SGLang baseline data is in
[benchmark_results.json](benchmark_results.json) and
[benchmark_results.csv](benchmark_results.csv).

Detailed dated reports:

- [2026-08-17 SGLang benchmark](reports/2026-08-17-sglang-benchmark.md)
- [2026-08-18 vLLM/NInfer follow-up](reports/2026-08-18-backend-follow-up.md)

Primary external references:

- [NInfer repository, requirements, capabilities, and current benchmarks](https://github.com/Neroued/ninfer)
- [NInfer Qwen3.8 NVFP4 artifact](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
- [vLLM MTP documentation](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/)
- [Independent RTX 5090 NInfer context-scaling report](https://www.reddit.com/r/LocalLLaMA/comments/1vrbyqz/qwen38_27b_speed_as_context_grows_1274/)
