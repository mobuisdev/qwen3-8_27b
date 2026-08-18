# RTX 5090 follow-up: reaching 100+ tok/s at long context

Date: 2026-08-18
Host: RTX 5090 32 GiB, Ryzen 9 9950X3D, 64 GiB RAM, Nobara 44

## Conclusion

The requested operating point is achievable on this machine. The best tested configuration is
NInfer at commit `b2b96bae4dd88f95b9ea8126d68fae3b88caa374`, using its registered Qwen3.8-27B
NVFP4 artifact, INT8 KV cache, MTP with three draft tokens, the optimized LM-head draft path, and
one active sequence.

At an **actual 190,003-token prompt**, three cold runs generated all 768 requested tokens at:

- **152.53, 156.52, and 155.42 tok/s** (median **155.42 tok/s**);
- cold TTFT of 67.64, 67.76, and 67.30 seconds;
- approximately 75.2–80.7% MTP draft acceptance according to the server log;
- peak VRAM 29.05 GiB and at least 52.88 GiB host RAM still available.

This is about 2.6–3.1 times the SGLang baseline generation rate at long context, and it clears the
100 tok/s target by a wide margin. The remaining long-context latency is cold prompt ingestion.
For a correctly shaped append-only agent conversation, compatible-prefix reuse restored 182,095
tokens, recomputed only 40, and reduced TTFT from 60.32 seconds to **0.40 seconds**.

## Why vLLM was not the final recommendation

Missing dependencies were not a fundamental blocker. An isolated vLLM 0.27.1 environment was
installed and made operational with:

- PyTorch 2.13.0, CUDA 13.0 Python packages, Transformers 5.15.0, and FlashInfer 0.6.16.post3;
- pinned `nvidia-cuda-nvcc`, `nvidia-cuda-crt`, and `nvidia-nvvm` 13.0.88 packages so the
  FlashInfer JIT compiler matched its CUDA 13.0 headers;
- the one required FP4 FlashInfer kernel compiled once with `ninja -j1` inside the 40 GiB/no-swap
  memory guard.

vLLM is a good general-purpose backend, and it did reach about 100 tok/s. It did not meet
**100 tok/s and approximately 190k context simultaneously** with either locally tested checkpoint:

| vLLM checkpoint/configuration | Usable KV/context | Prompt tokens | Median decode | Median TTFT |
|---|---:|---:|---:|---:|
| Unsloth NVFP4, MTP3, FP8 KV | 121,600 | 1,024 | 106.93 tok/s | 0.13 s |
| same | same | 32,768 | 101.25 tok/s | 4.94 s |
| same | same | 100,000 | 97.82 tok/s | 25.15 s |
| same | same | 115,000 | 91.12 tok/s | 31.47 s |
| Gittensor ModelOpt NVFP4, no MTP, FP8 KV | 246,176 | 1,024 | 53.68 tok/s | 0.07 s |
| same | same | 100,000 | 48.28 tok/s | 20.73 s |
| same | same | 190,000 | 44.40 tok/s | 64.18 s |

The Unsloth MTP state consumes enough VRAM that only 4.74 GiB remains for KV cache. The smaller
ModelOpt export leaves enough KV cache for 190k, but its decode path on this host remained around
44–54 tok/s. Raising `max_num_seqs`, using a slightly broader but still safe graph set, and matching
the model-card-style 256-token chat request did not materially change that result. Its advertised
80.6 tok/s short-context result was not reproduced here, and vLLM also warned that some FP8
attention scale values were absent or fell back to 1.0. It is therefore not the preferred checkpoint
for this machine without a fuller quality evaluation.

The vLLM result should be read as a VRAM trade-off, not an installation failure:

- MTP on: near the speed target, but about 121k maximum context;
- MTP off with the smaller export: more than 190k context, but near the baseline speed;
- CPU offload was deliberately excluded because it would reduce speed and increase host-memory
  risk.

## NInfer measured results

The main workload used one OpenAI-compatible streaming request at a time, greedy decoding,
thinking disabled, a code-generation instruction, 768 output tokens, a 1,024-token prefill chunk,
INT8 group-64 KV, CUDA graphs, MTP3, and no prefix reuse between independent samples.

The first four rows below are three repetitions each. The final row is the separately calibrated
three-run test that renders to exactly 190,003 prompt tokens.

| Actual prompt | Median decode | Range | Median TTFT | Median prefill | Aggregate MTP acceptance |
|---:|---:|---:|---:|---:|---:|
| 997 | 198.01 tok/s | 188.09–199.64 | 0.13 s | 7,908 tok/s | 75.9% |
| 31,418 | 191.76 tok/s | 188.36–192.59 | 4.53 s | 6,992 tok/s | 79.6% |
| 95,849 | 162.88 tok/s | 161.59–174.29 | 21.99 s | 4,379 tok/s | 74.1% |
| 182,099 | 154.55 tok/s | 154.17–161.94 | 62.76 s | 2,911 tok/s | 78.1% |
| **190,003** | **155.42 tok/s** | **152.53–156.52** | **67.64 s** | about 2,821 tok/s | 75.2–80.7% |

All samples reached the full 768-token output limit. NInfer resolved a fixed 200,000-token KV pool:

```text
weights uploaded       19.73 GiB
INT8 KV cache            7.24 GiB / 200,000 tokens
free after startup       2.79 GiB
sizing headroom          1.00 GiB
```

Client telemetry at 190,003 tokens recorded:

```text
peak VRAM                29.05 GiB
peak GPU utilization       100%
peak board power         607.61 W
peak GPU temperature       71 C
peak host RAM used         9.54 GiB
minimum host RAM free     52.88 GiB
```

The workstation already had roughly 4.16 GiB of system swap occupied, but it stayed essentially
flat. The server itself ran with `MemorySwapMax=0` and therefore could not add swap pressure.

### Prefix reuse

Two prefix tests were intentionally distinguished:

1. Mutating the end of an existing user message produced no cache hit and repeated the full
   60-second prefill.
2. Keeping prior turns byte-for-byte stable, appending the first assistant answer and a new user
   turn produced `restore_turn_checkpoint`, restored 182,095 tokens, and computed 40 new tokens.

The second shape is representative of a coding agent. Its TTFT was 0.397 seconds and its 256-token
continuation decoded at 129.44 tok/s. Agent clients should append messages and avoid rewriting earlier
system/user/assistant turns if they want this benefit.

### Focused behavior checks

All focused checks passed through NInfer's HTTP server:

| Check | Result |
|---|---|
| Simple logic | pass |
| Python implementation with asserts | pass |
| Exact JSON | pass |
| Parsed `get_weather({"city":"Paris"})` tool call | pass |
| Needle at 10% of a 176,271-token prompt | pass |
| Needle at 50% | pass |
| Needle at 90% | pass |

These are smoke checks, not a replacement for a full coding/reasoning benchmark. NInfer upstream
has not yet published its full Qwen3.8 capability evaluation, so the backend should still be
treated as a specialized, rapidly moving runtime.

## Internet and forum findings

The investigation initially found a LocalLLaMA report from another RTX 5090/NInfer user showing
169.7 tok/s below 50k, 156.9 tok/s at 100–150k, and 149.0 tok/s at 150–200k across 1,274 agent
generations. The local 190k median of 155.42 tok/s is consistent with that report rather than merely
repeating its claim:

- [LocalLLaMA: Qwen3.8 27B speed as context grows](https://www.reddit.com/r/LocalLLaMA/comments/1vrbyqz/qwen38_27b_speed_as_context_grows_1274/)

Current NInfer upstream explicitly supports the Qwen3.8-27B NVFP4 artifact, MTP, INT8 KV,
compatible-prefix reuse, OpenAI Chat/Responses APIs, Anthropic Messages, and parsed tool calls. Its
published Qwen3.8 figures span 151.4–195.2 tok/s for long reasoning and 194.3 tok/s for its code
category, depending strongly on MTP acceptance:

- [NInfer repository and current benchmark tables](https://github.com/Neroued/ninfer)
- [NInfer Qwen3.8 NVFP4 artifact](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)

vLLM's documentation confirms why MTP can improve decode without changing the verifier's output
distribution: draft tokens are accepted only after target-model verification. The benefit depends
on workload acceptance, which is why code/structured output can be much faster than free-form
story generation:

- [vLLM MTP documentation](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/)
- [vLLM speculative decoding overview](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

## Installed and built locally

No RPM was installed system-wide: interactive `sudo` was unavailable, so the exact Fedora 44
FFmpeg development/runtime RPMs and the missing CUDA 13.2 static runtime RPM were downloaded and
unpacked under `vendor/`. CMake was pointed at that local sysroot. The
header-only NVTX3 headers came from the already isolated vLLM CUDA package.

NInfer was compiled with GCC/G++ 15, CUDA 13.2.51, CMake, Ninja, and one build job inside the same
40 GiB/no-swap cgroup. Its official 21,492,695,040-byte model artifact was verified as:

```text
bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32
```

The OS CUDA packages, NVIDIA driver, and FFmpeg runtime were not replaced.

## Recommended everyday launch

Run:

```bash
./scripts/launch/launch_ninfer_5090.sh
```

Then point the coding agent at:

```text
base URL: http://127.0.0.1:32000/v1
model:    qwen38-ninfer-nvfp4
```

The launcher defaults to the measured configuration: 200k logical context/KV capacity, MTP3,
INT8 KV, one active sequence, vision disabled, compatible-prefix reuse enabled, and localhost-only
binding. It hard-limits the process to 40 GiB host RAM, starts reclaim pressure at 36 GiB, forbids
server swap, and kills only the server scope on OOM.

Useful overrides include:

```bash
QWEN_NINFER_PORT=32001 ./scripts/launch/launch_ninfer_5090.sh
QWEN_NINFER_MAX_CONTEXT=131072 ./scripts/launch/launch_ninfer_5090.sh
QWEN_NINFER_DRAFT_TOKENS=0 ./scripts/launch/launch_ninfer_5090.sh
```

Do not run NInfer, SGLang, and vLLM at the same time on this GPU. For a production agent, retain
append-only chat/Responses history so NInfer can restore its turn checkpoint. A 190k prompt plus
completion must remain below the configured 200k total sequence ceiling.

## Result artifacts

- `backend_results_2026-08-18.json`: compact measurements used by this report
- `scripts/launch/launch_ninfer_5090.sh`: measured NInfer profile
- `scripts/launch/launch_vllm_5090.sh`: measured vLLM profile
- `docs/RERUN.md`: complete publication matrix and aggregation workflow
