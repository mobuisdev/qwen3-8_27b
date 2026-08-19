# Qwen3.8-27B NVFP4 on RTX 5090: benchmark report

Date: 2026-08-17
Machine: RTX 5090 32 GB, Ryzen 9 9950X3D, 64 GB RAM, Nobara Linux 44

## Executive summary

| Question | Result |
|---|---|
| Best model | **RadixArk/Qwen3.8-27B-NVFP4** |
| Best overall settings | RadixArk, MTP off, FP8 E4M3 KV, BF16 Mamba state, four Mamba slots, FlashInfer, chunk 2,048, 0.91 static fraction, batch-1 decode graph |
| MTP worth using? | **Yes for short-context maximum speed; no for long-context serving** |
| Maximum stable configured context | 262,144 tokens |
| Runtime-reported GPU KV capacity | 202,763 tokens for the optimized non-MTP profile |
| Maximum repeated real prompt | 200,000 tokens, three repetitions |
| Maximum repeated prompt + output | **200,000 + 1,024 = 201,024 tokens** |
| GPU-only 262K achieved? | **No**; 262K can be configured but the physical GPU pool is ~203K |
| Best decode result | 126.02 tok/s median over 256 tokens; 109.74 tok/s median over 768 tokens (RadixArk MTP) |
| Everyday decode | 60.90 tok/s over 768 tokens; 50.48 tok/s at a 200K prompt |
| Best measured prefill | 11,729 tok/s at 8K; 2,956 tok/s at 200K |
| Winning-run peak VRAM | 30.41 GiB at 200K + 1,024 |
| Winning-run peak host RAM | 10.81 GiB |
| Swap during winning requests | Zero swap-out; small page-ins from pre-existing swap |
| CPU offloading detected | No; `--cpu-offload-gb 0`, GPU KV, no offload log markers |

The best everyday and maximum-context profile is the same RadixArk non-MTP
configuration. Its four-slot Mamba pool is the most important capacity
optimization: it raises reported KV capacity from about 114K to about 203K by
avoiding dozens of unused cached Mamba states at concurrency one. It processed
200K input plus 1,024 output three times without OOM, CPU offload, or swap-out.

The maximum-speed profile enables integrated EAGLE MTP. With batch-1 graphs it
delivered 109.74 tok/s over a 768-token output, 80.2% faster than the matching
non-MTP long-output result. The price is severe: the 5.53 GiB draft model leaves
only about 19.9K KV tokens. MTP is therefore workload-dependent, not a universal
default.

The experiment also found a quality qualification that must not be hidden.
Both FP8-KV runs warn that no numeric scale factors were supplied, so SGLang
uses 1.0. Long-context needle failures usually emitted immediate EOS rather
than a wrong fact. Explicit BF16 KV improved one 64K case but did not eliminate
the pattern. Treat ~200K as allocation/performance validated, not uniformly
quality validated; for quality-sensitive use, retest representative documents
and consider the conservative BF16-KV profile.

## Validity and compatibility

This is day-zero source support, not a released stable SGLang configuration.
The exact runtime is SGLang Qwen3.8 PR #34859 at commit
`374a6b24f2f2b52fc131417d8d0e4e78900f7a5d`, plus the narrow
`ParallelLMHead` fix from PR #34904 required by Unsloth. SGLang reports
`0.0.0.dev1+g374a6b24f.d20260817`.

Selected packages are Python 3.12.12, PyTorch 2.13.0+cu130, CUDA runtime 13.0,
host CUDA toolkit/driver interface 13.2, driver 595.91.07, Transformers 5.12.1,
FlashInfer 0.6.17, sglang-kernel 0.4.6.post1, compressed-tensors 0.18, and
Triton 3.7.1. PyTorch reports compute capability 12.0/SM120 and a direct CUDA
allocation and matrix multiplication passed.

RadixArk is the checkpoint recommended by the official Qwen3.8 SGLang recipe.
Unsloth's own model card says its checkpoint does not work with released
SGLang. Consequently all Unsloth results here are **experimental patched-runtime
results**, not proof of official support. See
[compatibility-notes.md](../../../docs/compatibility-notes.md) and
[sglang-environment.txt](sglang-environment.txt) for
sources and the complete package freeze.

## Machine and baseline

| Item | Observed |
|---|---|
| OS | Nobara Linux 44 KDE, native Linux; kernel 7.1.4-200.nobara.fc44 |
| CPU | Ryzen 9 9950X3D, 16 cores / 32 threads |
| Physical RAM | 64 GiB installed; 62.4 GiB visible |
| Swap | 8 GiB zram |
| GPU | GeForce RTX 5090, 32,607 MiB, SM120 |
| Idle display VRAM | Approximately 1.4–1.6 GiB |
| PCIe | Maximum Gen5 x16; idle sample Gen1 x8 (power-saving state) |
| Disk/cache | Btrfs on `/home`; about 157 GiB free before model preparation |
| Host Python | 3.14.6; isolated benchmark Python is 3.12.12 |

The GPU stayed resident: model weights, Mamba states, and full-attention KV
were CUDA allocations. Host RAM handled loading, tokenization, compilation,
page cache, and monitoring only. No `llama.cpp` installation was changed.

## Checkpoint audit

| Property | RadixArk | Unsloth |
|---|---:|---:|
| Revision | `554ebba9b5f1b79dc11246341960360e6ef05ef4` | `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` |
| Repository bytes | 21,945,295,265 | 23,444,511,832 |
| Indexed weight bytes | 21,921,428,072 | 23,417,592,488 |
| Runtime weight memory, target | 20.14 GiB | 21.48 GiB |
| Quantization loader | ModelOpt mixed | compressed-tensors mixed |
| Language MLP | Primarily NVFP4 W4A4 | NVFP4 except last eight FP8 MLP blocks |
| Attention | FP8 W8A8 | FP8 W8A8 |
| `lm_head` | NVFP4 | FP8; needs the local SGLang fix |
| MTP draft weight memory | 5.53 GiB | 4.35 GiB |
| Native context / MTP layers | 262,144 / 1 | 262,144 / 1 |

Both are mixed checkpoints, not uniformly W4A4. Startup logs verify the
ModelOpt/compressed-tensors loaders, fused FP4 dense MLP path, actual FlashInfer
attention, Triton GDN kernels, and actual KV dtype. Detailed tensor targets and
hashes are in [model_comparison.md](../../../models/model_comparison.md) and
[model_audit.json](../../../models/model_audit.json).

## Methodology

Every primary performance point used one unreported warm-up and three measured
requests at concurrency one. The client flushes the radix cache between runs,
uses exact tokenizer-generated input lengths, fixed seeds, `top_k=1`, and
`ignore_eos=true` for performance. TTFT is client-observed HTTP start to first
completion token, so prefill throughput (`prompt tokens / TTFT`) includes HTTP,
scheduling, and prefill. Decode throughput excludes the first output token.

Monitoring sampled NVML and host memory every 0.5 seconds. Each measured result
stores raw SGLang metadata, command, log, VRAM, power, temperature, host RAM,
and swap I/O. `results.csv` and `results.json` contain the normalized rows;
[METHODOLOGY.md](../../../docs/METHODOLOGY.md) describes prompt
construction.

## Four primary configurations

This table uses the same 1,024-token prompt, 256-token output, FP8 KV, BF16
Mamba state, FlashInfer, chunk 2,048, static fraction 0.91, batch-1 graphs, and
three repetitions. Values are medians except peak columns. RadixArk non-MTP
also includes the validated four-slot capacity optimization.

| ID | Model | MTP | Runtime KV capacity | TTFT ms | Prefill tok/s | Decode tok/s | Peak VRAM GiB | Peak host GiB | Stable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | RadixArk | Off | 202,746 | 85.6 | 11,966 | 60.82 | 30.04 | 10.82 | Yes |
| B | RadixArk | On | 19,507 | 91.1 | 11,244 | **126.02** | 25.62 | 10.83 | Yes |
| C | Unsloth (patched) | Off | 90,601 | 93.4 | 10,959 | 42.61 | 30.04 | 10.98 | Yes |
| D | Unsloth (patched) | On | 16,624 | 103.9 | 9,857 | 84.78 | 26.76 | 11.15 | Yes |

RadixArk is 42.7% faster than Unsloth without MTP and 48.6% faster with MTP in
this graph-enabled comparison. Unsloth is also 1.34 GiB larger on disk, uses
1.34 GiB more target-weight VRAM, and exposes less KV capacity.

The low total MTP VRAM peaks do **not** mean MTP is memory-free. RadixArk loads
5.53 GiB extra draft weights and Unsloth 4.35 GiB. Total observed usage is
lower only because hybrid-state constraints collapse the KV pool. At 0.85
RadixArk MTP cannot allocate any state cache; at 0.90/BF16 it gets three slots
but needs four. It first becomes viable at 0.91/BF16.

## MTP conclusion

| Metric | RadixArk off | RadixArk MTP | Change |
|---|---:|---:|---:|
| 256-token decode, graph | 60.82 | 126.02 | +107.2% |
| 768-token decode, graph | 60.90 | 109.74 | +80.2% |
| 768-token median acceptance | — | 49.4% | 2.49 accepted length |
| TTFT, 256-token run | 85.6 ms | 91.1 ms | +6.4% |
| Runtime KV capacity | 202,746 | 19,507–19,859 | about -90.2% |
| Extra draft weights | — | 5.53 GiB | material |

MTP is emphatically worthwhile for single-user decode when the working context
is below roughly 16K and maximum speed matters. It is unsuitable for the
everyday/long-context profile because it cuts usable capacity by about 90%.
RadixArk MTP passed all six 8K/16K needle checks. Unsloth MTP also passed all
six needles but its coding response exhausted 384 tokens in reasoning and did
not emit the requested code.

## Context capacity and stress results

| Configuration | Configured limit | Reported capacity | Largest repeated input | Output | Result |
|---|---:|---:|---:|---:|---|
| RadixArk off, auto Mamba pool | 262,144 | 114,165 | 98,304 | 16 | 3/3 |
| RadixArk off, four Mamba slots | 262,144 | 202,763 | **200,000** | **1,024** | **3/3** |
| RadixArk MTP | 32,768 | 19,859 | 16,384 quality prompt | up to 128 | Stable |
| Unsloth off, auto Mamba pool | 262,144 | 91,695 | 81,920 | 16 | 3/3 |
| Unsloth MTP | 32,768 | 17,052 | 16,384 quality prompt | up to 128 | Stable |
| RadixArk, BF16 KV, four slots | 262,144 | 101,373 | 98,304 quality prompt | up to 128 | Stable |

The final 200K + 1,024 stress run filled about 99% of the token pool. Across
three repetitions its median TTFT was 67.65 s, prefill 2,956 tok/s, decode
50.48 tok/s, and total time 87.91 s. Peak VRAM was 30.41 GiB, minimum free VRAM
1.43 GiB, peak host RAM 10.81 GiB, peak board power 597 W, peak temperature
67 C, and swap-out zero. All three completions stopped at the requested 1,024
tokens. No CPU offload was present.

Therefore **262,144 GPU-only tokens were not achieved**. The largest repeatedly
stable actual total is 201,024. A 240K–250K stress prompt was not attempted
because the runtime had already measured the physical pool at only ~203K; it
would be a known-invalid allocation, not useful evidence.

### Prompt-ingestion scaling

Non-MTP, FP8 KV, 0.91, FlashInfer, chunk 2,048, medians of three:

| Prompt | RadixArk tok/s | RadixArk TTFT | Unsloth tok/s | Unsloth TTFT |
|---:|---:|---:|---:|---:|
| 1,024 | 9,016 | 0.114 s | 8,206 | 0.125 s |
| 8,192 | 11,729 | 0.698 s | 9,691 | 0.845 s |
| 32,768 | 8,616 | 3.803 s | 7,397 | 4.430 s |
| 65,536 | 6,160 | 10.639 s | 5,514 | 11.884 s |
| 81,920 | — | — | 4,888 | 16.761 s |
| 98,304 | 4,779 | 20.569 s | — | — |
| 131,072, four slots | 3,904 | 33.574 s | — | — |
| 196,608, four slots | 2,842 | 69.186 s | — | — |
| 200,000 + 1,024 stress | 2,956 | 67.652 s | — | — |

RadixArk leads Unsloth by about 10% at 1K, 21% at 8K, 17% at 32K, and 12% at
64K. Differences exceed normal run-to-run noise.

## Selective tuning

### Attention backend

FlashInfer beat Triton at every shared point. Median prefill was 9,016 versus
8,707 tok/s at 1K (+3.6%), 11,729 versus 10,973 at 8K (+6.9%), and 8,616 versus
7,722 at 32K (+11.6%). At 1K, eager decode was 34.54 versus 33.30 tok/s (+3.7%).
Triton saved roughly 0.4 GiB peak VRAM but did not improve capacity. FlashInfer
is selected.

### Chunked prefill

| Chunk | 8K tok/s | 32K tok/s | 64K tok/s | Peak VRAM GiB |
|---:|---:|---:|---:|---:|
| 1,024 | 10,496 | 7,487 | 5,285 | 29.98 |
| **2,048** | **11,729** | **8,616** | **6,160** | 30.28 |
| 4,096 | 10,942 | 8,213 | 5,983 | 30.90 |

The official 2,048 starting point remains best on this machine. A 1,024 chunk
saves about 0.30 GiB but is substantially slower; 4,096 is slower and uses
more VRAM.

### CUDA graphs

The safe batch-1 decode graph increased RadixArk non-MTP from 34.54 to 60.76
tok/s (+75.9%). Capture took 1.42 s and 0.09 GiB. RadixArk MTP increased from
58.51 to 126.02 tok/s in the 256-token test. Only decode/verify/draft graphs at
batch one are enabled; prefill graphs remain disabled.

The broad default graph capture is unsafe on this 64 GiB desktop: it captured
42 shapes and contributed to roughly 61 GiB total host use and full zram.
Never substitute default graph settings for the validated batch-1 flags.

### Static fraction and Mamba pool

0.85 is safe for non-MTP but provides only about 85K RadixArk KV tokens. MTP
does not fit there. At 0.90/BF16, MTP has three state slots but needs four. A
0.91 fraction is the first viable matched MTP setting and leaves adequate
runtime/display headroom: the final stress test still had 1.43 GiB free VRAM.
More aggressive 0.92/0.95 settings were not pursued after 0.91 plus explicit
Mamba sizing reached 201K real tokens; risking the display GPU for a known
sub-262K marginal gain would violate the stability objective.

At concurrency one, auto Mamba sizing reserves 20 FP32 or 41 BF16 slots. An
overlap-safe request requires four. `--max-mamba-cache-size 4` lowers Mamba
state to 0.35 GiB and raises FP8 KV capacity from ~114K to ~203K. Three slots
are invalid; SGLang rejects them. `extra_buffer_lazy` also edges `extra_buffer`
and `no_buffer` in capacity while retaining overlap scheduling.

### KV dtype

FP8 E4M3 doubles capacity relative to BF16: 202,763 versus 101,373 tokens with
the four-slot pool. However, both checkpoint logs warn that no numeric FP8 KV
scales are present and default to 1.0. Explicit BF16 removes that warning and
improved RadixArk 64K needle retrieval from 1/3 to 2/3, but 96K remained 2/3.
FP8 is used for speed/capacity profiles with an explicit quality caveat;
BF16 is supplied as a conservative reference.

## Quality sanity checks

The four small checks cover logic, coding, instruction content, and factual
explanation. Content-term results were 4/4 for both non-MTP checkpoints and
RadixArk MTP, and 3/4 for Unsloth MTP. The latter spent the entire 384-token
coding budget reasoning and never emitted code. All variants included visible
reasoning before the requested minified JSON, so the JSON substring was right
but strict “exactly one JSON object” compliance failed.

| Configuration | 8K | 16K | 32K | 64K | 80/96K | 128K | 192K |
|---|---:|---:|---:|---:|---:|---:|---:|
| RadixArk off, FP8 KV | — | — | 3/3 | 1/3 | 2/3 at 96K | 1/3 | 0/3 |
| RadixArk off, BF16 KV | — | — | — | 2/3 | 2/3 at 96K | — | — |
| RadixArk MTP | 3/3 | 3/3 | — | — | — | — | — |
| Unsloth off, FP8 KV | — | — | 3/3 | 3/3 | 1/3 at 80K | — | — |
| Unsloth MTP | 3/3 | 3/3 | — | — | — | — | — |

Needles were inserted at 10%, 50%, and 90%. Most failures were a one-token
immediate EOS/empty response, not a fabricated code. The non-monotonic pattern
and fixed deterministic seeds make this a sanity check, not an academic
long-context evaluation. Still, it is sufficient to reject an unqualified
claim of robust 192K retrieval.

## Host RAM, swap, and crash diagnosis

| Item | Result |
|---|---|
| Physical RAM | 64 GiB installed / 62.4 GiB visible |
| Typical winning-run system RAM | About 10–11 GiB used |
| Winning-run peak | 11.15 GiB across primary comparisons; 10.81 GiB final stress |
| Unsafe startup peak | Roughly 61–62 GiB total system use |
| Peak swap observed overall | About 5.6 GiB during unsafe JIT/graph experiments |
| Winning-run pre-existing swap | About 3.3–3.7 GiB, left from earlier pressure |
| Winning measured swap-out | 0 bytes in every reported primary request |
| CPU offload | None detected |

The apparent terminal/application crashes were host-memory exhaustion caused by
FlashInfer autotuning, broad first-run compilation, default multi-shape CUDA
graph capture, and a generating `/health` readiness probe. They were not a
terminal bug and not CUDA OOM.

The reproducible prevention is now built into every recommended script:

- `systemd-run --user --scope` with `MemoryHigh=36G`, `MemoryMax=40G`,
  `MemorySwapMax=0`, and `OOMPolicy=kill`;
- `TORCHINDUCTOR_COMPILE_THREADS=1` and `MAX_JOBS=1`;
- `--disable-flashinfer-autotune`;
- prefill graphs disabled and decode graphs restricted to batch one;
- `--skip-server-warmup`, with non-generating `/model_info` readiness in the
  harness;
- no CPU offload and concurrency one.

The first serialized FP4 build took longer than SGLang's 300-second watchdog,
but finished its 17 CUDA objects safely and is cached. The runner uses a
900-second watchdog. Subsequent starts and requests do not repeat that build.

The 64 GiB RAM was useful for downloading, model loading/staging, tokenization,
the filesystem cache, and compilation. It was never counted as VRAM and did not
constrain valid runs. Existing swap was not forcibly cleared because that would
be a disruptive system action; per-request swap I/O is recorded instead.

## Recommendations

### Recommended everyday / best overall

Run:

```bash
./scripts/launch/launch_sglang_long_5090.sh
```

This is RadixArk non-MTP with FP8 KV, four BF16 Mamba slots, FlashInfer, chunk
2,048, 0.91 static fraction, and batch-1 decode graphs. Expect about 60.9 tok/s
after a short prompt, about 50.5 tok/s at 200K, 11.7K prefill tok/s at 8K, and
about 3.0K at 200K. Although the benchmark configured 262,144 and measured a
physical pool of about 202.8K, the recommended launcher is deliberately capped
at 200,000 combined tokens. Use about 190K input plus 8K output for routine
maximum-context work. Peak tested VRAM is 30.41 GiB.

### Maximum speed

Run:

```bash
./scripts/launch/launch_sglang_fast_5090.sh
```

This enables integrated EAGLE MTP and the three validated batch-1 graph paths.
Expect about 110 tok/s over a 768-token output, with shorter high-acceptance
runs reaching about 126 tok/s. Runtime KV capacity is only about 19.9K; keep
real prompts near or below 16K to reserve output space. MTP is enabled, FP8 KV
and FlashInfer are active, and no CPU offload is used.

### Maximum context

Use the everyday script. It is also the maximum-context winner. The benchmark
configured 262,144 and measured about 202,763 tokens of runtime capacity; the
published launcher is capped at 200,000 combined tokens. The highest repeated
real workload was 200,000 input + 1,024 output under the larger benchmark cap,
while the recommended routine budget is 190K input plus 8K output. Minimum
observed free VRAM was 1.43 GiB. MTP must remain off. This is fully GPU-resident.

### Conservative BF16-KV reference

Run:

```bash
./scripts/launch/launch_sglang_bf16_kv_5090.sh
```

This removes the unscaled-FP8 warning and reports about 101K KV tokens, but its
retrieval test was still imperfect. It is supplied for quality-sensitive A/B
testing, not claimed as universally superior. The conservative script keeps
all graphs disabled; expect eager decode around 34–35 tok/s.

## What was not run

The optional `llama.cpp` reference was not run. A comparable recent GGUF was
not already present in this workspace, and downloading/validating another
20+ GiB model would have delayed the primary SGLang study. Therefore this
experiment does **not** empirically answer whether SGLang beats the user's
existing `llama.cpp` workflow. No CPU/RAM-offload experiment was performed.

## Reproducibility index

- [results.csv](results.csv): normalized request rows
- [results.json](results.json): normalized rows with extra fields
- [raw_results/](../../../raw_results/): locally generated server info, response metadata, and quality output
- [logs/](../../../logs/): locally generated startup, kernel, warning, and failure logs
- [monitoring/](../../../monitoring/): locally generated 0.5-second hardware/host traces
- `scripts/launch/launch_sglang_*_5090.sh`: publication SGLang profiles
- [run_configuration.py](../../../scripts/run_configuration.py): benchmark orchestrator
- [system-info.txt](system-info.txt): machine snapshot
- [sglang-environment.txt](sglang-environment.txt): complete environment and package freeze
- [compatibility-notes.md](../../../docs/compatibility-notes.md): source research and runtime qualifications
- [model_comparison.md](../../../models/model_comparison.md): checkpoint audit

All reported successes and failures remain in the normalized JSON/CSV. Failed
startup or harness-development rows are not silently discarded; the report
identifies configuration/runtime failures separately from measurement-harness
fixes.
