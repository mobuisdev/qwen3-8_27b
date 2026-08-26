# 2026-08-20 upstream candidate matrix

This workflow evaluates upstream changes without altering the fixed publication
matrix in [RERUN.md](RERUN.md). Candidate results are not comparable claims
until they pass the same quality checks and use the same actual prompt sizes.

All launchers remain localhost-only, concurrency one, GPU-resident, and inside
the 36 GiB/40 GiB/no-swap host-memory boundary.

## Immutable candidates

| Profile | Runtime | Target | Draft / important setting |
|---|---|---|---|
| vLLM target isolation | vLLM 0.27.1 | Gittensor LMHead4 `0cc27958cefbbe231782ec8511de8c4eb5233348` | No speculation, 200k, FP8 KV |
| vLLM source candidate | vLLM `1eab6fef01b78ec4eab6b7156bbf5f120e48d381` | Same | No speculation, 200k, FP8 KV |
| SGLang target alone | SGLang Qwen3.8 merge `8a1e6e4e461044246739b5a1ad579c8acc556a2d` | Same | No speculation, 200k, FP8 KV |
| SGLang DSpark | Same | Same | DSpark NVFP4 `eba1ac5a66c74902eaa95a4000a7c5eda96d8e95`, 122,880 context |
| NInfer groupwise | Publication NInfer `b2b96bae4dd88f95b9ea8126d68fae3b88caa374` | Groupwise-int `18dfc887423fa5aabf3cb56fac41490e462b3fab` | INT8 KV, MTP3, 200k |

The Gittensor target changed its `lm_head` from BF16 to NVFP4 after the
publication pin. Its publisher reports a smaller checkpoint and faster decode.
The matching 1.40 GB DSpark drafter is a separate short/medium-context option;
the publisher reports that speculation limits the single-5090 context to about
116k even though the target alone can serve the native window.

The vLLM source commit is a dated experiment, not a stable upgrade. It includes
post-0.27.1 Qwen GDN/MTP work and is deliberately isolated from
`.venv-vllm-0.27.1`.

## Measured post-reboot result

The controlled 2026-08-20 run used three repetitions, 768 generated tokens,
zero initial swap use, and actual server-reported prompt counts. The complete
evidence is in
[`../reports/runs/candidate_20260820_postreboot/results.md`](../reports/runs/candidate_20260820_postreboot/results.md).

| Actual prompt | Target-only decode | DSpark decode | DSpark gain | Target TTFT | DSpark TTFT |
|---:|---:|---:|---:|---:|---:|
| 1,037 | 60.72 tok/s | 119.87 tok/s | +97.4% | 0.397 s | 0.124 s |
| 31,459 | 58.53 tok/s | 111.41 tok/s | +90.4% | 3.717 s | 3.709 s |
| 95,889 | 52.52 tok/s | 100.14 tok/s | +90.7% | 22.790 s | 21.971 s |
| 182,139 | 47.23 tok/s | Does not fit | n/a | 72.773 s | n/a |

Both profiles passed logic, coding, exact JSON, tool calling, and all three
needle positions when the needle generation allowance was 128 tokens. With the
historical 64-token allowance, target-only passed 5/7 checks and DSpark passed
6/7: the failed responses found or searched for the correct sentence but
exhausted the allowance in reasoning before emitting the digits. Both the
strict result and the 128-token diagnostic remain in the candidate report.

DSpark is therefore a meaningful SGLang short/medium-context improvement, but
it does not replace NInfer: it remains about 39% slower at comparable short and
96k prompts, and it cannot serve the 190k workload. The production
recommendation remains NInfer NVFP4 with MTP3.

## Provisioning

Download the Gittensor target and drafter without creating a new engine:

```bash
./scripts/setup/setup_candidate_models_5090.sh
```

Build the isolated SGLang candidate and download those models:

```bash
./scripts/setup/setup_sglang_candidate_5090.sh
```

Optionally build the dated vLLM source candidate:

```bash
./scripts/setup/setup_vllm_candidate_5090.sh
```

Optionally add the 16.96 GiB NInfer groupwise artifact:

```bash
./scripts/setup/setup_candidate_models_5090.sh --with-ninfer-groupwise
```

The large downloads are content-addressed in `hf-home/`; reruns reuse completed
blobs. The NInfer groupwise file is stored separately from the recommended
NVFP4 artifact and is verified against SHA-256
`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`.

## Establish a candidate session

After a reboot, close other GPU-heavy applications and run:

```bash
export BENCH_SESSION="candidate_20260820_$(date -u +%Y%m%dT%H%M%SZ)"
export BENCH_PY="$PWD/.venv-tools/bin/python"
python3 scripts/preflight.py --backend vllm --candidate
```

Run only one server at a time. Re-run the matching candidate preflight between
servers. Do not append these rows to the archived SGLang baseline.

## 1. Isolate the updated target on stable vLLM

Start the server:

```bash
./scripts/launch/launch_vllm_gittensor_candidate_5090.sh
```

Then run:

```bash
"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --base-url http://127.0.0.1:31000 \
  --model qwen38-vllm-gittensor-lmhead4 \
  --model-path "$PWD/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348" \
  --label "${BENCH_SESSION}_vllm_stable_gittensor_lmhead4" \
  --prompt-sizes 1024,32768,100000,190000 \
  --output-tokens 768 --repetitions 3

"$BENCH_PY" scripts/check_openai_quality.py \
  --base-url http://127.0.0.1:31000 \
  --model qwen38-vllm-gittensor-lmhead4 \
  --model-path "$PWD/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348" \
  --label "${BENCH_SESSION}_vllm_stable_gittensor_lmhead4_quality" \
  --needle-context 190000
```

This is the clean checkpoint comparison against the publication vLLM/Gittensor
row. Stop the server before continuing.

## 2. Test the dated vLLM source candidate

Use the same launcher with the isolated environment:

```bash
QWEN_VLLM_VENV="$PWD/.venv-vllm-qwen38-candidate" \
./scripts/launch/launch_vllm_gittensor_candidate_5090.sh
```

Repeat the preceding speed and quality commands with labels beginning
`${BENCH_SESSION}_vllm_source_gittensor_lmhead4`. This separates the engine
delta from the checkpoint delta.

## 3. Test SGLang target-only long context

Start:

```bash
python3 scripts/preflight.py --backend sglang --candidate
./scripts/launch/launch_sglang_gittensor_candidate_long_5090.sh
```

Run:

```bash
"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --base-url http://127.0.0.1:30000 \
  --model qwen38-sglang-gittensor-lmhead4 --readiness-path /model_info \
  --model-path "$PWD/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348" \
  --label "${BENCH_SESSION}_sglang_gittensor_lmhead4" \
  --prompt-sizes 1024,32768,100000,190000 \
  --output-tokens 768 --repetitions 3

"$BENCH_PY" scripts/check_openai_quality.py \
  --base-url http://127.0.0.1:30000 \
  --model qwen38-sglang-gittensor-lmhead4 \
  --model-path "$PWD/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348" \
  --label "${BENCH_SESSION}_sglang_gittensor_lmhead4_quality" \
  --needle-context 190000
```

Stop the server before the speculative profile.

## 4. Test SGLang with the matching DSpark drafter

Start:

```bash
./scripts/launch/launch_sglang_gittensor_dspark_candidate_5090.sh
```

Run only prompt sizes below the speculative profile's measured capacity:

```bash
"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --base-url http://127.0.0.1:30000 \
  --model qwen38-sglang-gittensor-dspark --readiness-path /model_info \
  --model-path "$PWD/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348" \
  --label "${BENCH_SESSION}_sglang_gittensor_dspark" \
  --prompt-sizes 1024,32768,100000 \
  --output-tokens 768 --repetitions 3

"$BENCH_PY" scripts/check_openai_quality.py \
  --base-url http://127.0.0.1:30000 \
  --model qwen38-sglang-gittensor-dspark \
  --model-path "$PWD/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348" \
  --label "${BENCH_SESSION}_sglang_gittensor_dspark_quality" \
  --needle-context 100000
```

## 5. Optionally test NInfer groupwise-int

Start:

```bash
python3 scripts/preflight.py --backend ninfer --candidate --verify-model-hash
./scripts/launch/launch_ninfer_groupwise_candidate_5090.sh
```

Use the standard NInfer client endpoint with the candidate model ID:

```bash
"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --model qwen38-ninfer-groupwise \
  --label "${BENCH_SESSION}_ninfer_groupwise" \
  --prompt-sizes 1024,32768,100000,190000 \
  --output-tokens 768 --repetitions 3

"$BENCH_PY" scripts/check_openai_quality.py \
  --model qwen38-ninfer-groupwise \
  --label "${BENCH_SESSION}_ninfer_groupwise_quality" \
  --needle-context 190000
```

## Promotion criteria

Generate a separate report with
`python3 scripts/summarize_results.py --prefix "$BENCH_SESSION"`. Promote a
candidate into the main recommendation only after:

- all seven local quality checks pass or every difference is reviewed;
- actual prompt counts match the publication rows;
- 1k, 100k, and 190k throughput and TTFT are compared where the profile fits;
- prefix reuse is verified independently;
- startup and first-request memory remain inside the existing safety boundary.

DSpark cannot replace the 190k recommendation if it still cannot serve that
prompt. A short-context win should be documented as a separate profile rather
than presented as a universal replacement.
