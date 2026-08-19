# Publication benchmark rerun

This is the canonical post-reboot workflow for generating a fresh, comparable
Reddit/GitHub result set. It reruns the five publication profiles without
repeating unsafe startup failures or exploratory tuning attempts.

All servers are localhost-only, concurrency one, GPU-resident, and wrapped in
the 36 GiB/40 GiB/no-swap cgroup. Run only one backend at a time.

## Fixed publication matrix

| Label | Engine and revision | Checkpoint and revision | Important settings |
|---|---|---|---|
| NInfer long | NInfer `b2b96bae4dd88f95b9ea8126d68fae3b88caa374` | NInfer NVFP4 `d6d0b3b61a38262e57217e64e7f44cf4ce98bda1` | INT8 KV, MTP3, LM-head draft, 200k, CUDA graphs |
| vLLM fast | vLLM 0.27.1 | Unsloth NVFP4 `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | FP8 KV, MTP3, 0.90 GPU utilization, automatic maximum context |
| vLLM long | vLLM 0.27.1 | Gittensor ModelOpt `ec8ad26b9e3b33c7d05c0e5743b60f37f5139005` | FP8 KV, no MTP, 0.90 GPU utilization, automatic maximum context |
| SGLang long | SGLang `374a6b24f2f2b52fc131417d8d0e4e78900f7a5d` + tracked patch | RadixArk ModelOpt `554ebba9b5f1b79dc11246341960360e6ef05ef4` | FP8 KV, no MTP, 200k, batch-1 decode graph |
| SGLang fast | Same SGLang | Same RadixArk | FP8 KV, EAGLE MTP3/4 draft tokens, 32k logical context |

Keep these revisions fixed for the publication rerun. Updating packages or
checkpoints would create a different experiment and requires a fresh audit.
Keep vLLM GPU utilization at 0.90: this leaves room for FlashInfer's lazy
394 MiB workspace and first-use kernels. At 0.93, automatic KV sizing can
consume that reserve and fail the first generation request with CUDA OOM.

## 1. Reboot and establish the session

Close other GPU-heavy applications after reboot. From the repository root:

```bash
export BENCH_SESSION="reddit_$(date -u +%Y%m%dT%H%M%SZ)"

if [[ -x .venv-tools/bin/python ]]; then
  export BENCH_PY="$PWD/.venv-tools/bin/python"
else
  export BENCH_PY="$PWD/.venv-vllm-0.27.1/bin/python"
fi

python3 scripts/preflight.py --backend all --verify-model-hash
```

The preflight requires at least 45 GiB available host RAM and 29,000 MiB free
VRAM, reports existing swap use, refuses to proceed while another inference
server is running, checks all binaries/checkpoints, and optionally verifies the
20 GiB NInfer artifact.

Capture the publication environment without overwriting the dated baseline:

```bash
python3 scripts/collect_system_info.py \
  --output "reports/${BENCH_SESSION}-system-info.txt"

.venv/bin/python scripts/collect_environment.py \
  --output "reports/${BENCH_SESSION}-sglang-environment.txt"

UV_CACHE_DIR=.uv-cache uv pip freeze \
  --python .venv-vllm-0.27.1/bin/python \
  > "reports/${BENCH_SESSION}-vllm-freeze.txt"

git -C vendor/ninfer rev-parse HEAD \
  > "reports/${BENCH_SESSION}-ninfer-commit.txt"
```

Do not clear compiled kernel caches for this decode/long-context comparison.
Cold installation/JIT time is a separate measurement. Let the idle GPU return
to a similar temperature and power state before each backend.

## 2. NInfer long-context profile

Start in terminal A:

```bash
./scripts/launch/launch_ninfer_5090.sh
```

Run in terminal B after the server reports ready:

```bash
"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --label "${BENCH_SESSION}_ninfer" \
  --prompt-sizes 1024,32768,100000,190000 \
  --output-tokens 768 --repetitions 3

"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --label "${BENCH_SESSION}_ninfer_exact190k" \
  --prompt-sizes 198247 --output-tokens 768 --repetitions 3

"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --label "${BENCH_SESSION}_ninfer_prefix" \
  --prefix-cache-size 190000 --prompt-sizes '' --repetitions 1

"$BENCH_PY" scripts/check_openai_quality.py \
  --label "${BENCH_SESSION}_ninfer_quality" --needle-context 190000
```

The 198,247-token generated input becomes approximately 190,003 actual chat
prompt tokens with this template. The result table always reports actual
server counts. Stop terminal A with Ctrl-C, then rerun preflight before the
next backend.

## 3. vLLM fast MTP profile

Start in terminal A:

```bash
QWEN_VLLM_GPU_MEMORY_UTILIZATION=0.90 \
./scripts/launch/launch_vllm_5090.sh
```

Run in terminal B:

```bash
"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --base-url http://127.0.0.1:31000 \
  --model qwen38-vllm-unsloth \
  --model-path "$PWD/hf-home/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108" \
  --label "${BENCH_SESSION}_vllm_unsloth_mtp3" \
  --prompt-sizes 1024,32768,100000,115000 \
  --output-tokens 768 --repetitions 3

"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --base-url http://127.0.0.1:31000 \
  --model qwen38-vllm-unsloth \
  --model-path "$PWD/hf-home/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108" \
  --label "${BENCH_SESSION}_vllm_unsloth_prefix" \
  --prefix-cache-size 100000 --prompt-sizes '' --repetitions 1

"$BENCH_PY" scripts/check_openai_quality.py \
  --base-url http://127.0.0.1:31000 \
  --model qwen38-vllm-unsloth \
  --model-path "$PWD/hf-home/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108" \
  --label "${BENCH_SESSION}_vllm_unsloth_quality" \
  --needle-context 100000
```

Stop terminal A before continuing.

## 4. vLLM long-context profile

Start in terminal A:

```bash
QWEN_VLLM_MODEL_PATH="$PWD/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/ec8ad26b9e3b33c7d05c0e5743b60f37f5139005" \
QWEN_VLLM_SERVED_MODEL_NAME=qwen38-vllm-gittensor \
QWEN_VLLM_MTP_TOKENS=0 \
QWEN_VLLM_GPU_MEMORY_UTILIZATION=0.90 \
./scripts/launch/launch_vllm_5090.sh
```

Run in terminal B:

```bash
"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --base-url http://127.0.0.1:31000 \
  --model qwen38-vllm-gittensor \
  --model-path "$PWD/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/ec8ad26b9e3b33c7d05c0e5743b60f37f5139005" \
  --label "${BENCH_SESSION}_vllm_gittensor_no_mtp" \
  --prompt-sizes 1024,32768,100000,190000 \
  --output-tokens 768 --repetitions 3

"$BENCH_PY" scripts/check_openai_quality.py \
  --base-url http://127.0.0.1:31000 \
  --model qwen38-vllm-gittensor \
  --model-path "$PWD/hf-home/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/ec8ad26b9e3b33c7d05c0e5743b60f37f5139005" \
  --label "${BENCH_SESSION}_vllm_gittensor_quality" \
  --needle-context 190000
```

Stop terminal A before continuing.

## 5. SGLang long-context profile

Start in terminal A:

```bash
./scripts/launch/launch_sglang_long_5090.sh
```

Run in terminal B. The `/model_info` readiness override is important because
this dated SGLang branch's `/health` route performs generation:

```bash
"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --base-url http://127.0.0.1:30000 \
  --model qwen38-radixark --readiness-path /model_info \
  --label "${BENCH_SESSION}_sglang_radixark_no_mtp" \
  --prompt-sizes 1024,32768,100000,190000,198247 \
  --output-tokens 768 --repetitions 3

"$BENCH_PY" scripts/check_openai_quality.py \
  --base-url http://127.0.0.1:30000 \
  --model qwen38-radixark \
  --model-path "$PWD/hf-home/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/554ebba9b5f1b79dc11246341960360e6ef05ef4" \
  --label "${BENCH_SESSION}_sglang_radixark_quality" \
  --needle-context 190000
```

Stop terminal A before continuing.

## 6. SGLang short-context MTP profile

Start in terminal A:

```bash
./scripts/launch/launch_sglang_fast_5090.sh
```

Run in terminal B:

```bash
"$BENCH_PY" scripts/benchmark_openai_backend.py \
  --base-url http://127.0.0.1:30000 \
  --model qwen38-radixark-mtp --readiness-path /model_info \
  --label "${BENCH_SESSION}_sglang_radixark_mtp" \
  --prompt-sizes 1024,16384 \
  --output-tokens 768 --repetitions 3
```

Stop the server and run the final preflight. If swap grew materially or any
run recorded an error, investigate before publishing.

## 7. Generate publication tables

```bash
python3 scripts/summarize_results.py --prefix "$BENCH_SESSION"
```

This writes `reports/<session>-results.csv` and
`reports/<session>-results.md`, including medians, ranges, TTFT, VRAM, host
headroom, quality pass counts, failures, and raw source filenames. Failure
tracebacks stay in ignored raw results; the report includes only a redacted
error type and message. Review model outputs manually before treating the smoke
checks as quality evidence.

Commit the compact report, CSV, environment captures, configuration scripts,
and updated canonical `REPORT.md`. Raw outputs and monitoring remain ignored;
archive them privately by default. If readers need complete traces, inspect and
redact local paths, device UUIDs, hostnames, usernames, and process details
before attaching them publicly.
