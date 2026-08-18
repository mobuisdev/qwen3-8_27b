# Scripts

Run commands from the repository root. Pinned backend provisioning lives in
[`setup/`](setup/), while RAM-guarded measured profiles live in
[`launch/`](launch/). The Python files in this directory are backend-neutral
benchmark clients and supporting utilities.

The current backend-neutral clients are:

- `benchmark_openai_backend.py`: streaming speed, TTFT, prefix reuse, GPU, and
  host-memory measurements;
- `check_openai_quality.py`: logic, coding, exact-JSON, tool-call, and
  long-context needle smoke tests.
- `preflight.py`: refuses a run with insufficient host headroom, missing
  artifacts, or another inference server.
- `summarize_results.py`: builds one publication Markdown/CSV table from a
  shared run-ID prefix.

After launching the recommended NInfer server, a small repeatable check is:

```bash
.venv-tools/bin/python scripts/benchmark_openai_backend.py \
  --prompt-sizes 1024,32768 --output-tokens 768 --repetitions 3

.venv-tools/bin/python scripts/check_openai_quality.py \
  --needle-context 100000
```

Both clients default to the NInfer endpoint, model ID, and the tracked
`models/qwen_base` tokenizer metadata. Pass `--help` for backend overrides.
Generated output goes to the ignored `raw_results/` and `monitoring/`
directories.

Supporting scripts capture the environment, audit/download pinned metadata,
and reproduce the detailed SGLang measurements.

Maintenance scripts make tracked writes explicit: use
`fetch_model_metadata.py --refresh` to replace pinned repository metadata and
`audit_models.py --write` to regenerate the tracked audit. Their `--help`
paths are read-only. The dated SGLang harness `run_configuration.py` stores
normalized rows in its ignored raw result by default; pass
`--append-canonical-results` only when intentionally extending the tracked
SGLang CSV/JSON datasets.

## Benchmark another machine or server

The benchmark client works with any OpenAI-compatible chat/completions server;
the RTX 5090 launchers are not required. From the repository root, create
the small client environment:

```bash
uv venv .venv-tools
UV_CACHE_DIR=.uv-cache uv pip install \
  --python .venv-tools/bin/python -r requirements-benchmark.txt
```

Then point it at your server and local tokenizer metadata:

```bash
.venv-tools/bin/python scripts/benchmark_openai_backend.py \
  --base-url http://127.0.0.1:8000 \
  --model YOUR_SERVED_MODEL_NAME \
  --model-path /path/to/tokenizer \
  --gpu-index 0 \
  --prompt-sizes 1024,32768 \
  --output-tokens 768 --repetitions 3
```

Use only prompt sizes supported by that server. The supplied launch/setup
profiles are intentionally specific to one Linux RTX 5090 workstation; on
other GPUs or operating systems, run your own backend and reuse these clients.
Pass `--no-monitor` for a remote or non-NVIDIA server; the result will omit
local GPU and host-memory telemetry.
