# Prompt methodology

Two harnesses produced the published measurements. Their prompt paths differ,
so results should be interpreted using the method associated with their date.

## 2026-08-17 SGLang baseline

`scripts/run_configuration.py` sends requests to SGLang's `/generate` route.
A neutral paragraph is tokenized with the checkpoint's own tokenizer, repeated,
and sliced so the server receives exactly the target count as `input_ids`. Each
measured request begins with a nonce and follows an explicit `/flush_cache`,
preventing prefix-cache reuse. Performance requests set `ignore_eos=true`;
quality checks allow normal end-of-sequence behavior.

## Backend follow-up and publication reruns

`scripts/benchmark_openai_backend.py` uses the OpenAI-compatible
`/v1/chat/completions` route by default. It deterministically constructs a
target number of tokenizer IDs, decodes them to text, and lets the server apply
its chat template. A nonce at the beginning prevents meaningful prefix reuse.
Because decoding and chat templating can change the count, the server-reported
prompt-token count is authoritative. For example, the requested 198,247-token
input becomes approximately 190,003 actual prompt tokens with the tested
template.

Chat-completion mode does not force `ignore_eos`. Every published speed sample
still reached its requested 768 output tokens; future reports must identify any
early stop rather than treating it as an equivalent speed sample. The harness's
completion-mode path supports exact token IDs and `ignore_eos` for servers that
expose a compatible completion endpoint.

## Metrics and quality checks

Client TTFT is measured from beginning the HTTP POST until the first streamed
event containing a completion token. It therefore includes HTTP, scheduling,
and prefill. Prefill throughput is prompt tokens divided by this TTFT and is
reported with that qualification. Decode throughput is `(completion_tokens -
1) / (last_event_time - first_token_time)`. Backend metrics are recorded
alongside client values where exposed.

The small sanity set covers logic, Python coding, exact-format instruction
following, and factual explanation. Long-context tests insert the exact fact
`The cobalt falcon access code is 739184.` at 10%, 50%, and 90% of a synthetic
archive and ask for that code. Automatic checks are intentionally elementary;
raw output is preserved for manual inspection.

Every primary performance measurement has one unreported warm-up followed by
at least three measured repetitions at concurrency 1. Identical parameter
settings are used for paired comparisons.
