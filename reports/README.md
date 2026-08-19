# Curated benchmark evidence

These dated reports preserve the detailed measurements and compatibility
findings behind the canonical recommendation. They describe one machine and
benchmark session, but they are publication artifacts rather than personal
files. Hostnames, usernames, local absolute paths, device UUIDs, process lists,
and secrets are omitted.

- [2026-08-17 SGLang benchmark](archive/2026-08-17-sglang/report.md): checkpoint,
  capacity, quality, and host-OOM baseline.
- [2026-08-18 backend follow-up](archive/2026-08-18-backend-follow-up/report.md): detailed
  vLLM and NInfer installation and measurements.
- [2026-08-17 system snapshot](archive/2026-08-17-sglang/system-info.txt): public-safe host
  and accelerator details for the SGLang baseline.
- [2026-08-17 SGLang environment](archive/2026-08-17-sglang/sglang-environment.txt): exact
  runtime versions and package freeze for that baseline.

[../REPORT.md](../REPORT.md) is the canonical current recommendation.
[../docs/RERUN.md](../docs/RERUN.md) generates each new Markdown/CSV report,
manifest, and environment capture under `runs/<session>/` without overwriting
the dated reports. `LATEST` names the most recently summarized session.

Review and redact every new report before committing it. Raw server logs and
uncurated captures belong in the ignored `logs/`, `monitoring/`, and
`raw_results/` directories.
