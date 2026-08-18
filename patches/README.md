# SGLang patch

`sglang-pr-34904-lm-head.patch` is a narrow backport derived from the SGLang
project and is distributed under its Apache-2.0 license. The repository-wide
MIT license does not replace the upstream license for this patch.

The pinned SGLang checkout needs this dispatch fix for compressed-tensors
checkpoints with a quantized `ParallelLMHead`, including the tested Unsloth
checkpoint. At publication on 2026-08-18, the upstream bug and pull request
were still open, and SGLang v0.5.17 and `main` did not contain an equivalent
fix. Keep the patch with the pinned benchmark configuration; evaluate a newer
SGLang revision as a separate configuration before removing it.

Upstream source and license: <https://github.com/sgl-project/sglang>

- Bug: <https://github.com/sgl-project/sglang/issues/34895>
- Proposed upstream fix: <https://github.com/sgl-project/sglang/pull/34904>
