# Benchmark results: candidate_20260820_postreboot

Status: **complete_with_quality_failures**.

All throughput values are client-observed medians. Prompt counts are the actual
server-reported counts, not requested target sizes.

| Configuration | Model | Actual prompt | n | Output | TTFT (s) | Decode tok/s | Range | Peak VRAM GiB | Min host available GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate_20260820_postreboot_sglang_gittensor_dspark | qwen38-sglang-gittensor-dspark | 1037 | 3 | 768 | 0.124 | 119.87 | 113.71-131.71 | 28.87 | 52.12 |
| candidate_20260820_postreboot_sglang_gittensor_dspark | qwen38-sglang-gittensor-dspark | 31459 | 3 | 768 | 3.709 | 111.41 | 102.27-129.91 | 28.87 | 52.17 |
| candidate_20260820_postreboot_sglang_gittensor_dspark | qwen38-sglang-gittensor-dspark | 95889 | 3 | 768 | 21.971 | 100.14 | 92.92-101.97 | 28.78 | 52.17 |
| candidate_20260820_postreboot_sglang_gittensor_lmhead4 | qwen38-sglang-gittensor-lmhead4 | 1037 | 3 | 768 | 0.397 | 60.72 | 60.2-61.2 | 29.96 | 52.9 |
| candidate_20260820_postreboot_sglang_gittensor_lmhead4 | qwen38-sglang-gittensor-lmhead4 | 31459 | 3 | 768 | 3.717 | 58.53 | 58.46-58.54 | 29.85 | 53.07 |
| candidate_20260820_postreboot_sglang_gittensor_lmhead4 | qwen38-sglang-gittensor-lmhead4 | 95889 | 3 | 768 | 22.79 | 52.52 | 51.89-52.79 | 30.23 | 52.13 |
| candidate_20260820_postreboot_sglang_gittensor_lmhead4 | qwen38-sglang-gittensor-lmhead4 | 182139 | 3 | 768 | 72.773 | 47.23 | 46.31-49.92 | 30.02 | 52.06 |

## Quality smoke tests

- candidate_20260820_postreboot_sglang_gittensor_dspark_quality_128diag: 7/7 passed
- candidate_20260820_postreboot_sglang_gittensor_dspark_quality: 6/7 passed
- candidate_20260820_postreboot_sglang_gittensor_lmhead4_quality_128diag: 7/7 passed
- candidate_20260820_postreboot_sglang_gittensor_lmhead4_quality: 5/7 passed

## Raw sources

- `raw_results/candidate_20260820_postreboot_sglang_gittensor_dspark_20260820T131402_e89be5f7/candidate_20260820_postreboot_sglang_gittensor_dspark_20260820T131402_e89be5f7.json`
- `raw_results/candidate_20260820_postreboot_sglang_gittensor_dspark_quality_128diag_20260820T131847_169d3151/candidate_20260820_postreboot_sglang_gittensor_dspark_quality_128diag_20260820T131847_169d3151.json`
- `raw_results/candidate_20260820_postreboot_sglang_gittensor_dspark_quality_20260820T131653_b019d32a/candidate_20260820_postreboot_sglang_gittensor_dspark_quality_20260820T131653_b019d32a.json`
- `raw_results/candidate_20260820_postreboot_sglang_gittensor_lmhead4_20260820T130052_854364c5/candidate_20260820_postreboot_sglang_gittensor_lmhead4_20260820T130052_854364c5.json`
- `raw_results/candidate_20260820_postreboot_sglang_gittensor_lmhead4_quality_128diag_20260820T132104_26bf8482/candidate_20260820_postreboot_sglang_gittensor_lmhead4_quality_128diag_20260820T132104_26bf8482.json`
- `raw_results/candidate_20260820_postreboot_sglang_gittensor_lmhead4_quality_20260820T130901_4e185631/candidate_20260820_postreboot_sglang_gittensor_lmhead4_quality_20260820T130901_4e185631.json`
