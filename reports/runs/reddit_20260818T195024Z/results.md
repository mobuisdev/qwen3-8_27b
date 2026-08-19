# Benchmark results: reddit_20260818T195024Z

Status: **failed**.

All throughput values are client-observed medians. Prompt counts are the actual
server-reported counts, not requested target sizes.

| Configuration | Model | Actual prompt | n | Output | TTFT (s) | Decode tok/s | Range | Peak VRAM GiB | Min host available GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reddit_20260818T195024Z_ninfer | qwen38-ninfer-nvfp4 | 997 | 3 | 768 | 0.129 | 200.39 | 199.47-203.01 | 29.14 | 55.77 |
| reddit_20260818T195024Z_ninfer | qwen38-ninfer-nvfp4 | 31419 | 3 | 768 | 4.528 | 179.7 | 162.7-189.04 | 29.14 | 55.77 |
| reddit_20260818T195024Z_ninfer | qwen38-ninfer-nvfp4 | 95849 | 3 | 768 | 22.042 | 169.0 | 167.44-173.27 | 29.14 | 55.41 |
| reddit_20260818T195024Z_ninfer | qwen38-ninfer-nvfp4 | 182099 | 3 | 768 | 62.84 | 152.75 | 141.22-158.82 | 28.81 | 55.41 |
| reddit_20260818T195024Z_ninfer_exact190k | qwen38-ninfer-nvfp4 | 190003 | 3 | 768 | 67.76 | 150.73 | 143.36-155.39 | 28.72 | 55.34 |
| reddit_20260818T195024Z_sglang_radixark_mtp | qwen38-radixark-mtp | 1037 | 3 | 768 | 0.098 | 121.5 | 115.41-128.48 | 25.7 | 52.22 |
| reddit_20260818T195024Z_sglang_radixark_mtp | qwen38-radixark-mtp | 15757 | 3 | 768 | 1.548 | 112.69 | 111.27-114.98 | 25.97 | 52.27 |
| reddit_20260818T195024Z_sglang_radixark_no_mtp | qwen38-radixark | 1038 | 3 | 768 | 0.147 | 63.81 | 63.8-63.86 | 30.05 | 52.36 |
| reddit_20260818T195024Z_sglang_radixark_no_mtp | qwen38-radixark | 31459 | 3 | 768 | 3.533 | 61.11 | 61.1-61.11 | 30.16 | 52.41 |
| reddit_20260818T195024Z_sglang_radixark_no_mtp | qwen38-radixark | 95890 | 3 | 768 | 18.964 | 56.54 | 56.51-56.54 | 30.17 | 52.44 |
| reddit_20260818T195024Z_sglang_radixark_no_mtp | qwen38-radixark | 182140 | 3 | 768 | 57.672 | 51.4 | 51.4-51.41 | 30.17 | 52.43 |
| reddit_20260818T195024Z_sglang_radixark_no_mtp | qwen38-radixark | 190043 | 3 | 768 | 62.21 | 50.96 | 50.96-50.96 | 30.17 | 52.41 |
| reddit_20260818T195024Z_vllm_gittensor_no_mtp | qwen38-vllm-gittensor | 1038 | 3 | 768 | 0.081 | 55.49 | 55.48-55.5 | 28.7 | 53.03 |
| reddit_20260818T195024Z_vllm_gittensor_no_mtp | qwen38-vllm-gittensor | 31459 | 3 | 768 | 3.45 | 53.49 | 53.49-53.5 | 28.88 | 53.02 |
| reddit_20260818T195024Z_vllm_gittensor_no_mtp | qwen38-vllm-gittensor | 95890 | 3 | 768 | 19.35 | 50.04 | 50.02-50.04 | 28.88 | 52.97 |
| reddit_20260818T195024Z_vllm_gittensor_no_mtp | qwen38-vllm-gittensor | 182140 | 3 | 768 | 59.319 | 46.01 | 46.0-46.06 | 28.88 | 52.8 |
| reddit_20260818T195024Z_vllm_unsloth_mtp3 | qwen38-vllm-unsloth | 1037 | 3 | 768 | 0.147 | 100.41 | 98.8-100.97 | 29.51 | 52.76 |
| reddit_20260818T195024Z_vllm_unsloth_mtp3 | qwen38-vllm-unsloth | 31459 | 3 | 768 | 4.705 | 93.19 | 91.13-97.59 | 29.7 | 52.82 |
| reddit_20260818T195024Z_vllm_unsloth_mtp3 | qwen38-vllm-unsloth | 95890 | 3 | 768 | 23.452 | 91.31 | 86.17-94.25 | 29.7 | 52.77 |
| reddit_20260818T195024Z_vllm_unsloth_mtp3 | qwen38-vllm-unsloth | 110265 | 3 | 768 | 29.354 | 89.48 | 85.04-93.88 | 29.75 | 52.79 |

## Quality smoke tests

- reddit_20260818T195024Z_ninfer_quality: 7/7 passed
- reddit_20260818T195024Z_sglang_radixark_quality: 5/7 passed
- reddit_20260818T195024Z_vllm_gittensor_quality: 5/7 passed
- reddit_20260818T195024Z_vllm_unsloth_quality: 5/7 passed

## Prefix reuse

| Configuration | Shared target tokens | Prime TTFT (s) | Cached TTFT (s) | Cached decode tok/s |
|---|---:|---:|---:|---:|
| reddit_20260818T195024Z_ninfer_prefix | 190000 | 62.841 | 0.409 | 119.55 |
| reddit_20260818T195024Z_vllm_unsloth_prefix | 100000 | 23.443 | 1.348 | 107.69 |

## Recorded failures

- reddit_20260818T195024Z_vllm_unsloth_mtp3_20260818T221918_706f4baf.json: IncompleteMeasurement: decode throughput unavailable for 1 request(s)

## Raw sources

- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_ninfer_20260818T215207_9221d68e.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_ninfer_exact190k_20260818T215751_7850fa45.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_ninfer_prefix_20260818T220144_2b97d0c1.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_ninfer_quality_20260818T220303_6bd16581.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_sglang_radixark_mtp_20260818T230945_2384eeb8.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_sglang_radixark_no_mtp_20260818T225319_9c28519b.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_sglang_radixark_quality_20260818T230437_21a8543f.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_vllm_gittensor_no_mtp_20260818T223217_1d26df94.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_vllm_gittensor_quality_20260818T223948_67586967.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_vllm_unsloth_mtp3_20260818T221918_706f4baf.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_vllm_unsloth_mtp3_20260818T222240_1f6bdfba.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_vllm_unsloth_prefix_20260818T222732_18e463b2.json`
- `raw_results/reddit_20260818T195024Z/reddit_20260818T195024Z_vllm_unsloth_quality_20260818T222820_59f3f163.json`
