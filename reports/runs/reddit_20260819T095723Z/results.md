# Benchmark results: reddit_20260819T095723Z

Status: **complete_with_quality_failures**.

All throughput values are client-observed medians. Prompt counts are the actual
server-reported counts, not requested target sizes.

| Configuration | Model | Actual prompt | n | Output | TTFT (s) | Decode tok/s | Range | Peak VRAM GiB | Min host available GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reddit_20260819T095723Z_ninfer | qwen38-ninfer-nvfp4 | 997 | 3 | 768 | 0.128 | 197.44 | 188.97-201.33 | 29.2 | 55.27 |
| reddit_20260819T095723Z_ninfer | qwen38-ninfer-nvfp4 | 31419 | 3 | 768 | 4.518 | 182.14 | 178.25-189.28 | 29.2 | 55.0 |
| reddit_20260819T095723Z_ninfer | qwen38-ninfer-nvfp4 | 95849 | 3 | 768 | 21.986 | 173.61 | 158.34-174.28 | 28.92 | 54.9 |
| reddit_20260819T095723Z_ninfer | qwen38-ninfer-nvfp4 | 182099 | 3 | 768 | 62.672 | 150.46 | 135.51-151.64 | 28.92 | 54.89 |
| reddit_20260819T095723Z_ninfer_exact190k | qwen38-ninfer-nvfp4 | 190003 | 3 | 768 | 67.697 | 155.95 | 152.64-156.7 | 28.91 | 54.89 |
| reddit_20260819T095723Z_sglang_radixark_mtp | qwen38-radixark-mtp | 1037 | 3 | 768 | 0.109 | 108.84 | 104.37-112.48 | 25.69 | 51.06 |
| reddit_20260819T095723Z_sglang_radixark_mtp | qwen38-radixark-mtp | 15757 | 3 | 768 | 1.701 | 105.02 | 104.93-106.46 | 25.96 | 51.08 |
| reddit_20260819T095723Z_sglang_radixark_no_mtp | qwen38-radixark | 1037 | 3 | 768 | 0.097 | 59.3 | 59.28-59.33 | 30.18 | 51.26 |
| reddit_20260819T095723Z_sglang_radixark_no_mtp | qwen38-radixark | 31459 | 3 | 768 | 3.79 | 56.76 | 56.27-56.84 | 30.45 | 51.33 |
| reddit_20260819T095723Z_sglang_radixark_no_mtp | qwen38-radixark | 95890 | 3 | 768 | 20.503 | 52.51 | 52.51-52.52 | 30.45 | 51.34 |
| reddit_20260819T095723Z_sglang_radixark_no_mtp | qwen38-radixark | 182140 | 3 | 768 | 64.506 | 47.35 | 45.96-47.72 | 30.5 | 51.24 |
| reddit_20260819T095723Z_sglang_radixark_no_mtp | qwen38-radixark | 190043 | 3 | 768 | 67.284 | 48.42 | 47.28-49.18 | 30.51 | 51.19 |
| reddit_20260819T095723Z_vllm_gittensor_no_mtp | qwen38-vllm-gittensor | 1038 | 3 | 768 | 0.085 | 52.24 | 51.01-52.36 | 30.66 | 52.27 |
| reddit_20260819T095723Z_vllm_gittensor_no_mtp | qwen38-vllm-gittensor | 31459 | 3 | 768 | 3.75 | 49.77 | 49.56-49.85 | 30.77 | 52.36 |
| reddit_20260819T095723Z_vllm_gittensor_no_mtp | qwen38-vllm-gittensor | 95890 | 3 | 768 | 20.96 | 46.57 | 46.57-46.59 | 30.81 | 52.33 |
| reddit_20260819T095723Z_vllm_gittensor_no_mtp | qwen38-vllm-gittensor | 182140 | 3 | 768 | 64.33 | 43.19 | 42.51-43.42 | 30.79 | 52.17 |
| reddit_20260819T095723Z_vllm_unsloth_mtp3 | qwen38-vllm-unsloth | 1038 | 3 | 768 | 0.147 | 98.98 | 97.98-108.18 | 30.84 | 52.51 |
| reddit_20260819T095723Z_vllm_unsloth_mtp3 | qwen38-vllm-unsloth | 31459 | 3 | 768 | 4.732 | 95.86 | 95.2-97.9 | 31.13 | 52.54 |
| reddit_20260819T095723Z_vllm_unsloth_mtp3 | qwen38-vllm-unsloth | 95890 | 3 | 768 | 23.523 | 98.13 | 93.7-103.76 | 31.13 | 52.53 |
| reddit_20260819T095723Z_vllm_unsloth_mtp3 | qwen38-vllm-unsloth | 110265 | 3 | 768 | 29.464 | 89.76 | 85.51-94.34 | 31.13 | 52.53 |

## Quality smoke tests

- reddit_20260819T095723Z_ninfer_quality: 7/7 passed
- reddit_20260819T095723Z_sglang_radixark_quality: 6/7 passed
- reddit_20260819T095723Z_vllm_gittensor_quality: 6/7 passed
- reddit_20260819T095723Z_vllm_unsloth_quality: 5/7 passed

## Prefix reuse

| Configuration | Shared target tokens | Prime TTFT (s) | Cached TTFT (s) | Cached decode tok/s |
|---|---:|---:|---:|---:|
| reddit_20260819T095723Z_ninfer_prefix | 190000 | 62.888 | 0.420 | 120.57 |
| reddit_20260819T095723Z_vllm_unsloth_prefix | 100000 | 23.555 | 1.357 | 83.12 |

## Raw sources

- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_ninfer_20260819T115834_a40f3ede.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_ninfer_exact190k_20260819T120416_5270cc0c.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_ninfer_prefix_20260819T120808_7dc2c4fb.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_ninfer_quality_20260819T120927_3522f888.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_sglang_radixark_mtp_20260819T125713_45a78cde.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_sglang_radixark_no_mtp_20260819T124036_9e3dc740.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_sglang_radixark_quality_20260819T125226_4096a456.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_vllm_gittensor_no_mtp_20260819T122411_e7a02f44.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_vllm_gittensor_quality_20260819T123216_78f28a7a.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_vllm_unsloth_mtp3_20260819T121419_26d7a0b9.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_vllm_unsloth_prefix_20260819T121915_04b0b694.json`
- `raw_results/reddit_20260819T095723Z/reddit_20260819T095723Z_vllm_unsloth_quality_20260819T121959_a49ff92f.json`
