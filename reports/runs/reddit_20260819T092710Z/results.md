# Benchmark results: reddit_20260819T092710Z

Status: **aborted**.

All throughput values are client-observed medians. Prompt counts are the actual
server-reported counts, not requested target sizes.

| Configuration | Model | Actual prompt | n | Output | TTFT (s) | Decode tok/s | Range | Peak VRAM GiB | Min host available GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reddit_20260819T092710Z_ninfer | qwen38-ninfer-nvfp4 | 997 | 3 | 768 | 0.129 | 198.56 | 184.97-199.36 | 29.14 | 54.37 |
| reddit_20260819T092710Z_ninfer | qwen38-ninfer-nvfp4 | 31418 | 3 | 768 | 4.513 | 185.67 | 175.43-189.72 | 29.14 | 54.32 |
| reddit_20260819T092710Z_ninfer | qwen38-ninfer-nvfp4 | 95849 | 3 | 768 | 21.953 | 165.96 | 150.76-167.36 | 29.14 | 54.25 |
| reddit_20260819T092710Z_ninfer | qwen38-ninfer-nvfp4 | 182099 | 3 | 768 | 62.633 | 151.8 | 148.08-155.09 | 29.15 | 54.08 |
| reddit_20260819T092710Z_ninfer_exact190k | qwen38-ninfer-nvfp4 | 190003 | 3 | 768 | 67.698 | 153.94 | 146.38-155.36 | 28.88 | 54.05 |

## Quality smoke tests

- reddit_20260819T092710Z_ninfer_quality: 7/7 passed

## Prefix reuse

| Configuration | Shared target tokens | Prime TTFT (s) | Cached TTFT (s) | Cached decode tok/s |
|---|---:|---:|---:|---:|
| reddit_20260819T092710Z_ninfer_prefix | 190000 | 62.722 | 0.413 | 133.20 |

## Raw sources

- `raw_results/reddit_20260819T092710Z/reddit_20260819T092710Z_ninfer_20260819T112835_aa02bfed.json`
- `raw_results/reddit_20260819T092710Z/reddit_20260819T092710Z_ninfer_exact190k_20260819T113417_576acea5.json`
- `raw_results/reddit_20260819T092710Z/reddit_20260819T092710Z_ninfer_prefix_20260819T113808_81569f74.json`
- `raw_results/reddit_20260819T092710Z/reddit_20260819T092710Z_ninfer_quality_20260819T113926_f34ea7f6.json`
- `raw_results/reddit_20260819T092710Z/reddit_20260819T092710Z_vllm_unsloth_mtp3_20260819T114500_0017d79f.json`
- `raw_results/reddit_20260819T092710Z/reddit_20260819T092710Z_vllm_unsloth_mtp3_20260819T114718_d46beb1c.json`
