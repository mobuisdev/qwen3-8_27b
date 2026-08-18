# Third-party notices

This repository includes compact metadata copied from model repositories and a
small source patch derived from SGLang. Model weights and framework source
trees are not committed.

| Tracked path | Upstream project | License and source |
|---|---|---|
| `models/qwen_base/` | Qwen/Qwen3.8-27B | Apache-2.0; [source](https://huggingface.co/Qwen/Qwen3.8-27B); [license](models/qwen_base/LICENSE) |
| `models/radixark/` | RadixArk/Qwen3.8-27B-NVFP4 | Apache-2.0; [source](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4); [license](models/radixark/LICENSE) |
| `models/unsloth/` | unsloth/Qwen3.8-27B-NVFP4 | Apache-2.0 as declared by its model card; [source](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); [license](models/unsloth/LICENSE) |
| `patches/sglang-pr-34904-lm-head.patch` | SGLang PR #34904 | Apache-2.0; [source](https://github.com/sgl-project/sglang/pull/34904); [upstream license](https://github.com/sgl-project/sglang/blob/main/LICENSE) |

The revisions actually tested are pinned in `models/repositories.json`,
`docs/RERUN.md`, and the setup scripts. The repository's MIT license applies only
to the benchmark code and documentation authored for this project; it does not
replace these upstream terms.
