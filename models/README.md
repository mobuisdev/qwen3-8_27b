# Model metadata

This tracked directory contains compact repository metadata, configuration
files, revisions, licenses, and checkpoint audits. The audit scripts use it to
compare the tested checkpoints without committing model weights. Repository-
specific Git attributes are excluded because this repository tracks tokenizer
metadata as ordinary Git objects rather than Hugging Face LFS pointers.

Large model objects remain in the ignored `../hf-home/` and
`../ninfer-models/` directories. `downloaded_snapshots.json` is also ignored:
it is a generated machine-local manifest containing absolute paths and can be
recreated with `../scripts/download_models.py`.

Some tokenizer, processor, and license files are byte-identical across pinned
checkpoints. They remain in each snapshot so every model directory is complete
and auditable in isolation; Git stores identical content only once internally.

`upstream_pins.json` records immutable publication revisions, dated candidate
revisions, and the deployed DeepSeek Harness revision.
`../scripts/check_upstream_updates.py` compares them with official upstream
heads without downloading weights. `repositories.json` also inventories
candidate and NInfer repositories, but only the three inputs used by
`audit_models.py` have complete metadata snapshots copied into this directory;
candidate weights and frontend files remain in ignored caches.
