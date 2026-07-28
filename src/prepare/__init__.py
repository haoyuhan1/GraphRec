"""Per-dataset raw → TFRecord preparation for the 14 paper benchmarks.

Every module here exposes a `build(raw_dir, out_dir, **kw)` that returns
`(sequences, item_text)` and writes the GRID TFRecord layout through
`common.write_dataset`. Run them via `python -m src.prepare <dataset>` or
`scripts/prepare_all.sh`.
"""
