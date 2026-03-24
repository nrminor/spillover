# spillover

`spillover` is a generic Rust crate for building disk-spilling (external) sort pipelines for datasets that do not fit in memory.

It provides reusable sorting primitives and lets you plug in your own:

- sort key extraction (`SortKey`)
- comparison (`Compare`)
- temporary on-disk format (`Codec` / `KeyedCodec`)
- post-merge deduplication (`Dedup`)
- in-memory chunk sorting strategy (`ChunkSorter`)

For project-level context and examples across both crates, see the main repository README:

- <https://github.com/nrminor/spillover/blob/main/README.md>

For API documentation:

- <https://docs.rs/spillover>
