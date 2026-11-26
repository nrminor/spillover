# spillover

`spillover` is a generic Rust crate for building disk-spilling (external) sort pipelines for datasets that do not fit in memory.

It provides reusable sorting primitives and lets you plug in your own:

- sort key extraction (`SortKey`) or complete record ordering
- comparison (`Compare`)
- run reading (`DecodeRun` or `Codec` / `KeyedCodec`) and writing (`EncodeRun` / `EncodeKeyedRun`)
- post-merge adjacent deduplication (`AdjacentDeduplicator`)
- in-memory chunk sorting strategy (`ChunkSorter`)

For project-level context and examples across both crates, see the main repository README:

- <https://github.com/nrminor/spillover/blob/main/README.md>

For API documentation:

- <https://docs.rs/spillover>
