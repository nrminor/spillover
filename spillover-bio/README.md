# spillover-bio

`spillover-bio` is a genomics-focused sorting crate built on top of [ `spillover` ](https://crates.io/crates/spillover).

It provides a ready-to-use disk-spilling sort pipeline for FASTQ/FASTA-style sequence records, with practical defaults for:

- sequence-aware sort orders (including quality tie-breaking)
- dryice-backed temporary storage codecs
- keyed merge acceleration for large spill-heavy workloads
- optional adjacent deduplication strategies
- optional arena-backed ingest for allocation-conscious callers

Builders return Core sorters configured with Bio's orders, codec policy, and
deduplication. Owned ingress accepts `SeqRecord`; convert borrowed views with
`.into()` when using that path. For more allocation-sensitive ingestion paths,
`SeqRecordArena` offers an alternate way to copy records into the sorter: the
sorter copies each pushed record into arena storage and reuses that allocation
across spill windows. This is not zero-copy from parser input, but it can
amortize allocation cost after startup when records are read through reusable
buffers.

Finished output supports owned iteration or borrowed
`output.items().try_for_each(...)` traversal. Both share adjacent deduplication
history. Custom orders implement Core's comparison/key traits and Bio's
`OrderCodecPolicy`; `RawRecords` permits only raw sequence, quality, and name
encoding. Custom chunk sorters implement Core's algorithm traits directly.

See [`examples/07_arena_backed_ingest.rs`](./examples/07_arena_backed_ingest.rs)
for a small arena-backed example.

For project-level context, architecture, and runnable examples, see the main repository README:

- <https://github.com/nrminor/spillover/blob/main/README.md>

For API documentation:

- <https://docs.rs/spillover-bio>
