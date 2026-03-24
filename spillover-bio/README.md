# spillover-bio

`spillover-bio` is a genomics-focused sorting crate built on top of [ `spillover` ](https://crates.io/crates/spillover).

It provides a ready-to-use disk-spilling sort pipeline for FASTQ/FASTA-style sequence records, with practical defaults for:

- sequence-aware sort orders (including quality tie-breaking)
- dryice-backed temporary storage codecs
- keyed merge acceleration for large spill-heavy workloads
- optional adjacent deduplication strategies

For project-level context, architecture, and runnable examples, see the main repository README:

- <https://github.com/nrminor/spillover/blob/main/README.md>

For API documentation:

- <https://docs.rs/spillover-bio>
