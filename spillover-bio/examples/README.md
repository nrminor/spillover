# spillover-bio examples

These examples focus on `spillover-bio`'s ready-to-use sorting pipeline for sequence records, including keyed merge acceleration, codec choices, deduplication, and alternate sort orders.
Application types are imported from the crate root; specialized order
constants remain in `sort`. `Builder` spills to disk, while
`InMemoryBuilder` explicitly keeps the input in memory.

- [`01_illumina_quickstart.rs`](./01_illumina_quickstart.rs): default "just sort my reads" flow using `sort_by_illumina()`.
- [`02_keyed_vs_unkeyed_tradeoff.rs`](./02_keyed_vs_unkeyed_tradeoff.rs): side-by-side accelerated Illumina and exact sequence sorting comparison.
- [`03_quality_tiebreak_across_spills.rs`](./03_quality_tiebreak_across_spills.rs): equal-sequence records across spills, showing quality tiebreak preservation.
- [`04_dedup_sequence.rs`](./04_dedup_sequence.rs): adjacent sequence dedup for high-duplicate datasets.
- [`05_codec_compaction.rs`](./05_codec_compaction.rs): dryice codec options (`two_bit_exact`, `binned_quality`, `split_names`).
- [`06_alternate_orders.rs`](./06_alternate_orders.rs): name sorting, length sorting, and reversed sequence order.
- [`07_arena_backed_ingest.rs`](./07_arena_backed_ingest.rs): borrowed input views, arena loan release before output consumption, and borrowed output through `SeqRecordSink`.
- [`08_sort_read_pairs.rs`](./08_sort_read_pairs.rs): assemble whole pairs, spill with a one-pair limit, and consume sorted pairs without exchanging R1/R2 roles.
- [`09_in_memory_sorting.rs`](./09_in_memory_sorting.rs): sort and compact existing collections, transfer a Vec into a push/finish sorter, traverse arena views, reuse arena capacity, and sort whole pairs without temporary files.

Run against the local workspace code, rather than the crates.io release:

```bash
cargo run -p spillover-bio --example 01_illumina_quickstart
```
