# spillover-bio examples

These examples focus on `spillover-bio`'s ready-to-use sorting pipeline for sequence records, including keyed merge acceleration, codec choices, deduplication, and alternate sort orders.

- [`01_illumina_quickstart.rs`](./01_illumina_quickstart.rs): default "just sort my reads" flow using `sort_by_illumina()`.
- [`02_keyed_vs_unkeyed_tradeoff.rs`](./02_keyed_vs_unkeyed_tradeoff.rs): side-by-side keyed and unkeyed timing/behavior comparison.
- [`03_quality_tiebreak_across_spills.rs`](./03_quality_tiebreak_across_spills.rs): equal-sequence records across spills, showing quality tiebreak preservation.
- [`04_dedup_sequence.rs`](./04_dedup_sequence.rs): adjacent sequence dedup for high-duplicate datasets.
- [`05_codec_compaction.rs`](./05_codec_compaction.rs): dryice codec options (`two_bit_exact`, `binned_quality`, `split_names`).
- [`06_alternate_orders.rs`](./06_alternate_orders.rs): name sorting, length sorting, and reversed sequence order.
- [`07_arena_backed_ingest.rs`](./07_arena_backed_ingest.rs): arena-backed ingest for amortizing allocation when pushing borrowed record views.

Run an example from the workspace root with:

```bash
cargo run -p spillover-bio --example 01_illumina_quickstart
```
