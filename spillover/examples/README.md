# spillover examples

These examples focus on the generic `spillover` crate and show how to compose sort keys, comparators, codecs, dedup strategies, and merge controls.

- [`01_basic_u64_sort.rs`](./01_basic_u64_sort.rs): minimal external sort over `u64` values using the complete `Natural` order and a tiny binary codec.
- [`02_custom_record_and_sort_key.rs`](./02_custom_record_and_sort_key.rs): sorting a custom record type with an owned tuple sort key.
- [`03_reverse_and_custom_compare.rs`](./03_reverse_and_custom_compare.rs): a complete case-insensitive record order wrapped in `Reverse`.
- [`04_adjacent_dedup.rs`](./04_adjacent_dedup.rs): post-merge adjacent deduplication with `AdjacentDedup`.
- [`05_keyed_codec_merge.rs`](./05_keyed_codec_merge.rs): `KeyedOrder` and order-side `DeriveKey` supply coarse keys and full-record tie-breaking; the codec preserves keys through merging.
- [`06_spill_controls.rs`](./06_spill_controls.rs): item-count and fan-in controls to force disk spilling and intermediate merge passes.

Run against the local workspace code, rather than the crates.io release:

```bash
cargo run -p spillover --example 01_basic_u64_sort
```
