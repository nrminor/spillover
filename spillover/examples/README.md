# spillover examples

These examples focus on the generic `spillover` crate and show how to compose sort keys, comparators, codecs, dedup strategies, and merge controls.

- [`01_basic_u64_sort.rs`](./01_basic_u64_sort.rs): minimal external sort over `u64` values using `Owned` sort keys and a tiny binary codec.
- [`02_custom_record_and_sort_key.rs`](./02_custom_record_and_sort_key.rs): sorting a custom record type with a borrowed tuple sort key.
- [`03_reverse_and_custom_compare.rs`](./03_reverse_and_custom_compare.rs): custom comparator for case-insensitive ordering plus reverse sort direction.
- [`04_adjacent_dedup.rs`](./04_adjacent_dedup.rs): post-merge adjacent deduplication with `AdjacentDedup`.
- [`05_keyed_codec_merge.rs`](./05_keyed_codec_merge.rs): keyed merge path with coarse precomputed keys and fallback full-record ordering.
- [`06_spill_controls.rs`](./06_spill_controls.rs): memory/fan-in controls to force disk spilling and intermediate merge passes.

Run an example from the workspace root with:

```bash
cargo run -p spillover --example 01_basic_u64_sort
```
