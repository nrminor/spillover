# spillover-bio

`spillover-bio` is a genomics-focused sorting crate built on top of [ `spillover` ](https://crates.io/crates/spillover).

It provides a ready-to-use disk-spilling sort pipeline for FASTQ/FASTA-style sequence records, with practical defaults for:

- sequence-aware sort orders (including quality tie-breaking)
- dryice-backed temporary storage codecs
- keyed merge acceleration for large spill-heavy workloads
- optional adjacent deduplication strategies
- optional arena-backed ingest for allocation-conscious callers

Builders return Core sorters configured with Bio's orders, codec compatibility, and
deduplication. Single-read owned ingress accepts `SeqRecord`; convert borrowed views with
`.into()` when using that path. For more allocation-sensitive ingestion paths,
`SeqRecordArena` offers an alternate way to copy records into the sorter: the
sorter copies each pushed record into arena storage and reuses that allocation
across spill windows. This is not zero-copy from parser input, but it can
amortize allocation cost after startup when records are read through reusable
buffers.

Finished output supports owned iteration or borrowed
`output.items().try_for_each(...)` traversal. Both share adjacent deduplication
history. Custom orders implement Core's comparison/key traits, designate one owned
input with `BioOrder`, and grant `CompatibleWith<C>` for supported base codecs.
Custom chunk sorters implement Core's algorithm traits directly.

Whole read pairs use the same builder and sorting lifecycle:

```rust
use spillover_bio::{
    record::{ReadPair, SeqRecord},
    sort::Builder,
};

let mut sorter = Builder::new()
    .sort_by_illumina_pairs()
    .max_buffer_items(1)
    .build();
let r1 = SeqRecord::new(b"read/1", b"AC", b"!!");
let r2 = SeqRecord::new(b"read/2", b"GT", b"##");
let pair = ReadPair::from_mates(r1, r2);
sorter.push(pair)?;
let output = sorter.finish()?;
for result in output {
    let (r1, r2) = result?.into_mates();
    // Deliver both mates, retaining their R1/R2 roles.
}
```

Applications assemble mates and choose how to handle orphans; `from_mates` does
not validate identity. Optional `pair.validate_with(callback)` borrows the pair
and preserves the application's error type. Sorting and decoding never invoke it.
Do not use `zip` to silently truncate unequal input streams.

`sort_by_pair_sequence_exact()` compares `(seq1, seq2, qual1, qual2, name1, name2)`.
`sort_by_illumina_pairs()` compares a 38-byte R1 packed key first, then that full
tuple. The 64-byte variant is `sort_by(ILLUMINA_250_PAIR_ORDER)`.
`sort_by_pair_name()` compares both names; `PAIR_QUALITY_ORDER` and
`PAIR_LENGTHS_ORDER` work with `sort_by_unkeyed`. Reversal changes ordering,
never mate roles. Equal pairs have no stable-input-order guarantee.

Initial pair sorting uses raw component encodings, sequential comparison sorting,
and preserves duplicates. Single-record dedup selectors and arena storage are
rejected for pair input. Existing `sort_by_paired_end()` still sorts individual
reads with a wider key; it does not assemble or sort pairs.

Item limits count pairs; dryice block sizes count physical reads. Temporary runs
store R1 then R2, repeating the supplied key on both mates for keyed runs. Pairs
may cross block boundaries. Decoders reject missing mates and disagreeing keys,
but do not establish biological pairing. Each open merge cursor eagerly retains
both owned mates plus block buffers; the measured active-window budget is not a
process-RSS limit.

For custom orders migrating from `OrderCodecPolicy`, replace the policy declaration
with an owned-input declaration and explicit codec grants:

```rust
impl BioOrder for MyOrder {
    type Input = SeqRecord; // or ReadPair<SeqRecord>
}
impl CompatibleWith<DryIceCodec> for MyOrder {}
```

The default `DryIceCodec` grant accepts raw encodings only. Changing encodings
does not change the owned input. Extensions must establish ordering coherence
through encoding and repeated merges; a grant does not prove it.
Default output now carries core `NoDedup`; selecting a dedup method changes the
builder/output type to `RecordDedup`. Ordinary method chains retain their
spelling, but conditional reassignment into the default builder type must be
restructured. `SortedRecordStream<I>` names the explicit-dedup stream.

See [`examples/07_arena_backed_ingest.rs`](./examples/07_arena_backed_ingest.rs)
for a small arena-backed example.
See [`examples/08_sort_read_pairs.rs`](./examples/08_sort_read_pairs.rs) for a runnable
pair-sorting example that forces spilling:

```bash
mise exec -- cargo run -p spillover-bio --example 08_sort_read_pairs
```

For project-level context, architecture, and runnable examples, see the main repository README:

- <https://github.com/nrminor/spillover/blob/main/README.md>

For API documentation:

- <https://docs.rs/spillover-bio>
