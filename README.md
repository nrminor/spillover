# spillover: disk-spilling, generic merge sorting for larger-than-memory data

[![CI](https://github.com/nrminor/spillover/actions/workflows/ci.yml/badge.svg)](https://github.com/nrminor/spillover/actions/workflows/ci.yml)
[![crates.io: spillover](https://img.shields.io/crates/v/spillover.svg)](https://crates.io/crates/spillover)
[![docs.rs: spillover](https://docs.rs/spillover/badge.svg)](https://docs.rs/spillover)
[![crates.io: spillover-bio](https://img.shields.io/crates/v/spillover-bio.svg)](https://crates.io/crates/spillover-bio)
[![docs.rs: spillover-bio](https://docs.rs/spillover-bio/badge.svg)](https://docs.rs/spillover-bio)

## Overview

`spillover` is a generic Rust library for building external (disk-spilling) sort pipelines.

The core crate is intentionally unopinionated: it knows how to spill sorted runs to disk and merge them back, but leaves domain behavior to user-provided traits. You can plug in your own implementations for:

1. sort key extraction (`SortKey`)
2. comparison (`Compare`)
3. on-disk format (`Codec` / `KeyedCodec`)
4. post-merge deduplication (`Dedup`)
5. in-memory chunk sorting strategy (`ChunkSorter`)
6. flush strategy (byte-based or item-count based budgets)

`spillover-bio` builds on `spillover` with genomics-focused defaults and convenience APIs. It uses [`dryice`](https://github.com/nrminor/dryice) for temporary on-disk storage, includes packed sequence keys for keyed merge acceleration, provides FASTQ/FASTA-friendly record handling, and includes practical sort orders (sequence, quality tiebreaking, name, length, reverse order).

That said, `spillover-bio` is still extensible: users can choose codecs, sort orders, dedup strategies, and flush controls. When you need full control, you can always drop down to `spillover` directly.

Here is a small `spillover-bio` quickstart:

```rust
use spillover_bio::{codec::DryIceCodec, record::SeqRecord, sort::Builder};

fn rec(name: &str, seq: &str, qual: &str) -> SeqRecord {
    SeqRecord::new(
        name.as_bytes().to_vec(),
        seq.as_bytes().to_vec(),
        qual.as_bytes().to_vec(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    for record in [
        rec("r1", "TTTTTTTT", "IIIIIIII"),
        rec("r2", "AAAAAAAA", "!!!!!!!!"),
        rec("r3", "CCCCCCCC", "########"),
    ] {
        sorter.push(record)?;
    }

    let sorted: Vec<SeqRecord> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    assert_eq!(sorted[0].sequence(), b"AAAAAAAA");
    assert_eq!(sorted[1].sequence(), b"CCCCCCCC");
    assert_eq!(sorted[2].sequence(), b"TTTTTTTT");

    Ok(())
}
```

Further runnable examples:

- `spillover`: [`spillover/examples/README.md`](./spillover/examples/README.md)
- `spillover-bio`: [`spillover-bio/examples/README.md`](./spillover-bio/examples/README.md)

## Getting Started

### Installation

```bash
cargo add spillover
# or
cargo add spillover-bio
```

### First steps

- If you want a ready-to-use genomics pipeline, start with `spillover-bio`.
- If you need a custom domain pipeline, start with `spillover` and implement your own `Codec` and `SortKey`.
- Use small buffer budgets in development to force spills and validate merge behavior early.

You can run examples from this workspace root:

```bash
cargo run -p spillover --example 01_basic_u64_sort
cargo run -p spillover-bio --example 01_illumina_quickstart
```

## Concepts & Architecture

At a high level, both crates follow the same pipeline:

1. collect records into an in-memory buffer
2. sort the buffer using the selected key/comparator/chunk sorter
3. spill sorted runs to temporary files when flush budget is reached
4. merge runs with bounded fan-in
5. optionally apply post-merge deduplication

Put another way, in `spillover`, sorting is an engine with interchangeable parts:

- Define what order means by choosing a `SortKey` and `Compare` pair. `SortKey` is the item you sort on, and `Compare` defines the sorting algorithm.
- Define how temporary runs are represented with `Codec` (or `KeyedCodec` when you can store compact precomputed keys). These types describe how records are represented on disk as well as how they should be parsed and written.
- Define what "duplicate" means in your domain with `Dedup`. Custom deduplication logic can be introduced with this interface.
- Define memory behavior explicitly with flush budgets (`measured_budget`, `max_buffer_items`, etc.). `spillover` ultimately uses n-way merge sorting. The budgets provided here determine when `spillover` flushes batches from memory onto disk to keep memory usage constant.

Three practical mental models help when designing a pipeline:

1. **Sort semantics are a contract.** Your key/comparator pair should fully capture the order your downstream logic depends on.
2. **Disk format is a performance lever.** Better temporary encodings and keyed merge support can significantly reduce merge-time work.
3. **Memory limits are part of correctness.** Budgets are not tuning afterthoughts; they define when data spills and therefore how the algorithm behaves at scale.

`spillover-bio` adds domain-specific defaults around that pipeline: sequence-aware sort orders (with quality tie-breaking), dryice-backed codecs, keyed merge acceleration, and convenience builders for common genomics workflows.

## Citation

If this project helps your work, please cite the repository and crate versions you used:

- repository: <https://github.com/nrminor/spillover>
- crates: <https://crates.io/crates/spillover>, <https://crates.io/crates/spillover-bio>
