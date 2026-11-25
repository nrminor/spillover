# spillover: disk-spilling, generic merge sorting for larger-than-memory data

[![CI](https://github.com/nrminor/spillover/actions/workflows/ci.yml/badge.svg)](https://github.com/nrminor/spillover/actions/workflows/ci.yml)
[![crates.io: spillover](https://img.shields.io/crates/v/spillover.svg)](https://crates.io/crates/spillover)
[![docs.rs: spillover](https://docs.rs/spillover/badge.svg)](https://docs.rs/spillover)
[![crates.io: spillover-bio](https://img.shields.io/crates/v/spillover-bio.svg)](https://crates.io/crates/spillover-bio)
[![docs.rs: spillover-bio](https://docs.rs/spillover-bio/badge.svg)](https://docs.rs/spillover-bio)

## Overview

`spillover` is a generic Rust library for building external (disk-spilling) sort pipelines.

The core crate is intentionally unopinionated: it knows how to spill sorted runs to disk and merge them back, but leaves domain behavior to user-provided traits. You can plug in your own implementations for:

1. sort key extraction (`SortKey`) or complete record ordering
2. comparison (`Compare`)
3. run reading (`DecodeRun` or `Codec` / `KeyedCodec`) and writing (`EncodeRun` / `EncodeKeyedRun`)
4. post-merge deduplication (`Dedup`)
5. in-memory chunk sorting strategy (`ChunkSorter`)
6. flush strategy (byte-based or item-count based budgets)

`spillover-bio` builds on `spillover` with genomics-focused defaults and convenience APIs. It uses [`dryice`](https://github.com/nrminor/dryice) for temporary on-disk storage, includes packed sequence keys for keyed merge acceleration, provides FASTQ/FASTA-friendly record handling, and includes practical sort orders (sequence, quality tiebreaking, name, length, reverse order).

That said, `spillover-bio` is still extensible: users can configure dryice encodings, sort orders, built-in deduplication modes, and flush controls. When you need full control, you can always drop down to `spillover` directly.

Here is a small `spillover-bio` quickstart:

```rust
use spillover_bio::{codec::DryIceCodec, record::SeqRecord, sort::Builder};

fn rec(name: &str, seq: &str, qual: &str) -> SeqRecord {
    SeqRecord::new(
        name.as_bytes(),
        seq.as_bytes(),
        qual.as_bytes(),
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
- If you need a custom domain pipeline, start with `spillover` and provide ordering plus run-reading and run-writing implementations.
- Use small buffer budgets in development to force spills and validate merge behavior early.

You can run examples from this workspace root:

```bash
cargo run -p spillover --example 01_basic_u64_sort
cargo run -p spillover-bio --example 01_illumina_quickstart
```

## Development

This repository uses [mise](https://mise.jdx.dev/) for its development toolchain and task runner. Git is also required to fetch the read-only reference repositories used during development.

```bash
mise trust
mise install
mise run setup
mise tasks
mise run check
```

## Concepts & Architecture

At a high level, both crates follow the same pipeline:

1. collect records into an in-memory buffer
2. sort the buffer using the selected key/comparator/chunk sorter
3. spill sorted runs to temporary files when flush budget is reached
4. merge runs with bounded fan-in
5. optionally apply post-merge deduplication

Put another way, in `spillover`, sorting is an engine with interchangeable parts:

- Define ordering with a `SortKey`/`Compare` pair or a complete record order; keyed complete orders also implement `KeyedOrder` for stored-key comparison. `ChunkSorter` selects the in-memory sorting algorithm.
- Define temporary-run reading with `DecodeRun` for ordinary owned-record iterators, or `Codec` / `KeyedCodec` for lower-level control; define writing with `EncodeRun` / `EncodeKeyedRun`.
- Define what "duplicate" means in your domain with `Dedup`. Custom deduplication logic can be introduced with this interface.
- Control spilling with byte budgets (`measured_budget`) or item limits (`max_buffer_items`); these are not hard limits on total process memory.

Three practical mental models help when designing a pipeline:

1. **Sort semantics are a contract.** Your configured order must capture the order your downstream logic depends on, including stored-key comparisons and record tie-breaking.
2. **Disk format is a performance lever.** Better temporary encodings and keyed merge support can significantly reduce merge-time work.
3. **Memory limits are part of correctness.** Budgets are not tuning afterthoughts; they define when data spills and therefore how the algorithm behaves at scale.

`spillover-bio` adds domain-specific defaults around that pipeline: sequence-aware sort orders (with quality tie-breaking), dryice-backed codecs, keyed merge acceleration, and convenience builders for common genomics workflows.

## Citation

If this project helps your work, please cite the repository and crate versions you used:

- repository: <https://github.com/nrminor/spillover>
- crates: <https://crates.io/crates/spillover>, <https://crates.io/crates/spillover-bio>
