//! In-memory sorting without codecs, temporary files, or a memory budget.
//!
//! All records, arena views, algorithm scratch, and dedup history must fit in
//! memory. Built-in algorithms do not promise stable ordering of equal items.
//! Records retain their original
//! fields: no codec transforms, normalizes, or omits them.
//!
//! Sort a collection in place, or build a sorter to accept records incrementally:
//!
//! ```
//! use spillover_bio::{record::SeqRecord, sort::in_memory::Builder};
//! let mut reads = vec![SeqRecord::new("b", "C", "!"), SeqRecord::new("a", "A", "!")];
//! Builder::new().sort_by_name().sort_slice(&mut reads);
//! assert_eq!(reads[0].name(), b"a");
//! let mut sorter = Builder::new().sort_by_name().build_from(reads);
//! sorter.push(SeqRecord::new("c", "G", "!"));
//! let output: Vec<_> = sorter.finish().collect::<Result<_, _>>().unwrap();
//! assert_eq!(output.len(), 3);
//! ```
//!
//! Deduplication applies to adjacent records, so the chosen order must make
//! matches consecutive. `sort_vec` compacts in place; slices cannot shrink:
//!
//! ```compile_fail
//! use spillover_bio::{record::SeqRecord, sort::in_memory::Builder};
//! let mut reads: Vec<SeqRecord> = Vec::new();
//! Builder::new().sort_by_name().dedup_by_name().sort_slice(&mut reads);
//! ```
//!
//! Arena ingestion copies input bytes into reusable storage. Output iteration
//! moves views, not independently owned records. Constructing owned records
//! from those views copies their fields while arena storage is still live.
//! Both iteration and `items().try_for_each` share deduplication history.
//! Memory output has an `Infallible` source error; a failing visitor is reported
//! separately as a sink error.
//!
//! ```
//! use std::convert::Infallible;
//! use spillover_bio::{record::{SeqRecord, SeqRecordArena}, sort::in_memory::Builder};
//! let mut arena = SeqRecordArena::new();
//! let mut sorter = Builder::new().sort_by_name().arena(&mut arena).build();
//! sorter.push(&SeqRecord::new("a", "A", "!"));
//! let mut owned = Vec::new();
//! sorter.finish().items().try_for_each(|view| {
//!     owned.push(SeqRecord::from_slices(view.name(), view.sequence(), view.quality()));
//!     Ok::<(), Infallible>(())
//! }).expect("visitor succeeds");
//! assert_eq!(owned[0].name(), b"a");
//! ```
//!
//! Arena output yields views into the arena. Dropping output does not release
//! borrows held by views that escaped it:
//!
//! ```compile_fail
//! use spillover_bio::{record::{SeqRecord, SeqRecordArena}, sort::in_memory::Builder};
//! let mut arena = SeqRecordArena::new();
//! let mut sorter = Builder::new().sort_by_name().arena(&mut arena).build();
//! sorter.push(&SeqRecord::new("a", "A", "!"));
//! let mut output = sorter.finish();
//! let view = output.next().unwrap().unwrap();
//! drop(output);
//! let next = Builder::new().sort_by_name().arena(&mut arena).build();
//! assert_eq!(view.sequence(), b"A");
//! ```
//!
//! A view copied out of borrowed visitation also keeps the arena borrowed:
//!
//! ```compile_fail
//! use std::convert::Infallible;
//! use spillover_bio::{record::{SeqRecord, SeqRecordArena}, sort::in_memory::Builder};
//! let mut arena = SeqRecordArena::new();
//! let mut sorter = Builder::new().sort_by_name().arena(&mut arena).build();
//! sorter.push(&SeqRecord::new("a", "A", "!"));
//! let mut escaped = None;
//! sorter.finish().items().try_for_each(|view| {
//!     escaped = Some(*view);
//!     Ok::<(), Infallible>(())
//! }).unwrap();
//! let next = Builder::new().sort_by_name().arena(&mut arena).build();
//! assert_eq!(escaped.unwrap().sequence(), b"A");
//! ```
//!
//! ```compile_fail
//! use spillover_bio::{record::SeqRecordArena, sort::in_memory::Builder};
//! let mut arena = SeqRecordArena::new();
//! let output = Builder::new().sort_by_name().arena(&mut arena).build().finish();
//! let next = Builder::new().sort_by_name().arena(&mut arena).build();
//! assert_eq!(output.count(), 0);
//! ```
//!
//! Pairs cannot use arena storage or the built-in single-record deduplicators:
//!
//! ```compile_fail
//! use spillover_bio::{record::SeqRecordArena, sort::in_memory::Builder};
//! let mut arena = SeqRecordArena::new();
//! let sorter = Builder::new().sort_by_illumina_pairs().arena(&mut arena).build();
//! ```
//! ```compile_fail
//! use spillover_bio::sort::in_memory::Builder;
//! let sorter = Builder::new().sort_by_illumina_pairs().dedup_by_sequence().build();
//! ```
//! Disk settings are not available:
//! ```compile_fail
//! use spillover_bio::{codec::DryIceCodec, sort::in_memory::Builder};
//! let sorter = Builder::new().sort_by_name().codec(DryIceCodec::new()).build();
//! ```
//! ```compile_fail
//! use spillover_bio::sort::in_memory::Builder;
//! let sorter = Builder::new().sort_by_name().measured_budget(1024).build();
//! ```
//! ```compile_fail
//! use spillover::{merge::MergeConfig};
//! use spillover_bio::sort::in_memory::Builder;
//! let sorter = Builder::new().sort_by_name().merge_config(MergeConfig::default()).build();
//! ```

use super::{
    ArenaStorage, BioOrder, ILLUMINA_ORDER, ILLUMINA_PAIR_ORDER, LENGTH_ORDER, LONG_READ_ORDER,
    LengthOrder, NAME_ORDER, NameOrder, OwnedStorage, PAIR_NAME_ORDER, PAIR_SEQUENCE_EXACT_ORDER,
    PAIRED_END_ORDER, PairNameOrder, PairSequenceExactOrder, PairSequenceOrder, RecordDedup,
    SequenceOrder, UnkeyedSequenceOrder,
};
use crate::{
    radix::RadixThenRefine,
    record::{SeqRecord, SeqRecordArena, SeqRecordView},
};
use spillover::{
    chunk::{ChunkSorter, Sequential},
    compare::Compare,
    dedup::{AdjacentDeduplicator, NoDedup},
    sorter::in_memory as core,
};

/// Configure a memory-only sorter or sort an existing collection.
pub struct Builder<O = (), D = NoDedup, CS = Sequential, S = OwnedStorage> {
    core: core::Builder<O, D, CS>,
    storage: S,
}

impl Builder {
    /// Start a new memory-only configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            core: core::Builder::new(),
            storage: OwnedStorage,
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl<D, CS, S> Builder<(), D, CS, S> {
    /// Select an order. Collections only require [`Compare`]; `build` also
    /// uses [`BioOrder`] to infer the owned input.
    #[must_use]
    pub fn sort_by<O>(self, order: O) -> Builder<O, D, CS, S> {
        Builder {
            core: self.core.sort_by(order),
            storage: self.storage,
        }
    }

    /// Sort single reads using the 150-base packed sequence order.
    #[must_use]
    pub fn sort_by_illumina(self) -> Builder<SequenceOrder<38>, D, RadixThenRefine<38>, S> {
        let builder = self.sort_by(ILLUMINA_ORDER);
        builder.chunk_sort(RadixThenRefine)
    }

    /// Sort single reads using the 250-base packed sequence order.
    #[must_use]
    pub fn sort_by_paired_end(self) -> Builder<SequenceOrder<64>, D, RadixThenRefine<64>, S> {
        let builder = self.sort_by(PAIRED_END_ORDER);
        builder.chunk_sort(RadixThenRefine)
    }

    /// Sort single reads using the long-read packed sequence order.
    #[must_use]
    pub fn sort_by_long_read(self) -> Builder<SequenceOrder<128>, D, RadixThenRefine<128>, S> {
        let builder = self.sort_by(LONG_READ_ORDER);
        builder.chunk_sort(RadixThenRefine)
    }

    /// Sort by exact raw sequence and quality, using sequential comparison.
    #[must_use]
    pub fn sort_by_sequence_exact(self) -> Builder<UnkeyedSequenceOrder, D, Sequential, S> {
        let builder = self.sort_by(UnkeyedSequenceOrder);
        builder.sort_with_sequential()
    }

    /// Sort by name.
    #[must_use]
    pub fn sort_by_name(self) -> Builder<NameOrder, D, CS, S> {
        self.sort_by(NAME_ORDER)
    }

    /// Sort by sequence length.
    #[must_use]
    pub fn sort_by_length(self) -> Builder<LengthOrder, D, CS, S> {
        self.sort_by(LENGTH_ORDER)
    }

    /// Sort whole pairs with the packed Illumina order.
    #[must_use]
    pub fn sort_by_illumina_pairs(self) -> Builder<PairSequenceOrder<38>, D, Sequential, S> {
        let builder = self.sort_by(ILLUMINA_PAIR_ORDER);
        builder.sort_with_sequential()
    }

    /// Sort whole pairs by exact sequence, quality, and name.
    #[must_use]
    pub fn sort_by_pair_sequence_exact(self) -> Builder<PairSequenceExactOrder, D, Sequential, S> {
        let builder = self.sort_by(PAIR_SEQUENCE_EXACT_ORDER);
        builder.sort_with_sequential()
    }

    /// Sort whole pairs by their names.
    #[must_use]
    pub fn sort_by_pair_name(self) -> Builder<PairNameOrder, D, CS, S> {
        self.sort_by(PAIR_NAME_ORDER)
    }
}

impl<O, D, CS, S> Builder<O, D, CS, S> {
    /// Select a compatible sorting algorithm.
    #[must_use]
    pub fn chunk_sort<Next>(self, algorithm: Next) -> Builder<O, D, Next, S> {
        Builder {
            core: self.core.chunk_sort(algorithm),
            storage: self.storage,
        }
    }

    /// Use sequential comparison sorting.
    #[must_use]
    pub fn sort_with_sequential(self) -> Builder<O, D, Sequential, S> {
        self.chunk_sort(Sequential)
    }

    /// Use parallel comparison sorting.
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn sort_with_parallel(self) -> Builder<O, D, spillover::chunk::Parallel, S> {
        self.chunk_sort(spillover::chunk::Parallel)
    }

    /// Select an adjacent deduplication algorithm.
    #[must_use]
    pub fn dedup<Next>(self, algorithm: Next) -> Builder<O, Next, CS, S> {
        Builder {
            core: self.core.dedup(algorithm),
            storage: self.storage,
        }
    }

    /// Deduplicate adjacent single reads by name.
    #[must_use]
    pub fn dedup_by_name(self) -> Builder<O, RecordDedup, CS, S> {
        self.dedup(RecordDedup::by_name())
    }

    /// Deduplicate adjacent single reads by sequence.
    #[must_use]
    pub fn dedup_by_sequence(self) -> Builder<O, RecordDedup, CS, S> {
        self.dedup(RecordDedup::by_sequence())
    }

    /// Deduplicate adjacent single reads by sequence and quality.
    #[must_use]
    pub fn dedup_by_sequence_and_quality(self) -> Builder<O, RecordDedup, CS, S> {
        self.dedup(RecordDedup::by_sequence_and_quality())
    }
}

impl<O, D, CS> Builder<O, D, CS> {
    /// Use caller-owned scratch storage, cleared at build while retaining capacity.
    #[must_use]
    pub fn arena(self, arena: &mut SeqRecordArena) -> Builder<O, D, CS, ArenaStorage<'_>> {
        Builder {
            core: self.core,
            storage: ArenaStorage { arena },
        }
    }

    /// Sort and deduplicate a vector in place, retaining its allocation.
    pub fn sort_vec<R>(self, records: &mut Vec<R>)
    where
        O: Compare<R>,
        CS: ChunkSorter<R, O>,
        D: AdjacentDeduplicator<R>,
    {
        self.core.sort_vec(records);
    }

    /// Transfer a vector and its allocation into a sorter that accepts more items.
    #[must_use]
    pub fn build_from<R>(self, records: Vec<R>) -> core::Sorter<Vec<R>, O, D, CS>
    where
        O: Compare<R>,
        CS: ChunkSorter<R, O>,
        D: AdjacentDeduplicator<R>,
    {
        self.core.build_from(records)
    }
}

impl<O, CS> Builder<O, NoDedup, CS> {
    /// Sort a slice in place without deduplication.
    pub fn sort_slice<R>(&self, records: &mut [R])
    where
        O: Compare<R>,
        CS: ChunkSorter<R, O>,
    {
        self.core.sort_slice(records);
    }
}

impl<O, D, CS> Builder<O, D, CS>
where
    O: BioOrder + Compare<O::Input>,
    D: AdjacentDeduplicator<O::Input>,
    CS: ChunkSorter<O::Input, O>,
{
    /// Build an owned sorter, inferring its input from the order.
    #[must_use]
    pub fn build(self) -> core::Sorter<Vec<O::Input>, O, D, CS> {
        self.core.build()
    }
}

impl<'a, O, D, CS> Builder<O, D, CS, ArenaStorage<'a>>
where
    O: BioOrder<Input = SeqRecord> + Compare<SeqRecordView<'a>>,
    CS: ChunkSorter<SeqRecordView<'a>, O>,
    D: AdjacentDeduplicator<SeqRecordView<'a>>,
{
    /// Build an arena sorter. Its output borrows the arena.
    #[must_use]
    pub fn build(self) -> core::Sorter<&'a mut SeqRecordArena, O, D, CS> {
        self.core.build_in(self.storage.arena)
    }
}
