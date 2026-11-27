//! Genomics-specific sort keys, sort orders, and the sorter builder.
//!
//! Sort keys extract the value to sort by from record-shaped values.
//! Sort orders implement core comparison and key-derivation traits;
//! Bio selects merge strategies and compatible codecs and algorithms.
//!
//! Accelerated sequence orders compare packed sequence keys first, then
//! raw sequence and quality bytes. Exact sequence ordering compares
//! raw sequence bytes, then quality bytes.
//!
//! The [`Builder`] is the main entry point for creating a sorter.

use std::{fs::File, io::BufReader, marker::PhantomData};

use dryice::{
    DryIceWriter, NameCodec, QualityCodec, RawAsciiCodec, RawNameCodec, RawQualityCodec, RecordKey,
    SequenceCodec,
};
use get_size2::GetSize;
use spillover::{
    chunk::{ChunkSorter, Sequential},
    codec::{Codec, CodecCursor, EncodeKeyedRun, EncodeRun, KeyedCodec},
    compare::Compare,
    dedup::{AdjacentDeduplicator, NoDedup},
    key::{DeriveKey, KeyedOrder, SortKey},
    merge::MergeConfig,
    sorter::{
        ByteLimit, HasOrder, ItemLimit, Keyed as CoreKeyed, NeedsOrder as CoreNeedsOrder,
        Unkeyed as CoreUnkeyed,
    },
};

use crate::{
    codec::{DryIceCodec, KeyedDryIceCodec, PairCodec, SeqRecordPartsAdapter},
    key::PackedSequenceKey,
    radix::RadixThenRefine,
    record::{ReadPair, SeqRecord, SeqRecordArena, SeqRecordParts, SeqRecordView},
};

pub mod in_memory;
mod pair;
pub use pair::{
    ILLUMINA_250_PAIR_ORDER, ILLUMINA_PAIR_ORDER, PAIR_LENGTHS_ORDER, PAIR_NAME_ORDER,
    PAIR_QUALITY_ORDER, PAIR_SEQUENCE_EXACT_ORDER, PairLengthsOrder, PairNameOrder,
    PairQualityOrder, PairSequenceExactOrder, PairSequenceOrder,
};
use sealed::RunShape;

/// Sort key that extracts sequence and quality as a tuple.
///
/// The tuple `(&[u8], &[u8])` implements [`Ord`] lexicographically,
/// so sequence is compared first and quality serves as the
/// tiebreaker — no custom comparator needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct SequenceQualityKey;

impl<R: SeqRecordParts> SortKey<R> for SequenceQualityKey {
    type Key<'a>
        = (&'a [u8], &'a [u8])
    where
        R: 'a;

    fn key<'a>(&self, item: &'a R) -> (&'a [u8], &'a [u8]) {
        (item.sequence(), item.quality())
    }
}

/// Sort key that extracts the record name/identifier.
#[derive(Debug, Clone, Copy, Default)]
pub struct NameKey;

impl<R: SeqRecordParts> SortKey<R> for NameKey {
    type Key<'a>
        = &'a [u8]
    where
        R: 'a;

    fn key<'a>(&self, item: &'a R) -> &'a [u8] {
        item.name()
    }
}

/// Sort key that extracts the sequence length as a `u64`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LengthKey;

impl<R: SeqRecordParts> SortKey<R> for LengthKey {
    type Key<'a>
        = u64
    where
        R: 'a;

    #[allow(clippy::cast_possible_truncation)]
    fn key(&self, item: &R) -> u64 {
        item.sequence().len() as u64
    }
}

// ── Owned input and codec compatibility ──────────────────

mod sealed {
    /// Resolves codec type-state to a concrete [`crate::codec::DryIceCodec`].
    /// `NeedsCodec` yields the raw (no-encoding) default;
    /// `HasCodec` passes through the configured choice.
    pub trait ResolveCodec {
        type S: dryice::SequenceCodec + Copy + 'static;
        type Q: dryice::QualityCodec + Copy + 'static;
        type N: dryice::NameCodec + Copy + 'static;

        fn resolve(self) -> crate::codec::DryIceCodec<Self::S, Self::Q, Self::N>;
    }

    /// Selects framing for supported owned inputs and dedup configurations.
    pub trait RunShape<C, D> {
        type Codec;
        fn frame(codec: C) -> Self::Codec;
    }
}

/// Declares an order's owned input for a sorting configuration.
///
/// Bio supports [`SeqRecord`] and [`ReadPair<SeqRecord>`] as owned inputs.
/// Comparison can remain generic
/// over borrowed record representations using core's [`Compare`] and
/// [`DeriveKey`] traits. Other owned shapes can use core's builder directly.
///
/// Grant compatibility for each supported base codec or for
/// [`in_memory::InMemory`] when building an empty memory-only sorter:
///
/// ```
/// use std::cmp::Ordering;
/// use spillover::compare::Compare;
/// use spillover_bio::{
///     codec::DryIceCodec,
///     record::{SeqRecord, SeqRecordParts},
///     sort::{Builder, CompatibleWith},
/// };
///
/// struct QualityThenName;
///
/// impl<R: SeqRecordParts + ?Sized> Compare<R> for QualityThenName {
///     fn compare(&self, left: &R, right: &R) -> Ordering {
///         let quality = left.quality().cmp(right.quality());
///         quality.then_with(|| left.name().cmp(right.name()))
///     }
/// }
///
/// impl CompatibleWith<DryIceCodec> for QualityThenName {
///     type Input = SeqRecord;
/// }
///
/// let mut sorter = Builder::new()
///     .sort_by_unkeyed(QualityThenName)
///     .codec(DryIceCodec::new())
///     .max_buffer_items(1)
///     .build();
/// sorter.push(SeqRecord::new(b"read", b"A", b"!"))?;
/// let collected: Result<Vec<SeqRecord>, _> = sorter.finish()?.collect();
/// assert_eq!(collected?.len(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// A grant is not proof of ordering coherence. Extension authors must preserve
/// comparisons through encoding, decoding, and intermediate merges. Delegating
/// to a built-in order's grants does not establish those laws for another order.
/// Granting just [`DryIceCodec`] conservatively accepts only raw sequence,
/// quality, and name codecs. No universal raw-codec grant is implicit.
///
/// Changing an ungranted component is rejected at build:
///
/// ```compile_fail
/// use std::cmp::Ordering;
/// use spillover::compare::Compare;
/// use spillover_bio::{
///     codec::DryIceCodec,
///     record::{SeqRecord, SeqRecordParts},
///     sort::{Builder, CompatibleWith},
/// };
///
/// struct ByName;
/// impl<R: SeqRecordParts + ?Sized> Compare<R> for ByName {
///     fn compare(&self, left: &R, right: &R) -> Ordering {
///         left.name().cmp(right.name())
///     }
/// }
/// impl CompatibleWith<DryIceCodec> for ByName {
///     type Input = SeqRecord;
/// }
/// let codec = DryIceCodec::new().omit_names();
/// let _sorter = Builder::new()
///     .sort_by_unkeyed(ByName)
///     .codec(codec)
///     .max_buffer_items(1)
///     .build();
/// ```
///
/// Sequence and quality encodings must also remain raw:
///
/// ```compile_fail
/// # use std::cmp::Ordering;
/// # use spillover::compare::Compare;
/// # use spillover_bio::{codec::DryIceCodec, record::{SeqRecord, SeqRecordParts}, sort::{Builder, CompatibleWith}};
/// # struct ByName;
/// # impl<R: SeqRecordParts + ?Sized> Compare<R> for ByName {
/// #     fn compare(&self, left: &R, right: &R) -> Ordering { left.name().cmp(right.name()) }
/// # }
/// # impl CompatibleWith<DryIceCodec> for ByName { type Input = SeqRecord; }
/// let codec = DryIceCodec::new().two_bit_lossy_n();
/// let _sorter = Builder::new()
///     .sort_by_unkeyed(ByName)
///     .codec(codec)
///     .max_buffer_items(1)
///     .build();
/// ```
///
/// ```compile_fail
/// # use std::cmp::Ordering;
/// # use spillover::compare::Compare;
/// # use spillover_bio::{codec::DryIceCodec, record::{SeqRecord, SeqRecordParts}, sort::{Builder, CompatibleWith}};
/// # struct ByName;
/// # impl<R: SeqRecordParts + ?Sized> Compare<R> for ByName {
/// #     fn compare(&self, left: &R, right: &R) -> Ordering { left.name().cmp(right.name()) }
/// # }
/// # impl CompatibleWith<DryIceCodec> for ByName { type Input = SeqRecord; }
/// let codec = DryIceCodec::new().binned_quality();
/// let _sorter = Builder::new()
///     .sort_by_unkeyed(ByName)
///     .codec(codec)
///     .max_buffer_items(1)
///     .build();
/// ```
pub trait CompatibleWith<C> {
    /// Canonical owned item for this configuration.
    type Input;
}

// ── Keyed sort orders ────────────────────────────────────

/// Sort by nucleotide sequence with quality tiebreaker,
/// using a 2-bit packed key of width N bytes (N×4 bases).
///
/// For most users, the convenience aliases are easier:
/// [`ILLUMINA_ORDER`], [`PAIRED_END_ORDER`], [`LONG_READ_ORDER`].
#[derive(Debug, Clone, Copy)]
pub struct SequenceOrder<const N: usize>;

impl<const N: usize, R: SeqRecordParts + ?Sized> Compare<R> for SequenceOrder<N> {
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        PackedSequenceKey::<N>::from_sequence(a.sequence())
            .cmp(&PackedSequenceKey::<N>::from_sequence(b.sequence()))
            .then_with(|| a.sequence().cmp(b.sequence()))
            .then_with(|| a.quality().cmp(b.quality()))
    }
}

impl<const N: usize> SequenceOrder<N> {
    /// Switch to exact raw sequence-then-quality ordering
    /// without record key acceleration.
    #[must_use]
    pub fn unkeyed(self) -> UnkeyedSequenceOrder {
        UnkeyedSequenceOrder
    }
}

/// Sort by record name, using a 16-byte name prefix as the
/// record key.
#[derive(Debug, Clone, Copy)]
pub struct NameOrder;

impl<R: SeqRecordParts + ?Sized> Compare<R> for NameOrder {
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        a.name().cmp(b.name())
    }
}

fn derive_name_key<R: SeqRecordParts + ?Sized>(record: &R) -> dryice::Bytes16Key {
    let mut key = [0u8; 16];
    let len = record.name().len().min(16);
    key[..len].copy_from_slice(&record.name()[..len]);
    dryice::Bytes16Key(key)
}

impl NameOrder {
    /// Opt out of record key acceleration.
    #[must_use]
    pub fn unkeyed(self) -> UnkeyedNameOrder {
        UnkeyedNameOrder
    }
}

/// Sort by sequence length, using an 8-byte big-endian u64
/// as the record key.
///
/// Records with the same sequence length compare equal under this order; their
/// relative order is not specified by `LengthOrder`.
#[derive(Debug, Clone, Copy)]
pub struct LengthOrder;

impl<R: SeqRecordParts + ?Sized> Compare<R> for LengthOrder {
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        a.sequence().len().cmp(&b.sequence().len())
    }
}

#[allow(clippy::cast_possible_truncation)]
fn derive_length_key<R: SeqRecordParts + ?Sized>(record: &R) -> dryice::Bytes8Key {
    dryice::Bytes8Key((record.sequence().len() as u64).to_be_bytes())
}

impl LengthOrder {
    /// Opt out of record key acceleration.
    #[must_use]
    pub fn unkeyed(self) -> UnkeyedLengthOrder {
        UnkeyedLengthOrder
    }
}

impl<const N: usize> spillover::key::KeyedOrder for SequenceOrder<N> {
    type StoredKey = PackedSequenceKey<N>;
    fn compare_keys(&self, left: &Self::StoredKey, right: &Self::StoredKey) -> std::cmp::Ordering {
        left.cmp(right)
    }
}
impl<const N: usize, R: SeqRecordParts + ?Sized> spillover::key::DeriveKey<R> for SequenceOrder<N> {
    fn derive_key(&self, item: &R) -> Self::StoredKey {
        PackedSequenceKey::from_sequence(item.sequence())
    }
}
impl<const N: usize, R: SeqRecordParts + ?Sized> spillover::key::EstimateKeyMemory<R>
    for SequenceOrder<N>
{
    fn estimate_key_memory(&self, _: &R) -> spillover::key::KeyMemory {
        spillover::key::KeyMemory::NONE
    }
}

impl spillover::key::KeyedOrder for NameOrder {
    type StoredKey = dryice::Bytes16Key;
    fn compare_keys(&self, left: &Self::StoredKey, right: &Self::StoredKey) -> std::cmp::Ordering {
        left.cmp(right)
    }
}
impl<R: SeqRecordParts + ?Sized> spillover::key::DeriveKey<R> for NameOrder {
    fn derive_key(&self, item: &R) -> Self::StoredKey {
        derive_name_key(item)
    }
}
impl<R: SeqRecordParts + ?Sized> spillover::key::EstimateKeyMemory<R> for NameOrder {
    fn estimate_key_memory(&self, _: &R) -> spillover::key::KeyMemory {
        spillover::key::KeyMemory::NONE
    }
}

impl spillover::key::KeyedOrder for LengthOrder {
    type StoredKey = dryice::Bytes8Key;
    fn compare_keys(&self, left: &Self::StoredKey, right: &Self::StoredKey) -> std::cmp::Ordering {
        left.cmp(right)
    }
}
impl<R: SeqRecordParts + ?Sized> spillover::key::DeriveKey<R> for LengthOrder {
    fn derive_key(&self, item: &R) -> Self::StoredKey {
        derive_length_key(item)
    }
}
impl<R: SeqRecordParts + ?Sized> spillover::key::EstimateKeyMemory<R> for LengthOrder {
    fn estimate_key_memory(&self, _: &R) -> spillover::key::KeyMemory {
        spillover::key::KeyMemory::NONE
    }
}

// ── Unkeyed sort orders ──────────────────────────────────

/// Sort by sequence without record key acceleration.
#[derive(Debug, Clone, Copy)]
pub struct UnkeyedSequenceOrder;

impl<R: SeqRecordParts + ?Sized> Compare<R> for UnkeyedSequenceOrder {
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        a.sequence()
            .cmp(b.sequence())
            .then_with(|| a.quality().cmp(b.quality()))
    }
}

/// Sort by name without record key acceleration.
#[derive(Debug, Clone, Copy)]
pub struct UnkeyedNameOrder;

impl<R: SeqRecordParts + ?Sized> Compare<R> for UnkeyedNameOrder {
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        a.name().cmp(b.name())
    }
}

/// Sort by length without record key acceleration.
#[derive(Debug, Clone, Copy)]
pub struct UnkeyedLengthOrder;

impl<R: SeqRecordParts + ?Sized> Compare<R> for UnkeyedLengthOrder {
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        a.sequence().len().cmp(&b.sequence().len())
    }
}

// ── Reverse wrapper ──────────────────────────────────────

/// Reverse any sort order. Flips the comparison direction
/// but keeps the same key pairing and merge strategy.
///
/// ```ignore
/// .sort_by(Reverse(ILLUMINA_ORDER))   // Z→A
/// .sort_by(Reverse(LengthOrder))     // longest first
/// ```
pub use spillover::compare::Reverse;

impl<O: CompatibleWith<C>, C> CompatibleWith<C> for Reverse<O> {
    type Input = <O as CompatibleWith<C>>::Input;
}

macro_rules! existing_codec_choices {
    ($($order:ty),+ $(,)?) => {$(
        impl<S, Q, N> CompatibleWith<DryIceCodec<S, Q, N>> for $order
        where
            S: SequenceCodec + Copy + 'static,
            Q: QualityCodec + Copy + 'static,
            N: NameCodec + Copy + 'static,
        {
            type Input = SeqRecord;
        }
    )+};
}

existing_codec_choices!(NameOrder, LengthOrder, UnkeyedNameOrder, UnkeyedLengthOrder);

impl<const WIDTH: usize, S, Q, N> CompatibleWith<DryIceCodec<S, Q, N>> for SequenceOrder<WIDTH>
where
    S: SequenceCodec + Copy + 'static,
    Q: QualityCodec + Copy + 'static,
    N: NameCodec + Copy + 'static,
{
    type Input = SeqRecord;
}

impl<N: NameCodec + Copy + 'static> CompatibleWith<DryIceCodec<RawAsciiCodec, RawQualityCodec, N>>
    for UnkeyedSequenceOrder
{
    type Input = SeqRecord;
}

macro_rules! raw_pair_orders {
    ($($order:ty),+ $(,)?) => {$(
        impl CompatibleWith<DryIceCodec> for $order {
            type Input = ReadPair<SeqRecord>;
        }
        impl CompatibleWith<in_memory::InMemory> for $order {
            type Input = ReadPair<SeqRecord>;
        }
    )+};
}

raw_pair_orders!(
    PairSequenceExactOrder,
    PairNameOrder,
    PairQualityOrder,
    PairLengthsOrder
);

impl<const N: usize> CompatibleWith<DryIceCodec> for PairSequenceOrder<N> {
    type Input = ReadPair<SeqRecord>;
}

impl<const N: usize> CompatibleWith<in_memory::InMemory> for PairSequenceOrder<N> {
    type Input = ReadPair<SeqRecord>;
}

macro_rules! memory_single_orders {
    ($($order:ty),+ $(,)?) => {$(
        impl CompatibleWith<in_memory::InMemory> for $order {
            type Input = SeqRecord;
        }
    )+};
}

memory_single_orders!(
    NameOrder,
    LengthOrder,
    UnkeyedNameOrder,
    UnkeyedLengthOrder,
    UnkeyedSequenceOrder
);

impl<const N: usize> CompatibleWith<in_memory::InMemory> for SequenceOrder<N> {
    type Input = SeqRecord;
}

// ── Convenience aliases ──────────────────────────────────

/// Sequence sort with 38-byte key (152 bases, covers 150bp Illumina).
pub const ILLUMINA_ORDER: SequenceOrder<38> = SequenceOrder;

/// Sequence sort with 64-byte key (256 bases, covers 250bp paired-end).
pub const PAIRED_END_ORDER: SequenceOrder<64> = SequenceOrder;

/// Sequence sort with 128-byte key (512 bases, prefix for long reads).
pub const LONG_READ_ORDER: SequenceOrder<128> = SequenceOrder;

/// Name sort with 16-byte prefix key.
pub const NAME_ORDER: NameOrder = NameOrder;

/// Length sort with 8-byte big-endian key.
pub const LENGTH_ORDER: LengthOrder = LengthOrder;

// ── Builder ──────────────────────────────────────────────

/// Post-merge adjacent deduplication, applied through Core's [`AdjacentDeduplicator`].
///
/// Each enabled mode keeps the first record in a consecutive group of matches.
/// Global deduplication requires an order that makes all matching records
/// contiguous. Owned iteration and borrowed visits share the same history.
/// This stateful algorithm is not `Copy`. Its retained packed record reuses
/// allocated capacity when later accepted records replace it. The retained
/// output history is outside the modeled initial-spill memory budget.
#[derive(Debug)]
pub struct RecordDedup {
    mode: RecordDedupMode,
    previous: Option<RecordRepresentative>,
}

#[derive(Debug)]
enum RecordDedupMode {
    Name,
    Sequence,
    SequenceAndQuality,
}

impl RecordDedup {
    /// Match record names.
    #[must_use]
    pub fn by_name() -> Self {
        Self {
            mode: RecordDedupMode::Name,
            previous: None,
        }
    }

    /// Match sequence bytes.
    #[must_use]
    pub fn by_sequence() -> Self {
        Self {
            mode: RecordDedupMode::Sequence,
            previous: None,
        }
    }

    /// Match sequence and quality bytes.
    #[must_use]
    pub fn by_sequence_and_quality() -> Self {
        Self {
            mode: RecordDedupMode::SequenceAndQuality,
            previous: None,
        }
    }
}

/// The associated representative of [`RecordDedup`], with private packed fields.
///
/// It implements [`SeqRecordParts`] to expose the captured record's fields.
#[derive(Debug, Default)]
pub struct RecordRepresentative {
    bytes: Vec<u8>,
    name_len: usize,
    sequence_len: usize,
}

impl RecordRepresentative {
    fn copy_from<R: SeqRecordParts + ?Sized>(&mut self, record: &R) {
        let total_len = record.name().len() + record.sequence().len() + record.quality().len();

        self.bytes.clear();
        self.bytes.reserve(total_len);
        self.name_len = record.name().len();
        self.sequence_len = record.sequence().len();
        self.bytes.extend_from_slice(record.name());
        self.bytes.extend_from_slice(record.sequence());
        self.bytes.extend_from_slice(record.quality());
    }

    fn sequence_start(&self) -> usize {
        self.name_len
    }

    fn sequence_end(&self) -> usize {
        self.name_len + self.sequence_len
    }
}

impl SeqRecordParts for RecordRepresentative {
    fn name(&self) -> &[u8] {
        &self.bytes[..self.name_len]
    }

    fn sequence(&self) -> &[u8] {
        &self.bytes[self.sequence_start()..self.sequence_end()]
    }

    fn quality(&self) -> &[u8] {
        &self.bytes[self.sequence_end()..]
    }
}

impl<R: SeqRecordParts + ?Sized> AdjacentDeduplicator<R> for RecordDedup {
    type Representative = RecordRepresentative;

    fn matches(&self, current: &R) -> bool {
        self.previous
            .as_ref()
            .is_some_and(|previous| match self.mode {
                RecordDedupMode::Name => previous.has_same_name_as(current),
                RecordDedupMode::Sequence => previous.has_same_sequence_as(current),
                RecordDedupMode::SequenceAndQuality => {
                    previous.has_same_sequence_and_quality_as(current)
                }
            })
    }

    fn capture(&self, record: &R) -> RecordRepresentative {
        let mut representative = RecordRepresentative::default();
        representative.copy_from(record);
        representative
    }

    fn replace(&mut self, representative: RecordRepresentative) {
        self.previous = Some(representative);
    }

    fn update(&mut self, record: &R) {
        let previous = self
            .previous
            .get_or_insert_with(RecordRepresentative::default);
        previous.copy_from(record);
    }
}

/// Builder marker: no codec chosen yet.
pub struct NeedsCodec;
/// Builder marker: codec has been chosen.
pub struct HasCodec<S, Q, N>(DryIceCodec<S, Q, N>);

/// Builder marker: use the default owned-record buffer.
pub struct OwnedStorage;

/// Builder marker: use caller-provided arena storage for the spill window.
pub struct ArenaStorage<'a> {
    arena: &'a mut SeqRecordArena,
}

/// Default memory budget when `.measured_budget()` is not called:
/// 1 GiB.
const DEFAULT_BUDGET: usize = 1 << 30;

// ── Sealed trait impls for default resolution ────────────

impl sealed::ResolveCodec for NeedsCodec {
    type S = RawAsciiCodec;
    type Q = RawQualityCodec;
    type N = RawNameCodec;

    fn resolve(self) -> DryIceCodec {
        DryIceCodec::new()
    }
}

impl<S, Q, N> sealed::ResolveCodec for HasCodec<S, Q, N>
where
    S: SequenceCodec + Copy + 'static,
    Q: QualityCodec + Copy + 'static,
    N: NameCodec + Copy + 'static,
{
    type S = S;
    type Q = Q;
    type N = N;

    fn resolve(self) -> DryIceCodec<S, Q, N> {
        self.0
    }
}

impl<C, D> RunShape<C, D> for SeqRecord {
    type Codec = C;

    fn frame(codec: C) -> C {
        codec
    }
}

impl<S, Q, N> RunShape<DryIceCodec<S, Q, N>, NoDedup> for ReadPair<SeqRecord> {
    type Codec = PairCodec<DryIceCodec<S, Q, N>>;

    fn frame(codec: DryIceCodec<S, Q, N>) -> Self::Codec {
        PairCodec::new(codec)
    }
}

impl<K, S, Q, N> RunShape<KeyedDryIceCodec<K, S, Q, N>, NoDedup> for ReadPair<SeqRecord> {
    type Codec = PairCodec<KeyedDryIceCodec<K, S, Q, N>>;

    fn frame(codec: KeyedDryIceCodec<K, S, Q, N>) -> Self::Codec {
        PairCodec::new(codec)
    }
}

/// Builder for configuring a genomics-oriented external sorter.
///
/// Built sorters are Core sorters: owned ingress accepts the order's
/// [`CompatibleWith::Input`] (`SeqRecord` or `ReadPair<SeqRecord>`), and
/// arena ingress can use [`SeqRecordView`]. Convert other owned inputs explicitly
/// before pushing; normalize parser records to views for measured keyed arenas.
/// Final output supports owned iteration or Core's `items().try_for_each(...)`
/// borrowed traversal. Bio supplies codec compatibility and deduplication, not another
/// sorter lifecycle.
///
/// Only a sort order is required — everything else has sensible
/// defaults:
///
/// ```ignore
/// let mut sorter = Builder::new()
///     .sort_by_illumina()
///     .build();
/// ```
///
/// Defaults: raw (uncompressed) dryice codec, 1 GiB memory budget,
/// and sequential comparison sorting. The Illumina, paired-end,
/// and long-read convenience methods select radix chunk sorting for
/// their compatible accelerated sequence orders.
///
/// Deduplication is disabled by default. Selecting a `dedup_by_*` method
/// changes the builder's concrete type to carry [`RecordDedup`]; selecting
/// another dedup mode replaces the previous choice. Conditional configuration
/// must therefore branch before building rather than reassign an enabled
/// builder into a variable holding the default type.
///
/// Pair sorting preserves whole pairs and duplicates. The single-record dedup
/// selectors and arena storage are not supported for pairs, regardless of
/// configuration order:
///
/// ```compile_fail
/// use spillover_bio::sort::Builder;
/// let _sorter = Builder::new().dedup_by_name().sort_by_pair_name().build();
/// ```
///
/// ```compile_fail
/// use spillover_bio::sort::Builder;
/// let _sorter = Builder::new().sort_by_pair_name().dedup_by_name().build();
/// ```
///
/// ```compile_fail
/// use spillover_bio::sort::Builder;
/// let _sorter = Builder::new().dedup_by_sequence().sort_by_illumina_pairs().build();
/// ```
///
/// ```compile_fail
/// use spillover_bio::sort::Builder;
/// let _sorter = Builder::new().sort_by_illumina_pairs().dedup_by_sequence_and_quality().build();
/// ```
///
/// ```compile_fail
/// use spillover_bio::{record::SeqRecordArena, sort::Builder};
/// let mut arena = SeqRecordArena::new();
/// let _sorter = Builder::new().arena(&mut arena).sort_by_pair_name().build();
/// ```
///
/// ```compile_fail
/// use spillover_bio::{record::SeqRecordArena, sort::Builder};
/// let mut arena = SeqRecordArena::new();
/// let _sorter = Builder::new().sort_by_illumina_pairs().arena(&mut arena).build();
/// ```
///
/// Pair input cannot be replaced by a single read:
///
/// ```compile_fail
/// use spillover_bio::{record::SeqRecord, sort::Builder};
/// let mut sorter = Builder::new().sort_by_pair_name().build();
/// sorter.push(SeqRecord::new(b"r1", b"A", b"!")).unwrap();
/// ```
///
/// Built-in pair orders initially require raw encodings for all fields:
///
/// ```compile_fail
/// use spillover_bio::{codec::DryIceCodec, sort::Builder};
/// let codec = DryIceCodec::new().omit_quality();
/// let _sorter = Builder::new().sort_by_pair_sequence_exact().codec(codec).build();
/// ```
///
/// Override any default with the corresponding builder method:
///
/// ```ignore
/// let mut sorter = Builder::new()
///     .sort_by_paired_end()
///     .codec(DryIceCodec::new().two_bit_exact().binned_quality())
///     .measured_budget(4 * 1024 * 1024 * 1024)
///     .sort_with_sequential()
///     .dedup_by_sequence()
///     .build();
/// ```
///
/// For high-duplicate datasets, consider using an external bloom
/// filter in your ingest path to skip likely-duplicate records
/// before they enter the sort pipeline. This can reduce memory
/// usage, disk I/O, and merge work:
///
/// ```ignore
/// let mut bloom = fastbloom::BloomFilter::with_false_pos(0.001)
///     .expected_items(10_000_000);
///
/// for record in records {
///     if !bloom.contains(record.sequence()) {
///         bloom.insert(record.sequence());
///         sorter.push(record)?;
///     }
/// }
/// ```
pub struct Builder<Core = InitialCore, C = NeedsCodec, S = OwnedStorage, M = ()> {
    core: Core,
    codec: C,
    storage: S,
    mode: PhantomData<fn() -> M>,
}

type CoreBuilder<SK, L, CS, D> =
    spillover::sorter::Builder<SK, spillover::sorter::NeedsCodec, L, (), D, CS>;
type OrderedCore<O, L, CS, D> = CoreBuilder<HasOrder<O>, L, CS, D>;
type InitialCore = CoreBuilder<CoreNeedsOrder, ByteLimit, Sequential, NoDedup>;

/// Finalized stream of sorted sequence records with explicit deduplication.
///
/// `SortedRecordStream` owns the sorted output source after a sorter has
/// been finished. It implements [`Iterator`] for owned materialization, so
/// ordinary `.collect()` callsites continue to work.
/// Core wraps this adapter in [`spillover::Sorted`]. Its borrowed traversal uses
/// the same deduplication history as owned iteration.
/// Default builder output instead uses [`NoDedup`] and preserves all records.
pub type SortedRecordStream<I> = spillover::dedup::Deduplicated<I, RecordDedup>;

/// Destination for sequence records produced by a sorted stream.
///
/// Use Core's `output.items().try_for_each(...)` to deliver borrowed records.
/// For writers with an inherent `write_record` method, such as [`DryIceWriter`],
/// call `SeqRecordSink::write_record(&mut writer, record)` to select this adapter.
pub trait SeqRecordSink {
    /// Error returned when the destination fails to write a record.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Write one sequence record to the destination.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination rejects the record or fails while
    /// writing it.
    fn write_record<R: SeqRecordParts + ?Sized>(&mut self, record: &R) -> Result<(), Self::Error>;
}

impl<W, S, Q, N> SeqRecordSink for DryIceWriter<W, S, Q, N, dryice::NoRecordKey>
where
    W: std::io::Write,
    S: SequenceCodec,
    Q: QualityCodec,
    N: NameCodec,
{
    type Error = dryice::DryIceError;

    fn write_record<R: SeqRecordParts + ?Sized>(&mut self, record: &R) -> Result<(), Self::Error> {
        self.write_record(&SeqRecordPartsAdapter(record))
    }
}

impl Builder {
    /// Start building a sorter with a default 1 GiB measured byte budget.
    ///
    /// Later limit settings replace this default; the last setting wins.
    #[must_use]
    pub fn new() -> Self {
        let limit = ByteLimit::new(DEFAULT_BUDGET);
        let core = spillover::sorter::Builder::for_order().byte_limit(limit);
        Builder {
            core,
            codec: NeedsCodec,
            storage: OwnedStorage,
            mode: PhantomData,
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

// Sort order selection.

impl<Core, C, M> Builder<Core, C, OwnedStorage, M> {
    /// Use caller-provided arena storage for the active spill window.
    #[must_use]
    pub fn arena(self, arena: &mut SeqRecordArena) -> Builder<Core, C, ArenaStorage<'_>, M> {
        Builder {
            codec: self.codec,
            core: self.core,
            storage: ArenaStorage { arena },
            mode: self.mode,
        }
    }
}

impl<C, L, CS, D, S> Builder<CoreBuilder<CoreNeedsOrder, L, CS, D>, C, S> {
    /// Set a keyed sort order (merge compares record keys).
    #[must_use]
    pub fn sort_by<O: KeyedOrder>(
        self,
        order: O,
    ) -> Builder<OrderedCore<O, L, CS, D>, C, S, CoreKeyed> {
        self.map_core(|core| core.select_order(order))
    }

    /// Set an unkeyed sort order (merge deserializes full records).
    #[must_use]
    pub fn sort_by_unkeyed<O>(
        self,
        order: O,
    ) -> Builder<OrderedCore<O, L, CS, D>, C, S, CoreUnkeyed> {
        self.map_core(|core| core.select_order(order))
    }

    /// Sort by sequence (Illumina 150bp, 38-byte packed key).
    /// Defaults to radix sort for the chunk sorting engine.
    #[must_use]
    pub fn sort_by_illumina(
        self,
    ) -> Builder<OrderedCore<SequenceOrder<38>, L, RadixThenRefine<38>, D>, C, S, CoreKeyed> {
        self.sort_by(ILLUMINA_ORDER)
            .chunk_sort(RadixThenRefine::<38>)
    }

    /// Sort by exact raw sequence bytes, then raw quality bytes.
    /// This order requires raw sequence and quality codecs; the name codec may vary.
    ///
    /// This uses the unkeyed sequence order rather than the accelerated
    /// packed-key order used by [`Self::sort_by_illumina`]. It also selects the
    /// sequential comparator-based chunk sorter, even if a chunk sorter was
    /// configured earlier on the builder, because packed-key radix sorting is not
    /// compatible with exact raw sequence ordering.
    ///
    /// ```compile_fail
    /// use spillover_bio::{codec::DryIceCodec, sort::Builder};
    ///
    /// let codec = DryIceCodec::new().two_bit_lossy_n();
    /// let _sorter = Builder::new().sort_by_sequence_exact().codec(codec).build();
    /// ```
    #[must_use]
    pub fn sort_by_sequence_exact(
        self,
    ) -> Builder<OrderedCore<UnkeyedSequenceOrder, L, Sequential, D>, C, S, CoreUnkeyed> {
        self.sort_by_unkeyed(UnkeyedSequenceOrder)
            .sort_with_sequential()
    }

    /// Sort by sequence (250bp paired-end, 64-byte packed key).
    /// Defaults to radix sort for the chunk sorting engine.
    #[must_use]
    pub fn sort_by_paired_end(
        self,
    ) -> Builder<OrderedCore<SequenceOrder<64>, L, RadixThenRefine<64>, D>, C, S, CoreKeyed> {
        self.sort_by(PAIRED_END_ORDER)
            .chunk_sort(RadixThenRefine::<64>)
    }

    /// Sort by sequence (long reads, 128-byte prefix key).
    /// Defaults to radix sort for the chunk sorting engine.
    #[must_use]
    pub fn sort_by_long_read(
        self,
    ) -> Builder<OrderedCore<SequenceOrder<128>, L, RadixThenRefine<128>, D>, C, S, CoreKeyed> {
        self.sort_by(LONG_READ_ORDER)
            .chunk_sort(RadixThenRefine::<128>)
    }

    /// Sort by record name.
    #[must_use]
    pub fn sort_by_name(self) -> Builder<OrderedCore<NameOrder, L, CS, D>, C, S, CoreKeyed> {
        self.sort_by(NAME_ORDER)
    }

    /// Sort by sequence length.
    #[must_use]
    pub fn sort_by_length(self) -> Builder<OrderedCore<LengthOrder, L, CS, D>, C, S, CoreKeyed> {
        self.sort_by(LENGTH_ORDER)
    }

    /// Sort whole read pairs using a 38-byte R1 key, then exact pair refinement.
    ///
    /// Selects [`Sequential`] chunk sorting. Unlike [`Self::sort_by_paired_end`],
    /// each item is a complete [`ReadPair`], not an individual read.
    #[must_use]
    pub fn sort_by_illumina_pairs(
        self,
    ) -> Builder<OrderedCore<PairSequenceOrder<38>, L, Sequential, D>, C, S, CoreKeyed> {
        self.sort_by(ILLUMINA_PAIR_ORDER).sort_with_sequential()
    }

    /// Sort whole pairs by both sequences, both qualities, then both names.
    ///
    /// Selects [`Sequential`] chunk sorting and requires raw encoding.
    #[must_use]
    pub fn sort_by_pair_sequence_exact(
        self,
    ) -> Builder<OrderedCore<PairSequenceExactOrder, L, Sequential, D>, C, S, CoreUnkeyed> {
        self.sort_by_unkeyed(PAIR_SEQUENCE_EXACT_ORDER)
            .sort_with_sequential()
    }

    /// Sort whole pairs by R1 name, then R2 name, without deriving keys.
    #[must_use]
    pub fn sort_by_pair_name(
        self,
    ) -> Builder<OrderedCore<PairNameOrder, L, CS, D>, C, S, CoreUnkeyed> {
        self.sort_by_unkeyed(PAIR_NAME_ORDER)
    }
}

// Codec selection.

impl<Core, St, M> Builder<Core, NeedsCodec, St, M> {
    /// Set the dryice codec for temporary file encoding.
    #[must_use]
    pub fn codec<S, Q, N>(
        self,
        codec: DryIceCodec<S, Q, N>,
    ) -> Builder<Core, HasCodec<S, Q, N>, St, M> {
        Builder {
            codec: HasCodec(codec),
            core: self.core,
            storage: self.storage,
            mode: self.mode,
        }
    }
}

// Flush strategy selection.

impl<SK, C, L, CS, D, S, M> Builder<CoreBuilder<SK, L, CS, D>, C, S, M> {
    /// Set a byte budget for modeled active-window and initial-spill memory.
    ///
    /// Replaces any previous item or byte limit. The final configuration must
    /// provide the applicable storage, key, and algorithm memory estimates.
    ///
    /// An algorithm without memory estimation cannot use a final byte budget:
    ///
    /// ```compile_fail
    /// use spillover::{chunk::ChunkSorter, compare::Compare};
    /// use spillover_bio::sort::Builder;
    ///
    /// struct CountOnly;
    /// impl<T, O: Compare<T>> ChunkSorter<T, O> for CountOnly {
    ///     fn sort(&self, records: &mut [T], order: &O) {
    ///         records.sort_unstable_by(|left, right| order.compare(left, right));
    ///     }
    /// }
    ///
    /// let _sorter = Builder::new()
    ///     .sort_by_name()
    ///     .chunk_sort(CountOnly)
    ///     .max_buffer_items(1)
    ///     .measured_budget(1024)
    ///     .build();
    /// ```
    #[must_use]
    pub fn measured_budget(
        self,
        budget: usize,
    ) -> Builder<CoreBuilder<SK, ByteLimit, CS, D>, C, S, M> {
        let limit = ByteLimit::new(budget);
        self.map_core(|core| core.byte_limit(limit))
    }

    /// Flush when the buffer reaches `max_items` records.
    ///
    /// Replaces any previous item or byte limit, including the default byte
    /// budget. An item-limited sorter does not require memory estimates.
    #[must_use]
    pub fn max_buffer_items(
        self,
        max_items: usize,
    ) -> Builder<CoreBuilder<SK, ItemLimit, CS, D>, C, S, M> {
        let limit = ItemLimit::new(max_items);
        self.map_core(|core| core.item_limit(limit))
    }
}

impl<Core, C, S, M> Builder<Core, C, S, M> {
    fn map_core<NextCore, NextMode>(
        self,
        update: impl FnOnce(Core) -> NextCore,
    ) -> Builder<NextCore, C, S, NextMode> {
        let core = update(self.core);
        Builder {
            core,
            codec: self.codec,
            storage: self.storage,
            mode: PhantomData,
        }
    }
}

impl<SK, C, L, CS, D, S, M> Builder<CoreBuilder<SK, L, CS, D>, C, S, M> {
    /// Deduplicate adjacent records with identical names.
    #[must_use]
    pub fn dedup_by_name(self) -> Builder<CoreBuilder<SK, L, CS, RecordDedup>, C, S, M> {
        self.map_core(|core| core.dedup(RecordDedup::by_name()))
    }

    /// Deduplicate adjacent records with identical sequences.
    #[must_use]
    pub fn dedup_by_sequence(self) -> Builder<CoreBuilder<SK, L, CS, RecordDedup>, C, S, M> {
        self.map_core(|core| core.dedup(RecordDedup::by_sequence()))
    }

    /// Deduplicate adjacent records with identical sequence and quality.
    #[must_use]
    pub fn dedup_by_sequence_and_quality(
        self,
    ) -> Builder<CoreBuilder<SK, L, CS, RecordDedup>, C, S, M> {
        self.map_core(|core| core.dedup(RecordDedup::by_sequence_and_quality()))
    }
}

// Optional core configuration.

impl<SK, C, L, CS, D, S, M> Builder<CoreBuilder<SK, L, CS, D>, C, S, M> {
    /// Set a custom chunk sorting strategy. Defaults to
    /// [`Sequential`].
    ///
    /// The selected chunk sorter must be compatible with the selected sort order
    /// when [`build`](Self::build) is called. Comparator-based sorters such as
    /// [`Sequential`] are compatible with every built-in order. Specialized
    /// sorters such as [`RadixThenRefine`] are accepted only for the matching
    /// accelerated sequence order.
    #[must_use]
    pub fn chunk_sort<CS2>(self, chunk_sort: CS2) -> Builder<CoreBuilder<SK, L, CS2, D>, C, S, M> {
        self.map_core(|core| core.chunk_sort(chunk_sort))
    }

    /// Override the merge configuration.
    #[must_use]
    pub fn merge_config(mut self, config: MergeConfig) -> Self {
        self.core = self.core.merge_config(config);
        self
    }

    /// Use single-threaded comparison sort for in-memory chunks.
    ///
    /// This overrides the current chunk sorting strategy. The
    /// Illumina, paired-end, and long-read convenience methods select
    /// radix chunk sorting; most other builder paths keep the default
    /// sequential comparison sorter unless [`Self::chunk_sort`] is
    /// called explicitly.
    #[must_use]
    pub fn sort_with_sequential(self) -> Builder<CoreBuilder<SK, L, Sequential, D>, C, S, M> {
        self.chunk_sort(Sequential)
    }

    /// Use rayon parallel sort for in-memory chunks.
    ///
    /// This overrides the current chunk sorting strategy. Requires the
    /// `rayon` feature to be enabled on spillover.
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn sort_with_parallel(
        self,
    ) -> Builder<CoreBuilder<SK, L, spillover::chunk::Parallel, D>, C, S, M> {
        self.chunk_sort(spillover::chunk::Parallel)
    }
}

// build() for keyed sort orders.

type NativeKeyedCodec<O, C> = KeyedDryIceCodec<
    <O as KeyedOrder>::StoredKey,
    <C as sealed::ResolveCodec>::S,
    <C as sealed::ResolveCodec>::Q,
    <C as sealed::ResolveCodec>::N,
>;
type OwnedInput<O, C> = <O as CompatibleWith<ResolvedCodec<C>>>::Input;
type KeyedRunCodec<O, C, D> = <OwnedInput<O, C> as RunShape<NativeKeyedCodec<O, C>, D>>::Codec;
type NativeRecord<C> = <<C as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record;
type NativeSorter<O, C, CS, L, D> = spillover::sorter::Sorter<
    OwnedInput<O, C>,
    O,
    KeyedRunCodec<O, C, D>,
    (),
    D,
    CS,
    spillover::sorter::Keyed,
    L,
>;

macro_rules! owned_keyed_builder {
    ($limit:ty; $($extra:tt)*) => {
        impl<O, C, CS, D> Builder<OrderedCore<O, $limit, CS, D>, C, OwnedStorage, CoreKeyed>
        where
            O: CompatibleWith<ResolvedCodec<C>> + DeriveKey<OwnedInput<O, C>>
                + Compare<OwnedInput<O, C>> + Compare<NativeRecord<KeyedRunCodec<O, C, D>>>,
            O::StoredKey: RecordKey + Clone + 'static,
            C: sealed::ResolveCodec,
            OwnedInput<O, C>: RunShape<NativeKeyedCodec<O, C>, D>,
            KeyedRunCodec<O, C, D>: KeyedCodec<Key = O::StoredKey>
                + EncodeKeyedRun<OwnedInput<O, C>>
                + EncodeKeyedRun<NativeRecord<KeyedRunCodec<O, C, D>>>,
            for<'a> CS: ChunkSorter<spillover::key::Keyed<usize, O::StoredKey>, spillover::key::KeyedEntryCompare<'a, OwnedInput<O, C>, O>>,
            $($extra)*
        {
            /// Build the configured owned keyed sorter.
            #[must_use]
            pub fn build(self) -> NativeSorter<O, C, CS, $limit, D> {
                let codec = self.codec.resolve();
                let keyed_codec: NativeKeyedCodec<O, C> = codec.with_record_key();
                let framed = OwnedInput::<O, C>::frame(keyed_codec);
                self.core.keyed_codec(framed).build()
            }
        }
    };
}

owned_keyed_builder!(ByteLimit;
    OwnedInput<O, C>: GetSize,
    O: spillover::key::EstimateKeyMemory<OwnedInput<O, C>>,
    CS: spillover::chunk::EstimateSortMemory<spillover::key::Keyed<usize, O::StoredKey>>,
);
owned_keyed_builder!(ItemLimit;);

// build() for unkeyed sort orders.

type ResolvedCodec<C> = DryIceCodec<
    <C as sealed::ResolveCodec>::S,
    <C as sealed::ResolveCodec>::Q,
    <C as sealed::ResolveCodec>::N,
>;
type RunCodec<O, C, D> = <OwnedInput<O, C> as RunShape<ResolvedCodec<C>, D>>::Codec;
type NativeOrdinaryRecord<C> = <<C as Codec>::Cursor<BufReader<File>> as CodecCursor>::Record;
type NativeUnkeyedSorter<O, C, CS, L, D> = spillover::sorter::Sorter<
    OwnedInput<O, C>,
    O,
    RunCodec<O, C, D>,
    (),
    D,
    CS,
    spillover::sorter::Unkeyed,
    L,
>;
type NativeUnkeyedBuilt<O, C, CS, L, D> = NativeUnkeyedSorter<O, C, CS, L, D>;

impl<O, C, CS, D> Builder<OrderedCore<O, ItemLimit, CS, D>, C, OwnedStorage, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: CompatibleWith<ResolvedCodec<C>>
        + Compare<OwnedInput<O, C>>
        + Compare<NativeOrdinaryRecord<RunCodec<O, C, D>>>,
    OwnedInput<O, C>: RunShape<ResolvedCodec<C>, D>,
    RunCodec<O, C, D>:
        Codec + EncodeRun<OwnedInput<O, C>> + EncodeRun<NativeOrdinaryRecord<RunCodec<O, C, D>>>,
    CS: ChunkSorter<OwnedInput<O, C>, O>,
{
    /// Build an item-limited sorter using the complete unkeyed order.
    #[must_use]
    pub fn build(self) -> NativeUnkeyedBuilt<O, C, CS, spillover::sorter::ItemLimit, D> {
        let codec = self.codec.resolve();
        let framed = OwnedInput::<O, C>::frame(codec);
        self.core.codec(framed).build()
    }
}

impl<O, C, CS, D> Builder<OrderedCore<O, ByteLimit, CS, D>, C, OwnedStorage, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: CompatibleWith<ResolvedCodec<C>>
        + Compare<OwnedInput<O, C>>
        + Compare<NativeOrdinaryRecord<RunCodec<O, C, D>>>,
    OwnedInput<O, C>: RunShape<ResolvedCodec<C>, D> + GetSize,
    RunCodec<O, C, D>:
        Codec + EncodeRun<OwnedInput<O, C>> + EncodeRun<NativeOrdinaryRecord<RunCodec<O, C, D>>>,
    CS: ChunkSorter<OwnedInput<O, C>, O> + spillover::chunk::EstimateSortMemory<OwnedInput<O, C>>,
{
    /// Build an owned unkeyed sorter under a measured byte budget.
    #[must_use]
    pub fn build(self) -> NativeUnkeyedBuilt<O, C, CS, spillover::sorter::ByteLimit, D> {
        let codec = self.codec.resolve();
        let framed = OwnedInput::<O, C>::frame(codec);
        self.core.codec(framed).build()
    }
}

type NativeArenaUnkeyed<'a, O, C, CS, D> = spillover::sorter::Sorter<
    (),
    O,
    C,
    (),
    D,
    CS,
    spillover::sorter::Unkeyed,
    spillover::sorter::ArenaBuffer<'a, SeqRecordArena>,
>;
type NativeArenaBuilt<'a, O, C, CS, D> = NativeArenaUnkeyed<'a, O, ResolvedCodec<C>, CS, D>;

impl<'a, O, C, CS, D> Builder<OrderedCore<O, ItemLimit, CS, D>, C, ArenaStorage<'a>, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: CompatibleWith<ResolvedCodec<C>, Input = SeqRecord>
        + Compare<NativeOrdinaryRecord<ResolvedCodec<C>>>,
    for<'v> O: Compare<SeqRecordView<'v>>,
    for<'v> CS: ChunkSorter<SeqRecordView<'v>, O>,
{
    /// Build an item-limited sorter over the selected arena.
    #[must_use]
    pub fn build(self) -> NativeArenaBuilt<'a, O, C, CS, D> {
        let codec = self.codec.resolve();
        let storage = self.storage.arena;
        let builder = self.core.codec(codec);
        let build_in: fn(_, &'a mut SeqRecordArena) -> _ = spillover::sorter::Builder::build_in;
        build_in(builder, storage)
    }
}

type NativeMeasuredArenaUnkeyed<'a, O, C, CS, D> = spillover::sorter::Sorter<
    (),
    O,
    C,
    (),
    D,
    CS,
    spillover::sorter::Unkeyed,
    spillover::sorter::MeasuredArena<'a, SeqRecordArena>,
>;
type NativeMeasuredArenaBuilt<'a, O, C, CS, D> =
    NativeMeasuredArenaUnkeyed<'a, O, ResolvedCodec<C>, CS, D>;

impl<'a, O, C, CS, D> Builder<OrderedCore<O, ByteLimit, CS, D>, C, ArenaStorage<'a>, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: CompatibleWith<ResolvedCodec<C>, Input = SeqRecord>
        + Compare<NativeOrdinaryRecord<ResolvedCodec<C>>>,
    for<'v> O: Compare<SeqRecordView<'v>>,
    for<'v> CS:
        ChunkSorter<SeqRecordView<'v>, O> + spillover::chunk::EstimateSortMemory<SeqRecordView<'v>>,
{
    /// Build a measured sorter over the selected arena.
    #[must_use]
    pub fn build(self) -> NativeMeasuredArenaBuilt<'a, O, C, CS, D> {
        let codec = self.codec.resolve();
        let storage = self.storage.arena;
        let builder = self.core.codec(codec);
        let build_measured_in: fn(_, &'a mut SeqRecordArena) -> _ =
            spillover::sorter::Builder::build_measured_in;
        build_measured_in(builder, storage)
    }
}

type NativeArenaKeyed<'a, O, C, CS, D> = spillover::sorter::Sorter<
    (),
    O,
    C,
    (),
    D,
    CS,
    spillover::sorter::Keyed,
    spillover::sorter::ArenaBuffer<'a, SeqRecordArena>,
>;

impl<'a, O, C, CS, D> Builder<OrderedCore<O, ItemLimit, CS, D>, C, ArenaStorage<'a>, CoreKeyed>
where
    O: CompatibleWith<ResolvedCodec<C>, Input = SeqRecord>
        + KeyedOrder
        + Compare<NativeRecord<NativeKeyedCodec<O, C>>>,
    O::StoredKey: RecordKey + Clone + 'static,
    C: sealed::ResolveCodec,
    for<'v> O: DeriveKey<SeqRecordView<'v>> + Compare<SeqRecordView<'v>>,
    for<'v, 'order> CS: ChunkSorter<
            spillover::key::Keyed<usize, O::StoredKey>,
            spillover::key::KeyedEntryCompare<'order, SeqRecordView<'v>, O>,
        >,
{
    /// Build an item-limited keyed sorter over the selected arena.
    #[must_use]
    pub fn build(self) -> NativeArenaKeyed<'a, O, NativeKeyedCodec<O, C>, CS, D> {
        let codec = self.codec.resolve();
        let keyed_codec: NativeKeyedCodec<O, C> = codec.with_record_key();
        let storage = self.storage.arena;
        let builder = self.core.keyed_codec(keyed_codec);
        let build_keyed_in: fn(_, &'a mut SeqRecordArena) -> _ =
            spillover::sorter::Builder::build_keyed_in;
        build_keyed_in(builder, storage)
    }
}

type NativeMeasuredArenaKeyed<'a, O, C, CS, D> = spillover::sorter::Sorter<
    (),
    O,
    C,
    (),
    D,
    CS,
    spillover::sorter::Keyed,
    spillover::sorter::MeasuredArena<'a, SeqRecordArena>,
>;

impl<'a, O, C, CS, D> Builder<OrderedCore<O, ByteLimit, CS, D>, C, ArenaStorage<'a>, CoreKeyed>
where
    O: CompatibleWith<ResolvedCodec<C>, Input = SeqRecord>
        + KeyedOrder
        + Compare<NativeRecord<NativeKeyedCodec<O, C>>>,
    O::StoredKey: RecordKey + Clone + 'static,
    C: sealed::ResolveCodec,
    CS: spillover::chunk::EstimateSortMemory<spillover::key::Keyed<usize, O::StoredKey>>,
    for<'v> O: DeriveKey<SeqRecordView<'v>>
        + spillover::key::EstimateKeyMemory<SeqRecordView<'v>>
        + Compare<SeqRecordView<'v>>,
    for<'v, 'order> CS: ChunkSorter<
            spillover::key::Keyed<usize, O::StoredKey>,
            spillover::key::KeyedEntryCompare<'order, SeqRecordView<'v>, O>,
        >,
{
    /// Build a measured keyed sorter using the arena.
    #[must_use]
    pub fn build(self) -> NativeMeasuredArenaKeyed<'a, O, NativeKeyedCodec<O, C>, CS, D> {
        let codec = self.codec.resolve();
        let keyed_codec: NativeKeyedCodec<O, C> = codec.with_record_key();
        let storage = self.storage.arena;
        let builder = self.core.keyed_codec(keyed_codec);
        let build_keyed_measured_in: fn(_, &'a mut SeqRecordArena) -> _ =
            spillover::sorter::Builder::build_keyed_measured_in;
        build_keyed_measured_in(builder, storage)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use spillover::compare::{Compare, Natural};
    use spillover::key::SortKey;

    use super::*;

    fn make_record(name: &[u8], seq: &[u8], qual: &[u8]) -> SeqRecord {
        SeqRecord::new(name, seq, qual)
    }

    #[test]
    fn record_dedup_reuses_representative_capacity_on_update() {
        let first = make_record(b"long-name", b"ACGTACGT", b"!!!!!!!!");
        let second = make_record(b"x", b"C", b"#");
        let mut algorithm = RecordDedup::by_sequence();
        let representative = algorithm.capture(&first);
        <RecordDedup as AdjacentDeduplicator<SeqRecord>>::replace(&mut algorithm, representative);
        let original = algorithm.previous.as_ref().expect("representative");
        let capacity = original.bytes.capacity();
        let pointer = original.bytes.as_ptr();

        algorithm.update(&second);
        let updated = algorithm.previous.as_ref().expect("representative");
        assert_eq!(updated.bytes.capacity(), capacity);
        assert_eq!(updated.bytes.as_ptr(), pointer);
        assert_eq!(updated.name(), b"x");
        assert_eq!(updated.sequence(), b"C");
        assert_eq!(updated.quality(), b"#");
    }

    #[test]
    fn record_dedup_updates_empty_state_in_every_mode() {
        let record = make_record(b"name", b"AC", b"!#");
        for mut algorithm in [
            RecordDedup::by_name(),
            RecordDedup::by_sequence(),
            RecordDedup::by_sequence_and_quality(),
        ] {
            algorithm.update(&record);
            assert!(algorithm.matches(&record));
            let previous = algorithm.previous.as_ref().expect("representative");
            assert_eq!(previous.name(), record.name());
            assert_eq!(previous.sequence(), record.sequence());
            assert_eq!(previous.quality(), record.quality());
        }
    }

    #[test]
    fn record_representative_updates_long_short_long_and_empty_fields() {
        let records = [
            make_record(b"long-name", b"ACGTACGT", b"!!!!!!!!"),
            make_record(b"", b"C", b"#"),
            make_record(b"another-long-name", b"TTTTTTTTTT", b"##########"),
            make_record(b"", b"", b""),
            make_record(b"last", b"G", b"!"),
        ];
        let mut algorithm = RecordDedup::by_sequence();
        for record in &records {
            algorithm.update(record);
            let previous = algorithm.previous.as_ref().expect("representative");
            assert_eq!(previous.name(), record.name());
            assert_eq!(previous.sequence(), record.sequence());
            assert_eq!(previous.quality(), record.quality());
            assert!(algorithm.matches(record));
        }
    }

    // ── Sort key tests ───────────────────────────────────

    #[test]
    fn sequence_quality_key_extracts_tuple() {
        let rec = make_record(b"r1", b"ACGT", b"!!!!");
        let (seq, qual) = SequenceQualityKey.key(&rec);
        assert_eq!(seq, b"ACGT");
        assert_eq!(qual, b"!!!!");
    }

    #[test]
    fn sequence_quality_key_extracts_from_record_views() {
        let rec = make_record(b"r1", b"ACGT", b"!!!!");
        let view = rec.as_view();

        let (seq, qual) = SequenceQualityKey.key(&view);

        assert_eq!(seq, b"ACGT");
        assert_eq!(qual, b"!!!!");
    }

    #[test]
    fn sequence_quality_key_sorts_by_sequence_first() {
        let a = make_record(b"r1", b"AAAA", b"IIII");
        let b = make_record(b"r2", b"CCCC", b"!!!!");

        let cmp = SequenceQualityKey.item_cmp(&Natural);
        assert_eq!(
            cmp(&a, &b),
            Ordering::Less,
            "AAAA < CCCC regardless of quality"
        );
    }

    #[test]
    fn sequence_quality_key_tiebreaks_by_quality() {
        let a = make_record(b"r1", b"ACGT", b"IIII");
        let b = make_record(b"r2", b"ACGT", b"!!!!");

        let cmp = SequenceQualityKey.item_cmp(&Natural);
        assert_eq!(
            cmp(&a, &b),
            Ordering::Greater,
            "same sequence, IIII > !!!! in ASCII"
        );
    }

    #[test]
    fn sequence_quality_key_equal_when_both_match() {
        let a = make_record(b"r1", b"ACGT", b"!!!!");
        let b = make_record(b"r2", b"ACGT", b"!!!!");

        let cmp = SequenceQualityKey.item_cmp(&Natural);
        assert_eq!(
            cmp(&a, &b),
            Ordering::Equal,
            "same sequence and quality should be equal regardless of name"
        );
    }

    #[test]
    fn sequence_quality_key_can_sort_records() {
        let mut records = [
            make_record(b"r1", b"ACGT", b"!!!!"),
            make_record(b"r2", b"ACGT", b"IIII"),
            make_record(b"r3", b"AAAA", b"!!!!"),
        ];

        let cmp = SequenceQualityKey.item_cmp(&Natural);
        records.sort_by(cmp);

        assert_eq!(records[0].sequence(), b"AAAA");
        assert_eq!(records[1].quality(), b"!!!!");
        assert_eq!(records[2].quality(), b"IIII");
    }

    #[test]
    fn name_key_extracts_name() {
        let rec = make_record(b"read_001", b"ACGT", b"!!!!");
        assert_eq!(NameKey.key(&rec), b"read_001");
    }

    #[test]
    fn name_key_extracts_from_record_views() {
        let rec = make_record(b"read_001", b"ACGT", b"!!!!");
        assert_eq!(NameKey.key(&rec.as_view()), b"read_001");
    }

    #[test]
    fn name_key_sorts_lexicographically() {
        let a = make_record(b"aaa", b"ACGT", b"!!!!");
        let b = make_record(b"bbb", b"ACGT", b"!!!!");

        let cmp = NameKey.item_cmp(&Natural);
        assert_eq!(cmp(&a, &b), Ordering::Less);
    }

    #[test]
    fn length_key_extracts_length() {
        let rec = make_record(b"r1", b"ACGTACGT", b"!!!!!!!!");
        assert_eq!(LengthKey.key(&rec), 8);
    }

    #[test]
    fn length_key_extracts_from_record_views() {
        let rec = make_record(b"r1", b"ACGTACGT", b"!!!!!!!!");
        assert_eq!(LengthKey.key(&rec.as_view()), 8);
    }

    #[test]
    fn length_key_empty_sequence() {
        let rec = make_record(b"r1", b"", b"");
        assert_eq!(LengthKey.key(&rec), 0);
    }

    #[test]
    fn length_key_sorts_by_length() {
        let short = make_record(b"r1", b"AC", b"!!");
        let long = make_record(b"r2", b"ACGTACGT", b"!!!!!!!!");

        let cmp = LengthKey.item_cmp(&Natural);
        assert_eq!(cmp(&short, &long), Ordering::Less);
    }

    #[test]
    fn sort_keys_are_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<SequenceQualityKey>();
        assert_copy::<NameKey>();
        assert_copy::<LengthKey>();
    }

    #[test]
    fn builder_accepts_arena_storage_mode() {
        let mut arena = SeqRecordArena::new();

        let _builder = Builder::new().sort_by_illumina().arena(&mut arena);
    }

    // ── Sort order tests ─────────────────────────────────

    #[test]
    fn illumina_order_produces_correct_key() {
        let rec = make_record(b"r1", b"ACGT", b"!!!!");
        let key = ILLUMINA_ORDER.derive_key(&rec);
        let expected = crate::key::PackedSequenceKey::<38>::from_sequence(b"ACGT");
        assert_eq!(key, expected);
    }

    #[test]
    fn sort_by_sequence_exact_selects_unkeyed_sequence_order() {
        fn assert_builder_type<T>(_: &T) {}

        let builder = Builder::new().sort_by_sequence_exact();

        assert_builder_type::<
            Builder<
                OrderedCore<UnkeyedSequenceOrder, ByteLimit, Sequential, NoDedup>,
                NeedsCodec,
                OwnedStorage,
                CoreUnkeyed,
            >,
        >(&builder);
    }

    #[test]
    fn sort_by_sequence_exact_overrides_chunk_sorter_to_sequential() {
        fn assert_builder_type<T>(_: &T) {}

        let builder = Builder::new()
            .chunk_sort(RadixThenRefine::<38>)
            .sort_by_sequence_exact();

        assert_builder_type::<
            Builder<
                OrderedCore<UnkeyedSequenceOrder, ByteLimit, Sequential, NoDedup>,
                NeedsCodec,
                OwnedStorage,
                CoreUnkeyed,
            >,
        >(&builder);
    }

    #[test]
    fn reverse_sequence_order_uses_sequential_by_default() {
        fn assert_builder_type<T>(_: &T) {}

        let builder = Builder::new().sort_by(Reverse(ILLUMINA_ORDER));

        assert_builder_type::<
            Builder<
                OrderedCore<Reverse<SequenceOrder<38>>, ByteLimit, Sequential, NoDedup>,
                NeedsCodec,
                OwnedStorage,
                CoreKeyed,
            >,
        >(&builder);
    }

    #[test]
    fn compatible_chunk_sorters_build() {
        let _illumina = Builder::new()
            .sort_by_illumina()
            .codec(DryIceCodec::new())
            .max_buffer_items(1)
            .build();

        let _exact = Builder::new()
            .sort_by_sequence_exact()
            .codec(DryIceCodec::new())
            .max_buffer_items(1)
            .build();

        let _reverse = Builder::new()
            .sort_by(Reverse(ILLUMINA_ORDER))
            .codec(DryIceCodec::new())
            .max_buffer_items(1)
            .build();
    }

    #[test]
    fn name_order_produces_correct_key() {
        let rec = make_record(b"read_001", b"ACGT", b"!!!!");
        let key = NameOrder.derive_key(&rec);
        let mut expected = [0u8; 16];
        expected[..8].copy_from_slice(b"read_001");
        assert_eq!(key, dryice::Bytes16Key(expected));
    }

    #[test]
    fn name_order_truncates_long_names() {
        let rec = make_record(b"a_very_long_name_that_exceeds_16_bytes", b"ACGT", b"!!!!");
        let key = NameOrder.derive_key(&rec);
        assert_eq!(&key.0, b"a_very_long_name");
    }

    #[test]
    fn length_order_produces_correct_key() {
        let rec = make_record(b"r1", b"ACGTACGT", b"!!!!!!!!");
        let key = LengthOrder.derive_key(&rec);
        assert_eq!(key, dryice::Bytes8Key(8u64.to_be_bytes()));
    }

    #[test]
    fn length_order_big_endian_preserves_ord() {
        let short = make_record(b"r1", b"AC", b"!!");
        let long = make_record(b"r2", b"ACGTACGT", b"!!!!!!!!");
        let key_short = LengthOrder.derive_key(&short);
        let key_long = LengthOrder.derive_key(&long);
        assert!(
            key_short < key_long,
            "big-endian u64 should preserve ordering"
        );
    }

    #[test]
    fn sequence_order_refines_stored_key_ties_with_quality() {
        let order = ILLUMINA_ORDER;

        let a = make_record(b"r1", b"ACGT", b"IIII");
        let b = make_record(b"r2", b"ACGT", b"!!!!");

        let ka = order.derive_key(&a);
        let kb = order.derive_key(&b);

        assert_eq!(order.compare_keys(&ka, &kb), Ordering::Equal);
        assert_eq!(
            order.compare(&a, &b),
            Ordering::Greater,
            "same sequence, IIII > !!!! under the complete order"
        );
    }

    #[test]
    fn reverse_flips_key_ordering() {
        let order = Reverse(ILLUMINA_ORDER);

        let a = make_record(b"r1", b"AAAA", b"!!!!");
        let b = make_record(b"r2", b"TTTT", b"!!!!");

        let ka = order.derive_key(&a);
        let kb = order.derive_key(&b);

        assert_eq!(
            order.compare_keys(&ka, &kb),
            Ordering::Greater,
            "reversed: AAAA key should be greater than TTTT key"
        );
    }

    #[test]
    fn reverse_preserves_record_key() {
        let rec = make_record(b"r1", b"ACGT", b"!!!!");
        let forward_key = ILLUMINA_ORDER.derive_key(&rec);
        let reverse_key = Reverse(ILLUMINA_ORDER).derive_key(&rec);
        assert_eq!(
            forward_key, reverse_key,
            "reverse should not change the record key"
        );
    }

    #[test]
    fn exact_sequence_order_requires_raw_compared_fields() {
        let codec = DryIceCodec::new().omit_names();
        let _forward = Builder::new()
            .sort_by_unkeyed(UnkeyedSequenceOrder)
            .codec(codec)
            .build();
        let _reverse = Builder::new()
            .sort_by_unkeyed(Reverse(UnkeyedSequenceOrder))
            .codec(codec)
            .build();
        let _nested_reverse = Builder::new()
            .sort_by_unkeyed(Reverse(Reverse(UnkeyedSequenceOrder)))
            .codec(codec)
            .build();
    }

    #[test]
    fn accelerated_sequence_order_retains_codec_choices() {
        let codec = DryIceCodec::new()
            .two_bit_lossy_n()
            .binned_quality()
            .omit_names();
        let _forward = Builder::new().sort_by(ILLUMINA_ORDER).codec(codec).build();
        let _reverse = Builder::new()
            .sort_by(Reverse(ILLUMINA_ORDER))
            .codec(codec)
            .build();
    }

    #[test]
    fn all_orders_are_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<SequenceOrder<38>>();
        assert_copy::<SequenceOrder<64>>();
        assert_copy::<SequenceOrder<128>>();
        assert_copy::<NameOrder>();
        assert_copy::<LengthOrder>();
        assert_copy::<UnkeyedSequenceOrder>();
        assert_copy::<UnkeyedNameOrder>();
        assert_copy::<UnkeyedLengthOrder>();
        assert_copy::<Reverse<SequenceOrder<38>>>();
    }

    #[test]
    fn unkeyed_from_keyed() {
        let _: UnkeyedSequenceOrder = ILLUMINA_ORDER.unkeyed();
        let _: UnkeyedNameOrder = NameOrder.unkeyed();
        let _: UnkeyedLengthOrder = LengthOrder.unkeyed();
    }
}
