//! Genomics-specific sort keys, sort orders, and the sorter builder.
//!
//! Sort keys extract the value to sort by from record-shaped values.
//! Sort orders bundle a sort key with a compatible dryice record
//! key and a merge strategy, preventing invalid combinations.
//!
//! The primary sort strategy for genomic data is sequence-first
//! with quality as the tiebreaker, expressed as a tuple key
//! `(&[u8], &[u8])` whose [`Ord`] implementation compares
//! sequence first, then quality lexicographically.
//!
//! The [`Builder`] is the main entry point for creating a sorter.

use std::{fs::File, io::BufReader, marker::PhantomData};

use dryice::{
    DryIceWriter, NameCodec, QualityCodec, RawAsciiCodec, RawNameCodec, RawQualityCodec, RecordKey,
    SequenceCodec,
};
use spillover::{
    SortedItemsError,
    chunk::{ChunkSorter, Sequential},
    codec::{Codec, CodecCursor, EncodeKeyedRun, EncodeRun, KeyedCodec},
    compare::{Compare, Natural},
    dedup::Identity,
    key::{KeyCompare, SortKey},
    merge::{KeyedRunMerge, MergeConfig, MergeError, RunMerge},
    sorter::{
        ByteLimit, HasOrder, ItemLimit, Keyed as CoreKeyed, NeedsLimit,
        NeedsOrder as CoreNeedsOrder, Unkeyed as CoreUnkeyed, VisitSortedItems,
    },
};

use crate::{
    codec::{DryIceCodec, KeyedDryIceCodec},
    error::SortedRecordStreamError,
    key::PackedSequenceKey,
    radix::RadixThenRefine,
    record::{SeqRecord, SeqRecordArena, SeqRecordParts, SeqRecordView},
};

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

// ── Sort order traits (sealed) ────────────────────────────

mod sealed {
    pub trait Sealed {}

    pub trait CompatibleChunkSorter<O> {}

    /// Resolves codec type-state to a concrete [`crate::codec::DryIceCodec`].
    /// `NeedsCodec` yields the raw (no-encoding) default;
    /// `HasCodec` passes through the user's choice.
    pub trait ResolveCodec {
        type S: dryice::SequenceCodec + Copy + 'static;
        type Q: dryice::QualityCodec + Copy + 'static;
        type N: dryice::NameCodec + Copy + 'static;

        fn resolve(self) -> crate::codec::DryIceCodec<Self::S, Self::Q, Self::N>;
    }
}

/// Defines a complete sort order: what to sort by, how to
/// compare, and what merge strategy to use. Sealed — only
/// spillover-bio's built-in orders implement this.
pub trait SortOrder: sealed::Sealed + Copy {
    /// The sort key for extracting the comparison value.
    type SortKey: SortKey<SeqRecord> + Copy + Send + Sync + 'static;

    /// Comparator for record-shaped values under this order.
    ///
    /// This is the comparator callers should use when they already have records
    /// or record views in memory.
    type Comparator: Copy + Send + Sync + 'static;

    /// Comparator for extracted sort keys and compact merge keys.
    ///
    /// This is the lower-level comparator used by the generic core sorter and
    /// keyed merge path after records have been encoded into sorted runs.
    type KeyComparator: Copy + Send + Sync + 'static;

    /// The merge strategy marker: [`Unkeyed`] or [`Keyed`].
    type Strategy;

    /// The sort key extractor.
    fn sort_key(&self) -> Self::SortKey;

    /// The record comparator.
    fn comparator(&self) -> Self::Comparator;

    /// The key comparator.
    fn key_comparator(&self) -> Self::KeyComparator;
}

/// Derives dryice record keys from record-shaped values.
pub trait RecordKeyer: Copy {
    /// The dryice record key type.
    type RecordKey: dryice::RecordKey + Clone + Copy;

    /// Derive a record key from any record-shaped value.
    fn record_key<R: SeqRecordParts + ?Sized>(&self, record: &R) -> Self::RecordKey;
}

/// Extension for sort orders that use record keys for merge
/// acceleration. Only available when `Strategy = Keyed`.
pub trait KeyedSortOrder: SortOrder<Strategy = Keyed> + RecordKeyer {}

impl<O> KeyedSortOrder for O where O: SortOrder<Strategy = Keyed> + RecordKeyer {}

impl<O> sealed::CompatibleChunkSorter<O> for Sequential {}

#[cfg(feature = "rayon")]
impl<O> sealed::CompatibleChunkSorter<O> for spillover::chunk::Parallel {}

impl<const N: usize> sealed::CompatibleChunkSorter<SequenceOrder<N>> for RadixThenRefine<N> {}

/// Marker: unkeyed merge path (no record keys).
pub struct Unkeyed;

/// Marker: keyed merge path (record keys for merge acceleration).
pub struct Keyed;

// ── Keyed sort orders ────────────────────────────────────

/// Sort by nucleotide sequence with quality tiebreaker,
/// using a 2-bit packed key of width N bytes (N×4 bases).
///
/// For most users, the convenience aliases are easier:
/// [`ILLUMINA_ORDER`], [`PAIRED_END_ORDER`], [`LONG_READ_ORDER`].
#[derive(Debug, Clone, Copy)]
pub struct SequenceOrder<const N: usize>;

impl<const N: usize> sealed::Sealed for SequenceOrder<N> {}

impl<const N: usize> SortOrder for SequenceOrder<N> {
    type SortKey = SequenceQualityKey;
    type Comparator = Self;
    type KeyComparator = Natural;
    type Strategy = Keyed;

    fn sort_key(&self) -> SequenceQualityKey {
        SequenceQualityKey
    }

    fn comparator(&self) -> Self {
        *self
    }

    fn key_comparator(&self) -> Natural {
        Natural
    }
}

impl<const N: usize, R: SeqRecordParts + ?Sized> Compare<R> for SequenceOrder<N> {
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        PackedSequenceKey::<N>::from_sequence(a.sequence())
            .cmp(&PackedSequenceKey::<N>::from_sequence(b.sequence()))
            .then_with(|| a.sequence().cmp(b.sequence()))
            .then_with(|| a.quality().cmp(b.quality()))
    }
}

impl<const N: usize> RecordKeyer for SequenceOrder<N> {
    type RecordKey = PackedSequenceKey<N>;

    fn record_key<R: SeqRecordParts + ?Sized>(&self, record: &R) -> PackedSequenceKey<N> {
        PackedSequenceKey::from_sequence(record.sequence())
    }
}

impl<const N: usize> SequenceOrder<N> {
    /// Opt out of record key acceleration, using the base
    /// merge path with full record deserialization.
    #[must_use]
    pub fn unkeyed(self) -> UnkeyedSequenceOrder {
        UnkeyedSequenceOrder
    }
}

/// Sort by record name, using a 16-byte name prefix as the
/// record key.
#[derive(Debug, Clone, Copy)]
pub struct NameOrder;

impl sealed::Sealed for NameOrder {}

impl SortOrder for NameOrder {
    type SortKey = NameKey;
    type Comparator = Self;
    type KeyComparator = Natural;
    type Strategy = Keyed;

    fn sort_key(&self) -> NameKey {
        NameKey
    }

    fn comparator(&self) -> Self {
        *self
    }

    fn key_comparator(&self) -> Natural {
        Natural
    }
}

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

impl RecordKeyer for NameOrder {
    type RecordKey = dryice::Bytes16Key;

    fn record_key<R: SeqRecordParts + ?Sized>(&self, record: &R) -> dryice::Bytes16Key {
        derive_name_key(record)
    }
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

impl sealed::Sealed for LengthOrder {}

impl SortOrder for LengthOrder {
    type SortKey = LengthKey;
    type Comparator = Self;
    type KeyComparator = Natural;
    type Strategy = Keyed;

    fn sort_key(&self) -> LengthKey {
        LengthKey
    }

    fn comparator(&self) -> Self {
        *self
    }

    fn key_comparator(&self) -> Natural {
        Natural
    }
}

impl<R: SeqRecordParts + ?Sized> Compare<R> for LengthOrder {
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        a.sequence().len().cmp(&b.sequence().len())
    }
}

#[allow(clippy::cast_possible_truncation)]
fn derive_length_key<R: SeqRecordParts + ?Sized>(record: &R) -> dryice::Bytes8Key {
    dryice::Bytes8Key((record.sequence().len() as u64).to_be_bytes())
}

impl RecordKeyer for LengthOrder {
    type RecordKey = dryice::Bytes8Key;

    fn record_key<R: SeqRecordParts + ?Sized>(&self, record: &R) -> dryice::Bytes8Key {
        derive_length_key(record)
    }
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

impl sealed::Sealed for UnkeyedSequenceOrder {}

impl SortOrder for UnkeyedSequenceOrder {
    type SortKey = SequenceQualityKey;
    type Comparator = Self;
    type KeyComparator = Natural;
    type Strategy = Unkeyed;

    fn sort_key(&self) -> SequenceQualityKey {
        SequenceQualityKey
    }

    fn comparator(&self) -> Self {
        *self
    }

    fn key_comparator(&self) -> Natural {
        Natural
    }
}

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

impl sealed::Sealed for UnkeyedNameOrder {}

impl SortOrder for UnkeyedNameOrder {
    type SortKey = NameKey;
    type Comparator = Self;
    type KeyComparator = Natural;
    type Strategy = Unkeyed;

    fn sort_key(&self) -> NameKey {
        NameKey
    }

    fn comparator(&self) -> Self {
        *self
    }

    fn key_comparator(&self) -> Natural {
        Natural
    }
}

impl<R: SeqRecordParts + ?Sized> Compare<R> for UnkeyedNameOrder {
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        a.name().cmp(b.name())
    }
}

/// Sort by length without record key acceleration.
#[derive(Debug, Clone, Copy)]
pub struct UnkeyedLengthOrder;

impl sealed::Sealed for UnkeyedLengthOrder {}

impl SortOrder for UnkeyedLengthOrder {
    type SortKey = LengthKey;
    type Comparator = Self;
    type KeyComparator = Natural;
    type Strategy = Unkeyed;

    fn sort_key(&self) -> LengthKey {
        LengthKey
    }

    fn comparator(&self) -> Self {
        *self
    }

    fn key_comparator(&self) -> Natural {
        Natural
    }
}

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
#[derive(Debug, Clone, Copy)]
pub struct Reverse<O>(pub O);

impl<O: sealed::Sealed> sealed::Sealed for Reverse<O> {}

impl<O: SortOrder + Send + Sync + 'static> SortOrder for Reverse<O> {
    type SortKey = O::SortKey;
    type Comparator = Self;
    type KeyComparator = spillover::compare::Reverse<O::KeyComparator>;
    type Strategy = O::Strategy;

    fn sort_key(&self) -> Self::SortKey {
        self.0.sort_key()
    }

    fn comparator(&self) -> Self {
        *self
    }

    fn key_comparator(&self) -> spillover::compare::Reverse<O::KeyComparator> {
        spillover::compare::Reverse(self.0.key_comparator())
    }
}

impl<O, R> Compare<R> for Reverse<O>
where
    O: SortOrder + Compare<R>,
    R: SeqRecordParts + ?Sized,
{
    fn compare(&self, a: &R, b: &R) -> std::cmp::Ordering {
        self.0.compare(a, b).reverse()
    }
}

impl<O: RecordKeyer> RecordKeyer for Reverse<O> {
    type RecordKey = O::RecordKey;

    fn record_key<R: SeqRecordParts + ?Sized>(&self, record: &R) -> Self::RecordKey {
        self.0.record_key(record)
    }
}

impl<O: spillover::key::KeyedOrder> spillover::key::KeyedOrder for Reverse<O> {
    type StoredKey = O::StoredKey;
    fn compare_keys(&self, left: &Self::StoredKey, right: &Self::StoredKey) -> std::cmp::Ordering {
        self.0.compare_keys(left, right).reverse()
    }
}
impl<R: ?Sized, O: spillover::key::DeriveKey<R>> spillover::key::DeriveKey<R> for Reverse<O> {
    fn derive_key(&self, item: &R) -> Self::StoredKey {
        self.0.derive_key(item)
    }
}
impl<R: ?Sized, O: spillover::key::EstimateKeyMemory<R>> spillover::key::EstimateKeyMemory<R>
    for Reverse<O>
{
    fn estimate_key_memory(&self, item: &R) -> spillover::key::KeyMemory {
        self.0.estimate_key_memory(item)
    }
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

#[derive(Debug, Clone, Copy)]
enum RecordDedup {
    None,
    Name,
    Sequence,
    SequenceAndQuality,
}

impl RecordDedup {
    #[inline]
    fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    #[inline]
    fn is_duplicate<R: SeqRecordParts + ?Sized>(
        self,
        previous: Option<&PreviousRecord>,
        current: &R,
    ) -> bool {
        previous.is_some_and(|previous| self.matches(previous, current))
    }

    #[inline]
    fn matches<R: SeqRecordParts + ?Sized>(self, previous: &PreviousRecord, current: &R) -> bool {
        match self {
            Self::None => false,
            Self::Name => previous.has_same_name_as(current),
            Self::Sequence => previous.has_same_sequence_as(current),
            Self::SequenceAndQuality => previous.has_same_sequence_and_quality_as(current),
        }
    }
}

#[derive(Debug, Default)]
struct PreviousRecord {
    bytes: Vec<u8>,
    name_len: usize,
    sequence_len: usize,
}

impl PreviousRecord {
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

impl SeqRecordParts for PreviousRecord {
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

struct RecordDedupState {
    mode: RecordDedup,
    previous: Option<PreviousRecord>,
}

impl RecordDedupState {
    fn new(mode: RecordDedup) -> Self {
        Self {
            mode,
            previous: None,
        }
    }

    fn is_enabled(&self) -> bool {
        self.mode.is_enabled()
    }

    fn accept<R: SeqRecordParts + ?Sized>(&mut self, record: &R) -> bool {
        if !self.is_enabled() {
            return true;
        }
        if self.mode.is_duplicate(self.previous.as_ref(), record) {
            return false;
        }
        let previous = self.previous.get_or_insert_with(PreviousRecord::default);
        previous.copy_from(record);
        true
    }

    fn write<R, W>(&mut self, record: &R, writer: &mut W) -> Result<(), W::Error>
    where
        R: SeqRecordParts + ?Sized,
        W: SeqRecordSink,
    {
        if !self.accept(record) {
            return Ok(());
        }
        match &self.previous {
            Some(previous) => writer.write_record(previous),
            None => writer.write_record(record),
        }
    }
}

/// Legacy marker for a builder with no sort order.
///
/// Retained as a public name; the current [`Builder`] does not use it.
pub struct NeedsOrder;
/// Legacy marker for a builder with a keyed sort order.
///
/// Retained as a public name; the current [`Builder`] does not use it.
pub struct HasKeyedOrder<O>(O);
/// Legacy marker for a builder with an unkeyed sort order.
///
/// Retained as a public name; the current [`Builder`] does not use it.
pub struct HasUnkeyedOrder<O>(O);

/// Builder marker: no codec chosen yet.
pub struct NeedsCodec;
/// Builder marker: codec has been chosen.
pub struct HasCodec<S, Q, N>(DryIceCodec<S, Q, N>);

/// Legacy marker for a builder with no flush strategy.
///
/// Retained as a public name; the current [`Builder`] does not use it.
pub struct NeedsFlush;
/// Legacy marker for a builder with a flush strategy.
///
/// Retained as a public name; the current [`Builder`] does not use it.
pub struct HasFlush<L = spillover::sorter::ByteLimit>(L);

/// Builder marker: use the default owned-record buffer.
pub struct OwnedStorage;

/// Builder marker: use caller-provided arena storage for the spill window.
pub struct ArenaStorage<'a> {
    arena: &'a mut SeqRecordArena,
}

// Retained for public compatibility.
#[doc(hidden)]
pub enum FlushConfig {
    MeasuredBudget(usize),
    MaxItems(usize),
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

/// Builder for configuring a genomics-oriented external sorter.
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
    dedup: RecordDedup,
    storage: S,
    mode: PhantomData<fn() -> M>,
}

type CoreBuilder<SK, L, CS> =
    spillover::sorter::Builder<SK, spillover::sorter::NeedsCodec, L, (), Identity, CS>;
type OrderedCore<O, L, CS> = CoreBuilder<HasOrder<O>, L, CS>;
type InitialCore = CoreBuilder<CoreNeedsOrder, NeedsLimit, Sequential>;

/// Active genomics sorter.
///
/// Construct a sorter with [`Builder`], push unsorted records into it, then call
/// [`finish`](Self::finish) to produce a [`SortedRecordStream`].
pub struct Sorter<SK, Cod, Cmp, CS, M, B = OwnedBackend<SK, Cod, Cmp, CS, M>> {
    inner: B,
    dedup: RecordDedup,
    _types: std::marker::PhantomData<SorterTypeMarker<SK, Cod, Cmp, CS, M>>,
}

type SorterTypeMarker<SK, Cod, Cmp, CS, M> = fn() -> (SK, Cod, Cmp, CS, M);

/// Default owned-record backend for [`Sorter`].
#[doc(hidden)]
pub struct OwnedBackend<
    SK,
    Cod,
    Cmp,
    CS,
    M,
    Inner = spillover::sorter::Sorter<SeqRecord, SK, Cod, Cmp, Identity, CS, M>,
> {
    inner: Inner,
    _types: std::marker::PhantomData<SorterTypeMarker<SK, Cod, Cmp, CS, M>>,
}

/// Finalized stream of sorted sequence records.
///
/// `SortedRecordStream` owns the sorted output source after a [`Sorter`] has
/// been finished. It implements [`Iterator`] for owned materialization, so
/// ordinary `.collect()` callsites continue to work.
pub struct SortedRecordStream<I> {
    inner: I,
    dedup: RecordDedupState,
    fused: bool,
}

impl<I> SortedRecordStream<I> {
    fn new(inner: I, dedup: RecordDedupState) -> Self {
        Self {
            inner,
            dedup,
            fused: false,
        }
    }
}

/// Destination for sequence records produced by a sorted stream.
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

struct SeqRecordPartsAdapter<'a, R: ?Sized>(&'a R);

impl<R: SeqRecordParts + ?Sized> dryice::SeqRecordLike for SeqRecordPartsAdapter<'_, R> {
    fn name(&self) -> &[u8] {
        self.0.name()
    }

    fn sequence(&self) -> &[u8] {
        self.0.sequence()
    }

    fn quality(&self) -> &[u8] {
        self.0.quality()
    }
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

impl<I, E> Iterator for SortedRecordStream<I>
where
    I: Iterator<Item = Result<SeqRecord, E>>,
{
    type Item = Result<SeqRecord, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.dedup.is_enabled() {
            return self.inner.next();
        }

        if self.fused {
            return None;
        }

        loop {
            match self.inner.next() {
                Some(Ok(record)) => {
                    if self.dedup.accept(&record) {
                        return Some(Ok(record));
                    }
                }
                Some(Err(err)) => {
                    self.fused = true;
                    return Some(Err(err));
                }
                None => return None,
            }
        }
    }
}

impl<I, RE> SortedRecordStream<I>
where
    I: VisitSortedItems<Error = RE>,
    I::Record: SeqRecordParts,
    RE: std::error::Error + Send + Sync + 'static,
{
    /// Write all sorted records to a destination sink.
    ///
    /// This drains through the core current-item traversal seam. Keyed dryice
    /// sources can write borrowed current record views, while sources without a
    /// borrowed current representation may still materialize internally.
    ///
    /// # Errors
    ///
    /// Returns [`SortedRecordStreamError::Source`] if the sorted stream fails to
    /// produce a record, or [`SortedRecordStreamError::Sink`] if the destination
    /// sink fails to write a record.
    pub fn write_to<W>(self, writer: &mut W) -> Result<(), SortedRecordStreamError<RE, W::Error>>
    where
        W: SeqRecordSink,
    {
        let Self {
            inner,
            mut dedup,
            fused,
        } = self;

        if fused {
            return Ok(());
        }

        inner
            .visit_items(|record| dedup.write(record, writer))
            .map_err(|err| match err {
                SortedItemsError::Source(err) => SortedRecordStreamError::Source(err),
                SortedItemsError::Sink(err) => SortedRecordStreamError::Sink(err),
            })
    }
}

impl Builder {
    /// Start building a new sorter.
    #[must_use]
    pub fn new() -> Self {
        Builder {
            core: spillover::sorter::Builder::for_order(),
            codec: NeedsCodec,
            dedup: RecordDedup::None,
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

// These bounds intentionally mirror the core sorter's record and keyed
// push/finish implementations. The bio wrapper owns the public lifecycle
// vocabulary; core owns the sorting mechanics.

impl<SK, Cod, Cmp, CS>
    Sorter<
        SK,
        Cod,
        Cmp,
        CS,
        spillover::sorter::Unkeyed,
        OwnedBackend<SK, Cod, Cmp, CS, spillover::sorter::Unkeyed>,
    >
where
    SK: SortKey<SeqRecord> + Copy + Send + Sync + 'static,
    Cod: Codec + Copy + 'static,
    Cod: EncodeRun<SeqRecord>,
    Cod: EncodeRun<<<Cod as Codec>::Cursor<BufReader<File>> as CodecCursor>::Record>,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy + Send + Sync + 'static,
    KeyCompare<SK, Cmp>: Compare<<<Cod as Codec>::Cursor<BufReader<File>> as CodecCursor>::Record>,
    CS: ChunkSorter<SeqRecord, KeyCompare<SK, Cmp>>,
{
    /// Add an unsorted record to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing buffered records to disk fails.
    pub fn push(&mut self, record: impl Into<SeqRecord>) -> Result<(), MergeError<Cod::Error>> {
        self.inner.inner.push(record.into())
    }

    /// Finish sorting and return the finalized sorted record stream.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing, merging, or deduplicating output fails.
    #[allow(clippy::type_complexity)]
    pub fn finish(
        self,
    ) -> Result<
        SortedRecordStream<spillover::Sorted<RunMerge<Cod, KeyCompare<SK, Cmp>>>>,
        MergeError<Cod::Error>,
    > {
        let inner = self.inner.inner.finish()?;
        let dedup = RecordDedupState::new(self.dedup);
        Ok(SortedRecordStream::new(inner, dedup))
    }
}

impl<SK, Cod, Cmp, CS>
    Sorter<
        SK,
        Cod,
        Cmp,
        CS,
        spillover::sorter::Keyed,
        OwnedBackend<SK, Cod, Cmp, CS, spillover::sorter::Keyed>,
    >
where
    SK: SortKey<SeqRecord> + Copy + Send + Sync + 'static,
    Cod: KeyedCodec + EncodeKeyedRun<SeqRecord> + Copy + 'static,
    Cod: EncodeKeyedRun<<<Cod as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>,
    Cmp: for<'a> Compare<SK::Key<'a>> + Compare<Cod::Key> + Copy + Send + Sync + 'static,
    KeyCompare<SK, Cmp>:
        Compare<<<Cod as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>,
    CS: ChunkSorter<SeqRecord, KeyCompare<SK, Cmp>>,
{
    /// Add an unsorted record to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing buffered records to disk fails.
    pub fn push(&mut self, record: impl Into<SeqRecord>) -> Result<(), MergeError<Cod::Error>> {
        self.inner.inner.push(record.into())
    }

    /// Finish sorting and return the finalized sorted record stream.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing, merging, or deduplicating output fails.
    #[allow(clippy::type_complexity)]
    pub fn finish(
        self,
    ) -> Result<
        SortedRecordStream<spillover::Sorted<KeyedRunMerge<Cod, Cmp, KeyCompare<SK, Cmp>>>>,
        MergeError<Cod::Error>,
    > {
        let inner = self.inner.inner.finish()?;
        let dedup = RecordDedupState::new(self.dedup);
        Ok(SortedRecordStream::new(inner, dedup))
    }
}

macro_rules! native_keyed_methods {
    ($limit:ty; $($extra:tt)*) => {
        impl<O, C, CS> NativeSorter<O, C, CS, $limit>
        where
            C: KeyedCodec + EncodeKeyedRun<SeqRecord> + EncodeKeyedRun<NativeRecord<C>>,
            O: spillover::key::DeriveKey<SeqRecord, StoredKey = C::Key> + Compare<SeqRecord> + Compare<NativeRecord<C>>,
            for<'a> CS: ChunkSorter<spillover::key::Keyed<usize, C::Key>, spillover::key::KeyedEntryCompare<'a, SeqRecord, O>>,
            $($extra)*
        {
            /// Add an unsorted record.
            ///
            /// # Errors
            /// Returns an error if spilling fails.
            pub fn push(&mut self, record: impl Into<SeqRecord>) -> Result<(), MergeError<C::Error>> {
                self.inner.inner.push(record.into())
            }

            /// Finish sorting and return a sorted stream.
            ///
            /// # Errors
            /// Returns an error if spilling or merging fails.
            pub fn finish(self) -> Result<NativeOutput<C, O>, MergeError<C::Error>> {
                let inner = self.inner.inner.finish()?;
                let dedup = RecordDedupState::new(self.dedup);
                Ok(SortedRecordStream::new(inner, dedup))
            }
        }
    };
}

type NativeOutput<C, O> =
    SortedRecordStream<spillover::Sorted<KeyedRunMerge<C, O, (), spillover::sorter::Keyed>>>;

native_keyed_methods!(spillover::sorter::ItemLimit;);
native_keyed_methods!(spillover::sorter::ByteLimit;
    O: spillover::key::EstimateKeyMemory<SeqRecord>,
    CS: spillover::chunk::EstimateSortMemory<spillover::key::Keyed<usize, C::Key>>,
);

// Sort order selection.

impl<Core, C, M> Builder<Core, C, OwnedStorage, M> {
    /// Use caller-provided arena storage for the active spill window.
    #[must_use]
    pub fn arena(self, arena: &mut SeqRecordArena) -> Builder<Core, C, ArenaStorage<'_>, M> {
        Builder {
            codec: self.codec,
            dedup: self.dedup,
            core: self.core,
            storage: ArenaStorage { arena },
            mode: self.mode,
        }
    }
}

impl<C, L, CS, S> Builder<CoreBuilder<CoreNeedsOrder, L, CS>, C, S> {
    /// Set a keyed sort order (merge compares record keys).
    #[must_use]
    pub fn sort_by<O: KeyedSortOrder>(
        self,
        order: O,
    ) -> Builder<OrderedCore<O, L, CS>, C, S, CoreKeyed> {
        self.map_core(|core| core.select_order(order))
    }

    /// Set an unkeyed sort order (merge deserializes full records).
    #[must_use]
    pub fn sort_by_unkeyed<O: SortOrder<Strategy = Unkeyed>>(
        self,
        order: O,
    ) -> Builder<OrderedCore<O, L, CS>, C, S, CoreUnkeyed> {
        self.map_core(|core| core.select_order(order))
    }

    /// Sort by sequence (Illumina 150bp, 38-byte packed key).
    /// Defaults to radix sort for the chunk sorting engine.
    #[must_use]
    pub fn sort_by_illumina(
        self,
    ) -> Builder<OrderedCore<SequenceOrder<38>, L, RadixThenRefine<38>>, C, S, CoreKeyed> {
        self.sort_by(ILLUMINA_ORDER)
            .chunk_sort(RadixThenRefine::<38>)
    }

    /// Sort by exact raw sequence bytes, then raw quality bytes.
    ///
    /// This uses the unkeyed sequence order rather than the accelerated
    /// packed-key order used by [`Self::sort_by_illumina`]. It also selects the
    /// sequential comparator-based chunk sorter, even if a chunk sorter was
    /// configured earlier on the builder, because packed-key radix sorting is not
    /// compatible with exact raw sequence ordering.
    #[must_use]
    pub fn sort_by_sequence_exact(
        self,
    ) -> Builder<OrderedCore<UnkeyedSequenceOrder, L, Sequential>, C, S, CoreUnkeyed> {
        self.sort_by_unkeyed(UnkeyedSequenceOrder)
            .sort_with_sequential()
    }

    /// Sort by sequence (250bp paired-end, 64-byte packed key).
    /// Defaults to radix sort for the chunk sorting engine.
    #[must_use]
    pub fn sort_by_paired_end(
        self,
    ) -> Builder<OrderedCore<SequenceOrder<64>, L, RadixThenRefine<64>>, C, S, CoreKeyed> {
        self.sort_by(PAIRED_END_ORDER)
            .chunk_sort(RadixThenRefine::<64>)
    }

    /// Sort by sequence (long reads, 128-byte prefix key).
    /// Defaults to radix sort for the chunk sorting engine.
    #[must_use]
    pub fn sort_by_long_read(
        self,
    ) -> Builder<OrderedCore<SequenceOrder<128>, L, RadixThenRefine<128>>, C, S, CoreKeyed> {
        self.sort_by(LONG_READ_ORDER)
            .chunk_sort(RadixThenRefine::<128>)
    }

    /// Sort by record name.
    #[must_use]
    pub fn sort_by_name(self) -> Builder<OrderedCore<NameOrder, L, CS>, C, S, CoreKeyed> {
        self.sort_by(NAME_ORDER)
    }

    /// Sort by sequence length.
    #[must_use]
    pub fn sort_by_length(self) -> Builder<OrderedCore<LengthOrder, L, CS>, C, S, CoreKeyed> {
        self.sort_by(LENGTH_ORDER)
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
            dedup: self.dedup,
            core: self.core,
            storage: self.storage,
            mode: self.mode,
        }
    }
}

// Flush strategy selection.

impl<SK, C, CS, S, M> Builder<CoreBuilder<SK, NeedsLimit, CS>, C, S, M> {
    /// Flush when estimated memory usage exceeds `budget` bytes.
    /// Requires [`SeqRecord`] to implement `GetSize`.
    #[must_use]
    pub fn measured_budget(
        self,
        budget: usize,
    ) -> Builder<CoreBuilder<SK, ByteLimit, CS>, C, S, M> {
        let limit = ByteLimit::new(budget);
        self.map_core(|core| core.byte_limit(limit))
    }

    /// Flush when the buffer reaches `max_items` records.
    #[must_use]
    pub fn max_buffer_items(
        self,
        max_items: usize,
    ) -> Builder<CoreBuilder<SK, ItemLimit, CS>, C, S, M> {
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
            dedup: self.dedup,
            mode: PhantomData,
        }
    }

    /// Deduplicate adjacent records with identical names.
    #[must_use]
    pub fn dedup_by_name(mut self) -> Self {
        self.dedup = RecordDedup::Name;
        self
    }

    /// Deduplicate adjacent records with identical sequences.
    #[must_use]
    pub fn dedup_by_sequence(mut self) -> Self {
        self.dedup = RecordDedup::Sequence;
        self
    }

    /// Deduplicate adjacent records with identical sequence and quality.
    #[must_use]
    pub fn dedup_by_sequence_and_quality(mut self) -> Self {
        self.dedup = RecordDedup::SequenceAndQuality;
        self
    }
}

// Optional core configuration.

impl<SK, C, L, CS, S, M> Builder<CoreBuilder<SK, L, CS>, C, S, M> {
    /// Set a custom chunk sorting strategy. Defaults to
    /// [`Sequential`].
    ///
    /// The selected chunk sorter must be compatible with the selected sort order
    /// when [`build`](Self::build) is called. Comparator-based sorters such as
    /// [`Sequential`] are compatible with every built-in order. Specialized
    /// sorters such as [`RadixThenRefine`] are accepted only for the matching
    /// accelerated sequence order.
    #[must_use]
    pub fn chunk_sort<CS2>(self, chunk_sort: CS2) -> Builder<CoreBuilder<SK, L, CS2>, C, S, M> {
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
    pub fn sort_with_sequential(self) -> Builder<CoreBuilder<SK, L, Sequential>, C, S, M> {
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
    ) -> Builder<CoreBuilder<SK, L, spillover::chunk::Parallel>, C, S, M> {
        self.chunk_sort(spillover::chunk::Parallel)
    }
}

// build() for keyed sort orders.

type NativeKeyedCodec<O, C> = KeyedDryIceCodec<
    O,
    <C as sealed::ResolveCodec>::S,
    <C as sealed::ResolveCodec>::Q,
    <C as sealed::ResolveCodec>::N,
>;
type NativeRecord<C> = <<C as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record;
type NativeCore<O, C, CS, L> =
    spillover::sorter::Sorter<SeqRecord, O, C, (), Identity, CS, spillover::sorter::Keyed, L>;
type NativeSorter<O, C, CS, L> = Sorter<
    O,
    C,
    (),
    CS,
    spillover::sorter::Keyed,
    OwnedBackend<O, C, (), CS, spillover::sorter::Keyed, NativeCore<O, C, CS, L>>,
>;

macro_rules! owned_keyed_builder {
    ($limit:ty; $($extra:tt)*) => {
        impl<O, C, CS> Builder<OrderedCore<O, $limit, CS>, C, OwnedStorage, CoreKeyed>
        where
            O: KeyedSortOrder + spillover::key::DeriveKey<SeqRecord, StoredKey = <O as RecordKeyer>::RecordKey>
                + Compare<SeqRecord> + Compare<NativeRecord<NativeKeyedCodec<O, C>>>,
            <O as RecordKeyer>::RecordKey: RecordKey + Clone + 'static,
            C: sealed::ResolveCodec,
            CS: sealed::CompatibleChunkSorter<O>,
            for<'a> CS: ChunkSorter<spillover::key::Keyed<usize, <O as RecordKeyer>::RecordKey>, spillover::key::KeyedEntryCompare<'a, SeqRecord, O>>,
            $($extra)*
        {
            /// Build the configured owned keyed sorter.
            #[must_use]
            pub fn build(self) -> NativeSorter<O, NativeKeyedCodec<O, C>, CS, $limit> {
                let record_keyer = *self.core.configured_order();
                let codec = self.codec.resolve();
                let keyed_codec = codec.with_record_keyer(record_keyer);
                let inner = self.core.keyed_codec(keyed_codec).build();
                Sorter {
                    inner: OwnedBackend { inner, _types: std::marker::PhantomData },
                    dedup: self.dedup,
                    _types: std::marker::PhantomData,
                }
            }
        }
    };
}

owned_keyed_builder!(ByteLimit;
    O: spillover::key::EstimateKeyMemory<SeqRecord>,
    CS: spillover::chunk::EstimateSortMemory<spillover::key::Keyed<usize, <O as RecordKeyer>::RecordKey>>,
);
owned_keyed_builder!(ItemLimit;);

impl<O, C, CS> Builder<OrderedCore<O, NeedsLimit, CS>, C, OwnedStorage, CoreKeyed>
where
    O: KeyedSortOrder
        + spillover::key::DeriveKey<SeqRecord, StoredKey = <O as RecordKeyer>::RecordKey>
        + spillover::key::EstimateKeyMemory<SeqRecord>
        + Compare<SeqRecord>
        + Compare<NativeRecord<NativeKeyedCodec<O, C>>>,
    <O as RecordKeyer>::RecordKey: RecordKey + Clone + 'static,
    C: sealed::ResolveCodec,
    CS: sealed::CompatibleChunkSorter<O>
        + spillover::chunk::EstimateSortMemory<
            spillover::key::Keyed<usize, <O as RecordKeyer>::RecordKey>,
        >,
    for<'a> CS: ChunkSorter<
            spillover::key::Keyed<usize, <O as RecordKeyer>::RecordKey>,
            spillover::key::KeyedEntryCompare<'a, SeqRecord, O>,
        >,
{
    /// Build with the default 1 GiB measured budget.
    #[must_use]
    pub fn build(self) -> NativeSorter<O, NativeKeyedCodec<O, C>, CS, ByteLimit> {
        self.measured_budget(DEFAULT_BUDGET).build()
    }
}

// build() for unkeyed sort orders.

type NativeOrdinaryRecord<C> = <<C as Codec>::Cursor<BufReader<File>> as CodecCursor>::Record;
type NativeUnkeyedCore<O, C, CS, L> =
    spillover::sorter::Sorter<SeqRecord, O, C, (), Identity, CS, spillover::sorter::Unkeyed, L>;
type NativeUnkeyedSorter<O, C, CS, L> = Sorter<
    O,
    C,
    (),
    CS,
    spillover::sorter::Unkeyed,
    OwnedBackend<O, C, (), CS, spillover::sorter::Unkeyed, NativeUnkeyedCore<O, C, CS, L>>,
>;
type NativeUnkeyedOutput<C, O> = SortedRecordStream<spillover::Sorted<RunMerge<C, O>>>;
type NativeUnkeyedBuilt<O, C, CS, L> = NativeUnkeyedSorter<
    O,
    DryIceCodec<
        <C as sealed::ResolveCodec>::S,
        <C as sealed::ResolveCodec>::Q,
        <C as sealed::ResolveCodec>::N,
    >,
    CS,
    L,
>;

impl<O, C, CS> Builder<OrderedCore<O, ItemLimit, CS>, C, OwnedStorage, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: SortOrder<Strategy = Unkeyed>
        + Compare<SeqRecord>
        + Compare<NativeOrdinaryRecord<DryIceCodec<C::S, C::Q, C::N>>>,
    CS: ChunkSorter<SeqRecord, O> + sealed::CompatibleChunkSorter<O>,
{
    /// Build an item-limited sorter using the complete unkeyed order.
    #[must_use]
    pub fn build(self) -> NativeUnkeyedBuilt<O, C, CS, spillover::sorter::ItemLimit> {
        let codec = self.codec.resolve();
        let inner = self.core.codec(codec).build();
        Sorter {
            inner: OwnedBackend {
                inner,
                _types: std::marker::PhantomData,
            },
            dedup: self.dedup,
            _types: std::marker::PhantomData,
        }
    }
}

impl<O, C, CS> Builder<OrderedCore<O, NeedsLimit, CS>, C, OwnedStorage, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: SortOrder<Strategy = Unkeyed>
        + Compare<SeqRecord>
        + Compare<NativeOrdinaryRecord<DryIceCodec<C::S, C::Q, C::N>>>,
    CS: ChunkSorter<SeqRecord, O>
        + spillover::chunk::EstimateSortMemory<SeqRecord>
        + sealed::CompatibleChunkSorter<O>,
{
    /// Build with the default 1 GiB measured budget.
    #[must_use]
    pub fn build(self) -> NativeUnkeyedBuilt<O, C, CS, spillover::sorter::ByteLimit> {
        self.measured_budget(DEFAULT_BUDGET).build()
    }
}

impl<O, C, CS> Builder<OrderedCore<O, ByteLimit, CS>, C, OwnedStorage, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: SortOrder<Strategy = Unkeyed>
        + Compare<SeqRecord>
        + Compare<NativeOrdinaryRecord<DryIceCodec<C::S, C::Q, C::N>>>,
    CS: ChunkSorter<SeqRecord, O>
        + spillover::chunk::EstimateSortMemory<SeqRecord>
        + sealed::CompatibleChunkSorter<O>,
{
    /// Build an owned unkeyed sorter under a measured byte budget.
    #[must_use]
    pub fn build(self) -> NativeUnkeyedBuilt<O, C, CS, spillover::sorter::ByteLimit> {
        let codec = self.codec.resolve();
        let inner = self.core.codec(codec).build();
        Sorter {
            inner: OwnedBackend {
                inner,
                _types: std::marker::PhantomData,
            },
            dedup: self.dedup,
            _types: std::marker::PhantomData,
        }
    }
}

impl<O, C, CS> NativeUnkeyedSorter<O, C, CS, spillover::sorter::ByteLimit>
where
    C: Codec + EncodeRun<SeqRecord> + EncodeRun<NativeOrdinaryRecord<C>>,
    O: Compare<SeqRecord> + Compare<NativeOrdinaryRecord<C>>,
    CS: ChunkSorter<SeqRecord, O> + spillover::chunk::EstimateSortMemory<SeqRecord>,
{
    /// Add a record under the measured byte budget.
    ///
    /// # Errors
    /// Returns an error if spilling fails.
    pub fn push(&mut self, record: impl Into<SeqRecord>) -> Result<(), MergeError<C::Error>> {
        self.inner.inner.push(record.into())
    }

    /// Finish sorting and return the finalized stream.
    ///
    /// # Errors
    /// Returns an error if spilling or merging fails.
    pub fn finish(self) -> Result<NativeUnkeyedOutput<C, O>, MergeError<C::Error>> {
        let inner = self.inner.inner.finish()?;
        let dedup = RecordDedupState::new(self.dedup);
        Ok(SortedRecordStream::new(inner, dedup))
    }
}

impl<O, C, CS> NativeUnkeyedSorter<O, C, CS, spillover::sorter::ItemLimit>
where
    C: Codec + EncodeRun<SeqRecord> + EncodeRun<NativeOrdinaryRecord<C>>,
    O: Compare<SeqRecord> + Compare<NativeOrdinaryRecord<C>>,
    CS: ChunkSorter<SeqRecord, O>,
{
    /// Add an unsorted record to the sorter.
    ///
    /// # Errors
    /// Returns an error if spilling fails.
    pub fn push(&mut self, record: impl Into<SeqRecord>) -> Result<(), MergeError<C::Error>> {
        self.inner.inner.push(record.into())
    }

    /// Finish sorting and return the finalized sorted record stream.
    ///
    /// # Errors
    /// Returns an error if spilling or merging fails.
    pub fn finish(self) -> Result<NativeUnkeyedOutput<C, O>, MergeError<C::Error>> {
        let inner = self.inner.inner.finish()?;
        let dedup = RecordDedupState::new(self.dedup);
        Ok(SortedRecordStream::new(inner, dedup))
    }
}

type ArenaUnkeyedCore<'a, O, C, CS> = spillover::sorter::Sorter<
    (),
    O,
    C,
    (),
    Identity,
    CS,
    spillover::sorter::Unkeyed,
    spillover::sorter::ArenaBuffer<'a, SeqRecordArena>,
>;
type NativeArenaUnkeyed<'a, O, C, CS> =
    Sorter<O, C, (), CS, spillover::sorter::Unkeyed, ArenaUnkeyedCore<'a, O, C, CS>>;
type NativeArenaBuilt<'a, O, C, CS> = NativeArenaUnkeyed<
    'a,
    O,
    DryIceCodec<
        <C as sealed::ResolveCodec>::S,
        <C as sealed::ResolveCodec>::Q,
        <C as sealed::ResolveCodec>::N,
    >,
    CS,
>;

impl<'a, O, C, CS> Builder<OrderedCore<O, ItemLimit, CS>, C, ArenaStorage<'a>, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: SortOrder<Strategy = Unkeyed> + Compare<NativeOrdinaryRecord<DryIceCodec<C::S, C::Q, C::N>>>,
    for<'v> O: Compare<SeqRecordView<'v>>,
    CS: sealed::CompatibleChunkSorter<O>,
    for<'v> CS: ChunkSorter<SeqRecordView<'v>, O>,
{
    /// Build an item-limited sorter over the selected arena.
    #[must_use]
    pub fn build(self) -> NativeArenaBuilt<'a, O, C, CS> {
        let codec = self.codec.resolve();
        let storage = self.storage.arena;
        let builder = self.core.codec(codec);
        let build_in: fn(_, &'a mut SeqRecordArena) -> _ = spillover::sorter::Builder::build_in;
        let inner = build_in(builder, storage);
        Sorter {
            inner,
            dedup: self.dedup,
            _types: std::marker::PhantomData,
        }
    }
}

impl<O, C, CS> NativeArenaUnkeyed<'_, O, C, CS>
where
    C: Codec + EncodeRun<NativeOrdinaryRecord<C>>,
    for<'v> C: EncodeRun<SeqRecordView<'v>>,
    O: Compare<NativeOrdinaryRecord<C>>,
    for<'v> O: Compare<SeqRecordView<'v>>,
    for<'v> CS: ChunkSorter<SeqRecordView<'v>, O>,
{
    /// Add a borrowed record to the arena-backed sorter.
    ///
    /// # Errors
    /// Returns an error if spilling fails.
    pub fn push<R: SeqRecordParts + ?Sized>(
        &mut self,
        record: &R,
    ) -> Result<(), MergeError<C::Error>> {
        self.inner.push(record)
    }

    /// Finish the sorter and return its sorted record stream.
    ///
    /// # Errors
    /// Returns an error if spilling or merging fails.
    pub fn finish(self) -> Result<NativeUnkeyedOutput<C, O>, MergeError<C::Error>> {
        let inner = self.inner.finish()?;
        let dedup = RecordDedupState::new(self.dedup);
        Ok(SortedRecordStream::new(inner, dedup))
    }
}

type MeasuredArenaUnkeyedCore<'a, O, C, CS> = spillover::sorter::Sorter<
    (),
    O,
    C,
    (),
    Identity,
    CS,
    spillover::sorter::Unkeyed,
    spillover::sorter::MeasuredArena<'a, SeqRecordArena>,
>;
type NativeMeasuredArenaUnkeyed<'a, O, C, CS> =
    Sorter<O, C, (), CS, spillover::sorter::Unkeyed, MeasuredArenaUnkeyedCore<'a, O, C, CS>>;
type NativeMeasuredArenaBuilt<'a, O, C, CS> = NativeMeasuredArenaUnkeyed<
    'a,
    O,
    DryIceCodec<
        <C as sealed::ResolveCodec>::S,
        <C as sealed::ResolveCodec>::Q,
        <C as sealed::ResolveCodec>::N,
    >,
    CS,
>;

impl<'a, O, C, CS> Builder<OrderedCore<O, ByteLimit, CS>, C, ArenaStorage<'a>, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: SortOrder<Strategy = Unkeyed> + Compare<NativeOrdinaryRecord<DryIceCodec<C::S, C::Q, C::N>>>,
    for<'v> O: Compare<SeqRecordView<'v>>,
    CS: sealed::CompatibleChunkSorter<O>,
    for<'v> CS:
        ChunkSorter<SeqRecordView<'v>, O> + spillover::chunk::EstimateSortMemory<SeqRecordView<'v>>,
{
    /// Build a measured sorter over the selected arena.
    #[must_use]
    pub fn build(self) -> NativeMeasuredArenaBuilt<'a, O, C, CS> {
        let codec = self.codec.resolve();
        let storage = self.storage.arena;
        let builder = self.core.codec(codec);
        let build_measured_in: fn(_, &'a mut SeqRecordArena) -> _ =
            spillover::sorter::Builder::build_measured_in;
        let inner = build_measured_in(builder, storage);
        Sorter {
            inner,
            dedup: self.dedup,
            _types: std::marker::PhantomData,
        }
    }
}

impl<'a, O, C, CS> Builder<OrderedCore<O, NeedsLimit, CS>, C, ArenaStorage<'a>, CoreUnkeyed>
where
    C: sealed::ResolveCodec,
    O: SortOrder<Strategy = Unkeyed> + Compare<NativeOrdinaryRecord<DryIceCodec<C::S, C::Q, C::N>>>,
    for<'v> O: Compare<SeqRecordView<'v>>,
    CS: sealed::CompatibleChunkSorter<O>,
    for<'v> CS:
        ChunkSorter<SeqRecordView<'v>, O> + spillover::chunk::EstimateSortMemory<SeqRecordView<'v>>,
{
    /// Build a measured sorter over the selected arena.
    #[must_use]
    pub fn build(self) -> NativeMeasuredArenaBuilt<'a, O, C, CS> {
        self.measured_budget(DEFAULT_BUDGET).build()
    }
}

impl<O, C, CS> NativeMeasuredArenaUnkeyed<'_, O, C, CS>
where
    C: Codec + EncodeRun<NativeOrdinaryRecord<C>>,
    for<'v> C: EncodeRun<SeqRecordView<'v>>,
    O: Compare<NativeOrdinaryRecord<C>>,
    for<'v> O: Compare<SeqRecordView<'v>>,
    for<'v> CS:
        ChunkSorter<SeqRecordView<'v>, O> + spillover::chunk::EstimateSortMemory<SeqRecordView<'v>>,
{
    /// Add a borrowed record to the arena-backed sorter.
    ///
    /// # Errors
    /// Returns an error if spilling fails.
    pub fn push<R: SeqRecordParts + ?Sized>(
        &mut self,
        record: &R,
    ) -> Result<(), MergeError<C::Error>> {
        self.inner.push(record)
    }

    /// Finish the sorter and return its sorted record stream.
    ///
    /// # Errors
    /// Returns an error if spilling or merging fails.
    pub fn finish(self) -> Result<NativeUnkeyedOutput<C, O>, MergeError<C::Error>> {
        let inner = self.inner.finish()?;
        let dedup = RecordDedupState::new(self.dedup);
        Ok(SortedRecordStream::new(inner, dedup))
    }
}

type ArenaKeyedCore<'a, O, C, CS> = spillover::sorter::Sorter<
    (),
    O,
    C,
    (),
    Identity,
    CS,
    spillover::sorter::Keyed,
    spillover::sorter::ArenaBuffer<'a, SeqRecordArena>,
>;
type NativeArenaKeyed<'a, O, C, CS> =
    Sorter<O, C, (), CS, spillover::sorter::Keyed, ArenaKeyedCore<'a, O, C, CS>>;

impl<'a, O, C, CS> Builder<OrderedCore<O, ItemLimit, CS>, C, ArenaStorage<'a>, CoreKeyed>
where
    O: KeyedSortOrder + Compare<NativeRecord<NativeKeyedCodec<O, C>>>,
    <O as RecordKeyer>::RecordKey: RecordKey + Clone + 'static,
    C: sealed::ResolveCodec,
    CS: sealed::CompatibleChunkSorter<O>,
    for<'v> O: spillover::key::DeriveKey<SeqRecordView<'v>, StoredKey = <O as RecordKeyer>::RecordKey>
        + Compare<SeqRecordView<'v>>,
    for<'v, 'order> CS: ChunkSorter<
            spillover::key::Keyed<usize, <O as RecordKeyer>::RecordKey>,
            spillover::key::KeyedEntryCompare<'order, SeqRecordView<'v>, O>,
        >,
{
    /// Build an item-limited keyed sorter over the selected arena.
    #[must_use]
    pub fn build(self) -> NativeArenaKeyed<'a, O, NativeKeyedCodec<O, C>, CS> {
        let record_keyer = *self.core.configured_order();
        let codec = self.codec.resolve();
        let keyed_codec = codec.with_record_keyer(record_keyer);
        let storage = self.storage.arena;
        let builder = self.core.keyed_codec(keyed_codec);
        let build_keyed_in: fn(_, &'a mut SeqRecordArena) -> _ =
            spillover::sorter::Builder::build_keyed_in;
        let inner = build_keyed_in(builder, storage);
        Sorter {
            inner,
            dedup: self.dedup,
            _types: std::marker::PhantomData,
        }
    }
}

type MeasuredArenaKeyedCore<'a, O, C, CS> = spillover::sorter::Sorter<
    (),
    O,
    C,
    (),
    Identity,
    CS,
    spillover::sorter::Keyed,
    spillover::sorter::MeasuredArena<'a, SeqRecordArena>,
>;
type NativeMeasuredArenaKeyed<'a, O, C, CS> =
    Sorter<O, C, (), CS, spillover::sorter::Keyed, MeasuredArenaKeyedCore<'a, O, C, CS>>;

impl<'a, O, C, CS> Builder<OrderedCore<O, ByteLimit, CS>, C, ArenaStorage<'a>, CoreKeyed>
where
    O: KeyedSortOrder + Compare<NativeRecord<NativeKeyedCodec<O, C>>>,
    <O as RecordKeyer>::RecordKey: RecordKey + Clone + 'static,
    C: sealed::ResolveCodec,
    CS: sealed::CompatibleChunkSorter<O>
        + spillover::chunk::EstimateSortMemory<
            spillover::key::Keyed<usize, <O as RecordKeyer>::RecordKey>,
        >,
    for<'v> O: spillover::key::DeriveKey<SeqRecordView<'v>, StoredKey = <O as RecordKeyer>::RecordKey>
        + spillover::key::EstimateKeyMemory<SeqRecordView<'v>>
        + Compare<SeqRecordView<'v>>,
    for<'v, 'order> CS: ChunkSorter<
            spillover::key::Keyed<usize, <O as RecordKeyer>::RecordKey>,
            spillover::key::KeyedEntryCompare<'order, SeqRecordView<'v>, O>,
        >,
{
    /// Build a measured keyed sorter using the arena.
    #[must_use]
    pub fn build(self) -> NativeMeasuredArenaKeyed<'a, O, NativeKeyedCodec<O, C>, CS> {
        let record_keyer = *self.core.configured_order();
        let codec = self.codec.resolve();
        let keyed_codec = codec.with_record_keyer(record_keyer);
        let storage = self.storage.arena;
        let builder = self.core.keyed_codec(keyed_codec);
        let build_keyed_measured_in: fn(_, &'a mut SeqRecordArena) -> _ =
            spillover::sorter::Builder::build_keyed_measured_in;
        let inner = build_keyed_measured_in(builder, storage);
        Sorter {
            inner,
            dedup: self.dedup,
            _types: std::marker::PhantomData,
        }
    }
}

impl<'a, O, C, CS> Builder<OrderedCore<O, NeedsLimit, CS>, C, ArenaStorage<'a>, CoreKeyed>
where
    O: KeyedSortOrder + Compare<NativeRecord<NativeKeyedCodec<O, C>>>,
    <O as RecordKeyer>::RecordKey: RecordKey + Clone + 'static,
    C: sealed::ResolveCodec,
    CS: sealed::CompatibleChunkSorter<O>
        + spillover::chunk::EstimateSortMemory<
            spillover::key::Keyed<usize, <O as RecordKeyer>::RecordKey>,
        >,
    for<'v> O: spillover::key::DeriveKey<SeqRecordView<'v>, StoredKey = <O as RecordKeyer>::RecordKey>
        + spillover::key::EstimateKeyMemory<SeqRecordView<'v>>
        + Compare<SeqRecordView<'v>>,
    for<'v, 'order> CS: ChunkSorter<
            spillover::key::Keyed<usize, <O as RecordKeyer>::RecordKey>,
            spillover::key::KeyedEntryCompare<'order, SeqRecordView<'v>, O>,
        >,
{
    /// Build a keyed sorter with the default measured budget.
    #[must_use]
    pub fn build(self) -> NativeMeasuredArenaKeyed<'a, O, NativeKeyedCodec<O, C>, CS> {
        self.measured_budget(DEFAULT_BUDGET).build()
    }
}

impl<O, C, CS> NativeMeasuredArenaKeyed<'_, O, C, CS>
where
    C: KeyedCodec + EncodeKeyedRun<NativeRecord<C>>,
    for<'v> C: EncodeKeyedRun<SeqRecordView<'v>>,
    O: Compare<NativeRecord<C>>,
    for<'v> O: spillover::key::DeriveKey<SeqRecordView<'v>, StoredKey = C::Key>
        + spillover::key::EstimateKeyMemory<SeqRecordView<'v>>
        + Compare<SeqRecordView<'v>>,
    CS: spillover::chunk::EstimateSortMemory<spillover::key::Keyed<usize, C::Key>>,
    for<'v, 'order> CS: ChunkSorter<
            spillover::key::Keyed<usize, C::Key>,
            spillover::key::KeyedEntryCompare<'order, SeqRecordView<'v>, O>,
        >,
{
    /// Insert a borrowed record.
    ///
    /// # Errors
    /// Returns an error if completing a run fails.
    pub fn push<R: SeqRecordParts + ?Sized>(
        &mut self,
        record: &R,
    ) -> Result<(), MergeError<C::Error>> {
        let name = record.name();
        let sequence = record.sequence();
        let quality = record.quality();
        let view = SeqRecordView::new(name, sequence, quality);
        self.inner.push(&view)
    }

    /// Finish and return the sorted records.
    ///
    /// # Errors
    /// Returns an error if spilling or merging fails.
    pub fn finish(self) -> Result<NativeOutput<C, O>, MergeError<C::Error>> {
        let inner = self.inner.finish()?;
        let dedup = RecordDedupState::new(self.dedup);
        Ok(SortedRecordStream::new(inner, dedup))
    }
}

impl<O, C, CS> NativeArenaKeyed<'_, O, C, CS>
where
    C: KeyedCodec + EncodeKeyedRun<NativeRecord<C>>,
    for<'v> C: EncodeKeyedRun<SeqRecordView<'v>>,
    O: Compare<NativeRecord<C>>,
    for<'v> O: spillover::key::DeriveKey<SeqRecordView<'v>, StoredKey = C::Key>
        + Compare<SeqRecordView<'v>>,
    for<'v, 'order> CS: ChunkSorter<
            spillover::key::Keyed<usize, C::Key>,
            spillover::key::KeyedEntryCompare<'order, SeqRecordView<'v>, O>,
        >,
{
    /// Add a borrowed record.
    ///
    /// # Errors
    /// Returns a spill error.
    pub fn push<R: SeqRecordParts + ?Sized>(
        &mut self,
        record: &R,
    ) -> Result<(), MergeError<C::Error>> {
        self.inner.push(record)
    }

    /// Finish sorting.
    ///
    /// # Errors
    /// Returns a spill or merge error.
    pub fn finish(self) -> Result<NativeOutput<C, O>, MergeError<C::Error>> {
        let inner = self.inner.finish()?;
        let dedup = RecordDedupState::new(self.dedup);
        Ok(SortedRecordStream::new(inner, dedup))
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
        let key = ILLUMINA_ORDER.record_key(&rec);
        let expected = crate::key::PackedSequenceKey::<38>::from_sequence(b"ACGT");
        assert_eq!(key, expected);
    }

    #[test]
    fn sort_by_sequence_exact_selects_unkeyed_sequence_order() {
        fn assert_builder_type<T>(_: &T) {}

        let builder = Builder::new().sort_by_sequence_exact();

        assert_builder_type::<
            Builder<
                OrderedCore<UnkeyedSequenceOrder, NeedsLimit, Sequential>,
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
                OrderedCore<UnkeyedSequenceOrder, NeedsLimit, Sequential>,
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
                OrderedCore<Reverse<SequenceOrder<38>>, NeedsLimit, Sequential>,
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
        let key = NameOrder.record_key(&rec);
        let mut expected = [0u8; 16];
        expected[..8].copy_from_slice(b"read_001");
        assert_eq!(key, dryice::Bytes16Key(expected));
    }

    #[test]
    fn name_order_truncates_long_names() {
        let rec = make_record(b"a_very_long_name_that_exceeds_16_bytes", b"ACGT", b"!!!!");
        let key = NameOrder.record_key(&rec);
        assert_eq!(&key.0, b"a_very_long_name");
    }

    #[test]
    fn length_order_produces_correct_key() {
        let rec = make_record(b"r1", b"ACGTACGT", b"!!!!!!!!");
        let key = LengthOrder.record_key(&rec);
        assert_eq!(key, dryice::Bytes8Key(8u64.to_be_bytes()));
    }

    #[test]
    fn length_order_big_endian_preserves_ord() {
        let short = make_record(b"r1", b"AC", b"!!");
        let long = make_record(b"r2", b"ACGTACGT", b"!!!!!!!!");
        let key_short = LengthOrder.record_key(&short);
        let key_long = LengthOrder.record_key(&long);
        assert!(
            key_short < key_long,
            "big-endian u64 should preserve ordering"
        );
    }

    #[test]
    fn sequence_order_uses_tuple_key_with_tiebreaker() {
        let order = ILLUMINA_ORDER;
        let sk = order.sort_key();
        let cmp = order.key_comparator();

        let a = make_record(b"r1", b"ACGT", b"IIII");
        let b = make_record(b"r2", b"ACGT", b"!!!!");

        let ka = sk.key(&a);
        let kb = sk.key(&b);

        assert_eq!(
            cmp.compare(&ka, &kb),
            Ordering::Greater,
            "same sequence, IIII > !!!! via tuple Ord"
        );
    }

    #[test]
    fn reverse_flips_key_ordering() {
        let order = Reverse(ILLUMINA_ORDER);
        let sk = order.sort_key();
        let cmp = order.key_comparator();

        let a = make_record(b"r1", b"AAAA", b"!!!!");
        let b = make_record(b"r2", b"TTTT", b"!!!!");

        let ka = sk.key(&a);
        let kb = sk.key(&b);

        assert_eq!(
            cmp.compare(&ka, &kb),
            Ordering::Greater,
            "reversed: AAAA key should be greater than TTTT key"
        );
    }

    #[test]
    fn reverse_preserves_record_key() {
        let rec = make_record(b"r1", b"ACGT", b"!!!!");
        let forward_key = ILLUMINA_ORDER.record_key(&rec);
        let reverse_key = Reverse(ILLUMINA_ORDER).record_key(&rec);
        assert_eq!(
            forward_key, reverse_key,
            "reverse should not change the record key"
        );
    }

    #[test]
    fn unkeyed_sequence_order_has_unkeyed_strategy() {
        fn assert_unkeyed<O: SortOrder<Strategy = Unkeyed>>() {}
        assert_unkeyed::<UnkeyedSequenceOrder>();
    }

    #[test]
    fn keyed_sequence_order_has_keyed_strategy() {
        fn assert_keyed<O: SortOrder<Strategy = Keyed>>() {}
        assert_keyed::<SequenceOrder<38>>();
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
        fn assert_unkeyed<O: SortOrder<Strategy = Unkeyed>>() {}

        let _unkeyed = ILLUMINA_ORDER.unkeyed();
        assert_unkeyed::<UnkeyedSequenceOrder>();

        let _unkeyed = NameOrder.unkeyed();
        assert_unkeyed::<UnkeyedNameOrder>();

        let _unkeyed = LengthOrder.unkeyed();
        assert_unkeyed::<UnkeyedLengthOrder>();
    }
}
