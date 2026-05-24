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

use dryice::{
    DryIceWriter, NameCodec, QualityCodec, RawAsciiCodec, RawNameCodec, RawQualityCodec, RecordKey,
    SequenceCodec,
};
use spillover::{
    SortedItemsError,
    chunk::{ChunkSorter, Sequential},
    codec::{Codec, CodecWriter, DeriveKey, KeyedCodec, KeyedCodecWriter},
    compare::{Compare, Natural},
    dedup::Identity,
    key::{KeyCompare, SortKey},
    merge::{KeyedRunMerge, MergeConfig, MergeError, RunMerge},
    sorter::VisitSortedItems,
};

use crate::{
    codec::{DryIceCodec, KeyedDryIceCodec},
    error::SortedRecordStreamError,
    key::PackedSequenceKey,
    radix::RadixThenRefine,
    record::{SeqRecord, SeqRecordArena, SeqRecordParts},
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

    /// Resolves codec type-state to a concrete [`crate::codec::DryIceCodec`].
    /// `NeedsCodec` yields the raw (no-encoding) default;
    /// `HasCodec` passes through the user's choice.
    pub trait ResolveCodec {
        type S: dryice::SequenceCodec + Copy + 'static;
        type Q: dryice::QualityCodec + Copy + 'static;
        type N: dryice::NameCodec + Copy + 'static;

        fn resolve(self) -> crate::codec::DryIceCodec<Self::S, Self::Q, Self::N>;
    }

    /// Resolves flush type-state to a concrete [`super::FlushConfig`].
    /// `NeedsFlush` yields a 1 GiB measured budget;
    /// `HasFlush` passes through the user's choice.
    pub trait ResolveFlush {
        fn resolve(self) -> super::FlushConfig;
    }
}

/// Defines a complete sort order: what to sort by, how to
/// compare, and what merge strategy to use. Sealed — only
/// spillover-bio's built-in orders implement this.
pub trait SortOrder: sealed::Sealed + Copy {
    /// The sort key for extracting the comparison value.
    type SortKey: SortKey<SeqRecord> + Copy + Send + Sync + 'static;

    /// The comparator for key comparison. Must implement
    /// `Compare<Key>` for whatever key type the `SortKey` produces.
    type Compare: Copy + Send + Sync + 'static;

    /// The merge strategy marker: [`Basic`] or [`Keyed`].
    type Strategy;

    /// The sort key extractor.
    fn sort_key(&self) -> Self::SortKey;

    /// The comparator.
    fn compare(&self) -> Self::Compare;
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

/// Marker: base merge path (no record keys).
pub struct Basic;

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
    type Compare = Natural;
    type Strategy = Keyed;

    fn sort_key(&self) -> SequenceQualityKey {
        SequenceQualityKey
    }

    fn compare(&self) -> Natural {
        Natural
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
    type Compare = Natural;
    type Strategy = Keyed;

    fn sort_key(&self) -> NameKey {
        NameKey
    }

    fn compare(&self) -> Natural {
        Natural
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
#[derive(Debug, Clone, Copy)]
pub struct LengthOrder;

impl sealed::Sealed for LengthOrder {}

impl SortOrder for LengthOrder {
    type SortKey = LengthKey;
    type Compare = Natural;
    type Strategy = Keyed;

    fn sort_key(&self) -> LengthKey {
        LengthKey
    }

    fn compare(&self) -> Natural {
        Natural
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

// ── Unkeyed sort orders ──────────────────────────────────

/// Sort by sequence without record key acceleration.
#[derive(Debug, Clone, Copy)]
pub struct UnkeyedSequenceOrder;

impl sealed::Sealed for UnkeyedSequenceOrder {}

impl SortOrder for UnkeyedSequenceOrder {
    type SortKey = SequenceQualityKey;
    type Compare = Natural;
    type Strategy = Basic;

    fn sort_key(&self) -> SequenceQualityKey {
        SequenceQualityKey
    }

    fn compare(&self) -> Natural {
        Natural
    }
}

/// Sort by name without record key acceleration.
#[derive(Debug, Clone, Copy)]
pub struct UnkeyedNameOrder;

impl sealed::Sealed for UnkeyedNameOrder {}

impl SortOrder for UnkeyedNameOrder {
    type SortKey = NameKey;
    type Compare = Natural;
    type Strategy = Basic;

    fn sort_key(&self) -> NameKey {
        NameKey
    }

    fn compare(&self) -> Natural {
        Natural
    }
}

/// Sort by length without record key acceleration.
#[derive(Debug, Clone, Copy)]
pub struct UnkeyedLengthOrder;

impl sealed::Sealed for UnkeyedLengthOrder {}

impl SortOrder for UnkeyedLengthOrder {
    type SortKey = LengthKey;
    type Compare = Natural;
    type Strategy = Basic;

    fn sort_key(&self) -> LengthKey {
        LengthKey
    }

    fn compare(&self) -> Natural {
        Natural
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

impl<O: SortOrder> SortOrder for Reverse<O> {
    type SortKey = O::SortKey;
    type Compare = spillover::compare::Reverse<O::Compare>;
    type Strategy = O::Strategy;

    fn sort_key(&self) -> Self::SortKey {
        self.0.sort_key()
    }

    fn compare(&self) -> spillover::compare::Reverse<O::Compare> {
        spillover::compare::Reverse(self.0.compare())
    }
}

impl<O: RecordKeyer> RecordKeyer for Reverse<O> {
    type RecordKey = O::RecordKey;

    fn record_key<R: SeqRecordParts + ?Sized>(&self, record: &R) -> Self::RecordKey {
        self.0.record_key(record)
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

/// Builder marker: no sort order chosen yet.
pub struct NeedsOrder;
/// Builder marker: keyed sort order has been chosen.
pub struct HasKeyedOrder<O>(O);
/// Builder marker: unkeyed sort order has been chosen.
pub struct HasUnkeyedOrder<O>(O);

/// Builder marker: no codec chosen yet.
pub struct NeedsCodec;
/// Builder marker: codec has been chosen.
pub struct HasCodec<S, Q, N>(DryIceCodec<S, Q, N>);

/// Builder marker: no flush strategy chosen yet.
pub struct NeedsFlush;
/// Builder marker: flush strategy has been chosen.
pub struct HasFlush(FlushConfig);

/// Builder marker: use the default owned-record buffer.
pub struct OwnedStorage;

/// Builder marker: use caller-provided arena storage for the spill window.
pub struct ArenaStorage<'a> {
    #[allow(
        dead_code,
        reason = "arena-backed build will consume the arena storage marker in the next phase"
    )]
    arena: &'a mut SeqRecordArena,
}

// pub visibility required because `sealed::ResolveFlush`
// returns it. The sealed module prevents external impls.
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

impl sealed::ResolveFlush for NeedsFlush {
    fn resolve(self) -> FlushConfig {
        FlushConfig::MeasuredBudget(DEFAULT_BUDGET)
    }
}

impl sealed::ResolveFlush for HasFlush {
    fn resolve(self) -> FlushConfig {
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
/// radix sort for sequence orders, sequential comparison sort for
/// name/length orders.
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
pub struct Builder<
    O = NeedsOrder,
    C = NeedsCodec,
    F = NeedsFlush,
    CS = Sequential,
    S = OwnedStorage,
> {
    order: O,
    codec: C,
    flush: F,
    dedup: RecordDedup,
    chunk_sort: CS,
    storage: S,
    merge_config: MergeConfig,
}

/// Active genomics sorter.
///
/// Construct a sorter with [`Builder`], push unsorted records into it, then call
/// [`finish`](Self::finish) to produce a [`SortedRecordStream`].
pub struct Sorter<SK, Cod, Cmp, CS, M> {
    inner: spillover::sorter::Sorter<SeqRecord, SK, Cod, Cmp, Identity, CS, M>,
    dedup: RecordDedup,
}

/// Finalized stream of sorted sequence records.
///
/// `SortedRecordStream` owns the sorted output source after a [`Sorter`] has
/// been finished. It implements [`Iterator`] for owned materialization, so
/// ordinary `.collect()` callsites continue to work.
pub struct SortedRecordStream<I> {
    inner: spillover::Sorted<I>,
    dedup: RecordDedup,
    previous: Option<PreviousRecord>,
    fused: bool,
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
    spillover::Sorted<I>: Iterator<Item = Result<SeqRecord, E>>,
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
                    if self.dedup.is_duplicate(self.previous.as_ref(), &record) {
                        continue;
                    }

                    self.previous
                        .get_or_insert_with(PreviousRecord::default)
                        .copy_from(&record);
                    return Some(Ok(record));
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
    for<'a> I::Item<'a>: SeqRecordParts,
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
            dedup,
            mut previous,
            fused,
        } = self;

        if fused {
            return Ok(());
        }

        inner
            .items()
            .try_for_each(|record| {
                if dedup.is_duplicate(previous.as_ref(), &record) {
                    return Ok(());
                }

                if dedup.is_enabled() {
                    let previous = previous.get_or_insert_with(PreviousRecord::default);
                    previous.copy_from(&record);
                    writer.write_record(previous)
                } else {
                    writer.write_record(&record)
                }
            })
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
            order: NeedsOrder,
            codec: NeedsCodec,
            flush: NeedsFlush,
            dedup: RecordDedup::None,
            chunk_sort: Sequential,
            storage: OwnedStorage,
            merge_config: MergeConfig::default(),
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

// These bounds intentionally mirror the core sorter's basic and keyed
// push/finish implementations. The bio wrapper owns the public lifecycle
// vocabulary; core owns the sorting mechanics.

impl<SK, Cod, Cmp, CS> Sorter<SK, Cod, Cmp, CS, spillover::sorter::Basic>
where
    SK: SortKey<SeqRecord> + Copy + Send + Sync + 'static,
    Cod: Codec<Item = SeqRecord> + Copy + 'static,
    for<'a> Cod::Writer<&'a mut std::fs::File>: CodecWriter<SeqRecord, Error = Cod::Error>,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy + Send + Sync + 'static,
    CS: ChunkSorter<SeqRecord>,
{
    /// Add an unsorted record to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing buffered records to disk fails.
    pub fn push(&mut self, record: impl Into<SeqRecord>) -> Result<(), MergeError<Cod::Error>> {
        self.inner.push(record.into())
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
        SortedRecordStream<RunMerge<SeqRecord, Cod, KeyCompare<SK, Cmp>>>,
        MergeError<Cod::Error>,
    > {
        Ok(SortedRecordStream {
            inner: self.inner.finish()?,
            dedup: self.dedup,
            previous: None,
            fused: false,
        })
    }
}

impl<SK, Cod, Cmp, CS> Sorter<SK, Cod, Cmp, CS, spillover::sorter::Keyed>
where
    SK: SortKey<SeqRecord> + Copy + Send + Sync + 'static,
    Cod: KeyedCodec<Item = SeqRecord> + DeriveKey<SeqRecord> + Copy + 'static,
    for<'a> Cod::KeyedWriter<&'a mut std::fs::File>:
        KeyedCodecWriter<SeqRecord, Cod::Key, Error = Cod::Error>,
    Cmp: for<'a> Compare<SK::Key<'a>> + Compare<Cod::Key> + Copy + Send + Sync + 'static,
    CS: ChunkSorter<SeqRecord>,
{
    /// Add an unsorted record to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing buffered records to disk fails.
    pub fn push(&mut self, record: impl Into<SeqRecord>) -> Result<(), MergeError<Cod::Error>> {
        self.inner.push(record.into())
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
        SortedRecordStream<KeyedRunMerge<SeqRecord, Cod, Cmp, KeyCompare<SK, Cmp>>>,
        MergeError<Cod::Error>,
    > {
        Ok(SortedRecordStream {
            inner: self.inner.finish()?,
            dedup: self.dedup,
            previous: None,
            fused: false,
        })
    }
}

// Sort order selection.

impl<O, C, F, CS> Builder<O, C, F, CS, OwnedStorage> {
    /// Use caller-provided arena storage for the active spill window.
    #[must_use]
    pub fn arena(self, arena: &mut SeqRecordArena) -> Builder<O, C, F, CS, ArenaStorage<'_>> {
        Builder {
            order: self.order,
            codec: self.codec,
            flush: self.flush,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            storage: ArenaStorage { arena },
            merge_config: self.merge_config,
        }
    }
}

impl<C, F, CS, S> Builder<NeedsOrder, C, F, CS, S> {
    /// Set a keyed sort order (merge compares record keys).
    #[must_use]
    pub fn sort_by<O: KeyedSortOrder>(self, order: O) -> Builder<HasKeyedOrder<O>, C, F, CS, S> {
        Builder {
            order: HasKeyedOrder(order),
            codec: self.codec,
            flush: self.flush,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            storage: self.storage,
            merge_config: self.merge_config,
        }
    }

    /// Set an unkeyed sort order (merge deserializes full records).
    #[must_use]
    pub fn sort_by_unkeyed<O: SortOrder<Strategy = Basic>>(
        self,
        order: O,
    ) -> Builder<HasUnkeyedOrder<O>, C, F, CS, S> {
        Builder {
            order: HasUnkeyedOrder(order),
            codec: self.codec,
            flush: self.flush,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            storage: self.storage,
            merge_config: self.merge_config,
        }
    }

    /// Sort by sequence (Illumina 150bp, 38-byte packed key).
    /// Defaults to radix sort for the chunk sorting engine.
    #[must_use]
    pub fn sort_by_illumina(
        self,
    ) -> Builder<HasKeyedOrder<SequenceOrder<38>>, C, F, RadixThenRefine<38>, S> {
        self.sort_by(ILLUMINA_ORDER)
            .chunk_sort(RadixThenRefine::<38>)
    }

    /// Sort by sequence (250bp paired-end, 64-byte packed key).
    /// Defaults to radix sort for the chunk sorting engine.
    #[must_use]
    pub fn sort_by_paired_end(
        self,
    ) -> Builder<HasKeyedOrder<SequenceOrder<64>>, C, F, RadixThenRefine<64>, S> {
        self.sort_by(PAIRED_END_ORDER)
            .chunk_sort(RadixThenRefine::<64>)
    }

    /// Sort by sequence (long reads, 128-byte prefix key).
    /// Defaults to radix sort for the chunk sorting engine.
    #[must_use]
    pub fn sort_by_long_read(
        self,
    ) -> Builder<HasKeyedOrder<SequenceOrder<128>>, C, F, RadixThenRefine<128>, S> {
        self.sort_by(LONG_READ_ORDER)
            .chunk_sort(RadixThenRefine::<128>)
    }

    /// Sort by record name.
    #[must_use]
    pub fn sort_by_name(self) -> Builder<HasKeyedOrder<NameOrder>, C, F, CS, S> {
        self.sort_by(NAME_ORDER)
    }

    /// Sort by sequence length.
    #[must_use]
    pub fn sort_by_length(self) -> Builder<HasKeyedOrder<LengthOrder>, C, F, CS, S> {
        self.sort_by(LENGTH_ORDER)
    }
}

// Codec selection.

impl<O, F, CS, St> Builder<O, NeedsCodec, F, CS, St> {
    /// Set the dryice codec for temporary file encoding.
    #[must_use]
    pub fn codec<S, Q, N>(
        self,
        codec: DryIceCodec<S, Q, N>,
    ) -> Builder<O, HasCodec<S, Q, N>, F, CS, St> {
        Builder {
            order: self.order,
            codec: HasCodec(codec),
            flush: self.flush,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            storage: self.storage,
            merge_config: self.merge_config,
        }
    }
}

// Flush strategy selection.

impl<O, C, CS, S> Builder<O, C, NeedsFlush, CS, S> {
    /// Flush when estimated memory usage exceeds `budget` bytes.
    /// Requires [`SeqRecord`] to implement `GetSize`.
    #[must_use]
    pub fn measured_budget(self, budget: usize) -> Builder<O, C, HasFlush, CS, S> {
        Builder {
            order: self.order,
            codec: self.codec,
            flush: HasFlush(FlushConfig::MeasuredBudget(budget)),
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            storage: self.storage,
            merge_config: self.merge_config,
        }
    }

    /// Flush when the buffer reaches `max_items` records.
    #[must_use]
    pub fn max_buffer_items(self, max_items: usize) -> Builder<O, C, HasFlush, CS, S> {
        Builder {
            order: self.order,
            codec: self.codec,
            flush: HasFlush(FlushConfig::MaxItems(max_items)),
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            storage: self.storage,
            merge_config: self.merge_config,
        }
    }
}

// Optional configuration.

impl<O, C, F, CS, S> Builder<O, C, F, CS, S> {
    /// Set a custom chunk sorting strategy. Defaults to
    /// [`Sequential`].
    #[must_use]
    pub fn chunk_sort<CS2>(self, chunk_sort: CS2) -> Builder<O, C, F, CS2, S> {
        Builder {
            order: self.order,
            codec: self.codec,
            flush: self.flush,
            dedup: self.dedup,
            chunk_sort,
            storage: self.storage,
            merge_config: self.merge_config,
        }
    }

    /// Override the merge configuration.
    #[must_use]
    pub fn merge_config(mut self, config: MergeConfig) -> Self {
        self.merge_config = config;
        self
    }

    /// Deduplicate adjacent records with identical names.
    #[must_use]
    pub fn dedup_by_name(self) -> Builder<O, C, F, CS, S> {
        Builder {
            order: self.order,
            codec: self.codec,
            flush: self.flush,
            dedup: RecordDedup::Name,
            chunk_sort: self.chunk_sort,
            storage: self.storage,
            merge_config: self.merge_config,
        }
    }

    /// Deduplicate adjacent records with identical sequences.
    #[must_use]
    pub fn dedup_by_sequence(self) -> Builder<O, C, F, CS, S> {
        Builder {
            order: self.order,
            codec: self.codec,
            flush: self.flush,
            dedup: RecordDedup::Sequence,
            chunk_sort: self.chunk_sort,
            storage: self.storage,
            merge_config: self.merge_config,
        }
    }

    /// Deduplicate adjacent records with identical sequence and
    /// quality.
    #[must_use]
    pub fn dedup_by_sequence_and_quality(self) -> Builder<O, C, F, CS, S> {
        Builder {
            order: self.order,
            codec: self.codec,
            flush: self.flush,
            dedup: RecordDedup::SequenceAndQuality,
            chunk_sort: self.chunk_sort,
            storage: self.storage,
            merge_config: self.merge_config,
        }
    }

    /// Use single-threaded comparison sort for in-memory chunks.
    ///
    /// This overrides the default sort engine (which is radix sort
    /// for sequence orders, and sequential for everything else).
    #[must_use]
    pub fn sort_with_sequential(self) -> Builder<O, C, F, Sequential, S> {
        self.chunk_sort(Sequential)
    }

    /// Use rayon parallel sort for in-memory chunks.
    ///
    /// This overrides the default sort engine. Requires the `rayon`
    /// feature to be enabled on spillover.
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn sort_with_parallel(self) -> Builder<O, C, F, spillover::chunk::Parallel, S> {
        self.chunk_sort(spillover::chunk::Parallel)
    }
}

// build() for keyed sort orders.
//
// The `ResolveCodec` and `ResolveFlush` traits handle defaults:
// `NeedsCodec` → raw DryIceCodec, `NeedsFlush` → 1 GiB budget.
// This single impl covers all four combinations of has/needs.

impl<O, C, F, CS> Builder<HasKeyedOrder<O>, C, F, CS, OwnedStorage>
where
    O: KeyedSortOrder,
    O::SortKey: Send + Sync + 'static,
    O::Compare: for<'a> Compare<<O::SortKey as SortKey<SeqRecord>>::Key<'a>>
        + Compare<<O as RecordKeyer>::RecordKey>
        + Send
        + Sync
        + 'static,
    <O as RecordKeyer>::RecordKey: RecordKey + Clone + 'static,
    C: sealed::ResolveCodec,
    F: sealed::ResolveFlush,
    CS: ChunkSorter<SeqRecord>,
{
    /// Build the sorter (keyed merge path).
    ///
    /// If `.codec()` was not called, defaults to raw (uncompressed)
    /// dryice encoding. If `.measured_budget()` / `.max_buffer_items()`
    /// was not called, defaults to a 1 GiB measured memory budget.
    #[allow(clippy::type_complexity)]
    #[must_use]
    pub fn build(
        self,
    ) -> Sorter<
        O::SortKey,
        KeyedDryIceCodec<O, C::S, C::Q, C::N>,
        O::Compare,
        CS,
        spillover::sorter::Keyed,
    > {
        let order = self.order.0;
        let _storage = self.storage;
        let codec = self.codec.resolve();
        let keyed_codec = codec.with_record_keyer(order);
        let flush = self.flush.resolve();

        let builder = spillover::sorter::Builder::new()
            .key(order.sort_key())
            .compare(order.compare())
            .keyed_codec(keyed_codec)
            .dedup(Identity)
            .chunk_sort(self.chunk_sort)
            .merge_config(self.merge_config);

        let inner = match flush {
            FlushConfig::MeasuredBudget(budget) => {
                builder.measured_budget::<SeqRecord>(budget).build()
            }
            FlushConfig::MaxItems(max_items) => {
                builder.max_buffer_items::<SeqRecord>(max_items).build()
            }
        };

        Sorter {
            inner,
            dedup: self.dedup,
        }
    }
}

// build() for unkeyed sort orders.

impl<O, C, F, CS> Builder<HasUnkeyedOrder<O>, C, F, CS, OwnedStorage>
where
    O: SortOrder<Strategy = Basic>,
    O::SortKey: Send + Sync + 'static,
    O::Compare:
        for<'a> Compare<<O::SortKey as SortKey<SeqRecord>>::Key<'a>> + Send + Sync + 'static,
    C: sealed::ResolveCodec,
    F: sealed::ResolveFlush,
    CS: ChunkSorter<SeqRecord>,
{
    /// Build the sorter (basic merge path, no record keys).
    ///
    /// If `.codec()` was not called, defaults to raw (uncompressed)
    /// dryice encoding. If `.measured_budget()` / `.max_buffer_items()`
    /// was not called, defaults to a 1 GiB measured memory budget.
    #[allow(clippy::type_complexity)]
    #[must_use]
    pub fn build(
        self,
    ) -> Sorter<O::SortKey, DryIceCodec<C::S, C::Q, C::N>, O::Compare, CS, spillover::sorter::Basic>
    {
        let order = self.order.0;
        let _storage = self.storage;
        let codec = self.codec.resolve();
        let flush = self.flush.resolve();

        let builder = spillover::sorter::Builder::new()
            .key(order.sort_key())
            .compare(order.compare())
            .codec(codec)
            .dedup(Identity)
            .chunk_sort(self.chunk_sort)
            .merge_config(self.merge_config);

        let inner = match flush {
            FlushConfig::MeasuredBudget(budget) => {
                builder.measured_budget::<SeqRecord>(budget).build()
            }
            FlushConfig::MaxItems(max_items) => {
                builder.max_buffer_items::<SeqRecord>(max_items).build()
            }
        };

        Sorter {
            inner,
            dedup: self.dedup,
        }
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
        let cmp = order.compare();

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
        let cmp = order.compare();

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
    fn unkeyed_sequence_order_has_basic_strategy() {
        fn assert_basic<O: SortOrder<Strategy = Basic>>() {}
        assert_basic::<UnkeyedSequenceOrder>();
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
        fn assert_basic<O: SortOrder<Strategy = Basic>>() {}

        let _unkeyed = ILLUMINA_ORDER.unkeyed();
        assert_basic::<UnkeyedSequenceOrder>();

        let _unkeyed = NameOrder.unkeyed();
        assert_basic::<UnkeyedNameOrder>();

        let _unkeyed = LengthOrder.unkeyed();
        assert_basic::<UnkeyedLengthOrder>();
    }
}
