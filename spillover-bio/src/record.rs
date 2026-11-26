//! Sequence record type for spillover-bio.
//!
//! [`SeqRecord`] is the item type sorted by spillover-bio. It holds
//! owned name, sequence, and quality bytes in one contiguous backing
//! allocation, and implements [`GetSize`] for accurate memory budget
//! tracking.
//!
//! Prefer passing borrowed records directly to dryice writers through
//! [`SeqRecordLike`]. If an owned dryice record is needed, prefer
//! [`dryice::SeqRecord::from_slices`] for explicit construction that leaves the
//! source available. [`TryFrom<SeqRecord>`] is also provided as a consuming
//! convenience; it copies the fields and rejects unequal sequence and quality
//! lengths. Conversion from a dryice record uses [`From`] and copies its fields.

use std::alloc::Layout;

use dryice::SeqRecordLike;
use get_size2::GetSize;
use spillover::sorter::{
    Buffer, BufferMemory, ChunkBuffer, EstimateBufferMemory, ProjectBufferMemory,
};

/// Two caller-supplied mates kept together in R1, R2 order.
///
/// Construction does not check biological identity or discover missing mates.
#[derive(Debug, Clone, PartialEq, Eq, GetSize)]
pub struct ReadPair<R> {
    r1: R,
    r2: R,
}

impl<R> ReadPair<R> {
    /// Assemble two mates without validation.
    #[must_use]
    pub fn from_mates(r1: R, r2: R) -> Self {
        Self { r1, r2 }
    }

    /// Borrow the first mate.
    #[must_use]
    pub fn r1(&self) -> &R {
        &self.r1
    }

    /// Borrow the second mate.
    #[must_use]
    pub fn r2(&self) -> &R {
        &self.r2
    }

    /// Recover both mates in their original order.
    #[must_use]
    pub fn into_mates(self) -> (R, R) {
        (self.r1, self.r2)
    }

    /// Validate this pair using application-defined rules.
    ///
    /// # Errors
    ///
    /// Returns the error produced by the supplied validator.
    pub fn validate_with<E>(&self, validate: impl FnOnce(&Self) -> Result<(), E>) -> Result<(), E> {
        validate(self)
    }
}

/// Read-only access to sequence record fields.
///
/// This trait lets generic code work with both owned records and
/// borrowed record views without caring where the bytes live.
pub trait SeqRecordParts {
    /// The record name/identifier.
    fn name(&self) -> &[u8];

    /// The nucleotide sequence.
    fn sequence(&self) -> &[u8];

    /// The quality scores.
    fn quality(&self) -> &[u8];

    /// The nucleotide sequence length.
    fn sequence_len(&self) -> usize {
        self.sequence().len()
    }

    /// Whether the nucleotide sequence is empty.
    fn is_empty(&self) -> bool {
        self.sequence().is_empty()
    }

    /// Whether this record has the same name as another record-shaped value.
    fn has_same_name_as<R: SeqRecordParts + ?Sized>(&self, other: &R) -> bool {
        self.name() == other.name()
    }

    /// Whether this record has the same nucleotide sequence as another
    /// record-shaped value.
    fn has_same_sequence_as<R: SeqRecordParts + ?Sized>(&self, other: &R) -> bool {
        self.sequence() == other.sequence()
    }

    /// Whether this record has the same nucleotide sequence and quality scores
    /// as another record-shaped value.
    fn has_same_sequence_and_quality_as<R: SeqRecordParts + ?Sized>(&self, other: &R) -> bool {
        self.sequence() == other.sequence() && self.quality() == other.quality()
    }
}

/// A borrowed sequence record view.
///
/// `SeqRecordView` owns no bytes. It is a small value whose fields
/// borrow name, sequence, and quality slices from some other storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqRecordView<'a> {
    name: &'a [u8],
    sequence: &'a [u8],
    quality: &'a [u8],
}

impl<'a> SeqRecordView<'a> {
    /// Create a borrowed record view.
    #[must_use]
    pub fn new(name: &'a [u8], sequence: &'a [u8], quality: &'a [u8]) -> Self {
        Self {
            name,
            sequence,
            quality,
        }
    }

    /// The record name/identifier.
    #[must_use]
    pub fn name(&self) -> &'a [u8] {
        self.name
    }

    /// The nucleotide sequence.
    #[must_use]
    pub fn sequence(&self) -> &'a [u8] {
        self.sequence
    }

    /// The quality scores.
    #[must_use]
    pub fn quality(&self) -> &'a [u8] {
        self.quality
    }
}

/// An owned sequence record with name, sequence, and quality data.
///
/// This is the primary item type for spillover-bio's sorter. It
/// mirrors `dryice::SeqRecord` but is owned by this crate to
/// decouple the public API from dryice's semver.
#[derive(Debug, Clone, PartialEq, Eq, GetSize)]
pub struct SeqRecord {
    bytes: Box<[u8]>,
    name_len: usize,
    sequence_len: usize,
}

/// Reusable byte storage for arena-backed sequence record buffering.
///
/// `SeqRecordArena` is storage, not a record type. It lets arena-backed sorter
/// internals copy incoming record-shaped values into one reusable byte buffer for
/// the current spill window, then use record metadata to borrow
/// [`SeqRecordView`] values while flushing that window.
#[derive(Debug, Default)]
pub struct SeqRecordArena {
    bytes: Vec<u8>,
    records: Vec<ArenaRecord>,
}

#[derive(Debug, Clone, Copy)]
struct ArenaRecord {
    start: usize,
    name_len: usize,
    sequence_len: usize,
    quality_len: usize,
}

fn growth_target<T>(items: &Vec<T>, required: usize) -> Option<usize> {
    let layout = Layout::array::<T>(required).ok()?;
    let capacity = items.capacity();
    if layout.size() == 0 || required <= capacity {
        return Some(required);
    }
    let doubled = capacity.checked_mul(2).unwrap_or(required);
    let requested = doubled.max(required);
    if Layout::array::<T>(requested).is_ok() {
        Some(requested)
    } else {
        Some(required)
    }
}

impl SeqRecord {
    /// Create a new record by copying field bytes into one
    /// contiguous backing allocation.
    #[must_use]
    pub fn new(
        name: impl AsRef<[u8]>,
        sequence: impl AsRef<[u8]>,
        quality: impl AsRef<[u8]>,
    ) -> Self {
        Self::from_slices(name.as_ref(), sequence.as_ref(), quality.as_ref())
    }

    /// Create a new record by copying borrowed field slices into one
    /// contiguous backing allocation.
    #[must_use]
    pub fn from_slices(name: &[u8], sequence: &[u8], quality: &[u8]) -> Self {
        let name_len = name.len();
        let sequence_len = sequence.len();

        let mut bytes = Vec::with_capacity(name.len() + sequence.len() + quality.len());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(sequence);
        bytes.extend_from_slice(quality);

        Self {
            bytes: bytes.into_boxed_slice(),
            name_len,
            sequence_len,
        }
    }

    /// The record name/identifier.
    #[must_use]
    pub fn name(&self) -> &[u8] {
        &self.bytes[..self.name_len]
    }

    /// The nucleotide sequence.
    #[must_use]
    pub fn sequence(&self) -> &[u8] {
        let start = self.name_len;
        let end = start + self.sequence_len;
        &self.bytes[start..end]
    }

    /// The quality scores.
    #[must_use]
    pub fn quality(&self) -> &[u8] {
        let start = self.name_len + self.sequence_len;
        &self.bytes[start..]
    }

    /// Borrow this record as a [`SeqRecordView`].
    #[must_use]
    pub fn as_view(&self) -> SeqRecordView<'_> {
        SeqRecordView::new(self.name(), self.sequence(), self.quality())
    }
}

impl SeqRecordArena {
    /// Create an empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty arena with space for at least `bytes` record bytes.
    #[must_use]
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            records: Vec::new(),
        }
    }

    /// Number of record bytes currently stored in the arena.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Current byte capacity retained by the arena.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    /// Whether the arena currently stores no record bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub(crate) fn store<R: SeqRecordParts + ?Sized>(&mut self, record: &R) {
        let start = self.bytes.len();
        let name = record.name();
        let sequence = record.sequence();
        let quality = record.quality();
        let name_len = name.len();
        let sequence_len = sequence.len();
        let quality_len = quality.len();
        let payload_len = start
            .checked_add(name_len)
            .and_then(|len| len.checked_add(sequence_len))
            .and_then(|len| len.checked_add(quality_len))
            .expect("capacity overflow");
        let item_count = self
            .records
            .len()
            .checked_add(1)
            .expect("capacity overflow");
        let payload_target = growth_target(&self.bytes, payload_len).expect("capacity overflow");
        let metadata_target = growth_target(&self.records, item_count).expect("capacity overflow");

        if payload_target > self.bytes.capacity() {
            self.bytes.reserve_exact(payload_target - self.bytes.len());
        }
        self.bytes.extend_from_slice(name);
        self.bytes.extend_from_slice(sequence);
        self.bytes.extend_from_slice(quality);

        if metadata_target > self.records.capacity() {
            self.records
                .reserve_exact(metadata_target - self.records.len());
        }

        self.records.push(ArenaRecord {
            start,
            name_len,
            sequence_len,
            quality_len,
        });
    }

    pub(crate) fn views(&self) -> Vec<SeqRecordView<'_>> {
        let mut views = Vec::with_capacity(self.records.len());
        views.extend(self.records.iter().map(|record| {
            let name_start = record.start;
            let sequence_start = name_start + record.name_len;
            let quality_start = sequence_start + record.sequence_len;
            let end = quality_start + record.quality_len;

            SeqRecordView::new(
                &self.bytes[name_start..sequence_start],
                &self.bytes[sequence_start..quality_start],
                &self.bytes[quality_start..end],
            )
        }));
        views
    }

    fn memory_for_lengths(&self, payload_bytes: Option<usize>, item_count: usize) -> BufferMemory {
        let costs = (|| {
            let payload_bytes = payload_bytes?;
            let payload_target = growth_target(&self.bytes, payload_bytes)?;
            let metadata_target = growth_target(&self.records, item_count)?;
            let metadata_item_bytes = std::mem::size_of::<ArenaRecord>();
            let metadata_before = self.records.len().checked_mul(metadata_item_bytes)?;
            let metadata_after = item_count.checked_mul(metadata_item_bytes)?;
            let payload_storage = if payload_target > self.bytes.capacity() {
                self.bytes.capacity().checked_add(payload_target)?
            } else {
                payload_bytes
            };
            let payload_phase = payload_storage.checked_add(metadata_before)?;
            let metadata_storage = if metadata_target > self.records.capacity() {
                let overlapping_slots = self.records.capacity().checked_add(metadata_target)?;
                overlapping_slots.checked_mul(metadata_item_bytes)?
            } else {
                metadata_after
            };
            let metadata_phase = payload_bytes.checked_add(metadata_storage)?;
            let insertion_peak_bytes = payload_phase.max(metadata_phase);
            let view_bytes = Layout::array::<SeqRecordView<'_>>(item_count).ok()?.size();
            let spill_peak_bytes = payload_bytes
                .checked_add(metadata_after)?
                .checked_add(view_bytes)?;
            Some((insertion_peak_bytes, spill_peak_bytes))
        })();
        match costs {
            Some((insertion, spill)) => BufferMemory::new(item_count, insertion, spill),
            None => BufferMemory::new(item_count, usize::MAX, usize::MAX),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.bytes.clear();
        self.records.clear();
    }
}

impl Buffer for SeqRecordArena {
    fn len(&self) -> usize {
        self.records.len()
    }
}

impl EstimateBufferMemory for SeqRecordArena {
    fn memory(&self) -> BufferMemory {
        self.memory_for_lengths(Some(self.bytes.len()), self.records.len())
    }
}

impl<R: SeqRecordParts + ?Sized> ProjectBufferMemory<&R> for SeqRecordArena {
    fn memory_after(&self, record: &&R) -> BufferMemory {
        let payload_bytes = self
            .bytes
            .len()
            .checked_add(record.name().len())
            .and_then(|bytes| bytes.checked_add(record.sequence().len()))
            .and_then(|bytes| bytes.checked_add(record.quality().len()));
        let item_count = self.records.len().saturating_add(1);
        self.memory_for_lengths(payload_bytes, item_count)
    }
}

impl<'input, R: SeqRecordParts + ?Sized> Extend<&'input R> for SeqRecordArena {
    fn extend<I: IntoIterator<Item = &'input R>>(&mut self, inputs: I) {
        for record in inputs {
            self.store(record);
        }
    }
}

impl<'items> ChunkBuffer for &'items mut SeqRecordArena {
    type Item = SeqRecordView<'items>;
    type Items = Vec<Self::Item>;

    fn items(self) -> Self::Items {
        self.views()
    }

    fn clear(self) {
        SeqRecordArena::clear(self);
    }
}

impl SeqRecordParts for SeqRecord {
    fn name(&self) -> &[u8] {
        self.name()
    }

    fn sequence(&self) -> &[u8] {
        self.sequence()
    }

    fn quality(&self) -> &[u8] {
        self.quality()
    }
}

impl SeqRecordParts for SeqRecordView<'_> {
    fn name(&self) -> &[u8] {
        self.name()
    }

    fn sequence(&self) -> &[u8] {
        self.sequence()
    }

    fn quality(&self) -> &[u8] {
        self.quality()
    }
}

impl From<SeqRecordView<'_>> for SeqRecord {
    fn from(record: SeqRecordView<'_>) -> Self {
        Self::from_slices(record.name(), record.sequence(), record.quality())
    }
}

impl SeqRecordLike for SeqRecord {
    fn name(&self) -> &[u8] {
        self.name()
    }

    fn sequence(&self) -> &[u8] {
        self.sequence()
    }

    fn quality(&self) -> &[u8] {
        self.quality()
    }
}

impl SeqRecordLike for SeqRecordView<'_> {
    fn name(&self) -> &[u8] {
        self.name()
    }

    fn sequence(&self) -> &[u8] {
        self.sequence()
    }

    fn quality(&self) -> &[u8] {
        self.quality()
    }
}

impl From<dryice::SeqRecord> for SeqRecord {
    fn from(rec: dryice::SeqRecord) -> Self {
        Self::from_slices(rec.name(), rec.sequence(), rec.quality())
    }
}

impl TryFrom<SeqRecord> for dryice::SeqRecord {
    type Error = dryice::DryIceError;

    /// Copy the fields into an owned dryice record, consuming the source even
    /// when conversion fails.
    ///
    /// # Errors
    ///
    /// Returns [`dryice::DryIceError::MismatchedSequenceAndQualityLengths`] when
    /// sequence and quality lengths differ.
    fn try_from(record: SeqRecord) -> Result<Self, Self::Error> {
        dryice::SeqRecord::from_slices(record.name(), record.sequence(), record.quality())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_accessors() {
        let rec = SeqRecord::new(b"read1", b"ACGTACGT", b"!!!!!!!!");
        assert_eq!(rec.name(), b"read1");
        assert_eq!(rec.sequence(), b"ACGTACGT");
        assert_eq!(rec.quality(), b"!!!!!!!!");
    }

    #[test]
    fn round_trip_to_dryice() {
        let rec = SeqRecord::new(b"read1", b"ACGT", b"!!!!");
        let dryice_rec: dryice::SeqRecord = rec
            .clone()
            .try_into()
            .expect("matching sequence and quality lengths");
        let back: SeqRecord = dryice_rec.into();
        assert_eq!(rec, back);
    }

    #[test]
    fn conversion_to_dryice_rejects_mismatched_lengths() {
        for (sequence, quality) in [
            (b"AC".as_slice(), b"!".as_slice()),
            (b"A".as_slice(), b"!!".as_slice()),
            (b"AC".as_slice(), b"".as_slice()),
        ] {
            let record = SeqRecord::new(b"read", sequence, quality);
            let result = dryice::SeqRecord::try_from(record);
            match result {
                Err(dryice::DryIceError::MismatchedSequenceAndQualityLengths {
                    sequence_len,
                    quality_len,
                }) => {
                    assert_eq!(sequence_len, sequence.len());
                    assert_eq!(quality_len, quality.len());
                }
                other => panic!("expected mismatched lengths, got {other:?}"),
            }
        }
    }

    #[test]
    fn empty_record_round_trips_through_dryice() {
        let record = SeqRecord::new(b"", b"", b"");
        let converted = dryice::SeqRecord::try_from(record.clone())
            .expect("empty sequence and quality have matching lengths");
        let recovered = SeqRecord::from(converted);
        assert_eq!(recovered, record);
    }

    #[test]
    fn implements_seq_record_like() {
        let rec = SeqRecord::new(b"r1", b"ACGT", b"!!!!");
        // SeqRecordLike is used by dryice writers.
        let like: &dyn SeqRecordLike = &rec;
        assert_eq!(like.sequence(), b"ACGT");
    }

    #[test]
    fn record_view_borrows_fields() {
        let view = SeqRecordView::new(b"r1", b"ACGT", b"!!!!");

        assert_eq!(view.name(), b"r1");
        assert_eq!(view.sequence(), b"ACGT");
        assert_eq!(view.quality(), b"!!!!");
    }

    #[test]
    fn seq_record_as_view_borrows_owned_record() {
        let rec = SeqRecord::new(b"r1", b"ACGT", b"!!!!");
        let view = rec.as_view();

        assert_eq!(view.name(), rec.name());
        assert_eq!(view.sequence(), rec.sequence());
        assert_eq!(view.quality(), rec.quality());
    }

    #[test]
    fn record_view_converts_to_owned_compact_record() {
        let view = SeqRecordView::new(b"r1", b"ACGT", b"!!!!");
        let rec = SeqRecord::from(view);

        assert_eq!(rec.name(), b"r1");
        assert_eq!(rec.sequence(), b"ACGT");
        assert_eq!(rec.quality(), b"!!!!");
    }

    #[test]
    fn arena_stores_records_and_returns_views() {
        let mut arena = SeqRecordArena::new();

        arena.store(&SeqRecordView::new(b"r1", b"ACGT", b"!!!!"));
        arena.store(&SeqRecordView::new(b"r2", b"TGCA", b"####"));

        assert_eq!(arena.len(), 2);
        assert_eq!(arena.byte_len(), 20);

        let views = arena.views();
        let first_view = &views[0];
        assert_eq!(first_view.name(), b"r1");
        assert_eq!(first_view.sequence(), b"ACGT");
        assert_eq!(first_view.quality(), b"!!!!");

        let second_view = &views[1];
        assert_eq!(second_view.name(), b"r2");
        assert_eq!(second_view.sequence(), b"TGCA");
        assert_eq!(second_view.quality(), b"####");
    }

    #[test]
    fn arena_copies_input_bytes() {
        let mut arena = SeqRecordArena::new();
        let mut sequence = b"ACGT".to_vec();

        arena.store(&SeqRecordView::new(b"r1", &sequence, b"!!!!"));
        sequence.copy_from_slice(b"TGCA");

        let views = arena.views();
        assert_eq!(views[0].sequence(), b"ACGT");
    }

    #[test]
    fn arena_clear_reuses_capacity() {
        let mut arena = SeqRecordArena::with_capacity(128);
        let initial_capacity = arena.capacity();

        arena.store(&SeqRecordView::new(b"r1", b"ACGT", b"!!!!"));
        arena.clear();

        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.capacity(), initial_capacity);
    }

    #[test]
    fn parts_trait_works_for_owned_and_view_records() {
        fn sequence_len<R: SeqRecordParts>(record: &R) -> usize {
            record.sequence_len()
        }

        let rec = SeqRecord::new(b"r1", b"ACGT", b"!!!!");
        let view = rec.as_view();

        assert_eq!(sequence_len(&rec), 4);
        assert_eq!(sequence_len(&view), 4);
        assert!(!SeqRecordParts::is_empty(&view));
    }

    #[test]
    fn parts_trait_compares_record_fields() {
        let record = SeqRecord::new(b"r1", b"ACGT", b"!!!!");
        let same_sequence = SeqRecordView::new(b"r2", b"ACGT", b"####");
        let same_name = SeqRecordView::new(b"r1", b"TGCA", b"!!!!");
        let same_sequence_quality = SeqRecordView::new(b"r3", b"ACGT", b"!!!!");

        assert!(record.has_same_sequence_as(&same_sequence));
        assert!(!record.has_same_name_as(&same_sequence));
        assert!(record.has_same_name_as(&same_name));
        assert!(record.has_same_sequence_and_quality_as(&same_sequence_quality));
        assert!(!record.has_same_sequence_and_quality_as(&same_sequence));
    }

    #[test]
    fn record_view_implements_seq_record_like() {
        let view = SeqRecordView::new(b"r1", b"ACGT", b"!!!!");
        let like: &dyn SeqRecordLike = &view;

        assert_eq!(like.name(), b"r1");
        assert_eq!(like.sequence(), b"ACGT");
        assert_eq!(like.quality(), b"!!!!");
    }

    #[test]
    fn get_size_accounts_for_heap() {
        let rec = SeqRecord::new(b"read1", b"ACGTACGT", b"!!!!!!!!");
        let size = rec.get_size();
        assert!(
            size > std::mem::size_of::<SeqRecord>(),
            "GetSize should account for heap allocations, got {size}"
        );
    }
}
