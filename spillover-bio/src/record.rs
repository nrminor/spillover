//! Sequence record type for spillover-bio.
//!
//! [`SeqRecord`] is the item type sorted by spillover-bio. It holds
//! owned name, sequence, and quality bytes in one contiguous backing
//! allocation, and implements [`GetSize`] for accurate memory budget
//! tracking.
//!
//! Conversions to and from `dryice::SeqRecord` are provided via
//! `From` impls, so records move between spillover-bio and dryice
//! without manual field copying.

use dryice::SeqRecordLike;
use get_size2::GetSize;

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
/// the current spill window, then resolve private handles back into
/// [`SeqRecordView`] values while flushing that window.
#[derive(Debug, Default)]
pub struct SeqRecordArena {
    bytes: Vec<u8>,
    records: Vec<ArenaRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SeqRecordHandle {
    index: usize,
}

#[derive(Debug, Clone, Copy)]
struct ArenaRecord {
    start: usize,
    name_len: usize,
    sequence_len: usize,
    quality_len: usize,
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

    /// Estimated live bytes owned by the arena for the current spill window.
    pub(crate) fn live_byte_len(&self) -> usize {
        Self::live_byte_len_for(self.byte_len(), self.record_count())
    }

    /// Estimated live arena bytes for a projected spill window.
    pub(crate) fn live_byte_len_for(record_bytes: usize, record_count: usize) -> usize {
        record_bytes.saturating_add(Self::metadata_bytes_for(record_count))
    }

    /// Estimated live bytes used for private per-record boundary metadata.
    pub(crate) fn metadata_bytes_for(record_count: usize) -> usize {
        record_count.saturating_mul(std::mem::size_of::<ArenaRecord>())
    }

    pub(crate) fn record_count(&self) -> usize {
        self.records.len()
    }

    pub(crate) fn store<R: SeqRecordParts + ?Sized>(&mut self, record: &R) -> SeqRecordHandle {
        let start = self.bytes.len();
        let name_len = record.name().len();
        let sequence_len = record.sequence().len();
        let quality_len = record.quality().len();

        self.bytes.extend_from_slice(record.name());
        self.bytes.extend_from_slice(record.sequence());
        self.bytes.extend_from_slice(record.quality());

        let index = self.records.len();
        self.records.push(ArenaRecord {
            start,
            name_len,
            sequence_len,
            quality_len,
        });

        SeqRecordHandle { index }
    }

    pub(crate) fn view(&self, handle: SeqRecordHandle) -> SeqRecordView<'_> {
        let record = self.records[handle.index];
        let name_start = record.start;
        let sequence_start = name_start + record.name_len;
        let quality_start = sequence_start + record.sequence_len;
        let end = quality_start + record.quality_len;

        SeqRecordView::new(
            &self.bytes[name_start..sequence_start],
            &self.bytes[sequence_start..quality_start],
            &self.bytes[quality_start..end],
        )
    }

    pub(crate) fn clear(&mut self) {
        self.bytes.clear();
        self.records.clear();
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

impl From<SeqRecord> for dryice::SeqRecord {
    fn from(rec: SeqRecord) -> Self {
        dryice::SeqRecord::new(
            rec.name().to_vec(),
            rec.sequence().to_vec(),
            rec.quality().to_vec(),
        )
        .expect("spillover-bio SeqRecord fields are valid for dryice")
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
        let dryice_rec: dryice::SeqRecord = rec.clone().into();
        let back: SeqRecord = dryice_rec.into();
        assert_eq!(rec, back);
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

        let first = arena.store(&SeqRecordView::new(b"r1", b"ACGT", b"!!!!"));
        let second = arena.store(&SeqRecordView::new(b"r2", b"TGCA", b"####"));

        assert_eq!(arena.record_count(), 2);
        assert_eq!(arena.byte_len(), 20);

        let first_view = arena.view(first);
        assert_eq!(first_view.name(), b"r1");
        assert_eq!(first_view.sequence(), b"ACGT");
        assert_eq!(first_view.quality(), b"!!!!");

        let second_view = arena.view(second);
        assert_eq!(second_view.name(), b"r2");
        assert_eq!(second_view.sequence(), b"TGCA");
        assert_eq!(second_view.quality(), b"####");
    }

    #[test]
    fn arena_copies_input_bytes() {
        let mut arena = SeqRecordArena::new();
        let mut sequence = b"ACGT".to_vec();

        let handle = arena.store(&SeqRecordView::new(b"r1", &sequence, b"!!!!"));
        sequence.copy_from_slice(b"TGCA");

        assert_eq!(arena.view(handle).sequence(), b"ACGT");
    }

    #[test]
    fn arena_clear_reuses_capacity() {
        let mut arena = SeqRecordArena::with_capacity(128);
        let initial_capacity = arena.capacity();

        arena.store(&SeqRecordView::new(b"r1", b"ACGT", b"!!!!"));
        arena.clear();

        assert!(arena.is_empty());
        assert_eq!(arena.record_count(), 0);
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
