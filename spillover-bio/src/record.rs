//! Sequence record type for spillover-bio.
//!
//! [`SeqRecord`] is the item type sorted by spillover-bio. It
//! holds owned name, sequence, and quality bytes, and implements
//! [`GetSize`] for accurate memory budget tracking.
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
    name: Vec<u8>,
    sequence: Vec<u8>,
    quality: Vec<u8>,
}

impl SeqRecord {
    /// Create a new record from owned byte vectors.
    #[must_use]
    pub fn new(name: Vec<u8>, sequence: Vec<u8>, quality: Vec<u8>) -> Self {
        Self {
            name,
            sequence,
            quality,
        }
    }

    /// The record name/identifier.
    #[must_use]
    pub fn name(&self) -> &[u8] {
        &self.name
    }

    /// The nucleotide sequence.
    #[must_use]
    pub fn sequence(&self) -> &[u8] {
        &self.sequence
    }

    /// The quality scores.
    #[must_use]
    pub fn quality(&self) -> &[u8] {
        &self.quality
    }

    /// Borrow this record as a [`SeqRecordView`].
    #[must_use]
    pub fn as_view(&self) -> SeqRecordView<'_> {
        SeqRecordView::new(self.name(), self.sequence(), self.quality())
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

impl SeqRecordLike for SeqRecord {
    fn name(&self) -> &[u8] {
        &self.name
    }

    fn sequence(&self) -> &[u8] {
        &self.sequence
    }

    fn quality(&self) -> &[u8] {
        &self.quality
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
        Self {
            name: rec.name().to_vec(),
            sequence: rec.sequence().to_vec(),
            quality: rec.quality().to_vec(),
        }
    }
}

impl From<SeqRecord> for dryice::SeqRecord {
    fn from(rec: SeqRecord) -> Self {
        dryice::SeqRecord::new(rec.name, rec.sequence, rec.quality)
            .expect("spillover-bio SeqRecord fields are valid for dryice")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_accessors() {
        let rec = SeqRecord::new(
            b"read1".to_vec(),
            b"ACGTACGT".to_vec(),
            b"!!!!!!!!".to_vec(),
        );
        assert_eq!(rec.name(), b"read1");
        assert_eq!(rec.sequence(), b"ACGTACGT");
        assert_eq!(rec.quality(), b"!!!!!!!!");
    }

    #[test]
    fn round_trip_to_dryice() {
        let rec = SeqRecord::new(b"read1".to_vec(), b"ACGT".to_vec(), b"!!!!".to_vec());
        let dryice_rec: dryice::SeqRecord = rec.clone().into();
        let back: SeqRecord = dryice_rec.into();
        assert_eq!(rec, back);
    }

    #[test]
    fn implements_seq_record_like() {
        let rec = SeqRecord::new(b"r1".to_vec(), b"ACGT".to_vec(), b"!!!!".to_vec());
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
        let rec = SeqRecord::new(b"r1".to_vec(), b"ACGT".to_vec(), b"!!!!".to_vec());
        let view = rec.as_view();

        assert_eq!(view.name(), rec.name());
        assert_eq!(view.sequence(), rec.sequence());
        assert_eq!(view.quality(), rec.quality());
    }

    #[test]
    fn parts_trait_works_for_owned_and_view_records() {
        fn sequence_len<R: SeqRecordParts>(record: &R) -> usize {
            record.sequence_len()
        }

        let rec = SeqRecord::new(b"r1".to_vec(), b"ACGT".to_vec(), b"!!!!".to_vec());
        let view = rec.as_view();

        assert_eq!(sequence_len(&rec), 4);
        assert_eq!(sequence_len(&view), 4);
        assert!(!SeqRecordParts::is_empty(&view));
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
        let rec = SeqRecord::new(
            b"read1".to_vec(),
            b"ACGTACGT".to_vec(),
            b"!!!!!!!!".to_vec(),
        );
        let size = rec.get_size();
        assert!(
            size > std::mem::size_of::<SeqRecord>(),
            "GetSize should account for heap allocations, got {size}"
        );
    }
}
