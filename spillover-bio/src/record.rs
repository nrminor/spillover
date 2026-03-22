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
