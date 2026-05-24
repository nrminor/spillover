//! In-place MSD radix sort for genomic sequence records.
//!
//! [`RadixThenRefine`] implements spillover's [`ChunkSorter`]
//! trait using an in-place most-significant-digit radix sort on
//! packed sequence key bytes, followed by a comparison-based
//! refinement pass for tiebreaking within equal-key groups.
//!
//! The radix sort operates via swaps (`slice::swap`), not copies,
//! so it works on non-`Copy` record-shaped values without extra
//! allocation.

use std::cmp::Ordering;

use spillover::chunk::ChunkSorter;

use crate::record::SeqRecordParts;

/// In-place MSD radix sort on packed sequence keys, with
/// comparison-based refinement for tiebreaking.
///
/// The const parameter `N` is the packed key width in bytes.
/// Records are sorted by 2-bit-packed sequence bytes first
/// (O(n×N) radix passes), then equal-key groups are refined
/// with the full comparator (quality tiebreaking) via
/// comparison sort.
///
/// For small groups (below the insertion threshold), the radix
/// sort falls through to comparison sort directly, avoiding
/// the overhead of counting and partitioning for tiny slices.
#[derive(Debug, Clone, Copy)]
pub struct RadixThenRefine<const N: usize>;

/// Groups smaller than this are sorted by comparison instead
/// of continuing the radix recursion. Tuned for the cost of
/// a 256-bucket counting pass vs. a short comparison sort.
const INSERTION_THRESHOLD: usize = 64;

impl<const N: usize, R: SeqRecordParts> ChunkSorter<R> for RadixThenRefine<N> {
    fn sort(&self, chunk: &mut [R], cmp: impl Fn(&R, &R) -> Ordering + Send + Sync) {
        Self::sort_impl(chunk, &cmp);
    }
}

/// Inner radix sort with const generic key width.
fn msd_radix_sort_inner<const N: usize, R: SeqRecordParts>(
    records: &mut [R],
    keys: &mut [[u8; N]],
    byte_pos: usize,
    cmp: &impl Fn(&R, &R) -> Ordering,
) {
    if records.len() <= INSERTION_THRESHOLD || byte_pos >= N {
        // Small group or exhausted key bytes: comparison sort
        // handles tiebreaking (quality) and final ordering.
        records.sort_unstable_by(|a, b| cmp(a, b));
        return;
    }

    // Count occurrences of each byte value at this position.
    let mut counts = [0usize; 256];
    for key in keys.iter() {
        counts[key[byte_pos] as usize] += 1;
    }

    // Compute bucket start offsets (prefix sum).
    let mut offsets = [0usize; 256];
    let mut running = 0;
    for i in 0..256 {
        offsets[i] = running;
        running += counts[i];
    }

    // In-place permutation via cycle sort. For each bucket, walk
    // its assigned range and swap any out-of-place items to their
    // correct bucket until all items are in the right place.
    let mut cursors = offsets;
    for bucket in 0..256 {
        let bucket_end = offsets[bucket] + counts[bucket];
        while cursors[bucket] < bucket_end {
            let item_bucket = keys[cursors[bucket]][byte_pos] as usize;
            if item_bucket == bucket {
                cursors[bucket] += 1;
            } else {
                let target = cursors[item_bucket];
                records.swap(cursors[bucket], target);
                keys.swap(cursors[bucket], target);
                cursors[item_bucket] += 1;
            }
        }
    }

    // Recurse on each non-trivial bucket.
    let mut start = 0;
    for &count in &counts {
        let end = start + count;
        if end - start > 1 {
            msd_radix_sort_inner(
                &mut records[start..end],
                &mut keys[start..end],
                byte_pos + 1,
                cmp,
            );
        }
        start = end;
    }
}

// Fix the ChunkSorter impl to call msd_radix_sort_inner directly.
impl<const N: usize> RadixThenRefine<N> {
    fn sort_impl<R: SeqRecordParts>(chunk: &mut [R], cmp: &impl Fn(&R, &R) -> Ordering) {
        let mut keys: Vec<[u8; N]> = chunk
            .iter()
            .map(|rec| crate::key::PackedSequenceKey::<N>::from_sequence(rec.sequence()).0)
            .collect();

        msd_radix_sort_inner(chunk, &mut keys, 0, cmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{SeqRecord, SeqRecordView};

    fn make_record(name: &[u8], seq: &[u8], qual: &[u8]) -> SeqRecord {
        SeqRecord::new(name, seq, qual)
    }

    fn seq_qual_cmp(a: &SeqRecord, b: &SeqRecord) -> Ordering {
        a.sequence()
            .cmp(b.sequence())
            .then_with(|| a.quality().cmp(b.quality()))
    }

    fn view_seq_qual_cmp(a: &SeqRecordView<'_>, b: &SeqRecordView<'_>) -> Ordering {
        a.sequence()
            .cmp(b.sequence())
            .then_with(|| a.quality().cmp(b.quality()))
    }

    #[test]
    fn sorts_by_sequence() {
        let mut records = vec![
            make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
            make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
            make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
        ];

        RadixThenRefine::<2>::sort_impl(&mut records, &seq_qual_cmp);

        assert_eq!(records[0].sequence(), b"AAAAAAAA");
        assert_eq!(records[1].sequence(), b"CCCCCCCC");
        assert_eq!(records[2].sequence(), b"TTTTTTTT");
    }

    #[test]
    fn sorts_record_views() {
        let records = [
            make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
            make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
            make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
        ];
        let mut views: Vec<_> = records.iter().map(SeqRecord::as_view).collect();

        RadixThenRefine::<2>.sort(&mut views, view_seq_qual_cmp);

        assert_eq!(views[0].sequence(), b"AAAAAAAA");
        assert_eq!(views[1].sequence(), b"CCCCCCCC");
        assert_eq!(views[2].sequence(), b"TTTTTTTT");
    }

    #[test]
    fn handles_quality_tiebreaker() {
        let mut records = vec![
            make_record(b"r1", b"ACGTACGT", b"!!!!!!!!"),
            make_record(b"r2", b"ACGTACGT", b"IIIIIIII"),
            make_record(b"r3", b"AAAAAAAA", b"!!!!!!!!"),
        ];

        RadixThenRefine::<2>::sort_impl(&mut records, &seq_qual_cmp);

        assert_eq!(records[0].sequence(), b"AAAAAAAA");
        assert_eq!(records[1].quality(), b"!!!!!!!!");
        assert_eq!(records[2].quality(), b"IIIIIIII");
    }

    #[test]
    fn empty_slice() {
        let mut records: Vec<SeqRecord> = vec![];
        RadixThenRefine::<2>::sort_impl(&mut records, &seq_qual_cmp);
        assert!(records.is_empty());
    }

    #[test]
    fn single_record() {
        let mut records = vec![make_record(b"r1", b"ACGT", b"!!!!")];
        RadixThenRefine::<2>::sort_impl(&mut records, &seq_qual_cmp);
        assert_eq!(records[0].sequence(), b"ACGT");
    }

    #[test]
    fn already_sorted() {
        let mut records = vec![
            make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
            make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
            make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
        ];

        RadixThenRefine::<2>::sort_impl(&mut records, &seq_qual_cmp);

        assert_eq!(records[0].sequence(), b"AAAAAAAA");
        assert_eq!(records[1].sequence(), b"CCCCCCCC");
        assert_eq!(records[2].sequence(), b"TTTTTTTT");
    }

    #[test]
    fn all_identical_sequences() {
        let mut records = vec![
            make_record(b"r1", b"ACGTACGT", b"IIIIIIII"),
            make_record(b"r2", b"ACGTACGT", b"!!!!!!!!"),
            make_record(b"r3", b"ACGTACGT", b"########"),
        ];

        RadixThenRefine::<2>::sort_impl(&mut records, &seq_qual_cmp);

        // Quality tiebreaker: ! < # < I in ASCII
        assert_eq!(records[0].quality(), b"!!!!!!!!");
        assert_eq!(records[1].quality(), b"########");
        assert_eq!(records[2].quality(), b"IIIIIIII");
    }

    #[test]
    fn matches_comparison_sort() {
        // Generate diverse records and verify radix sort matches
        // standard comparison sort.
        let bases = [b'A', b'C', b'G', b'T'];
        let mut records: Vec<SeqRecord> = (0..200)
            .map(|i| {
                let seq: Vec<u8> = (0..16).map(|j| bases[(i * 7 + j * 13) % 4]).collect();
                let qual = vec![b'!' + u8::try_from(i % 40).expect("fits"); 16];
                make_record(format!("r{i}").as_bytes(), &seq, &qual)
            })
            .collect();

        let mut expected = records.clone();
        expected.sort_by(seq_qual_cmp);

        RadixThenRefine::<4>::sort_impl(&mut records, &seq_qual_cmp);

        for (i, (got, exp)) in records.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.sequence(),
                exp.sequence(),
                "sequence mismatch at position {i}"
            );
            assert_eq!(
                got.quality(),
                exp.quality(),
                "quality mismatch at position {i}"
            );
        }
    }

    #[test]
    fn large_dataset_matches_comparison_sort() {
        let bases = [b'A', b'C', b'G', b'T'];
        let mut records: Vec<SeqRecord> = (0..1000)
            .map(|i| {
                let seq: Vec<u8> = (0..32)
                    .map(|j| bases[(i * 3 + j * 17 + i / 4) % 4])
                    .collect();
                let qual = vec![b'!' + u8::try_from(i % 40).expect("fits"); 32];
                make_record(format!("r{i}").as_bytes(), &seq, &qual)
            })
            .collect();

        let mut expected = records.clone();
        expected.sort_by(seq_qual_cmp);

        RadixThenRefine::<8>::sort_impl(&mut records, &seq_qual_cmp);

        for (i, (got, exp)) in records.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got.sequence(),
                exp.sequence(),
                "sequence mismatch at position {i}"
            );
            assert_eq!(
                got.quality(),
                exp.quality(),
                "quality mismatch at position {i}"
            );
        }
    }
}
