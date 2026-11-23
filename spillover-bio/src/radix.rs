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

use spillover::{
    chunk::{ChunkSorter, EstimateSortMemory, SortMemory},
    compare::Compare,
};

use crate::{key::PackedSequenceKey, record::SeqRecordParts};

/// In-place MSD radix sort on packed sequence keys, with
/// comparison-based refinement for tiebreaking.
///
/// The const parameter `N` is the packed key width in bytes.
/// Records are sorted by 2-bit-packed sequence bytes first
/// (O(n×N) radix passes), then equal-key groups are refined
/// with the full comparator (quality tiebreaking) via
/// comparison sort.
///
/// This is not a general-purpose comparator sort. It imposes
/// ascending [`PackedSequenceKey`]
/// bucket order before calling the supplied comparator, and the
/// comparator is used only to refine records whose packed keys tie.
/// Use it only when ascending `PackedSequenceKey<N>` is the primary
/// ordering component of the comparator, such as with
/// [`SequenceOrder<N>`](crate::sort::SequenceOrder). It is not
/// compatible with exact raw sequence ordering, reverse sequence
/// ordering, name or length ordering, or arbitrary custom
/// comparators.
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
const BUCKETS: usize = 256;
type BucketTable = [usize; BUCKETS];

struct TemporaryEntry<const N: usize> {
    source: usize,
    key: PackedSequenceKey<N>,
}

impl<const N: usize, R: SeqRecordParts> ChunkSorter<R, crate::sort::SequenceOrder<N>>
    for RadixThenRefine<N>
{
    fn sort(&self, chunk: &mut [R], cmp: &crate::sort::SequenceOrder<N>) {
        Self::sort_impl(chunk, &|left, right| cmp.compare(left, right));
    }
}

impl<'a, const N: usize, I>
    ChunkSorter<
        spillover::key::Keyed<usize, PackedSequenceKey<N>>,
        spillover::key::KeyedEntryCompare<'a, I, crate::sort::SequenceOrder<N>>,
    > for RadixThenRefine<N>
where
    crate::sort::SequenceOrder<N>: Compare<I>,
{
    fn sort(
        &self,
        entries: &mut [spillover::key::Keyed<usize, PackedSequenceKey<N>>],
        cmp: &spillover::key::KeyedEntryCompare<'a, I, crate::sort::SequenceOrder<N>>,
    ) {
        sort_entries(entries, 0, &|entry| entry.stored_key(), &|left, right| {
            cmp.compare(left, right)
        });
    }
}

impl<const N: usize> EstimateSortMemory<spillover::key::Keyed<usize, PackedSequenceKey<N>>>
    for RadixThenRefine<N>
{
    fn sort_memory(&self, count: usize, (): &()) -> SortMemory {
        SortMemory::temporary(Self::partition_workspace(count).unwrap_or(usize::MAX))
    }
}

impl<const N: usize, R: SeqRecordParts> EstimateSortMemory<R> for RadixThenRefine<N> {
    fn sort_memory(&self, count: usize, (): &()) -> SortMemory {
        let bytes = std::alloc::Layout::array::<TemporaryEntry<N>>(count)
            .ok()
            .and_then(|layout| layout.size().checked_add(Self::partition_workspace(count)?))
            .unwrap_or(usize::MAX);
        SortMemory::temporary(bytes)
    }
}

fn sort_entries<const N: usize, E>(
    entries: &mut [E],
    byte_pos: usize,
    key: &impl Fn(&E) -> &PackedSequenceKey<N>,
    compare: &impl Fn(&E, &E) -> Ordering,
) {
    if entries.len() <= INSERTION_THRESHOLD || byte_pos >= N {
        entries.sort_unstable_by(compare);
        return;
    }

    let mut counts: BucketTable = [0; BUCKETS];
    for entry in entries.iter() {
        let bucket = usize::from(key(entry).0[byte_pos]);
        counts[bucket] += 1;
    }

    let mut offsets: BucketTable = [0; BUCKETS];
    let mut running = 0;
    for (offset, count) in offsets.iter_mut().zip(&counts) {
        *offset = running;
        running += count;
    }

    let mut cursors = offsets;
    for bucket in 0..BUCKETS {
        let bucket_end = offsets[bucket] + counts[bucket];
        while cursors[bucket] < bucket_end {
            let position = cursors[bucket];
            let item_bucket = usize::from(key(&entries[position]).0[byte_pos]);
            if item_bucket == bucket {
                cursors[bucket] += 1;
            } else {
                let target = cursors[item_bucket];
                entries.swap(position, target);
                cursors[item_bucket] += 1;
            }
        }
    }

    let mut start = 0;
    for &count in &counts {
        let end = start + count;
        if count > 1 {
            let bucket = &mut entries[start..end];
            sort_entries(bucket, byte_pos + 1, key, compare);
        }
        start = end;
    }
}

fn apply_permutation<const N: usize, R>(records: &mut [R], entries: &mut [TemporaryEntry<N>]) {
    for start in 0..entries.len() {
        let mut destination = start;
        loop {
            let source = entries[destination].source;
            entries[destination].source = destination;
            if source == start {
                break;
            }
            records.swap(destination, source);
            destination = source;
        }
    }
}

impl<const N: usize> RadixThenRefine<N> {
    fn sort_impl<R: SeqRecordParts>(chunk: &mut [R], cmp: &impl Fn(&R, &R) -> Ordering) {
        let mut entries: Vec<TemporaryEntry<N>> = chunk
            .iter()
            .enumerate()
            .map(|(source, record)| {
                let key = PackedSequenceKey::from_sequence(record.sequence());
                TemporaryEntry { source, key }
            })
            .collect();

        let compare = |left: &TemporaryEntry<N>, right: &TemporaryEntry<N>| {
            cmp(&chunk[left.source], &chunk[right.source])
        };
        sort_entries(&mut entries, 0, &|entry| &entry.key, &compare);
        apply_permutation(chunk, &mut entries);
    }

    pub(crate) fn partition_workspace(count: usize) -> Option<usize> {
        if count <= INSERTION_THRESHOLD {
            return Some(0);
        }
        let tables = N.checked_mul(3)?;
        let layout = std::alloc::Layout::array::<BucketTable>(tables).ok()?;
        Some(layout.size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::SeqRecord;
    use crate::sort::SequenceOrder;

    fn make_record(name: &[u8], seq: &[u8], qual: &[u8]) -> SeqRecord {
        SeqRecord::new(name, seq, qual)
    }

    fn seq_qual_cmp(a: &SeqRecord, b: &SeqRecord) -> Ordering {
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

        let order: SequenceOrder<2> = SequenceOrder;
        RadixThenRefine.sort(&mut views, &order);

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
    fn matches_sequence_order_above_threshold() {
        let bases = [b'A', b'C', b'G', b'T', b'N', b'a', b'c', b'x'];
        let mut records: Vec<SeqRecord> = (0..200)
            .map(|i| {
                let seq: Vec<u8> = (0..12)
                    .map(|j| bases[(i * 11 + j * 5 + i / 3) % bases.len()])
                    .collect();
                let qual = vec![b'!' + u8::try_from(i % 40).expect("fits"); 12];
                make_record(format!("r{i:03}").as_bytes(), &seq, &qual)
            })
            .collect();

        let mut expected = records.clone();
        let order = SequenceOrder::<3>;
        let cmp = |a: &SeqRecord, b: &SeqRecord| order.compare(a, b);

        // Use more than INSERTION_THRESHOLD records so this exercises radix
        // buckets rather than only the comparison-sort fallback.
        expected.sort_unstable_by(cmp);
        RadixThenRefine::<3>.sort(&mut records, &order);

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
