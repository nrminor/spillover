//! K-way merge of sorted runs from temporary files on disk.
//!
//! This module provides the infrastructure for writing pre-sorted
//! items to temporary files and merging them back into a single
//! sorted stream via a min-heap. When the number of runs exceeds
//! a configurable fan-in limit, intermediate merge passes spill
//! to temporary files automatically.
//!
//! Serialization is handled by the [`Codec`] trait — the merge
//! engine has no opinion about the on-disk format. Ordering is
//! handled by a [`Compare`] implementation wrapped via
//! [`WithOrd`] so that the standard [`BinaryHeap`] works
//! without requiring `Ord` on the item type.

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    io::{BufReader, BufWriter, Seek, SeekFrom, Write},
    num::NonZeroUsize,
    path::PathBuf,
};

use crate::{
    codec::Codec,
    compare::{Compare, WithOrd},
};

/// Errors that can occur during merge operations.
///
/// Generic over `CE`, the codec's associated error type. This
/// enum is internal to the merge module and will be converted
/// at the `Sorter` boundary into whatever error type the user
/// has chosen.
#[derive(Debug)]
pub enum MergeError<CE> {
    /// An I/O error occurred reading or writing a temporary file.
    Io(std::io::Error),

    /// A temporary file ended with a partial record.
    TruncatedEntry,

    /// The codec failed to decode an entry from bytes.
    Codec(CE),
}

impl<CE: std::fmt::Display> std::fmt::Display for MergeError<CE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "merge I/O error: {e}"),
            Self::TruncatedEntry => write!(f, "sorted run ended with a partial record"),
            Self::Codec(e) => write!(f, "failed to decode entry from sorted run: {e}"),
        }
    }
}

impl<CE: std::error::Error + 'static> std::error::Error for MergeError<CE> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::TruncatedEntry => None,
            Self::Codec(e) => Some(e),
        }
    }
}

impl<CE> From<std::io::Error> for MergeError<CE> {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Configuration for the merge engine.
///
/// All fields have sensible defaults via the [`Default`]
/// implementation: 256-way fan-in, 64 KiB read/write buffers,
/// and the OS default temporary directory.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MergeConfig {
    /// Maximum number of sorted runs to merge in a single pass.
    /// When the number of runs exceeds this, intermediate merge
    /// passes spill to temporary files.
    pub max_fan_in: NonZeroUsize,

    /// Size of the read buffer (in bytes) for each run being
    /// merged. Larger buffers reduce syscall overhead at the
    /// cost of memory per open file.
    pub read_buffer_bytes: usize,

    /// Size of the write buffer (in bytes) when spilling sorted
    /// items or intermediate merge results to disk.
    pub write_buffer_bytes: usize,

    /// Directory for temporary files. When `None`, uses the
    /// operating system's default temporary directory.
    pub temp_dir: Option<PathBuf>,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            max_fan_in: NonZeroUsize::new(256).expect("256 is not zero"),
            read_buffer_bytes: 64 * 1024,
            write_buffer_bytes: 64 * 1024,
            temp_dir: None,
        }
    }
}

/// A sorted run that has been written to a temporary file on disk.
///
/// Created by [`RunMerger::spill_sorted`] and consumed by
/// [`RunMerger::merge`].
#[derive(Debug)]
pub struct SortedRun {
    file: std::fs::File,
}

/// Orchestrates the creation and merging of sorted runs on disk.
///
/// Generic over the item type `T`, the codec `C` that serializes
/// items, and the comparator `Cmp` that orders them. The
/// comparator should be a zero-sized type (like [`Natural`] or
/// [`Reverse`]) for best performance — `WithOrd` wrappers in the
/// merge heap will then add no memory overhead.
///
/// [`Natural`]: crate::compare::Natural
/// [`Reverse`]: crate::compare::Reverse
pub struct RunMerger<'c, T, C: Codec<T>, Cmp: Compare<T> + Clone> {
    codec: &'c C,
    cmp: Cmp,
    config: MergeConfig,
    _item: std::marker::PhantomData<fn() -> T>,
}

impl<'c, T, C: Codec<T>, Cmp: Compare<T> + Clone> RunMerger<'c, T, C, Cmp> {
    /// Create a new merger with the given codec, comparator, and
    /// configuration.
    #[must_use]
    pub fn new(codec: &'c C, cmp: Cmp, config: MergeConfig) -> Self {
        Self {
            codec,
            cmp,
            config,
            _item: std::marker::PhantomData,
        }
    }

    /// Write a pre-sorted iterator of items to a temporary file.
    ///
    /// The caller is responsible for ensuring the input is sorted
    /// according to the comparator. In debug builds, ordering is
    /// verified.
    ///
    /// # Errors
    ///
    /// Returns [`MergeError::Io`] if writing to the temporary
    /// file fails, or [`MergeError::Codec`] if encoding fails.
    pub fn spill_sorted(
        &self,
        items: impl IntoIterator<Item = T>,
    ) -> Result<SortedRun, MergeError<C::Error>> {
        let mut file = self.create_temp_file()?;
        let mut writer = BufWriter::with_capacity(self.config.write_buffer_bytes, &mut file);

        #[cfg(debug_assertions)]
        let mut prev: Option<T> = None;

        for item in items {
            #[cfg(debug_assertions)]
            if let Some(ref p) = prev {
                debug_assert!(
                    self.cmp.le(p, &item),
                    "spill_sorted received unsorted input"
                );
            }

            self.codec
                .write(&item, &mut writer)
                .map_err(MergeError::Codec)?;

            #[cfg(debug_assertions)]
            {
                prev = Some(item);
            }
        }

        writer.flush()?;
        drop(writer);
        file.seek(SeekFrom::Start(0))?;

        Ok(SortedRun { file })
    }

    /// Merge multiple sorted runs into a single sorted iterator.
    ///
    /// When the number of runs exceeds [`MergeConfig::max_fan_in`],
    /// intermediate merge passes are performed automatically,
    /// spilling to temporary files as needed.
    ///
    /// Each call to the returned iterator's `next()` may perform
    /// disk I/O and can therefore fail.
    ///
    /// # Errors
    ///
    /// Returns [`MergeError`] if opening or reading a temporary
    /// file fails during heap seeding or intermediate spilling.
    pub fn merge(
        &self,
        runs: Vec<SortedRun>,
    ) -> Result<MergedItems<'_, T, C>, MergeError<C::Error>> {
        let sources: Vec<MergeSource<'_, T, C>> = runs
            .into_iter()
            .map(|run| {
                MergeSource::File(EntryReader::new(
                    self.codec,
                    run.file,
                    self.config.read_buffer_bytes,
                ))
            })
            .collect();

        let merged = self.merge_sources(sources)?;
        Ok(MergedItems { source: merged })
    }

    /// Recursively merge sources, respecting the fan-in limit.
    fn merge_sources<'s>(
        &'s self,
        mut sources: Vec<MergeSource<'s, T, C>>,
    ) -> Result<MergeSource<'s, T, C>, MergeError<C::Error>> {
        if sources.is_empty() {
            return Ok(MergeSource::Heap(Box::new(HeapMerge::empty(
                self.cmp.clone(),
            ))));
        }

        let fan_in = self.config.max_fan_in.get();

        if sources.len() <= fan_in {
            Ok(MergeSource::Heap(Box::new(HeapMerge::new(
                sources,
                self.cmp.clone(),
            )?)))
        } else {
            let mut intermediate = Vec::new();

            while !sources.is_empty() {
                let chunk_end = sources.len().min(fan_in);
                let group: Vec<MergeSource<'_, T, C>> = sources.drain(..chunk_end).collect();
                let heap_merge = HeapMerge::new(group, self.cmp.clone())?;
                let file = self.spill_merge_to_disk(heap_merge)?;
                intermediate.push(MergeSource::File(EntryReader::new(
                    self.codec,
                    file,
                    self.config.read_buffer_bytes,
                )));
            }

            self.merge_sources(intermediate)
        }
    }

    /// Drain a heap merge into a temporary file.
    fn spill_merge_to_disk(
        &self,
        mut merge: HeapMerge<'_, T, C, Cmp>,
    ) -> Result<std::fs::File, MergeError<C::Error>> {
        let mut file = self.create_temp_file()?;
        let mut writer = BufWriter::with_capacity(self.config.write_buffer_bytes, &mut file);

        while let Some(item) = merge.next_item()? {
            self.codec
                .write(&item, &mut writer)
                .map_err(MergeError::Codec)?;
        }

        writer.flush()?;
        drop(writer);
        file.seek(SeekFrom::Start(0))?;
        Ok(file)
    }

    /// Create a temporary file in the configured directory.
    fn create_temp_file(&self) -> Result<std::fs::File, MergeError<C::Error>> {
        let file = match self.config.temp_dir {
            Some(ref dir) => tempfile::tempfile_in(dir)?,
            None => tempfile::tempfile()?,
        };
        Ok(file)
    }
}

/// Iterator over merged sorted items from multiple runs.
///
/// Created by [`RunMerger::merge`]. Each call to `next()` may
/// perform disk I/O.
pub struct MergedItems<'c, T, C: Codec<T>> {
    source: MergeSource<'c, T, C>,
}

impl<T, C: Codec<T>> Iterator for MergedItems<'_, T, C> {
    type Item = Result<T, MergeError<C::Error>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.source.next_item() {
            Ok(Some(item)) => Some(Ok(item)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Reads items sequentially from a sorted temporary file.
struct EntryReader<'c, T, C: Codec<T>> {
    codec: &'c C,
    reader: BufReader<std::fs::File>,
    _item: std::marker::PhantomData<fn() -> T>,
}

impl<'c, T, C: Codec<T>> EntryReader<'c, T, C> {
    fn new(codec: &'c C, file: std::fs::File, read_buffer_bytes: usize) -> Self {
        Self {
            codec,
            reader: BufReader::with_capacity(read_buffer_bytes, file),
            _item: std::marker::PhantomData,
        }
    }

    fn next_item(&mut self) -> Result<Option<T>, MergeError<C::Error>> {
        self.codec.read(&mut self.reader).map_err(MergeError::Codec)
    }
}

/// A source of sorted items: either a file on disk or an
/// in-progress heap merge of other sources.
enum MergeSource<'c, T, C: Codec<T>> {
    File(EntryReader<'c, T, C>),
    Heap(Box<dyn MergeSourceTrait<T, C> + 'c>),
}

impl<T, C: Codec<T>> MergeSource<'_, T, C> {
    #[inline]
    fn next_item(&mut self) -> Result<Option<T>, MergeError<C::Error>> {
        match self {
            Self::File(reader) => reader.next_item(),
            Self::Heap(merge) => merge.next_item(),
        }
    }
}

/// Trait object interface for heap merges, erasing the comparator
/// type from `MergeSource`.
trait MergeSourceTrait<T, C: Codec<T>> {
    fn next_item(&mut self) -> Result<Option<T>, MergeError<C::Error>>;
}

/// A heap entry carrying the item, its source index, and the
/// comparator via [`WithOrd`]. The comparator is a ZST for
/// common cases ([`Natural`], [`Reverse`]), so the entry has
/// no memory overhead beyond the item and index.
///
/// [`Natural`]: crate::compare::Natural
/// [`Reverse`]: crate::compare::Reverse
struct HeapEntry<T, Cmp: Compare<T> + Clone> {
    item: WithOrd<T, Cmp>,
    source_idx: usize,
}

impl<T, Cmp: Compare<T> + Clone> Eq for HeapEntry<T, Cmp> {}

impl<T, Cmp: Compare<T> + Clone> PartialEq for HeapEntry<T, Cmp> {
    fn eq(&self, other: &Self) -> bool {
        self.item == other.item && self.source_idx == other.source_idx
    }
}

impl<T, Cmp: Compare<T> + Clone> Ord for HeapEntry<T, Cmp> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.item
            .cmp(&other.item)
            .then(self.source_idx.cmp(&other.source_idx))
    }
}

impl<T, Cmp: Compare<T> + Clone> PartialOrd for HeapEntry<T, Cmp> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The k-way merge engine. Holds N sources and a min-heap that
/// always contains at most one item per source.
struct HeapMerge<'c, T, C: Codec<T>, Cmp: Compare<T> + Clone> {
    sources: Vec<MergeSource<'c, T, C>>,
    heap: BinaryHeap<Reverse<HeapEntry<T, Cmp>>>,
    cmp: Cmp,
}

impl<'c, T, C: Codec<T>, Cmp: Compare<T> + Clone> HeapMerge<'c, T, C, Cmp> {
    /// Create an empty merge that yields no items.
    fn empty(cmp: Cmp) -> Self {
        Self {
            sources: Vec::new(),
            heap: BinaryHeap::new(),
            cmp,
        }
    }

    /// Seed the heap by reading one item from each source.
    fn new(
        mut sources: Vec<MergeSource<'c, T, C>>,
        cmp: Cmp,
    ) -> Result<Self, MergeError<C::Error>> {
        let mut heap = BinaryHeap::with_capacity(sources.len());

        for (idx, source) in sources.iter_mut().enumerate() {
            if let Some(item) = source.next_item()? {
                heap.push(Reverse(HeapEntry {
                    item: WithOrd::new(item, cmp.clone()),
                    source_idx: idx,
                }));
            }
        }

        Ok(Self { sources, heap, cmp })
    }

    /// Pop the smallest item and advance its source.
    fn next_item(&mut self) -> Result<Option<T>, MergeError<C::Error>> {
        let Some(Reverse(entry)) = self.heap.pop() else {
            return Ok(None);
        };

        if let Some(next_item) = self.sources[entry.source_idx].next_item()? {
            self.heap.push(Reverse(HeapEntry {
                item: WithOrd::new(next_item, self.cmp.clone()),
                source_idx: entry.source_idx,
            }));
        }

        Ok(Some(entry.item.into_inner()))
    }
}

impl<T, C: Codec<T>, Cmp: Compare<T> + Clone> MergeSourceTrait<T, C> for HeapMerge<'_, T, C, Cmp> {
    fn next_item(&mut self) -> Result<Option<T>, MergeError<C::Error>> {
        self.next_item()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;
    use crate::compare::Natural;

    struct U64Codec;

    impl Codec<u64> for U64Codec {
        type Error = std::io::Error;

        fn write(&self, item: &u64, writer: &mut impl Write) -> Result<(), Self::Error> {
            writer.write_all(&item.to_le_bytes())
        }

        fn read(&self, reader: &mut impl Read) -> Result<Option<u64>, Self::Error> {
            let mut buf = [0u8; 8];
            match reader.read(&mut buf[..1]) {
                Ok(0) => Ok(None),
                Ok(_) => {
                    reader.read_exact(&mut buf[1..])?;
                    Ok(Some(u64::from_le_bytes(buf)))
                }
                Err(e) => Err(e),
            }
        }
    }

    fn default_merger(codec: &U64Codec) -> RunMerger<'_, u64, U64Codec, Natural> {
        RunMerger::new(codec, Natural, MergeConfig::default())
    }

    #[test]
    fn spill_and_merge_single_run() {
        let codec = U64Codec;
        let merger = default_merger(&codec);
        let run = merger
            .spill_sorted(vec![1u64, 3, 5, 7, 9])
            .expect("spilling should succeed");

        let results: Vec<u64> = merger
            .merge(vec![run])
            .expect("merging should succeed")
            .map(|r| r.expect("reading should succeed"))
            .collect();

        assert_eq!(results, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn merge_two_interleaved_runs() {
        let codec = U64Codec;
        let merger = default_merger(&codec);
        let a = merger.spill_sorted(vec![1u64, 3, 5]).expect("spill A");
        let b = merger.spill_sorted(vec![2u64, 4, 6]).expect("spill B");

        let results: Vec<u64> = merger
            .merge(vec![a, b])
            .expect("merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn merge_preserves_duplicates_across_runs() {
        let codec = U64Codec;
        let merger = default_merger(&codec);
        let a = merger.spill_sorted(vec![1u64, 3, 5]).expect("spill");
        let b = merger.spill_sorted(vec![1u64, 3, 7]).expect("spill");

        let results: Vec<u64> = merger
            .merge(vec![a, b])
            .expect("merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 1, 3, 3, 5, 7]);
    }

    #[test]
    fn merge_empty_run_list() {
        let codec = U64Codec;
        let merger = default_merger(&codec);
        let results: Vec<u64> = merger
            .merge(vec![])
            .expect("merge zero runs")
            .map(|r| r.expect("read"))
            .collect();

        assert!(results.is_empty());
    }

    #[test]
    fn merge_single_empty_run() {
        let codec = U64Codec;
        let merger = default_merger(&codec);
        let run = merger
            .spill_sorted(std::iter::empty::<u64>())
            .expect("spill empty");

        let results: Vec<u64> = merger
            .merge(vec![run])
            .expect("merge")
            .map(|r| r.expect("read"))
            .collect();

        assert!(results.is_empty());
    }

    #[test]
    fn merge_three_runs() {
        let codec = U64Codec;
        let merger = default_merger(&codec);
        let a = merger.spill_sorted(vec![1u64, 4, 7]).expect("spill");
        let b = merger.spill_sorted(vec![2u64, 5, 8]).expect("spill");
        let c = merger.spill_sorted(vec![3u64, 6, 9]).expect("spill");

        let results: Vec<u64> = merger
            .merge(vec![a, b, c])
            .expect("merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn bounded_fan_in_triggers_intermediate_spill() {
        let codec = U64Codec;
        let config = MergeConfig {
            max_fan_in: NonZeroUsize::new(2).expect("2 is not zero"),
            ..MergeConfig::default()
        };
        let merger = RunMerger::new(&codec, Natural, config);

        let a = merger.spill_sorted(vec![1u64, 4]).expect("spill");
        let b = merger.spill_sorted(vec![2u64, 5]).expect("spill");
        let c = merger.spill_sorted(vec![3u64, 6]).expect("spill");

        let results: Vec<u64> = merger
            .merge(vec![a, b, c])
            .expect("merge fan-in=2")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn merge_many_runs_with_small_fan_in() {
        let codec = U64Codec;
        let config = MergeConfig {
            max_fan_in: NonZeroUsize::new(3).expect("3 is not zero"),
            ..MergeConfig::default()
        };
        let merger = RunMerger::new(&codec, Natural, config);

        let runs: Vec<SortedRun> = (0..10)
            .map(|i: u64| {
                let start = i * 3;
                merger
                    .spill_sorted(vec![start, start + 1, start + 2])
                    .expect("spill")
            })
            .collect();

        let results: Vec<u64> = merger
            .merge(runs)
            .expect("merge 10 runs")
            .map(|r| r.expect("read"))
            .collect();

        let expected: Vec<u64> = (0..30).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn merge_different_sized_runs() {
        let codec = U64Codec;
        let merger = default_merger(&codec);
        let a = merger.spill_sorted(vec![1u64]).expect("spill");
        let b = merger.spill_sorted(vec![2u64, 3, 4, 5, 6]).expect("spill");
        let c = merger.spill_sorted(vec![7u64, 8]).expect("spill");

        let results: Vec<u64> = merger
            .merge(vec![a, b, c])
            .expect("merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn merge_config_default_has_sensible_values() {
        let config = MergeConfig::default();
        assert_eq!(config.max_fan_in.get(), 256);
        assert_eq!(config.read_buffer_bytes, 64 * 1024);
        assert_eq!(config.write_buffer_bytes, 64 * 1024);
        assert!(config.temp_dir.is_none());
    }

    #[test]
    fn custom_temp_dir_works() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let codec = U64Codec;
        let config = MergeConfig {
            temp_dir: Some(temp_dir.path().to_path_buf()),
            ..MergeConfig::default()
        };
        let merger = RunMerger::new(&codec, Natural, config);

        let run = merger.spill_sorted(vec![1u64, 2, 3]).expect("spill");
        let results: Vec<u64> = merger
            .merge(vec![run])
            .expect("merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn nonexistent_temp_dir_returns_io_error() {
        let codec = U64Codec;
        let config = MergeConfig {
            temp_dir: Some(PathBuf::from("/nonexistent/path/should/not/exist")),
            ..MergeConfig::default()
        };
        let merger = RunMerger::new(&codec, Natural, config);

        let result = merger.spill_sorted(vec![1u64, 2, 3]);
        assert!(
            matches!(result, Err(MergeError::Io(_))),
            "nonexistent temp dir should produce Io error, got: {result:?}"
        );
    }

    mod proptests {
        use proptest::prelude::*;

        use super::*;

        fn arb_sorted_u64_vec() -> impl Strategy<Value = Vec<u64>> {
            proptest::collection::vec(0u64..1_000, 0..50).prop_map(|mut v| {
                v.sort_unstable();
                v
            })
        }

        fn spill_runs(
            merger: &RunMerger<'_, u64, U64Codec, Natural>,
            batches: &[Vec<u64>],
        ) -> Vec<SortedRun> {
            batches
                .iter()
                .map(|b| {
                    merger
                        .spill_sorted(b.iter().copied())
                        .expect("spill in proptest")
                })
                .collect()
        }

        fn collect_merged(
            merger: &RunMerger<'_, u64, U64Codec, Natural>,
            runs: Vec<SortedRun>,
        ) -> Vec<u64> {
            merger
                .merge(runs)
                .expect("merge in proptest")
                .map(|r| r.expect("read in proptest"))
                .collect()
        }

        proptest! {
            #[test]
            fn merged_output_is_always_sorted(
                batches in proptest::collection::vec(arb_sorted_u64_vec(), 0..6),
            ) {
                let codec = U64Codec;
                let merger = RunMerger::new(&codec, Natural, MergeConfig::default());
                let runs = spill_runs(&merger, &batches);
                let results = collect_merged(&merger, runs);

                prop_assert!(
                    results.windows(2).all(|w| w[0] <= w[1]),
                    "merged output must be sorted, got: {results:?}"
                );
            }

            #[test]
            fn merge_preserves_total_entry_count(
                batches in proptest::collection::vec(arb_sorted_u64_vec(), 0..6),
            ) {
                let codec = U64Codec;
                let merger = RunMerger::new(&codec, Natural, MergeConfig::default());
                let total_input: usize = batches.iter().map(Vec::len).sum();
                let runs = spill_runs(&merger, &batches);
                let output_count = collect_merged(&merger, runs).len();

                prop_assert_eq!(total_input, output_count);
            }

            #[test]
            fn fan_in_does_not_affect_merge_output(
                batches in proptest::collection::vec(arb_sorted_u64_vec(), 2..8),
            ) {
                let codec = U64Codec;
                let merger_wide = RunMerger::new(&codec, Natural, MergeConfig::default());
                let merger_narrow = RunMerger::new(&codec, Natural, MergeConfig {
                    max_fan_in: NonZeroUsize::new(2).expect("2 is not zero"),
                    ..MergeConfig::default()
                });

                let results_wide = collect_merged(
                    &merger_wide,
                    spill_runs(&merger_wide, &batches),
                );
                let results_narrow = collect_merged(
                    &merger_narrow,
                    spill_runs(&merger_narrow, &batches),
                );

                prop_assert_eq!(results_wide, results_narrow);
            }

            #[test]
            fn merge_output_matches_reference_sort(
                batches in proptest::collection::vec(arb_sorted_u64_vec(), 0..6),
            ) {
                let codec = U64Codec;
                let merger = RunMerger::new(&codec, Natural, MergeConfig::default());

                let mut reference: Vec<u64> = batches.iter().flatten().copied().collect();
                reference.sort_unstable();

                let runs = spill_runs(&merger, &batches);
                let results = collect_merged(&merger, runs);

                prop_assert_eq!(results, reference);
            }
        }
    }
}
