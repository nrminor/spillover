//! K-way merge of sorted runs from temporary files on disk.
//!
//! This module provides the infrastructure for writing pre-sorted
//! items to temporary files and merging them back into a single
//! sorted stream via a min-heap. When the number of runs exceeds
//! a configurable fan-in limit, intermediate merge passes spill
//! to temporary files automatically.
//!
//! The merge engine supports two paths:
//! - base path: heap holds full records
//! - keyed path: heap holds compact keys and falls back to
//!   full-record comparison when keys tie
//!
//! Both paths share fan-in recursion and intermediate spilling.

use std::{cmp::Reverse, collections::BinaryHeap, num::NonZeroUsize, path::PathBuf};

use crate::{
    SortedItemsError,
    codec::{
        Codec, CodecCursor, CodecWriter, DeriveKey, KeyedCodec, KeyedCodecCursor, KeyedCodecWriter,
    },
    compare::{Compare, WithOrd},
    sorter::{VisitSortedItems, sealed},
};

/// Errors that can occur during merge operations.
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

/// Shorthand for results carrying a [`MergeError`].
pub type MergeResult<T, CE> = Result<T, MergeError<CE>>;

/// Configuration for the merge engine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MergeConfig {
    /// Maximum number of sorted runs to merge in a single pass.
    pub max_fan_in: NonZeroUsize,

    /// Size of the read buffer (in bytes) for each run being merged.
    pub read_buffer_bytes: usize,

    /// Size of the write buffer (in bytes) when spilling.
    pub write_buffer_bytes: usize,

    /// Directory for temporary files. `None` uses the OS default.
    pub temp_dir: Option<PathBuf>,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            max_fan_in: NonZeroUsize::new(128).expect("128 is not zero"),
            read_buffer_bytes: 64 * 1024,
            write_buffer_bytes: 64 * 1024,
            temp_dir: None,
        }
    }
}

/// A sorted run that has been written to a temporary file on disk.
/// The file handle is closed after writing; the path is retained
/// for reopening during the merge. The temp file is automatically
/// deleted when the `SortedRun` is dropped.
#[derive(Debug)]
pub struct SortedRun {
    pub(crate) path: tempfile::TempPath,
}

impl SortedRun {
    /// Reopen the temp file for reading.
    fn reopen(&self) -> std::io::Result<std::fs::File> {
        std::fs::File::open(&self.path)
    }
}

// ── MergeReader trait ────────────────────────────────────

/// Abstracts how the merge engine reads from sorted runs.
///
/// For the basic path, the heap holds full records and
/// [`output`](Self::output) returns the record directly.
/// For the keyed path, the heap holds compact keys and
/// `output` fetches the full record from the reader only
/// for the merge winner.
pub(crate) trait MergeReader {
    /// What goes in the merge heap (full record or compact key).
    type HeapItem;

    /// What the merge emits (always the full record).
    type Output;

    /// The error type from the underlying codec.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Read the next heap item from this source.
    /// Returns `None` at clean EOF.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or decoding fails.
    fn next(&mut self) -> Result<Option<Self::HeapItem>, Self::Error>;

    /// Convert a popped heap item into the output value.
    /// For the basic path this returns the item itself.
    /// For the keyed path this fetches the full record
    /// from the reader.
    ///
    /// # Errors
    ///
    /// Returns an error if fetching the record fails.
    fn output(&mut self, heap_item: Self::HeapItem) -> Result<Self::Output, Self::Error>;
}

/// Basic merge reader: heap holds full records.
pub(crate) struct BasicMergeReader<T, C: Codec<Item = T>> {
    reader: C::Cursor<std::fs::File>,
    _item: std::marker::PhantomData<fn() -> T>,
}

impl<T, C: Codec<Item = T>> BasicMergeReader<T, C> {
    pub fn new(codec: C, file: std::fs::File) -> Self {
        Self {
            reader: codec.cursor(file),
            _item: std::marker::PhantomData,
        }
    }
}

impl<T, C> MergeReader for BasicMergeReader<T, C>
where
    C: Codec<Item = T>,
{
    type HeapItem = T;
    type Output = T;
    type Error = C::Error;

    fn next(&mut self) -> Result<Option<T>, C::Error> {
        if self.reader.advance()? {
            self.reader.current().map(Some)
        } else {
            Ok(None)
        }
    }

    fn output(&mut self, heap_item: T) -> Result<T, C::Error> {
        Ok(heap_item)
    }
}

/// Keyed merge reader: heap holds compact keys, records
/// fetched on demand for the winner only.
pub(crate) struct KeyedMergeReader<T, C: KeyedCodec<Item = T>> {
    reader: C::KeyedCursor<std::fs::File>,
    _item: std::marker::PhantomData<fn() -> T>,
}

impl<T, C: KeyedCodec<Item = T>> KeyedMergeReader<T, C> {
    pub fn new(codec: C, file: std::fs::File) -> Self {
        Self {
            reader: codec.keyed_cursor(file),
            _item: std::marker::PhantomData,
        }
    }

    fn current(&mut self) -> Result<T, C::Error> {
        self.reader.current()
    }

    fn with_current<'a, R>(
        &'a mut self,
        f: impl FnOnce(<C::KeyedCursor<std::fs::File> as CodecCursor<T>>::Current<'a>) -> R,
    ) -> Result<R, C::Error> {
        self.reader.with_current(f)
    }
}

impl<T, C> MergeReader for KeyedMergeReader<T, C>
where
    C: KeyedCodec<Item = T>,
{
    type HeapItem = C::Key;
    type Output = T;
    type Error = C::Error;

    fn next(&mut self) -> Result<Option<C::Key>, C::Error> {
        if self.reader.advance()? {
            self.reader.current_key().map(Some)
        } else {
            Ok(None)
        }
    }

    fn output(&mut self, _key: C::Key) -> Result<T, C::Error> {
        self.reader.current()
    }
}

// ── RunMerger ────────────────────────────────────────────

/// Orchestrates the creation and merging of sorted runs on disk.
pub struct RunMerger<T, C: Codec<Item = T>, Cmp: Compare<T> + Copy> {
    codec: C,
    cmp: Cmp,
    config: MergeConfig,
    _item: std::marker::PhantomData<fn() -> T>,
}

/// Writes sorted entries into temporary run files.
pub(crate) struct RunWriter<C, M> {
    codec: C,
    config: MergeConfig,
    _mode: std::marker::PhantomData<fn() -> M>,
}

pub(crate) struct Unkeyed;

pub(crate) struct Keyed;

impl<C> RunWriter<C, Unkeyed> {
    pub(crate) fn new_unkeyed(codec: C, config: MergeConfig) -> Self {
        Self {
            codec,
            config,
            _mode: std::marker::PhantomData,
        }
    }
}

impl<C> RunWriter<C, Keyed> {
    pub(crate) fn new_keyed(codec: C, config: MergeConfig) -> Self {
        Self {
            codec,
            config,
            _mode: std::marker::PhantomData,
        }
    }
}

impl<C, M> RunWriter<C, M> {
    fn create_temp_file<CE>(&self) -> MergeResult<tempfile::NamedTempFile, CE> {
        let file = match self.config.temp_dir {
            Some(ref dir) => tempfile::NamedTempFile::new_in(dir)?,
            None => tempfile::NamedTempFile::new()?,
        };
        Ok(file)
    }
}

impl<C> RunWriter<C, Unkeyed>
where
    C: Codec + Copy,
{
    pub(crate) fn write_sorted<T>(&self, items: &[T]) -> MergeResult<SortedRun, C::Error>
    where
        C: Codec<Item = T>,
        for<'a> C::Writer<&'a mut std::fs::File>: CodecWriter<T, Error = C::Error>,
    {
        let named = self.create_temp_file()?;
        let mut file = named.reopen().map_err(MergeError::Io)?;
        let mut writer = self.codec.writer(&mut file);

        for item in items {
            writer.write(item).map_err(MergeError::Codec)?;
        }

        writer.finish().map_err(MergeError::Codec)?;
        drop(file);

        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }
}

impl<C> RunWriter<C, Keyed>
where
    C: KeyedCodec + Copy,
{
    pub(crate) fn write_sorted<T>(&self, items: &[T]) -> MergeResult<SortedRun, C::Error>
    where
        C: KeyedCodec<Item = T> + DeriveKey<T>,
        for<'a> C::KeyedWriter<&'a mut std::fs::File>:
            KeyedCodecWriter<T, C::Key, Error = C::Error>,
    {
        let named = self.create_temp_file()?;
        let mut file = named.reopen().map_err(MergeError::Io)?;
        let mut writer = self.codec.keyed_writer(&mut file);

        for item in items {
            let key = self.codec.derive_key(item);
            writer.write_keyed(item, &key).map_err(MergeError::Codec)?;
        }

        writer.finish().map_err(MergeError::Codec)?;
        drop(file);

        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }
}

impl<T: 'static, C: Codec<Item = T> + Copy + 'static, Cmp: Compare<T> + Copy + 'static>
    RunMerger<T, C, Cmp>
{
    /// Create a new merger.
    #[must_use]
    pub fn new(codec: C, cmp: Cmp, config: MergeConfig) -> Self {
        Self {
            codec,
            cmp,
            config,
            _item: std::marker::PhantomData,
        }
    }

    fn create_temp_file(&self) -> MergeResult<tempfile::NamedTempFile, C::Error> {
        let file = match self.config.temp_dir {
            Some(ref dir) => tempfile::NamedTempFile::new_in(dir)?,
            None => tempfile::NamedTempFile::new()?,
        };
        Ok(file)
    }
}

impl<T: 'static, C: Codec<Item = T> + Copy + 'static, Cmp: Compare<T> + Copy + 'static>
    RunMerger<T, C, Cmp>
where
    for<'a> C::Writer<&'a mut std::fs::File>: CodecWriter<T, Error = C::Error>,
{
    /// Write a pre-sorted iterator of items to a temporary file.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn spill_sorted(
        &self,
        items: impl IntoIterator<Item = T>,
    ) -> Result<SortedRun, MergeError<C::Error>> {
        let named = self.create_temp_file()?;
        let mut file = named.reopen().map_err(MergeError::Io)?;
        let mut writer = self.codec.writer(&mut file);

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

            writer.write(&item).map_err(MergeError::Codec)?;

            #[cfg(debug_assertions)]
            {
                prev = Some(item);
            }
        }

        writer.finish().map_err(MergeError::Codec)?;
        drop(file); // close the write handle

        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }

    /// Merge sorted runs using the basic path (full record
    /// deserialization for comparison).
    ///
    /// When there are more runs than [`MergeConfig::max_fan_in`],
    /// intermediate merge passes spill to temp files automatically.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or merging fails.
    pub fn merge(&self, mut runs: Vec<SortedRun>) -> MergeResult<RunMerge<T, C, Cmp>, C::Error> {
        let codec = self.codec;
        let cmp = self.cmp;
        let fan_in = self.config.max_fan_in.get();

        while runs.len() > fan_in {
            let mut intermediate = Vec::new();
            while !runs.is_empty() {
                let chunk_end = runs.len().min(fan_in);
                let group: Vec<SortedRun> = runs.drain(..chunk_end).collect();
                let readers = open_basic_readers(group, codec)?;
                let mut heap = HeapMerge::new(readers, cmp)?;
                let run = self.spill_heap_to_disk(&mut heap, codec)?;
                intermediate.push(run);
            }
            runs = intermediate;
        }

        Ok(RunMerge::new(runs, codec, cmp))
    }

    /// Drain a heap merge into a temp file via the base codec writer.
    fn spill_heap_to_disk<
        MR: MergeReader<Output = T, Error = C::Error>,
        HCmp: Compare<MR::HeapItem> + Copy,
    >(
        &self,
        heap: &mut HeapMerge<MR, HCmp>,
        codec: C,
    ) -> MergeResult<SortedRun, C::Error> {
        let named = self.create_temp_file()?;
        let mut file = named.reopen().map_err(MergeError::Io)?;
        let mut writer = codec.writer(&mut file);
        while let Some(item) = heap.next_output()? {
            writer.write(&item).map_err(MergeError::Codec)?;
        }
        writer.finish().map_err(MergeError::Codec)?;
        drop(file);
        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }
}

// Keyed merge — only available when the codec supports keys.
impl<T: 'static, C: KeyedCodec<Item = T> + Copy + 'static, Cmp: Compare<T> + Copy + 'static>
    RunMerger<T, C, Cmp>
where
    C: DeriveKey<T>,
    for<'a> C::KeyedWriter<&'a mut std::fs::File>: KeyedCodecWriter<T, C::Key, Error = C::Error>,
{
    /// Merge sorted runs using the keyed path (key-only
    /// comparison, full deserialization only for the winner).
    ///
    /// When there are more runs than [`MergeConfig::max_fan_in`],
    /// intermediate merge passes spill to temp files automatically.
    /// Intermediate files are written with record keys so the next
    /// merge level can use keyed comparison.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or merging fails.
    pub fn merge_keyed<KeyCmp: Compare<C::Key> + Copy>(
        &self,
        mut runs: Vec<SortedRun>,
        key_cmp: KeyCmp,
    ) -> MergeResult<KeyedRunMerge<T, C, KeyCmp, Cmp>, C::Error> {
        let codec = self.codec;
        let fan_in = self.config.max_fan_in.get();

        while runs.len() > fan_in {
            let mut intermediate = Vec::new();
            while !runs.is_empty() {
                let chunk_end = runs.len().min(fan_in);
                let group: Vec<SortedRun> = runs.drain(..chunk_end).collect();

                let readers = open_keyed_readers(group, codec)?;
                let mut heap = KeyedHeapMerge::new(readers, key_cmp, self.cmp)?;
                let run = self.spill_keyed_heap_to_disk(&mut heap, codec)?;
                intermediate.push(run);
            }
            runs = intermediate;
        }

        Ok(KeyedRunMerge::new(runs, codec, key_cmp, self.cmp))
    }

    /// Drain a keyed heap merge into a temp file, re-deriving
    /// record keys so the next merge level can use keyed comparison.
    fn spill_keyed_heap_to_disk<KeyCmp: Compare<C::Key> + Copy>(
        &self,
        heap: &mut KeyedHeapMerge<T, C, KeyCmp, Cmp>,
        codec: C,
    ) -> MergeResult<SortedRun, C::Error> {
        let named = self.create_temp_file()?;
        let mut file = named.reopen().map_err(MergeError::Io)?;
        let mut writer = codec.keyed_writer(&mut file);
        while let Some(item) = heap.next_output()? {
            let key = codec.derive_key(&item);
            writer.write_keyed(&item, &key).map_err(MergeError::Codec)?;
        }
        writer.finish().map_err(MergeError::Codec)?;
        drop(file);
        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }
}

// ── Reader construction helpers ──────────────────────────

fn open_basic_readers<T, C: Codec<Item = T> + Copy>(
    runs: Vec<SortedRun>,
    codec: C,
) -> MergeResult<Vec<BasicMergeReader<T, C>>, C::Error> {
    runs.into_iter()
        .map(|run| {
            let file = run.reopen().map_err(MergeError::Io)?;
            Ok(BasicMergeReader::new(codec, file))
        })
        .collect()
}

fn open_keyed_readers<T, C: KeyedCodec<Item = T> + Copy>(
    runs: Vec<SortedRun>,
    codec: C,
) -> MergeResult<Vec<KeyedMergeReader<T, C>>, C::Error> {
    runs.into_iter()
        .map(|run| {
            let file = run.reopen().map_err(MergeError::Io)?;
            Ok(KeyedMergeReader::new(codec, file))
        })
        .collect()
}

// ── Heap merge engine ────────────────────────────────────

struct HeapEntry<MR: MergeReader, Cmp: Compare<MR::HeapItem> + Copy> {
    item: WithOrd<MR::HeapItem, Cmp>,
    source_idx: usize,
}

impl<MR: MergeReader, Cmp: Compare<MR::HeapItem> + Copy> Eq for HeapEntry<MR, Cmp> {}

impl<MR: MergeReader, Cmp: Compare<MR::HeapItem> + Copy> PartialEq for HeapEntry<MR, Cmp> {
    fn eq(&self, other: &Self) -> bool {
        self.item == other.item && self.source_idx == other.source_idx
    }
}

impl<MR: MergeReader, Cmp: Compare<MR::HeapItem> + Copy> Ord for HeapEntry<MR, Cmp> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.item
            .cmp(&other.item)
            .then(self.source_idx.cmp(&other.source_idx))
    }
}

impl<MR: MergeReader, Cmp: Compare<MR::HeapItem> + Copy> PartialOrd for HeapEntry<MR, Cmp> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The k-way merge engine. Holds N readers and a min-heap
/// with at most one entry per reader.
struct HeapMerge<MR: MergeReader, Cmp: Compare<MR::HeapItem> + Copy> {
    readers: Vec<MR>,
    heap: BinaryHeap<Reverse<HeapEntry<MR, Cmp>>>,
    cmp: Cmp,
}

impl<MR: MergeReader, Cmp: Compare<MR::HeapItem> + Copy> HeapMerge<MR, Cmp> {
    /// Seed the heap by reading one item from each reader.
    fn new(mut readers: Vec<MR>, cmp: Cmp) -> Result<Self, MergeError<MR::Error>> {
        let mut heap = BinaryHeap::with_capacity(readers.len());

        for (idx, reader) in readers.iter_mut().enumerate() {
            if let Some(item) = reader.next().map_err(MergeError::Codec)? {
                heap.push(Reverse(HeapEntry {
                    item: WithOrd::new(item, cmp),
                    source_idx: idx,
                }));
            }
        }

        Ok(Self { readers, heap, cmp })
    }

    /// Pop the smallest entry, produce the output, and advance.
    fn next_output(&mut self) -> Result<Option<MR::Output>, MergeError<MR::Error>> {
        let Some(Reverse(entry)) = self.heap.pop() else {
            return Ok(None);
        };

        let output = self.readers[entry.source_idx]
            .output(entry.item.into_inner())
            .map_err(MergeError::Codec)?;

        if let Some(next) = self.readers[entry.source_idx]
            .next()
            .map_err(MergeError::Codec)?
        {
            self.heap.push(Reverse(HeapEntry {
                item: WithOrd::new(next, self.cmp),
                source_idx: entry.source_idx,
            }));
        }

        Ok(Some(output))
    }
}

/// Keyed k-way merge engine. Heap compares record keys, and when
/// keys tie across runs it falls back to full-record comparison.
struct KeyedHeapMerge<
    T,
    C: KeyedCodec<Item = T>,
    KeyCmp: Compare<C::Key> + Copy,
    ItemCmp: Compare<T> + Copy,
> {
    readers: Vec<KeyedMergeReader<T, C>>,
    heap: BinaryHeap<Reverse<HeapEntry<KeyedMergeReader<T, C>, KeyCmp>>>,
    key_cmp: KeyCmp,
    item_cmp: ItemCmp,
}

impl<T, C, KeyCmp, ItemCmp> KeyedHeapMerge<T, C, KeyCmp, ItemCmp>
where
    C: KeyedCodec<Item = T>,
    KeyCmp: Compare<C::Key> + Copy,
    ItemCmp: Compare<T> + Copy,
{
    fn new(
        mut readers: Vec<KeyedMergeReader<T, C>>,
        key_cmp: KeyCmp,
        item_cmp: ItemCmp,
    ) -> Result<Self, MergeError<C::Error>> {
        let mut heap = BinaryHeap::with_capacity(readers.len());

        for (idx, reader) in readers.iter_mut().enumerate() {
            if let Some(key) = reader.next().map_err(MergeError::Codec)? {
                heap.push(Reverse(HeapEntry {
                    item: WithOrd::new(key, key_cmp),
                    source_idx: idx,
                }));
            }
        }

        Ok(Self {
            readers,
            heap,
            key_cmp,
            item_cmp,
        })
    }

    fn next_output(&mut self) -> Result<Option<T>, MergeError<C::Error>> {
        let Some(Reverse(first)) = self.heap.pop() else {
            return Ok(None);
        };

        if !self.has_key_tie(&first) {
            let source_idx = first.source_idx;
            let output = self.readers[source_idx]
                .output(first.item.into_inner())
                .map_err(MergeError::Codec)?;

            self.advance_source(source_idx)?;

            return Ok(Some(output));
        }

        let tied = self.collect_key_ties(first);
        let (winner, winner_record) = self.select_tied_winner(&tied)?;
        let winner_source = tied[winner].source_idx;

        self.reinsert_tied_losers(tied, winner);
        self.advance_source(winner_source)?;

        Ok(Some(winner_record))
    }

    fn has_key_tie(&self, first: &HeapEntry<KeyedMergeReader<T, C>, KeyCmp>) -> bool {
        self.heap
            .peek()
            .is_some_and(|Reverse(next)| self.key_cmp.eq(next.item.as_ref(), first.item.as_ref()))
    }

    fn collect_key_ties(
        &mut self,
        first: HeapEntry<KeyedMergeReader<T, C>, KeyCmp>,
    ) -> Vec<HeapEntry<KeyedMergeReader<T, C>, KeyCmp>> {
        let mut tied = vec![first];
        while let Some(Reverse(peek)) = self.heap.peek() {
            if self.key_cmp.eq(peek.item.as_ref(), tied[0].item.as_ref()) {
                let Reverse(entry) = self
                    .heap
                    .pop()
                    .expect("heap.peek returned Some but pop failed");
                tied.push(entry);
            } else {
                break;
            }
        }

        tied
    }

    fn select_tied_winner(
        &mut self,
        tied: &[HeapEntry<KeyedMergeReader<T, C>, KeyCmp>],
    ) -> Result<(usize, T), MergeError<C::Error>> {
        let mut winner = 0usize;
        let mut winner_record = self.readers[tied[0].source_idx]
            .current()
            .map_err(MergeError::Codec)?;

        for (idx, entry) in tied.iter().enumerate().skip(1) {
            let candidate = self.readers[entry.source_idx]
                .current()
                .map_err(MergeError::Codec)?;
            let ordering = self.item_cmp.compare(&candidate, &winner_record);

            if ordering.is_lt() || (ordering.is_eq() && entry.source_idx < tied[winner].source_idx)
            {
                winner = idx;
                winner_record = candidate;
            }
        }

        Ok((winner, winner_record))
    }

    fn reinsert_tied_losers(
        &mut self,
        tied: Vec<HeapEntry<KeyedMergeReader<T, C>, KeyCmp>>,
        winner: usize,
    ) {
        for (idx, entry) in tied.into_iter().enumerate() {
            if idx != winner {
                self.heap.push(Reverse(entry));
            }
        }
    }

    fn advance_source(&mut self, source_idx: usize) -> Result<(), MergeError<C::Error>> {
        if let Some(next_key) = self.readers[source_idx].next().map_err(MergeError::Codec)? {
            self.heap.push(Reverse(HeapEntry {
                item: WithOrd::new(next_key, self.key_cmp),
                source_idx,
            }));
        }

        Ok(())
    }

    fn visit_next<F, FE>(
        &mut self,
        mut visit_item: F,
    ) -> Result<bool, SortedItemsError<MergeError<C::Error>, FE>>
    where
        F: for<'a> FnMut(
            <C::KeyedCursor<std::fs::File> as CodecCursor<T>>::Current<'a>,
        ) -> Result<(), FE>,
    {
        let Some(Reverse(first)) = self.heap.pop() else {
            return Ok(false);
        };

        if !self.has_key_tie(&first) {
            let source_idx = first.source_idx;
            self.readers[source_idx]
                .with_current(&mut visit_item)
                .map_err(MergeError::Codec)
                .map_err(SortedItemsError::Source)?
                .map_err(SortedItemsError::Sink)?;

            self.advance_source(source_idx)
                .map_err(SortedItemsError::Source)?;

            return Ok(true);
        }

        let tied = self.collect_key_ties(first);
        let (winner, winner_record) = self
            .select_tied_winner(&tied)
            .map_err(SortedItemsError::Source)?;
        let winner_source = tied[winner].source_idx;

        self.reinsert_tied_losers(tied, winner);

        drop(winner_record);

        self.readers[winner_source]
            .with_current(visit_item)
            .map_err(MergeError::Codec)
            .map_err(SortedItemsError::Source)?
            .map_err(SortedItemsError::Sink)?;

        self.advance_source(winner_source)
            .map_err(SortedItemsError::Source)?;

        Ok(true)
    }
}

/// Lazy merge over sorted runs spilled to disk.
///
/// `RunMerge` owns the final sorted runs and constructs merge state when it is
/// consumed. Owned iteration materializes items through the codec.
pub struct RunMerge<T, C: Codec<Item = T>, Cmp: Compare<T> + Copy> {
    runs: Option<Vec<SortedRun>>,
    heap: Option<BasicHeap<T, C, Cmp>>,
    codec: C,
    cmp: Cmp,
    failed: bool,
}

type BasicHeap<T, C, Cmp> = HeapMerge<BasicMergeReader<T, C>, Cmp>;

impl<T, C, Cmp> RunMerge<T, C, Cmp>
where
    C: Codec<Item = T> + Copy,
    Cmp: Compare<T> + Copy,
{
    fn new(runs: Vec<SortedRun>, codec: C, cmp: Cmp) -> Self {
        Self {
            runs: Some(runs),
            heap: None,
            codec,
            cmp,
            failed: false,
        }
    }

    fn heap(&mut self) -> Result<&mut BasicHeap<T, C, Cmp>, MergeError<C::Error>> {
        if self.heap.is_none() {
            let runs = self.runs.take().expect("runs are initialized at most once");
            let readers = open_basic_readers(runs, self.codec)?;
            self.heap = Some(HeapMerge::new(readers, self.cmp)?);
        }

        Ok(self.heap.as_mut().expect("heap initialized before access"))
    }
}

impl<T, C, Cmp> Iterator for RunMerge<T, C, Cmp>
where
    C: Codec<Item = T> + Copy,
    Cmp: Compare<T> + Copy,
{
    type Item = Result<T, MergeError<C::Error>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }

        let result = match self.heap() {
            Ok(heap) => heap.next_output(),
            Err(err) => {
                self.failed = true;
                return Some(Err(err));
            }
        };

        match result {
            Ok(Some(item)) => Some(Ok(item)),
            Ok(None) => None,
            Err(e) => {
                self.failed = true;
                Some(Err(e))
            }
        }
    }
}

impl<T, C, Cmp> sealed::Sealed for RunMerge<T, C, Cmp>
where
    T: 'static,
    C: Codec<Item = T> + Copy + 'static,
    Cmp: Compare<T> + Copy + 'static,
{
}

impl<T, C, Cmp> VisitSortedItems for RunMerge<T, C, Cmp>
where
    T: 'static,
    C: Codec<Item = T> + Copy + 'static,
    Cmp: Compare<T> + Copy + 'static,
{
    type Item<'a>
        = T
    where
        Self: 'a;

    type Error = MergeError<C::Error>;

    fn visit_items<F, FE>(self, mut visit_item: F) -> Result<(), SortedItemsError<Self::Error, FE>>
    where
        F: for<'a> FnMut(Self::Item<'a>) -> Result<(), FE>,
    {
        for item in self {
            let item = item.map_err(SortedItemsError::Source)?;
            visit_item(item).map_err(SortedItemsError::Sink)?;
        }

        Ok(())
    }
}

/// Lazy keyed merge over sorted runs spilled to disk.
///
/// `KeyedRunMerge` owns the final keyed runs and constructs keyed merge state
/// when consumed. Owned iteration materializes records; current-item traversal
/// can visit the winning cursor's current representation before advancing it.
pub struct KeyedRunMerge<
    T,
    C: KeyedCodec<Item = T>,
    KeyCmp: Compare<C::Key> + Copy,
    ItemCmp: Compare<T> + Copy,
> {
    runs: Option<Vec<SortedRun>>,
    heap: Option<KeyedHeapMerge<T, C, KeyCmp, ItemCmp>>,
    codec: C,
    key_cmp: KeyCmp,
    item_cmp: ItemCmp,
    failed: bool,
}

impl<T, C, KeyCmp, ItemCmp> KeyedRunMerge<T, C, KeyCmp, ItemCmp>
where
    C: KeyedCodec<Item = T>,
    KeyCmp: Compare<C::Key> + Copy,
    ItemCmp: Compare<T> + Copy,
{
    fn new(runs: Vec<SortedRun>, codec: C, key_cmp: KeyCmp, item_cmp: ItemCmp) -> Self {
        Self {
            runs: Some(runs),
            heap: None,
            codec,
            key_cmp,
            item_cmp,
            failed: false,
        }
    }

    fn heap(&mut self) -> Result<&mut KeyedHeapMerge<T, C, KeyCmp, ItemCmp>, MergeError<C::Error>> {
        if self.heap.is_none() {
            let runs = self
                .runs
                .take()
                .expect("keyed runs are initialized at most once");
            let readers = open_keyed_readers(runs, self.codec)?;
            self.heap = Some(KeyedHeapMerge::new(readers, self.key_cmp, self.item_cmp)?);
        }

        Ok(self
            .heap
            .as_mut()
            .expect("keyed heap is initialized before access"))
    }
}

impl<T, C, KeyCmp, ItemCmp> Iterator for KeyedRunMerge<T, C, KeyCmp, ItemCmp>
where
    C: KeyedCodec<Item = T>,
    KeyCmp: Compare<C::Key> + Copy,
    ItemCmp: Compare<T> + Copy,
{
    type Item = Result<T, MergeError<C::Error>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }

        let result = match self.heap() {
            Ok(heap) => heap.next_output(),
            Err(err) => {
                self.failed = true;
                return Some(Err(err));
            }
        };

        match result {
            Ok(Some(item)) => Some(Ok(item)),
            Ok(None) => None,
            Err(e) => {
                self.failed = true;
                Some(Err(e))
            }
        }
    }
}

impl<T, C, KeyCmp, ItemCmp> sealed::Sealed for KeyedRunMerge<T, C, KeyCmp, ItemCmp>
where
    T: 'static,
    C: KeyedCodec<Item = T> + 'static,
    KeyCmp: Compare<C::Key> + Copy + 'static,
    ItemCmp: Compare<T> + Copy + 'static,
{
}

impl<T, C, KeyCmp, ItemCmp> VisitSortedItems for KeyedRunMerge<T, C, KeyCmp, ItemCmp>
where
    T: 'static,
    C: KeyedCodec<Item = T> + 'static,
    KeyCmp: Compare<C::Key> + Copy + 'static,
    ItemCmp: Compare<T> + Copy + 'static,
{
    type Item<'a>
        = <C::KeyedCursor<std::fs::File> as CodecCursor<T>>::Current<'a>
    where
        Self: 'a;

    type Error = MergeError<C::Error>;

    fn visit_items<F, FE>(mut self, mut f: F) -> Result<(), SortedItemsError<Self::Error, FE>>
    where
        F: for<'a> FnMut(Self::Item<'a>) -> Result<(), FE>,
    {
        loop {
            let visited = {
                let heap = self.heap().map_err(SortedItemsError::Source)?;
                heap.visit_next(|item| f(item))?
            };

            if !visited {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufWriter, Read, Write};

    use super::*;
    use crate::compare::Natural;

    #[derive(Clone, Copy)]
    struct U64Codec;

    #[derive(Clone, Copy)]
    struct U64KeyedCodec;

    struct U64Writer<W: Write> {
        inner: BufWriter<W>,
    }

    impl<W: Write> CodecWriter<u64> for U64Writer<W> {
        type Error = std::io::Error;

        fn write(&mut self, item: &u64) -> Result<(), Self::Error> {
            self.inner.write_all(&item.to_le_bytes())
        }

        fn finish(mut self) -> Result<(), Self::Error> {
            self.inner.flush()
        }
    }

    struct U64Reader<R: Read> {
        inner: R,
        current: Option<u64>,
    }

    impl<R: Read> CodecCursor<u64> for U64Reader<R> {
        type Error = std::io::Error;
        type Current<'a>
            = u64
        where
            Self: 'a;

        fn advance(&mut self) -> Result<bool, Self::Error> {
            let mut buf = [0u8; 8];
            match self.inner.read(&mut buf[..1]) {
                Ok(0) => {
                    self.current = None;
                    Ok(false)
                }
                Ok(_) => {
                    self.inner.read_exact(&mut buf[1..])?;
                    self.current = Some(u64::from_le_bytes(buf));
                    Ok(true)
                }
                Err(e) => Err(e),
            }
        }

        fn current(&mut self) -> Result<u64, Self::Error> {
            self.current
                .ok_or_else(|| std::io::Error::other("current called before advance"))
        }

        fn with_current<'a, F>(
            &'a mut self,
            f: impl FnOnce(Self::Current<'a>) -> F,
        ) -> Result<F, Self::Error> {
            self.current().map(f)
        }
    }

    impl Codec for U64Codec {
        type Item = u64;
        type Error = std::io::Error;
        type Writer<W: Write> = U64Writer<W>;
        type Cursor<R: Read> = U64Reader<R>;

        fn writer<W: Write>(&self, dest: W) -> U64Writer<W> {
            U64Writer {
                inner: BufWriter::new(dest),
            }
        }

        fn cursor<R: Read>(&self, source: R) -> U64Reader<R> {
            U64Reader {
                inner: source,
                current: None,
            }
        }
    }

    struct U64KeyedWriter<W: Write> {
        inner: BufWriter<W>,
    }

    impl<W: Write> CodecWriter<u64> for U64KeyedWriter<W> {
        type Error = std::io::Error;

        fn write(&mut self, item: &u64) -> Result<(), Self::Error> {
            self.inner.write_all(&item.to_le_bytes())
        }

        fn finish(mut self) -> Result<(), Self::Error> {
            self.inner.flush()
        }
    }

    struct U64KeyedReader<R: Read> {
        inner: R,
        current: Option<u64>,
    }

    impl<R: Read> CodecCursor<u64> for U64KeyedReader<R> {
        type Error = std::io::Error;
        type Current<'a>
            = u64
        where
            Self: 'a;

        fn advance(&mut self) -> Result<bool, Self::Error> {
            let mut buf = [0u8; 8];
            match self.inner.read(&mut buf[..1]) {
                Ok(0) => {
                    self.current = None;
                    Ok(false)
                }
                Ok(_) => {
                    self.inner.read_exact(&mut buf[1..])?;
                    self.current = Some(u64::from_le_bytes(buf));
                    Ok(true)
                }
                Err(e) => Err(e),
            }
        }

        fn current(&mut self) -> Result<u64, Self::Error> {
            self.current
                .ok_or_else(|| std::io::Error::other("current called before advance"))
        }

        fn with_current<'a, F>(
            &'a mut self,
            f: impl FnOnce(Self::Current<'a>) -> F,
        ) -> Result<F, Self::Error> {
            self.current().map(f)
        }
    }

    impl Codec for U64KeyedCodec {
        type Item = u64;
        type Error = std::io::Error;
        type Writer<W: Write> = U64KeyedWriter<W>;
        type Cursor<R: Read> = U64KeyedReader<R>;

        fn writer<W: Write>(&self, dest: W) -> U64KeyedWriter<W> {
            U64KeyedWriter {
                inner: BufWriter::new(dest),
            }
        }

        fn cursor<R: Read>(&self, source: R) -> U64KeyedReader<R> {
            U64KeyedReader {
                inner: source,
                current: None,
            }
        }
    }

    struct U64OnlyKeyedWriter<W: Write> {
        inner: BufWriter<W>,
    }

    impl<W: Write> KeyedCodecWriter<u64, u8> for U64OnlyKeyedWriter<W> {
        type Error = std::io::Error;

        fn write_keyed(&mut self, item: &u64, key: &u8) -> Result<(), Self::Error> {
            self.inner.write_all(&[*key])?;
            self.inner.write_all(&item.to_le_bytes())
        }

        fn finish(mut self) -> Result<(), Self::Error> {
            self.inner.flush()
        }
    }

    struct U64OnlyKeyedReader<R: Read> {
        inner: R,
        current_key: Option<u8>,
        current: Option<u64>,
    }

    impl<R: Read> CodecCursor<u64> for U64OnlyKeyedReader<R> {
        type Error = std::io::Error;
        type Current<'a>
            = u64
        where
            Self: 'a;

        fn advance(&mut self) -> Result<bool, Self::Error> {
            let mut key = [0u8; 1];
            match self.inner.read(&mut key) {
                Ok(0) => {
                    self.current_key = None;
                    self.current = None;
                    Ok(false)
                }
                Ok(_) => {
                    let mut item = [0u8; 8];
                    self.inner.read_exact(&mut item)?;
                    self.current_key = Some(key[0]);
                    self.current = Some(u64::from_le_bytes(item));
                    Ok(true)
                }
                Err(e) => Err(e),
            }
        }

        fn current(&mut self) -> Result<u64, Self::Error> {
            self.current
                .ok_or_else(|| std::io::Error::other("current called before advance"))
        }

        fn with_current<'a, T>(
            &'a mut self,
            f: impl FnOnce(Self::Current<'a>) -> T,
        ) -> Result<T, Self::Error> {
            self.current().map(f)
        }
    }

    impl<R: Read> KeyedCodecCursor<u64, u8> for U64OnlyKeyedReader<R> {
        fn current_key(&self) -> Result<u8, Self::Error> {
            self.current_key
                .ok_or_else(|| std::io::Error::other("current_key called before advance"))
        }
    }

    impl KeyedCodec for U64KeyedCodec {
        type Key = u8;
        type KeyedWriter<W: Write> = U64OnlyKeyedWriter<W>;
        type KeyedCursor<R: Read> = U64OnlyKeyedReader<R>;

        fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
            U64OnlyKeyedWriter {
                inner: BufWriter::new(dest),
            }
        }

        fn keyed_cursor<R: Read>(&self, source: R) -> Self::KeyedCursor<R> {
            U64OnlyKeyedReader {
                inner: source,
                current_key: None,
                current: None,
            }
        }
    }

    impl DeriveKey<u64> for U64KeyedCodec {
        fn derive_key(&self, item: &u64) -> u8 {
            // coarse key: values in the same decade tie
            u8::try_from(*item / 10).expect("test values should fit in u8")
        }
    }

    fn default_merger() -> RunMerger<u64, U64Codec, Natural> {
        RunMerger::new(U64Codec, Natural, MergeConfig::default())
    }

    #[test]
    fn spill_and_merge_single_run() {
        let merger = default_merger();
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
        let merger = default_merger();
        let run_a = merger.spill_sorted(vec![1u64, 3, 5]).expect("spill A");
        let run_b = merger.spill_sorted(vec![2u64, 4, 6]).expect("spill B");

        let results: Vec<u64> = merger
            .merge(vec![run_a, run_b])
            .expect("merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn merge_preserves_duplicates_across_runs() {
        let merger = default_merger();
        let run_a = merger.spill_sorted(vec![1u64, 3, 5]).expect("spill");
        let run_b = merger.spill_sorted(vec![1u64, 3, 7]).expect("spill");

        let results: Vec<u64> = merger
            .merge(vec![run_a, run_b])
            .expect("merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 1, 3, 3, 5, 7]);
    }

    #[test]
    fn merge_empty_run_list() {
        let merger = default_merger();
        let results: Vec<u64> = merger
            .merge(vec![])
            .expect("merge zero runs")
            .map(|r| r.expect("read"))
            .collect();

        assert!(results.is_empty());
    }

    #[test]
    fn merge_single_empty_run() {
        let merger = default_merger();
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
        let merger = default_merger();
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
    fn merge_different_sized_runs() {
        let merger = default_merger();
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
    fn bounded_fan_in_triggers_intermediate_spill() {
        let config = MergeConfig {
            max_fan_in: NonZeroUsize::new(2).expect("2 is not zero"),
            ..MergeConfig::default()
        };
        let merger = RunMerger::new(U64Codec, Natural, config);

        let a = merger.spill_sorted(vec![1u64, 4]).expect("spill");
        let b = merger.spill_sorted(vec![2u64, 5]).expect("spill");
        let c = merger.spill_sorted(vec![3u64, 6]).expect("spill");

        let results: Vec<u64> = merger
            .merge(vec![a, b, c])
            .expect("merge with fan-in=2")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn merge_many_runs_with_small_fan_in() {
        let config = MergeConfig {
            max_fan_in: NonZeroUsize::new(3).expect("3 is not zero"),
            ..MergeConfig::default()
        };
        let merger = RunMerger::new(U64Codec, Natural, config);

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
    fn merge_config_default_has_sensible_values() {
        let config = MergeConfig::default();
        assert_eq!(config.max_fan_in.get(), 128);
        assert_eq!(config.read_buffer_bytes, 64 * 1024);
        assert_eq!(config.write_buffer_bytes, 64 * 1024);
        assert!(config.temp_dir.is_none());
    }

    #[test]
    fn custom_temp_dir_works() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let config = MergeConfig {
            temp_dir: Some(temp_dir.path().to_path_buf()),
            ..MergeConfig::default()
        };
        let merger = RunMerger::new(U64Codec, Natural, config);

        let run = merger.spill_sorted(vec![1u64, 2, 3]).expect("spill");
        let results: Vec<u64> = merger
            .merge(vec![run])
            .expect("merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![1, 2, 3]);
    }

    fn spill_sorted_keyed(codec: U64KeyedCodec, items: impl IntoIterator<Item = u64>) -> SortedRun {
        let named = tempfile::NamedTempFile::new().expect("create temp file");
        let mut file = named.reopen().expect("reopen temp file");
        let mut writer = codec.keyed_writer(&mut file);
        for item in items {
            let key = codec.derive_key(&item);
            writer.write_keyed(&item, &key).expect("write keyed record");
        }
        writer.finish().expect("finish keyed writer");
        drop(file);

        SortedRun {
            path: named.into_temp_path(),
        }
    }

    #[test]
    fn keyed_merge_falls_back_to_full_record_order_when_keys_tie() {
        let merger = RunMerger::new(U64KeyedCodec, Natural, MergeConfig::default());

        // Keys are item/10, so 11/12/18/19 all tie on key=1 across runs.
        // Correct output requires fallback comparison on the full record.
        let run_a = spill_sorted_keyed(U64KeyedCodec, vec![11u64, 19, 25]);
        let run_b = spill_sorted_keyed(U64KeyedCodec, vec![12u64, 18, 26]);

        let results: Vec<u64> = merger
            .merge_keyed(vec![run_a, run_b], Natural)
            .expect("keyed merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![11, 12, 18, 19, 25, 26]);
    }

    #[test]
    fn nonexistent_temp_dir_returns_io_error() {
        let config = MergeConfig {
            temp_dir: Some(PathBuf::from("/nonexistent/path/should/not/exist")),
            ..MergeConfig::default()
        };
        let merger = RunMerger::new(U64Codec, Natural, config);

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
            merger: &RunMerger<u64, U64Codec, Natural>,
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
            merger: &RunMerger<u64, U64Codec, Natural>,
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
                let merger = RunMerger::new(U64Codec, Natural, MergeConfig::default());
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
                let merger = RunMerger::new(U64Codec, Natural, MergeConfig::default());
                let total_input: usize = batches.iter().map(Vec::len).sum();
                let runs = spill_runs(&merger, &batches);
                let output_count = collect_merged(&merger, runs).len();

                prop_assert_eq!(total_input, output_count);
            }

            #[test]
            fn merge_output_matches_reference_sort(
                batches in proptest::collection::vec(arb_sorted_u64_vec(), 0..6),
            ) {
                let merger = RunMerger::new(U64Codec, Natural, MergeConfig::default());

                let mut reference: Vec<u64> = batches.iter().flatten().copied().collect();
                reference.sort_unstable();

                let runs = spill_runs(&merger, &batches);
                let results = collect_merged(&merger, runs);

                prop_assert_eq!(results, reference);
            }
        }
    }
}
