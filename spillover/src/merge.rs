//! K-way merge of sorted runs from temporary files on disk.
//!
//! This module provides the infrastructure for writing pre-sorted
//! items to temporary files and merging them back into a single
//! sorted stream via a min-heap. When the number of runs exceeds
//! a configurable fan-in limit, intermediate merge passes spill
//! to temporary files automatically.
//!
//! The merge engine supports two paths:
//! - record path: heap holds source indices; cursors retain current records
//! - keyed path: heap holds compact keys and falls back to
//!   full-record comparison when keys tie
//!
//! Both paths share fan-in recursion and intermediate spilling.

use std::{
    cmp::Ordering,
    convert::Infallible,
    io::{BufReader, BufWriter},
    num::NonZeroUsize,
    path::PathBuf,
};

use crate::{
    SortedItemsError,
    codec::{
        Codec, CodecCursor, CodecWriter, EncodeKeyedRun, EncodeRun, KeyedCodec, KeyedCodecCursor,
        KeyedCodecWriter, TakeOwned,
    },
    compare::Compare,
    sorter::VisitSortedItems,
};

/// Errors that can occur during merge operations.
#[derive(Debug)]
pub enum MergeError<CE> {
    /// An I/O error occurred reading or writing a temporary file.
    Io(std::io::Error),

    /// A temporary file ended with a partial record.
    TruncatedEntry,

    /// A selected cursor did not provide its current record.
    MissingCurrent,

    /// The codec failed to decode an entry from bytes.
    Codec(CE),
}

impl<CE: std::fmt::Display> std::fmt::Display for MergeError<CE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "merge I/O error: {e}"),
            Self::TruncatedEntry => write!(f, "sorted run ended with a partial record"),
            Self::MissingCurrent => write!(f, "selected cursor has no current record"),
            Self::Codec(e) => write!(f, "failed to decode entry from sorted run: {e}"),
        }
    }
}

impl<CE: std::error::Error + 'static> std::error::Error for MergeError<CE> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::TruncatedEntry | Self::MissingCurrent => None,
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

    /// Size of the outer read buffer (in bytes) for each temporary run being
    /// merged.
    ///
    /// This configures core-owned transport buffering for temporary run files.
    /// Codecs may still maintain additional format-specific buffers internally.
    pub read_buffer_bytes: usize,

    /// Size of the outer write buffer (in bytes) when spilling temporary runs.
    ///
    /// This configures core-owned transport buffering for temporary run files.
    /// Codecs may still maintain additional format-specific buffers internally.
    pub write_buffer_bytes: usize,

    /// Directory for temporary files. `None` uses the OS default.
    pub temp_dir: Option<PathBuf>,
}

type RunReader = BufReader<std::fs::File>;
pub(crate) type RunWriterStream = BufWriter<std::fs::File>;

impl MergeConfig {
    fn run_reader<CE>(&self, run: &SortedRun) -> MergeResult<RunReader, CE> {
        let file = run.reopen().map_err(MergeError::Io)?;
        Ok(BufReader::with_capacity(self.read_buffer_bytes, file))
    }

    fn run_writer(&self, file: std::fs::File) -> RunWriterStream {
        BufWriter::with_capacity(self.write_buffer_bytes, file)
    }
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

// ── RunMerger ────────────────────────────────────────────

/// Orchestrates the creation and merging of sorted runs on disk.
pub struct RunMerger<C: Codec, Cmp> {
    codec: C,
    cmp: Cmp,
    config: MergeConfig,
}

/// Writes sorted entries into temporary run files.
pub(crate) struct RunWriter<'config, C, M> {
    codec: C,
    config: &'config MergeConfig,
    _mode: std::marker::PhantomData<fn() -> M>,
}

pub(crate) struct Unkeyed;

pub(crate) struct Keyed;

impl<'config, C> RunWriter<'config, C, Unkeyed> {
    pub(crate) fn new_unkeyed(codec: C, config: &'config MergeConfig) -> Self {
        Self {
            codec,
            config,
            _mode: std::marker::PhantomData,
        }
    }
}

impl<'config, C> RunWriter<'config, C, Keyed> {
    pub(crate) fn new_keyed(codec: C, config: &'config MergeConfig) -> Self {
        Self {
            codec,
            config,
            _mode: std::marker::PhantomData,
        }
    }
}

impl<C, M> RunWriter<'_, C, M> {
    fn create_temp_file<CE>(&self) -> MergeResult<tempfile::NamedTempFile, CE> {
        let file = match self.config.temp_dir {
            Some(ref dir) => tempfile::NamedTempFile::new_in(dir)?,
            None => tempfile::NamedTempFile::new()?,
        };
        Ok(file)
    }
}

impl<C> RunWriter<'_, C, Unkeyed>
where
    C: Codec,
{
    pub(crate) fn write_sorted<T>(&self, items: &[T]) -> MergeResult<SortedRun, C::Error>
    where
        C: EncodeRun<T>,
    {
        let named = self.create_temp_file()?;
        let file = named.reopen().map_err(MergeError::Io)?;
        let mut run_writer = self.config.run_writer(file);
        let mut writer = self.codec.writer(&mut run_writer);

        for item in items {
            writer.write(item).map_err(MergeError::Codec)?;
        }

        CodecWriter::<T>::finish(writer).map_err(MergeError::Codec)?;
        std::io::Write::flush(&mut run_writer).map_err(MergeError::Io)?;

        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }
}

impl<C> RunWriter<'_, C, Keyed>
where
    C: KeyedCodec,
{
    pub(crate) fn write_sorted<T>(&self, items: &[T]) -> MergeResult<SortedRun, C::Error>
    where
        C: EncodeKeyedRun<T>,
    {
        let named = self.create_temp_file()?;
        let file = named.reopen().map_err(MergeError::Io)?;
        let mut run_writer = self.config.run_writer(file);
        let mut writer = self.codec.keyed_writer(&mut run_writer);

        for item in items {
            let key = self.codec.derive_key(item);
            writer.write_keyed(item, &key).map_err(MergeError::Codec)?;
        }

        writer.finish().map_err(MergeError::Codec)?;
        std::io::Write::flush(&mut run_writer).map_err(MergeError::Io)?;

        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }
}

impl<C: Codec, Cmp> RunMerger<C, Cmp> {
    /// Create a new merger.
    #[must_use]
    pub fn new(codec: C, cmp: Cmp, config: MergeConfig) -> Self {
        Self { codec, cmp, config }
    }

    fn create_temp_file(&self) -> MergeResult<tempfile::NamedTempFile, C::Error> {
        let file = match self.config.temp_dir {
            Some(ref dir) => tempfile::NamedTempFile::new_in(dir)?,
            None => tempfile::NamedTempFile::new()?,
        };
        Ok(file)
    }

    /// Write a pre-sorted iterator of items to a temporary file.
    ///
    /// The write-side item type may differ from the cursor's borrowed record.
    /// The caller is responsible for providing items in sorted order.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn spill_sorted<I>(
        &self,
        items: impl IntoIterator<Item = I>,
    ) -> Result<SortedRun, MergeError<C::Error>>
    where
        C: EncodeRun<I>,
    {
        let named = self.create_temp_file()?;
        let file = named.reopen().map_err(MergeError::Io)?;
        let mut run_writer = self.config.run_writer(file);
        let mut writer = self.codec.writer(&mut run_writer);

        for item in items {
            writer.write(&item).map_err(MergeError::Codec)?;
        }

        writer.finish().map_err(MergeError::Codec)?;
        std::io::Write::flush(&mut run_writer).map_err(MergeError::Io)?;

        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }

    /// Consume this merger to merge runs by their borrowed current records.
    ///
    /// When there are more runs than [`MergeConfig::max_fan_in`],
    /// intermediate merge passes spill to temp files automatically.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or merging fails.
    pub fn merge(self, runs: Vec<SortedRun>) -> MergeResult<RunMerge<C, Cmp>, C::Error>
    where
        C: EncodeRun<<<C as Codec>::Cursor<RunReader> as CodecCursor>::Record>,
        Cmp: Compare<<C::Cursor<RunReader> as CodecCursor>::Record>,
    {
        let fan_in = self.config.max_fan_in.get();
        let final_runs = reduce_runs(runs, fan_in, |group| self.spill_group(group))?;
        let execution = Execution::new(self, final_runs);
        Ok(RunMerge { execution })
    }

    fn spill_group(&self, runs: Vec<SortedRun>) -> MergeResult<SortedRun, C::Error>
    where
        C: EncodeRun<<<C as Codec>::Cursor<RunReader> as CodecCursor>::Record>,
        Cmp: Compare<<C::Cursor<RunReader> as CodecCursor>::Record>,
    {
        let named = self.create_temp_file()?;
        let file = named.reopen().map_err(MergeError::Io)?;
        let mut run_writer = self.config.run_writer(file);
        let mut writer = self.codec.writer(&mut run_writer);
        let execution = Execution::new(self, runs);
        execution
            .visit_items(|record| writer.write(record))
            .map_err(intermediate_error)?;
        writer.finish().map_err(MergeError::Codec)?;
        std::io::Write::flush(&mut run_writer).map_err(MergeError::Io)?;
        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }
}

// Keyed merge — only available when the codec supports keys.
impl<C: KeyedCodec, Cmp> RunMerger<C, Cmp> {
    /// Write a pre-sorted iterator of items and stored keys to a temporary file.
    ///
    /// The write-side item type may differ from the cursor's record. Keys are
    /// derived from each write-side item and stored alongside the record so the
    /// resulting run can be consumed by [`merge_keyed`](Self::merge_keyed).
    /// The caller is responsible for providing items in sorted order.
    ///
    /// # Errors
    ///
    /// Returns an error if key derivation, encoding, or writing fails.
    pub fn spill_sorted_with_keys<I>(
        &self,
        items: impl IntoIterator<Item = I>,
    ) -> MergeResult<SortedRun, C::Error>
    where
        C: EncodeKeyedRun<I>,
    {
        let named = self.create_temp_file()?;
        let file = named.reopen().map_err(MergeError::Io)?;
        let mut run_writer = self.config.run_writer(file);
        let mut writer = self.codec.keyed_writer(&mut run_writer);

        for item in items {
            let key = self.codec.derive_key(&item);
            writer.write_keyed(&item, &key).map_err(MergeError::Codec)?;
        }

        KeyedCodecWriter::<I, C::Key>::finish(writer).map_err(MergeError::Codec)?;
        std::io::Write::flush(&mut run_writer).map_err(MergeError::Io)?;

        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }

    /// Consume this merger to merge runs by stored key, comparing borrowed
    /// records when keys tie.
    ///
    /// When there are more runs than [`MergeConfig::max_fan_in`],
    /// intermediate merge passes spill to temp files automatically.
    /// Intermediate files preserve each winning entry's original stored key.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or merging fails.
    pub fn merge_keyed<KeyCmp: Compare<C::Key>>(
        self,
        runs: Vec<SortedRun>,
        key_cmp: KeyCmp,
    ) -> MergeResult<KeyedRunMerge<C, KeyCmp, Cmp>, C::Error>
    where
        C: EncodeKeyedRun<<<C as KeyedCodec>::KeyedCursor<RunReader> as CodecCursor>::Record>,
        Cmp: Compare<<C::KeyedCursor<RunReader> as CodecCursor>::Record>,
    {
        let fan_in = self.config.max_fan_in.get();
        let context = KeyedContext {
            merger: self,
            key_cmp,
        };
        let final_runs = reduce_runs(runs, fan_in, |group| context.spill_group(group))?;
        let execution = Execution::new(context, final_runs);
        Ok(KeyedRunMerge { execution })
    }
}

fn reduce_runs<CE>(
    mut runs: Vec<SortedRun>,
    fan_in: usize,
    mut spill_group: impl FnMut(Vec<SortedRun>) -> MergeResult<SortedRun, CE>,
) -> MergeResult<Vec<SortedRun>, CE> {
    while runs.len() > fan_in {
        let mut intermediate = Vec::new();
        while !runs.is_empty() {
            let chunk_end = runs.len().min(fan_in);
            let group = runs.drain(..chunk_end).collect();
            let run = spill_group(group)?;
            intermediate.push(run);
        }
        runs = intermediate;
    }
    Ok(runs)
}

fn intermediate_error<CE>(error: SortedItemsError<MergeError<CE>, CE>) -> MergeError<CE> {
    match error {
        SortedItemsError::Source(error) => error,
        SortedItemsError::Sink(error) => MergeError::Codec(error),
    }
}

// These are the only mode-specific operations. Selection, progress, and delivery
// are shared. A borrowed context lets intermediate passes use the final owner's
// configuration without copying its comparators.
trait MergeContext {
    type Error: std::error::Error + Send + Sync + 'static;
    type Cursor: CodecCursor<Error = Self::Error>;
    type Key;

    fn config(&self) -> &MergeConfig;
    fn open(&self, reader: RunReader) -> Result<Self::Cursor, Self::Error>;
    fn current_key(&self, cursor: &Self::Cursor) -> Result<Self::Key, Self::Error>;
    fn compare(
        &self,
        readers: &mut [Self::Cursor],
        left: &HeapEntry<Self::Key>,
        right: &HeapEntry<Self::Key>,
    ) -> Result<Ordering, Self::Error>;
}

impl<M: MergeContext> MergeContext for &M {
    type Error = M::Error;
    type Cursor = M::Cursor;
    type Key = M::Key;

    fn config(&self) -> &MergeConfig {
        M::config(self)
    }

    fn open(&self, reader: RunReader) -> Result<Self::Cursor, Self::Error> {
        M::open(self, reader)
    }

    fn current_key(&self, cursor: &Self::Cursor) -> Result<Self::Key, Self::Error> {
        M::current_key(self, cursor)
    }

    fn compare(
        &self,
        readers: &mut [Self::Cursor],
        left: &HeapEntry<Self::Key>,
        right: &HeapEntry<Self::Key>,
    ) -> Result<Ordering, Self::Error> {
        M::compare(self, readers, left, right)
    }
}

impl<C, Cmp> MergeContext for RunMerger<C, Cmp>
where
    C: Codec,
    Cmp: Compare<<C::Cursor<RunReader> as CodecCursor>::Record>,
{
    type Error = C::Error;
    type Cursor = C::Cursor<RunReader>;
    type Key = ();

    fn config(&self) -> &MergeConfig {
        &self.config
    }

    fn open(&self, reader: RunReader) -> Result<Self::Cursor, Self::Error> {
        self.codec.cursor(reader)
    }

    fn current_key(&self, _cursor: &Self::Cursor) -> Result<(), Self::Error> {
        Ok(())
    }

    fn compare(
        &self,
        readers: &mut [Self::Cursor],
        left: &HeapEntry<()>,
        right: &HeapEntry<()>,
    ) -> Result<Ordering, Self::Error> {
        let ordering = compare_records(readers, left.source, right.source, &self.cmp)?;
        Ok(ordering.then_with(|| left.source.cmp(&right.source)))
    }
}

struct KeyedContext<C: KeyedCodec, KeyCmp, ItemCmp> {
    merger: RunMerger<C, ItemCmp>,
    key_cmp: KeyCmp,
}

impl<C, KeyCmp, ItemCmp> MergeContext for KeyedContext<C, KeyCmp, ItemCmp>
where
    C: KeyedCodec,
    KeyCmp: Compare<C::Key>,
    ItemCmp: Compare<<C::KeyedCursor<RunReader> as CodecCursor>::Record>,
{
    type Error = C::Error;
    type Cursor = C::KeyedCursor<RunReader>;
    type Key = C::Key;

    fn config(&self) -> &MergeConfig {
        &self.merger.config
    }

    fn open(&self, reader: RunReader) -> Result<Self::Cursor, Self::Error> {
        self.merger.codec.keyed_cursor(reader)
    }

    fn current_key(&self, cursor: &Self::Cursor) -> Result<Self::Key, Self::Error> {
        cursor.current_key()
    }

    fn compare(
        &self,
        readers: &mut [Self::Cursor],
        left: &HeapEntry<Self::Key>,
        right: &HeapEntry<Self::Key>,
    ) -> Result<Ordering, Self::Error> {
        let key_ordering = self.key_cmp.compare(&left.key, &right.key);
        if !key_ordering.is_eq() {
            return Ok(key_ordering);
        }
        let ordering = compare_records(readers, left.source, right.source, &self.merger.cmp)?;
        Ok(ordering.then_with(|| left.source.cmp(&right.source)))
    }
}

impl<C, KeyCmp, ItemCmp> KeyedContext<C, KeyCmp, ItemCmp>
where
    C: KeyedCodec
        + EncodeKeyedRun<<<C as KeyedCodec>::KeyedCursor<RunReader> as CodecCursor>::Record>,
    KeyCmp: Compare<C::Key>,
    ItemCmp: Compare<<C::KeyedCursor<RunReader> as CodecCursor>::Record>,
{
    fn spill_group(&self, runs: Vec<SortedRun>) -> MergeResult<SortedRun, C::Error> {
        let named = self.merger.create_temp_file()?;
        let file = named.reopen().map_err(MergeError::Io)?;
        let mut run_writer = self.merger.config.run_writer(file);
        let mut writer = self.merger.codec.keyed_writer(&mut run_writer);
        let mut execution = Execution::new(self, runs);
        loop {
            let delivered = execution
                .next_with(|state, winner| {
                    state.with_record(winner, |record| writer.write_keyed(record, &winner.key))
                })
                .map_err(intermediate_error)?;
            if delivered.is_none() {
                break;
            }
        }
        writer.finish().map_err(MergeError::Codec)?;
        std::io::Write::flush(&mut run_writer).map_err(MergeError::Io)?;
        Ok(SortedRun {
            path: named.into_temp_path(),
        })
    }
}

// Both indices identify positioned cursors. The split keeps their short-lived
// mutable loans disjoint even when record access performs lazy decoding.
fn compare_records<R, Cmp>(
    readers: &mut [R],
    left: usize,
    right: usize,
    cmp: &Cmp,
) -> Result<Ordering, R::Error>
where
    R: CodecCursor,
    Cmp: Compare<R::Record>,
{
    if left == right {
        return Ok(Ordering::Equal);
    }
    let (left_reader, right_reader) = if left < right {
        let (before, after) = readers.split_at_mut(right);
        (&mut before[left], &mut after[0])
    } else {
        let (before, after) = readers.split_at_mut(left);
        (&mut after[0], &mut before[right])
    };
    left_reader.with_current(|left_record| {
        right_reader.with_current(|right_record| cmp.compare(left_record, right_record))
    })?
}

struct HeapEntry<K> {
    source: usize,
    key: K,
}

struct Selection<R, K> {
    readers: Vec<R>,
    heap: Vec<HeapEntry<K>>,
}

impl<R: CodecCursor, K> Selection<R, K> {
    fn open<M>(context: &M, runs: Vec<SortedRun>) -> MergeResult<Self, R::Error>
    where
        M: MergeContext<Cursor = R, Key = K, Error = R::Error>,
    {
        let mut readers = Vec::with_capacity(runs.len());
        let heap = Vec::with_capacity(runs.len());
        for run in runs {
            let transport = context.config().run_reader(&run)?;
            let cursor = context.open(transport).map_err(MergeError::Codec)?;
            readers.push(cursor);
        }
        let mut state = Self { readers, heap };
        for source in 0..state.readers.len() {
            state.refill(source, context)?;
        }
        Ok(state)
    }

    fn refill<M>(&mut self, source: usize, context: &M) -> MergeResult<(), R::Error>
    where
        M: MergeContext<Cursor = R, Key = K, Error = R::Error>,
    {
        let cursor = &mut self.readers[source];
        let positioned = cursor.advance().map_err(MergeError::Codec)?;
        if positioned {
            let key = context.current_key(cursor).map_err(MergeError::Codec)?;
            let entry = HeapEntry { source, key };
            heap_push_by(&mut self.heap, entry, |left, right| {
                context.compare(&mut self.readers, left, right)
            })
            .map_err(MergeError::Codec)?;
        }
        Ok(())
    }

    fn take_winner<M>(&mut self, context: &M) -> MergeResult<Option<HeapEntry<K>>, R::Error>
    where
        M: MergeContext<Cursor = R, Key = K, Error = R::Error>,
    {
        heap_pop_by(&mut self.heap, |left, right| {
            context.compare(&mut self.readers, left, right)
        })
        .map_err(MergeError::Codec)
    }

    fn with_record<Output, FE>(
        &mut self,
        winner: &HeapEntry<K>,
        visit: impl FnOnce(&R::Record) -> Result<Output, FE>,
    ) -> Result<Output, SortedItemsError<MergeError<R::Error>, FE>> {
        let visited = self.readers[winner.source]
            .with_current(visit)
            .map_err(MergeError::Codec)
            .map_err(SortedItemsError::Source)?;
        visited.map_err(SortedItemsError::Sink)
    }
}

// The caller supplies a min-heap and a consistent ordering for its live entries.
// Failed comparisons may leave the heap unordered; the execution then drops it.
fn heap_push_by<T, E>(
    heap: &mut Vec<T>,
    entry: T,
    mut compare: impl FnMut(&T, &T) -> Result<Ordering, E>,
) -> Result<(), E> {
    let mut child = heap.len();
    heap.push(entry);
    while child > 0 {
        let parent = (child - 1) / 2;
        let ordering = compare(&heap[child], &heap[parent])?;
        if !ordering.is_lt() {
            break;
        }
        heap.swap(child, parent);
        child = parent;
    }
    Ok(())
}

fn heap_pop_by<T, E>(
    heap: &mut Vec<T>,
    mut compare: impl FnMut(&T, &T) -> Result<Ordering, E>,
) -> Result<Option<T>, E> {
    if heap.is_empty() {
        return Ok(None);
    }
    let winner = heap.swap_remove(0);
    let mut parent = 0;
    // Indices below len / 2 are exactly the nodes with children.
    while parent < heap.len() / 2 {
        let mut child = parent * 2 + 1;
        let right = child + 1;
        if right < heap.len() {
            let ordering = compare(&heap[right], &heap[child])?;
            if ordering.is_lt() {
                child = right;
            }
        }
        let ordering = compare(&heap[child], &heap[parent])?;
        if !ordering.is_lt() {
            break;
        }
        heap.swap(parent, child);
        parent = child;
    }
    Ok(Some(winner))
}

enum MergePhase<S> {
    Pending(Vec<SortedRun>),
    Active(S),
    NeedsRefill { state: S, source: usize },
    Exhausted,
    Failed,
}

struct Execution<M, R, K> {
    context: M,
    phase: MergePhase<Selection<R, K>>,
}

impl<M: MergeContext> Execution<M, M::Cursor, M::Key> {
    fn new(context: M, runs: Vec<SortedRun>) -> Self {
        Self {
            context,
            phase: MergePhase::Pending(runs),
        }
    }

    // Delivery may access only the removed winner. It never advances a cursor
    // or changes selection. All fallible work runs with Failed already installed.
    fn next_with<Output, FE>(
        &mut self,
        deliver: impl FnOnce(
            &mut Selection<M::Cursor, M::Key>,
            &HeapEntry<M::Key>,
        ) -> Result<Output, SortedItemsError<MergeError<M::Error>, FE>>,
    ) -> Result<Option<Output>, SortedItemsError<MergeError<M::Error>, FE>> {
        loop {
            let phase = std::mem::replace(&mut self.phase, MergePhase::Failed);
            match phase {
                MergePhase::Pending(runs) => {
                    let state =
                        Selection::open(&self.context, runs).map_err(SortedItemsError::Source)?;
                    self.phase = MergePhase::Active(state);
                }
                MergePhase::NeedsRefill { mut state, source } => {
                    state
                        .refill(source, &self.context)
                        .map_err(SortedItemsError::Source)?;
                    self.phase = MergePhase::Active(state);
                }
                MergePhase::Active(mut state) => {
                    let selected = state
                        .take_winner(&self.context)
                        .map_err(SortedItemsError::Source)?;
                    let Some(winner) = selected else {
                        self.phase = MergePhase::Exhausted;
                        return Ok(None);
                    };
                    let source = winner.source;
                    let output = deliver(&mut state, &winner)?;
                    self.phase = MergePhase::NeedsRefill { state, source };
                    return Ok(Some(output));
                }
                MergePhase::Exhausted => {
                    self.phase = MergePhase::Exhausted;
                    return Ok(None);
                }
                MergePhase::Failed => return Ok(None),
            }
        }
    }

    fn next_owned(&mut self) -> Option<MergeResult<<M::Cursor as TakeOwned>::Owned, M::Error>>
    where
        M::Cursor: TakeOwned,
    {
        let delivered: Result<_, SortedItemsError<_, Infallible>> =
            self.next_with(|state, winner| {
                let extracted = state.readers[winner.source]
                    .take_owned()
                    .map_err(MergeError::Codec)
                    .map_err(SortedItemsError::Source)?;
                let record = extracted
                    .ok_or(MergeError::MissingCurrent)
                    .map_err(SortedItemsError::Source)?;
                Ok(record)
            });
        match delivered {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => None,
            Err(SortedItemsError::Source(error)) => Some(Err(error)),
            Err(SortedItemsError::Sink(impossible)) => match impossible {},
        }
    }

    fn visit_items<F, FE>(
        mut self,
        mut visit: F,
    ) -> Result<(), SortedItemsError<MergeError<M::Error>, FE>>
    where
        F: FnMut(&<M::Cursor as CodecCursor>::Record) -> Result<(), FE>,
    {
        loop {
            let delivered =
                self.next_with(|state, winner| state.with_record(winner, &mut visit))?;
            if delivered.is_none() {
                return Ok(());
            }
        }
    }
}

/// Lazy merge over sorted runs spilled to disk.
///
/// The final runs open on first consumption. Borrowed traversal visits cursor
/// records; owned iteration is available only when the cursor implements
/// [`TakeOwned`]. The merger owns its codec, comparator, and configuration.
pub struct RunMerge<C: Codec, Cmp> {
    execution: Execution<RunMerger<C, Cmp>, C::Cursor<RunReader>, ()>,
}

impl<C, Cmp> Iterator for RunMerge<C, Cmp>
where
    C: Codec,
    C::Cursor<RunReader>: TakeOwned,
    Cmp: Compare<<C::Cursor<RunReader> as CodecCursor>::Record>,
{
    type Item = MergeResult<<C::Cursor<RunReader> as TakeOwned>::Owned, C::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.execution.next_owned()
    }
}

impl<C, Cmp> VisitSortedItems for RunMerge<C, Cmp>
where
    C: Codec,
    Cmp: Compare<<C::Cursor<RunReader> as CodecCursor>::Record>,
{
    type Record = <C::Cursor<RunReader> as CodecCursor>::Record;
    type Error = MergeError<C::Error>;

    fn visit_items<F, FE>(self, visit: F) -> Result<(), SortedItemsError<Self::Error, FE>>
    where
        F: FnMut(&Self::Record) -> Result<(), FE>,
    {
        self.execution.visit_items(visit)
    }
}

/// Lazy keyed merge over sorted runs spilled to disk.
///
/// Stored keys determine order, with borrowed record comparison on ties.
/// Traversal borrows the winning record; owned iteration additionally requires
/// the keyed cursor's [`TakeOwned`] capability.
pub struct KeyedRunMerge<C: KeyedCodec, KeyCmp, ItemCmp> {
    execution: Execution<KeyedContext<C, KeyCmp, ItemCmp>, C::KeyedCursor<RunReader>, C::Key>,
}

impl<C, KeyCmp, ItemCmp> Iterator for KeyedRunMerge<C, KeyCmp, ItemCmp>
where
    C: KeyedCodec,
    C::KeyedCursor<RunReader>: TakeOwned,
    KeyCmp: Compare<C::Key>,
    ItemCmp: Compare<<C::KeyedCursor<RunReader> as CodecCursor>::Record>,
{
    type Item = MergeResult<<C::KeyedCursor<RunReader> as TakeOwned>::Owned, C::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.execution.next_owned()
    }
}

impl<C, KeyCmp, ItemCmp> VisitSortedItems for KeyedRunMerge<C, KeyCmp, ItemCmp>
where
    C: KeyedCodec,
    KeyCmp: Compare<C::Key>,
    ItemCmp: Compare<<C::KeyedCursor<RunReader> as CodecCursor>::Record>,
{
    type Record = <C::KeyedCursor<RunReader> as CodecCursor>::Record;
    type Error = MergeError<C::Error>;

    fn visit_items<F, FE>(self, visit: F) -> Result<(), SortedItemsError<Self::Error, FE>>
    where
        F: FnMut(&Self::Record) -> Result<(), FE>,
    {
        self.execution.visit_items(visit)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufWriter, Read, Write};

    use super::*;
    use crate::{codec::DeriveKey, compare::Natural};

    #[derive(Clone, Copy)]
    struct U64Codec;

    #[derive(Clone, Copy)]
    struct U64KeyedCodec;

    #[derive(Clone, Copy)]
    struct U64View(u64);

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

    impl<W: Write> CodecWriter<U64View> for U64Writer<W> {
        type Error = std::io::Error;

        fn write(&mut self, item: &U64View) -> Result<(), Self::Error> {
            self.inner.write_all(&item.0.to_le_bytes())
        }

        fn finish(mut self) -> Result<(), Self::Error> {
            self.inner.flush()
        }
    }

    struct U64Reader<R: Read> {
        inner: R,
        current: Option<u64>,
    }

    impl<R: Read> CodecCursor for U64Reader<R> {
        type Error = std::io::Error;
        type Record = u64;

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

        fn with_current<F>(
            &mut self,
            f: impl FnOnce(&Self::Record) -> F,
        ) -> Result<F, Self::Error> {
            let record = self
                .current
                .as_ref()
                .ok_or_else(|| std::io::Error::other("current called before advance"))?;
            Ok(f(record))
        }
    }

    impl<R: Read> TakeOwned for U64Reader<R> {
        type Owned = u64;

        fn take_owned(&mut self) -> Result<Option<Self::Owned>, Self::Error> {
            Ok(self.current.take())
        }
    }

    impl Codec for U64Codec {
        type Error = std::io::Error;
        type Cursor<R: Read> = U64Reader<R>;

        fn cursor<R: Read>(&self, source: R) -> Result<U64Reader<R>, Self::Error> {
            Ok(U64Reader {
                inner: source,
                current: None,
            })
        }
    }

    impl EncodeRun<u64> for U64Codec {
        type Writer<W: Write> = U64Writer<W>;

        fn writer<W: Write>(&self, dest: W) -> U64Writer<W> {
            U64Writer {
                inner: BufWriter::new(dest),
            }
        }
    }

    impl EncodeRun<U64View> for U64Codec {
        type Writer<W: Write> = U64Writer<W>;

        fn writer<W: Write>(&self, dest: W) -> U64Writer<W> {
            U64Writer {
                inner: BufWriter::new(dest),
            }
        }
    }

    struct U64KeyedReader<R: Read> {
        inner: R,
        current: Option<u64>,
    }

    impl<R: Read> CodecCursor for U64KeyedReader<R> {
        type Error = std::io::Error;
        type Record = u64;

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

        fn with_current<F>(
            &mut self,
            f: impl FnOnce(&Self::Record) -> F,
        ) -> Result<F, Self::Error> {
            let record = self
                .current
                .as_ref()
                .ok_or_else(|| std::io::Error::other("current called before advance"))?;
            Ok(f(record))
        }
    }

    impl<R: Read> TakeOwned for U64KeyedReader<R> {
        type Owned = u64;

        fn take_owned(&mut self) -> Result<Option<Self::Owned>, Self::Error> {
            Ok(self.current.take())
        }
    }

    impl Codec for U64KeyedCodec {
        type Error = std::io::Error;
        type Cursor<R: Read> = U64KeyedReader<R>;

        fn cursor<R: Read>(&self, source: R) -> Result<U64KeyedReader<R>, Self::Error> {
            Ok(U64KeyedReader {
                inner: source,
                current: None,
            })
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

    impl<W: Write> KeyedCodecWriter<U64View, u8> for U64OnlyKeyedWriter<W> {
        type Error = std::io::Error;

        fn write_keyed(&mut self, item: &U64View, key: &u8) -> Result<(), Self::Error> {
            self.inner.write_all(&[*key])?;
            self.inner.write_all(&item.0.to_le_bytes())
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

    impl<R: Read> CodecCursor for U64OnlyKeyedReader<R> {
        type Error = std::io::Error;
        type Record = u64;

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

        fn with_current<T>(
            &mut self,
            f: impl FnOnce(&Self::Record) -> T,
        ) -> Result<T, Self::Error> {
            let record = self
                .current
                .as_ref()
                .ok_or_else(|| std::io::Error::other("current called before advance"))?;
            Ok(f(record))
        }
    }

    impl<R: Read> TakeOwned for U64OnlyKeyedReader<R> {
        type Owned = u64;

        fn take_owned(&mut self) -> Result<Option<Self::Owned>, Self::Error> {
            Ok(self.current.take())
        }
    }

    impl<R: Read> KeyedCodecCursor<u8> for U64OnlyKeyedReader<R> {
        fn current_key(&self) -> Result<u8, Self::Error> {
            self.current_key
                .ok_or_else(|| std::io::Error::other("current_key called before advance"))
        }
    }

    impl KeyedCodec for U64KeyedCodec {
        type Key = u8;
        type KeyedCursor<R: Read> = U64OnlyKeyedReader<R>;

        fn keyed_cursor<R: Read>(&self, source: R) -> Result<Self::KeyedCursor<R>, Self::Error> {
            Ok(U64OnlyKeyedReader {
                inner: source,
                current_key: None,
                current: None,
            })
        }
    }

    impl EncodeKeyedRun<u64> for U64KeyedCodec {
        type KeyedWriter<W: Write> = U64OnlyKeyedWriter<W>;

        fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
            U64OnlyKeyedWriter {
                inner: BufWriter::new(dest),
            }
        }
    }

    impl EncodeKeyedRun<U64View> for U64KeyedCodec {
        type KeyedWriter<W: Write> = U64OnlyKeyedWriter<W>;

        fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
            U64OnlyKeyedWriter {
                inner: BufWriter::new(dest),
            }
        }
    }

    impl DeriveKey<u64> for U64KeyedCodec {
        fn derive_key(&self, item: &u64) -> u8 {
            // coarse key: values in the same decade tie
            u8::try_from(*item / 10).expect("test values should fit in u8")
        }
    }

    impl DeriveKey<U64View> for U64KeyedCodec {
        fn derive_key(&self, item: &U64View) -> u8 {
            u8::try_from(item.0 / 10).expect("test values should fit in u8")
        }
    }

    fn default_merger() -> RunMerger<U64Codec, Natural> {
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
    fn spill_sorted_accepts_write_side_items_that_differ_from_read_items() {
        let merger = default_merger();
        let run = merger
            .spill_sorted([U64View(1), U64View(3), U64View(5)])
            .expect("spilling views should succeed");

        let results: Vec<u64> = merger
            .merge(vec![run])
            .expect("merging should succeed")
            .map(|r| r.expect("reading should succeed"))
            .collect();

        assert_eq!(results, vec![1, 3, 5]);
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

    #[test]
    fn keyed_merge_falls_back_to_full_record_order_when_keys_tie() {
        let merger = RunMerger::new(U64KeyedCodec, Natural, MergeConfig::default());

        // Keys are item/10, so 11/12/18/19 all tie on key=1 across runs.
        // Correct output requires fallback comparison on the full record.
        let run_a = merger
            .spill_sorted_with_keys(vec![11u64, 19, 25])
            .expect("spill A");
        let run_b = merger
            .spill_sorted_with_keys(vec![12u64, 18, 26])
            .expect("spill B");

        let results: Vec<u64> = merger
            .merge_keyed(vec![run_a, run_b], Natural)
            .expect("keyed merge")
            .map(|r| r.expect("read"))
            .collect();

        assert_eq!(results, vec![11, 12, 18, 19, 25, 26]);
    }

    #[test]
    fn spill_sorted_with_keys_accepts_write_side_items_that_differ_from_read_items() {
        let merger = RunMerger::new(U64KeyedCodec, Natural, MergeConfig::default());
        let run = merger
            .spill_sorted_with_keys([U64View(11), U64View(19), U64View(25)])
            .expect("spilling keyed views should succeed");

        let results: Vec<u64> = merger
            .merge_keyed(vec![run], Natural)
            .expect("keyed merge should succeed")
            .map(|r| r.expect("reading should succeed"))
            .collect();

        assert_eq!(results, vec![11, 19, 25]);
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
            merger: &RunMerger<U64Codec, Natural>,
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

        fn collect_merged(merger: RunMerger<U64Codec, Natural>, runs: Vec<SortedRun>) -> Vec<u64> {
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
                let results = collect_merged(merger, runs);

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
                let output_count = collect_merged(merger, runs).len();

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
                let results = collect_merged(merger, runs);

                prop_assert_eq!(results, reference);
            }
        }
    }
}
