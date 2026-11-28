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

    /// A codec operation failed.
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
    pub(crate) fn write_sorted<Yield, Record>(
        &self,
        items: impl IntoIterator<Item = Yield>,
        project_record: impl Fn(&Yield) -> &Record,
    ) -> MergeResult<SortedRun, C::Error>
    where
        C: EncodeKeyedRun<Record> + crate::codec::DeriveKey<Record>,
    {
        self.write_with(|writer: &mut C::KeyedWriter<&mut RunWriterStream>| {
            for item in items {
                let record = project_record(&item);
                let key = self.codec.derive_key(record);
                writer
                    .write_keyed(crate::key::Keyed::trusted(record, &key))
                    .map_err(MergeError::Codec)?;
            }
            Ok(())
        })
    }

    pub(crate) fn write_supplied<'item, Record: ?Sized + 'item>(
        &self,
        items: impl IntoIterator<Item = crate::key::Keyed<&'item Record, &'item C::Key>>,
    ) -> MergeResult<SortedRun, C::Error>
    where
        C: EncodeKeyedRun<Record>,
        C::Key: 'item,
    {
        self.write_with(|writer: &mut C::KeyedWriter<&mut RunWriterStream>| {
            for record in items {
                writer.write_keyed(record).map_err(MergeError::Codec)?;
            }
            Ok(())
        })
    }

    fn write_with<Record: ?Sized>(
        &self,
        write: impl FnOnce(&mut C::KeyedWriter<&mut RunWriterStream>) -> MergeResult<(), C::Error>,
    ) -> MergeResult<SortedRun, C::Error>
    where
        C: EncodeKeyedRun<Record>,
    {
        let named = self.create_temp_file()?;
        let file = named.reopen().map_err(MergeError::Io)?;
        let mut run_writer = self.config.run_writer(file);
        let mut writer = self.codec.keyed_writer(&mut run_writer);
        write(&mut writer)?;
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
        C: EncodeKeyedRun<I> + crate::codec::DeriveKey<I>,
    {
        let run_writer = RunWriter::new_keyed(self.codec, &self.config);
        run_writer.write_sorted(items, |item| item)
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
            codec: self.codec,
            config: self.config,
            primary: key_cmp,
            secondary: self.cmp,
            mode: std::marker::PhantomData,
        };
        let final_runs = reduce_runs(runs, fan_in, |group| context.spill_group(group))?;
        let execution = Execution::new(context, final_runs);
        Ok(KeyedRunMerge { execution })
    }
    pub(crate) fn merge_under_order(
        self,
        runs: Vec<SortedRun>,
    ) -> MergeResult<KeyedRunMerge<C, Cmp, (), crate::sorter::Keyed>, C::Error>
    where
        C: EncodeKeyedRun<<<C as KeyedCodec>::KeyedCursor<RunReader> as CodecCursor>::Record>,
        Cmp: crate::key::KeyedOrder<StoredKey = C::Key>
            + Compare<<C::KeyedCursor<RunReader> as CodecCursor>::Record>,
    {
        let fan_in = self.config.max_fan_in.get();
        let context = KeyedContext {
            codec: self.codec,
            config: self.config,
            primary: self.cmp,
            secondary: (),
            mode: std::marker::PhantomData,
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

struct KeyedContext<C: KeyedCodec, Primary, Secondary, M = ()> {
    codec: C,
    config: MergeConfig,
    primary: Primary,
    secondary: Secondary,
    mode: std::marker::PhantomData<fn() -> M>,
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
        &self.config
    }

    fn open(&self, reader: RunReader) -> Result<Self::Cursor, Self::Error> {
        self.codec.keyed_cursor(reader)
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
        let key_ordering = self.primary.compare(&left.key, &right.key);
        compare_keyed_records(readers, left, right, key_ordering, &self.secondary)
    }
}

impl<C, O> MergeContext for KeyedContext<C, O, (), crate::sorter::Keyed>
where
    C: KeyedCodec,
    O: crate::key::KeyedOrder<StoredKey = C::Key>
        + Compare<<C::KeyedCursor<RunReader> as CodecCursor>::Record>,
{
    type Error = C::Error;
    type Cursor = C::KeyedCursor<RunReader>;
    type Key = C::Key;

    fn config(&self) -> &MergeConfig {
        &self.config
    }

    fn open(&self, reader: RunReader) -> Result<Self::Cursor, Self::Error> {
        self.codec.keyed_cursor(reader)
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
        let order = &self.primary;
        let key_ordering = order.compare_keys(&left.key, &right.key);
        compare_keyed_records(readers, left, right, key_ordering, order)
    }
}

fn compare_keyed_records<R, K, Cmp>(
    readers: &mut [R],
    left: &HeapEntry<K>,
    right: &HeapEntry<K>,
    key_ordering: Ordering,
    item_cmp: &Cmp,
) -> Result<Ordering, R::Error>
where
    R: CodecCursor,
    Cmp: Compare<R::Record>,
{
    if !key_ordering.is_eq() {
        return Ok(key_ordering);
    }
    let ordering = compare_records(readers, left.source, right.source, item_cmp)?;
    Ok(ordering.then_with(|| left.source.cmp(&right.source)))
}

impl<C, KeyCmp, ItemCmp, M> KeyedContext<C, KeyCmp, ItemCmp, M>
where
    C: KeyedCodec
        + EncodeKeyedRun<<<C as KeyedCodec>::KeyedCursor<RunReader> as CodecCursor>::Record>,
    Self: MergeContext<Error = C::Error, Cursor = C::KeyedCursor<RunReader>, Key = C::Key>,
{
    fn spill_group(&self, runs: Vec<SortedRun>) -> MergeResult<SortedRun, C::Error> {
        let named = match &self.config.temp_dir {
            Some(dir) => tempfile::NamedTempFile::new_in(dir)?,
            None => tempfile::NamedTempFile::new()?,
        };
        let file = named.reopen().map_err(MergeError::Io)?;
        let mut run_writer = self.config.run_writer(file);
        let mut writer = self.codec.keyed_writer(&mut run_writer);
        let mut execution = Execution::new(self, runs);
        loop {
            let delivered = execution
                .next_with(|state, winner| {
                    state.with_record(winner, |record| {
                        let entry = crate::key::Keyed::trusted(record, &winner.key);
                        writer.write_keyed(entry)
                    })
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

type KeyedExecution<C, K, I, M> = Execution<
    KeyedContext<C, K, I, M>,
    <C as KeyedCodec>::KeyedCursor<RunReader>,
    <C as KeyedCodec>::Key,
>;

/// Lazy keyed merge over sorted runs spilled to disk.
///
/// Stored keys determine order, with borrowed record comparison on ties.
/// Traversal borrows the winning record; owned iteration additionally requires
/// the keyed cursor's [`TakeOwned`] capability.
pub struct KeyedRunMerge<C: KeyedCodec, KeyCmp, ItemCmp, M = ()> {
    execution: KeyedExecution<C, KeyCmp, ItemCmp, M>,
}

impl<C, KeyCmp, ItemCmp, M> Iterator for KeyedRunMerge<C, KeyCmp, ItemCmp, M>
where
    C: KeyedCodec,
    C::KeyedCursor<RunReader>: TakeOwned,
    KeyedContext<C, KeyCmp, ItemCmp, M>:
        MergeContext<Error = C::Error, Cursor = C::KeyedCursor<RunReader>, Key = C::Key>,
{
    type Item = MergeResult<<C::KeyedCursor<RunReader> as TakeOwned>::Owned, C::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.execution.next_owned()
    }
}

impl<C, KeyCmp, ItemCmp, M> VisitSortedItems for KeyedRunMerge<C, KeyCmp, ItemCmp, M>
where
    C: KeyedCodec,
    KeyedContext<C, KeyCmp, ItemCmp, M>:
        MergeContext<Error = C::Error, Cursor = C::KeyedCursor<RunReader>, Key = C::Key>,
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

        fn write_keyed(&mut self, entry: crate::key::Keyed<&u64, &u8>) -> Result<(), Self::Error> {
            let item = *entry.item();
            let key = *entry.stored_key();
            self.inner.write_all(&[*key])?;
            self.inner.write_all(&item.to_le_bytes())
        }

        fn finish(mut self) -> Result<(), Self::Error> {
            self.inner.flush()
        }
    }

    impl<W: Write> KeyedCodecWriter<U64View, u8> for U64OnlyKeyedWriter<W> {
        type Error = std::io::Error;

        fn write_keyed(
            &mut self,
            entry: crate::key::Keyed<&U64View, &u8>,
        ) -> Result<(), Self::Error> {
            let item = *entry.item();
            let key = *entry.stored_key();
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
        use proptest::test_runner::FileFailurePersistence;

        use super::*;

        // The high bytes identify a record; only the low byte participates in ordering.
        #[derive(Clone, Copy)]
        struct RoundedScoreCodec;

        #[derive(Clone, Copy)]
        struct ScoreOnly;

        impl Compare<u64> for ScoreOnly {
            fn compare(&self, left: &u64, right: &u64) -> Ordering {
                (left & 255).cmp(&(right & 255))
            }
        }

        impl Codec for RoundedScoreCodec {
            type Error = std::io::Error;
            type Cursor<R: Read> = U64KeyedReader<R>;

            fn cursor<R: Read>(&self, source: R) -> Result<Self::Cursor<R>, Self::Error> {
                U64KeyedCodec.cursor(source)
            }
        }

        struct RoundedScoreReader<R: Read>(U64OnlyKeyedReader<R>);

        impl<R: Read> CodecCursor for RoundedScoreReader<R> {
            type Error = std::io::Error;
            type Record = u64;

            fn advance(&mut self) -> Result<bool, Self::Error> {
                let advanced = self.0.advance()?;
                if let Some(record) = &mut self.0.current {
                    *record &= !3;
                }
                Ok(advanced)
            }

            fn with_current<T>(&mut self, f: impl FnOnce(&u64) -> T) -> Result<T, Self::Error> {
                self.0.with_current(f)
            }
        }

        impl<R: Read> TakeOwned for RoundedScoreReader<R> {
            type Owned = u64;

            fn take_owned(&mut self) -> Result<Option<u64>, Self::Error> {
                self.0.take_owned()
            }
        }

        impl<R: Read> KeyedCodecCursor<u8> for RoundedScoreReader<R> {
            fn current_key(&self) -> Result<u8, Self::Error> {
                self.0.current_key()
            }
        }

        impl KeyedCodec for RoundedScoreCodec {
            type Key = u8;
            type KeyedCursor<R: Read> = RoundedScoreReader<R>;

            fn keyed_cursor<R: Read>(
                &self,
                source: R,
            ) -> Result<Self::KeyedCursor<R>, Self::Error> {
                let reader = U64KeyedCodec.keyed_cursor(source)?;
                Ok(RoundedScoreReader(reader))
            }
        }

        impl EncodeKeyedRun<u64> for RoundedScoreCodec {
            type KeyedWriter<W: Write> = U64OnlyKeyedWriter<W>;

            fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
                U64OnlyKeyedWriter {
                    inner: BufWriter::new(dest),
                }
            }
        }

        impl DeriveKey<u64> for RoundedScoreCodec {
            fn derive_key(&self, item: &u64) -> u8 {
                (item & 255) as u8
            }
        }

        fn inspect_scored_run(run: &SortedRun, context: &str) -> Vec<(u64, u8)> {
            let file = run
                .reopen()
                .unwrap_or_else(|error| panic!("{context}: open sorted run file: {error}"));
            let mut cursor = RoundedScoreCodec
                .keyed_cursor(file)
                .unwrap_or_else(|error| panic!("{context}: create keyed run cursor: {error}"));
            let mut entries = Vec::new();

            loop {
                let index = entries.len();
                let advanced = cursor.advance().unwrap_or_else(|error| {
                    panic!("{context}: advance to record {index}: {error}")
                });
                if !advanced {
                    break;
                }

                let key = cursor.current_key().unwrap_or_else(|error| {
                    panic!("{context}: read stored key for record {index}: {error}")
                });
                let record = cursor
                    .with_current(|record| *record)
                    .unwrap_or_else(|error| {
                        panic!("{context}: read record data at index {index}: {error}")
                    });
                entries.push((record, key));
            }

            entries
        }

        fn independently_rounded_record(record: u64) -> u64 {
            let identity = record >> 8;
            let original_score = record & 255;
            let rounded_score = original_score / 4 * 4;

            (identity << 8) | rounded_score
        }

        fn assert_final_scored_records(
            original: &[u64],
            actual: &[u64],
            fan_in: usize,
            count: usize,
        ) {
            let original_scores: Vec<u8> = {
                let mut scores = original
                    .iter()
                    .map(|record| (record & 255) as u8)
                    .collect::<Vec<_>>();
                scores.sort_unstable();
                scores
            };

            let actual_scores: Vec<u8> = actual
                .iter()
                .enumerate()
                .map(|(index, record)| {
                    let identity = (record >> 8) as usize;
                    let original_record = original.get(identity).unwrap_or_else(|| {
                        panic!("read final record {index}: unknown identity {identity}: fan-in={fan_in}, initial runs={count}")
                    });
                    (original_record & 255) as u8
                })
                .collect();

            assert_eq!(
                actual_scores, original_scores,
                "final original-key order: fan-in={fan_in}, initial runs={count}"
            );
            let expected: Vec<u64> = {
                let mut records = original
                    .iter()
                    .copied()
                    .map(independently_rounded_record)
                    .collect::<Vec<_>>();
                records.sort_unstable();
                records
            };
            let actual_sorted = {
                let mut records = actual.to_vec();
                records.sort_unstable();
                records
            };

            assert_eq!(
                actual_sorted, expected,
                "final transformed record multiset: fan-in={fan_in}, initial runs={count}"
            );
        }

        fn scored_input_runs(scores: Vec<Vec<u8>>) -> Vec<Vec<u64>> {
            let mut next_identity = 0_u64;
            let mut runs = Vec::new();

            for mut batch in scores {
                // Every run contains the same non-multiple of four.
                batch.push(1);
                batch.sort_unstable();

                let first_identity = next_identity;
                next_identity += batch.len() as u64;
                let records = batch
                    .into_iter()
                    .enumerate()
                    .map(|(index, score)| {
                        let identity = first_identity + index as u64;
                        (identity << 8) | u64::from(score)
                    })
                    .collect();
                runs.push(records);
            }

            runs
        }

        fn assert_intermediate_scored_run(
            original: &[u64],
            expected: Vec<(u64, u8)>,
            actual: Vec<(u64, u8)>,
            label: &str,
        ) {
            for (record_index, &(record, key)) in actual.iter().enumerate() {
                let identity = (record >> 8) as usize;
                let original_record = original.get(identity).unwrap_or_else(|| {
                    panic!("{label}: unknown identity at record {record_index}: {identity}")
                });
                let original_score = original_record & 255;
                assert_eq!(
                    u64::from(key),
                    original_score,
                    "{label}: original key at record {record_index}, identity={identity}"
                );
            }

            let keys_are_ordered = actual.windows(2).all(|pair| pair[0].1 <= pair[1].1);
            assert!(
                keys_are_ordered,
                "{label}: intermediate records are not in stored key order"
            );

            let expected = {
                let mut entries = expected;
                entries.sort_unstable();
                entries
            };
            let actual = {
                let mut entries = actual;
                entries.sort_unstable();
                entries
            };
            assert_eq!(actual, expected, "{label}: record and key multiset");
        }

        fn assert_original_keys_through_two_passes(fan_in: usize, scores: Vec<Vec<u8>>) {
            let count = fan_in * fan_in + 1;
            let config = merge_config(fan_in, count);
            let merger = RunMerger::new(RoundedScoreCodec, ScoreOnly, config.clone());
            let batches = scored_input_runs(scores);
            let original: Vec<u64> = batches.iter().flatten().copied().collect();
            let runs: Vec<SortedRun> = batches
                .into_iter()
                .enumerate()
                .map(|(run_index, records)| {
                    merger.spill_sorted_with_keys(records).unwrap_or_else(|error| {
                        panic!("spill input run {run_index}: fan-in={fan_in}, initial runs={count}: {error:?}")
                    })
                })
                .collect();

            assert_eq!(
                runs.len(),
                count,
                "initial run count: fan-in={fan_in}, initial runs={count}"
            );
            let context = KeyedContext {
                codec: RoundedScoreCodec,
                config,
                primary: Natural,
                secondary: ScoreOnly,
                mode: std::marker::PhantomData::<fn() -> ()>,
            };

            let mut writes = 0;
            let reduced = reduce_runs(runs, fan_in, |group| {
                // f²+1 inputs produce f+1 writes in pass one, then two in pass two.
                let pass = if writes < fan_in + 1 { 1 } else { 2 };
                let run_index = if pass == 1 {
                    writes
                } else {
                    writes - fan_in - 1
                };
                let label = format!(
                    "intermediate pass {pass}, run {run_index}: fan-in={fan_in}, initial runs={count}"
                );

                let source = if pass == 1 { "initial" } else { "previous intermediate" };
                let expected: Vec<_> = group
                    .iter()
                    .enumerate()
                    .flat_map(|(input_index, run)| {
                        let input_label = format!("{label}: read {source} input file {input_index}");
                        inspect_scored_run(run, &input_label)
                    })
                    .collect();

                let run = context
                    .spill_group(group)
                    .unwrap_or_else(|error| panic!("{label}: write intermediate sorted run: {error:?}"));

                let output_label = format!("{label}: read newly written intermediate file");
                let actual = inspect_scored_run(&run, &output_label);
                assert_intermediate_scored_run(&original, expected, actual, &label);

                writes += 1;
                Ok(run)
            })
            .unwrap_or_else(|error: MergeError<std::io::Error>| {
                panic!("merge input sorted runs into intermediate runs: fan-in={fan_in}, initial runs={count}: {error:?}")
            });

            assert_eq!(
                writes,
                fan_in + 3,
                "two intermediate passes: fan-in={fan_in}, initial runs={count}"
            );
            assert_eq!(
                reduced.len(),
                2,
                "runs after two intermediate passes: fan-in={fan_in}, initial runs={count}"
            );
            let output = merger
                .merge_keyed(reduced, Natural)
                .unwrap_or_else(|error| {
                    panic!("start final merge: fan-in={fan_in}, initial runs={count}: {error:?}")
                });
            let actual: Vec<u64> = output
                .enumerate()
                .map(|(index, result)| {
                    result.unwrap_or_else(|error| {
                        panic!("read final record {index}: fan-in={fan_in}, initial runs={count}: {error:?}")
                    })
                })
                .collect();

            assert_final_scored_records(&original, &actual, fan_in, count);
        }

        #[test]
        fn original_keys_survive_two_passes_for_fan_in_two_three_and_four() {
            for fan_in in 2..=4 {
                let count = fan_in * fan_in + 1;
                let scores = vec![vec![0, 3, 4, 7]; count];
                assert_original_keys_through_two_passes(fan_in, scores);
            }
        }

        #[derive(Clone, Copy, Debug)]
        enum InitialRunCount {
            OneBelowFanIn,
            EqualToFanIn,
            OneAboveFanIn,
            EqualToFanInSquared,
            OneAboveFanInSquared,
        }

        fn initial_run_count(fan_in: usize, scenario: InitialRunCount) -> usize {
            match scenario {
                InitialRunCount::OneBelowFanIn => fan_in - 1,
                InitialRunCount::EqualToFanIn => fan_in,
                InitialRunCount::OneAboveFanIn => fan_in + 1,
                InitialRunCount::EqualToFanInSquared => fan_in * fan_in,
                InitialRunCount::OneAboveFanInSquared => fan_in * fan_in + 1,
            }
        }

        fn sorted_input_runs(
            scenario: InitialRunCount,
        ) -> impl Strategy<Value = (usize, Vec<Vec<u64>>)> {
            (2usize..=4).prop_flat_map(move |fan_in| {
                let count = initial_run_count(fan_in, scenario);
                let value = prop_oneof![8 => 0u64..8, 1 => Just(u64::MAX), 1 => 0u64..=u64::MAX];

                let batch = proptest::collection::vec(value, 1..=16).prop_map(|mut values| {
                    values.sort_unstable();
                    values
                });

                proptest::collection::vec(batch, count).prop_map(move |batches| (fan_in, batches))
            })
        }

        fn merge_config(fan_in: usize, initial_count: usize) -> MergeConfig {
            let max_fan_in = NonZeroUsize::new(fan_in).unwrap_or_else(|| {
                panic!(
                    "configure merge: fan-in {fan_in} must be nonzero; initial runs={initial_count}"
                )
            });
            MergeConfig {
                max_fan_in,
                ..MergeConfig::default()
            }
        }

        fn spill_runs(
            merger: &RunMerger<U64Codec, Natural>,
            batches: &[Vec<u64>],
            fan_in: usize,
        ) -> Vec<SortedRun> {
            let initial_count = batches.len();
            batches
                .iter()
                .enumerate()
                .map(|(run_index, batch)| {
                    merger
                        .spill_sorted(batch.iter().copied())
                        .unwrap_or_else(|error| {
                            panic!("spill input run {run_index}: fan-in={fan_in}, initial runs={initial_count}: {error:?}")
                        })
                })
                .collect()
        }

        fn collect_merged_values(
            merger: RunMerger<U64Codec, Natural>,
            runs: Vec<SortedRun>,
            fan_in: usize,
            initial_count: usize,
        ) -> Vec<u64> {
            let output = merger
                .merge(runs)
                .unwrap_or_else(|error| {
                    panic!("failed to start merging sorted runs: fan-in={fan_in}, initial runs={initial_count}: {error:?}")
                });

            output
                .enumerate()
                .map(|(record_index, result)| {
                    result.unwrap_or_else(|error| {
                        panic!("read merged output record {record_index}: fan-in={fan_in}, initial runs={initial_count}: {error:?}")
                    })
                })
                .collect()
        }

        fn assert_merge_preserves_sorted_input_values(
            fan_in: usize,
            batches: &[Vec<u64>],
        ) -> Result<(), TestCaseError> {
            let initial_count = batches.len();
            let expected: Vec<u64> = {
                let mut values = batches.iter().flatten().copied().collect::<Vec<_>>();
                values.sort_unstable();
                values
            };

            let config = merge_config(fan_in, initial_count);
            let merger = RunMerger::new(U64Codec, Natural, config);
            let runs = spill_runs(&merger, batches, fan_in);

            prop_assert_eq!(
                runs.len(),
                initial_count,
                "spilled run count: fan-in={}, initial runs={}",
                fan_in,
                initial_count
            );
            let actual = collect_merged_values(merger, runs, fan_in, initial_count);

            prop_assert_eq!(
                actual,
                expected,
                "merged values differ from sorted input values: fan-in={}, initial runs={}",
                fan_in,
                initial_count
            );

            Ok(())
        }

        #[test]
        fn merge_preserves_all_values_including_when_input_runs_are_empty() {
            let scenarios = [
                InitialRunCount::OneBelowFanIn,
                InitialRunCount::EqualToFanIn,
                InitialRunCount::OneAboveFanIn,
                InitialRunCount::EqualToFanInSquared,
                InitialRunCount::OneAboveFanInSquared,
            ];

            for fan_in in 2..=4 {
                for scenario in scenarios {
                    let count = initial_run_count(fan_in, scenario);
                    let batches = vec![vec![0, 1, 1, u64::MAX]; count];
                    assert_merge_preserves_sorted_input_values(fan_in, &batches)
                        .unwrap_or_else(|error| panic!("check initial run count scenario {scenario:?}: fan-in={fan_in}, initial runs={count}: {error}"));
                }

                for batches in [
                    vec![],
                    vec![vec![], vec![]],
                    vec![vec![], vec![0, 1], vec![]],
                ] {
                    let count = batches.len();
                    assert_merge_preserves_sorted_input_values(fan_in, &batches)
                        .unwrap_or_else(|error| panic!("check empty input runs: fan-in={fan_in}, initial runs={count}: {error}"));
                }

                let mut mixed = vec![vec![]; fan_in * fan_in + 1];
                mixed[fan_in] = vec![0, 1, 1, u64::MAX];
                let count = mixed.len();
                assert_merge_preserves_sorted_input_values(fan_in, &mixed)
                    .unwrap_or_else(|error| panic!("check mixed empty runs through intermediate merges: fan-in={fan_in}, initial runs={count}: {error}"));
            }
        }

        #[test]
        fn intermediate_merge_passes_respect_fan_in_and_expected_run_counts() {
            let scenarios = [
                InitialRunCount::OneBelowFanIn,
                InitialRunCount::EqualToFanIn,
                InitialRunCount::OneAboveFanIn,
                InitialRunCount::EqualToFanInSquared,
                InitialRunCount::OneAboveFanInSquared,
            ];

            for fan_in in 2..=4 {
                for scenario in scenarios {
                    let count = initial_run_count(fan_in, scenario);
                    let batches = vec![vec![1]; count];
                    let config = merge_config(fan_in, count);
                    let merger = RunMerger::new(U64Codec, Natural, config);
                    let runs = spill_runs(&merger, &batches, fan_in);

                    let mut group_sizes = Vec::new();
                    let reduced = reduce_runs(runs, fan_in, |group| {
                        group_sizes.push(group.len());
                        merger.spill_group(group)
                    })
                    .unwrap_or_else(|error| panic!("failed to write intermediate sorted run: fan-in={fan_in}, initial runs={count}, input run counts for each intermediate merge={group_sizes:?}: {error:?}"));

                    let (expected_groups, expected_count) = match scenario {
                        InitialRunCount::OneBelowFanIn | InitialRunCount::EqualToFanIn => {
                            (vec![], count)
                        }
                        InitialRunCount::OneAboveFanIn => (vec![fan_in, 1], 2),
                        InitialRunCount::EqualToFanInSquared => (vec![fan_in; fan_in], fan_in),
                        InitialRunCount::OneAboveFanInSquared => {
                            let mut groups = vec![fan_in; fan_in];
                            groups.extend([1, fan_in, 1]);
                            (groups, 2)
                        }
                    };

                    assert_eq!(
                        group_sizes, expected_groups,
                        "input run counts for each intermediate merge: fan-in={fan_in}, initial runs={count}, scenario={scenario:?}"
                    );
                    assert_eq!(
                        reduced.len(),
                        expected_count,
                        "runs after intermediate merges: fan-in={fan_in}, initial runs={count}, scenario={scenario:?}"
                    );
                    if fan_in == 2 && count == 5 {
                        assert_eq!(
                            group_sizes,
                            [2, 2, 1, 2, 1],
                            "two intermediate merge passes: fan-in={fan_in}, initial runs={count}"
                        );
                    }

                    let output = collect_merged_values(merger, reduced, fan_in, count);
                    assert_eq!(
                        output,
                        vec![1; count],
                        "intermediate and final merges preserve values: fan-in={fan_in}, initial runs={count}"
                    );
                }
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 32,
                max_shrink_iters: 512,
                failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
                    env!("CARGO_MANIFEST_DIR"), "/proptest-regressions/merge.txt"
                )))),
                ..ProptestConfig::default()
            })]

            #[test]
            fn original_keys_stay_with_records_through_two_intermediate_passes(
                (fan_in, scores) in (2usize..=4).prop_flat_map(|fan_in| {
                    let run_count = fan_in * fan_in + 1;
                    let run = proptest::collection::vec(0u8..64, 0..4);
                    proptest::collection::vec(run, run_count)
                        .prop_map(move |scores| (fan_in, scores))
                })
            ) {
                assert_original_keys_through_two_passes(fan_in, scores);
            }

            #[test]
            fn merge_preserves_values_with_one_fewer_run_than_fan_in((fan_in, batches) in sorted_input_runs(InitialRunCount::OneBelowFanIn)) {
                assert_merge_preserves_sorted_input_values(fan_in, &batches)?;
            }

            #[test]
            fn merge_preserves_values_with_runs_equal_to_fan_in((fan_in, batches) in sorted_input_runs(InitialRunCount::EqualToFanIn)) {
                assert_merge_preserves_sorted_input_values(fan_in, &batches)?;
            }

            #[test]
            fn merge_preserves_values_with_one_more_run_than_fan_in((fan_in, batches) in sorted_input_runs(InitialRunCount::OneAboveFanIn)) {
                assert_merge_preserves_sorted_input_values(fan_in, &batches)?;
            }

            #[test]
            fn merge_preserves_values_with_runs_equal_to_fan_in_squared((fan_in, batches) in sorted_input_runs(InitialRunCount::EqualToFanInSquared)) {
                assert_merge_preserves_sorted_input_values(fan_in, &batches)?;
            }

            #[test]
            fn merge_preserves_values_with_one_more_run_than_fan_in_squared((fan_in, batches) in sorted_input_runs(InitialRunCount::OneAboveFanInSquared)) {
                assert_merge_preserves_sorted_input_values(fan_in, &batches)?;
            }
        }
    }
}
