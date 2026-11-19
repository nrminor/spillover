//! The external sorter — the main entry point for sorting
//! larger-than-memory datasets.
//!
//! [`Sorter`] ties together all the trait axes ([`SortKey`],
//! [`Compare`], [`Codec`], [`Dedup`], [`ChunkSorter`]) and the
//! merge engine into a single, configurable pipeline. Construct
//! one via the type-state [`Builder`], push items into it, and
//! call [`finish`](Sorter::finish) to get a finalized sorted output
//! handle.
//!
//! [`SortKey`]: crate::key::SortKey
//! [`Compare`]: crate::compare::Compare
//! [`Codec`]: crate::codec::Codec
//! [`Dedup`]: crate::dedup::Dedup
//! [`ChunkSorter`]: crate::chunk::ChunkSorter

use std::{fs::File, io::BufReader};

use get_size2::GetSize;

use crate::{
    SortedItemsError,
    chunk::{ChunkSorter, Sequential},
    codec::{Codec, CodecCursor, EncodeKeyedRun, EncodeRun, KeyedCodec},
    compare::{Compare, Natural},
    dedup::{Dedup, Identity},
    key::{KeyCompare, SortKey},
    merge::{KeyedRunMerge, MergeConfig, MergeError, RunMerge, RunMerger, RunWriter, SortedRun},
};

/// How the sorter decides when to flush the in-memory buffer to
/// disk.
enum FlushStrategy {
    /// Flush when the estimated byte usage exceeds a budget.
    Bytes { budget: usize },

    /// Flush when the item count exceeds a limit.
    Items { max_items: usize },
}

/// Configuration for the [`Sorter`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SorterConfig {
    /// Configuration for the merge engine.
    pub merge: MergeConfig,
}

// ── Type-state markers for the builder ───────────────────────

/// Marker: no sort key provided yet.
pub struct NeedsSortKey;
/// Marker: sort key has been provided.
pub struct HasSortKey<SK>(SK);

/// Marker: no codec provided yet.
pub struct NeedsCodec;
/// Marker: codec has been provided for the unkeyed merge path.
pub struct HasCodec<Cod>(Cod);
/// Marker: keyed codec has been provided (keyed merge path).
pub struct HasKeyedCodec<Cod>(Cod);

/// Marker: no flush strategy provided yet.
pub struct NeedsFlushStrategy;
/// Marker: flush strategy has been provided.
pub struct HasFlushStrategy<T>(FlushStrategy, MaybeItemSize<T>);

/// Marker: unkeyed merge path — temporary runs store records without record keys.
pub struct Unkeyed;
/// Marker: keyed merge path — codec implements [`KeyedCodec`],
/// enabling key-first comparisons during merge with fallback
/// full-record comparison when keys tie.
pub struct Keyed;

/// Type-state builder for [`Sorter`].
pub struct Builder<SK, Cod, Flush, Cmp = Natural, D = Identity, CS = Sequential> {
    sort_key: SK,
    codec: Cod,
    flush: Flush,
    compare: Cmp,
    dedup: D,
    chunk_sort: CS,
    config: SorterConfig,
}

impl Builder<NeedsSortKey, NeedsCodec, NeedsFlushStrategy> {
    /// Start building a new [`Sorter`].
    #[must_use]
    pub fn new() -> Self {
        Builder {
            sort_key: NeedsSortKey,
            codec: NeedsCodec,
            flush: NeedsFlushStrategy,
            compare: Natural,
            dedup: Identity,
            chunk_sort: Sequential,
            config: SorterConfig::default(),
        }
    }
}

impl Default for Builder<NeedsSortKey, NeedsCodec, NeedsFlushStrategy> {
    fn default() -> Self {
        Self::new()
    }
}

impl<SK, Cod, Flush, Cmp, D, CS> Builder<SK, Cod, Flush, Cmp, D, CS> {
    /// Set the sort key extractor.
    #[must_use]
    pub fn key<SK2>(self, sort_key: SK2) -> Builder<HasSortKey<SK2>, Cod, Flush, Cmp, D, CS> {
        Builder {
            sort_key: HasSortKey(sort_key),
            codec: self.codec,
            flush: self.flush,
            compare: self.compare,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }

    /// Set the codec (unkeyed merge path).
    #[must_use]
    pub fn codec<Cod2>(self, codec: Cod2) -> Builder<SK, HasCodec<Cod2>, Flush, Cmp, D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: HasCodec(codec),
            flush: self.flush,
            compare: self.compare,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }

    /// Set a keyed codec (keyed merge path).
    #[must_use]
    pub fn keyed_codec<Cod2>(
        self,
        codec: Cod2,
    ) -> Builder<SK, HasKeyedCodec<Cod2>, Flush, Cmp, D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: HasKeyedCodec(codec),
            flush: self.flush,
            compare: self.compare,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }

    /// Set the comparator. Defaults to [`Natural`].
    #[must_use]
    pub fn compare<Cmp2>(self, compare: Cmp2) -> Builder<SK, Cod, Flush, Cmp2, D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: self.flush,
            compare,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }

    /// Set the dedup strategy. Defaults to [`Identity`].
    #[must_use]
    pub fn dedup<D2>(self, dedup: D2) -> Builder<SK, Cod, Flush, Cmp, D2, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: self.flush,
            compare: self.compare,
            dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }

    /// Set the chunk sorting strategy. Defaults to [`Sequential`].
    #[must_use]
    pub fn chunk_sort<CS2>(self, chunk_sort: CS2) -> Builder<SK, Cod, Flush, Cmp, D, CS2> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: self.flush,
            compare: self.compare,
            dedup: self.dedup,
            chunk_sort,
            config: self.config,
        }
    }

    /// Override the merge configuration.
    #[must_use]
    pub fn merge_config(mut self, merge: MergeConfig) -> Self {
        self.config.merge = merge;
        self
    }
}

// Flush strategy methods.

impl<SK, Cod, Cmp, D, CS> Builder<SK, Cod, NeedsFlushStrategy, Cmp, D, CS> {
    /// Byte-based budget with a custom sizing function.
    #[must_use]
    pub fn memory_budget<T: 'static>(
        self,
        budget: usize,
        item_size: impl Fn(&T) -> usize + Send + Sync + 'static,
    ) -> Builder<SK, Cod, HasFlushStrategy<T>, Cmp, D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: HasFlushStrategy(FlushStrategy::Bytes { budget }, Some(Box::new(item_size))),
            compare: self.compare,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }

    /// Byte-based budget for types implementing [`GetSize`].
    #[must_use]
    pub fn measured_budget<T: GetSize + 'static>(
        self,
        budget: usize,
    ) -> Builder<SK, Cod, HasFlushStrategy<T>, Cmp, D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: HasFlushStrategy(
                FlushStrategy::Bytes { budget },
                Some(Box::new(GetSize::get_size)),
            ),
            compare: self.compare,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }

    /// Byte-based budget using `size_of` (fixed-size types only).
    #[must_use]
    pub fn fixed_size_budget<T: 'static>(
        self,
        budget: usize,
    ) -> Builder<SK, Cod, HasFlushStrategy<T>, Cmp, D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: HasFlushStrategy(
                FlushStrategy::Bytes { budget },
                Some(Box::new(|_| std::mem::size_of::<T>())),
            ),
            compare: self.compare,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }

    /// Count-based buffer limit.
    #[must_use]
    pub fn max_buffer_items<T>(
        self,
        max_items: usize,
    ) -> Builder<SK, Cod, HasFlushStrategy<T>, Cmp, D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: HasFlushStrategy(FlushStrategy::Items { max_items }, None),
            compare: self.compare,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }
}

// build() for unkeyed merge path.
impl<T, SK, Cod, Cmp, D, CS> Builder<HasSortKey<SK>, HasCodec<Cod>, HasFlushStrategy<T>, Cmp, D, CS>
where
    SK: SortKey<T> + Copy,
    Cod: Codec + Copy,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy,
    CS: ChunkSorter<T>,
{
    /// Build the [`Sorter`] (unkeyed merge path).
    #[must_use]
    pub fn build(self) -> Sorter<T, SK, Cod, Cmp, D, CS, Unkeyed> {
        Sorter {
            session: Session::new(
                OwnedBuffer::new(self.flush.1),
                RunAccumulator::new(
                    self.sort_key.0,
                    self.codec.0,
                    self.compare,
                    self.chunk_sort,
                    self.config.merge,
                ),
            ),
            dedup: self.dedup,
            flush: self.flush.0,
            _marker: std::marker::PhantomData,
        }
    }
}

// build() for keyed path.
impl<T, SK, Cod, Cmp, D, CS>
    Builder<HasSortKey<SK>, HasKeyedCodec<Cod>, HasFlushStrategy<T>, Cmp, D, CS>
where
    SK: SortKey<T> + Copy,
    Cod: KeyedCodec + Copy,
    Cmp: for<'a> Compare<SK::Key<'a>> + Compare<Cod::Key> + Copy,
    CS: ChunkSorter<T>,
{
    /// Build the [`Sorter`] (keyed merge path).
    #[must_use]
    pub fn build(self) -> Sorter<T, SK, Cod, Cmp, D, CS, Keyed> {
        Sorter {
            session: Session::new(
                OwnedBuffer::new(self.flush.1),
                RunAccumulator::new(
                    self.sort_key.0,
                    self.codec.0,
                    self.compare,
                    self.chunk_sort,
                    self.config.merge,
                ),
            ),
            dedup: self.dedup,
            flush: self.flush.0,
            _marker: std::marker::PhantomData,
        }
    }
}

/// A disk-spilling external sorter.
///
/// Push items via [`push`](Self::push), then call
/// [`finish`](Self::finish) to flush remaining items, merge all
/// sorted runs, apply deduplication, and produce a sorted output
/// handle.
///
/// Buffered inputs and decoded records may have different representations.
/// The sort key must support both the input and the selected cursor's record.
/// Owned iteration additionally requires the cursor's
/// [`TakeOwned`](crate::codec::TakeOwned) capability.
///
/// - `T`: the item type being sorted
/// - `SK`: [`SortKey`] — extracts the value to sort by from each
///   item (e.g. sequence bytes, a name, a length)
/// - `Cod`: [`Codec`] — serializes items to/from temporary files
///   on disk. On the keyed path ([`KeyedCodec`]), the codec also
///   stores a compact *record key* alongside each item for merge
///   acceleration
/// - `Cmp`: [`Compare`] — the ordering relation, applied to both
///   sort keys (during chunk sort) and record keys (during merge).
///   These are different representations of the same underlying
///   data, so the comparator must handle both types
/// - `D`: [`Dedup`] — post-merge deduplication
/// - `CS`: [`ChunkSorter`] — in-memory sort algorithm and
///   threading model
/// - `M`: merge strategy marker ([`Unkeyed`] or [`Keyed`])
pub struct Sorter<T, SK, Cod, Cmp, D, CS, M = Unkeyed> {
    session: SorterSession<T, SK, Cod, Cmp, CS, M>,
    dedup: D,
    flush: FlushStrategy,
    _marker: std::marker::PhantomData<M>,
}

type SorterSession<T, SK, Cod, Cmp, CS, M> =
    Session<OwnedBuffer<T>, RunAccumulator<T, SK, Cod, Cmp, CS, M>>;

type ItemSize<T> = Box<dyn Fn(&T) -> usize + Send + Sync>;
type MaybeItemSize<T> = Option<ItemSize<T>>;

struct OwnedBuffer<T> {
    items: Vec<T>,
    active_bytes: usize,
    item_size: MaybeItemSize<T>,
}

impl<T> OwnedBuffer<T> {
    fn new(item_size: MaybeItemSize<T>) -> Self {
        Self {
            items: Vec::new(),
            active_bytes: 0,
            item_size,
        }
    }
}

impl<T> Extend<T> for OwnedBuffer<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for item in iter {
            if let Some(item_size) = &self.item_size {
                self.active_bytes += item_size(&item);
            }
            self.items.push(item);
        }
    }
}

trait ChunkBuffer {
    type Item;
    type Items: AsMut<[Self::Item]>;

    fn items(self) -> Self::Items;
    fn is_empty(&self) -> bool;
    fn clear(self);
}

impl<'items, T> ChunkBuffer for &'items mut OwnedBuffer<T> {
    type Item = T;
    type Items = &'items mut [T];

    fn items(self) -> Self::Items {
        &mut self.items
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn clear(self) {
        self.items.clear();
        self.active_bytes = 0;
    }
}

trait FlushPolicy<B, I> {
    fn should_flush_before_push(&self, buffer: &B, input: &I) -> bool;
    fn should_flush_after_push(&self, buffer: &B) -> bool;
}

impl<T> FlushPolicy<OwnedBuffer<T>, T> for FlushStrategy {
    fn should_flush_before_push(&self, _buffer: &OwnedBuffer<T>, _input: &T) -> bool {
        false
    }

    fn should_flush_after_push(&self, buffer: &OwnedBuffer<T>) -> bool {
        match self {
            FlushStrategy::Bytes { budget } => buffer.active_bytes >= *budget,
            FlushStrategy::Items { max_items } => buffer.items.len() >= *max_items,
        }
    }
}

struct Session<B, A> {
    buffer: B,
    accumulator: A,
}

impl<B, A> Session<B, A> {
    fn new(buffer: B, accumulator: A) -> Self {
        Self {
            buffer,
            accumulator,
        }
    }
}

trait AccumulateItems<I> {
    type Error;
    type Final;

    fn accumulate(&mut self, items: &mut [I]) -> Result<(), Self::Error>;
    fn finish(self) -> Result<Self::Final, Self::Error>;
}

impl<B, A> Session<B, A>
where
    for<'items> &'items mut B: ChunkBuffer,
{
    fn commit<E>(&mut self) -> Result<(), E>
    where
        for<'items> A: AccumulateItems<<&'items mut B as ChunkBuffer>::Item, Error = E>,
    {
        let is_empty = {
            let buffer = &mut self.buffer;
            ChunkBuffer::is_empty(&buffer)
        };

        if is_empty {
            return Ok(());
        }

        {
            let mut items = ChunkBuffer::items(&mut self.buffer);
            self.accumulator.accumulate(items.as_mut())?;
        }

        ChunkBuffer::clear(&mut self.buffer);
        Ok(())
    }

    fn finish<E, F>(mut self) -> Result<F, E>
    where
        for<'items> A: AccumulateItems<<&'items mut B as ChunkBuffer>::Item, Error = E, Final = F>,
    {
        self.commit()?;
        self.accumulator.finish()
    }
}

struct RunAccumulator<T, SK, Cod, Cmp, CS, M> {
    sort_key: SK,
    codec: Cod,
    compare: Cmp,
    chunk_sort: CS,
    merge_config: MergeConfig,
    runs: Vec<SortedRun>,
    _marker: std::marker::PhantomData<fn() -> (T, M)>,
}

impl<T, SK, Cod, Cmp, CS, M> RunAccumulator<T, SK, Cod, Cmp, CS, M> {
    fn new(sort_key: SK, codec: Cod, compare: Cmp, chunk_sort: CS, config: MergeConfig) -> Self {
        Self {
            sort_key,
            codec,
            compare,
            chunk_sort,
            merge_config: config,
            runs: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, SK, Cod, Cmp, CS> AccumulateItems<T> for RunAccumulator<T, SK, Cod, Cmp, CS, Unkeyed>
where
    T: 'static,
    SK: SortKey<T> + Copy + Send + Sync + 'static,
    Cod: Codec
        + EncodeRun<T>
        + EncodeRun<<<Cod as Codec>::Cursor<BufReader<File>> as CodecCursor>::Record>
        + Copy
        + 'static,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy + Send + Sync + 'static,
    KeyCompare<SK, Cmp>: Compare<<<Cod as Codec>::Cursor<BufReader<File>> as CodecCursor>::Record>,
    CS: ChunkSorter<T>,
{
    type Error = MergeError<Cod::Error>;
    type Final = RunMerge<Cod, KeyCompare<SK, Cmp>>;

    fn accumulate(&mut self, items: &mut [T]) -> Result<(), MergeError<Cod::Error>> {
        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        let compare_items = move |left: &T, right: &T| {
            <KeyCompare<SK, Cmp> as Compare<T>>::compare(&item_cmp, left, right)
        };
        self.chunk_sort.sort(items, compare_items);

        let run_writer = RunWriter::new_unkeyed(self.codec, self.merge_config.clone());
        let run = run_writer.write_sorted(items)?;
        self.runs.push(run);

        Ok(())
    }

    fn finish(self) -> Result<Self::Final, MergeError<Cod::Error>> {
        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        let run_merger = RunMerger::new(self.codec, item_cmp, self.merge_config);

        run_merger.merge(self.runs)
    }
}

impl<T, SK, Cod, Cmp, CS> AccumulateItems<T> for RunAccumulator<T, SK, Cod, Cmp, CS, Keyed>
where
    T: 'static,
    SK: SortKey<T> + Copy + Send + Sync + 'static,
    Cod: KeyedCodec
        + EncodeKeyedRun<T>
        + EncodeKeyedRun<<<Cod as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>
        + Copy
        + 'static,
    Cmp: for<'a> Compare<SK::Key<'a>> + Compare<Cod::Key> + Copy + Send + Sync + 'static,
    KeyCompare<SK, Cmp>:
        Compare<<<Cod as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>,
    CS: ChunkSorter<T>,
{
    type Error = MergeError<Cod::Error>;
    type Final = KeyedRunMerge<Cod, Cmp, KeyCompare<SK, Cmp>>;

    fn accumulate(&mut self, items: &mut [T]) -> Result<(), MergeError<Cod::Error>> {
        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        let compare_items = move |left: &T, right: &T| {
            <KeyCompare<SK, Cmp> as Compare<T>>::compare(&item_cmp, left, right)
        };
        self.chunk_sort.sort(items, compare_items);

        let run_writer = RunWriter::new_keyed(self.codec, self.merge_config.clone());
        let run = run_writer.write_sorted(items)?;
        self.runs.push(run);

        Ok(())
    }

    fn finish(self) -> Result<Self::Final, MergeError<Cod::Error>> {
        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        let run_merger = RunMerger::new(self.codec, item_cmp, self.merge_config);

        run_merger.merge_keyed(self.runs, self.compare)
    }
}

/// Finalized sorted output.
///
/// `Sorted` owns the state needed to produce sorted items after a
/// [`Sorter`] has been finished. It implements [`Iterator`] for the
/// owned materialization path, so ordinary collection remains the
/// primary way to collect sorted output into memory:
///
/// ```ignore
/// let items: Vec<_> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
/// ```
pub struct Sorted<I> {
    source: I,
}

/// Adapter for visiting sorted output items without using the owned
/// [`Iterator`] namespace on [`Sorted`].
///
/// Constructed by [`Sorted::items`]. Like standard iterator adapter structs,
/// this type is public and nameable but has private fields.
pub struct SortedItems<S> {
    source: S,
}

/// A sorted source that can be consumed through borrowed record visits.
pub trait VisitSortedItems {
    /// The record representation passed to the item sink.
    type Record: ?Sized;

    /// The source error type.
    type Error;

    /// Visit each item in source order.
    ///
    /// # Errors
    ///
    /// Returns [`SortedItemsError::Source`] if producing a record fails, or
    /// [`SortedItemsError::Sink`] if the callback fails.
    fn visit_items<F, FE>(self, f: F) -> Result<(), SortedItemsError<Self::Error, FE>>
    where
        F: FnMut(&Self::Record) -> Result<(), FE>;
}

impl<I> Sorted<I> {
    fn new(source: I) -> Self {
        Self { source }
    }

    /// Consume this sorted output handle and visit items through the
    /// current-item traversal namespace.
    pub fn items(self) -> SortedItems<I> {
        SortedItems {
            source: self.source,
        }
    }
}

impl<S> SortedItems<S>
where
    S: VisitSortedItems,
{
    /// Visit every sorted item in order.
    ///
    /// This method lives under the [`Sorted::items`] namespace so it does not
    /// conflict with owned [`Iterator`] traversal on [`Sorted`].
    ///
    /// # Errors
    ///
    /// Returns [`SortedItemsError::Source`] if the sorted source fails, or
    /// [`SortedItemsError::Sink`] if the callback fails.
    pub fn try_for_each<F, FE>(self, f: F) -> Result<(), SortedItemsError<S::Error, FE>>
    where
        F: FnMut(&S::Record) -> Result<(), FE>,
    {
        self.source.visit_items(f)
    }
}

impl<I> Iterator for Sorted<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next()
    }
}

impl<I> VisitSortedItems for Sorted<I>
where
    I: VisitSortedItems,
{
    type Record = I::Record;
    type Error = I::Error;

    fn visit_items<F, FE>(self, f: F) -> Result<(), SortedItemsError<Self::Error, FE>>
    where
        F: FnMut(&Self::Record) -> Result<(), FE>,
    {
        self.source.visit_items(f)
    }
}

// ── Ordinary record path: push + flush + finish ───────────

impl<T, SK, Cod, Cmp, D, CS> Sorter<T, SK, Cod, Cmp, D, CS, Unkeyed>
where
    T: 'static,
    SK: SortKey<T> + Copy + Send + Sync + 'static,
    Cod: Codec
        + EncodeRun<T>
        + EncodeRun<<<Cod as Codec>::Cursor<BufReader<File>> as CodecCursor>::Record>
        + Copy
        + 'static,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy + Send + Sync + 'static,
    KeyCompare<SK, Cmp>: Compare<<<Cod as Codec>::Cursor<BufReader<File>> as CodecCursor>::Record>,
    CS: ChunkSorter<T>,
{
    /// Add an item to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing to disk fails.
    pub fn push(&mut self, item: T) -> Result<(), MergeError<Cod::Error>> {
        if self
            .flush
            .should_flush_before_push(&self.session.buffer, &item)
        {
            self.session.commit()?;
        }

        self.session.buffer.extend([item]);

        if self.flush.should_flush_after_push(&self.session.buffer) {
            self.session.commit()?;
        }

        Ok(())
    }

    /// Flush remaining items, merge all sorted runs, apply dedup,
    /// and return finalized sorted output.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or merging fails.
    ///
    #[allow(clippy::type_complexity)]
    pub fn finish(
        self,
    ) -> Result<
        Sorted<<D as Dedup<RunMerge<Cod, KeyCompare<SK, Cmp>>>>::Deduped>,
        MergeError<Cod::Error>,
    >
    where
        D: Dedup<RunMerge<Cod, KeyCompare<SK, Cmp>>>,
    {
        let dedup = self.dedup;
        let merged = self.session.finish()?;

        Ok(Sorted::new(dedup.dedup(merged)))
    }
}

// ── Keyed path: push + flush + finish ────────────────────

impl<T, SK, Cod, Cmp, D, CS> Sorter<T, SK, Cod, Cmp, D, CS, Keyed>
where
    T: 'static,
    SK: SortKey<T> + Copy + Send + Sync + 'static,
    Cod: KeyedCodec
        + EncodeKeyedRun<T>
        + EncodeKeyedRun<<<Cod as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>
        + Copy
        + 'static,
    Cmp: for<'a> Compare<SK::Key<'a>> + Compare<Cod::Key> + Copy + Send + Sync + 'static,
    KeyCompare<SK, Cmp>:
        Compare<<<Cod as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>,
    CS: ChunkSorter<T>,
{
    /// Add an item to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing to disk fails.
    pub fn push(&mut self, item: T) -> Result<(), MergeError<Cod::Error>> {
        if self
            .flush
            .should_flush_before_push(&self.session.buffer, &item)
        {
            self.session.commit()?;
        }

        self.session.buffer.extend([item]);

        if self.flush.should_flush_after_push(&self.session.buffer) {
            self.session.commit()?;
        }

        Ok(())
    }

    /// Flush remaining items, merge all sorted runs, apply dedup,
    /// and return finalized sorted output.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or merging fails.
    ///
    #[allow(clippy::type_complexity)]
    pub fn finish(
        self,
    ) -> Result<
        Sorted<<D as Dedup<KeyedRunMerge<Cod, Cmp, KeyCompare<SK, Cmp>>>>::Deduped>,
        MergeError<Cod::Error>,
    >
    where
        D: Dedup<KeyedRunMerge<Cod, Cmp, KeyCompare<SK, Cmp>>>,
    {
        let dedup = self.dedup;
        let merged = self.session.finish()?;

        Ok(Sorted::new(dedup.dedup(merged)))
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufWriter, Read, Write};

    use super::*;
    use crate::{
        codec::{
            CodecCursor, CodecWriter, DeriveKey, EncodeKeyedRun, EncodeRun, KeyedCodec,
            KeyedCodecCursor, KeyedCodecWriter, TakeOwned,
        },
        compare::Reverse,
        dedup::AdjacentDedup,
        key::Owned,
    };

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

    impl Codec for U64KeyedCodec {
        type Error = std::io::Error;
        type Cursor<R: Read> = U64Reader<R>;

        fn cursor<R: Read>(&self, source: R) -> Result<U64Reader<R>, Self::Error> {
            Ok(U64Reader {
                inner: source,
                current: None,
            })
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

    impl DeriveKey<u64> for U64KeyedCodec {
        fn derive_key(&self, item: &u64) -> u8 {
            u8::try_from(*item / 10).expect("test values should fit in u8")
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

    type U64Sorter = Sorter<u64, Owned<fn(&u64) -> u64>, U64Codec, Natural, Identity, Sequential>;

    #[allow(clippy::type_complexity)]
    type U64KeyedSorter =
        Sorter<u64, Owned<fn(&u64) -> u64>, U64KeyedCodec, Natural, Identity, Sequential, Keyed>;

    fn u64_sorter(max_items: usize) -> U64Sorter {
        Builder::new()
            .key(Owned((|v: &u64| *v) as fn(&u64) -> u64))
            .codec(U64Codec)
            .max_buffer_items::<u64>(max_items)
            .build()
    }

    fn u64_keyed_sorter(max_items: usize) -> U64KeyedSorter {
        Builder::new()
            .key(Owned((|v: &u64| *v) as fn(&u64) -> u64))
            .keyed_codec(U64KeyedCodec)
            .max_buffer_items::<u64>(max_items)
            .build()
    }

    #[allow(clippy::type_complexity)]
    fn u64_accumulator(
        config: MergeConfig,
    ) -> RunAccumulator<u64, Owned<fn(&u64) -> u64>, U64Codec, Natural, Sequential, Unkeyed> {
        RunAccumulator::new(
            Owned((|v: &u64| *v) as fn(&u64) -> u64),
            U64Codec,
            Natural,
            Sequential,
            config,
        )
    }

    #[allow(clippy::type_complexity)]
    fn u64_keyed_accumulator(
        config: MergeConfig,
    ) -> RunAccumulator<u64, Owned<fn(&u64) -> u64>, U64KeyedCodec, Natural, Sequential, Keyed>
    {
        RunAccumulator::new(
            Owned((|v: &u64| *v) as fn(&u64) -> u64),
            U64KeyedCodec,
            Natural,
            Sequential,
            config,
        )
    }

    #[allow(clippy::type_complexity)]
    fn u64_session(
        config: MergeConfig,
    ) -> Session<
        OwnedBuffer<u64>,
        RunAccumulator<u64, Owned<fn(&u64) -> u64>, U64Codec, Natural, Sequential, Unkeyed>,
    > {
        Session::new(
            OwnedBuffer::new(Some(Box::new(|_| 8))),
            u64_accumulator(config),
        )
    }

    #[allow(clippy::type_complexity)]
    fn collect_unkeyed(
        merged: RunMerge<U64Codec, KeyCompare<Owned<fn(&u64) -> u64>, Natural>>,
    ) -> Vec<u64> {
        merged.map(|r| r.expect("read")).collect()
    }

    #[allow(clippy::type_complexity)]
    fn collect_keyed(
        merged: KeyedRunMerge<U64KeyedCodec, Natural, KeyCompare<Owned<fn(&u64) -> u64>, Natural>>,
    ) -> Vec<u64> {
        merged.map(|r| r.expect("read")).collect()
    }

    struct TestVisitable {
        items: Vec<u64>,
    }

    impl VisitSortedItems for TestVisitable {
        type Record = u64;
        type Error = std::convert::Infallible;

        fn visit_items<F, FE>(self, mut f: F) -> Result<(), SortedItemsError<Self::Error, FE>>
        where
            F: FnMut(&u64) -> Result<(), FE>,
        {
            for item in self.items {
                f(&item).map_err(SortedItemsError::Sink)?;
            }

            Ok(())
        }
    }

    #[test]
    fn sort_single_item() {
        let mut sorter = u64_sorter(100);
        sorter.push(42).expect("push");
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r| r.expect("read"))
            .collect();
        assert_eq!(results, vec![42]);
    }

    #[test]
    fn unkeyed_accumulator_sorts_and_finalizes_multiple_runs() {
        let mut accumulator = u64_accumulator(MergeConfig::default());
        let mut first = vec![9, 3, 7];
        let mut second = vec![8, 1, 5];

        accumulator.accumulate(&mut first).expect("first run");
        accumulator.accumulate(&mut second).expect("second run");
        assert_eq!(accumulator.runs.len(), 2);

        let merged = accumulator.finish().expect("finish");
        assert_eq!(collect_unkeyed(merged), vec![1, 3, 5, 7, 8, 9]);
    }

    #[test]
    fn keyed_accumulator_sorts_and_finalizes_multiple_runs() {
        let mut accumulator = u64_keyed_accumulator(MergeConfig::default());
        let mut first = vec![24, 11, 19];
        let mut second = vec![12, 25, 18];

        accumulator.accumulate(&mut first).expect("first run");
        accumulator.accumulate(&mut second).expect("second run");
        assert_eq!(accumulator.runs.len(), 2);

        let merged = accumulator.finish().expect("finish");
        assert_eq!(collect_keyed(merged), vec![11, 12, 18, 19, 24, 25]);
    }

    #[test]
    fn empty_accumulator_finalizes_to_empty_output() {
        let accumulator = u64_accumulator(MergeConfig::default());
        assert_eq!(accumulator.runs.len(), 0);

        let merged = accumulator.finish().expect("finish");
        assert!(collect_unkeyed(merged).is_empty());
    }

    #[test]
    fn empty_keyed_accumulator_finalizes_to_empty_output() {
        let accumulator = u64_keyed_accumulator(MergeConfig::default());
        assert_eq!(accumulator.runs.len(), 0);

        let merged = accumulator.finish().expect("finish");
        assert!(collect_keyed(merged).is_empty());
    }

    #[test]
    fn failed_accumulation_registers_zero_runs() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let config = MergeConfig {
            temp_dir: Some(temp_dir.path().join("missing")),
            ..MergeConfig::default()
        };
        let mut accumulator = u64_accumulator(config);
        let mut items = vec![3, 1, 2];

        let result = accumulator.accumulate(&mut items);
        assert!(matches!(result, Err(MergeError::Io(_))));
        assert_eq!(accumulator.runs.len(), 0);
    }

    #[test]
    fn failed_keyed_accumulation_registers_zero_runs() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let config = MergeConfig {
            temp_dir: Some(temp_dir.path().join("missing")),
            ..MergeConfig::default()
        };
        let mut accumulator = u64_keyed_accumulator(config);
        let mut items = vec![24, 11, 19];

        let result = accumulator.accumulate(&mut items);
        assert!(matches!(result, Err(MergeError::Io(_))));
        assert_eq!(accumulator.runs.len(), 0);
    }

    #[test]
    fn successful_session_commit_registers_one_run_and_clears_owned_buffer() {
        let mut session = u64_session(MergeConfig::default());
        session.buffer.extend([3, 1, 2]);
        let capacity = session.buffer.items.capacity();

        session.commit().expect("commit");

        assert_eq!(session.accumulator.runs.len(), 1);
        assert!(session.buffer.items.is_empty());
        assert_eq!(session.buffer.active_bytes, 0);
        assert_eq!(session.buffer.items.capacity(), capacity);
    }

    #[test]
    fn failed_session_commit_keeps_items_accounting_and_registers_no_run() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let config = MergeConfig {
            temp_dir: Some(temp_dir.path().join("missing")),
            ..MergeConfig::default()
        };
        let mut session = u64_session(config);
        session.buffer.extend([3, 1, 2]);

        let result = session.commit();

        assert!(matches!(result, Err(MergeError::Io(_))));
        assert_eq!(session.accumulator.runs.len(), 0);
        assert_eq!(session.buffer.items.len(), 3);
        assert_eq!(session.buffer.active_bytes, 24);

        let mut retained = session.buffer.items.clone();
        retained.sort_unstable();
        assert_eq!(retained, vec![1, 2, 3]);
    }

    #[test]
    fn session_finish_includes_nonempty_residual_chunk() {
        let mut session = u64_session(MergeConfig::default());
        session.buffer.extend([5, 1, 3]);

        let merged = session.finish().expect("finish");

        assert_eq!(collect_unkeyed(merged), vec![1, 3, 5]);
    }

    #[test]
    fn empty_session_commit_leaves_zero_private_runs() {
        let mut session = u64_session(MergeConfig::default());

        session.commit().expect("commit");

        assert_eq!(session.accumulator.runs.len(), 0);
        assert!(session.buffer.items.is_empty());
        assert_eq!(session.buffer.active_bytes, 0);
    }

    #[test]
    fn empty_finish_creates_no_temp_run_artifact() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let sorter = Builder::new()
            .key(Owned((|v: &u64| *v) as fn(&u64) -> u64))
            .codec(U64Codec)
            .max_buffer_items::<u64>(2)
            .merge_config(MergeConfig {
                temp_dir: Some(temp_dir.path().to_path_buf()),
                ..MergeConfig::default()
            })
            .build();

        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r| r.expect("read"))
            .collect();
        let artifacts = std::fs::read_dir(temp_dir.path())
            .expect("read temp dir")
            .count();

        assert!(results.is_empty());
        assert_eq!(artifacts, 0);
    }

    #[test]
    fn unkeyed_public_push_commits_through_session() {
        let mut sorter = u64_sorter(2);

        sorter.push(2).expect("push");
        sorter.push(1).expect("push");

        assert_eq!(sorter.session.accumulator.runs.len(), 1);
        assert!(sorter.session.buffer.items.is_empty());
        assert_eq!(sorter.session.buffer.active_bytes, 0);
    }

    #[test]
    fn keyed_public_push_commits_through_session() {
        let mut sorter = u64_keyed_sorter(2);

        sorter.push(22).expect("push");
        sorter.push(11).expect("push");

        assert_eq!(sorter.session.accumulator.runs.len(), 1);
        assert!(sorter.session.buffer.items.is_empty());
        assert_eq!(sorter.session.buffer.active_bytes, 0);
    }

    #[test]
    fn item_count_sorter_carries_no_byte_sizing_operation() {
        let mut sorter = u64_sorter(3);

        sorter.push(10).expect("push");

        assert!(sorter.session.buffer.item_size.is_none());
        assert_eq!(sorter.session.buffer.active_bytes, 0);
        assert_eq!(sorter.session.buffer.items, vec![10]);
    }

    #[test]
    fn byte_budget_sorter_accounts_and_commits_at_existing_threshold() {
        let mut sorter = Builder::new()
            .key(Owned((|v: &u64| *v) as fn(&u64) -> u64))
            .codec(U64Codec)
            .memory_budget::<u64>(10, |v| usize::try_from(*v).expect("test value fits usize"))
            .build();

        sorter.push(4).expect("first push");
        assert_eq!(sorter.session.accumulator.runs.len(), 0);
        assert_eq!(sorter.session.buffer.active_bytes, 4);
        assert_eq!(sorter.session.buffer.items, vec![4]);

        sorter.push(5).expect("second push");
        assert_eq!(sorter.session.accumulator.runs.len(), 0);
        assert_eq!(sorter.session.buffer.active_bytes, 9);
        assert_eq!(sorter.session.buffer.items, vec![4, 5]);

        sorter.push(1).expect("threshold push");
        assert_eq!(sorter.session.accumulator.runs.len(), 1);
        assert!(sorter.session.buffer.items.is_empty());
        assert_eq!(sorter.session.buffer.active_bytes, 0);
        assert!(sorter.session.buffer.item_size.is_some());
    }

    #[test]
    fn sorted_items_visits_items_in_order() {
        let mut visited = Vec::new();
        Sorted::new(TestVisitable {
            items: vec![1, 2, 3],
        })
        .items()
        .try_for_each(|item| {
            visited.push(*item);
            Ok::<(), std::convert::Infallible>(())
        })
        .expect("visit items");

        assert_eq!(visited, vec![1, 2, 3]);
    }

    #[test]
    fn sorted_items_stops_on_visitor_error() {
        #[derive(Debug, PartialEq, Eq)]
        struct SinkError;

        impl std::fmt::Display for SinkError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("sink error")
            }
        }

        impl std::error::Error for SinkError {}

        let mut visited = Vec::new();
        let err = Sorted::new(TestVisitable {
            items: vec![1, 2, 3],
        })
        .items()
        .try_for_each(|item| {
            visited.push(*item);
            if *item == 2 { Err(SinkError) } else { Ok(()) }
        })
        .expect_err("sink error should stop traversal");

        assert!(matches!(err, SortedItemsError::Sink(SinkError)));
        assert_eq!(visited, vec![1, 2]);
    }

    #[test]
    fn sort_already_sorted() {
        let mut sorter = u64_sorter(100);
        for v in [1, 2, 3, 4, 5] {
            sorter.push(v).expect("push");
        }
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r| r.expect("read"))
            .collect();
        assert_eq!(results, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sort_reverse_input() {
        let mut sorter = u64_sorter(100);
        for v in [5, 4, 3, 2, 1] {
            sorter.push(v).expect("push");
        }
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r| r.expect("read"))
            .collect();
        assert_eq!(results, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sort_with_spilling() {
        let mut sorter = u64_sorter(3);
        for v in [9, 7, 5, 3, 1, 2, 4, 6, 8, 10] {
            sorter.push(v).expect("push");
        }
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r| r.expect("read"))
            .collect();
        assert_eq!(results, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn sort_empty_input() {
        let sorter = u64_sorter(100);
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r| r.expect("read"))
            .collect();
        assert!(results.is_empty());
    }

    #[test]
    fn sort_preserves_duplicates() {
        let mut sorter = u64_sorter(3);
        for v in [3, 1, 2, 1, 3, 2] {
            sorter.push(v).expect("push");
        }
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r| r.expect("read"))
            .collect();
        assert_eq!(results, vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn sort_with_reverse_comparator() {
        let mut sorter = Builder::new()
            .key(Owned((|v: &u64| *v) as fn(&u64) -> u64))
            .codec(U64Codec)
            .compare(Reverse(Natural))
            .max_buffer_items::<u64>(3)
            .build();

        for v in [1, 5, 3, 2, 4] {
            sorter.push(v).expect("push");
        }
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r| r.expect("read"))
            .collect();
        assert_eq!(results, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn sort_with_dedup() {
        let mut sorter = Builder::new()
            .key(Owned((|v: &u64| *v) as fn(&u64) -> u64))
            .codec(U64Codec)
            .dedup(AdjacentDedup::new(|a: &u64, b: &u64| a == b))
            .max_buffer_items::<u64>(3)
            .build();

        for v in [3, 1, 2, 1, 3, 2] {
            sorter.push(v).expect("push");
        }
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r: Result<u64, _>| r.expect("read"))
            .collect();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn sort_with_byte_budget() {
        let mut sorter = Builder::new()
            .key(Owned((|v: &u64| *v) as fn(&u64) -> u64))
            .codec(U64Codec)
            .fixed_size_budget::<u64>(24)
            .build();

        for v in [9, 7, 5, 3, 1, 2, 4, 6, 8, 10] {
            sorter.push(v).expect("push");
        }
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r: Result<u64, _>| r.expect("read"))
            .collect();
        assert_eq!(results, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn sort_all_in_memory_no_spill() {
        let mut sorter = u64_sorter(1000);
        for v in [5, 3, 1, 4, 2] {
            sorter.push(v).expect("push");
        }
        let results: Vec<u64> = sorter
            .finish()
            .expect("finish")
            .map(|r| r.expect("read"))
            .collect();
        assert_eq!(results, vec![1, 2, 3, 4, 5]);
    }

    mod proptests {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            #[test]
            fn output_is_always_sorted(
                data in proptest::collection::vec(0u64..10_000, 0..500),
                max_items in 3usize..50,
            ) {
                let mut sorter = u64_sorter(max_items);
                for v in &data {
                    sorter.push(*v).expect("push");
                }
                let results: Vec<u64> = sorter
                    .finish()
                    .expect("finish")
                    .map(|r| r.expect("read"))
                    .collect();

                prop_assert!(
                    results.windows(2).all(|w| w[0] <= w[1]),
                    "output must be sorted"
                );
            }

            #[test]
            fn output_preserves_all_items(
                data in proptest::collection::vec(0u64..1_000, 0..200),
                max_items in 3usize..50,
            ) {
                let mut sorter = u64_sorter(max_items);
                for v in &data {
                    sorter.push(*v).expect("push");
                }
                let mut results: Vec<u64> = sorter
                    .finish()
                    .expect("finish")
                    .map(|r| r.expect("read"))
                    .collect();

                let mut expected = data;
                expected.sort_unstable();
                results.sort_unstable();

                prop_assert_eq!(results, expected);
            }
        }
    }
}
