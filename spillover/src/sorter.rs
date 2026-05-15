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

use get_size2::GetSize;

use crate::{
    SortedItemsError,
    chunk::{ChunkSorter, Sequential},
    codec::{Codec, CodecWriter, KeyedCodec, KeyedCodecWriter},
    compare::{Compare, Natural},
    dedup::{Dedup, Identity},
    key::{KeyCompare, SortKey},
    merge::{KeyedRunMerge, MergeConfig, MergeError, RunMerge, RunMerger, SortedRun},
};

/// How the sorter decides when to flush the in-memory buffer to
/// disk.
enum FlushStrategy<T> {
    /// Flush when the estimated byte usage exceeds a budget.
    /// The closure returns the memory footprint of each item.
    Bytes {
        budget: usize,
        item_size: Box<dyn Fn(&T) -> usize + Send + Sync>,
    },

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
/// Marker: codec has been provided (base path).
pub struct HasCodec<Cod>(Cod);
/// Marker: keyed codec has been provided (keyed merge path).
pub struct HasKeyedCodec<Cod>(Cod);

/// Marker: no flush strategy provided yet.
pub struct NeedsFlushStrategy;
/// Marker: flush strategy has been provided.
pub struct HasFlushStrategy<T>(FlushStrategy<T>);

/// Marker: base merge path — codec implements [`Codec`] only.
pub struct Basic;
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

    /// Set the codec (base merge path).
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
            flush: HasFlushStrategy(FlushStrategy::Bytes {
                budget,
                item_size: Box::new(item_size),
            }),
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
            flush: HasFlushStrategy(FlushStrategy::Bytes {
                budget,
                item_size: Box::new(GetSize::get_size),
            }),
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
            flush: HasFlushStrategy(FlushStrategy::Bytes {
                budget,
                item_size: Box::new(|_| std::mem::size_of::<T>()),
            }),
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
            flush: HasFlushStrategy(FlushStrategy::Items { max_items }),
            compare: self.compare,
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }
}

// build() for basic path.
impl<T, SK, Cod, Cmp, D, CS> Builder<HasSortKey<SK>, HasCodec<Cod>, HasFlushStrategy<T>, Cmp, D, CS>
where
    SK: SortKey<T> + Copy,
    Cod: Codec<Item = T> + Copy,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy,
    CS: ChunkSorter<T>,
{
    /// Build the [`Sorter`] (base merge path).
    #[must_use]
    pub fn build(self) -> Sorter<T, SK, Cod, Cmp, D, CS, Basic> {
        Sorter {
            sort_key: self.sort_key.0,
            codec: self.codec.0,
            compare: self.compare,
            dedup: Some(self.dedup),
            chunk_sort: self.chunk_sort,
            flush: self.flush.0,
            buffer: Vec::new(),
            buffer_bytes: 0,
            spilled_runs: Vec::new(),
            config: self.config,
            _marker: std::marker::PhantomData,
        }
    }
}

// build() for keyed path.
impl<T, SK, Cod, Cmp, D, CS>
    Builder<HasSortKey<SK>, HasKeyedCodec<Cod>, HasFlushStrategy<T>, Cmp, D, CS>
where
    SK: SortKey<T> + Copy,
    Cod: KeyedCodec<Item = T> + Copy,
    Cmp: for<'a> Compare<SK::Key<'a>> + Compare<Cod::Key> + Copy,
    CS: ChunkSorter<T>,
{
    /// Build the [`Sorter`] (keyed merge path).
    #[must_use]
    pub fn build(self) -> Sorter<T, SK, Cod, Cmp, D, CS, Keyed> {
        Sorter {
            sort_key: self.sort_key.0,
            codec: self.codec.0,
            compare: self.compare,
            dedup: Some(self.dedup),
            chunk_sort: self.chunk_sort,
            flush: self.flush.0,
            buffer: Vec::new(),
            buffer_bytes: 0,
            spilled_runs: Vec::new(),
            config: self.config,
            _marker: std::marker::PhantomData,
        }
    }
}

/// A disk-spilling external sorter.
///
/// Push items via [`push`](Self::push), then call
/// [`finish`](Self::finish) to flush remaining items, merge all
/// sorted runs, apply deduplication, and produce a sorted output
/// iterator.
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
/// - `M`: merge strategy marker ([`Basic`] or [`Keyed`])
pub struct Sorter<T, SK, Cod, Cmp, D, CS, M = Basic> {
    sort_key: SK,
    codec: Cod,
    compare: Cmp,
    dedup: Option<D>,
    chunk_sort: CS,
    flush: FlushStrategy<T>,
    buffer: Vec<T>,
    buffer_bytes: usize,
    spilled_runs: Vec<SortedRun>,
    config: SorterConfig,
    _marker: std::marker::PhantomData<M>,
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

#[doc(hidden)]
pub mod sealed {
    pub trait Sealed {}
}

/// Implementation detail for sources that can be drained through
/// [`SortedItems`].
///
/// This trait is sealed. Users should not implement it directly; use
/// [`Sorted::items`] and [`SortedItems::try_for_each`] instead.
#[doc(hidden)]
pub trait VisitSortedItems: sealed::Sealed {
    /// The item representation passed to the item sink.
    type Item<'a>
    where
        Self: 'a;

    /// The source error type.
    type Error;

    /// Visit each item in source order.
    fn visit_items<F, FE>(self, f: F) -> Result<(), SortedItemsError<Self::Error, FE>>
    where
        F: for<'a> FnMut(Self::Item<'a>) -> Result<(), FE>;
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
        F: for<'a> FnMut(S::Item<'a>) -> Result<(), FE>,
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

// ── Basic path: push + flush + finish ────────────────────

impl<T, SK, Cod, Cmp, D, CS> Sorter<T, SK, Cod, Cmp, D, CS, Basic>
where
    T: 'static,
    SK: SortKey<T> + Copy + Send + Sync + 'static,
    Cod: Codec<Item = T> + Copy + 'static,
    for<'a> Cod::Writer<&'a mut std::fs::File>: CodecWriter<T, Error = Cod::Error>,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy + Send + Sync + 'static,
    CS: ChunkSorter<T>,
{
    /// Add an item to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing to disk fails.
    pub fn push(&mut self, item: T) -> Result<(), MergeError<Cod::Error>> {
        match &self.flush {
            FlushStrategy::Bytes { budget, item_size } => {
                self.buffer_bytes += item_size(&item);
                self.buffer.push(item);
                if self.buffer_bytes >= *budget {
                    self.flush_basic()?;
                }
            }
            FlushStrategy::Items { max_items } => {
                self.buffer.push(item);
                if self.buffer.len() >= *max_items {
                    self.flush_basic()?;
                }
            }
        }
        Ok(())
    }

    fn flush_basic(&mut self) -> Result<(), MergeError<Cod::Error>> {
        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        self.chunk_sort.sort(&mut self.buffer, move |a: &T, b: &T| {
            Compare::compare(&item_cmp, a, b)
        });

        let run_merger = RunMerger::new(self.codec, item_cmp, self.config.merge.clone());
        let run = run_merger.spill_sorted(self.buffer.drain(..))?;
        self.spilled_runs.push(run);
        self.buffer_bytes = 0;

        Ok(())
    }

    /// Flush remaining items, merge all sorted runs, apply dedup,
    /// and return finalized sorted output.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or merging fails.
    ///
    /// # Panics
    ///
    /// Panics if called more than once.
    #[allow(clippy::type_complexity)]
    pub fn finish(
        mut self,
    ) -> Result<
        Sorted<<D as Dedup<RunMerge<T, Cod, KeyCompare<SK, Cmp>>>>::Deduped>,
        MergeError<Cod::Error>,
    >
    where
        D: Dedup<RunMerge<T, Cod, KeyCompare<SK, Cmp>>>,
    {
        if !self.buffer.is_empty() {
            self.flush_basic()?;
        }

        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        let run_merger = RunMerger::new(self.codec, item_cmp, self.config.merge.clone());
        let merged = run_merger.merge(std::mem::take(&mut self.spilled_runs))?;

        let dedup = self
            .dedup
            .take()
            .expect("dedup is always Some until finish() consumes it");

        Ok(Sorted::new(dedup.dedup(merged)))
    }
}

// ── Keyed path: push + flush + finish ────────────────────

impl<T, SK, Cod, Cmp, D, CS> Sorter<T, SK, Cod, Cmp, D, CS, Keyed>
where
    T: 'static,
    SK: SortKey<T> + Copy + Send + Sync + 'static,
    Cod: KeyedCodec<Item = T> + Copy + 'static,
    for<'a> Cod::Writer<&'a mut std::fs::File>: CodecWriter<T, Error = Cod::Error>,
    for<'a> Cod::KeyedWriter<&'a mut std::fs::File>:
        KeyedCodecWriter<T, Cod::Key, Error = Cod::Error>,
    Cmp: for<'a> Compare<SK::Key<'a>> + Compare<Cod::Key> + Copy + Send + Sync + 'static,
    CS: ChunkSorter<T>,
{
    /// Add an item to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing to disk fails.
    pub fn push(&mut self, item: T) -> Result<(), MergeError<Cod::Error>> {
        match &self.flush {
            FlushStrategy::Bytes { budget, item_size } => {
                self.buffer_bytes += item_size(&item);
                self.buffer.push(item);
                if self.buffer_bytes >= *budget {
                    self.flush_keyed()?;
                }
            }
            FlushStrategy::Items { max_items } => {
                self.buffer.push(item);
                if self.buffer.len() >= *max_items {
                    self.flush_keyed()?;
                }
            }
        }
        Ok(())
    }

    fn flush_keyed(&mut self) -> Result<(), MergeError<Cod::Error>> {
        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        self.chunk_sort.sort(&mut self.buffer, move |a: &T, b: &T| {
            Compare::compare(&item_cmp, a, b)
        });

        let named = match self.config.merge.temp_dir {
            Some(ref dir) => tempfile::NamedTempFile::new_in(dir).map_err(MergeError::Io)?,
            None => tempfile::NamedTempFile::new().map_err(MergeError::Io)?,
        };
        let mut file = named.reopen().map_err(MergeError::Io)?;
        let mut writer = self.codec.keyed_writer(&mut file);
        for item in &self.buffer {
            let key = self.codec.derive_key(item);
            writer.write_keyed(item, &key).map_err(MergeError::Codec)?;
        }
        writer.finish().map_err(MergeError::Codec)?;
        drop(file);

        self.spilled_runs.push(SortedRun {
            path: named.into_temp_path(),
        });
        self.buffer.clear();
        self.buffer_bytes = 0;

        Ok(())
    }

    /// Flush remaining items, merge all sorted runs, apply dedup,
    /// and return finalized sorted output.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or merging fails.
    ///
    /// # Panics
    ///
    /// Panics if called more than once.
    #[allow(clippy::type_complexity)]
    pub fn finish(
        mut self,
    ) -> Result<
        Sorted<<D as Dedup<KeyedRunMerge<T, Cod, Cmp, KeyCompare<SK, Cmp>>>>::Deduped>,
        MergeError<Cod::Error>,
    >
    where
        D: Dedup<KeyedRunMerge<T, Cod, Cmp, KeyCompare<SK, Cmp>>>,
    {
        if !self.buffer.is_empty() {
            self.flush_keyed()?;
        }

        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        let run_merger = RunMerger::new(self.codec, item_cmp, self.config.merge.clone());
        let merged =
            run_merger.merge_keyed(std::mem::take(&mut self.spilled_runs), self.compare)?;

        let dedup = self
            .dedup
            .take()
            .expect("dedup is always Some until finish() consumes it");

        Ok(Sorted::new(dedup.dedup(merged)))
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufWriter, Read, Write};

    use super::*;
    use crate::{
        codec::{CodecCursor, CodecWriter},
        compare::Reverse,
        dedup::AdjacentDedup,
        key::Owned,
    };

    #[derive(Clone, Copy)]
    struct U64Codec;

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

    type U64Sorter = Sorter<u64, Owned<fn(&u64) -> u64>, U64Codec, Natural, Identity, Sequential>;

    fn u64_sorter(max_items: usize) -> U64Sorter {
        Builder::new()
            .key(Owned((|v: &u64| *v) as fn(&u64) -> u64))
            .codec(U64Codec)
            .max_buffer_items::<u64>(max_items)
            .build()
    }

    struct TestVisitable {
        items: Vec<u64>,
    }

    impl sealed::Sealed for TestVisitable {}

    impl VisitSortedItems for TestVisitable {
        type Item<'a>
            = u64
        where
            Self: 'a;

        type Error = std::convert::Infallible;

        fn visit_items<F, FE>(self, mut f: F) -> Result<(), SortedItemsError<Self::Error, FE>>
        where
            F: for<'a> FnMut(u64) -> Result<(), FE>,
        {
            for item in self.items {
                f(item).map_err(SortedItemsError::Sink)?;
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
    fn sorted_items_visits_items_in_order() {
        let mut visited = Vec::new();
        Sorted::new(TestVisitable {
            items: vec![1, 2, 3],
        })
        .items()
        .try_for_each(|item| {
            visited.push(item);
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
            visited.push(item);
            if item == 2 { Err(SinkError) } else { Ok(()) }
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
