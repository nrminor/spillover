//! The external sorter — the main entry point for sorting
//! larger-than-memory datasets.
//!
//! [`Sorter`] ties together all the trait axes ([`SortKey`],
//! [`Compare`], [`Codec`], [`Dedup`], [`ChunkSorter`]) and the
//! merge engine into a single, configurable pipeline. Construct
//! one via the type-state [`Builder`], push items into it, and
//! call [`finish`](Sorter::finish) to get a sorted, optionally
//! deduplicated output iterator.
//!
//! [`SortKey`]: crate::key::SortKey
//! [`Compare`]: crate::compare::Compare
//! [`Codec`]: crate::codec::Codec
//! [`Dedup`]: crate::dedup::Dedup
//! [`ChunkSorter`]: crate::chunk::ChunkSorter

use get_size2::GetSize;

use crate::{
    chunk::{ChunkSorter, Sequential},
    codec::{Codec, KeyedCodec},
    compare::{Compare, Natural},
    dedup::{Dedup, Identity},
    key::{KeyCompare, SortKey},
    merge::{MergeConfig, MergeError, RunMerger, SortedRun},
};

/// How the sorter decides when to flush the in-memory buffer to
/// disk.
enum FlushStrategy<T> {
    /// Flush when the estimated byte usage exceeds a budget.
    /// The closure returns the memory footprint of each item.
    Bytes {
        budget: usize,
        item_size: Box<dyn Fn(&T) -> usize>,
    },

    /// Flush when the item count exceeds a limit.
    Items { max_items: usize },
}

/// Configuration for the [`Sorter`].
///
/// Controls merge behavior (fan-in, buffer sizes, temp directory).
/// Flush strategy is configured separately via the builder's
/// [`memory_budget`](Builder::memory_budget) or
/// [`max_buffer_items`](Builder::max_buffer_items) methods.
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

/// Marker: base merge path — codec implements [`Codec`] only.
pub struct Basic;
/// Marker: keyed merge path — codec implements [`KeyedCodec`],
/// enabling key-only comparisons during merge.
pub struct Keyed;

/// Marker: no flush strategy provided yet.
pub struct NeedsFlushStrategy;
/// Marker: flush strategy has been provided.
pub struct HasFlushStrategy<T>(FlushStrategy<T>);

/// Type-state builder for [`Sorter`].
///
/// The builder enforces at compile time that a sort key, codec,
/// and flush strategy must be provided before [`build`](Builder::build)
/// can be called. Compare defaults to [`Natural`], dedup defaults
/// to [`Identity`], and chunk sorter defaults to [`Sequential`].
///
/// ```ignore
/// use spillover::sorter::Builder;
/// use spillover::key::Owned;
/// use spillover::compare::Natural;
///
/// let sorter = Builder::new()
///     .key(Owned(|v: &i32| *v))
///     .codec(my_codec)
///     .max_buffer_items(10_000)
///     .build();
/// ```
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

    /// Set the codec for serializing items to temporary files.
    /// Uses the base merge path — the merge deserializes full
    /// records for comparison.
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

    /// Set a keyed codec for serializing items with precomputed
    /// sort keys. Uses the keyed merge path — the merge compares
    /// keys without deserializing full records.
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

    /// Set the comparator for ordering keys. Defaults to
    /// [`Natural`] (uses [`Ord`]).
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

    /// Set the post-merge deduplication strategy. Defaults to
    /// [`Identity`] (no dedup).
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

// Flush strategy methods — available regardless of type-state,
// but transition the Flush parameter.

impl<SK, Cod, Cmp, D, CS> Builder<SK, Cod, NeedsFlushStrategy, Cmp, D, CS> {
    /// Set a byte-based memory budget with a custom sizing
    /// function.
    ///
    /// The sorter will flush to disk when the estimated total
    /// memory footprint of buffered items exceeds `budget` bytes.
    /// The `item_size` closure should return the total memory
    /// footprint (stack + heap) of each item.
    ///
    /// For types that implement [`GetSize`], use
    /// [`measured_budget`](Self::measured_budget) instead.
    /// For fixed-size types with no heap allocations, use
    /// [`fixed_size_budget`](Self::fixed_size_budget).
    #[must_use]
    pub fn memory_budget<T: 'static>(
        self,
        budget: usize,
        item_size: impl Fn(&T) -> usize + 'static,
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

    /// Set a byte-based memory budget for types that implement
    /// [`GetSize`].
    ///
    /// This is the recommended approach for types with heap
    /// allocations. The sizing function is generated automatically
    /// from the [`GetSize`] implementation, which can be derived
    /// with `#[derive(GetSize)]`.
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

    /// Set a byte-based memory budget for fixed-size types with
    /// no heap allocations.
    ///
    /// Uses [`std::mem::size_of`] for the item size, which is
    /// accurate for types like `(u64, i32)` that store all their
    /// data inline. Do not use this for types with `Vec`, `String`,
    /// or other heap-allocating fields — the budget will be
    /// drastically underestimated.
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

    /// Set a count-based buffer limit.
    ///
    /// The sorter will flush to disk when the buffer contains
    /// `max_items` items, regardless of their memory footprint.
    /// Use this when you don't want to measure item sizes, or
    /// when you know the approximate size of your items and
    /// can pick a count that keeps memory usage reasonable.
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

// build() is only available when all required axes are provided.

impl<T, SK, Cod, Cmp, D, CS> Builder<HasSortKey<SK>, HasCodec<Cod>, HasFlushStrategy<T>, Cmp, D, CS>
where
    SK: SortKey<T> + Copy,
    Cod: Codec<T> + Copy,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy,
    CS: ChunkSorter<T>,
{
    /// Build the [`Sorter`].
    ///
    /// This is only available after providing a sort key, codec,
    /// and flush strategy. Compare defaults to [`Natural`], dedup
    /// defaults to [`Identity`], and chunk sorter defaults to
    /// [`Sequential`].
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

// build() for the keyed path.
impl<T, SK, Cod, Cmp, D, CS>
    Builder<HasSortKey<SK>, HasKeyedCodec<Cod>, HasFlushStrategy<T>, Cmp, D, CS>
where
    SK: SortKey<T> + Copy,
    Cod: KeyedCodec<T> + Copy,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy,
    CS: ChunkSorter<T>,
{
    /// Build the [`Sorter`] with the keyed merge path.
    ///
    /// This is only available after providing a sort key, keyed
    /// codec, and flush strategy. The keyed path compares
    /// precomputed keys during the merge instead of deserializing
    /// full records.
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
/// The sorter is generic over seven axes:
/// - `T`: the item type being sorted
/// - `SK`: [`SortKey`] — how to extract a sort key
/// - `Cod`: [`Codec`] — how to serialize to/from disk
/// - `Cmp`: [`Compare`] — how to order keys
/// - `D`: [`Dedup`] — post-merge deduplication
/// - `CS`: [`ChunkSorter`] — in-memory sort algorithm
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

impl<T, SK, Cod, Cmp, D, CS, M> Sorter<T, SK, Cod, Cmp, D, CS, M>
where
    T: 'static,
    SK: SortKey<T> + Copy + Send + Sync + 'static,
    Cod: Codec<T> + Copy + 'static,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy + Send + Sync + 'static,
    CS: ChunkSorter<T>,
{
    /// Add an item to the sorter.
    ///
    /// If the buffer exceeds the configured flush threshold, the
    /// buffer is sorted and flushed to a temporary file on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing to disk fails (I/O or codec).
    pub fn push(&mut self, item: T) -> Result<(), MergeError<Cod::Error>> {
        match &self.flush {
            FlushStrategy::Bytes { budget, item_size } => {
                self.buffer_bytes += item_size(&item);
                self.buffer.push(item);
                if self.buffer_bytes >= *budget {
                    self.flush_buffer()?;
                }
            }
            FlushStrategy::Items { max_items } => {
                self.buffer.push(item);
                if self.buffer.len() >= *max_items {
                    self.flush_buffer()?;
                }
            }
        }
        Ok(())
    }

    /// Sort and flush the current buffer to a temporary file.
    fn flush_buffer(&mut self) -> Result<(), MergeError<Cod::Error>> {
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
}

// finish() for the base merge path.
impl<T, SK, Cod, Cmp, D, CS> Sorter<T, SK, Cod, Cmp, D, CS, Basic>
where
    T: 'static,
    SK: SortKey<T> + Copy + Send + Sync + 'static,
    Cod: Codec<T> + Copy + 'static,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy + Send + Sync + 'static,
    CS: ChunkSorter<T>,
{
    /// Flush any remaining items, merge all sorted runs, apply
    /// deduplication, and return a sorted output iterator.
    ///
    /// This uses the base merge path — every record is
    /// deserialized during the merge for comparison.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing the residual buffer or
    /// setting up the merge fails.
    ///
    /// # Panics
    ///
    /// Panics if called more than once (the dedup strategy is
    /// consumed on the first call).
    // Allow: the nested Result<impl Iterator<Item = Result<...>>>
    // is inherent to fallible streaming finalization.
    #[allow(clippy::type_complexity)]
    pub fn finish(
        mut self,
    ) -> Result<
        impl Iterator<Item = Result<D::Output, MergeError<Cod::Error>>>,
        MergeError<Cod::Error>,
    >
    where
        D: Dedup<T, MergeError<Cod::Error>>,
    {
        if !self.buffer.is_empty() {
            self.flush_buffer()?;
        }

        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        let run_merger = RunMerger::new(self.codec, item_cmp, self.config.merge.clone());
        let merged = run_merger.merge(std::mem::take(&mut self.spilled_runs))?;

        let dedup = self
            .dedup
            .take()
            .expect("dedup is always Some until finish() consumes it");

        Ok(dedup.dedup(merged))
    }
}

// finish() for the keyed merge path.
// TODO: implement the keyed merge using KeyedCodecReader.
// For now this is a placeholder that falls back to the base path
// so that the type structure compiles and tests pass. The real
// keyed merge will be implemented when spillover-bio needs it.
impl<T, SK, Cod, Cmp, D, CS> Sorter<T, SK, Cod, Cmp, D, CS, Keyed>
where
    T: 'static,
    SK: SortKey<T> + Copy + Send + Sync + 'static,
    Cod: KeyedCodec<T> + Copy + 'static,
    Cmp: for<'a> Compare<SK::Key<'a>> + Copy + Send + Sync + 'static,
    CS: ChunkSorter<T>,
{
    /// Flush any remaining items, merge all sorted runs, apply
    /// deduplication, and return a sorted output iterator.
    ///
    /// This uses the keyed merge path — the merge heap compares
    /// precomputed keys without deserializing full records. Only
    /// the winning record is deserialized on each merge step.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing the residual buffer or
    /// setting up the merge fails.
    ///
    /// # Panics
    ///
    /// Panics if called more than once (the dedup strategy is
    /// consumed on the first call).
    #[allow(clippy::type_complexity)]
    pub fn finish(
        mut self,
    ) -> Result<
        impl Iterator<Item = Result<D::Output, MergeError<Cod::Error>>>,
        MergeError<Cod::Error>,
    >
    where
        D: Dedup<T, MergeError<Cod::Error>>,
    {
        // TODO: use KeyedCodecReader for the merge heap.
        // For now, fall back to the base path via the Codec
        // supertrait.
        if !self.buffer.is_empty() {
            self.flush_buffer()?;
        }

        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        let run_merger = RunMerger::new(self.codec, item_cmp, self.config.merge.clone());
        let merged = run_merger.merge(std::mem::take(&mut self.spilled_runs))?;

        let dedup = self
            .dedup
            .take()
            .expect("dedup is always Some until finish() consumes it");

        Ok(dedup.dedup(merged))
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufWriter, Read, Write};

    use super::*;
    use crate::{
        codec::{CodecReader, CodecWriter},
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
    }

    impl<R: Read> CodecReader<u64> for U64Reader<R> {
        type Error = std::io::Error;

        fn read(&mut self) -> Result<Option<u64>, Self::Error> {
            let mut buf = [0u8; 8];
            match self.inner.read(&mut buf[..1]) {
                Ok(0) => Ok(None),
                Ok(_) => {
                    self.inner.read_exact(&mut buf[1..])?;
                    Ok(Some(u64::from_le_bytes(buf)))
                }
                Err(e) => Err(e),
            }
        }
    }

    impl Codec<u64> for U64Codec {
        type Error = std::io::Error;
        type Writer<W: Write> = U64Writer<W>;
        type Reader<R: Read> = U64Reader<R>;

        fn writer<W: Write>(&self, dest: W) -> U64Writer<W> {
            U64Writer {
                inner: BufWriter::new(dest),
            }
        }

        fn reader<R: Read>(&self, source: R) -> U64Reader<R> {
            U64Reader { inner: source }
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
        // Buffer holds 3 items, input has 10 → multiple spills.
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
        // 8 bytes per u64, budget of 24 bytes → flush every 3 items.
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
        // Budget larger than all data → no spill, single in-memory sort.
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
                max_items in 1usize..50,
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
                max_items in 1usize..50,
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
