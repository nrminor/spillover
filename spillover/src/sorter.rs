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

use std::{alloc::Layout, fs::File, io::BufReader};

use get_size2::GetSize;

use crate::{
    SortedItemsError,
    chunk::{ChunkSorter, EstimateSortMemory, Sequential},
    codec::{Codec, CodecCursor, EncodeKeyedRun, EncodeRun, KeyedCodec},
    compare::{Compare, Natural},
    dedup::{Dedup, Identity},
    key::{DeriveKey, EstimateKeyMemory, KeyCompare, KeyMemory, KeyedEntryCompare, SortKey},
    merge::{KeyedRunMerge, MergeConfig, MergeError, RunMerge, RunMerger, RunWriter, SortedRun},
};

/// How the sorter decides when to flush the in-memory buffer to
/// disk.
enum FlushStrategy {
    /// Flush when the estimated byte usage exceeds a budget.
    Bytes { budget: usize },

    /// Account for fixed-size owned storage and the configured spill transport.
    FixedSize { budget: usize },

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

/// Marker: a complete order has been selected.
pub struct HasOrder<O>(O);
/// Marker: the complete-order path needs a limit.
pub struct NeedsLimit;
/// Maximum number of buffered items on the complete-order path.
pub struct ItemLimit {
    max_items: usize,
}

impl ItemLimit {
    /// Construct an item-count limit.
    #[must_use]
    pub const fn new(max_items: usize) -> Self {
        Self { max_items }
    }

    /// Return the configured item limit.
    #[must_use]
    pub fn max_items(&self) -> usize {
        self.max_items
    }
}

/// Count-based algorithm workspace estimation.
pub struct CountBased;

/// A byte budget for the modeled active window and initial spill.
pub struct ByteLimit<M = CountBased> {
    bytes: usize,
    mode: std::marker::PhantomData<fn() -> M>,
}

impl ByteLimit {
    /// Construct a count-based byte budget.
    #[must_use]
    pub fn new(bytes: usize) -> Self {
        Self {
            bytes,
            mode: std::marker::PhantomData,
        }
    }
}

impl<M> ByteLimit<M> {
    /// Return the configured byte budget.
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.bytes
    }
}

impl From<usize> for ByteLimit {
    fn from(bytes: usize) -> Self {
        Self::new(bytes)
    }
}

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

impl<Cod, D, CS> Builder<NeedsSortKey, Cod, NeedsFlushStrategy, Natural, D, CS> {
    /// Select one complete order for initial runs and merging.
    #[must_use]
    pub fn order<O>(self, order: O) -> Builder<HasOrder<O>, Cod, NeedsLimit, (), D, CS> {
        Builder {
            sort_key: HasOrder(order),
            codec: self.codec,
            flush: NeedsLimit,
            compare: (),
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }
}

impl<O, Cod, D, CS> Builder<HasOrder<O>, Cod, NeedsLimit, (), D, CS> {
    /// Limit the number of buffered items.
    #[must_use]
    pub fn max_buffer_items(
        self,
        max_items: usize,
    ) -> Builder<HasOrder<O>, Cod, ItemLimit, (), D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: ItemLimit { max_items },
            compare: (),
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
    }
    /// Set the modeled byte budget for the complete-order path.
    #[must_use]
    pub fn memory_budget(
        self,
        budget: impl Into<ByteLimit>,
    ) -> Builder<HasOrder<O>, Cod, ByteLimit, (), D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: budget.into(),
            compare: (),
            dedup: self.dedup,
            chunk_sort: self.chunk_sort,
            config: self.config,
        }
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
    ///
    /// The unkeyed path accounts for occupied inline storage, requested backing
    /// replacement overlap, and the configured write buffer during spilling.
    /// The keyed path retains its item-size-only byte threshold.
    /// Neither path accounts for arbitrary sorting scratch or codec allocations.
    #[must_use]
    pub fn fixed_size_budget<T: 'static>(
        self,
        budget: usize,
    ) -> Builder<SK, Cod, HasFlushStrategy<T>, Cmp, D, CS> {
        Builder {
            sort_key: self.sort_key,
            codec: self.codec,
            flush: HasFlushStrategy(FlushStrategy::FixedSize { budget }, None),
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
    CS: ChunkSorter<T, KeyCompare<SK, Cmp>>,
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
    CS: ChunkSorter<T, KeyCompare<SK, Cmp>>,
{
    /// Build the [`Sorter`] (keyed merge path).
    #[must_use]
    pub fn build(self) -> Sorter<T, SK, Cod, Cmp, D, CS, Keyed> {
        let HasFlushStrategy(flush, item_size) = self.flush;
        let (flush, item_size) = match flush {
            FlushStrategy::FixedSize { budget } => {
                let item_size: ItemSize<T> = Box::new(|_| std::mem::size_of::<T>());
                (FlushStrategy::Bytes { budget }, Some(item_size))
            }
            legacy => (legacy, item_size),
        };

        Sorter {
            session: Session::new(
                OwnedBuffer::new(item_size),
                RunAccumulator::new(
                    self.sort_key.0,
                    self.codec.0,
                    self.compare,
                    self.chunk_sort,
                    self.config.merge,
                ),
            ),
            dedup: self.dedup,
            flush,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<O, Cod, D, CS> Builder<HasOrder<O>, HasKeyedCodec<Cod>, ItemLimit, (), D, CS> {
    /// Build an item-limited sorter using the complete order.
    #[must_use]
    pub fn build<T>(self) -> Sorter<T, O, Cod, (), D, CS, Keyed, ItemLimit>
    where
        O: DeriveKey<T, StoredKey = Cod::Key> + Compare<T>,
        Cod: KeyedCodec
            + EncodeKeyedRun<T>
            + EncodeKeyedRun<
                <<Cod as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record,
            >,
        O: Compare<<<Cod as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>,
        for<'a> CS: ChunkSorter<crate::key::Keyed<usize, Cod::Key>, KeyedEntryCompare<'a, T, O>>,
    {
        let accumulator = OrderedAccumulator {
            order: self.sort_key.0,
            codec: self.codec.0,
            chunk_sort: self.chunk_sort,
            merge_config: self.config.merge,
            runs: Vec::new(),
            pending: Vec::new(),
            accounting: (),
        };
        Sorter {
            session: Session::new(OwnedBuffer::new(None), accumulator),
            dedup: self.dedup,
            flush: FlushStrategy::Items {
                max_items: self.flush.max_items,
            },
            _marker: std::marker::PhantomData,
        }
    }
}

impl<O, Cod, D, CS> Builder<HasOrder<O>, HasKeyedCodec<Cod>, ByteLimit, (), D, CS> {
    /// Build a byte-limited sorter using the complete order.
    #[must_use]
    pub fn build<T>(self) -> Sorter<T, O, Cod, (), D, CS, Keyed, ByteLimit>
    where
        T: GetSize,
        Cod: KeyedCodec + EncodeKeyedRun<T> + EncodeKeyedRun<KeyedRecord<Cod>>,
        O: DeriveKey<T, StoredKey = Cod::Key>
            + EstimateKeyMemory<T>
            + Compare<T>
            + Compare<KeyedRecord<Cod>>,
        CS: EstimateSortMemory<crate::key::Keyed<usize, Cod::Key>>,
        for<'a> CS: ChunkSorter<crate::key::Keyed<usize, Cod::Key>, KeyedEntryCompare<'a, T, O>>,
    {
        let accumulator = OrderedAccumulator {
            order: self.sort_key.0,
            codec: self.codec.0,
            chunk_sort: self.chunk_sort,
            merge_config: self.config.merge,
            runs: Vec::new(),
            pending: Vec::new(),
            accounting: ByteAccounting::empty(),
        };
        Sorter {
            session: Session::new(OwnedBuffer::new(None), accumulator),
            dedup: self.dedup,
            flush: FlushStrategy::Bytes {
                budget: self.flush.bytes,
            },
            _marker: std::marker::PhantomData,
        }
    }
}

type KeyedRecord<C> = <<C as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record;

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
pub struct Sorter<T, SK, Cod, Cmp, D, CS, M = Unkeyed, L = ()>
where
    L: binding::Binding<T, SK, Cod, Cmp, CS, M>,
{
    session: <L as binding::Binding<T, SK, Cod, Cmp, CS, M>>::Session,
    dedup: D,
    flush: FlushStrategy,
    _marker: std::marker::PhantomData<M>,
}

type SorterSession<T, SK, Cod, Cmp, CS, M> =
    Session<OwnedBuffer<T>, RunAccumulator<SK, Cod, Cmp, CS, M>>;

mod binding {
    use super::{KeyedCodec, MaybeItemSize, MergeConfig, SortedRun};

    pub struct OwnedBuffer<T> {
        pub(super) items: Vec<T>,
        pub(super) active_bytes: usize,
        pub(super) item_size: MaybeItemSize<T>,
    }

    pub struct Session<B, A> {
        pub(super) buffer: B,
        pub(super) accumulator: A,
    }

    pub struct RunAccumulator<SK, Cod, Cmp, CS, M> {
        pub(super) sort_key: SK,
        pub(super) codec: Cod,
        pub(super) compare: Cmp,
        pub(super) chunk_sort: CS,
        pub(super) merge_config: MergeConfig,
        pub(super) runs: Vec<SortedRun>,
        pub(super) _marker: std::marker::PhantomData<fn() -> M>,
    }

    pub struct OrderedAccumulator<O, C: KeyedCodec, CS, A = ()> {
        pub(super) order: O,
        pub(super) codec: C,
        pub(super) chunk_sort: CS,
        pub(super) merge_config: MergeConfig,
        pub(super) runs: Vec<SortedRun>,
        pub(super) pending: Vec<crate::key::Keyed<usize, C::Key>>,
        pub(super) accounting: A,
    }

    #[derive(Clone)]
    pub struct ByteAccounting {
        pub(super) payload_heap: Option<usize>,
        pub(super) retained_keys: Option<usize>,
        pub(super) missing_keys: Option<usize>,
        pub(super) derivation_scratch: usize,
    }

    pub trait Binding<T, SK, Cod, Cmp, CS, M> {
        type Session;
    }

    impl<T, SK, Cod, Cmp, CS, M> Binding<T, SK, Cod, Cmp, CS, M> for () {
        type Session = super::SorterSession<T, SK, Cod, Cmp, CS, M>;
    }

    impl<T, O, Cod: crate::codec::KeyedCodec, CS> Binding<T, O, Cod, (), CS, super::Keyed>
        for super::ItemLimit
    {
        type Session = Session<OwnedBuffer<T>, OrderedAccumulator<O, Cod, CS>>;
    }
    impl<T, O, Cod: crate::codec::KeyedCodec, CS> Binding<T, O, Cod, (), CS, super::Keyed>
        for super::ByteLimit
    {
        type Session = Session<OwnedBuffer<T>, OrderedAccumulator<O, Cod, CS, ByteAccounting>>;
    }
}

use binding::{ByteAccounting, OrderedAccumulator, OwnedBuffer, RunAccumulator, Session};

type ItemSize<T> = Box<dyn Fn(&T) -> usize + Send + Sync>;
type MaybeItemSize<T> = Option<ItemSize<T>>;

/// Shared observation of a buffer's logical contents.
///
/// The count includes items with no payload bytes and must agree with the number
/// of items exposed for the next commit. Observation must be cheap: use existing
/// metadata rather than scanning records or constructing views.
///
/// This capability does not imply insertion, traversal, memory usage, allocation
/// behavior, or lifecycle ownership.
pub trait Buffer {
    /// Return the number of logical items currently buffered.
    fn len(&self) -> usize;

    /// Return whether the logical item count is zero.
    ///
    /// Overrides must remain equivalent to `self.len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Buffer for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<T> Buffer for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<T, const N: usize> Buffer for [T; N] {
    fn len(&self) -> usize {
        N
    }
}

fn growth_target<T>(items: &Vec<T>, required: usize) -> Option<usize> {
    let layout = Layout::array::<T>(required).ok()?;
    let capacity = items.capacity();
    if layout.size() == 0 || required <= capacity {
        return Some(required);
    }
    let doubled = capacity.checked_mul(2).unwrap_or(required);
    let requested = doubled.max(required);
    if Layout::array::<T>(requested).is_ok() {
        Some(requested)
    } else {
        Some(required)
    }
}

struct OwnedStorageMemory {
    item_count: usize,
    insertion_peak_bytes: usize,
    spill_peak_bytes: usize,
}

impl<T> OwnedBuffer<T> {
    fn new(item_size: MaybeItemSize<T>) -> Self {
        Self {
            items: Vec::new(),
            active_bytes: 0,
            item_size,
        }
    }

    fn growth_target(&self, required: usize) -> Option<usize> {
        growth_target(&self.items, required)
    }

    fn storage_memory(&self, additional_items: usize) -> OwnedStorageMemory {
        let item_count = self.items.len().checked_add(additional_items);
        let occupied_bytes = item_count.and_then(|count| {
            let layout = Layout::array::<T>(count).ok()?;
            Some(layout.size())
        });
        let insertion_peak_bytes = item_count.and_then(|required| {
            let requested_capacity = self.growth_target(required)?;
            let current_capacity = self.items.capacity();
            if requested_capacity <= current_capacity {
                return occupied_bytes;
            }

            let old_layout = Layout::array::<T>(current_capacity).ok()?;
            let requested_layout = Layout::array::<T>(requested_capacity).ok()?;
            let old_bytes = old_layout.size();
            let requested_bytes = requested_layout.size();
            old_bytes.checked_add(requested_bytes)
        });

        OwnedStorageMemory {
            item_count: item_count.unwrap_or(usize::MAX),
            insertion_peak_bytes: insertion_peak_bytes.unwrap_or(usize::MAX),
            spill_peak_bytes: occupied_bytes.unwrap_or(usize::MAX),
        }
    }

    fn reserve_for_push(&mut self) {
        let required = self.items.len().checked_add(1).expect("capacity overflow");
        let requested_capacity = self.growth_target(required).expect("capacity overflow");
        if requested_capacity > self.items.capacity() {
            let additional = requested_capacity - self.items.len();
            self.items.reserve_exact(additional);
        }
    }
}

impl<T> Buffer for OwnedBuffer<T> {
    fn len(&self) -> usize {
        self.items.len()
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

trait StorageOwner {
    type Storage;

    fn storage(&self) -> &Self::Storage;
    fn storage_mut(&mut self) -> &mut Self::Storage;
}

impl<T> StorageOwner for OwnedBuffer<T> {
    type Storage = Self;

    fn storage(&self) -> &Self::Storage {
        self
    }

    fn storage_mut(&mut self) -> &mut Self::Storage {
        self
    }
}

trait ChunkBuffer {
    type Item;
    type Items: AsMut<[Self::Item]>;

    fn items(self) -> Self::Items;
    fn clear(self);
}

impl<'items, T> ChunkBuffer for &'items mut OwnedBuffer<T> {
    type Item = T;
    type Items = &'items mut [T];

    fn items(self) -> Self::Items {
        &mut self.items
    }

    fn clear(self) {
        self.items.clear();
        self.active_bytes = 0;
    }
}

impl FlushStrategy {
    fn fixed_size_peak(memory: &OwnedStorageMemory, write_buffer_bytes: usize) -> usize {
        let spill_peak_bytes = if memory.item_count == 0 {
            0
        } else {
            memory.spill_peak_bytes.saturating_add(write_buffer_bytes)
        };
        memory.insertion_peak_bytes.max(spill_peak_bytes)
    }

    fn should_flush_before_push<T>(
        &self,
        buffer: &OwnedBuffer<T>,
        write_buffer_bytes: usize,
    ) -> bool {
        let Self::FixedSize { budget } = self else {
            return false;
        };
        let memory = buffer.storage_memory(1);
        let projected_peak = Self::fixed_size_peak(&memory, write_buffer_bytes);
        projected_peak >= *budget
    }

    fn should_flush_after_push<T>(
        &self,
        buffer: &OwnedBuffer<T>,
        write_buffer_bytes: usize,
    ) -> bool {
        match self {
            FlushStrategy::Bytes { budget } => buffer.active_bytes >= *budget,
            FlushStrategy::FixedSize { budget } => {
                let memory = buffer.storage_memory(0);
                let current_peak = Self::fixed_size_peak(&memory, write_buffer_bytes);
                current_peak >= *budget
            }
            FlushStrategy::Items { max_items } => buffer.len() >= *max_items,
        }
    }
}

impl<B, A> Session<B, A> {
    fn new(buffer: B, accumulator: A) -> Self {
        Self {
            buffer,
            accumulator,
        }
    }

    fn buffer(&self) -> &B::Storage
    where
        B: StorageOwner,
    {
        self.buffer.storage()
    }
}

trait AccumulateItems<I> {
    type Error;
    type Final;

    fn accumulate(&mut self, items: &mut [I]) -> Result<(), Self::Error>;
    fn finish(self) -> Result<Self::Final, Self::Error>;
}

impl<B, A> Session<B, A> {
    fn commit<E>(&mut self) -> Result<(), E>
    where
        B: StorageOwner,
        B::Storage: Buffer,
        for<'items> &'items mut B::Storage: ChunkBuffer,
        for<'items> A: AccumulateItems<<&'items mut B::Storage as ChunkBuffer>::Item, Error = E>,
    {
        if self.buffer().is_empty() {
            return Ok(());
        }

        let storage = self.buffer.storage_mut();
        {
            let mut items = ChunkBuffer::items(&mut *storage);
            self.accumulator.accumulate(items.as_mut())?;
        }

        ChunkBuffer::clear(storage);
        Ok(())
    }

    fn finish<E, F>(mut self) -> Result<F, E>
    where
        B: StorageOwner,
        B::Storage: Buffer,
        for<'items> &'items mut B::Storage: ChunkBuffer,
        for<'items> A:
            AccumulateItems<<&'items mut B::Storage as ChunkBuffer>::Item, Error = E, Final = F>,
    {
        self.commit()?;
        let finish_accumulator =
            <A as AccumulateItems<<&mut B::Storage as ChunkBuffer>::Item>>::finish;
        finish_accumulator(self.accumulator)
    }
}

impl<T, SK, Cod, Cmp, CS, M> Session<OwnedBuffer<T>, RunAccumulator<SK, Cod, Cmp, CS, M>> {
    fn push<E>(&mut self, item: T, policy: &FlushStrategy) -> Result<(), E>
    where
        RunAccumulator<SK, Cod, Cmp, CS, M>: AccumulateItems<T, Error = E>,
    {
        let write_buffer_bytes = self.accumulator.merge_config.write_buffer_bytes;
        if !self.buffer.is_empty()
            && policy.should_flush_before_push(&self.buffer, write_buffer_bytes)
        {
            self.commit()?;
        }

        if matches!(policy, FlushStrategy::FixedSize { .. }) {
            self.buffer.reserve_for_push();
        }
        self.buffer.extend([item]);

        if policy.should_flush_after_push(&self.buffer, write_buffer_bytes) {
            self.commit()?;
        }

        Ok(())
    }
}

impl<SK, Cod, Cmp, CS, M> RunAccumulator<SK, Cod, Cmp, CS, M> {
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

impl<T, SK, Cod, Cmp, CS> AccumulateItems<T> for RunAccumulator<SK, Cod, Cmp, CS, Unkeyed>
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
    CS: ChunkSorter<T, KeyCompare<SK, Cmp>>,
{
    type Error = MergeError<Cod::Error>;
    type Final = RunMerge<Cod, KeyCompare<SK, Cmp>>;

    fn accumulate(&mut self, items: &mut [T]) -> Result<(), MergeError<Cod::Error>> {
        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        self.chunk_sort.sort(items, &item_cmp);

        let run_writer = RunWriter::new_unkeyed(self.codec, &self.merge_config);
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

impl<T, SK, Cod, Cmp, CS> AccumulateItems<T> for RunAccumulator<SK, Cod, Cmp, CS, Keyed>
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
    CS: ChunkSorter<T, KeyCompare<SK, Cmp>>,
{
    type Error = MergeError<Cod::Error>;
    type Final = KeyedRunMerge<Cod, Cmp, KeyCompare<SK, Cmp>>;

    fn accumulate(&mut self, items: &mut [T]) -> Result<(), MergeError<Cod::Error>> {
        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        self.chunk_sort.sort(items, &item_cmp);

        let run_writer = RunWriter::new_keyed(self.codec, &self.merge_config);
        let records = items.iter();
        let run = run_writer.write_sorted(records, |item| *item)?;
        self.runs.push(run);

        Ok(())
    }

    fn finish(self) -> Result<Self::Final, MergeError<Cod::Error>> {
        let item_cmp = KeyCompare::new(self.sort_key, self.compare);
        let run_merger = RunMerger::new(self.codec, item_cmp, self.merge_config);

        run_merger.merge_keyed(self.runs, self.compare)
    }
}

trait Accounting {
    fn prepared(&mut self);
    fn retire(&mut self);
}

impl Accounting for () {
    fn prepared(&mut self) {}
    fn retire(&mut self) {}
}

impl ByteAccounting {
    fn empty() -> Self {
        Self {
            payload_heap: Some(0),
            retained_keys: Some(0),
            missing_keys: Some(0),
            derivation_scratch: 0,
        }
    }

    fn include(&mut self, heap: usize, key: &KeyMemory) {
        self.payload_heap = self.payload_heap.and_then(|n| n.checked_add(heap));
        self.missing_keys = self
            .missing_keys
            .and_then(|n| n.checked_add(key.retained_heap_bytes()));
        self.derivation_scratch = self.derivation_scratch.max(key.derivation_scratch_bytes());
    }
}

impl Accounting for ByteAccounting {
    fn prepared(&mut self) {
        self.retained_keys = self
            .retained_keys
            .and_then(|n| n.checked_add(self.missing_keys?));
        self.missing_keys = Some(0);
        self.derivation_scratch = 0;
    }

    fn retire(&mut self) {
        *self = Self::empty();
    }
}

impl<T, O, C, CS, A: Accounting> AccumulateItems<T> for OrderedAccumulator<O, C, CS, A>
where
    C: KeyedCodec
        + EncodeKeyedRun<T>
        + EncodeKeyedRun<<<C as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>,
    O: DeriveKey<T, StoredKey = C::Key>
        + Compare<T>
        + Compare<<<C as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>,
    for<'a> CS: ChunkSorter<crate::key::Keyed<usize, C::Key>, KeyedEntryCompare<'a, T, O>>,
{
    type Error = MergeError<C::Error>;
    type Final = KeyedRunMerge<C, O, (), Keyed>;

    fn accumulate(&mut self, items: &mut [T]) -> Result<(), Self::Error> {
        let required = items.len();
        let requested = growth_target(&self.pending, required).expect("entry capacity overflow");
        if requested > self.pending.capacity() {
            self.pending.reserve_exact(requested - self.pending.len());
        }
        let comparison = KeyedEntryCompare::new(items, &self.order);
        comparison.prepare_missing_keys(&mut self.pending);
        self.accounting.prepared();
        self.chunk_sort.sort(&mut self.pending, &comparison);
        let writer = RunWriter::new_keyed(self.codec, &self.merge_config);
        let records = comparison.ordered_records(&self.pending);
        let run = writer.write_supplied(records)?;
        self.runs.push(run);
        self.pending.clear();
        self.accounting.retire();
        Ok(())
    }

    fn finish(self) -> Result<Self::Final, Self::Error> {
        let merger = RunMerger::new(self.codec, self.order, self.merge_config);
        merger.merge_under_order(self.runs)
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
    CS: ChunkSorter<T, KeyCompare<SK, Cmp>>,
{
    /// Add an item to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing to disk fails.
    pub fn push(&mut self, item: T) -> Result<(), MergeError<Cod::Error>> {
        self.session.push(item, &self.flush)
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
    CS: ChunkSorter<T, KeyCompare<SK, Cmp>>,
{
    /// Add an item to the sorter.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing to disk fails.
    pub fn push(&mut self, item: T) -> Result<(), MergeError<Cod::Error>> {
        self.session.push(item, &self.flush)
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

impl<T, O, C, CS> Session<OwnedBuffer<T>, OrderedAccumulator<O, C, CS, ByteAccounting>>
where
    T: GetSize,
    C: KeyedCodec + EncodeKeyedRun<T> + EncodeKeyedRun<KeyedRecord<C>>,
    O: DeriveKey<T, StoredKey = C::Key>
        + EstimateKeyMemory<T>
        + Compare<T>
        + Compare<KeyedRecord<C>>,
    CS: EstimateSortMemory<crate::key::Keyed<usize, C::Key>>,
    for<'a> CS: ChunkSorter<crate::key::Keyed<usize, C::Key>, KeyedEntryCompare<'a, T, O>>,
{
    fn peak(&self, additional: usize, accounting: &ByteAccounting) -> Option<usize> {
        let count = self.buffer.items.len().checked_add(additional)?;
        let occupied = Layout::array::<T>(count).ok()?.size();
        let requested = self.buffer.growth_target(count)?;
        let insertion_storage = if requested > self.buffer.items.capacity() {
            let old = Layout::array::<T>(self.buffer.items.capacity())
                .ok()?
                .size();
            let new = Layout::array::<T>(requested).ok()?.size();
            old.checked_add(new)?
        } else {
            occupied
        };
        let entries = &self.accumulator.pending;
        let existing_entries = Layout::array::<crate::key::Keyed<usize, C::Key>>(entries.len())
            .ok()?
            .size();
        let retained_keys = accounting.retained_keys?;
        let payload = accounting.payload_heap?;
        let sort_memory = self.accumulator.chunk_sort.sort_memory(count, &());
        let workspace = sort_memory.retained_bytes();
        let insertion = insertion_storage
            .checked_add(payload)?
            .checked_add(existing_entries)?
            .checked_add(retained_keys)?
            .checked_add(workspace)?;
        if count == 0 {
            return Some(insertion);
        }
        let completed_entries = Layout::array::<crate::key::Keyed<usize, C::Key>>(count)
            .ok()?
            .size();
        let entry_request = growth_target(entries, count)?;
        let entry_preparation = if entry_request > entries.capacity() {
            let old = Layout::array::<crate::key::Keyed<usize, C::Key>>(entries.capacity())
                .ok()?
                .size();
            let new = Layout::array::<crate::key::Keyed<usize, C::Key>>(entry_request)
                .ok()?
                .size();
            old.checked_add(new)?
        } else {
            completed_entries
        };
        let base = occupied.checked_add(payload)?.checked_add(workspace)?;
        let existing_key_heap = retained_keys;
        let all_key_heap = retained_keys.checked_add(accounting.missing_keys?)?;
        let reserve = base
            .checked_add(entry_preparation)?
            .checked_add(existing_key_heap)?;
        let completed = base
            .checked_add(completed_entries)?
            .checked_add(all_key_heap)?;
        let derive = completed.checked_add(accounting.derivation_scratch)?;
        let sorting = completed.checked_add(sort_memory.scratch_bytes())?;
        let writing = completed.checked_add(self.accumulator.merge_config.write_buffer_bytes)?;
        Some(insertion.max(reserve).max(derive).max(sorting).max(writing))
    }

    fn push_measured(&mut self, item: T, budget: usize) -> Result<(), MergeError<C::Error>> {
        let heap = item.get_heap_size();
        let key = self.accumulator.order.estimate_key_memory(&item);
        let mut projected = self.accumulator.accounting.clone();
        projected.include(heap, &key);
        let peak = self.peak(1, &projected).unwrap_or(usize::MAX);
        if !self.buffer.is_empty() && peak >= budget {
            self.commit()?;
        }
        self.buffer.reserve_for_push();
        self.buffer.extend([item]);
        self.accumulator.accounting.include(heap, &key);
        let peak = self
            .peak(0, &self.accumulator.accounting)
            .unwrap_or(usize::MAX);
        if peak >= budget {
            self.commit()?;
        }
        Ok(())
    }
}

type OrderedOutput<C, O, D> = Sorted<<D as Dedup<KeyedRunMerge<C, O, (), Keyed>>>::Deduped>;

impl<T, O, C, D, CS> Sorter<T, O, C, (), D, CS, Keyed, ItemLimit>
where
    C: KeyedCodec
        + EncodeKeyedRun<T>
        + EncodeKeyedRun<<<C as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>,
    O: DeriveKey<T, StoredKey = C::Key>
        + Compare<T>
        + Compare<<<C as KeyedCodec>::KeyedCursor<BufReader<File>> as CodecCursor>::Record>,
    for<'a> CS: ChunkSorter<crate::key::Keyed<usize, C::Key>, KeyedEntryCompare<'a, T, O>>,
{
    /// Add an item, spilling when the configured count is reached.
    ///
    /// # Errors
    /// Returns an error if spilling fails.
    pub fn push(&mut self, item: T) -> Result<(), MergeError<C::Error>> {
        self.session.buffer.extend([item]);
        if self.flush.should_flush_after_push(
            &self.session.buffer,
            self.session.accumulator.merge_config.write_buffer_bytes,
        ) {
            self.session.commit()?;
        }
        Ok(())
    }

    /// Finish the remaining run and return the merged output.
    ///
    /// # Errors
    /// Returns an error if spilling or merging fails.
    pub fn finish(self) -> Result<OrderedOutput<C, O, D>, MergeError<C::Error>>
    where
        D: Dedup<KeyedRunMerge<C, O, (), Keyed>>,
    {
        let dedup = self.dedup;
        let merged = self.session.finish()?;
        Ok(Sorted::new(dedup.dedup(merged)))
    }
}

impl<T, O, C, D, CS> Sorter<T, O, C, (), D, CS, Keyed, ByteLimit>
where
    T: GetSize,
    C: KeyedCodec + EncodeKeyedRun<T> + EncodeKeyedRun<KeyedRecord<C>>,
    O: DeriveKey<T, StoredKey = C::Key>
        + EstimateKeyMemory<T>
        + Compare<T>
        + Compare<KeyedRecord<C>>,
    CS: EstimateSortMemory<crate::key::Keyed<usize, C::Key>>,
    for<'a> CS: ChunkSorter<crate::key::Keyed<usize, C::Key>, KeyedEntryCompare<'a, T, O>>,
{
    /// Add an item under the modeled byte budget.
    ///
    /// # Errors
    /// Returns an error if spilling fails.
    pub fn push(&mut self, item: T) -> Result<(), MergeError<C::Error>> {
        let FlushStrategy::Bytes { budget } = self.flush else {
            unreachable!()
        };
        self.session.push_measured(item, budget)
    }

    /// Finish remaining runs and return the merged output.
    ///
    /// # Errors
    /// Returns an error if spilling or merging fails.
    pub fn finish(self) -> Result<OrderedOutput<C, O, D>, MergeError<C::Error>>
    where
        D: Dedup<KeyedRunMerge<C, O, (), Keyed>>,
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
    ) -> RunAccumulator<Owned<fn(&u64) -> u64>, U64Codec, Natural, Sequential, Unkeyed> {
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
    ) -> RunAccumulator<Owned<fn(&u64) -> u64>, U64KeyedCodec, Natural, Sequential, Keyed> {
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
        RunAccumulator<Owned<fn(&u64) -> u64>, U64Codec, Natural, Sequential, Unkeyed>,
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
