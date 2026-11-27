//! Sort a whole batch in memory, without a codec, disk, or memory budget.
//!
//! Inputs and sorting scratch must fit in memory. Select an order implementing
//! [`Compare`], optionally an [`AdjacentDeduplicator`] and a [`ChunkSorter`],
//! then sort an existing slice or vector, or build a sorter and push inputs.
//! Deduplication only removes adjacent matches, so the order must group them.
//!
//! ```
//! use spillover::{AdjacentDedup, InMemoryBuilder as Builder, Natural};
//!
//! let mut values = vec![3, 1, 1, 2];
//! Builder::new().sort_by(Natural)
//!     .dedup(AdjacentDedup::new(|a: &i32, b: &i32| a == b))
//!     .sort_vec(&mut values);
//! assert_eq!(values, [1, 2, 3]);
//!
//! let mut sorter = Builder::new().sort_by(Natural).build_from(values);
//! sorter.push(0);
//! let result = sorter.finish().collect::<Result<Vec<_>, _>>().unwrap();
//! assert_eq!(result, [0, 1, 2, 3]);
//! ```

use std::convert::Infallible;

use super::{ChunkBuffer, Sorted, VisitSortedItems};
use crate::{
    SortedItemsError,
    chunk::{ChunkSorter, Sequential},
    compare::Compare,
    dedup::{self, AdjacentDeduplicator, Deduplicated, NoDedup},
};

/// Configuration for a single in-memory batch.
pub struct Builder<O = (), D = NoDedup, CS = Sequential> {
    order: O,
    dedup: Option<D>,
    algorithm: CS,
}

impl Builder {
    /// Start without an order or deduplication, using [`Sequential`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            order: (),
            dedup: None,
            algorithm: Sequential,
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl<D, CS> Builder<(), D, CS> {
    /// Select the complete comparison order.
    #[must_use]
    pub fn sort_by<O>(self, order: O) -> Builder<O, D, CS> {
        Builder {
            order,
            dedup: self.dedup,
            algorithm: self.algorithm,
        }
    }
}

impl<O, D, CS> Builder<O, D, CS> {
    /// Select an adjacent deduplicator. The first item starts fresh history,
    /// even if the supplied algorithm already contains a representative.
    #[must_use]
    pub fn dedup<Next>(self, algorithm: Next) -> Builder<O, Next, CS> {
        Builder {
            order: self.order,
            dedup: Some(algorithm),
            algorithm: self.algorithm,
        }
    }

    /// Select the sorting algorithm; the default is [`Sequential`].
    #[must_use]
    pub fn chunk_sort<Next>(self, algorithm: Next) -> Builder<O, D, Next> {
        Builder {
            order: self.order,
            dedup: self.dedup,
            algorithm,
        }
    }

    /// Sort and compact a vector in place, consuming this configuration.
    ///
    /// No second result vector is allocated. The algorithm may allocate scratch
    /// or retain a representative; the vector retains its capacity.
    pub fn sort_vec<R>(mut self, records: &mut Vec<R>)
    where
        O: Compare<R>,
        CS: ChunkSorter<R, O>,
        D: AdjacentDeduplicator<R>,
    {
        self.algorithm.sort(records, &self.order);
        if let Some(algorithm) = self.dedup.as_mut() {
            let mut initialized = false;
            records.retain(|record| dedup::accept(algorithm, &mut initialized, record));
        }
    }

    /// Start an empty owned batch. The item type is inferred from pushes.
    #[must_use]
    pub fn build<R>(self) -> Sorter<Vec<R>, O, D, CS>
    where
        O: Compare<R>,
        CS: ChunkSorter<R, O>,
        D: AdjacentDeduplicator<R>,
    {
        self.build_from(Vec::new())
    }

    /// Transfer an existing vector and its spare capacity into a batch.
    /// Further items can be pushed before finishing.
    #[must_use]
    pub fn build_from<R>(self, records: Vec<R>) -> Sorter<Vec<R>, O, D, CS>
    where
        O: Compare<R>,
        CS: ChunkSorter<R, O>,
        D: AdjacentDeduplicator<R>,
    {
        Sorter {
            storage: records,
            configuration: self,
        }
    }

    /// Start a batch in reusable storage, clearing its logical contents first.
    ///
    /// Clearing retains any capacity according to the storage implementation.
    /// The storage is borrowed through the output and any yielded borrowed
    /// values; all such uses must end before it can be reused.
    ///
    /// Custom storage implements [`ChunkBuffer`] for its mutable borrow and
    /// [`Extend`] for inputs accepted by [`Sorter::push`]. Its item container
    /// must support both mutable slice access and ordered iteration, with
    /// iteration preserving the slice order established by the sort.
    ///
    /// A live output prevents another build:
    ///
    /// ```compile_fail
    /// use spillover::{sorter::ChunkBuffer, InMemoryBuilder as Builder, Natural};
    /// struct Arena(Vec<i32>);
    /// impl<'a> ChunkBuffer for &'a mut Arena {
    ///     type Item = &'a i32;
    ///     type Items = Vec<&'a i32>;
    ///     fn items(self) -> Self::Items { self.0.iter().collect() }
    ///     fn clear(self) { self.0.clear(); }
    /// }
    /// let mut arena = Arena(vec![1]);
    /// let output = Builder::new().sort_by(Natural).build_in(&mut arena).finish();
    /// let next = Builder::new().sort_by(Natural).build_in(&mut arena);
    /// drop(output);
    /// ```
    ///
    /// A yielded view can outlive the output and still prevent reuse:
    ///
    /// ```compile_fail
    /// use spillover::{sorter::ChunkBuffer, InMemoryBuilder as Builder, Natural};
    /// struct Arena(Vec<i32>);
    /// impl<'a> ChunkBuffer for &'a mut Arena {
    ///     type Item = &'a i32;
    ///     type Items = Vec<&'a i32>;
    ///     fn items(self) -> Self::Items { self.0.iter().collect() }
    ///     fn clear(self) { self.0.clear(); }
    /// }
    /// let mut arena = Arena(vec![1]);
    /// let mut output = Builder::new().sort_by(Natural).build_in(&mut arena).finish();
    /// let view = output.next();
    /// drop(output);
    /// let next = Builder::new().sort_by(Natural).build_in(&mut arena);
    /// println!("{view:?}");
    /// ```
    #[must_use]
    pub fn build_in<'a, B>(self, storage: &'a mut B) -> Sorter<&'a mut B, O, D, CS>
    where
        <&'a mut B as ChunkBuffer>::Items: IntoIterator<Item = <&'a mut B as ChunkBuffer>::Item>,
        O: Compare<<&'a mut B as ChunkBuffer>::Item>,
        CS: ChunkSorter<<&'a mut B as ChunkBuffer>::Item, O>,
        D: AdjacentDeduplicator<<&'a mut B as ChunkBuffer>::Item>,
        for<'v> &'v mut B: ChunkBuffer,
    {
        ChunkBuffer::clear(&mut *storage);
        Sorter {
            storage,
            configuration: self,
        }
    }
}

impl<O, CS> Builder<O, NoDedup, CS> {
    /// Sort a slice without deduplication. A slice cannot be compacted.
    ///
    /// ```compile_fail
    /// use spillover::{AdjacentDedup, InMemoryBuilder as Builder, Natural};
    /// let mut values = [1, 1];
    /// Builder::new().sort_by(Natural)
    ///     .dedup(AdjacentDedup::new(|a: &i32, b: &i32| a == b))
    ///     .sort_slice(&mut values);
    /// ```
    pub fn sort_slice<R>(&self, records: &mut [R])
    where
        O: Compare<R>,
        CS: ChunkSorter<R, O>,
    {
        self.algorithm.sort(records, &self.order);
    }
}

/// A batch retained entirely in memory until it is finished.
pub struct Sorter<B, O, D, CS> {
    storage: B,
    configuration: Builder<O, D, CS>,
}

impl<R, O, D, CS> Sorter<Vec<R>, O, D, CS> {
    /// Move an input into the batch.
    pub fn push(&mut self, record: R) {
        self.storage.push(record);
    }

    /// Sort the batch; deduplication is performed lazily during traversal.
    #[must_use]
    pub fn finish(mut self) -> Output<std::vec::IntoIter<R>, D>
    where
        O: Compare<R>,
        CS: ChunkSorter<R, O>,
    {
        self.configuration
            .algorithm
            .sort(&mut self.storage, &self.configuration.order);
        output(self.storage.into_iter(), self.configuration.dedup)
    }
}

impl<'a, B, O, D, CS> Sorter<&'a mut B, O, D, CS> {
    /// Insert one input via the storage's [`Extend`] implementation.
    pub fn push<I>(&mut self, record: I)
    where
        B: Extend<I>,
    {
        self.storage.extend([record]);
    }

    /// Sort the storage's item container and return its native values.
    ///
    /// Its `IntoIterator` must traverse items in the same order as its
    /// `AsMut` slice. The returned values may borrow storage; even after
    /// dropping the output, escaped values can prevent its reuse.
    #[must_use]
    pub fn finish(self) -> Output<<<&'a mut B as ChunkBuffer>::Items as IntoIterator>::IntoIter, D>
    where
        &'a mut B: ChunkBuffer,
        <&'a mut B as ChunkBuffer>::Items: IntoIterator<Item = <&'a mut B as ChunkBuffer>::Item>,
        O: Compare<<&'a mut B as ChunkBuffer>::Item>,
        CS: ChunkSorter<<&'a mut B as ChunkBuffer>::Item, O>,
    {
        let mut items = ChunkBuffer::items(self.storage);
        self.configuration
            .algorithm
            .sort(items.as_mut(), &self.configuration.order);
        output(items.into_iter(), self.configuration.dedup)
    }
}

/// Sorted native items with optional lazy adjacent deduplication.
///
/// Iteration yields `Result<I::Item, Infallible>`. Calling `items()` after
/// partial iteration visits the remaining items with the same dedup history.
/// A visitor can fail in its sink, but the source cannot fail.
pub type Output<I, D = NoDedup> = Sorted<Deduplicated<BufferedSource<I>, D>>;

/// Iterator adapter yielding native values, which can themselves be borrowed.
pub struct BufferedSource<I> {
    remaining: I,
}

fn output<I, D>(remaining: I, algorithm: Option<D>) -> Output<I, D> {
    let source = BufferedSource { remaining };
    let deduplicated = Deduplicated::new(source, algorithm);
    Sorted::new(deduplicated)
}

impl<I: Iterator> Iterator for BufferedSource<I> {
    type Item = Result<I::Item, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        self.remaining.next().map(Ok)
    }
}

impl<I: Iterator> VisitSortedItems for BufferedSource<I> {
    type Record = I::Item;
    type Error = Infallible;

    fn visit_items<F, E>(self, mut sink: F) -> Result<(), SortedItemsError<Infallible, E>>
    where
        F: FnMut(&Self::Record) -> Result<(), E>,
    {
        for record in self.remaining {
            sink(&record).map_err(SortedItemsError::Sink)?;
        }
        Ok(())
    }
}
