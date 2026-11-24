//! In-memory chunk sorting strategies.
//!
//! [`ChunkSorter`] controls the *mechanics* of sorting each
//! in-memory chunk before it is flushed to disk — the algorithm,
//! threading model, and memory strategy. It does **not** define
//! *what ordering* to use. The [`Sorter`](crate::sorter::Sorter)
//! passes a [`Compare`] implementation to [`ChunkSorter::sort`].
//!
//! The core crate provides [`Sequential`] (single-threaded, always
//! available) and `Parallel` (rayon-based, behind
//! `feature = "rayon"`). Domain crates can implement the trait for
//! specialized algorithms like radix sort that accelerate specific
//! data shapes while still falling back to the provided comparator
//! for tiebreaking.

use crate::compare::Compare;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Sort a mutable slice in place.
///
/// Implementations choose the sort algorithm and threading model
/// but must respect the ordering defined by `cmp`. The comparator
/// is provided by the [`Sorter`](crate::sorter::Sorter) — a
/// `ChunkSorter` should never impose its own ordering.
///
/// Implementations *may* use domain knowledge to sort faster (e.g.
/// radix sort on known key structure) as long as the final order
/// is consistent with `cmp`. When a fast path leaves items in
/// equal-key groups, `cmp` must be used to refine those groups.
///
/// ```
/// use spillover::chunk::{ChunkSorter, Sequential};
///
/// let mut data = vec![3, 1, 4, 1, 5];
/// Sequential.sort(&mut data, &spillover::compare::Natural);
/// assert_eq!(data, vec![1, 1, 3, 4, 5]);
/// ```
pub trait ChunkSorter<T, Cmp: ?Sized> {
    /// Sort `chunk` in place according to `cmp`.
    ///
    fn sort(&self, chunk: &mut [T], cmp: &Cmp);
}

/// Modeled algorithm workspace retained across sorting and temporary scratch.
pub struct SortMemory {
    retained_bytes: usize,
    scratch_bytes: usize,
}

impl SortMemory {
    /// No modeled dynamic workspace.
    pub const NONE: Self = Self {
        retained_bytes: 0,
        scratch_bytes: 0,
    };

    /// Additional workspace used only while sorting.
    #[must_use]
    pub fn temporary(bytes: usize) -> Self {
        Self {
            retained_bytes: 0,
            scratch_bytes: bytes,
        }
    }

    /// Workspace that may remain allocated between sorts.
    #[must_use]
    pub fn retained(bytes: usize) -> Self {
        Self {
            retained_bytes: bytes,
            scratch_bytes: 0,
        }
    }

    /// Set additional call-scoped workspace.
    #[must_use]
    pub fn with_scratch(mut self, bytes: usize) -> Self {
        self.scratch_bytes = bytes;
        self
    }

    /// Workspace that can overlap other phases.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Additional workspace during sorting.
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
}

/// Conservatively estimate modeled workspace for the actual sorted element.
///
/// Reports must bound retained workspace that can overlap an applicable
/// projection, as well as scratch used by the corresponding sort. Estimates
/// need not be monotone across different windows or summaries. These are
/// modeled allocation bounds, not measurements of process RSS.
pub trait EstimateSortMemory<T, Summary = ()> {
    /// Report retained and call-scoped workspace for this sort.
    fn sort_memory(&self, items: usize, summary: &Summary) -> SortMemory;
}

/// Supply the configured empty summary and bounds on summary construction.
///
/// In the returned `(H, W)`, `H` bounds the retained heap of each applicable
/// summary, including the configured empty summary and any supplied existing
/// summary; `W` bounds additional temporary construction scratch. The forecast
/// itself must not allocate, unless its allocations are included in the bounds.
pub trait SortMemorySummary {
    /// Summary consumed by the selected algorithm.
    type Summary;

    /// Construct the configured empty summary.
    fn summary(&self) -> Self::Summary;
    /// Bound each applicable summary and additional construction scratch.
    fn summary_allowance(&self, existing: Option<&Self::Summary>) -> (usize, usize);
}

/// Summarize an input and forecast summary construction before consuming it.
pub trait SummarizeSortInput<T: ?Sized>: SortMemorySummary {
    /// Produce the summary for an input.
    fn summarize(&self, input: &T) -> Self::Summary;
    /// Return `(H, W)` for this input and current summary.
    ///
    /// `H` bounds the retained heap of each current, incoming, combined, and
    /// configured-empty summary. `W` bounds additional construction scratch,
    /// including reset and recombination with an empty summary after a spill.
    /// Forecasting must not allocate, unless its allocations are included in
    /// these bounds. Forecasts need not be monotone.
    fn summary_allowance_for(&self, current: &Self::Summary, input: &T) -> (usize, usize);
}

/// Single-threaded sort via [`slice::sort_unstable_by`]. Always
/// available with no additional dependencies.
///
/// Unstable sort is preferred over stable sort because it avoids
/// an O(n) auxiliary allocation, which matters when the chunk is
/// large — this is, after all, a library for sorting data that
/// pushes memory limits.
///
/// ```
/// use spillover::chunk::{ChunkSorter, Sequential};
///
/// let mut data = vec!["banana", "apple", "cherry"];
/// Sequential.sort(&mut data, &spillover::compare::Natural);
/// assert_eq!(data, vec!["apple", "banana", "cherry"]);
/// ```
pub struct Sequential;

impl<T, Cmp: Compare<T> + ?Sized> ChunkSorter<T, Cmp> for Sequential {
    #[inline]
    fn sort(&self, chunk: &mut [T], cmp: &Cmp) {
        chunk.sort_unstable_by(|left, right| cmp.compare(left, right));
    }
}

impl<T> EstimateSortMemory<T> for Sequential {
    fn sort_memory(&self, _: usize, (): &()) -> SortMemory {
        SortMemory::NONE
    }
}

/// Parallel sort via rayon's [`par_sort_unstable_by`](rayon::slice::ParallelSliceMut::par_sort_unstable_by).
/// Available behind `feature = "rayon"`.
///
/// The `Send` bound on `T` and `Sync` bound on the comparison
/// function are inherent requirements of rayon and surface
/// naturally through this impl. Users who choose [`Sequential`]
/// never encounter these bounds.
#[cfg(feature = "rayon")]
pub struct Parallel;

#[cfg(feature = "rayon")]
impl<T: Send, Cmp: Compare<T> + Sync + ?Sized> ChunkSorter<T, Cmp> for Parallel {
    #[inline]
    fn sort(&self, chunk: &mut [T], cmp: &Cmp) {
        chunk.par_sort_unstable_by(|left, right| cmp.compare(left, right));
    }
}

#[cfg(feature = "rayon")]
impl<T> EstimateSortMemory<T> for Parallel {
    fn sort_memory(&self, _: usize, (): &()) -> SortMemory {
        SortMemory::NONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_sorts_integers() {
        let mut data = vec![5, 3, 1, 4, 2];
        Sequential.sort(&mut data, &crate::compare::Natural);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sequential_sorts_empty_slice() {
        let mut data: Vec<i32> = vec![];
        Sequential.sort(&mut data, &crate::compare::Natural);
        assert!(data.is_empty(), "sorting an empty slice should be a no-op");
    }

    #[test]
    fn sequential_sorts_single_element() {
        let mut data = vec![42];
        Sequential.sort(&mut data, &crate::compare::Natural);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn sequential_sorts_already_sorted() {
        let mut data = vec![1, 2, 3, 4, 5];
        Sequential.sort(&mut data, &crate::compare::Natural);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sequential_sorts_reverse() {
        let mut data = vec![5, 4, 3, 2, 1];
        Sequential.sort(&mut data, &crate::compare::Reverse(crate::compare::Natural));
        assert_eq!(
            data,
            vec![5, 4, 3, 2, 1],
            "reverse comparator should preserve reverse order"
        );
    }

    #[test]
    fn sequential_sorts_with_custom_comparator() {
        let mut data = vec![(3, "c"), (1, "a"), (2, "b")];
        Sequential.sort(
            &mut data,
            &crate::key::KeyCompare::new(
                crate::key::Owned(|a: &(i32, &str)| a.0),
                crate::compare::Natural,
            ),
        );
        assert_eq!(data, vec![(1, "a"), (2, "b"), (3, "c")]);
    }

    #[test]
    fn sequential_sorts_strings_by_length() {
        let mut data = vec!["hello", "hi", "hey"];
        Sequential.sort(
            &mut data,
            &crate::key::KeyCompare::new(
                crate::key::Owned(|a: &&str| a.len()),
                crate::compare::Natural,
            ),
        );
        assert_eq!(
            data,
            vec!["hi", "hey", "hello"],
            "should sort by string length"
        );
    }

    #[test]
    fn sequential_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Sequential>();
    }
}
