//! In-memory chunk sorting strategies.
//!
//! [`ChunkSorter`] abstracts the algorithm used to sort each
//! in-memory chunk before it is flushed to disk. The core crate
//! provides [`Sequential`] (single-threaded, always available) and
//! `Parallel` (rayon-based, behind `feature = "rayon"`). Users
//! with exotic needs can implement the trait themselves.

use std::cmp::Ordering;

/// Sort a mutable slice using a provided comparison function.
///
/// Implementations choose the sort algorithm and threading model.
/// The comparison function is provided by the sorter engine, which
/// composes `SortKey` and `Compare` into a single closure.
///
/// ```
/// use spillover::chunk::{ChunkSorter, Sequential};
///
/// let mut data = vec![3, 1, 4, 1, 5];
/// Sequential.sort(&mut data, Ord::cmp);
/// assert_eq!(data, vec![1, 1, 3, 4, 5]);
/// ```
pub trait ChunkSorter<T> {
    /// Sort `chunk` in place according to `cmp`.
    ///
    /// The `Send + Sync` bounds on the comparator are required to
    /// support parallel implementations like `Parallel`. For
    /// sequential use, these bounds are automatically satisfied by
    /// any closure that does not capture mutable references.
    fn sort(&self, chunk: &mut [T], cmp: impl Fn(&T, &T) -> Ordering + Send + Sync);
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
/// Sequential.sort(&mut data, Ord::cmp);
/// assert_eq!(data, vec!["apple", "banana", "cherry"]);
/// ```
pub struct Sequential;

impl<T> ChunkSorter<T> for Sequential {
    #[inline]
    fn sort(&self, chunk: &mut [T], cmp: impl Fn(&T, &T) -> Ordering + Send + Sync) {
        chunk.sort_unstable_by(cmp);
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
impl<T: Send> ChunkSorter<T> for Parallel {
    #[inline]
    fn sort(&self, chunk: &mut [T], cmp: impl Fn(&T, &T) -> Ordering + Send + Sync) {
        use rayon::prelude::*;
        chunk.par_sort_unstable_by(cmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_sorts_integers() {
        let mut data = vec![5, 3, 1, 4, 2];
        Sequential.sort(&mut data, Ord::cmp);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sequential_sorts_empty_slice() {
        let mut data: Vec<i32> = vec![];
        Sequential.sort(&mut data, Ord::cmp);
        assert!(data.is_empty(), "sorting an empty slice should be a no-op");
    }

    #[test]
    fn sequential_sorts_single_element() {
        let mut data = vec![42];
        Sequential.sort(&mut data, Ord::cmp);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn sequential_sorts_already_sorted() {
        let mut data = vec![1, 2, 3, 4, 5];
        Sequential.sort(&mut data, Ord::cmp);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn sequential_sorts_reverse() {
        let mut data = vec![5, 4, 3, 2, 1];
        Sequential.sort(&mut data, |a, b| b.cmp(a));
        assert_eq!(
            data,
            vec![5, 4, 3, 2, 1],
            "reverse comparator should preserve reverse order"
        );
    }

    #[test]
    fn sequential_sorts_with_custom_comparator() {
        let mut data = vec![(3, "c"), (1, "a"), (2, "b")];
        Sequential.sort(&mut data, |a, b| a.0.cmp(&b.0));
        assert_eq!(data, vec![(1, "a"), (2, "b"), (3, "c")]);
    }

    #[test]
    fn sequential_sorts_strings_by_length() {
        let mut data = vec!["hello", "hi", "hey"];
        Sequential.sort(&mut data, |a, b| a.len().cmp(&b.len()));
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
