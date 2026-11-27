//! Adjacent deduplication of sorted sources.
//!
//! Configure an [`AdjacentDeduplicator`] on the sorter builder, or wrap a
//! source directly with [`Deduplicated::new`]. Arbitrary transformations
//! formerly implemented using `Dedup` can instead be composed after `finish`.
//! Exact deduplication requires the sort order to place equivalent records
//! together; otherwise only adjacent matches are removed.

use crate::{SortedItemsError, sorter::VisitSortedItems};

/// Maintains the representative of the last accepted record.
///
/// The first record is always accepted: Core calls `capture` then `replace`,
/// even when the algorithm is pre-seeded. This can discard preallocated
/// storage and temporarily retain both old and new representatives. Later
/// matching records are skipped; nonmatches are accepted and passed to
/// `update`. Both replacement and update must make the accepted record the
/// representative. Override `update` to reuse storage after initialization.
///
/// These callbacks are infallible and must not fail. Implementations must
/// compare against the last accepted record rather than a growing seen-set.
/// Owned iteration and borrowed visitation have independent record bounds.
/// If both are supported, their implementations must have consistent
/// semantics and share state, including when traversal modes are mixed.
/// Retained representatives are outside the sorter's initial-spill memory budget.
///
/// Implement the algorithm, then pass it to [`Builder::dedup`](crate::sorter::Builder::dedup)
/// or use the supplied adapter directly:
///
/// ```
/// use std::convert::Infallible;
/// use spillover::dedup::{AdjacentDeduplicator, Deduplicated};
///
/// #[derive(Default)]
/// struct ByBucket {
///     previous: Option<u64>,
/// }
///
/// impl AdjacentDeduplicator<u64> for ByBucket {
///     type Representative = u64;
///
///     fn matches(&self, record: &u64) -> bool {
///         self.previous == Some(*record / 2)
///     }
///
///     fn capture(&self, record: &u64) -> u64 {
///         *record / 2
///     }
///
///     fn replace(&mut self, representative: u64) {
///         self.previous = Some(representative);
///     }
/// }
///
/// let source = [Ok::<_, Infallible>(1), Ok(2), Ok(3), Ok(4)].into_iter();
/// let output = Deduplicated::new(source, Some(ByBucket::default()));
/// let values = output.collect::<Result<Vec<_>, _>>()?;
/// assert_eq!(values, [1, 2, 4]);
/// # Ok::<(), Infallible>(())
/// ```
pub trait AdjacentDeduplicator<R: ?Sized> {
    /// Captured comparison data owned by the algorithm.
    type Representative;
    /// Whether the record matches the installed representative.
    fn matches(&self, record: &R) -> bool;
    /// Capture comparison data for a record.
    fn capture(&self, record: &R) -> Self::Representative;
    /// Install captured comparison data.
    ///
    /// If the algorithm implements this trait for several record types, a direct
    /// call may require a qualified trait path: the representative alone need not
    /// identify `R`. Normal builder use supplies that context internally.
    fn replace(&mut self, representative: Self::Representative);
    /// Make this algorithm represent the record, optionally reusing its storage.
    ///
    /// This also initializes an empty algorithm when called directly. Core calls
    /// it only after the first capture and installation.
    fn update(&mut self, record: &R) {
        let representative = self.capture(record);
        self.replace(representative);
    }
}

/// Type marker for a builder without deduplication.
pub struct NoDedup(());

impl<R: ?Sized> AdjacentDeduplicator<R> for NoDedup {
    type Representative = ();
    fn matches(&self, _: &R) -> bool {
        false
    }
    fn capture(&self, _: &R) {}
    fn replace(&mut self, (): ()) {}
}

/// Predicate-based adjacent deduplication retaining a clone of the last item.
///
/// The predicate receives the last accepted item and the candidate, in that
/// order. For direct use, pass this algorithm to [`Deduplicated::new`]:
///
/// ```
/// use spillover::{dedup::Deduplicated, AdjacentDedup};
/// let source = [Ok::<_, std::convert::Infallible>(1), Ok(1), Ok(2)].into_iter();
/// let algorithm = AdjacentDedup::new(|a: &i32, b: &i32| a == b);
/// let output: Vec<_> = Deduplicated::new(source, Some(algorithm))
///     .collect::<Result<_, _>>().unwrap();
/// assert_eq!(output, [1, 2]);
/// ```
pub struct AdjacentDedup<F, T> {
    eq: F,
    previous: Option<T>,
}

impl<F, T> AdjacentDedup<F, T> {
    /// Construct an initially empty predicate-based algorithm.
    pub fn new(eq: F) -> Self {
        Self { eq, previous: None }
    }
}

impl<F, T: Clone> AdjacentDeduplicator<T> for AdjacentDedup<F, T>
where
    F: Fn(&T, &T) -> bool,
{
    type Representative = T;
    fn matches(&self, record: &T) -> bool {
        self.previous
            .as_ref()
            .is_some_and(|previous| (self.eq)(previous, record))
    }
    fn capture(&self, record: &T) -> T {
        record.clone()
    }
    fn replace(&mut self, representative: T) {
        self.previous = Some(representative);
    }
    fn update(&mut self, record: &T) {
        match self.previous.as_mut() {
            Some(previous) => previous.clone_from(record),
            None => self.previous = Some(record.clone()),
        }
    }
}

/// Sorted source with optional adjacent deduplication.
///
/// Disabled traversal forwards directly, including the source's behavior after
/// an error. Enabled iteration stops permanently after a source error, but a
/// clean `None` does not fuse it or clear its history. Borrowed visitation
/// preserves source versus sink error classification and stops on sink error.
/// Iterator and visitor capabilities are independent. The borrowed record need
/// not be the owned item, and borrowed visitation does not require owned extraction.
pub struct Deduplicated<S, D> {
    source: S,
    algorithm: Option<D>,
    initialized: bool,
    failed: bool,
}

impl<S, D> Deduplicated<S, D> {
    /// Wrap a source; `None` disables deduplication.
    ///
    /// A disabled adapter still needs a type for its algorithm. Use
    /// `Deduplicated<_, NoDedup>` when no particular algorithm type is needed.
    pub fn new(source: S, algorithm: Option<D>) -> Self {
        Self {
            source,
            algorithm,
            initialized: false,
            failed: false,
        }
    }
}

pub(crate) fn accept<R: ?Sized, D: AdjacentDeduplicator<R>>(
    algorithm: &mut D,
    initialized: &mut bool,
    record: &R,
) -> bool {
    if !*initialized {
        let representative = algorithm.capture(record);
        algorithm.replace(representative);
        *initialized = true;
        return true;
    }
    if algorithm.matches(record) {
        return false;
    }
    algorithm.update(record);
    true
}

impl<S, D, T, E> Iterator for Deduplicated<S, D>
where
    S: Iterator<Item = Result<T, E>>,
    D: AdjacentDeduplicator<T>,
{
    type Item = Result<T, E>;
    fn next(&mut self) -> Option<Self::Item> {
        let Some(algorithm) = self.algorithm.as_mut() else {
            return self.source.next();
        };
        if self.failed {
            return None;
        }
        loop {
            match self.source.next() {
                Some(Ok(record)) => {
                    if accept(algorithm, &mut self.initialized, &record) {
                        return Some(Ok(record));
                    }
                }
                Some(Err(error)) => {
                    self.failed = true;
                    return Some(Err(error));
                }
                None => return None,
            }
        }
    }
}

impl<S, D> VisitSortedItems for Deduplicated<S, D>
where
    S: VisitSortedItems,
    D: AdjacentDeduplicator<S::Record>,
{
    type Record = S::Record;
    type Error = S::Error;
    fn visit_items<F, E>(self, mut sink: F) -> Result<(), SortedItemsError<S::Error, E>>
    where
        F: FnMut(&S::Record) -> Result<(), E>,
    {
        let Self {
            source,
            algorithm,
            mut initialized,
            failed,
        } = self;
        let Some(mut algorithm) = algorithm else {
            return source.visit_items(sink);
        };
        if failed {
            return Ok(());
        }
        source.visit_items(|record| {
            if accept(&mut algorithm, &mut initialized, record) {
                sink(record)
            } else {
                Ok(())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use proptest::prelude::*;

    use super::*;

    fn deduplicate<T: Clone, F: Fn(&T, &T) -> bool>(
        source: impl Iterator<Item = Result<T, Infallible>>,
        eq: F,
    ) -> Vec<T> {
        Deduplicated::new(source, Some(AdjacentDedup::new(eq)))
            .map(|item| item.expect("infallible"))
            .collect()
    }

    #[test]
    fn disabled_passes_items_and_errors_through() {
        let source = [Ok(1), Err("error"), Ok(3)].into_iter();
        let output: Vec<_> = Deduplicated::<_, NoDedup>::new(source, None).collect();
        assert_eq!(output.len(), 3);
        assert!(output[1].is_err());
        let source = [Ok::<_, Infallible>(1), Ok(2), Ok(3)].into_iter();
        let output: Vec<_> = Deduplicated::<_, NoDedup>::new(source, None).collect();
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn adjacent_examples() {
        let eq = |a: &i32, b: &i32| a == b;
        for (input, expected) in [
            (vec![1, 1, 2, 3, 3, 3], vec![1, 2, 3]),
            (vec![1, 2, 3], vec![1, 2, 3]),
            (vec![5, 5, 5, 5], vec![5]),
            (vec![], vec![]),
            (vec![42], vec![42]),
            (vec![1, 2, 1, 3, 2], vec![1, 2, 1, 3, 2]),
        ] {
            let source = input.into_iter().map(Ok);
            assert_eq!(deduplicate(source, eq), expected);
        }
        let input = [(1, "a"), (1, "b"), (2, "c"), (2, "d"), (3, "e")];
        let output = deduplicate(input.into_iter().map(Ok), |a, b| a.0 == b.0);
        assert_eq!(output, [(1, "a"), (2, "c"), (3, "e")]);
    }

    #[test]
    fn error_propagates_and_fuses() {
        let source = [Ok(1), Ok(2), Err("error"), Ok(3), Ok(3)].into_iter();
        let algorithm = AdjacentDedup::new(|a: &i32, b: &i32| a == b);
        let mut output = Deduplicated::new(source, Some(algorithm));
        assert_eq!(output.next(), Some(Ok(1)));
        assert_eq!(output.next(), Some(Ok(2)));
        assert_eq!(output.next(), Some(Err("error")));
        assert_eq!(output.next(), None);
        assert_eq!(output.next(), None);
    }

    proptest! {
        #[test]
        fn never_emits_consecutive_equal_items(data in proptest::collection::vec(0i32..10, 0..200)) {
            let output = deduplicate(data.into_iter().map(Ok), |a, b| a == b);
            prop_assert!(output.windows(2).all(|pair| pair[0] != pair[1]));
        }

        #[test]
        fn preserves_unique_values_from_sorted_input(mut data in proptest::collection::vec(0i32..100, 0..200)) {
            data.sort_unstable();
            let mut expected = data.clone();
            expected.dedup();
            let output = deduplicate(data.into_iter().map(Ok), |a, b| a == b);
            prop_assert_eq!(output, expected);
        }

        #[test]
        fn output_is_subset_of_input(data in proptest::collection::vec(0i32..10, 0..200)) {
            let output = deduplicate(data.clone().into_iter().map(Ok), |a, b| a == b);
            prop_assert!(output.len() <= data.len());
            for item in output {
                prop_assert!(data.contains(&item));
            }
        }
    }
}
