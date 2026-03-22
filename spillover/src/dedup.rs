//! Post-merge deduplication strategies.
//!
//! [`Dedup`] defines a transform applied to the fully sorted,
//! merged stream. [`Identity`] passes the stream through unchanged.
//! [`AdjacentDedup`] drops consecutive items that compare equal on
//! a user-provided predicate, keeping the first item from each run.

/// A transform applied to the sorted, merged stream.
///
/// The trait is generic over the stream's error type `E` so that
/// error composition is entirely in the user's hands. Infallible
/// dedups like [`Identity`] and [`AdjacentDedup`] pass source
/// errors through unchanged. A fallible dedup can require
/// `E: From<MyDedupError>` on its impl and use `?` to convert its
/// own errors, relying on the user's `#[from]` enum to unify them.
///
/// More specialized dedup strategies (like group-by-key-and-reduce
/// for database construction) can be implemented directly on this
/// trait by downstream crates.
pub trait Dedup<T, E> {
    /// The type of items in the output stream. For filtering dedups
    /// like [`AdjacentDedup`] this is the same as `T`. For dedups
    /// that collapse items (e.g., a group-reduce), it may differ.
    type Output;

    /// Transform the sorted stream.
    fn dedup(
        self,
        sorted: impl Iterator<Item = Result<T, E>>,
    ) -> impl Iterator<Item = Result<Self::Output, E>>;
}

/// Pass-through dedup that does nothing. This is the default when
/// no deduplication is needed.
///
/// ```
/// use spillover::dedup::{Dedup, Identity};
///
/// let stream = vec![Ok(1), Ok(2), Ok(3)];
/// let out: Vec<Result<i32, std::convert::Infallible>> =
///     Identity.dedup(stream.into_iter()).collect();
/// assert_eq!(out.len(), 3);
/// ```
pub struct Identity;

impl<T, E> Dedup<T, E> for Identity {
    type Output = T;

    fn dedup(
        self,
        sorted: impl Iterator<Item = Result<T, E>>,
    ) -> impl Iterator<Item = Result<T, E>> {
        sorted
    }
}

/// Drop consecutive items where a predicate says they are equal,
/// keeping the first item from each run of duplicates.
///
/// Since the stream is sorted, "consecutive" means "all duplicates
/// are adjacent," so this provides exact deduplication on whatever
/// the predicate tests. The predicate receives two items by
/// reference and returns `true` if they should be considered
/// duplicates.
///
/// Requires `T: Clone` because the iterator must retain the
/// previous item for comparison while also yielding it to the
/// caller.
///
/// ```
/// use spillover::dedup::{Dedup, AdjacentDedup};
///
/// let stream: Vec<Result<i32, std::convert::Infallible>> =
///     vec![Ok(1), Ok(1), Ok(2), Ok(3), Ok(3), Ok(3)];
///
/// let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
///     .dedup(stream.into_iter())
///     .map(|r| r.unwrap())
///     .collect();
///
/// assert_eq!(out, vec![1, 2, 3]);
/// ```
pub struct AdjacentDedup<F> {
    eq_fn: F,
}

impl<F> AdjacentDedup<F> {
    /// Create a new `AdjacentDedup` with the given equality
    /// predicate. The predicate should return `true` for items
    /// that are considered duplicates.
    pub fn new(eq_fn: F) -> Self {
        Self { eq_fn }
    }
}

impl<T, E, F> Dedup<T, E> for AdjacentDedup<F>
where
    T: Clone,
    F: Fn(&T, &T) -> bool,
{
    type Output = T;

    fn dedup(
        self,
        sorted: impl Iterator<Item = Result<T, E>>,
    ) -> impl Iterator<Item = Result<T, E>> {
        AdjacentDedupIter {
            source: sorted,
            eq_fn: self.eq_fn,
            last_emitted: None,
            fused: false,
        }
    }
}

/// The iterator produced by [`AdjacentDedup::dedup`].
struct AdjacentDedupIter<T, I, F> {
    source: I,
    eq_fn: F,
    last_emitted: Option<T>,
    fused: bool,
}

impl<T, E, I, F> Iterator for AdjacentDedupIter<T, I, F>
where
    T: Clone,
    I: Iterator<Item = Result<T, E>>,
    F: Fn(&T, &T) -> bool,
{
    type Item = Result<T, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.fused {
            return None;
        }

        loop {
            match self.source.next() {
                Some(Ok(item)) => {
                    let is_dup = self
                        .last_emitted
                        .as_ref()
                        .is_some_and(|prev| (self.eq_fn)(prev, &item));
                    if is_dup {
                        continue;
                    }
                    self.last_emitted = Some(item.clone());
                    return Some(Ok(item));
                }
                Some(Err(e)) => {
                    self.fused = true;
                    return Some(Err(e));
                }
                None => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;

    #[test]
    fn identity_passes_through() {
        let stream = vec![Ok(1), Ok(2), Ok(3)];
        let out: Vec<Result<i32, Infallible>> = Identity.dedup(stream.into_iter()).collect();
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn identity_passes_through_errors() {
        #[derive(Debug, PartialEq)]
        struct E;

        let stream: Vec<Result<i32, E>> = vec![Ok(1), Err(E), Ok(3)];
        let out: Vec<_> = Identity.dedup(stream.into_iter()).collect();
        assert_eq!(out.len(), 3);
        assert!(out[1].is_err());
    }

    #[test]
    fn adjacent_dedup_removes_consecutive_duplicates() {
        let stream: Vec<Result<i32, Infallible>> = vec![Ok(1), Ok(1), Ok(2), Ok(3), Ok(3), Ok(3)];
        let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
            .dedup(stream.into_iter())
            .map(|r| r.expect("infallible"))
            .collect();
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn adjacent_dedup_keeps_all_when_no_duplicates() {
        let stream: Vec<Result<i32, Infallible>> = vec![Ok(1), Ok(2), Ok(3)];
        let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
            .dedup(stream.into_iter())
            .map(|r| r.expect("infallible"))
            .collect();
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn adjacent_dedup_all_same_yields_one() {
        let stream: Vec<Result<i32, Infallible>> = vec![Ok(5), Ok(5), Ok(5), Ok(5)];
        let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
            .dedup(stream.into_iter())
            .map(|r| r.expect("infallible"))
            .collect();
        assert_eq!(out, vec![5]);
    }

    #[test]
    fn adjacent_dedup_empty_stream() {
        let stream: Vec<Result<i32, Infallible>> = vec![];
        let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
            .dedup(stream.into_iter())
            .map(|r| r.expect("infallible"))
            .collect();
        assert!(out.is_empty());
    }

    #[test]
    fn adjacent_dedup_single_item() {
        let stream: Vec<Result<i32, Infallible>> = vec![Ok(42)];
        let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
            .dedup(stream.into_iter())
            .map(|r| r.expect("infallible"))
            .collect();
        assert_eq!(out, vec![42]);
    }

    #[test]
    fn adjacent_dedup_custom_predicate() {
        // Dedup by first element of tuple only.
        let stream: Vec<Result<(i32, &str), Infallible>> = vec![
            Ok((1, "a")),
            Ok((1, "b")),
            Ok((2, "c")),
            Ok((2, "d")),
            Ok((3, "e")),
        ];
        let out: Vec<(i32, &str)> =
            AdjacentDedup::new(|a: &(i32, &str), b: &(i32, &str)| a.0 == b.0)
                .dedup(stream.into_iter())
                .map(|r| r.expect("infallible"))
                .collect();
        assert_eq!(
            out,
            vec![(1, "a"), (2, "c"), (3, "e")],
            "should keep the first item from each run"
        );
    }

    #[test]
    fn adjacent_dedup_error_propagates_and_fuses() {
        #[derive(Debug, PartialEq, Clone)]
        struct TestError;

        let stream: Vec<Result<i32, TestError>> = vec![Ok(1), Ok(2), Err(TestError), Ok(3), Ok(3)];
        let mut iter = AdjacentDedup::new(|a: &i32, b: &i32| a == b).dedup(stream.into_iter());

        assert_eq!(iter.next(), Some(Ok(1)));
        assert_eq!(iter.next(), Some(Ok(2)));
        assert_eq!(iter.next(), Some(Err(TestError)));
        assert_eq!(iter.next(), None, "should fuse after error");
        assert_eq!(iter.next(), None, "should stay fused");
    }

    #[test]
    fn adjacent_dedup_non_adjacent_duplicates_kept() {
        let stream: Vec<Result<i32, Infallible>> = vec![Ok(1), Ok(2), Ok(1), Ok(3), Ok(2)];
        let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
            .dedup(stream.into_iter())
            .map(|r| r.expect("infallible"))
            .collect();
        assert_eq!(
            out,
            vec![1, 2, 1, 3, 2],
            "non-adjacent duplicates should be kept (only adjacent runs are collapsed)"
        );
    }

    mod proptests {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            #[test]
            fn adjacent_dedup_never_emits_consecutive_equal_items(
                data in proptest::collection::vec(0i32..10, 0..200),
            ) {
                let stream = data.into_iter().map(Ok::<_, Infallible>);
                let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
                    .dedup(stream)
                    .map(|r| r.expect("infallible"))
                    .collect();

                prop_assert!(
                    out.windows(2).all(|w| w[0] != w[1]),
                    "output should never have consecutive equal items, got: {out:?}"
                );
            }

            #[test]
            fn adjacent_dedup_preserves_all_unique_values_from_sorted_input(
                mut data in proptest::collection::vec(0i32..100, 0..200),
            ) {
                data.sort_unstable();
                let expected: Vec<i32> = {
                    let mut v = data.clone();
                    v.dedup();
                    v
                };

                let stream = data.into_iter().map(Ok::<_, Infallible>);
                let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
                    .dedup(stream)
                    .map(|r| r.expect("infallible"))
                    .collect();

                prop_assert_eq!(out, expected);
            }

            #[test]
            fn adjacent_dedup_output_is_subset_of_input(
                data in proptest::collection::vec(0i32..10, 0..200),
            ) {
                let stream = data.clone().into_iter().map(Ok::<_, Infallible>);
                let out: Vec<i32> = AdjacentDedup::new(|a: &i32, b: &i32| a == b)
                    .dedup(stream)
                    .map(|r| r.expect("infallible"))
                    .collect();

                prop_assert!(
                    out.len() <= data.len(),
                    "output should never be larger than input"
                );
                for item in &out {
                    prop_assert!(
                        data.contains(item),
                        "every output item should appear in the input"
                    );
                }
            }
        }
    }
}
