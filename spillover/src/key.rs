//! Sort key extraction from items.
//!
//! [`SortKey`] defines how a sort key is derived from an item. The
//! key may borrow from the item via a GAT lifetime (e.g., `&'a [u8]`
//! for a sequence slice) or may be fully owned (e.g., `f64` for a
//! quality score). For owned keys, the [`Owned`] adapter eliminates
//! GAT boilerplate by lifting a closure into a `SortKey` impl.

use std::cmp::Ordering;

/// Extract a sort key from an item.
///
/// The GAT `Key<'a>` allows the key to borrow from the item. For
/// example, a sort key that returns `&'a [u8]` borrows the
/// sequence bytes directly from the record without copying. For
/// owned keys like `f64` or `u64`, the lifetime parameter is
/// simply unused — or use the [`Owned`] adapter to avoid writing
/// the GAT at all.
///
/// `SortKey` is deliberately separated from [`Compare`](crate::compare::Compare)
/// so that the same comparator can serve any key extractor that
/// produces the same key type, and vice versa.
///
/// ```
/// use spillover::key::{SortKey, Owned};
///
/// let quality_key = Owned(|scores: &Vec<u8>| {
///     scores.iter().map(|&q| f64::from(q)).sum::<f64>()
///         / scores.len() as f64
/// });
///
/// let scores = vec![30u8, 40, 35];
/// let key = quality_key.key(&scores);
/// assert!((key - 35.0).abs() < f64::EPSILON);
/// ```
pub trait SortKey<T> {
    /// The key type, which may borrow from the item for lifetime `'a`.
    type Key<'a>
    where
        T: 'a;

    /// Extract the sort key from an item.
    fn key<'a>(&self, item: &'a T) -> Self::Key<'a>;

    /// Build a comparator closure that compares two items by their
    /// keys, using the provided comparison function. This is a
    /// convenience for passing to slice sort methods.
    fn item_cmp<'a, C>(&'a self, compare: &'a C) -> impl Fn(&T, &T) -> Ordering + 'a
    where
        C: for<'b> crate::compare::Compare<Self::Key<'b>>,
    {
        move |a, b| {
            let ka = self.key(a);
            let kb = self.key(b);
            compare.compare(&ka, &kb)
        }
    }
}

/// A [`Compare<T>`](crate::compare::Compare) implementation that
/// compares items by extracting keys via a [`SortKey`] and
/// delegating to a key comparator. This bridges the gap between
/// "compare keys" and "compare items" — the merge engine needs
/// the latter, but the user provides the former.
///
/// Cloning a `KeyCompare` is cheap when both the sort key and
/// comparator are zero-sized types (the common case).
#[derive(Clone, Copy)]
pub struct KeyCompare<SK, Cmp> {
    sort_key: SK,
    compare: Cmp,
}

impl<SK, Cmp> KeyCompare<SK, Cmp> {
    /// Create a new item comparator from a sort key and a key
    /// comparator.
    pub fn new(sort_key: SK, compare: Cmp) -> Self {
        Self { sort_key, compare }
    }
}

impl<T, SK, Cmp> crate::compare::Compare<T> for KeyCompare<SK, Cmp>
where
    SK: SortKey<T>,
    Cmp: for<'a> crate::compare::Compare<SK::Key<'a>>,
{
    fn compare(&self, a: &T, b: &T) -> Ordering {
        let ka = self.sort_key.key(a);
        let kb = self.sort_key.key(b);
        self.compare.compare(&ka, &kb)
    }
}

/// Adapter that lifts a key-extraction function into a [`SortKey`]
/// without requiring the caller to write out the GAT. Works for
/// any key that does not borrow from the item.
///
/// ```
/// use spillover::key::{SortKey, Owned};
///
/// let length_key = Owned(|s: &String| s.len());
/// assert_eq!(length_key.key(&"hello".to_string()), 5);
/// ```
#[derive(Clone, Copy)]
pub struct Owned<F>(pub F);

impl<T, K, F> SortKey<T> for Owned<F>
where
    F: Fn(&T) -> K,
{
    type Key<'a>
        = K
    where
        T: 'a;

    fn key(&self, item: &T) -> K {
        (self.0)(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::Natural;

    #[test]
    fn owned_adapter_extracts_key() {
        let key_fn = Owned(|v: &(i32, &str)| v.0);
        assert_eq!(
            key_fn.key(&(42, "hello")),
            42,
            "Owned should extract the key via the closure"
        );
    }

    #[test]
    fn owned_adapter_works_with_closures_that_compute() {
        let key_fn = Owned(|v: &Vec<i32>| v.iter().sum::<i32>());
        assert_eq!(
            key_fn.key(&vec![1, 2, 3]),
            6,
            "Owned should support arbitrary computation in the closure"
        );
    }

    #[test]
    fn borrowed_sort_key_returns_reference() {
        struct SeqKey;

        impl SortKey<(Vec<u8>, Vec<u8>)> for SeqKey {
            type Key<'a> = &'a [u8];

            fn key<'a>(&self, item: &'a (Vec<u8>, Vec<u8>)) -> &'a [u8] {
                &item.0
            }
        }

        let record = (b"ACGT".to_vec(), b"!!!!".to_vec());
        let key = SeqKey.key(&record);
        assert_eq!(
            key, b"ACGT",
            "a borrowed SortKey should return a reference into the item"
        );
    }

    #[test]
    fn item_cmp_orders_by_extracted_key() {
        let key_fn = Owned(|v: &(i32, &str)| v.0);
        let cmp = key_fn.item_cmp(&Natural);

        let a = (1, "first");
        let b = (3, "second");
        let c = (2, "third");

        assert_eq!(
            cmp(&a, &b),
            std::cmp::Ordering::Less,
            "item_cmp should order by the extracted key"
        );
        assert_eq!(
            cmp(&b, &c),
            std::cmp::Ordering::Greater,
            "item_cmp should order by the extracted key"
        );
        assert_eq!(
            cmp(&a, &a),
            std::cmp::Ordering::Equal,
            "item_cmp should return Equal for identical keys"
        );
    }

    #[test]
    fn item_cmp_with_borrowed_key() {
        struct NameKey;

        impl SortKey<(String, i32)> for NameKey {
            type Key<'a> = &'a str;

            fn key<'a>(&self, item: &'a (String, i32)) -> &'a str {
                &item.0
            }
        }

        let cmp = NameKey.item_cmp(&Natural);
        let alice = ("alice".to_string(), 1);
        let bob = ("bob".to_string(), 2);

        assert_eq!(
            cmp(&alice, &bob),
            std::cmp::Ordering::Less,
            "item_cmp should work with borrowed keys"
        );
    }

    #[test]
    fn item_cmp_can_sort_a_slice() {
        let key_fn = Owned(|v: &(i32, &str)| v.0);
        let cmp = key_fn.item_cmp(&Natural);

        let mut items = vec![(3, "c"), (1, "a"), (2, "b")];
        items.sort_by(&cmp);

        assert_eq!(
            items,
            vec![(1, "a"), (2, "b"), (3, "c")],
            "item_cmp should produce a closure usable with sort_by"
        );
    }
}
