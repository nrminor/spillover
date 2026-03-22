//! Key comparison traits and built-in comparators.
//!
//! [`Compare`] defines how two sort keys are ordered. It is
//! deliberately separated from [`SortKey`](crate::key::SortKey) so that
//! one comparator serves any key extractor producing the same key
//! type, and vice versa. Three built-in implementations cover the
//! common cases: [`Natural`] delegates to [`Ord`], [`Reverse`]
//! flips any comparator, and [`CompareVia`] compares through
//! [`AsRef`] so that e.g. `Vec<u8>` and `&[u8]` keys compare as
//! the underlying `[u8]`.

use std::{cmp::Ordering, marker::PhantomData};

/// Compare two keys by reference. Implementations define the
/// ordering used by the sort and merge engines.
///
/// The `?Sized` bound on `K` allows comparing unsized types like
/// `[u8]` and `str` directly, which is useful with [`CompareVia`].
///
/// ```
/// use spillover::compare::{Compare, Natural, Reverse};
///
/// assert_eq!(Natural.compare(&1, &2), std::cmp::Ordering::Less);
/// assert_eq!(Reverse(Natural).compare(&1, &2), std::cmp::Ordering::Greater);
/// ```
pub trait Compare<K: ?Sized> {
    /// Return the ordering of `a` relative to `b`.
    fn compare(&self, a: &K, b: &K) -> Ordering;
}

/// Natural ordering via [`Ord`]. Any key type that implements `Ord`
/// gets comparison for free through this blanket implementation.
///
/// This is the default comparator — users who are happy with `Ord`
/// never need to think about the `Compare` trait at all.
///
/// ```
/// use spillover::compare::{Compare, Natural};
///
/// // Works with any Ord type.
/// assert_eq!(Natural.compare(&"apple", &"banana"), std::cmp::Ordering::Less);
/// assert_eq!(Natural.compare(&42u64, &42u64), std::cmp::Ordering::Equal);
/// ```
pub struct Natural;

impl<K: Ord + ?Sized> Compare<K> for Natural {
    #[inline]
    fn compare(&self, a: &K, b: &K) -> Ordering {
        a.cmp(b)
    }
}

/// Reverse any comparator's ordering.
///
/// ```
/// use spillover::compare::{Compare, Natural, Reverse};
///
/// let rev = Reverse(Natural);
/// assert_eq!(rev.compare(&1, &2), std::cmp::Ordering::Greater);
/// assert_eq!(rev.compare(&2, &1), std::cmp::Ordering::Less);
/// assert_eq!(rev.compare(&1, &1), std::cmp::Ordering::Equal);
/// ```
pub struct Reverse<C>(pub C);

impl<K: ?Sized, C: Compare<K>> Compare<K> for Reverse<C> {
    #[inline]
    fn compare(&self, a: &K, b: &K) -> Ordering {
        self.0.compare(a, b).reverse()
    }
}

/// Compare keys through [`AsRef`], so that different key types
/// that dereference to the same underlying type compare
/// identically.
///
/// For example, `CompareVia<[u8]>` compares both `Vec<u8>` keys
/// and `&[u8]` keys as `[u8]`, producing the same lexicographic
/// ordering regardless of ownership.
///
/// ```
/// use spillover::compare::{Compare, CompareVia};
///
/// let cmp: CompareVia<[u8]> = CompareVia::new();
/// let a = vec![1u8, 2, 3];
/// let b = vec![1u8, 2, 4];
/// assert_eq!(cmp.compare(&a, &b), std::cmp::Ordering::Less);
/// ```
pub struct CompareVia<Target: ?Sized>(PhantomData<fn() -> *const Target>);

impl<Target: ?Sized> CompareVia<Target> {
    /// Create a new `CompareVia` comparator.
    #[must_use]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Target: ?Sized> Default for CompareVia<Target> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, Target> Compare<K> for CompareVia<Target>
where
    K: AsRef<Target>,
    Target: Ord + ?Sized,
{
    #[inline]
    fn compare(&self, a: &K, b: &K) -> Ordering {
        a.as_ref().cmp(b.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_orders_integers() {
        assert_eq!(Natural.compare(&1, &2), Ordering::Less);
        assert_eq!(Natural.compare(&2, &1), Ordering::Greater);
        assert_eq!(Natural.compare(&5, &5), Ordering::Equal);
    }

    #[test]
    fn natural_orders_strings() {
        assert_eq!(Natural.compare(&"aardvark", &"zebra"), Ordering::Less);
    }

    #[test]
    fn natural_orders_byte_slices() {
        let a: &[u8] = &[1, 2, 3];
        let b: &[u8] = &[1, 2, 4];
        assert_eq!(
            Natural.compare(a, b),
            Ordering::Less,
            "Natural should compare unsized [u8] slices"
        );
    }

    #[test]
    fn reverse_flips_ordering() {
        let rev = Reverse(Natural);
        assert_eq!(rev.compare(&1, &2), Ordering::Greater);
        assert_eq!(rev.compare(&2, &1), Ordering::Less);
        assert_eq!(
            rev.compare(&1, &1),
            Ordering::Equal,
            "Reverse should not affect Equal"
        );
    }

    #[test]
    fn reverse_composes() {
        let double_rev = Reverse(Reverse(Natural));
        assert_eq!(
            double_rev.compare(&1, &2),
            Ordering::Less,
            "reversing twice should restore natural ordering"
        );
    }

    #[test]
    fn compare_via_vec_u8_as_byte_slice() {
        let cmp: CompareVia<[u8]> = CompareVia::new();
        let a = vec![1u8, 2, 3];
        let b = vec![1u8, 2, 4];
        assert_eq!(cmp.compare(&a, &b), Ordering::Less);
    }

    #[test]
    fn compare_via_string_as_str() {
        let cmp: CompareVia<str> = CompareVia::new();
        let a = "alpha".to_string();
        let b = "beta".to_string();
        assert_eq!(cmp.compare(&a, &b), Ordering::Less);
    }

    #[test]
    fn compare_via_default() {
        let cmp: CompareVia<[u8]> = CompareVia::default();
        let a = vec![1u8];
        let b = vec![2u8];
        assert_eq!(
            cmp.compare(&a, &b),
            Ordering::Less,
            "Default should produce a working CompareVia"
        );
    }

    #[test]
    fn reverse_with_compare_via() {
        let cmp = Reverse(CompareVia::<[u8]>::new());
        let a = vec![1u8, 2];
        let b = vec![1u8, 3];
        assert_eq!(
            cmp.compare(&a, &b),
            Ordering::Greater,
            "Reverse(CompareVia) should reverse the AsRef-based ordering"
        );
    }

    #[test]
    fn natural_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Natural>();
    }

    #[test]
    fn reverse_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Reverse<Natural>>();
    }

    #[test]
    fn compare_via_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CompareVia<[u8]>>();
    }
}
