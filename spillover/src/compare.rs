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

    /// `a <= b`
    fn le(&self, a: &K, b: &K) -> bool {
        self.compare(a, b) != Ordering::Greater
    }

    /// `a < b`
    fn lt(&self, a: &K, b: &K) -> bool {
        self.compare(a, b) == Ordering::Less
    }

    /// `a >= b`
    fn ge(&self, a: &K, b: &K) -> bool {
        self.compare(a, b) != Ordering::Less
    }

    /// `a > b`
    fn gt(&self, a: &K, b: &K) -> bool {
        self.compare(a, b) == Ordering::Greater
    }

    /// `a == b` according to this comparator.
    fn eq(&self, a: &K, b: &K) -> bool {
        self.compare(a, b) == Ordering::Equal
    }

    /// Return a reference to the greater of `a` and `b`. If equal,
    /// returns `a`.
    fn max<'a>(&self, a: &'a K, b: &'a K) -> &'a K {
        if self.ge(a, b) { a } else { b }
    }

    /// Return a reference to the lesser of `a` and `b`. If equal,
    /// returns `a`.
    fn min<'a>(&self, a: &'a K, b: &'a K) -> &'a K {
        if self.le(a, b) { a } else { b }
    }
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
#[derive(Debug, Clone, Copy, Default)]
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
#[derive(Debug, Clone, Copy)]
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

/// A transparent wrapper that gives any `T` an [`Ord`]
/// implementation derived from a [`Compare`] implementation.
///
/// This is useful when you need to put a value into a container
/// that requires `Ord` (like [`BinaryHeap`](std::collections::BinaryHeap)
/// or [`BTreeMap`](std::collections::BTreeMap)) but your ordering
/// comes from a [`Compare`] impl rather than the type's own `Ord`.
///
/// For zero-sized comparators like [`Natural`], [`Reverse`], and
/// [`CompareVia`], the wrapper adds no memory overhead — a
/// `WithOrd<T, Natural>` is the same size as `T`.
///
/// The wrapper implements [`Deref`](std::ops::Deref),
/// [`AsRef`], and [`Borrow`](std::borrow::Borrow) so that
/// methods on `T` are callable through it and it can be used
/// as a lookup key in collections.
///
/// When `T: Ord`, you can convert directly via `From`:
///
/// ```
/// use spillover::compare::WithOrd;
///
/// let ordered: WithOrd<i32, _> = WithOrd::from(42);
/// assert_eq!(*ordered, 42);
/// ```
///
/// For custom comparators:
///
/// ```
/// use spillover::compare::{Compare, Reverse, Natural, WithOrd};
///
/// let a = WithOrd::new(1, Reverse(Natural));
/// let b = WithOrd::new(2, Reverse(Natural));
/// // Under Reverse ordering, 2 < 1:
/// assert!(a < b);
/// ```
#[derive(Clone)]
pub struct WithOrd<T, C> {
    value: T,
    cmp: C,
}

impl<T, C> WithOrd<T, C> {
    /// Wrap a value with a comparator.
    pub fn new(value: T, cmp: C) -> Self {
        Self { value, cmp }
    }

    /// Unwrap, returning the inner value.
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Borrow the comparator.
    pub fn comparator(&self) -> &C {
        &self.cmp
    }
}

impl<T: Ord> From<T> for WithOrd<T, Natural> {
    fn from(value: T) -> Self {
        Self {
            value,
            cmp: Natural,
        }
    }
}

impl<T: Copy, C: Copy> Copy for WithOrd<T, C> {}

impl<T, C> std::ops::Deref for WithOrd<T, C> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T, C> AsRef<T> for WithOrd<T, C> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T, C> std::borrow::Borrow<T> for WithOrd<T, C> {
    fn borrow(&self) -> &T {
        &self.value
    }
}

impl<T: std::fmt::Debug, C> std::fmt::Debug for WithOrd<T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

impl<T: std::fmt::Display, C> std::fmt::Display for WithOrd<T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

impl<T, C: Compare<T>> Ord for WithOrd<T, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp.compare(&self.value, &other.value)
    }
}

impl<T, C: Compare<T>> PartialOrd for WithOrd<T, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, C: Compare<T>> Eq for WithOrd<T, C> {}

impl<T, C: Compare<T>> PartialEq for WithOrd<T, C> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp.compare(&self.value, &other.value) == Ordering::Equal
    }
}

impl<T, C: Compare<T>> std::hash::Hash for WithOrd<T, C>
where
    T: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
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
    fn le_lt_ge_gt_eq_convenience_methods() {
        assert!(Natural.le(&1, &2));
        assert!(Natural.le(&1, &1));
        assert!(!Natural.le(&2, &1));

        assert!(Natural.lt(&1, &2));
        assert!(!Natural.lt(&1, &1));

        assert!(Natural.ge(&2, &1));
        assert!(Natural.ge(&1, &1));
        assert!(!Natural.ge(&1, &2));

        assert!(Natural.gt(&2, &1));
        assert!(!Natural.gt(&1, &1));

        assert!(Natural.eq(&1, &1));
        assert!(!Natural.eq(&1, &2));
    }

    #[test]
    fn min_max_return_correct_references() {
        assert_eq!(Natural.min(&1, &2), &1);
        assert_eq!(Natural.max(&1, &2), &2);
        assert_eq!(Natural.min(&1, &1), &1, "min should return a on equal");
        assert_eq!(Natural.max(&1, &1), &1, "max should return a on equal");
    }

    #[test]
    fn convenience_methods_work_with_reverse() {
        let rev = Reverse(Natural);
        assert!(rev.lt(&2, &1), "under Reverse, 2 should be 'less than' 1");
        assert!(
            rev.gt(&1, &2),
            "under Reverse, 1 should be 'greater than' 2"
        );
        assert_eq!(
            rev.min(&1, &2),
            &2,
            "under Reverse, min should return the naturally larger value"
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

    #[test]
    fn with_ord_from_ord_type() {
        let a: WithOrd<i32, _> = WithOrd::from(1);
        let b: WithOrd<i32, _> = WithOrd::from(2);
        assert!(a < b);
        assert_eq!(*a, 1, "Deref should give access to the inner value");
    }

    #[test]
    fn with_ord_custom_comparator() {
        let a = WithOrd::new(1, Reverse(Natural));
        let b = WithOrd::new(2, Reverse(Natural));
        assert!(
            a > b,
            "under Reverse, WithOrd(1) should be greater than WithOrd(2)"
        );
    }

    #[test]
    fn with_ord_into_inner() {
        let w = WithOrd::new(String::from("hello"), Natural);
        assert_eq!(&*w, "hello");
        let s = w.into_inner();
        assert_eq!(s, "hello");
    }

    #[test]
    fn with_ord_debug_shows_inner_value() {
        let w = WithOrd::new(42, Natural);
        assert_eq!(format!("{w:?}"), "42");
    }

    #[test]
    fn with_ord_display_shows_inner_value() {
        let w = WithOrd::new(42, Natural);
        assert_eq!(format!("{w}"), "42");
    }

    #[test]
    fn with_ord_in_binary_heap() {
        use std::collections::BinaryHeap;

        // BinaryHeap is a max-heap; wrapping with Reverse gives
        // min-heap behavior. Popping yields smallest first.
        let mut heap = BinaryHeap::new();
        heap.push(std::cmp::Reverse(WithOrd::new(3, Natural)));
        heap.push(std::cmp::Reverse(WithOrd::new(1, Natural)));
        heap.push(std::cmp::Reverse(WithOrd::new(2, Natural)));

        let mut drained = Vec::new();
        while let Some(std::cmp::Reverse(entry)) = heap.pop() {
            drained.push(entry.into_inner());
        }
        assert_eq!(drained, vec![1, 2, 3]);
    }

    #[test]
    fn with_ord_in_btree_map() {
        use std::collections::BTreeMap;

        let mut map = BTreeMap::new();
        map.insert(WithOrd::new("b", Natural), 2);
        map.insert(WithOrd::new("a", Natural), 1);
        map.insert(WithOrd::new("c", Natural), 3);

        let keys: Vec<&str> = map.keys().map(|k| **k).collect();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn with_ord_zst_comparator_is_zero_overhead() {
        assert_eq!(
            std::mem::size_of::<WithOrd<u64, Natural>>(),
            std::mem::size_of::<u64>(),
            "WithOrd with a ZST comparator should be the same size as the inner type"
        );
        assert_eq!(
            std::mem::size_of::<WithOrd<u64, Reverse<Natural>>>(),
            std::mem::size_of::<u64>(),
            "WithOrd with Reverse(Natural) should also be zero overhead"
        );
    }
}
