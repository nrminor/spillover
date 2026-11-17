//! Sort key extraction from items.
//!
//! A [`SortKey`] extracts the value to sort by from each item —
//! it answers "what do we compare?" rather than "how do we
//! compare?" (which is [`Compare`](crate::compare::Compare)'s
//! job) or "what algorithm do we use?" (which is
//! [`ChunkSorter`](crate::chunk::ChunkSorter)'s job).
//!
//! Sort keys are transient, potentially borrowed values used
//! during in-memory comparison. They are distinct from the
//! *record key* stored by a
//! [`KeyedCodec`](crate::codec::KeyedCodec), which is a compact,
//! owned encoding written to disk for merge acceleration. Both
//! are derived from the same underlying data, but they may use
//! different representations. The `SortKey` builder path requires its comparator
//! to support both; complete-order sorting uses [`KeyedOrder`] for stored keys
//! and [`Compare`](crate::compare::Compare) for records.
//!
//! The key may borrow from the item via a GAT lifetime (e.g.,
//! `&'a [u8]` for a sequence slice) or may be fully owned (e.g.,
//! `f64` for a quality score). For owned keys, the [`Owned`]
//! adapter eliminates GAT boilerplate by lifting a closure into a
//! `SortKey` impl.

use std::cmp::Ordering;

/// An item associated with its original stored key.
///
/// The key may have been preserved through encoding and decoding the item.
pub struct Keyed<I, K> {
    item: I,
    stored_key: K,
}

impl<I, K> Keyed<I, K> {
    fn new<Input, O>(input: Input, order: &O) -> Self
    where
        Input: std::ops::Deref + Into<I>,
        O: DeriveKey<Input::Target, StoredKey = K> + ?Sized,
    {
        let stored_key = order.derive_key(&input);
        let item = input.into();
        Self { item, stored_key }
    }

    /// The associated item.
    #[must_use]
    pub fn item(&self) -> &I {
        &self.item
    }

    /// The original stored key for the associated item.
    #[must_use]
    pub fn stored_key(&self) -> &K {
        &self.stored_key
    }
}

impl<'item, I: ?Sized, K> Keyed<&'item I, K> {
    /// Derive a stored key for this item under the given order.
    #[must_use]
    pub fn derive<O: DeriveKey<I, StoredKey = K> + ?Sized>(item: &'item I, order: &O) -> Self {
        Self::new(item, order)
    }

    /// Borrow the associated item and stored key together.
    #[must_use]
    pub fn borrowed(&self) -> Keyed<&I, &K> {
        Keyed {
            item: self.item,
            stored_key: &self.stored_key,
        }
    }
}

impl<'item, I: ?Sized, K> Keyed<&'item I, &'item K> {
    /// Pair a record with its key at a trusted source or merge boundary.
    pub(crate) fn trusted(item: &'item I, stored_key: &'item K) -> Self {
        Self { item, stored_key }
    }
}

struct Located<'source, I> {
    source: &'source [I],
    source_slot: usize,
}

impl<I> std::ops::Deref for Located<'_, I> {
    type Target = I;

    fn deref(&self) -> &I {
        &self.source[self.source_slot]
    }
}

impl<I> From<Located<'_, I>> for usize {
    fn from(located: Located<'_, I>) -> Self {
        located.source_slot
    }
}

/// Compares protected entries against their bound source and complete order.
pub struct KeyedEntryCompare<'a, I, O> {
    items: &'a [I],
    order: &'a O,
}

impl<'a, I, O: KeyedOrder> KeyedEntryCompare<'a, I, O> {
    pub(crate) fn new(items: &'a [I], order: &'a O) -> Self {
        Self { items, order }
    }

    pub(crate) fn prepare_missing_keys(&self, entries: &mut Vec<Keyed<usize, O::StoredKey>>)
    where
        O: DeriveKey<I>,
    {
        let first_missing_slot = entries.len();
        assert!(first_missing_slot <= self.items.len());
        for source_slot in first_missing_slot..self.items.len() {
            let located = Located {
                source: self.items,
                source_slot,
            };
            let entry = Keyed::new(located, self.order);
            entries.push(entry);
        }
    }

    pub(crate) fn ordered_records<'records>(
        &'records self,
        entries: &'records [Keyed<usize, O::StoredKey>],
    ) -> impl Iterator<Item = Keyed<&'records I, &'records O::StoredKey>> + 'records {
        entries.iter().map(|entry| {
            let item = &self.items[*entry.item()];
            Keyed::trusted(item, entry.stored_key())
        })
    }
}

impl<I, O> crate::compare::Compare<Keyed<usize, O::StoredKey>> for KeyedEntryCompare<'_, I, O>
where
    O: KeyedOrder + crate::compare::Compare<I>,
{
    fn compare(
        &self,
        left: &Keyed<usize, O::StoredKey>,
        right: &Keyed<usize, O::StoredKey>,
    ) -> Ordering {
        let key_order = self
            .order
            .compare_keys(left.stored_key(), right.stored_key());
        key_order.then_with(|| {
            self.order
                .compare(&self.items[*left.item()], &self.items[*right.item()])
        })
    }
}

/// The stored-key part of a complete order.
pub trait KeyedOrder {
    /// Key retained in temporary runs.
    type StoredKey;

    /// Compare two stored keys under this order.
    fn compare_keys(&self, left: &Self::StoredKey, right: &Self::StoredKey) -> Ordering;
}

/// Derive the stored key used by a complete order.
pub trait DeriveKey<I: ?Sized>: KeyedOrder {
    /// Derive the stored key for an item.
    fn derive_key(&self, item: &I) -> Self::StoredKey;

    /// Associate an item with its derived stored key.
    fn derive_keyed<'item>(&self, item: &'item I) -> Keyed<&'item I, Self::StoredKey> {
        Keyed::derive(item, self)
    }
}

/// An upper bound on a key's dynamic memory and its derivation scratch.
pub struct KeyMemory {
    retained_heap_bytes: usize,
    derivation_scratch_bytes: usize,
}

impl KeyMemory {
    /// No additional dynamic memory is required.
    pub const NONE: Self = Self {
        retained_heap_bytes: 0,
        derivation_scratch_bytes: 0,
    };

    /// Bound the heap retained by a derived key.
    #[must_use]
    pub fn retained_heap(bytes: usize) -> Self {
        Self {
            retained_heap_bytes: bytes,
            derivation_scratch_bytes: 0,
        }
    }

    /// Bound additional temporary memory during key derivation.
    #[must_use]
    pub fn with_derivation_scratch(mut self, bytes: usize) -> Self {
        self.derivation_scratch_bytes = bytes;
        self
    }

    pub(crate) fn retained_heap_bytes(&self) -> usize {
        self.retained_heap_bytes
    }

    pub(crate) fn derivation_scratch_bytes(&self) -> usize {
        self.derivation_scratch_bytes
    }
}

/// Estimate dynamic memory needed to derive and retain a key.
pub trait EstimateKeyMemory<I: ?Sized>: KeyedOrder {
    /// Return a conservative allowance for this input.
    fn estimate_key_memory(&self, input: &I) -> KeyMemory;
}

/// Extract a sort key from an item.
///
/// A sort key is the transient value that determines ordering
/// during in-memory chunk sort. It is distinct from the *record
/// key* on [`KeyedCodec`](crate::codec::KeyedCodec), which is a
/// compact on-disk proxy used for merge acceleration.
///
/// The GAT `Key<'a>` allows the key to borrow from the item. For
/// example, a sort key that returns `&'a [u8]` borrows the
/// sequence bytes directly from the record without copying. For
/// owned keys like `f64` or `u64`, the lifetime parameter is
/// simply unused — or use the [`Owned`] adapter to avoid writing
/// the GAT at all.
///
/// `SortKey` is deliberately separated from
/// [`Compare`](crate::compare::Compare) so that the two vary
/// independently: any `SortKey` that produces a given key type
/// can pair with any `Compare` that orders that key type.
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
pub trait SortKey<T: ?Sized> {
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
/// delegating to a key comparator. This adapts the `SortKey` builder path
/// to record comparison; complete-order sorting supplies record comparison directly.
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

impl<T: ?Sized, SK, Cmp> crate::compare::Compare<T> for KeyCompare<SK, Cmp>
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

impl<T: ?Sized, K, F> SortKey<T> for Owned<F>
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
