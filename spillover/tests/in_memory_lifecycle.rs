use std::{
    cell::{Cell, RefCell},
    cmp::Ordering,
    convert::Infallible,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
};

use spillover::{
    SortedItemsError,
    chunk::{ChunkSorter, Sequential},
    compare::{Compare, Natural},
    dedup::AdjacentDeduplicator,
    sorter::{ChunkBuffer, in_memory::Builder},
};

struct ObserveAllocation<'a> {
    pointer: &'a Cell<*const u64>,
    calls: &'a Cell<usize>,
}

impl<C: Compare<u64>> ChunkSorter<u64, C> for ObserveAllocation<'_> {
    fn sort(&self, values: &mut [u64], order: &C) {
        self.pointer.set(values.as_ptr());
        self.calls.set(self.calls.get() + 1);
        Sequential.sort(values, order);
    }
}

#[test]
fn build_from_keeps_element_allocation_and_spare_capacity_for_pushes() {
    let mut input = Vec::with_capacity(32);
    input.extend([9_u64, 1]);
    let original_pointer = input.as_ptr();
    let original_capacity = input.capacity();
    let observed_pointer = Cell::new(std::ptr::null());
    let calls = Cell::new(0);
    let algorithm = ObserveAllocation {
        pointer: &observed_pointer,
        calls: &calls,
    };
    let mut sorter = Builder::new()
        .sort_by(Natural)
        .chunk_sort(algorithm)
        .build_from(input);
    // Fill all transferred spare slots without requiring growth.
    for _ in 2..original_capacity {
        sorter.push(5);
    }
    let output = sorter.finish();
    assert_eq!(calls.get(), 1);
    assert_eq!(observed_pointer.get(), original_pointer);
    let values = output
        .collect::<Result<Vec<_>, Infallible>>()
        .expect("infallible output");
    assert_eq!(values.len(), original_capacity);
    assert_eq!(values.first(), Some(&1));
    assert_eq!(values.last(), Some(&9));
    assert!(values[1..values.len() - 1].iter().all(|value| *value == 5));
}

struct Tracked {
    id: usize,
    drops: Rc<RefCell<[usize; 4]>>,
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.drops.borrow_mut()[self.id] += 1;
    }
}

struct ById;

impl Compare<Tracked> for ById {
    fn compare(&self, left: &Tracked, right: &Tracked) -> Ordering {
        left.id.cmp(&right.id)
    }
}

fn tracked_records(drops: &Rc<RefCell<[usize; 4]>>) -> Vec<Tracked> {
    [3, 1, 0, 2]
        .into_iter()
        .map(|id| Tracked {
            id,
            drops: Rc::clone(drops),
        })
        .collect()
}

#[test]
fn dropping_unfinished_sorter_drops_each_original_once() {
    let drops = Rc::new(RefCell::new([0; 4]));
    let input = tracked_records(&drops);
    let sorter = Builder::new().sort_by(ById).build_from(input);
    assert_eq!(*drops.borrow(), [0; 4]);
    drop(sorter);
    assert_eq!(*drops.borrow(), [1; 4]);
}

#[test]
fn dropping_partial_output_preserves_yielded_record_and_drops_the_rest() {
    let drops = Rc::new(RefCell::new([0; 4]));
    let input = tracked_records(&drops);
    let mut output = Builder::new().sort_by(ById).build_from(input).finish();
    let first = output
        .next()
        .expect("first record")
        .expect("infallible output");
    assert_eq!(first.id, 0);
    drop(output);
    assert_eq!(*drops.borrow(), [0, 1, 1, 1]);
    drop(first);
    assert_eq!(*drops.borrow(), [1; 4]);
}

#[test]
fn sink_error_drops_current_and_unvisited_records_once() {
    let drops = Rc::new(RefCell::new([0; 4]));
    let input = tracked_records(&drops);
    let output = Builder::new().sort_by(ById).build_from(input).finish();
    let mut calls = 0;
    let result = output.items().try_for_each(|record| {
        calls += 1;
        assert_eq!(record.id, 0);
        Err::<(), _>("stop")
    });
    assert!(matches!(result, Err(SortedItemsError::Sink("stop"))));
    assert_eq!(calls, 1);
    assert_eq!(*drops.borrow(), [1; 4]);
}

#[test]
fn sink_panic_drops_current_and_unvisited_records_once() {
    let drops = Rc::new(RefCell::new([0; 4]));
    let input = tracked_records(&drops);
    let output = Builder::new().sort_by(ById).build_from(input).finish();
    let calls = Cell::new(0);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = output
            .items()
            .try_for_each(|record| -> Result<(), Infallible> {
                calls.set(calls.get() + 1);
                assert_eq!(record.id, 0);
                panic!("injected sink panic");
            });
    }));
    assert!(result.is_err());
    assert_eq!(calls.get(), 1);
    assert_eq!(*drops.borrow(), [1; 4]);
}

struct PanicComparison;

impl Compare<Tracked> for PanicComparison {
    fn compare(&self, _: &Tracked, _: &Tracked) -> Ordering {
        panic!("injected comparison panic");
    }
}

#[test]
fn comparison_panic_during_sort_vec_keeps_all_originals_and_allocation() {
    let drops = Rc::new(RefCell::new([0; 4]));
    let mut input = tracked_records(&drops);
    let pointer = input.as_ptr();
    let capacity = input.capacity();
    let result = catch_unwind(AssertUnwindSafe(|| {
        Builder::new().sort_by(PanicComparison).sort_vec(&mut input);
    }));
    assert!(result.is_err());
    assert_eq!(input.len(), 4);
    assert_eq!(input.as_ptr(), pointer);
    assert_eq!(input.capacity(), capacity);
    assert_eq!(*drops.borrow(), [0; 4]);
    // A failed sort promises validity, not rollback or any particular ordering.
    drop(input);
    assert_eq!(*drops.borrow(), [1; 4]);
}

struct RejectThenPanic {
    matches: Rc<Cell<usize>>,
}

impl AdjacentDeduplicator<Tracked> for RejectThenPanic {
    type Representative = ();

    fn capture(&self, _: &Tracked) {}

    fn replace(&mut self, (): ()) {}

    fn matches(&self, _: &Tracked) -> bool {
        let calls = self.matches.get() + 1;
        self.matches.set(calls);
        assert_eq!(calls, 1, "injected dedup panic after first rejection");
        true
    }
}

#[test]
fn dedup_panic_after_rejection_preserves_allocation_and_drops_each_original_once() {
    let drops = Rc::new(RefCell::new([0; 4]));
    let mut input = tracked_records(&drops);
    let pointer = input.as_ptr();
    let capacity = input.capacity();
    let matches = Rc::new(Cell::new(0));
    let algorithm = RejectThenPanic {
        matches: Rc::clone(&matches),
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        Builder::new()
            .sort_by(ById)
            .dedup(algorithm)
            .sort_vec(&mut input);
    }));
    assert!(result.is_err());
    assert_eq!(matches.get(), 2);
    assert_eq!(input.as_ptr(), pointer);
    assert_eq!(input.capacity(), capacity);
    assert_eq!(*drops.borrow(), [0, 1, 0, 0]);
    drop(input);
    assert_eq!(*drops.borrow(), [1; 4]);
}

struct TextStorage(Vec<String>);

struct TextItems<'a>(Vec<&'a str>);

impl<'a> AsMut<[&'a str]> for TextItems<'a> {
    fn as_mut(&mut self) -> &mut [&'a str] {
        &mut self.0
    }
}

impl<'a> IntoIterator for TextItems<'a> {
    type Item = &'a str;
    type IntoIter = std::vec::IntoIter<&'a str>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> ChunkBuffer for &'a mut TextStorage {
    type Item = &'a str;
    type Items = TextItems<'a>;

    fn items(self) -> Self::Items {
        let views = self.0.iter().map(String::as_str).collect();
        TextItems(views)
    }

    fn clear(self) {
        self.0.clear();
    }
}

impl Extend<String> for TextStorage {
    fn extend<I: IntoIterator<Item = String>>(&mut self, input: I) {
        self.0.extend(input);
    }
}

// Deliberately no codec, summary, memory estimate, Clone, Send, or Sync bounds.
struct LocalSort<'a>(&'a Cell<usize>);

impl<T, C: Compare<T>> ChunkSorter<T, C> for LocalSort<'_> {
    fn sort(&self, items: &mut [T], order: &C) {
        self.0.set(self.0.get() + 1);
        Sequential.sort(items, order);
    }
}

#[test]
fn custom_buffer_container_and_algorithm_need_only_memory_sort_capabilities() {
    let mut storage = TextStorage(vec![String::from("stale")]);
    let calls = Cell::new(0);
    let mut sorter = Builder::new()
        .sort_by(Natural)
        .chunk_sort(LocalSort(&calls))
        .build_in(&mut storage);
    sorter.push(String::from("zebra"));
    sorter.push(String::from("apple"));
    let mut output = sorter.finish();
    assert_eq!(calls.get(), 1);
    assert_eq!(output.next(), Some(Ok("apple")));
    let mut remaining = Vec::new();
    output
        .items()
        .try_for_each(|record| {
            remaining.push(*record);
            Ok::<(), Infallible>(())
        })
        .expect("infallible visitor");
    assert_eq!(remaining, ["zebra"]);
}
