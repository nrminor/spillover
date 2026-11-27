use std::{
    cell::Cell,
    convert::Infallible,
    panic::{AssertUnwindSafe, catch_unwind},
};

use spillover::{chunk::ChunkSorter, compare::Compare};
use spillover_bio::{
    record::{SeqRecord, SeqRecordArena, SeqRecordView},
    sort::in_memory::Builder,
};

fn assert_replacement_batch(arena: &mut SeqRecordArena, capacity: usize) {
    let mut sorter = Builder::new().sort_by_name().arena(arena).build();
    sorter.push(&SeqRecord::new("new-z", "GG", "!!"));
    sorter.push(&SeqRecord::new("new-a", "C", "?"));
    let mut records = Vec::new();
    sorter
        .finish()
        .items()
        .try_for_each(|view| {
            records.push((
                view.name().to_vec(),
                view.sequence().to_vec(),
                view.quality().to_vec(),
            ));
            Ok::<(), Infallible>(())
        })
        .expect("infallible visitor");
    assert_eq!(
        records,
        [
            (b"new-a".to_vec(), b"C".to_vec(), b"?".to_vec()),
            (b"new-z".to_vec(), b"GG".to_vec(), b"!!".to_vec()),
        ]
    );
    assert_eq!(arena.capacity(), capacity);
}

#[test]
fn dropping_unfinished_arena_sorter_allows_replacement_without_stale_records() {
    let mut arena = SeqRecordArena::with_capacity(1024);
    let capacity = arena.capacity();
    {
        let mut sorter = Builder::new().sort_by_name().arena(&mut arena).build();
        sorter.push(&SeqRecord::new("old-long-name", "AAAA", "!!!!"));
    }
    assert_replacement_batch(&mut arena, capacity);
}

#[test]
fn dropping_partial_arena_output_allows_replacement_without_stale_records() {
    let mut arena = SeqRecordArena::with_capacity(1024);
    let capacity = arena.capacity();
    let mut sorter = Builder::new().sort_by_name().arena(&mut arena).build();
    sorter.push(&SeqRecord::new("old-a", "AAAA", "!!!!"));
    sorter.push(&SeqRecord::new("old-b", "TT", "!!"));
    let mut output = sorter.finish();
    let first = output
        .next()
        .expect("first record")
        .expect("infallible output");
    assert_eq!(first.name(), b"old-a");
    drop(output);
    assert_replacement_batch(&mut arena, capacity);
}

struct PanicSort;

impl<T, C: Compare<T>> ChunkSorter<T, C> for PanicSort {
    fn sort(&self, _: &mut [T], _: &C) {
        panic!("injected arena sort panic");
    }
}

#[test]
fn arena_sort_panic_allows_replacement_and_retains_byte_capacity() {
    let mut arena = SeqRecordArena::with_capacity(1024);
    let capacity = arena.capacity();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut sorter = Builder::new()
            .sort_by_name()
            .chunk_sort(PanicSort)
            .arena(&mut arena)
            .build();
        sorter.push(&SeqRecord::new("old-long-name", "AAAA", "!!!!"));
        drop(sorter.finish());
    }));
    assert!(result.is_err());
    assert_replacement_batch(&mut arena, capacity);
}

#[test]
fn arena_visitor_panic_allows_replacement_and_retains_byte_capacity() {
    let mut arena = SeqRecordArena::with_capacity(1024);
    let capacity = arena.capacity();
    let calls = Cell::new(0);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut sorter = Builder::new().sort_by_name().arena(&mut arena).build();
        sorter.push(&SeqRecord::new("old-a", "AAAA", "!!!!"));
        sorter.push(&SeqRecord::new("old-b", "TT", "!!"));
        let _ = sorter
            .finish()
            .items()
            .try_for_each(|_| -> Result<(), Infallible> {
                calls.set(calls.get() + 1);
                panic!("injected arena visitor panic");
            });
    }));
    assert!(result.is_err());
    assert_eq!(calls.get(), 1);
    assert_replacement_batch(&mut arena, capacity);
}

#[test]
fn copied_visitor_view_remains_readable_after_output_is_consumed() {
    let mut arena = SeqRecordArena::with_capacity(1024);
    let capacity = arena.capacity();
    let mut sorter = Builder::new().sort_by_name().arena(&mut arena).build();
    sorter.push(&SeqRecord::new("old", "ACGT", "!?#$"));
    let mut escaped: Option<SeqRecordView<'_>> = None;
    sorter
        .finish()
        .items()
        .try_for_each(|view| {
            escaped = Some(*view);
            Ok::<(), Infallible>(())
        })
        .expect("infallible visitor");
    let escaped = escaped.expect("visited record");
    assert_eq!(escaped.name(), b"old");
    assert_eq!(escaped.sequence(), b"ACGT");
    assert_eq!(escaped.quality(), b"!?#$");
    // Rebuilding is legal only once the escaped view's final use has ended.
    assert_replacement_batch(&mut arena, capacity);
}
