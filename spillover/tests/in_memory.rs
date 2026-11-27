use std::convert::Infallible;

use spillover::{
    SortedItemsError,
    compare::Natural,
    dedup::AdjacentDedup,
    sorter::{ChunkBuffer, in_memory::Builder},
};

#[derive(Default)]
struct Arena(Vec<String>);

impl<'a> ChunkBuffer for &'a mut Arena {
    type Item = &'a str;
    type Items = Vec<&'a str>;

    fn items(self) -> Self::Items {
        self.0.iter().map(String::as_str).collect()
    }

    fn clear(self) {
        self.0.clear();
    }
}

impl Extend<String> for Arena {
    fn extend<I: IntoIterator<Item = String>>(&mut self, iter: I) {
        self.0.extend(iter);
    }
}

#[test]
fn sorts_slices_and_compacts_vectors_in_place() {
    let mut slice = [3, 1, 2, 1];
    Builder::new().sort_by(Natural).sort_slice(&mut slice);
    assert_eq!(slice, [1, 1, 2, 3]);
    let mut values = Vec::with_capacity(32);
    values.extend([3, 1, 2, 1]);
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    let dedup = AdjacentDedup::new(|a: &i32, b: &i32| a == b);
    Builder::new()
        .sort_by(Natural)
        .dedup(dedup)
        .sort_vec(&mut values);
    assert_eq!(values, [1, 2, 3]);
    assert_eq!(values.as_ptr(), pointer);
    assert_eq!(values.capacity(), capacity);
}

#[test]
fn owned_batches_move_values_and_support_borrowed_inputs() {
    let owned = String::from("b");
    let pointer = owned.as_ptr();
    let input = vec![owned];
    let mut sorter = Builder::new().sort_by(Natural).build_from(input);
    sorter.push(String::from("a"));
    let result = sorter
        .finish()
        .collect::<Result<Vec<_>, Infallible>>()
        .unwrap();
    assert_eq!(result, ["a", "b"]);
    assert_eq!(result[1].as_ptr(), pointer);

    let text = String::from("borrowed");
    let mut sorter = Builder::new().sort_by(Natural).build();
    sorter.push(text.as_str());
    assert_eq!(sorter.finish().next(), Some(Ok("borrowed")));
}

#[test]
fn iteration_and_visitation_share_dedup_history() {
    let dedup = AdjacentDedup::new(|a: &i32, b: &i32| a == b);
    let sorter = Builder::new()
        .sort_by(Natural)
        .dedup(dedup)
        .build_from(vec![2, 1, 1, 2, 3]);
    let mut output = sorter.finish();
    assert_eq!(output.next(), Some(Ok(1)));
    let mut visited = Vec::new();
    output
        .items()
        .try_for_each(|value| {
            visited.push(*value);
            Ok::<_, Infallible>(())
        })
        .expect("infallible visitor");
    assert_eq!(visited, [2, 3]);
}

#[test]
fn sink_failure_is_not_a_source_failure() {
    let sorter = Builder::new().sort_by(Natural).build_from(vec![2, 1]);
    let result = sorter
        .finish()
        .items()
        .try_for_each(|_| Err::<(), _>("stop"));
    assert!(matches!(result, Err(SortedItemsError::Sink("stop"))));
}

#[test]
fn arena_reuses_capacity_and_clears_previous_contents() {
    let mut arena = Arena::default();
    arena.0.reserve(20);
    let capacity = arena.0.capacity();
    let mut sorter = Builder::new().sort_by(Natural).build_in(&mut arena);
    sorter.push(String::from("b"));
    sorter.push(String::from("a"));
    let values = sorter
        .finish()
        .collect::<Result<Vec<_>, Infallible>>()
        .expect("infallible visitor");
    assert_eq!(values, ["a", "b"]);
    drop(values);
    let sorter = Builder::new().sort_by(Natural).build_in(&mut arena);
    assert_eq!(sorter.finish().next(), None);
    assert_eq!(arena.0.capacity(), capacity);
}

#[test]
fn arena_early_drop_allows_reuse() {
    let mut arena = Arena::default();
    let mut sorter = Builder::new().sort_by(Natural).build_in(&mut arena);
    sorter.push(String::from("old"));
    drop(sorter.finish());
    let mut sorter = Builder::new().sort_by(Natural).build_in(&mut arena);
    sorter.push(String::from("new"));
    assert_eq!(sorter.finish().next(), Some(Ok("new")));
}

#[test]
fn arena_dedup_and_visitation_share_history() {
    let mut arena = Arena::default();
    let dedup = AdjacentDedup::new(|a: &&str, b: &&str| a == b);
    let mut sorter = Builder::new()
        .sort_by(Natural)
        .dedup(dedup)
        .build_in(&mut arena);
    for value in ["b", "a", "a", "b", "c"] {
        sorter.push(value.to_owned());
    }
    let mut output = sorter.finish();
    assert_eq!(output.next(), Some(Ok("a")));
    let mut seen = Vec::new();
    output
        .items()
        .try_for_each(|value| {
            seen.push(*value);
            Ok::<_, Infallible>(())
        })
        .expect("infallible visitor");
    assert_eq!(seen, ["b", "c"]);
}
