use std::{
    cell::{Cell, RefCell},
    convert::Infallible,
};

use proptest::{prelude::*, test_runner::FileFailurePersistence};
use spillover::{
    SortedItemsError,
    dedup::{AdjacentDeduplicator, Deduplicated, NoDedup},
    sorter::VisitSortedItems,
};

#[derive(Default)]
struct Observed {
    captures: Cell<usize>,
    installations: Cell<usize>,
    matches: Cell<usize>,
    updates: Cell<usize>,
    storage: RefCell<Vec<(*const u8, usize)>>,
}

struct Bytes<'a> {
    previous: Option<Vec<u8>>,
    observed: &'a Observed,
}

impl<R: AsRef<[u8]> + ?Sized> AdjacentDeduplicator<R> for Bytes<'_> {
    type Representative = Vec<u8>;

    fn matches(&self, record: &R) -> bool {
        self.observed.matches.set(self.observed.matches.get() + 1);
        self.previous.as_deref() == Some(record.as_ref())
    }

    fn capture(&self, record: &R) -> Vec<u8> {
        self.observed.captures.set(self.observed.captures.get() + 1);
        record.as_ref().to_vec()
    }

    fn replace(&mut self, representative: Vec<u8>) {
        self.observed
            .installations
            .set(self.observed.installations.get() + 1);
        self.observed
            .storage
            .borrow_mut()
            .push((representative.as_ptr(), representative.capacity()));
        self.previous = Some(representative);
    }

    fn update(&mut self, record: &R) {
        self.observed.updates.set(self.observed.updates.get() + 1);
        let previous = self.previous.get_or_insert_with(Vec::new);
        previous.clear();
        previous.extend_from_slice(record.as_ref());
        self.observed
            .storage
            .borrow_mut()
            .push((previous.as_ptr(), previous.capacity()));
    }
}

struct Source(std::vec::IntoIter<Option<Result<Vec<u8>, &'static str>>>);

struct CountingSource<'a> {
    source: Source,
    visits: &'a Cell<usize>,
}

impl VisitSortedItems for CountingSource<'_> {
    type Record = [u8];
    type Error = &'static str;
    fn visit_items<F, E>(self, mut sink: F) -> Result<(), SortedItemsError<Self::Error, E>>
    where
        F: FnMut(&[u8]) -> Result<(), E>,
    {
        self.source.visit_items(|record| {
            self.visits.set(self.visits.get() + 1);
            sink(record)
        })
    }
}

impl Iterator for Source {
    type Item = Result<Vec<u8>, &'static str>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().flatten()
    }
}

impl VisitSortedItems for Source {
    type Record = [u8];
    type Error = &'static str;

    fn visit_items<F, E>(self, mut sink: F) -> Result<(), SortedItemsError<Self::Error, E>>
    where
        F: FnMut(&[u8]) -> Result<(), E>,
    {
        for event in self.0 {
            let Some(record) = event else { continue };
            let record = record.map_err(SortedItemsError::Source)?;
            sink(&record).map_err(SortedItemsError::Sink)?;
        }
        Ok(())
    }
}

struct Token(i32);
struct Near {
    previous: Option<Token>,
}

struct BorrowedOnly<'a>(&'a [&'a [u8]]);

impl VisitSortedItems for BorrowedOnly<'_> {
    type Record = [u8];
    type Error = Infallible;

    fn visit_items<F, E>(self, mut sink: F) -> Result<(), SortedItemsError<Infallible, E>>
    where
        F: FnMut(&[u8]) -> Result<(), E>,
    {
        for record in self.0 {
            sink(record).map_err(SortedItemsError::Sink)?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct BorrowedAlgorithm(Option<Vec<u8>>);

impl AdjacentDeduplicator<[u8]> for BorrowedAlgorithm {
    type Representative = Vec<u8>;
    fn matches(&self, record: &[u8]) -> bool {
        self.0.as_deref() == Some(record)
    }
    fn capture(&self, record: &[u8]) -> Vec<u8> {
        record.to_vec()
    }
    fn replace(&mut self, representative: Vec<u8>) {
        self.0 = Some(representative);
    }
}

struct PanicAlgorithm;

impl AdjacentDeduplicator<Vec<u8>> for PanicAlgorithm {
    type Representative = ();
    fn matches(&self, _: &Vec<u8>) -> bool {
        panic!("disabled callback")
    }
    fn capture(&self, _: &Vec<u8>) {
        panic!("disabled callback")
    }
    fn replace(&mut self, (): ()) {
        panic!("disabled callback")
    }
}

struct Trace<'a> {
    events: &'a RefCell<Vec<&'static str>>,
    previous: Option<Vec<u8>>,
}

impl AdjacentDeduplicator<[u8]> for Trace<'_> {
    type Representative = Vec<u8>;
    fn matches(&self, record: &[u8]) -> bool {
        self.events.borrow_mut().push("match");
        self.previous.as_deref() == Some(record)
    }
    fn capture(&self, record: &[u8]) -> Vec<u8> {
        self.events.borrow_mut().push("capture");
        record.to_vec()
    }
    fn replace(&mut self, representative: Vec<u8>) {
        self.events.borrow_mut().push("replace");
        self.previous = Some(representative);
    }
    fn update(&mut self, record: &[u8]) {
        self.events.borrow_mut().push("update");
        self.previous = Some(record.to_vec());
    }
}

impl AdjacentDeduplicator<Vec<u8>> for Trace<'_> {
    type Representative = Vec<u8>;
    fn matches(&self, record: &Vec<u8>) -> bool {
        <Self as AdjacentDeduplicator<[u8]>>::matches(self, record)
    }
    fn capture(&self, record: &Vec<u8>) -> Vec<u8> {
        <Self as AdjacentDeduplicator<[u8]>>::capture(self, record)
    }
    fn replace(&mut self, representative: Vec<u8>) {
        <Self as AdjacentDeduplicator<[u8]>>::replace(self, representative);
    }
    fn update(&mut self, record: &Vec<u8>) {
        <Self as AdjacentDeduplicator<[u8]>>::update(self, record);
    }
}

impl AdjacentDeduplicator<i32> for Near {
    type Representative = Token;
    fn matches(&self, record: &i32) -> bool {
        self.previous
            .as_ref()
            .is_some_and(|previous| (previous.0 - record).abs() <= 1)
    }
    fn capture(&self, record: &i32) -> Token {
        Token(*record)
    }
    fn replace(&mut self, representative: Token) {
        self.previous = Some(representative);
    }
}

#[test]
fn first_install_then_reuse_across_owned_and_borrowed_delivery() {
    let observed = Observed::default();
    let old = vec![1; 32];
    let old_pointer = old.as_ptr();
    let algorithm = Bytes {
        previous: Some(old),
        observed: &observed,
    };
    let events = vec![
        Some(Ok(vec![1; 8])),
        Some(Ok(vec![1; 8])),
        Some(Ok(vec![2; 4])),
    ];
    let mut output = Deduplicated::new(Source(events.into_iter()), Some(algorithm));
    assert_eq!(output.next(), Some(Ok(vec![1; 8])));
    assert_eq!(observed.captures.get(), 1);
    assert_eq!(observed.installations.get(), 1);
    assert_eq!(observed.matches.get(), 0);
    assert_eq!(observed.updates.get(), 0);
    let mut remaining = Vec::new();
    output
        .visit_items(|record| {
            remaining.push(record.to_vec());
            Ok::<_, Infallible>(())
        })
        .expect("borrowed traversal");
    assert_eq!(remaining, [vec![2; 4]]);
    assert_eq!(observed.matches.get(), 2);
    assert_eq!(observed.updates.get(), 1);
    let storage = observed.storage.borrow();
    assert_ne!(storage[0].0, old_pointer);
    assert_eq!(storage[0], storage[1]);
}

#[test]
fn preseed_cannot_reject_first_and_direct_update_can_initialize() {
    let observed = Observed::default();
    let mut algorithm = Bytes {
        previous: None,
        observed: &observed,
    };
    algorithm.update(&[7_u8][..]);
    assert_eq!(algorithm.previous.as_deref(), Some(&[7][..]));
    let source = [Ok::<_, Infallible>(vec![7])].into_iter();
    let records = Deduplicated::new(source, Some(algorithm)).collect::<Result<Vec<_>, _>>();
    assert_eq!(records.expect("infallible"), [vec![7]]);
    assert_eq!(observed.captures.get(), 1);
    assert_eq!(observed.matches.get(), 0);
}

#[test]
fn nonclone_nondefault_representative_and_nontransitive_matching() {
    let source = [Ok::<_, Infallible>(1), Ok(2), Ok(3)].into_iter();
    let algorithm = Near { previous: None };
    let output = Deduplicated::new(source, Some(algorithm));
    let records = output.collect::<Result<Vec<_>, _>>().expect("infallible");
    assert_eq!(records, [1, 3]);
}

#[test]
fn disabled_is_passthrough_including_errors_and_intermittent_none() {
    let events = vec![
        Some(Err("read")),
        Some(Ok(vec![1])),
        None,
        Some(Ok(vec![2])),
    ];
    let mut output: Deduplicated<_, NoDedup> = Deduplicated::new(Source(events.into_iter()), None);
    assert_eq!(output.next(), Some(Err("read")));
    assert_eq!(output.next(), Some(Ok(vec![1])));
    assert_eq!(output.next(), None);
    assert_eq!(output.next(), Some(Ok(vec![2])));
}

#[test]
fn enabled_none_retains_history_but_error_stops_all_delivery() {
    let observed = Observed::default();
    let algorithm = Bytes {
        previous: None,
        observed: &observed,
    };
    let events = vec![
        Some(Ok(vec![1])),
        None,
        Some(Ok(vec![1])),
        Some(Err("read")),
        Some(Ok(vec![2])),
    ];
    let mut output = Deduplicated::new(Source(events.into_iter()), Some(algorithm));
    assert_eq!(output.next(), Some(Ok(vec![1])));
    assert_eq!(output.next(), None);
    assert_eq!(output.next(), Some(Err("read")));
    assert_eq!(output.next(), None);
    output
        .visit_items(|_| -> Result<(), Infallible> { panic!("delivery after source error") })
        .expect("fused traversal");
}

#[test]
fn borrowed_errors_keep_source_sink_classification_and_stop_immediately() {
    let observed = Observed::default();
    let algorithm = Bytes {
        previous: None,
        observed: &observed,
    };
    let events = vec![Some(Ok(vec![1])), Some(Err("read"))];
    let output = Deduplicated::new(Source(events.into_iter()), Some(algorithm));
    let result = output.visit_items(|_| Err::<(), _>(42_u16));
    assert!(matches!(result, Err(SortedItemsError::Sink(42))));
    let algorithm = Bytes {
        previous: None,
        observed: &observed,
    };
    let events = vec![Some(Ok(vec![1])), Some(Err("read"))];
    let output = Deduplicated::new(Source(events.into_iter()), Some(algorithm));
    let result = output.visit_items(|_| Ok::<_, u16>(()));
    assert!(matches!(result, Err(SortedItemsError::Source("read"))));
}

#[test]
fn borrowed_only_unsized_record_requires_no_owned_algorithm() {
    let records: &[&[u8]] = &[b"a", b"a", b"b"];
    let output = Deduplicated::new(BorrowedOnly(records), Some(BorrowedAlgorithm::default()));
    let mut visited = Vec::new();
    output
        .visit_items(|record| {
            visited.push(record.to_vec());
            Ok::<_, Infallible>(())
        })
        .expect("borrowed traversal");
    assert_eq!(visited, [b"a".to_vec(), b"b".to_vec()]);
}

#[test]
fn runtime_disabled_actual_algorithm_forwards_every_event() {
    let events = vec![
        Some(Ok(vec![1])),
        Some(Ok(vec![1])),
        Some(Err("read")),
        None,
        Some(Ok(vec![2])),
    ];
    let mut output = Deduplicated::new(Source(events.into_iter()), None::<PanicAlgorithm>);
    assert_eq!(output.next(), Some(Ok(vec![1])));
    assert_eq!(output.next(), Some(Ok(vec![1])));
    assert_eq!(output.next(), Some(Err("read")));
    assert_eq!(output.next(), None);
    assert_eq!(output.next(), Some(Ok(vec![2])));
}

#[test]
fn borrowed_first_initialization_is_ordered_and_empty_or_error_never_calls_algorithm() {
    let events = RefCell::new(Vec::new());
    let algorithm = Trace {
        events: &events,
        previous: Some(vec![9]),
    };
    let source = [Ok::<_, Infallible>(vec![1]), Ok(vec![1]), Ok(vec![2])].into_iter();
    let result = Deduplicated::new(source, Some(algorithm)).collect::<Result<Vec<_>, _>>();
    assert_eq!(result.expect("owned"), [vec![1], vec![2]]);
    assert_eq!(
        *events.borrow(),
        ["capture", "replace", "match", "match", "update"]
    );
    events.borrow_mut().clear();
    let algorithm = Trace {
        events: &events,
        previous: Some(vec![9]),
    };
    let records: &[&[u8]] = &[b"a", b"a", b"b"];
    let output = Deduplicated::new(BorrowedOnly(records), Some(algorithm));
    output
        .visit_items(|_| Ok::<_, Infallible>(()))
        .expect("visit");
    assert_eq!(
        *events.borrow(),
        ["capture", "replace", "match", "match", "update"]
    );

    events.borrow_mut().clear();
    let algorithm = Trace {
        events: &events,
        previous: None,
    };
    let output = Deduplicated::new(BorrowedOnly(&[]), Some(algorithm));
    output
        .visit_items(|_| Ok::<_, Infallible>(()))
        .expect("empty");
    assert!(events.borrow().is_empty());

    let algorithm = Trace {
        events: &events,
        previous: None,
    };
    let source = Source(vec![Some(Err("read"))].into_iter());
    let output = Deduplicated::new(source, Some(algorithm));
    let result = output.visit_items(|_| Ok::<_, Infallible>(()));
    assert!(matches!(result, Err(SortedItemsError::Source("read"))));
    assert!(events.borrow().is_empty());
}

#[test]
fn sink_error_stops_source_visits_immediately() {
    let observed = Observed::default();
    let algorithm = Bytes {
        previous: None,
        observed: &observed,
    };
    let events = vec![Some(Ok(vec![1])), Some(Ok(vec![2])), Some(Ok(vec![3]))];
    let visits = Cell::new(0);
    let source = CountingSource {
        source: Source(events.into_iter()),
        visits: &visits,
    };
    let output = Deduplicated::new(source, Some(algorithm));
    let result = output.visit_items(|_| Err::<(), _>("stop"));
    assert!(matches!(result, Err(SortedItemsError::Sink("stop"))));
    assert_eq!(visits.get(), 1);
    assert_eq!(observed.matches.get(), 0);
}

mod proptests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            max_shrink_iters: 4096,
            failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
                env!("CARGO_MANIFEST_DIR"), "/proptest-regressions/dedup_controls.txt"
            )))),
            ..ProptestConfig::default()
        })]

        #[test]
        fn owned_prefix_and_borrowed_remainder_share_the_last_accepted_record(
            prefix in proptest::collection::vec(0u8..4, 1..12),
            seed in 0u8..4,
        ) {
            let accepted_prefix = {
                let mut accepted = Vec::new();
                for &value in &prefix {
                    if accepted.last() != Some(&value) {
                        accepted.push(value);
                    }
                }
                accepted
            };
            let owned_count = accepted_prefix.len();
            let last = *accepted_prefix.last()
                .expect("the nonempty input prefix must retain its first record before traversal switches");
            let different = (last + 1) % 4;
            let input = {
                let mut records = prefix;
                records.extend([last, different, last]);
                records
            };
            let expected = {
                let mut records = accepted_prefix;
                records.extend([different, last]);
                records
            };

            let observed = Observed::default();
            let algorithm = Bytes {
                previous: Some(vec![seed]),
                observed: &observed,
            };
            let events = input.iter()
                .map(|&value| Some(Ok(vec![value])))
                .collect::<Vec<_>>();
            let source = Source(events.into_iter());
            let mut output = Deduplicated::new(source, Some(algorithm));
            let mut actual = Vec::new();
            for index in 0..owned_count {
                let event = output.next()
                    .unwrap_or_else(|| panic!("owned prefix count {owned_count}, output index {index}: unexpected end"));
                let record = event
                    .unwrap_or_else(|error| panic!("owned prefix count {owned_count}, output index {index}: source error {error}"));
                actual.push(record[0]);
            }

            let result = output.visit_items(|record| {
                actual.push(record[0]);
                Ok::<_, Infallible>(())
            });
            result.unwrap_or_else(|error| {
                panic!("borrowed remainder after owned prefix count {owned_count}: {error:?}")
            });

            prop_assert_eq!(actual, expected, "owned prefix count {}, followed by borrowed remainder", owned_count);
            prop_assert_eq!(observed.captures.get(), 1, "owned prefix count {}", owned_count);
            prop_assert_eq!(observed.installations.get(), 1, "owned prefix count {}", owned_count);
            prop_assert_eq!(observed.matches.get(), input.len() - 1, "owned prefix count {}", owned_count);
            prop_assert_eq!(observed.updates.get(), owned_count + 1, "owned prefix count {}", owned_count);
        }

        #[test]
        fn nontransitive_matching_compares_with_the_last_accepted_record(
            start in -20i32..20,
            tail in proptest::collection::vec(-20i32..20, 0..12),
        ) {
            let input = {
                let mut records = vec![start, start + 1, start + 2];
                records.extend(tail);
                records
            };
            let expected = {
                let mut accepted = Vec::new();
                for &record in &input {
                    if accepted.last().is_none_or(|previous: &i32| (previous - record).abs() > 1) {
                        accepted.push(record);
                    }
                }
                accepted
            };

            let source = input.into_iter().map(Ok::<_, Infallible>);
            let algorithm = Near {
                previous: Some(Token(start)),
            };
            let output = Deduplicated::new(source, Some(algorithm));
            let actual = output.collect::<Result<Vec<_>, _>>()
                .expect("owned nontransitive history has an infallible source");

            prop_assert_eq!(actual, expected, "owned nontransitive history starting at {}", start);
        }

        #[test]
        fn clean_end_retains_history_and_source_error_fuses_only_enabled_output(
            first in 0u8..4,
            following in 0u8..4,
        ) {
            let events = vec![
                Some(Ok(vec![first])),
                None,
                Some(Ok(vec![first])),
                Some(Ok(vec![following])),
                Some(Err("source failure")),
                Some(Ok(vec![first])),
            ];
            let observed = Observed::default();
            let algorithm = Bytes {
                previous: Some(vec![first]),
                observed: &observed,
            };
            let mut enabled = Deduplicated::new(Source(events.clone().into_iter()), Some(algorithm));

            prop_assert_eq!(enabled.next(), Some(Ok(vec![first])), "enabled event 0");
            prop_assert_eq!(enabled.next(), None, "enabled clean end at event 1");
            let next = enabled.next();
            if following == first {
                prop_assert_eq!(next, Some(Err("source failure")), "enabled error after duplicate events 2 and 3");
            } else {
                prop_assert_eq!(next, Some(Ok(vec![following])), "enabled record at event 3");
                prop_assert_eq!(enabled.next(), Some(Err("source failure")), "enabled error at event 4");
            }
            prop_assert_eq!(enabled.next(), None, "enabled event after source error");

            let mut disabled: Deduplicated<_, NoDedup> = Deduplicated::new(Source(events.clone().into_iter()), None);
            for (index, event) in events.into_iter().enumerate() {
                prop_assert_eq!(disabled.next(), event, "disabled forwarding at event {}", index);
            }
        }
    }
}
