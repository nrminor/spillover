use std::{
    cell::{Cell, RefCell},
    convert::Infallible,
};

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
