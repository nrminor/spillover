use std::cmp::Ordering;

use proptest::prelude::*;
use spillover::compare::Compare;
use spillover_bio::{
    record::{SeqRecord, SeqRecordParts, SeqRecordView},
    sort::{
        ILLUMINA_ORDER, LengthOrder, NameOrder, RecordKeyer, Reverse, SequenceOrder, SortOrder,
    },
};

fn rec(sequence: &[u8], quality: &[u8]) -> SeqRecord {
    SeqRecord::new(b"record", sequence, quality)
}

fn named_rec(name: &[u8], sequence: &[u8], quality: &[u8]) -> SeqRecord {
    SeqRecord::new(name, sequence, quality)
}

fn assert_record_order<O, R>(order: O, left: &R, right: &R, expected: Ordering)
where
    O: SortOrder,
    R: SeqRecordParts + ?Sized,
    O::Comparator: Compare<R>,
{
    assert_eq!(order.comparator().compare(left, right), expected);
}

fn assert_record_comparator_is_pairwise_consistent<O, R>(order: O, records: &[R])
where
    O: SortOrder,
    R: SeqRecordParts,
    O::Comparator: Compare<R>,
{
    let cmp = order.comparator();

    for left in records {
        assert_eq!(cmp.compare(left, left), Ordering::Equal);

        for right in records {
            assert_eq!(cmp.compare(left, right), cmp.compare(right, left).reverse(),);
        }
    }
}

fn assert_key_comparator_is_pairwise_consistent<O, R>(order: O, records: &[R])
where
    O: SortOrder + RecordKeyer,
    R: SeqRecordParts,
    O::KeyComparator: Compare<O::RecordKey>,
{
    let cmp = order.key_comparator();

    for left in records {
        let left_key = order.record_key(left);
        assert_eq!(cmp.compare(&left_key, &left_key), Ordering::Equal);

        for right in records {
            let right_key = order.record_key(right);
            assert_eq!(
                cmp.compare(&left_key, &right_key),
                cmp.compare(&right_key, &left_key).reverse(),
            );
        }
    }
}

fn assert_key_and_record_order_are_compatible<O, R>(order: O, records: &[R])
where
    O: SortOrder + RecordKeyer,
    R: SeqRecordParts,
    O::Comparator: Compare<R>,
    O::KeyComparator: Compare<O::RecordKey>,
{
    let key_cmp = order.key_comparator();
    let record_cmp = order.comparator();

    for left in records {
        for right in records {
            let left_key = order.record_key(left);
            let right_key = order.record_key(right);
            let key_order = key_cmp.compare(&left_key, &right_key);
            let record_order = record_cmp.compare(left, right);

            match key_order {
                Ordering::Less => assert_ne!(record_order, Ordering::Greater),
                Ordering::Greater => assert_ne!(record_order, Ordering::Less),
                Ordering::Equal => {}
            }
        }
    }
}

fn assert_record_comparator_is_consistent<O, R>(order: O, records: &[R])
where
    O: SortOrder,
    R: SeqRecordParts,
    O::Comparator: Compare<R>,
{
    let cmp = order.comparator();

    assert_comparator_is_consistent(&cmp, records);
}

fn assert_key_comparator_is_consistent<O>(order: O, keys: &[O::RecordKey])
where
    O: SortOrder + RecordKeyer,
    O::KeyComparator: Compare<O::RecordKey>,
{
    let cmp = order.key_comparator();

    assert_comparator_is_consistent(&cmp, keys);
}

fn assert_comparator_is_consistent<C, T>(cmp: &C, values: &[T])
where
    C: Compare<T>,
{
    for a in values {
        assert_eq!(cmp.compare(a, a), Ordering::Equal);
    }

    for a in values {
        for b in values {
            assert_eq!(cmp.compare(a, b), cmp.compare(b, a).reverse());
        }
    }

    for a in values {
        for b in values {
            for c in values {
                if cmp.compare(a, b) != Ordering::Greater && cmp.compare(b, c) != Ordering::Greater
                {
                    assert_ne!(cmp.compare(a, c), Ordering::Greater);
                }
            }
        }
    }
}

#[test]
fn ordinary_bases_follow_packed_primary_order() {
    let order = SequenceOrder::<1>;

    assert_record_order(order, &rec(b"A", b"!"), &rec(b"C", b"!"), Ordering::Less);
    assert_record_order(order, &rec(b"C", b"!"), &rec(b"G", b"!"), Ordering::Less);
    assert_record_order(order, &rec(b"G", b"!"), &rec(b"T", b"!"), Ordering::Less);
}

#[test]
fn ambiguous_bases_follow_packed_primary_order() {
    let order = SequenceOrder::<1>;
    assert_record_order(order, &rec(b"N", b"!"), &rec(b"C", b"!"), Ordering::Less);
}

#[test]
fn lowercase_bases_follow_packed_primary_order() {
    let order = SequenceOrder::<1>;
    assert_record_order(order, &rec(b"a", b"!"), &rec(b"C", b"!"), Ordering::Less);
}

#[test]
fn ambiguous_packed_ties_refine_by_raw_sequence() {
    let order = SequenceOrder::<1>;
    assert_record_order(order, &rec(b"A", b"!"), &rec(b"N", b"!"), Ordering::Less);
}

#[test]
fn zero_padded_packed_ties_refine_by_raw_sequence() {
    let order = SequenceOrder::<1>;
    assert_record_order(order, &rec(b"A", b"!"), &rec(b"AA", b"!!"), Ordering::Less);
}

#[test]
fn identical_sequences_refine_by_quality() {
    let order = SequenceOrder::<1>;
    assert_record_order(
        order,
        &rec(b"ACGT", b"!!!!"),
        &rec(b"ACGT", b"IIII"),
        Ordering::Less,
    );
}

#[test]
fn empty_sequences_are_ordered() {
    let order = SequenceOrder::<1>;
    assert_record_order(order, &rec(b"", b""), &rec(b"C", b"!"), Ordering::Less);
}

fn sequence_order_examples() -> Vec<SeqRecord> {
    vec![
        rec(b"", b""),
        rec(b"A", b"!"),
        rec(b"C", b"!"),
        rec(b"G", b"!"),
        rec(b"T", b"!"),
        rec(b"a", b"!"),
        rec(b"N", b"!"),
        rec(b"-", b"!"),
        rec(b"AA", b"!!"),
        rec(b"AAAA", b"!!!!"),
        rec(b"AAAAA", b"!!!!!"),
        rec(b"ACGT", b"!!!!"),
        rec(b"ACGT", b"IIII"),
    ]
}

fn comparator_consistency_examples() -> Vec<SeqRecord> {
    vec![
        named_rec(b"read_000", b"", b""),
        named_rec(b"read_001", b"A", b"!"),
        named_rec(b"read_002", b"C", b"!"),
        named_rec(b"read_003", b"G", b"!"),
        named_rec(b"read_004", b"T", b"!"),
        named_rec(b"read_005", b"a", b"!"),
        named_rec(b"read_006", b"N", b"!"),
        named_rec(b"read_007", b"-", b"!"),
        named_rec(b"read_008", b"AA", b"!!"),
        named_rec(b"read_009", b"AAAA", b"!!!!"),
        named_rec(b"read_010", b"AAAAA", b"!!!!!"),
        named_rec(b"read_011", b"ACGT", b"!!!!"),
        named_rec(b"read_012", b"ACGT", b"IIII"),
        named_rec(b"same_len_a", b"AA", b"!!"),
        named_rec(b"same_len_b", b"TT", b"II"),
        named_rec(b"same_name", b"A", b"!"),
        named_rec(b"same_name", b"T", b"I"),
        named_rec(b"aaaaaaaaaaaaaaaaY", b"A", b"!"),
        named_rec(b"aaaaaaaaaaaaaaaaZ", b"C", b"!"),
    ]
}

fn name_order_examples() -> Vec<SeqRecord> {
    vec![
        named_rec(b"", b"A", b"!"),
        named_rec(b"A", b"A", b"!"),
        named_rec(b"AA", b"A", b"!"),
        named_rec(b"AB", b"A", b"!"),
        named_rec(b"B", b"A", b"!"),
        named_rec(b"A\0", b"A", b"!"),
        named_rec(b"A\0Z", b"A", b"!"),
        named_rec(b"aaaaaaaaaaaaaaaaY", b"A", b"!"),
        named_rec(b"aaaaaaaaaaaaaaaaZ", b"A", b"!"),
    ]
}

fn length_order_examples() -> Vec<SeqRecord> {
    vec![
        named_rec(b"empty", b"", b""),
        named_rec(b"one_a", b"A", b"!"),
        named_rec(b"two_a", b"AA", b"!!"),
        named_rec(b"two_t", b"TT", b"II"),
        named_rec(b"four", b"ACGT", b"!!!!"),
        named_rec(b"eight", b"ACGTACGT", b"!!!!!!!!"),
    ]
}

#[test]
fn owned_records_have_compatible_key_and_record_order() {
    let order = SequenceOrder::<1>;
    let records = sequence_order_examples();

    assert_record_comparator_is_pairwise_consistent(order, &records);
    assert_key_comparator_is_pairwise_consistent(order, &records);
    assert_key_and_record_order_are_compatible(order, &records);
}

#[test]
fn borrowed_records_have_compatible_key_and_record_order() {
    let order = SequenceOrder::<1>;
    let records = sequence_order_examples();

    let views: Vec<SeqRecordView<'_>> = records.iter().map(SeqRecord::as_view).collect();

    assert_record_comparator_is_pairwise_consistent(order, &views);
    assert_key_comparator_is_pairwise_consistent(order, &views);
    assert_key_and_record_order_are_compatible(order, &views);
}

fn illumina_sequence_len() -> impl Strategy<Value = usize> {
    prop_oneof![0usize..=16, 148usize..=156, 157usize..=220,]
}

fn sequence_byte() -> impl Strategy<Value = u8> {
    prop_oneof![
        Just(b'A'),
        Just(b'C'),
        Just(b'G'),
        Just(b'T'),
        Just(b'a'),
        Just(b'c'),
        Just(b'g'),
        Just(b't'),
        Just(b'N'),
        Just(b'n'),
        any::<u8>(),
    ]
}

fn illumina_record() -> impl Strategy<Value = SeqRecord> {
    illumina_sequence_len()
        .prop_flat_map(|len| {
            (
                prop::collection::vec(sequence_byte(), len),
                prop::collection::vec(any::<u8>(), len),
            )
        })
        .prop_map(|(sequence, quality)| SeqRecord::new(b"record", sequence, quality))
}

fn small_record() -> impl Strategy<Value = SeqRecord> {
    (
        prop::collection::vec(any::<u8>(), 0..=8),
        prop::collection::vec(any::<u8>(), 0..=8),
    )
        .prop_flat_map(|(name, sequence)| {
            let quality_len = sequence.len();

            (
                Just(name),
                Just(sequence),
                prop::collection::vec(any::<u8>(), quality_len),
            )
        })
        .prop_map(|(name, sequence, quality)| SeqRecord::new(name, sequence, quality))
}

fn small_record_triple() -> impl Strategy<Value = (SeqRecord, SeqRecord, SeqRecord)> {
    (small_record(), small_record(), small_record())
}

fn named_record() -> impl Strategy<Value = SeqRecord> {
    prop::collection::vec(any::<u8>(), 0..=40).prop_map(|name| SeqRecord::new(name, b"A", b"!"))
}

fn length_record() -> impl Strategy<Value = SeqRecord> {
    (
        prop::collection::vec(any::<u8>(), 0..=8),
        prop::collection::vec(any::<u8>(), 0..=64),
    )
        .prop_flat_map(|(name, sequence)| {
            let quality_len = sequence.len();

            (
                Just(name),
                Just(sequence),
                prop::collection::vec(any::<u8>(), quality_len),
            )
        })
        .prop_map(|(name, sequence, quality)| SeqRecord::new(name, sequence, quality))
}

macro_rules! record_comparator_consistency_test {
    ($test_name:ident, $order:expr, owned) => {
        #[test]
        fn $test_name() {
            let records = comparator_consistency_examples();

            assert_record_comparator_is_consistent($order, &records);
        }
    };

    ($test_name:ident, $order:expr, borrowed) => {
        #[test]
        fn $test_name() {
            let records = comparator_consistency_examples();
            let views: Vec<SeqRecordView<'_>> = records.iter().map(SeqRecord::as_view).collect();

            assert_record_comparator_is_consistent($order, &views);
        }
    };
}

macro_rules! key_comparator_consistency_test {
    ($test_name:ident, $order:expr) => {
        #[test]
        fn $test_name() {
            let order = $order;
            let records = comparator_consistency_examples();
            let keys: Vec<_> = records
                .iter()
                .map(|record| order.record_key(record))
                .collect();

            assert_key_comparator_is_consistent(order, &keys);
        }
    };
}

record_comparator_consistency_test!(
    sequence_order_record_comparator_is_consistent_for_owned_records,
    SequenceOrder::<1>,
    owned
);
record_comparator_consistency_test!(
    illumina_order_record_comparator_is_consistent_for_owned_records,
    ILLUMINA_ORDER,
    owned
);
record_comparator_consistency_test!(
    name_order_record_comparator_is_consistent_for_owned_records,
    NameOrder,
    owned
);
record_comparator_consistency_test!(
    length_order_record_comparator_is_consistent_for_owned_records,
    LengthOrder,
    owned
);
record_comparator_consistency_test!(
    reverse_sequence_order_record_comparator_is_consistent_for_owned_records,
    Reverse(SequenceOrder::<1>),
    owned
);
record_comparator_consistency_test!(
    reverse_illumina_order_record_comparator_is_consistent_for_owned_records,
    Reverse(ILLUMINA_ORDER),
    owned
);
record_comparator_consistency_test!(
    reverse_name_order_record_comparator_is_consistent_for_owned_records,
    Reverse(NameOrder),
    owned
);
record_comparator_consistency_test!(
    reverse_length_order_record_comparator_is_consistent_for_owned_records,
    Reverse(LengthOrder),
    owned
);

record_comparator_consistency_test!(
    sequence_order_record_comparator_is_consistent_for_borrowed_records,
    SequenceOrder::<1>,
    borrowed
);
record_comparator_consistency_test!(
    illumina_order_record_comparator_is_consistent_for_borrowed_records,
    ILLUMINA_ORDER,
    borrowed
);
record_comparator_consistency_test!(
    name_order_record_comparator_is_consistent_for_borrowed_records,
    NameOrder,
    borrowed
);
record_comparator_consistency_test!(
    length_order_record_comparator_is_consistent_for_borrowed_records,
    LengthOrder,
    borrowed
);
record_comparator_consistency_test!(
    reverse_sequence_order_record_comparator_is_consistent_for_borrowed_records,
    Reverse(SequenceOrder::<1>),
    borrowed
);
record_comparator_consistency_test!(
    reverse_illumina_order_record_comparator_is_consistent_for_borrowed_records,
    Reverse(ILLUMINA_ORDER),
    borrowed
);
record_comparator_consistency_test!(
    reverse_name_order_record_comparator_is_consistent_for_borrowed_records,
    Reverse(NameOrder),
    borrowed
);
record_comparator_consistency_test!(
    reverse_length_order_record_comparator_is_consistent_for_borrowed_records,
    Reverse(LengthOrder),
    borrowed
);

key_comparator_consistency_test!(
    sequence_order_key_comparator_is_consistent,
    SequenceOrder::<1>
);
key_comparator_consistency_test!(illumina_order_key_comparator_is_consistent, ILLUMINA_ORDER);
key_comparator_consistency_test!(name_order_key_comparator_is_consistent, NameOrder);
key_comparator_consistency_test!(length_order_key_comparator_is_consistent, LengthOrder);
key_comparator_consistency_test!(
    reverse_sequence_order_key_comparator_is_consistent,
    Reverse(SequenceOrder::<1>)
);
key_comparator_consistency_test!(
    reverse_illumina_order_key_comparator_is_consistent,
    Reverse(ILLUMINA_ORDER)
);
key_comparator_consistency_test!(
    reverse_name_order_key_comparator_is_consistent,
    Reverse(NameOrder)
);
key_comparator_consistency_test!(
    reverse_length_order_key_comparator_is_consistent,
    Reverse(LengthOrder)
);

#[test]
fn name_order_key_and_record_order_are_compatible_for_owned_records() {
    let records = name_order_examples();

    assert_key_and_record_order_are_compatible(NameOrder, &records);
}

#[test]
fn name_order_key_and_record_order_are_compatible_for_borrowed_records() {
    let records = name_order_examples();
    let views: Vec<SeqRecordView<'_>> = records.iter().map(SeqRecord::as_view).collect();

    assert_key_and_record_order_are_compatible(NameOrder, &views);
}

#[test]
fn reverse_name_order_key_and_record_order_are_compatible_for_owned_records() {
    let records = name_order_examples();

    assert_key_and_record_order_are_compatible(Reverse(NameOrder), &records);
}

#[test]
fn reverse_name_order_key_and_record_order_are_compatible_for_borrowed_records() {
    let records = name_order_examples();
    let views: Vec<SeqRecordView<'_>> = records.iter().map(SeqRecord::as_view).collect();

    assert_key_and_record_order_are_compatible(Reverse(NameOrder), &views);
}

#[test]
fn length_order_key_and_record_order_are_compatible_for_owned_records() {
    let records = length_order_examples();

    assert_key_and_record_order_are_compatible(LengthOrder, &records);
}

#[test]
fn length_order_key_and_record_order_are_compatible_for_borrowed_records() {
    let records = length_order_examples();
    let views: Vec<SeqRecordView<'_>> = records.iter().map(SeqRecord::as_view).collect();

    assert_key_and_record_order_are_compatible(LengthOrder, &views);
}

#[test]
fn reverse_length_order_key_and_record_order_are_compatible_for_owned_records() {
    let records = length_order_examples();

    assert_key_and_record_order_are_compatible(Reverse(LengthOrder), &records);
}

#[test]
fn reverse_length_order_key_and_record_order_are_compatible_for_borrowed_records() {
    let records = length_order_examples();
    let views: Vec<SeqRecordView<'_>> = records.iter().map(SeqRecord::as_view).collect();

    assert_key_and_record_order_are_compatible(Reverse(LengthOrder), &views);
}

#[test]
fn length_order_ties_distinct_equal_length_records() {
    let left = named_rec(b"same_len_a", b"AA", b"!!");
    let right = named_rec(b"same_len_b", b"TT", b"II");

    assert_eq!(
        LengthOrder.comparator().compare(&left, &right),
        Ordering::Equal
    );
    assert_eq!(
        LengthOrder.record_key(&left),
        LengthOrder.record_key(&right)
    );
}

proptest! {
    #[test]
    fn generated_illumina_records_have_compatible_key_and_record_order(
        left in illumina_record(),
        right in illumina_record(),
    ) {
        let records = vec![left, right];

        assert_record_comparator_is_pairwise_consistent(ILLUMINA_ORDER, &records);
        assert_key_comparator_is_pairwise_consistent(ILLUMINA_ORDER, &records);
        assert_key_and_record_order_are_compatible(ILLUMINA_ORDER, &records);
    }

    #[test]
    fn sequence_order_record_comparator_is_consistent_for_generated_records(
        (a, b, c) in small_record_triple(),
    ) {
        let records = [a, b, c];

        assert_record_comparator_is_consistent(SequenceOrder::<1>, &records);
    }

    #[test]
    fn illumina_order_record_comparator_is_consistent_for_generated_records(
        (a, b, c) in small_record_triple(),
    ) {
        let records = [a, b, c];

        assert_record_comparator_is_consistent(ILLUMINA_ORDER, &records);
    }

    #[test]
    fn name_order_record_comparator_is_consistent_for_generated_records(
        (a, b, c) in small_record_triple(),
    ) {
        let records = [a, b, c];

        assert_record_comparator_is_consistent(NameOrder, &records);
    }

    #[test]
    fn length_order_record_comparator_is_consistent_for_generated_records(
        (a, b, c) in small_record_triple(),
    ) {
        let records = [a, b, c];

        assert_record_comparator_is_consistent(LengthOrder, &records);
    }

    #[test]
    fn generated_name_records_have_compatible_key_and_record_order(
        left in named_record(),
        right in named_record(),
    ) {
        let records = [left, right];

        assert_key_and_record_order_are_compatible(NameOrder, &records);
    }

    #[test]
    fn generated_length_records_have_compatible_key_and_record_order(
        left in length_record(),
        right in length_record(),
    ) {
        let records = [left, right];

        assert_key_and_record_order_are_compatible(LengthOrder, &records);
    }
}
