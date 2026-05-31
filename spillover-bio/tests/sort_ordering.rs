use std::cmp::Ordering;

use proptest::prelude::*;
use spillover::compare::Compare;
use spillover_bio::{
    record::{SeqRecord, SeqRecordParts, SeqRecordView},
    sort::{ILLUMINA_ORDER, RecordKeyer, SequenceOrder, SortOrder},
};

fn rec(sequence: &[u8], quality: &[u8]) -> SeqRecord {
    SeqRecord::new(b"record", sequence, quality)
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
}
