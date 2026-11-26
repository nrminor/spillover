use std::{cmp::Ordering, mem::size_of};

use get_size2::GetSize;
use proptest::prelude::*;
use spillover::{
    compare::{Compare, Reverse},
    key::{DeriveKey, EstimateKeyMemory, KeyedOrder},
};
use spillover_bio::{
    record::{ReadPair, SeqRecord, SeqRecordView},
    sort::{
        ILLUMINA_250_PAIR_ORDER, ILLUMINA_PAIR_ORDER, PAIR_LENGTHS_ORDER, PAIR_NAME_ORDER,
        PAIR_QUALITY_ORDER, PAIR_SEQUENCE_EXACT_ORDER, PairSequenceOrder,
    },
};

fn pair(fields: [&[u8]; 6]) -> ReadPair<SeqRecord> {
    let [seq1, seq2, qual1, qual2, name1, name2] = fields;
    let r1 = SeqRecord::new(name1, seq1, qual1);
    let r2 = SeqRecord::new(name2, seq2, qual2);
    ReadPair::from_mates(r1, r2)
}

#[test]
fn exact_order_compares_every_field_in_selected_priority() {
    let baseline = [b"M".as_slice(); 6];
    let left = pair(baseline);
    for position in 0..6 {
        let mut fields = baseline;
        fields[position] = b"N";
        for later in &mut fields[position + 1..] {
            *later = b"A";
        }
        let right = pair(fields);
        assert_eq!(
            PAIR_SEQUENCE_EXACT_ORDER.compare(&left, &right),
            Ordering::Less
        );
        assert_eq!(
            PAIR_SEQUENCE_EXACT_ORDER.compare(&right, &left),
            Ordering::Greater
        );
    }
    assert_eq!(
        PAIR_SEQUENCE_EXACT_ORDER.compare(&left, &left),
        Ordering::Equal
    );
}

#[test]
fn packed_order_refines_collisions_and_uses_r1_key() {
    let order = PairSequenceOrder::<1>;
    for (first, second) in [
        (b"A".as_slice(), b"a".as_slice()),
        (b"A".as_slice(), b"N".as_slice()),
        (b"A".as_slice(), b"AA".as_slice()),
        (b"AAAAA".as_slice(), b"AAAAC".as_slice()),
    ] {
        let left = pair([first, b"T", b"z", b"!", b"a", b"a"]);
        let right = pair([second, b"A", b"!", b"!", b"a", b"a"]);
        assert_eq!(order.derive_key(&left), order.derive_key(&right));
        assert_eq!(
            order.compare(&left, &right),
            PAIR_SEQUENCE_EXACT_ORDER.compare(&left, &right)
        );
    }
    let left = pair([b"A", b"T", b"z", b"!", b"a", b"a"]);
    let right = pair([b"C", b"A", b"!", b"!", b"a", b"a"]);
    assert_eq!(order.compare(&left, &right), Ordering::Less);
    let _estimate = order.estimate_key_memory(&left);
    let _ = ILLUMINA_PAIR_ORDER.derive_key(&left);
    let _ = ILLUMINA_250_PAIR_ORDER.derive_key(&left);
}

#[test]
fn matewise_orders_ignore_unrelated_fields() {
    let left = pair([b"A", b"AA", b"a", b"z", b"a", b"z"]);
    let right = pair([b"AA", b"A", b"b", b"a", b"b", b"a"]);
    assert_eq!(PAIR_NAME_ORDER.compare(&left, &right), Ordering::Less);
    assert_eq!(PAIR_QUALITY_ORDER.compare(&left, &right), Ordering::Less);
    assert_eq!(PAIR_LENGTHS_ORDER.compare(&left, &right), Ordering::Less);

    let same_names = pair([b"T", b"TT", b"x", b"y", b"a", b"z"]);
    let same_qualities = pair([b"T", b"TT", b"a", b"z", b"x", b"y"]);
    let same_lengths = pair([b"T", b"TT", b"x", b"y", b"x", b"y"]);
    assert_eq!(PAIR_NAME_ORDER.compare(&left, &same_names), Ordering::Equal);
    assert_eq!(
        PAIR_QUALITY_ORDER.compare(&left, &same_qualities),
        Ordering::Equal
    );
    assert_eq!(
        PAIR_LENGTHS_ORDER.compare(&left, &same_lengths),
        Ordering::Equal
    );

    let second_greater = pair([b"A", b"AAA", b"a", b"zz", b"a", b"zz"]);
    assert_eq!(
        PAIR_NAME_ORDER.compare(&left, &second_greater),
        Ordering::Less
    );
    assert_eq!(
        PAIR_QUALITY_ORDER.compare(&left, &second_greater),
        Ordering::Less
    );
    assert_eq!(
        PAIR_LENGTHS_ORDER.compare(&left, &second_greater),
        Ordering::Less
    );
}

#[test]
fn borrowed_views_and_reverse_preserve_mate_roles() {
    let first = SeqRecordView::new(b"first", b"A", b"z");
    let second = SeqRecordView::new(b"second", b"C", b"a");
    let left = ReadPair::from_mates(first, second);
    let right = ReadPair::from_mates(second, first);
    assert_eq!(
        PAIR_SEQUENCE_EXACT_ORDER.compare(&left, &right),
        Ordering::Less
    );
    assert_eq!(ILLUMINA_PAIR_ORDER.compare(&left, &right), Ordering::Less);
    assert_eq!(PAIR_NAME_ORDER.compare(&left, &right), Ordering::Less);
    assert_eq!(PAIR_QUALITY_ORDER.compare(&left, &right), Ordering::Greater);
    assert_eq!(PAIR_LENGTHS_ORDER.compare(&left, &right), Ordering::Equal);
    let reversed = Reverse(PAIR_SEQUENCE_EXACT_ORDER);
    assert_eq!(reversed.compare(&left, &right), Ordering::Greater);
    assert_eq!(left.r1().name(), b"first");
    assert_eq!(left.r2().name(), b"second");
}

#[test]
fn validation_borrows_pair_and_preserves_owned_error_and_mates() {
    let left = SeqRecord::new(b"first", b"A", b"!");
    let right = SeqRecord::new(b"second", b"C", b"#");
    let pair = ReadPair::from_mates(left.clone(), right.clone());
    let reason = String::from("application rejection");
    let result = pair.validate_with(|candidate| {
        assert_eq!(candidate.r1(), &left);
        Err::<(), _>(reason)
    });
    assert_eq!(
        result.expect_err("application validation rejects"),
        "application rejection"
    );
    assert_eq!(pair.into_mates(), (left, right));
}

#[test]
fn pair_size_counts_inline_mates_and_both_payloads() {
    let first = SeqRecord::new(b"first", b"AC", b"!!");
    let second = SeqRecord::new(b"second", b"GTA", b"###");
    let expected = first.get_size() + second.get_size();
    let pair = ReadPair::from_mates(first, second);
    assert_eq!(size_of::<ReadPair<SeqRecord>>(), 2 * size_of::<SeqRecord>());
    assert_eq!(pair.get_size(), expected);
    assert!(pair.get_size() > size_of::<ReadPair<SeqRecord>>());
}

#[test]
fn validation_accepts_a_stateful_borrowed_callback() {
    let pair = ReadPair::from_mates(1, 2);
    let mut calls = 0;
    let mut validate = |candidate: &ReadPair<i32>| {
        calls += 1;
        assert_eq!((*candidate.r1(), *candidate.r2()), (1, 2));
        Ok::<(), ()>(())
    };
    pair.validate_with(&mut validate).expect("first validation");
    pair.validate_with(&mut validate)
        .expect("second validation");
    assert_eq!(calls, 2);
    assert_eq!(pair.into_mates(), (1, 2));
}

fn check_order_laws<O: Compare<ReadPair<SeqRecord>>>(
    order: &O,
    records: &[ReadPair<SeqRecord>; 3],
) {
    for left in records {
        assert_eq!(order.compare(left, left), Ordering::Equal);
        for right in records {
            let forward = order.compare(left, right);
            let backward = order.compare(right, left);
            assert_eq!(forward, backward.reverse());
            for third in records {
                if forward.is_le() && order.compare(right, third).is_le() {
                    assert!(order.compare(left, third).is_le());
                }
            }
        }
    }
}

fn check_key_refinement<const N: usize>(
    order: PairSequenceOrder<N>,
    records: &[ReadPair<SeqRecord>; 3],
) {
    for left in records {
        for right in records {
            let left_key = order.derive_key(left);
            let right_key = order.derive_key(right);
            let primary = order.compare_keys(&left_key, &right_key);
            let expected = if primary.is_eq() {
                PAIR_SEQUENCE_EXACT_ORDER.compare(left, right)
            } else {
                primary
            };
            assert_eq!(order.compare(left, right), expected);
        }
    }
}

proptest! {
    #[test]
    fn pair_orders_obey_comparator_and_key_laws(
        generated in prop::array::uniform3(prop::array::uniform6("[ACGTNacgtn]{0,170}"))
    ) {
        let records = generated.map(|fields| {
            let borrowed = fields.each_ref().map(String::as_bytes);
            pair(borrowed)
        });
        check_order_laws(&PAIR_SEQUENCE_EXACT_ORDER, &records);
        check_order_laws(&PAIR_NAME_ORDER, &records);
        check_order_laws(&PAIR_QUALITY_ORDER, &records);
        check_order_laws(&PAIR_LENGTHS_ORDER, &records);
        check_order_laws(&ILLUMINA_PAIR_ORDER, &records);
        check_order_laws(&ILLUMINA_250_PAIR_ORDER, &records);
        check_order_laws(&Reverse(ILLUMINA_PAIR_ORDER), &records);
        check_key_refinement(ILLUMINA_PAIR_ORDER, &records);
        check_key_refinement(ILLUMINA_250_PAIR_ORDER, &records);
    }
}
