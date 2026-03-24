//! End-to-end integration tests for the spillover-bio sort pipeline.

use spillover_bio::{codec::DryIceCodec, record::SeqRecord, sort::Builder};

fn make_record(name: &[u8], seq: &[u8], qual: &[u8]) -> SeqRecord {
    SeqRecord::new(name.to_vec(), seq.to_vec(), qual.to_vec())
}

#[test]
fn unsorted_records_come_out_sorted_by_sequence() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    let records = vec![
        make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
        make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
        make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
    ];

    for rec in records {
        sorter.push(rec).expect("push should succeed");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|r| r.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn sort_with_disk_spilling() {
    // Buffer holds 3 records, input has 10 → multiple flushes to disk.
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(3)
        .build();

    let bases = [b'A', b'C', b'G', b'T'];
    for i in (0..10).rev() {
        let seq: Vec<u8> = std::iter::repeat_n(bases[i % 4], 8).collect();
        let qual = vec![b'!'; 8];
        sorter
            .push(make_record(format!("r{i}").as_bytes(), &seq, &qual))
            .expect("push should succeed");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|r| r.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 10);
    // Verify sorted order
    for window in results.windows(2) {
        assert!(
            window[0].sequence() <= window[1].sequence(),
            "output should be sorted: {:?} should come before {:?}",
            std::str::from_utf8(window[0].sequence()),
            std::str::from_utf8(window[1].sequence()),
        );
    }
}

#[test]
fn quality_tiebreaker_orders_identical_sequences() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"r1", b"ACGTACGT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"ACGTACGT", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"ACGTACGT", b"########"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 3);
    // All same sequence, ordered by quality: ! < # < I in ASCII
    assert_eq!(results[0].quality(), b"!!!!!!!!");
    assert_eq!(results[1].quality(), b"########");
    assert_eq!(results[2].quality(), b"IIIIIIII");
}

#[test]
fn sort_by_name_orders_lexicographically() {
    let mut sorter = Builder::new()
        .sort_by_name()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"charlie", b"ACGT", b"!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"alice", b"TGCA", b"!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"bob", b"GGGG", b"!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].name(), b"alice");
    assert_eq!(results[1].name(), b"bob");
    assert_eq!(results[2].name(), b"charlie");
}

#[test]
fn reverse_sequence_order_produces_descending_output() {
    use spillover_bio::sort::{ILLUMINA_ORDER, Reverse};

    let mut sorter = Builder::new()
        .sort_by(Reverse(ILLUMINA_ORDER))
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"TTTTTTTT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"CCCCCCCC", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"TTTTTTTT");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"AAAAAAAA");
}

#[test]
fn sort_by_length_orders_shortest_first() {
    use spillover_bio::sort::LENGTH_ORDER;

    let mut sorter = Builder::new()
        .sort_by(LENGTH_ORDER)
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"r1", b"ACGTACGTACGT", b"!!!!!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"ACGT", b"!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"ACGTACGT", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence().len(), 4);
    assert_eq!(results[1].sequence().len(), 8);
    assert_eq!(results[2].sequence().len(), 12);
}

#[test]
fn dedup_removes_duplicate_sequences() {
    use spillover::dedup::AdjacentDedup;

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .dedup(AdjacentDedup::new(|a: &SeqRecord, b: &SeqRecord| {
            a.sequence() == b.sequence()
        }))
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"r1", b"ACGTACGT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"ACGTACGT", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r4", b"TTTTTTTT", b"########"))
        .expect("push");
    sorter
        .push(make_record(b"r5", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(
        results.len(),
        3,
        "duplicates should be collapsed to one per unique sequence"
    );
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"ACGTACGT");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn empty_input_produces_empty_output() {
    let sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert!(results.is_empty());
}

#[test]
fn large_dataset_matches_reference_sort() {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut input: Vec<SeqRecord> = (0..500)
        .map(|i| {
            let seq: Vec<u8> = (0..16).map(|j| bases[(i * 7 + j * 13) % 4]).collect();
            let qual: Vec<u8> = (0..16)
                .map(|j| b'!' + u8::try_from((i + j) % 40).expect("fits"))
                .collect();
            make_record(format!("r{i:04}").as_bytes(), &seq, &qual)
        })
        .collect();

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(50)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    // Build reference: sort by sequence only. The keyed merge
    // does not guarantee quality ordering across run boundaries.
    input.sort_by(|a, b| a.sequence().cmp(b.sequence()));

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i}"
        );
    }
}

#[test]
fn spilling_with_many_flushes_preserves_all_records() {
    // Buffer holds 2 records, input has 20 → 10 flushes.
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    let bases = [b'A', b'C', b'G', b'T'];
    for i in 0..20 {
        let seq: Vec<u8> = (0..8).map(|j| bases[(i * 3 + j * 7) % 4]).collect();
        let qual = vec![b'!'; 8];
        sorter
            .push(make_record(format!("r{i:02}").as_bytes(), &seq, &qual))
            .expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(
        results.len(),
        20,
        "all records should survive multiple flushes and merge"
    );

    for window in results.windows(2) {
        assert!(
            (window[0].sequence(), window[0].quality())
                <= (window[1].sequence(), window[1].quality()),
            "output should be sorted"
        );
    }
}

#[test]
fn compact_codecs_round_trip_correctly() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(
            DryIceCodec::new()
                .two_bit_exact()
                .binned_quality()
                .split_names(),
        )
        .max_buffer_items(100)
        .build();

    let records = vec![
        make_record(
            b"instrument:run:flowcell 1:N:0:ATCACG",
            b"TTTTTTTTTTTTTTTT",
            b"IIIIIIIIIIIIIIII",
        ),
        make_record(
            b"instrument:run:flowcell 2:N:0:ATCACG",
            b"AAAAAAAAAAAAAAAA",
            b"!!!!!!!!!!!!!!!!",
        ),
        make_record(
            b"instrument:run:flowcell 3:N:0:ATCACG",
            b"CCCCCCCCCCCCCCCC",
            b"################",
        ),
    ];

    for rec in records {
        sorter.push(rec).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"AAAAAAAAAAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCCCCCCCCCC");
    assert_eq!(results[2].sequence(), b"TTTTTTTTTTTTTTTT");
}

#[test]
fn realistic_150bp_reads_sort_correctly() {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut input: Vec<SeqRecord> = (0..100)
        .map(|i| {
            let seq: Vec<u8> = (0..150)
                .map(|j| bases[(i * 7 + j * 13 + j / 10) % 4])
                .collect();
            let qual: Vec<u8> = (0..150)
                .map(|j| b'!' + u8::try_from((i + j * 3) % 40).expect("fits"))
                .collect();
            make_record(format!("read_{i:04}").as_bytes(), &seq, &qual)
        })
        .collect();

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(20)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    input.sort_by(|a, b| a.sequence().cmp(b.sequence()));

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i}"
        );
    }
    // Quality ordering within equal sequences is guaranteed within
    // each sorted chunk but not across the merge, since the keyed
    // merge compares only packed keys, not full (sequence, quality)
    // tuples. Cross-run quality ordering requires the fallback
    // comparison path (not yet implemented).
}

#[test]
fn radix_sort_through_pipeline() {
    use spillover_bio::radix::RadixThenRefine;

    let bases = [b'A', b'C', b'G', b'T'];
    let mut input: Vec<SeqRecord> = (0..200)
        .map(|i| {
            let seq: Vec<u8> = (0..16).map(|j| bases[(i * 7 + j * 13) % 4]).collect();
            let qual: Vec<u8> = (0..16)
                .map(|j| b'!' + u8::try_from((i + j) % 40).expect("fits"))
                .collect();
            make_record(format!("r{i:03}").as_bytes(), &seq, &qual)
        })
        .collect();

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .chunk_sort(RadixThenRefine::<4>)
        .max_buffer_items(50)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    input.sort_by(|a, b| a.sequence().cmp(b.sequence()));

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i} with radix sort"
        );
    }
}

#[test]
fn sequences_with_ambiguous_bases_sort_correctly() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    // N maps to A in packed keys, so NNNNNNNN and AAAAAAAA
    // have the same packed key. Quality tiebreaker should
    // differentiate them.
    sorter
        .push(make_record(b"r1", b"NNNNNNNN", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 3);
    // AAAAAAAA and NNNNNNNN both sort before TTTTTTTT.
    // Between them, quality tiebreaker: ! < I
    assert_eq!(results[0].quality(), b"!!!!!!!!");
    assert_eq!(results[1].quality(), b"IIIIIIII");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn single_record_round_trips() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"only_record", b"ACGTACGT", b"IIIIIIII"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name(), b"only_record");
    assert_eq!(results[0].sequence(), b"ACGTACGT");
    assert_eq!(results[0].quality(), b"IIIIIIII");
}

#[test]
fn all_identical_records_sort_and_preserve_count() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(3)
        .build();

    for i in 0..10 {
        sorter
            .push(make_record(
                format!("r{i}").as_bytes(),
                b"ACGTACGT",
                b"!!!!!!!!",
            ))
            .expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(
        results.len(),
        10,
        "identical records should all be preserved without dedup"
    );
}

#[test]
fn dedup_with_spilling_removes_cross_chunk_duplicates() {
    use spillover::dedup::AdjacentDedup;

    // Buffer holds 3 records. Duplicates span chunk boundaries.
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .dedup(AdjacentDedup::new(|a: &SeqRecord, b: &SeqRecord| {
            a.sequence() == b.sequence()
        }))
        .max_buffer_items(3)
        .build();

    // Push records with duplicates that will end up in different chunks
    for seq in [
        b"AAAAAAAA",
        b"AAAAAAAA",
        b"CCCCCCCC",
        b"CCCCCCCC",
        b"CCCCCCCC",
        b"GGGGGGGG",
        b"TTTTTTTT",
        b"TTTTTTTT",
    ] {
        sorter
            .push(make_record(b"r", seq.as_slice(), &[b'!'; 8]))
            .expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(
        results.len(),
        4,
        "dedup should collapse duplicates even across chunk boundaries"
    );
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"GGGGGGGG");
    assert_eq!(results[3].sequence(), b"TTTTTTTT");
}

#[test]
fn unkeyed_sort_preserves_exact_sequence_and_quality_ordering() {
    use spillover_bio::sort::ILLUMINA_ORDER;

    // The unkeyed path uses full (sequence, quality) comparison
    // during merge, so quality tiebreaking is preserved across
    // run boundaries — unlike the keyed path.
    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .max_buffer_items(3)
        .build();

    sorter
        .push(make_record(b"r1", b"ACGTACGT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"ACGTACGT", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"ACGTACGT", b"########"))
        .expect("push");
    sorter
        .push(make_record(b"r4", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r5", b"TTTTTTTT", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 5);
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    // ACGT group: quality ordering preserved across runs
    assert_eq!(results[1].sequence(), b"ACGTACGT");
    assert_eq!(results[1].quality(), b"!!!!!!!!");
    assert_eq!(results[2].sequence(), b"ACGTACGT");
    assert_eq!(results[2].quality(), b"########");
    assert_eq!(results[3].sequence(), b"ACGTACGT");
    assert_eq!(results[3].quality(), b"IIIIIIII");
    assert_eq!(results[4].sequence(), b"TTTTTTTT");
}

#[test]
fn unkeyed_sort_with_variable_length_sequences() {
    use spillover_bio::sort::ILLUMINA_ORDER;

    // The unkeyed path doesn't have the packed key padding
    // ambiguity, so variable-length sequences sort correctly.
    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .max_buffer_items(3)
        .build();

    sorter.push(make_record(b"r1", b"TA", b"!!")).expect("push");
    sorter.push(make_record(b"r2", b"T", b"!")).expect("push");
    sorter
        .push(make_record(b"r3", b"TAA", b"!!!"))
        .expect("push");
    sorter.push(make_record(b"r4", b"A", b"!")).expect("push");
    sorter
        .push(make_record(b"r5", b"AAAA", b"!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 5);
    // Lexicographic: A < AAAA < T < TA < TAA
    assert_eq!(results[0].sequence(), b"A");
    assert_eq!(results[1].sequence(), b"AAAA");
    assert_eq!(results[2].sequence(), b"T");
    assert_eq!(results[3].sequence(), b"TA");
    assert_eq!(results[4].sequence(), b"TAA");
}

mod proptests {
    use proptest::prelude::*;

    use super::*;

    fn arb_fixed_len_sequence(len: usize) -> impl Strategy<Value = Vec<u8>> {
        proptest::collection::vec(
            proptest::sample::select(vec![b'A', b'C', b'G', b'T']),
            len..=len,
        )
    }

    fn arb_quality(len: usize) -> impl Strategy<Value = Vec<u8>> {
        proptest::collection::vec(b'!'..=b'I', len..=len)
    }

    /// Generate records with uniform sequence length. Packed key
    /// ordering is exact when all sequences are the same length
    /// (no padding ambiguity with trailing A's).
    fn arb_record(seq_len: usize) -> impl Strategy<Value = SeqRecord> {
        (arb_fixed_len_sequence(seq_len), arb_quality(seq_len))
            .prop_map(|(seq, qual)| SeqRecord::new(b"r".to_vec(), seq, qual))
    }

    proptest! {
        #[test]
        fn output_is_sorted_for_any_input(
            records in proptest::collection::vec(arb_record(32), 1..100),
            buffer_size in 2usize..20,
        ) {
            let mut sorter = Builder::new()
                .sort_by_illumina()
                .codec(DryIceCodec::new())
                .max_buffer_items(buffer_size)
                .build();

            for rec in &records {
                sorter.push(rec.clone()).expect("push");
            }

            let results: Vec<SeqRecord> = sorter
                .finish()
                .expect("finish")
                .map(|r| r.expect("decode"))
                .collect();

            prop_assert_eq!(results.len(), records.len());

            // Keyed merge guarantees sequence ordering but not
            // quality ordering across run boundaries.
            for window in results.windows(2) {
                prop_assert!(
                    window[0].sequence() <= window[1].sequence(),
                    "output must be sorted by sequence"
                );
            }
        }

        #[test]
        fn output_preserves_all_records(
            records in proptest::collection::vec(arb_record(32), 1..100),
            buffer_size in 2usize..20,
        ) {
            let mut sorter = Builder::new()
                .sort_by_illumina()
                .codec(DryIceCodec::new())
                .max_buffer_items(buffer_size)
                .build();

            for rec in &records {
                sorter.push(rec.clone()).expect("push");
            }

            let mut results: Vec<SeqRecord> = sorter
                .finish()
                .expect("finish")
                .map(|r| r.expect("decode"))
                .collect();

            let mut expected = records;
            expected.sort_by(|a, b| {
                a.sequence()
                    .cmp(b.sequence())
                    .then_with(|| a.quality().cmp(b.quality()))
            });

            prop_assert_eq!(results.len(), expected.len());

            // Sort both by (seq, qual, name) for comparison since
            // records with equal (seq, qual) may be in any relative order.
            let key = |r: &SeqRecord| {
                (r.sequence().to_vec(), r.quality().to_vec(), r.name().to_vec())
            };
            results.sort_by_key(|r| key(r));
            expected.sort_by_key(|r| key(r));

            for (got, exp) in results.iter().zip(expected.iter()) {
                prop_assert_eq!(got.sequence(), exp.sequence());
                prop_assert_eq!(got.quality(), exp.quality());
            }
        }

        #[test]
        fn name_sort_output_is_sorted(
            records in proptest::collection::vec(
                (proptest::collection::vec(b'a'..=b'z', 1..20), arb_fixed_len_sequence(16))
                    .prop_flat_map(|(name, seq)| {
                        let len = seq.len();
                        (Just(name), Just(seq), arb_quality(len))
                    })
                    .prop_map(|(name, seq, qual)| SeqRecord::new(name, seq, qual)),
                1..50,
            ),
            buffer_size in 2usize..10,
        ) {
            let mut sorter = Builder::new()
                .sort_by_name()
                .codec(DryIceCodec::new())
                .max_buffer_items(buffer_size)
                .build();

            for rec in &records {
                sorter.push(rec.clone()).expect("push");
            }

            let results: Vec<SeqRecord> = sorter
                .finish()
                .expect("finish")
                .map(|r| r.expect("decode"))
                .collect();

            prop_assert_eq!(results.len(), records.len());

            for window in results.windows(2) {
                prop_assert!(
                    window[0].name() <= window[1].name(),
                    "output must be sorted by name"
                );
            }
        }
    }
}
