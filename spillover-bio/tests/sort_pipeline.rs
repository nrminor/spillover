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

    // Build reference: sort by (sequence, quality)
    input.sort_by(|a, b| {
        a.sequence()
            .cmp(b.sequence())
            .then_with(|| a.quality().cmp(b.quality()))
    });

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i}"
        );
        assert_eq!(
            got.quality(),
            expected.quality(),
            "quality mismatch at position {i}"
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
