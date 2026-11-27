//! Sort existing collections or collect a batch without temporary files.
//!
//! All records and algorithm scratch must fit in memory.
//! Run with `cargo run -p spillover-bio --example 09_in_memory_sorting`.

use std::{convert::Infallible, error::Error};

use spillover_bio::{
    InMemoryBuilder as Builder, ReadPair, SeqRecord, SeqRecordArena, SeqRecordParts,
};

fn show(record: &impl SeqRecordParts) {
    println!(
        "{}\t{}",
        String::from_utf8_lossy(record.name()),
        String::from_utf8_lossy(record.sequence())
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut reads = vec![
        SeqRecord::new("read-c", "TT", "II"),
        SeqRecord::new("read-b", "AA", "II"),
        SeqRecord::new("read-a", "AA", "!!"),
    ];

    // A slice can be reordered but cannot shrink.
    Builder::new().sort_by_name().sort_slice(&mut reads);
    assert_eq!(reads[0].name(), b"read-a");

    // Sort and physically remove adjacent duplicates in the original Vec.
    // The first sorted AA record has the lower quality bytes and is retained.
    let allocation = reads.as_ptr();
    Builder::new()
        .sort_by_illumina()
        .dedup_by_sequence()
        .sort_vec(&mut reads);
    assert_eq!(reads.len(), 2);
    assert_eq!(reads.as_ptr(), allocation);
    println!("Compacted collection:");
    for read in &reads {
        show(read);
    }

    // Transfer that allocation into a sorter and append more input.
    // Use build() instead to start with an empty owned batch.
    let mut sorter = Builder::new().sort_by_name().build_from(reads);
    sorter.push(SeqRecord::new("read-b", "CC", "##"));
    println!("Owned output:");
    for result in sorter.finish() {
        let record = result?;
        show(&record);
    }

    // Ingest copies input bytes into reusable arena storage. finish returns
    // views, not independent owned records. Dedup still retains a representative.
    let mut arena = SeqRecordArena::with_capacity(1024);
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .dedup_by_sequence()
        .arena(&mut arena)
        .build();
    for record in [
        SeqRecord::new("arena-c", "TT", "II"),
        SeqRecord::new("arena-b", "AA", "II"),
        SeqRecord::new("arena-a", "AA", "!!"),
    ] {
        sorter.push(&record.as_view());
    }
    let mut output = sorter.finish();
    println!("Arena output:");
    if let Some(result) = output.next() {
        let view = result?;
        show(&view);
    }
    // Visitation continues with the same dedup history as iteration.
    output.items().try_for_each(|view| {
        show(view);
        Ok::<(), Infallible>(())
    })?;

    // All output views are now finished. The next build clears logical contents,
    // retaining capacity; it does not treat the old arena contents as input.
    assert!(!arena.is_empty());
    let capacity = arena.capacity();
    let next = Builder::new().sort_by_name().arena(&mut arena).build();
    assert_eq!(next.finish().count(), 0);
    assert!(arena.is_empty());
    assert_eq!(arena.capacity(), capacity);

    // Pairs remain atomic; pair arena storage and built-in dedup are unsupported.
    let mut sorter = Builder::new().sort_by_illumina_pairs().build();
    for (name, sequence) in [("pair-b", "TT"), ("pair-a", "AA")] {
        let r1 = SeqRecord::new(name, sequence, "!!");
        let r2 = SeqRecord::new(name, "CC", "##");
        sorter.push(ReadPair::from_mates(r1, r2));
    }
    println!("Pair output:");
    for result in sorter.finish() {
        let pair = result?;
        show(pair.r1());
        show(pair.r2());
    }
    Ok(())
}
