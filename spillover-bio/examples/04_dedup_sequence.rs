use spillover_bio::{codec::DryIceCodec, record::SeqRecord, sort::Builder};

fn rec(name: &str, seq: &str, qual: &str) -> SeqRecord {
    SeqRecord::new(
        name.as_bytes().to_vec(),
        seq.as_bytes().to_vec(),
        qual.as_bytes().to_vec(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .dedup_by_sequence()
        .max_buffer_items(3)
        .build();

    for record in [
        rec("a1", "AAAAAAAA", "!!!!!!!!"),
        rec("a2", "AAAAAAAA", "IIIIIIII"),
        rec("c1", "CCCCCCCC", "########"),
        rec("c2", "CCCCCCCC", "IIIIIIII"),
        rec("t1", "TTTTTTTT", "!!!!!!!!"),
    ] {
        sorter.push(record)?;
    }

    let deduped: Vec<SeqRecord> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    println!("deduped record count: {}", deduped.len());
    for r in deduped {
        println!(
            "{}\t{}",
            String::from_utf8_lossy(r.name()),
            String::from_utf8_lossy(r.sequence())
        );
    }

    Ok(())
}
