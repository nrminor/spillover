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
        .max_buffer_items(2)
        .build();

    for record in [
        rec("r1", "TTTTTTTT", "IIIIIIII"),
        rec("r2", "AAAAAAAA", "!!!!!!!!"),
        rec("r3", "CCCCCCCC", "########"),
        rec("r4", "AAAAAAAA", "IIIIIIII"),
    ] {
        sorter.push(record)?;
    }

    let sorted_records: Vec<SeqRecord> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    for r in sorted_records {
        println!(
            "{}\t{}\t{}",
            String::from_utf8_lossy(r.name()),
            String::from_utf8_lossy(r.sequence()),
            String::from_utf8_lossy(r.quality())
        );
    }

    Ok(())
}
