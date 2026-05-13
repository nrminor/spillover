use spillover_bio::{codec::DryIceCodec, record::SeqRecord, sort::Builder};

fn rec(name: &str, qual: &str) -> SeqRecord {
    SeqRecord::new(name.as_bytes(), b"ACGTACGT", qual.as_bytes())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    for record in [
        rec("r1", "IIIIIIII"),
        rec("r2", "!!!!!!!!"),
        rec("r3", "########"),
        rec("r4", "JJJJJJJJ"),
    ] {
        sorter.push(record)?;
    }

    let output: Vec<SeqRecord> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    for record in output {
        println!(
            "{}\t{}",
            String::from_utf8_lossy(record.name()),
            String::from_utf8_lossy(record.quality())
        );
    }

    Ok(())
}
