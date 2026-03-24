use spillover_bio::{codec::DryIceCodec, record::SeqRecord, sort::Builder};

fn rec(name: &str, seq: &str, qual: &str) -> SeqRecord {
    SeqRecord::new(
        name.as_bytes().to_vec(),
        seq.as_bytes().to_vec(),
        qual.as_bytes().to_vec(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compact_codec = DryIceCodec::new()
        .two_bit_exact()
        .binned_quality()
        .split_names();

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(compact_codec)
        .max_buffer_items(2)
        .build();

    for record in [
        rec(
            "instrument:run:flowcell 3:N:0:ATCACG",
            "TTTTTTTTTTTTTTTT",
            "IIIIIIIIIIIIIIII",
        ),
        rec(
            "instrument:run:flowcell 1:N:0:ATCACG",
            "AAAAAAAAAAAAAAAA",
            "!!!!!!!!!!!!!!!!",
        ),
        rec(
            "instrument:run:flowcell 2:N:0:ATCACG",
            "CCCCCCCCCCCCCCCC",
            "################",
        ),
    ] {
        sorter.push(record)?;
    }

    let output: Vec<SeqRecord> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    for record in output {
        println!(
            "{}\t{}",
            String::from_utf8_lossy(record.name()),
            String::from_utf8_lossy(record.sequence())
        );
    }

    Ok(())
}
