use spillover_bio::{
    codec::DryIceCodec,
    record::SeqRecord,
    sort::{Builder, ILLUMINA_ORDER, Reverse},
};

fn rec(name: &str, seq: &str, qual: &str) -> SeqRecord {
    SeqRecord::new(
        name.as_bytes().to_vec(),
        seq.as_bytes().to_vec(),
        qual.as_bytes().to_vec(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = vec![
        rec("charlie", "AC", "!!"),
        rec("alice", "TTTT", "IIII"),
        rec("bob", "G", "#"),
    ];

    let mut by_name = Builder::new()
        .sort_by_name()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();
    for rec in &input {
        by_name.push(rec.clone())?;
    }
    let by_name = by_name.finish()?.collect::<Result<Vec<_>, _>>()?;
    println!("by name:");
    for r in &by_name {
        println!("  {}", String::from_utf8_lossy(r.name()));
    }

    let mut by_length = Builder::new()
        .sort_by_length()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();
    for rec in &input {
        by_length.push(rec.clone())?;
    }
    let by_length = by_length.finish()?.collect::<Result<Vec<_>, _>>()?;
    println!("by length:");
    for r in &by_length {
        println!(
            "  {} (len {})",
            String::from_utf8_lossy(r.sequence()),
            r.sequence().len()
        );
    }

    let mut reverse_sequence = Builder::new()
        .sort_by(Reverse(ILLUMINA_ORDER))
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();
    for rec in &input {
        reverse_sequence.push(rec.clone())?;
    }
    let reverse_sequence = reverse_sequence.finish()?.collect::<Result<Vec<_>, _>>()?;
    println!("reverse sequence:");
    for r in &reverse_sequence {
        println!("  {}", String::from_utf8_lossy(r.sequence()));
    }

    Ok(())
}
