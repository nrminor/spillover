use std::time::Instant;

use spillover_bio::{
    codec::DryIceCodec,
    record::SeqRecord,
    sort::{Builder, ILLUMINA_ORDER},
};

fn generate_records(count: usize, len: usize) -> Vec<SeqRecord> {
    let mut seed: u64 = 0xD0E5_F17E_CAFE_BABE;
    let mut out = Vec::with_capacity(count);

    for idx in 0..count {
        let mut seq = Vec::with_capacity(len);
        let mut qual = Vec::with_capacity(len);

        for _ in 0..len {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let base = match seed & 3 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            };
            seq.push(base);

            let q = b'!' + u8::try_from((seed >> 8) % 40).expect("quality in range");
            qual.push(q);
        }

        out.push(SeqRecord::new(
            format!("read_{idx:06}").into_bytes(),
            seq,
            qual,
        ));
    }

    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let records = generate_records(4_000, 150);

    let keyed_start = Instant::now();
    let mut keyed = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(256)
        .build();
    for rec in &records {
        keyed.push(rec.clone())?;
    }
    let keyed_out: Vec<SeqRecord> = keyed.finish()?.collect::<Result<Vec<_>, _>>()?;
    let keyed_elapsed = keyed_start.elapsed();

    let unkeyed_start = Instant::now();
    let mut unkeyed = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .max_buffer_items(256)
        .build();
    for rec in &records {
        unkeyed.push(rec.clone())?;
    }
    let unkeyed_out: Vec<SeqRecord> = unkeyed.finish()?.collect::<Result<Vec<_>, _>>()?;
    let unkeyed_elapsed = unkeyed_start.elapsed();

    assert_eq!(keyed_out.len(), unkeyed_out.len());
    println!("records: {}", keyed_out.len());
    println!("keyed elapsed:   {keyed_elapsed:?}");
    println!("unkeyed elapsed: {unkeyed_elapsed:?}");
    println!(
        "same first sequence: {}",
        keyed_out.first().map(SeqRecord::sequence) == unkeyed_out.first().map(SeqRecord::sequence)
    );

    Ok(())
}
