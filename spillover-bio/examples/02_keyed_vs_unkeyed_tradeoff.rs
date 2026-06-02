use std::time::Instant;

use spillover_bio::{codec::DryIceCodec, record::SeqRecord, sort::Builder};

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

    // `sort_by_illumina()` is the high-throughput path for ordinary Illumina
    // reads. It uses a compact packed sequence key and radix chunk sorting, then
    // refines ties with the full sequence/quality comparator.
    let illumina_start = Instant::now();
    let mut illumina = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(256)
        .build();
    for rec in &records {
        illumina.push(rec.clone())?;
    }
    let illumina_out: Vec<SeqRecord> = illumina.finish()?.collect::<Result<Vec<_>, _>>()?;
    let illumina_elapsed = illumina_start.elapsed();

    // `sort_by_sequence_exact()` is the boring, literal order: raw sequence bytes
    // followed by raw quality bytes. It is the right choice when exact byte
    // lexicographic ordering matters, but it uses the full-record unkeyed path and
    // sequential comparison sorting, so it may be slower on large spill-heavy
    // Illumina datasets than the compact keyed path above.
    let exact_start = Instant::now();
    let mut exact = Builder::new()
        .sort_by_sequence_exact()
        .codec(DryIceCodec::new())
        .max_buffer_items(256)
        .build();
    for rec in &records {
        exact.push(rec.clone())?;
    }
    let exact_out: Vec<SeqRecord> = exact.finish()?.collect::<Result<Vec<_>, _>>()?;
    let exact_elapsed = exact_start.elapsed();

    assert_eq!(illumina_out.len(), exact_out.len());
    println!("records: {}", illumina_out.len());
    println!("illumina elapsed: {illumina_elapsed:?}");
    println!("exact elapsed:    {exact_elapsed:?}");
    println!(
        "same first sequence: {}",
        illumina_out.first().map(SeqRecord::sequence) == exact_out.first().map(SeqRecord::sequence)
    );

    Ok(())
}
