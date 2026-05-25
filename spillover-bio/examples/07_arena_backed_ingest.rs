use spillover_bio::{
    codec::DryIceCodec,
    record::{SeqRecord, SeqRecordArena, SeqRecordView},
    sort::Builder,
};

struct ParserRecord<'a> {
    name: &'a [u8],
    sequence: &'a [u8],
    quality: &'a [u8],
}

impl ParserRecord<'_> {
    fn as_view(&self) -> SeqRecordView<'_> {
        SeqRecordView::new(self.name, self.sequence, self.quality)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arena = SeqRecordArena::new();

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .arena(&mut arena)
        .measured_budget(64 * 1024)
        .build();

    // A real parser often reuses its own input buffers. Arena-backed ingest still
    // copies each record, but it copies into sorter-owned storage whose allocation
    // can be reused across spill windows.
    for record in [
        ParserRecord {
            name: b"r1",
            sequence: b"TTTTTTTT",
            quality: b"IIIIIIII",
        },
        ParserRecord {
            name: b"r2",
            sequence: b"AAAAAAAA",
            quality: b"!!!!!!!!",
        },
        ParserRecord {
            name: b"r3",
            sequence: b"CCCCCCCC",
            quality: b"########",
        },
    ] {
        sorter.push(&record.as_view())?;
    }

    let sorted_records: Vec<SeqRecord> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;

    assert!(
        arena.is_empty(),
        "finish releases the active arena contents"
    );

    for record in sorted_records {
        println!(
            "{}\t{}\t{}",
            String::from_utf8_lossy(record.name()),
            String::from_utf8_lossy(record.sequence()),
            String::from_utf8_lossy(record.quality())
        );
    }

    Ok(())
}
