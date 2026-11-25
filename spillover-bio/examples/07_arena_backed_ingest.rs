use std::io::{self, Write};

use spillover_bio::{
    codec::DryIceCodec,
    record::{SeqRecordArena, SeqRecordParts, SeqRecordView},
    sort::{Builder, SeqRecordSink},
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

struct TabSeparated<W> {
    writer: W,
}

impl<W: Write> SeqRecordSink for TabSeparated<W> {
    type Error = io::Error;

    fn write_record<R: SeqRecordParts + ?Sized>(&mut self, record: &R) -> io::Result<()> {
        self.writer.write_all(record.name())?;
        self.writer.write_all(b"\t")?;
        self.writer.write_all(record.sequence())?;
        self.writer.write_all(b"\t")?;
        self.writer.write_all(record.quality())?;
        self.writer.write_all(b"\n")
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
    // copies each record into caller-provided arena storage whose allocation
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

    let sorted_records = sorter.finish()?;

    // The output owns its runs, so the arena is available before consumption.
    assert!(
        arena.is_empty(),
        "finish releases the active arena contents"
    );

    let stdout = io::stdout();
    let mut sink = TabSeparated {
        writer: stdout.lock(),
    };
    sorted_records.write_to(&mut sink)?;

    Ok(())
}
