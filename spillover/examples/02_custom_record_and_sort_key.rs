use std::io::{BufRead, BufReader, BufWriter, Write};

use spillover::{
    codec::{Codec, CodecReader, CodecWriter},
    sorter::Builder,
};

#[derive(Debug, Clone)]
struct LogEvent {
    timestamp: u64,
    level: u8,
    message: String,
}

#[derive(Clone, Copy)]
struct LogCodec;

struct LogWriter<W: Write> {
    inner: BufWriter<W>,
}

impl<W: Write> CodecWriter<LogEvent> for LogWriter<W> {
    type Error = std::io::Error;

    fn write(&mut self, item: &LogEvent) -> Result<(), Self::Error> {
        writeln!(
            self.inner,
            "{}\t{}\t{}",
            item.timestamp, item.level, item.message
        )
    }

    fn finish(mut self) -> Result<(), Self::Error> {
        self.inner.flush()
    }
}

struct LogReader<R: std::io::Read> {
    inner: BufReader<R>,
    line: String,
}

impl<R: std::io::Read> CodecReader<LogEvent> for LogReader<R> {
    type Error = std::io::Error;

    fn read(&mut self) -> Result<Option<LogEvent>, Self::Error> {
        self.line.clear();
        let n = self.inner.read_line(&mut self.line)?;
        if n == 0 {
            return Ok(None);
        }

        let mut parts = self.line.trim_end().splitn(3, '\t');
        let ts = parts
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing ts"))?;
        let level = parts
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing level"))?;
        let msg = parts.next().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing message")
        })?;

        let timestamp = ts.parse::<u64>().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("invalid ts: {e}"))
        })?;
        let level = level.parse::<u8>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid level: {e}"),
            )
        })?;

        Ok(Some(LogEvent {
            timestamp,
            level,
            message: msg.to_string(),
        }))
    }
}

impl Codec<LogEvent> for LogCodec {
    type Error = std::io::Error;
    type Writer<W: Write> = LogWriter<W>;
    type Reader<R: std::io::Read> = LogReader<R>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        LogWriter {
            inner: BufWriter::new(dest),
        }
    }

    fn reader<R: std::io::Read>(&self, source: R) -> Self::Reader<R> {
        LogReader {
            inner: BufReader::new(source),
            line: String::new(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .key(spillover::key::Owned(|e: &LogEvent| (e.level, e.timestamp)))
        .codec(LogCodec)
        .max_buffer_items::<LogEvent>(2)
        .build();

    let input = [
        LogEvent {
            timestamp: 103,
            level: 3,
            message: "retrying index write".to_string(),
        },
        LogEvent {
            timestamp: 100,
            level: 1,
            message: "starting pipeline".to_string(),
        },
        LogEvent {
            timestamp: 101,
            level: 1,
            message: "loaded references".to_string(),
        },
        LogEvent {
            timestamp: 102,
            level: 2,
            message: "spilled first run".to_string(),
        },
    ];

    for event in input {
        sorter.push(event)?;
    }

    let output: Vec<LogEvent> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    for event in output {
        println!("L{} @ {}: {}", event.level, event.timestamp, event.message);
    }

    Ok(())
}
