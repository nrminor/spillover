use std::{
    cmp::Ordering,
    io::{self, BufRead, BufReader, BufWriter, Lines, Read, Write},
    iter::Map,
};

use spillover::{
    codec::{CodecWriter, DecodeRun, EncodeRun},
    compare::{Compare, Reverse},
    key::SortKey,
    sorter::Builder,
};

#[derive(Clone, Copy)]
struct LineCodec;

struct LineWriter<W: Write> {
    inner: BufWriter<W>,
}

impl<W: Write> CodecWriter<String> for LineWriter<W> {
    type Error = std::io::Error;

    fn write(&mut self, item: &String) -> Result<(), Self::Error> {
        writeln!(self.inner, "{item}")
    }

    fn finish(mut self) -> Result<(), Self::Error> {
        self.inner.flush()
    }
}

fn trim_record(record: io::Result<String>) -> io::Result<String> {
    let mut record = record?;
    let retained_length = record.trim_end().len();
    record.truncate(retained_length);
    Ok(record)
}

impl DecodeRun for LineCodec {
    type Record = String;
    type Error = std::io::Error;
    type Reader<R: Read> = Map<Lines<BufReader<R>>, fn(io::Result<String>) -> io::Result<String>>;

    fn reader<R: Read>(&self, source: R) -> Result<Self::Reader<R>, Self::Error> {
        let trim: fn(io::Result<String>) -> io::Result<String> = trim_record;
        let buffered = BufReader::new(source);
        let lines = buffered.lines();
        let records = lines.map(trim);
        Ok(records)
    }
}

impl EncodeRun<String> for LineCodec {
    type Writer<W: Write> = LineWriter<W>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        LineWriter {
            inner: BufWriter::new(dest),
        }
    }
}

#[derive(Clone, Copy)]
struct AsciiCaseInsensitive;

impl Compare<str> for AsciiCaseInsensitive {
    fn compare(&self, a: &str, b: &str) -> Ordering {
        a.to_ascii_lowercase().cmp(&b.to_ascii_lowercase())
    }
}

impl Compare<&str> for AsciiCaseInsensitive {
    fn compare(&self, a: &&str, b: &&str) -> Ordering {
        a.to_ascii_lowercase().cmp(&b.to_ascii_lowercase())
    }
}

#[derive(Clone, Copy)]
struct AsStrKey;

impl SortKey<String> for AsStrKey {
    type Key<'a>
        = &'a str
    where
        String: 'a;

    fn key<'a>(&self, item: &'a String) -> Self::Key<'a> {
        item.as_str()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .key(AsStrKey)
        .compare(Reverse(AsciiCaseInsensitive))
        .codec(LineCodec)
        .max_buffer_items(2)
        .build();

    for name in ["zeta", "Alpha", "beta", "Gamma", "delta"] {
        sorter.push(name.to_string())?;
    }

    let output_stream = sorter.finish()?;
    let collected: Result<Vec<String>, _> = output_stream.collect();
    let output = collected?;
    println!("case-insensitive descending: {output:?}");

    Ok(())
}
