use std::cmp::Ordering;
use std::io::{BufRead, BufReader, BufWriter, Write};

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

struct LineReader<R: std::io::Read> {
    inner: BufReader<R>,
}

impl<R: std::io::Read> Iterator for LineReader<R> {
    type Item = std::io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.inner.read_line(&mut line) {
            Ok(0) => None,
            Ok(_) => {
                let content_length = line.trim_end().len();
                line.truncate(content_length);
                Some(Ok(line))
            }
            Err(error) => Some(Err(error)),
        }
    }
}

impl DecodeRun for LineCodec {
    type Record = String;
    type Error = std::io::Error;
    type Reader<R: std::io::Read> = LineReader<R>;

    fn reader<R: std::io::Read>(&self, source: R) -> Result<Self::Reader<R>, Self::Error> {
        Ok(LineReader {
            inner: BufReader::new(source),
        })
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
