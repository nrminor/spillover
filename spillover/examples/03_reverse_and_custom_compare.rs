use std::cmp::Ordering;
use std::io::{BufRead, BufReader, BufWriter, Write};

use spillover::{
    codec::{Codec, CodecReader, CodecWriter},
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
    line: String,
}

impl<R: std::io::Read> CodecReader<String> for LineReader<R> {
    type Error = std::io::Error;

    fn read(&mut self) -> Result<Option<String>, Self::Error> {
        self.line.clear();
        let n = self.inner.read_line(&mut self.line)?;
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(self.line.trim_end().to_string()))
    }
}

impl Codec<String> for LineCodec {
    type Error = std::io::Error;
    type Writer<W: Write> = LineWriter<W>;
    type Reader<R: std::io::Read> = LineReader<R>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        LineWriter {
            inner: BufWriter::new(dest),
        }
    }

    fn reader<R: std::io::Read>(&self, source: R) -> Self::Reader<R> {
        LineReader {
            inner: BufReader::new(source),
            line: String::new(),
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
        .max_buffer_items::<String>(2)
        .build();

    for name in ["zeta", "Alpha", "beta", "Gamma", "delta"] {
        sorter.push(name.to_string())?;
    }

    let output: Vec<String> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    println!("case-insensitive descending: {output:?}");

    Ok(())
}
