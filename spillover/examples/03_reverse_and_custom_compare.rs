use std::cmp::Ordering;
use std::io::{BufRead, BufReader, BufWriter, Write};

use spillover::{
    codec::{Codec, CodecCursor, CodecWriter},
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
    current: Option<String>,
}

impl<R: std::io::Read> CodecCursor<String> for LineReader<R> {
    type Error = std::io::Error;
    type Current<'a>
        = &'a str
    where
        Self: 'a;

    fn advance(&mut self) -> Result<bool, Self::Error> {
        self.line.clear();
        let n = self.inner.read_line(&mut self.line)?;
        if n == 0 {
            self.current = None;
            return Ok(false);
        }
        self.current = Some(self.line.trim_end().to_string());
        Ok(true)
    }

    fn current(&mut self) -> Result<String, Self::Error> {
        self.current
            .take()
            .ok_or_else(|| std::io::Error::other("current called before advance"))
    }

    fn with_current<'a, T>(
        &'a mut self,
        f: impl FnOnce(Self::Current<'a>) -> T,
    ) -> Result<T, Self::Error> {
        match self.current.as_deref() {
            Some(current) => Ok(f(current)),
            None => Err(std::io::Error::other("current called before advance")),
        }
    }
}

impl Codec for LineCodec {
    type Item = String;
    type Error = std::io::Error;
    type Writer<W: Write> = LineWriter<W>;
    type Cursor<R: std::io::Read> = LineReader<R>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        LineWriter {
            inner: BufWriter::new(dest),
        }
    }

    fn cursor<R: std::io::Read>(&self, source: R) -> Self::Cursor<R> {
        LineReader {
            inner: BufReader::new(source),
            line: String::new(),
            current: None,
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
