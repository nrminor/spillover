use std::io::{BufReader, BufWriter, Read, Write};

use spillover::{
    codec::{Codec, CodecCursor, CodecWriter},
    key::Owned,
    sorter::Builder,
};

#[derive(Clone, Copy)]
struct U64Codec;

struct U64Writer<W: Write> {
    inner: BufWriter<W>,
}

impl<W: Write> CodecWriter<u64> for U64Writer<W> {
    type Error = std::io::Error;

    fn write(&mut self, item: &u64) -> Result<(), Self::Error> {
        self.inner.write_all(&item.to_le_bytes())
    }

    fn finish(mut self) -> Result<(), Self::Error> {
        self.inner.flush()
    }
}

struct U64Reader<R: Read> {
    inner: BufReader<R>,
    current: Option<u64>,
}

impl<R: Read> CodecCursor<u64> for U64Reader<R> {
    type Error = std::io::Error;
    type Current<'a>
        = u64
    where
        Self: 'a;

    fn advance(&mut self) -> Result<bool, Self::Error> {
        let mut bytes = [0_u8; 8];
        match self.inner.read(&mut bytes[..1]) {
            Ok(0) => {
                self.current = None;
                Ok(false)
            }
            Ok(_) => {
                self.inner.read_exact(&mut bytes[1..])?;
                self.current = Some(u64::from_le_bytes(bytes));
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }

    fn current(&mut self) -> Result<u64, Self::Error> {
        self.current
            .ok_or_else(|| std::io::Error::other("current called before advance"))
    }

    fn with_current<'a, T>(
        &'a mut self,
        f: impl FnOnce(Self::Current<'a>) -> T,
    ) -> Result<T, Self::Error> {
        self.current().map(f)
    }
}

impl Codec for U64Codec {
    type Item = u64;
    type Error = std::io::Error;
    type Writer<W: Write> = U64Writer<W>;
    type Cursor<R: Read> = U64Reader<R>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        U64Writer {
            inner: BufWriter::new(dest),
        }
    }

    fn cursor<R: Read>(&self, source: R) -> Self::Cursor<R> {
        U64Reader {
            inner: BufReader::new(source),
            current: None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .key(Owned(|v: &u64| *v))
        .codec(U64Codec)
        .max_buffer_items::<u64>(3)
        .build();

    for value in [42_u64, 7, 9, 1, 100, 3] {
        sorter.push(value)?;
    }

    let output: Vec<u64> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    println!("sorted values: {output:?}");

    Ok(())
}
