use std::io::{BufReader, BufWriter, Read, Write};

use spillover::{
    codec::{Codec, CodecCursor, CodecWriter, KeyedCodec, KeyedCodecCursor, KeyedCodecWriter},
    key::Owned,
    sorter::Builder,
};

#[derive(Clone, Copy)]
struct DecadeKeyedCodec;

struct Writer<W: Write> {
    inner: BufWriter<W>,
}

impl<W: Write> CodecWriter<u64> for Writer<W> {
    type Error = std::io::Error;

    fn write(&mut self, item: &u64) -> Result<(), Self::Error> {
        let key = u8::try_from(*item / 10).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("decade key overflow: {e}"),
            )
        })?;
        self.inner.write_all(&[key])?;
        self.inner.write_all(&item.to_le_bytes())
    }

    fn finish(mut self) -> Result<(), Self::Error> {
        self.inner.flush()
    }
}

impl<W: Write> KeyedCodecWriter<u64, u8> for Writer<W> {
    type Error = std::io::Error;

    fn write_keyed(&mut self, item: &u64, key: &u8) -> Result<(), Self::Error> {
        self.inner.write_all(&[*key])?;
        self.inner.write_all(&item.to_le_bytes())
    }

    fn finish(self) -> Result<(), Self::Error> {
        CodecWriter::finish(self)
    }
}

struct Reader<R: Read> {
    inner: BufReader<R>,
    current_key: Option<u8>,
    current: Option<u64>,
}

impl<R: Read> CodecCursor<u64> for Reader<R> {
    type Error = std::io::Error;
    type Current<'a>
        = u64
    where
        Self: 'a;

    fn advance(&mut self) -> Result<bool, Self::Error> {
        let mut key = [0_u8; 1];
        match self.inner.read(&mut key) {
            Ok(0) => {
                self.current_key = None;
                self.current = None;
                Ok(false)
            }
            Ok(_) => {
                let mut bytes = [0_u8; 8];
                self.inner.read_exact(&mut bytes)?;
                self.current_key = Some(key[0]);
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

impl<R: Read> KeyedCodecCursor<u64, u8> for Reader<R> {
    fn current_key(&self) -> Result<u8, Self::Error> {
        self.current_key.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "current_key called before advance",
            )
        })
    }
}

impl Codec for DecadeKeyedCodec {
    type Item = u64;
    type Error = std::io::Error;
    type Writer<W: Write> = Writer<W>;
    type Cursor<R: Read> = Reader<R>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        Writer {
            inner: BufWriter::new(dest),
        }
    }

    fn cursor<R: Read>(&self, source: R) -> Self::Cursor<R> {
        Reader {
            inner: BufReader::new(source),
            current_key: None,
            current: None,
        }
    }
}

impl KeyedCodec for DecadeKeyedCodec {
    type Key = u8;
    type KeyedWriter<W: Write> = Writer<W>;
    type KeyedCursor<R: Read> = Reader<R>;

    fn derive_key(&self, item: &u64) -> Self::Key {
        u8::try_from(*item / 10).expect("example values fit in u8 decade key")
    }

    fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
        self.writer(dest)
    }

    fn keyed_cursor<R: Read>(&self, source: R) -> Self::KeyedCursor<R> {
        self.cursor(source)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .key(Owned(|v: &u64| *v))
        .keyed_codec(DecadeKeyedCodec)
        .max_buffer_items::<u64>(2)
        .build();

    for value in [19_u64, 11, 25, 12, 18, 26] {
        sorter.push(value)?;
    }

    let output: Vec<u64> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    println!("keyed merge output: {output:?}");

    Ok(())
}
