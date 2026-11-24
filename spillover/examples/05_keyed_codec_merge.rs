use std::io::{BufReader, BufWriter, Read, Write};

use spillover::{
    codec::{
        Codec, CodecCursor, DeriveKey, EncodeKeyedRun, KeyedCodec, KeyedCodecCursor,
        KeyedCodecWriter, TakeOwned,
    },
    key::Owned,
    sorter::Builder,
};

#[derive(Clone, Copy)]
struct DecadeKeyedCodec;

struct Writer<W: Write> {
    inner: BufWriter<W>,
}

impl<W: Write> KeyedCodecWriter<u64, u8> for Writer<W> {
    type Error = std::io::Error;

    fn write_keyed(&mut self, entry: spillover::key::Keyed<&u64, &u8>) -> Result<(), Self::Error> {
        let item = *entry.item();
        let key = *entry.stored_key();
        self.inner.write_all(&[*key])?;
        self.inner.write_all(&item.to_le_bytes())
    }

    fn finish(mut self) -> Result<(), Self::Error> {
        self.inner.flush()
    }
}

struct Reader<R: Read> {
    inner: BufReader<R>,
    current_key: Option<u8>,
    current: Option<u64>,
}

impl<R: Read> CodecCursor for Reader<R> {
    type Error = std::io::Error;
    type Record = u64;

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

    fn with_current<T>(&mut self, f: impl FnOnce(&Self::Record) -> T) -> Result<T, Self::Error> {
        let record = self
            .current
            .as_ref()
            .ok_or_else(|| std::io::Error::other("current called before advance"))?;
        Ok(f(record))
    }
}

impl<R: Read> TakeOwned for Reader<R> {
    type Owned = u64;

    fn take_owned(&mut self) -> Result<Option<Self::Owned>, Self::Error> {
        Ok(self.current.take())
    }
}

impl<R: Read> KeyedCodecCursor<u8> for Reader<R> {
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
    type Error = std::io::Error;
    type Cursor<R: Read> = Reader<R>;

    fn cursor<R: Read>(&self, source: R) -> Result<Self::Cursor<R>, Self::Error> {
        Ok(Reader {
            inner: BufReader::new(source),
            current_key: None,
            current: None,
        })
    }
}

impl KeyedCodec for DecadeKeyedCodec {
    type Key = u8;
    type KeyedCursor<R: Read> = Reader<R>;

    fn keyed_cursor<R: Read>(&self, source: R) -> Result<Self::KeyedCursor<R>, Self::Error> {
        self.cursor(source)
    }
}

impl EncodeKeyedRun<u64> for DecadeKeyedCodec {
    type KeyedWriter<W: Write> = Writer<W>;

    fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
        Writer {
            inner: BufWriter::new(dest),
        }
    }
}

impl DeriveKey<u64> for DecadeKeyedCodec {
    fn derive_key(&self, item: &u64) -> Self::Key {
        u8::try_from(*item / 10).expect("example values fit in u8 decade key")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .key(Owned(|v: &u64| *v))
        .keyed_codec(DecadeKeyedCodec)
        .max_buffer_items(2)
        .build();

    for value in [19_u64, 11, 25, 12, 18, 26] {
        sorter.push(value)?;
    }

    let output_stream = sorter.finish()?;
    let collected: Result<Vec<u64>, _> = output_stream.collect();
    let output = collected?;
    println!("keyed merge output: {output:?}");

    Ok(())
}
