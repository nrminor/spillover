use std::io::{BufReader, BufWriter, Read, Write};

use spillover::{
    codec::{Codec, CodecReader, CodecWriter, KeyedCodec, KeyedCodecReader, KeyedCodecWriter},
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
    current: Option<u64>,
}

impl<R: Read> CodecReader<u64> for Reader<R> {
    type Error = std::io::Error;

    fn read(&mut self) -> Result<Option<u64>, Self::Error> {
        let mut key = [0_u8; 1];
        match self.inner.read_exact(&mut key) {
            Ok(()) => {
                let mut bytes = [0_u8; 8];
                self.inner.read_exact(&mut bytes)?;
                Ok(Some(u64::from_le_bytes(bytes)))
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl<R: Read> KeyedCodecReader<u64, u8> for Reader<R> {
    type Error = std::io::Error;

    fn next_key(&mut self) -> Result<Option<u8>, Self::Error> {
        let mut key = [0_u8; 1];
        match self.inner.read_exact(&mut key) {
            Ok(()) => {
                let mut bytes = [0_u8; 8];
                self.inner.read_exact(&mut bytes)?;
                self.current = Some(u64::from_le_bytes(bytes));
                Ok(Some(key[0]))
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.current = None;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    fn current_record(&mut self) -> Result<u64, Self::Error> {
        self.current.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "current_record called before next_key",
            )
        })
    }
}

impl Codec for DecadeKeyedCodec {
    type Item = u64;
    type Error = std::io::Error;
    type Writer<W: Write> = Writer<W>;
    type Reader<R: Read> = Reader<R>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        Writer {
            inner: BufWriter::new(dest),
        }
    }

    fn reader<R: Read>(&self, source: R) -> Self::Reader<R> {
        Reader {
            inner: BufReader::new(source),
            current: None,
        }
    }
}

impl KeyedCodec for DecadeKeyedCodec {
    type Key = u8;
    type KeyedWriter<W: Write> = Writer<W>;
    type KeyedReader<R: Read> = Reader<R>;

    fn derive_key(&self, item: &u64) -> Self::Key {
        u8::try_from(*item / 10).expect("example values fit in u8 decade key")
    }

    fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
        self.writer(dest)
    }

    fn keyed_reader<R: Read>(&self, source: R) -> Self::KeyedReader<R> {
        self.reader(source)
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
