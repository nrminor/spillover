use std::io::{BufReader, BufWriter, Read, Write};

use spillover::{
    codec::{Codec, CodecReader, CodecWriter},
    dedup::AdjacentDedup,
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
}

impl<R: Read> CodecReader<u64> for U64Reader<R> {
    type Error = std::io::Error;

    fn read(&mut self) -> Result<Option<u64>, Self::Error> {
        let mut bytes = [0_u8; 8];
        match self.inner.read_exact(&mut bytes) {
            Ok(()) => Ok(Some(u64::from_le_bytes(bytes))),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl Codec<u64> for U64Codec {
    type Error = std::io::Error;
    type Writer<W: Write> = U64Writer<W>;
    type Reader<R: Read> = U64Reader<R>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        U64Writer {
            inner: BufWriter::new(dest),
        }
    }

    fn reader<R: Read>(&self, source: R) -> Self::Reader<R> {
        U64Reader {
            inner: BufReader::new(source),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .key(Owned(|v: &u64| *v))
        .codec(U64Codec)
        .dedup(AdjacentDedup::new(|a: &u64, b: &u64| a == b))
        .max_buffer_items::<u64>(3)
        .build();

    for value in [3_u64, 3, 2, 2, 2, 1, 1, 4, 4] {
        sorter.push(value)?;
    }

    let output: Vec<u64> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    println!("deduplicated sorted values: {output:?}");

    Ok(())
}
