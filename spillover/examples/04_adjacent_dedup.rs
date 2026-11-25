use std::io::{BufReader, BufWriter, Read, Write};

use spillover::{
    codec::{CodecWriter, DecodeRun, EncodeRun},
    compare::Natural,
    dedup::AdjacentDedup,
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

impl<R: Read> Iterator for U64Reader<R> {
    type Item = std::io::Result<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut bytes = [0_u8; 8];
        match self.inner.read(&mut bytes[..1]) {
            Ok(0) => None,
            Ok(_) => {
                let decoded = self
                    .inner
                    .read_exact(&mut bytes[1..])
                    .map(|()| u64::from_le_bytes(bytes));
                Some(decoded)
            }
            Err(error) => Some(Err(error)),
        }
    }
}

impl DecodeRun for U64Codec {
    type Record = u64;
    type Error = std::io::Error;
    type Reader<R: Read> = U64Reader<R>;

    fn reader<R: Read>(&self, source: R) -> Result<Self::Reader<R>, Self::Error> {
        Ok(U64Reader {
            inner: BufReader::new(source),
        })
    }
}

impl EncodeRun<u64> for U64Codec {
    type Writer<W: Write> = U64Writer<W>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        U64Writer {
            inner: BufWriter::new(dest),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sorter = Builder::new()
        .order(Natural)
        .codec(U64Codec)
        .dedup(AdjacentDedup::new(|a: &u64, b: &u64| a == b))
        .max_buffer_items(3)
        .build();

    for value in [3_u64, 3, 2, 2, 2, 1, 1, 4, 4] {
        sorter.push(value)?;
    }

    let output_stream = sorter.finish()?;
    let collected: Result<Vec<u64>, _> = output_stream.collect();
    let output = collected?;
    assert_eq!(output, [1, 2, 3, 4]);
    println!("deduplicated sorted values: {output:?}");

    Ok(())
}
