use std::{
    io::{BufReader, BufWriter, Read, Write},
    num::NonZeroUsize,
};

use spillover::{
    codec::{Codec, CodecReader, CodecWriter},
    key::Owned,
    merge::MergeConfig,
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

impl<R: Read> CodecReader<u64> for U64Reader<R> {
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
    type Reader<R: Read> = U64Reader<R>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        U64Writer {
            inner: BufWriter::new(dest),
        }
    }

    fn reader<R: Read>(&self, source: R) -> Self::Reader<R> {
        U64Reader {
            inner: BufReader::new(source),
            current: None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut merge_config = MergeConfig::default();
    merge_config.max_fan_in = NonZeroUsize::new(2).expect("2 is non-zero");

    let mut sorter = Builder::new()
        .key(Owned(|v: &u64| *v))
        .codec(U64Codec)
        .max_buffer_items::<u64>(2)
        .merge_config(merge_config)
        .build();

    for value in [10_u64, 3, 7, 5, 1, 8, 9, 2, 4, 6] {
        sorter.push(value)?;
    }

    let output: Vec<u64> = sorter.finish()?.collect::<Result<Vec<_>, _>>()?;
    println!(
        "sorted {} values with forced small buffer + fan-in: {output:?}",
        output.len()
    );

    Ok(())
}
