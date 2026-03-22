//! Serialization traits for reading and writing items to disk.
//!
//! [`Codec`] defines how items are written to and read from
//! temporary files during the sort. [`KeyedCodec`] is an optional
//! extension for formats that can store a precomputed sort key
//! alongside each record, enabling the merge engine to compare
//! keys without deserializing full records.

use std::io::{Read, Write};

/// Serialize and deserialize items to and from temporary files.
///
/// A codec defines the on-disk format for sorted runs. The core
/// crate provides no built-in codecs — implementations live in
/// domain crates (e.g., `spillover-bio` provides a dryice-based
/// codec for sequence records) or in application code for simple
/// cases like fixed-width binary tuples.
///
/// ```ignore
/// // A minimal codec for (u64, i32) pairs.
/// struct PairCodec;
///
/// impl Codec<(u64, i32)> for PairCodec {
///     type Error = std::io::Error;
///
///     fn write(&self, item: &(u64, i32), w: &mut impl Write) -> Result<(), Self::Error> {
///         w.write_all(&item.0.to_le_bytes())?;
///         w.write_all(&item.1.to_le_bytes())?;
///         Ok(())
///     }
///
///     fn read(&self, r: &mut impl Read) -> Result<Option<(u64, i32)>, Self::Error> {
///         let mut buf = [0u8; 8];
///         match r.read(&mut buf[..1]) {
///             Ok(0) => return Ok(None),
///             Ok(_) => r.read_exact(&mut buf[1..])?,
///             Err(e) => return Err(e),
///         }
///         let key = u64::from_le_bytes(buf);
///         let mut vbuf = [0u8; 4];
///         r.read_exact(&mut vbuf)?;
///         let val = i32::from_le_bytes(vbuf);
///         Ok(Some((key, val)))
///     }
/// }
/// ```
pub trait Codec<T> {
    /// The error type for encode/decode failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Encode an item and write it to `writer`.
    ///
    /// # Errors
    ///
    /// Returns the codec's error type if encoding or writing fails.
    fn write(&self, item: &T, writer: &mut impl Write) -> Result<(), Self::Error>;

    /// Decode the next item from `reader`, or return `None` at
    /// clean EOF. A partial read (some bytes but not a complete
    /// item) should return an error, not `None`.
    ///
    /// # Errors
    ///
    /// Returns the codec's error type if decoding fails or the
    /// stream contains a partial record.
    fn read(&self, reader: &mut impl Read) -> Result<Option<T>, Self::Error>;
}

/// Extension trait for codecs that store precomputed sort keys
/// alongside records.
///
/// During the k-way merge, this allows the heap to compare small
/// fixed-width keys without deserializing full records — only
/// the winning record is deserialized on each merge step. Any
/// temp format that stores keys alongside records can implement
/// this; it is not coupled to any particular format.
///
/// The merge engine selects between the base [`Codec`] path
/// (deserialize everything, extract key, compare) and the
/// `KeyedCodec` fast path (read key only, compare, deserialize
/// winner) at compile time based on trait bounds.
pub trait KeyedCodec<T>: Codec<T> {
    /// The precomputed key stored alongside each record. `Clone`
    /// is required because the merge heap holds keys independently
    /// of reader state.
    type Key: Clone;

    /// A stateful reader that can retrieve keys and records
    /// independently. The GAT allows the reader to borrow from
    /// the codec or the underlying byte source.
    type Reader<'a>: KeyedReader<T, Self::Key, Error = Self::Error>
    where
        Self: 'a;

    /// Write a record with its precomputed key.
    ///
    /// # Errors
    ///
    /// Returns the codec's error type if encoding or writing fails.
    fn write_keyed(
        &self,
        item: &T,
        key: &Self::Key,
        writer: &mut impl Write,
    ) -> Result<(), Self::Error>;

    /// Open a keyed reader over a byte source.
    ///
    /// # Errors
    ///
    /// Returns the codec's error type if initializing the reader
    /// fails (e.g., reading a file header).
    fn keyed_reader<'a, R: Read + 'a>(&self, reader: R) -> Result<Self::Reader<'a>, Self::Error>;
}

/// Stateful reader that separates key access from full record
/// deserialization.
///
/// The merge engine calls [`next_key`](Self::next_key) to advance
/// the reader and feed the heap, then
/// [`current_record`](Self::current_record) only for the merge
/// winner. This avoids deserializing records that lose the heap
/// comparison.
pub trait KeyedReader<T, K> {
    /// The error type, which must match the parent codec's error.
    type Error;

    /// Advance to the next entry and return its key. Returns
    /// `None` at clean EOF.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or decoding the key fails.
    fn next_key(&mut self) -> Result<Option<K>, Self::Error>;

    /// Retrieve the full record at the current position. Only
    /// valid after [`next_key`](Self::next_key) returned `Some`.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or decoding the record fails.
    fn current_record(&mut self) -> Result<T, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct U64Codec;

    impl Codec<u64> for U64Codec {
        type Error = std::io::Error;

        fn write(&self, item: &u64, writer: &mut impl Write) -> Result<(), Self::Error> {
            writer.write_all(&item.to_le_bytes())
        }

        fn read(&self, reader: &mut impl Read) -> Result<Option<u64>, Self::Error> {
            let mut buf = [0u8; 8];
            match reader.read(&mut buf[..1]) {
                Ok(0) => Ok(None),
                Ok(_) => {
                    reader.read_exact(&mut buf[1..])?;
                    Ok(Some(u64::from_le_bytes(buf)))
                }
                Err(e) => Err(e),
            }
        }
    }

    #[test]
    fn codec_round_trips_single_item() {
        let mut buf = Vec::new();
        U64Codec
            .write(&42u64, &mut buf)
            .expect("write should succeed");
        assert_eq!(buf.len(), 8, "u64 should write exactly 8 bytes");

        let mut cursor = std::io::Cursor::new(&buf);
        let item = U64Codec
            .read(&mut cursor)
            .expect("read should succeed")
            .expect("should find one item");
        assert_eq!(item, 42, "round-tripped value should match");
    }

    #[test]
    fn codec_round_trips_multiple_items() {
        let values = vec![1u64, 2, 3, u64::MAX, 0];
        let mut buf = Vec::new();
        for v in &values {
            U64Codec.write(v, &mut buf).expect("write should succeed");
        }

        let mut cursor = std::io::Cursor::new(&buf);
        let mut recovered = Vec::new();
        while let Some(v) = U64Codec.read(&mut cursor).expect("read should succeed") {
            recovered.push(v);
        }

        assert_eq!(
            recovered, values,
            "all round-tripped values should match in order"
        );
    }

    #[test]
    fn codec_read_empty_returns_none() {
        let buf: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&buf);
        let result = U64Codec
            .read(&mut cursor)
            .expect("reading empty should not error");
        assert!(
            result.is_none(),
            "reading from an empty source should return None"
        );
    }

    #[test]
    fn codec_read_truncated_returns_error() {
        let buf = vec![0u8; 3]; // less than 8 bytes
        let mut cursor = std::io::Cursor::new(&buf);
        let result = U64Codec.read(&mut cursor);
        assert!(
            result.is_err(),
            "reading a partial record should return an error, not None"
        );
    }
}
