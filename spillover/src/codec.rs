//! Serialization traits for reading and writing items to disk.
//!
//! [`Codec`] defines how items are written to and read from
//! temporary files during the sort. The codec itself is a
//! stateless `Copy` configuration object — it spawns stateful
//! [`CodecWriter`] and [`CodecCursor`] instances that handle
//! the actual I/O. This separation allows block-oriented
//! formats (like dryice) to manage internal buffering in the
//! writer while keeping the codec trivially duplicatable.
//!
//! [`KeyedCodec`] is an optional extension for formats that
//! can store a precomputed sort key alongside each record,
//! enabling the merge engine to compare keys without
//! deserializing full records.

use std::io::{Read, Write};

/// A stateful writer created by a [`Codec`] for writing items
/// to a single sorted run.
///
/// The writer may hold internal state (e.g., a partially filled
/// block for block-oriented formats). [`finish`](Self::finish)
/// must be called after the last item to flush any buffered data
/// and finalize the format.
pub trait CodecWriter<I: ?Sized> {
    /// The error type for write failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Write one item.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding or writing fails.
    fn write(&mut self, item: &I) -> Result<(), Self::Error>;

    /// Flush any buffered data and finalize. Must be called
    /// after the last item — block-oriented formats need this
    /// to write partial blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or finalizing fails.
    fn finish(self) -> Result<(), Self::Error>;
}

/// A stateful cursor created by a [`Codec`] for reading items
/// back from a sorted run.
pub trait CodecCursor<T> {
    /// The error type for read failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// The cheapest representation of the current item for this cursor.
    ///
    /// Cursors backed by reusable decode buffers can expose borrowed views here.
    /// Owned-only cursors may use owned values or references to internally stored
    /// owned values.
    type Current<'a>
    where
        Self: 'a;

    /// Advance to the next item, returning `false` at clean EOF.
    /// A partial read (some bytes but not a complete item) should return an
    /// error, not `false`.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails or the stream contains a partial
    /// record.
    fn advance(&mut self) -> Result<bool, Self::Error>;

    /// Materialize the current item as an owned value.
    ///
    /// Valid only after [`advance`](Self::advance) returned `true`. Repeated
    /// calls to `current()` and [`with_current`](Self::with_current) must keep
    /// observing the same item until the next call to [`advance`](Self::advance).
    /// This method may allocate, but it must not advance the cursor or consume
    /// the current position.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding or materialization fails.
    fn current(&mut self) -> Result<T, Self::Error>;

    /// Visit the current item in this cursor's cheapest representation.
    ///
    /// The value passed to the callback is valid only for the callback. This is
    /// valid only after [`advance`](Self::advance) returned `true`. Calling this
    /// method must not advance the cursor or consume the current position.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding or current-item access fails.
    fn with_current<'a, R>(
        &'a mut self,
        f: impl FnOnce(Self::Current<'a>) -> R,
    ) -> Result<R, Self::Error>;
}

/// Defines the on-disk format for sorted runs.
///
/// A codec is a stateless, `Copy` configuration object that
/// knows how to create writers and cursors for its format.
/// The writers and cursors hold whatever state the format
/// needs (I/O buffers, block accumulators, etc.).
///
/// The core crate provides no built-in codecs — implementations
/// live in domain crates (e.g., `spillover-bio` provides a
/// dryice-based codec) or in application code for simple cases.
pub trait Codec: Copy {
    /// The item type materialized by cursors for this codec.
    type Item;

    /// The error type for encode/decode failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// A stateful writer for encoding items into a sorted run.
    type Writer<W: Write>;

    /// A stateful cursor for decoding items from a sorted run.
    type Cursor<R: Read>: CodecCursor<Self::Item, Error = Self::Error>;

    /// Create a writer that encodes items into `dest`.
    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W>;

    /// Create a cursor that decodes items from `source`.
    fn cursor<R: Read>(&self, source: R) -> Self::Cursor<R>;
}

/// A stateful writer that stores items alongside precomputed keys.
///
/// Created by [`KeyedCodec::keyed_writer`]. Like [`CodecWriter`],
/// [`finish`](Self::finish) must be called after the last item.
pub trait KeyedCodecWriter<I: ?Sized, K> {
    /// The error type for write failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Write one item with its precomputed key.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding or writing fails.
    fn write_keyed(&mut self, item: &I, key: &K) -> Result<(), Self::Error>;

    /// Flush any buffered data and finalize.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or finalizing fails.
    fn finish(self) -> Result<(), Self::Error>;
}

/// A stateful cursor that separates key access from full record
/// materialization.
///
/// The merge engine calls [`advance`](CodecCursor::advance) to position the
/// cursor, [`current_key`](Self::current_key) to feed the heap, and
/// [`current`](CodecCursor::current) or [`with_current`](CodecCursor::with_current)
/// only for the merge winner. This avoids materializing records that lose the
/// heap comparison.
pub trait KeyedCodecCursor<T, K>: CodecCursor<T> {
    /// Return the stored key for the current entry.
    ///
    /// Valid only after [`advance`](CodecCursor::advance) returned `true`.
    /// Repeated calls must return the same key until the next call to
    /// [`advance`](CodecCursor::advance).
    ///
    /// # Errors
    ///
    /// Returns an error if reading or decoding the key fails.
    fn current_key(&self) -> Result<K, Self::Error>;
}

/// Extension trait for codecs that store a compact *record key*
/// alongside each record on disk.
///
/// The record key is a compact, owned representation derived from
/// the same data as the [`SortKey`](crate::key::SortKey), but
/// potentially in a different encoding (e.g., 2-bit packed
/// nucleotides vs. raw ASCII bytes). During the k-way merge, the
/// heap compares these small fixed-width record keys without
/// deserializing full records — only the winning record is
/// deserialized on each merge step.
///
/// Because the record key and the sort key may use different
/// representations of the same underlying data, the user's
/// [`Compare`](crate::compare::Compare) implementation must
/// handle both types — sorting behaviour cannot be assumed to
/// transfer between encodings.
///
/// The merge engine selects between the base [`Codec`] path and
/// the `KeyedCodec` fast path at compile time based on whether
/// the user calls `.codec()` or `.keyed_codec()` on the builder.
pub trait KeyedCodec: Codec {
    /// The compact record key stored alongside each record on
    /// disk for merge acceleration. This is distinct from the
    /// [`SortKey`](crate::key::SortKey), which is a transient,
    /// potentially borrowed value used during in-memory chunk
    /// sort. `Clone` is required because the merge heap holds
    /// keys independently of reader state.
    type Key: Clone;

    /// A stateful writer that encodes items with their keys.
    type KeyedWriter<W: Write>;

    /// A stateful cursor that can retrieve keys and records
    /// independently.
    type KeyedCursor<R: Read>: KeyedCodecCursor<Self::Item, Self::Key, Error = Self::Error>;

    /// Create a keyed writer that encodes items with their keys
    /// into `dest`.
    fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W>;

    /// Create a keyed cursor over a byte source.
    fn keyed_cursor<R: Read>(&self, source: R) -> Self::KeyedCursor<R>;
}

/// Capability for deriving a keyed codec's stored key from a write-side item.
///
/// This is separate from [`KeyedCodec`] because the item written into a run need
/// not be the same representation that the codec materializes when reading the
/// run back. Owned sorters typically implement this for `Codec::Item`, while
/// allocation-conscious writers may implement it for borrowed current views.
pub trait DeriveKey<I: ?Sized>: KeyedCodec {
    /// Derive the stored key for `item`.
    fn derive_key(&self, item: &I) -> Self::Key;
}

#[cfg(test)]
mod tests {
    use std::io::BufWriter;

    use super::*;

    #[derive(Clone, Copy)]
    struct U64Codec;

    struct U64Writer<W: Write> {
        inner: BufWriter<W>,
    }

    impl<W: Write> CodecWriter<u64> for U64Writer<W> {
        type Error = std::io::Error;

        fn write(&mut self, item: &u64) -> Result<(), Self::Error> {
            use std::io::Write as _;
            self.inner.write_all(&item.to_le_bytes())
        }

        fn finish(mut self) -> Result<(), Self::Error> {
            use std::io::Write as _;
            self.inner.flush()
        }
    }

    struct U64Reader<R: Read> {
        inner: R,
        current: Option<u64>,
    }

    impl<R: Read> CodecCursor<u64> for U64Reader<R> {
        type Error = std::io::Error;
        type Current<'a>
            = u64
        where
            Self: 'a;

        fn advance(&mut self) -> Result<bool, Self::Error> {
            let mut buf = [0u8; 8];
            match self.inner.read(&mut buf[..1]) {
                Ok(0) => {
                    self.current = None;
                    Ok(false)
                }
                Ok(_) => {
                    self.inner.read_exact(&mut buf[1..])?;
                    self.current = Some(u64::from_le_bytes(buf));
                    Ok(true)
                }
                Err(e) => Err(e),
            }
        }

        fn current(&mut self) -> Result<u64, Self::Error> {
            self.current
                .ok_or_else(|| std::io::Error::other("current called before advance"))
        }

        fn with_current<'a, F>(
            &'a mut self,
            f: impl FnOnce(Self::Current<'a>) -> F,
        ) -> Result<F, Self::Error> {
            self.current().map(f)
        }
    }

    impl Codec for U64Codec {
        type Item = u64;
        type Error = std::io::Error;
        type Writer<W: Write> = U64Writer<W>;
        type Cursor<R: Read> = U64Reader<R>;

        fn writer<W: Write>(&self, dest: W) -> U64Writer<W> {
            U64Writer {
                inner: BufWriter::new(dest),
            }
        }

        fn cursor<R: Read>(&self, source: R) -> U64Reader<R> {
            U64Reader {
                inner: source,
                current: None,
            }
        }
    }

    #[test]
    fn codec_round_trips_single_item() {
        let mut buf = Vec::new();
        let mut writer = U64Codec.writer(&mut buf);
        writer.write(&42u64).expect("write should succeed");
        writer.finish().expect("finish should succeed");
        assert_eq!(buf.len(), 8, "u64 should write exactly 8 bytes");

        let mut reader = U64Codec.cursor(std::io::Cursor::new(&buf));
        assert!(reader.advance().expect("advance should succeed"));
        let visited = reader
            .with_current(|item| item)
            .expect("with_current should succeed");
        assert_eq!(visited, 42, "current representation should match");
        let item = reader.current().expect("current should succeed");
        assert_eq!(item, 42, "round-tripped value should match");
    }

    #[test]
    fn codec_round_trips_multiple_items() {
        let values = vec![1u64, 2, 3, u64::MAX, 0];
        let mut buf = Vec::new();
        let mut writer = U64Codec.writer(&mut buf);
        for v in &values {
            writer.write(v).expect("write should succeed");
        }
        writer.finish().expect("finish should succeed");

        let mut reader = U64Codec.cursor(std::io::Cursor::new(&buf));
        let mut recovered = Vec::new();
        while reader.advance().expect("advance should succeed") {
            recovered.push(reader.current().expect("current should succeed"));
        }

        assert_eq!(
            recovered, values,
            "all round-tripped values should match in order"
        );
    }

    #[test]
    fn codec_read_empty_returns_none() {
        let buf: Vec<u8> = Vec::new();
        let mut reader = U64Codec.cursor(std::io::Cursor::new(&buf));
        let result = reader.advance().expect("reading empty should not error");
        assert!(!result, "reading from an empty source should return false");
    }

    #[test]
    fn codec_read_truncated_returns_error() {
        let buf = vec![0u8; 3]; // less than 8 bytes
        let mut reader = U64Codec.cursor(std::io::Cursor::new(&buf));
        let result = reader.advance();
        assert!(
            result.is_err(),
            "reading a partial record should return an error, not None"
        );
    }
}
