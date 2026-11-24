//! Serialization traits for reading and writing items to disk.
//!
//! [`Codec`] defines how items are read back from temporary files during the
//! sort. [`EncodeRun`] and [`EncodeKeyedRun`] define which write-side item shapes
//! a format can encode into temporary runs. Core owns temporary-run transport
//! concerns such as opening run files and applying the outer read/write buffers
//! configured by [`MergeConfig`](crate::merge::MergeConfig). Codecs own record
//! encoding and decoding, and may still keep format-specific buffers such as
//! compression blocks or reusable decode storage while keeping the codec itself a
//! trivially duplicatable `Copy` configuration object.
//!
//! [`KeyedCodec`] is an optional extension for formats that
//! can store a precomputed sort key alongside each record,
//! enabling the merge engine to compare keys without
//! deserializing full records.

use std::io::{Read, Write};

/// A stateful writer created by [`EncodeRun`] for writing items
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
pub trait CodecCursor {
    /// The record representation borrowed from this cursor.
    type Record: ?Sized;

    /// The error type for read failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Advance to the next item, returning `false` at clean EOF.
    /// A partial read (some bytes but not a complete item) should return an
    /// error, not `false`.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails or the stream contains a partial
    /// record.
    fn advance(&mut self) -> Result<bool, Self::Error>;

    /// Visit the current item in this cursor's borrowed representation.
    ///
    /// The value passed to the callback is valid only for the callback. This is
    /// valid only after [`advance`](Self::advance) returned `true`. Calling this
    /// method must not advance the cursor or consume the current position.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding or current-item access fails.
    fn with_current<Output>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> Output,
    ) -> Result<Output, Self::Error>;
}

/// Optional capability to transfer or materialize an owned current record.
///
/// After taking a record, do not access the current position again before
/// advancing. `None` means the current record is unavailable, not end of input.
pub trait TakeOwned: CodecCursor {
    /// The owned representation returned by this cursor.
    type Owned;

    /// Take the current record without advancing the cursor.
    ///
    /// # Errors
    ///
    /// Returns an error if extracting or materializing the record fails.
    fn take_owned(&mut self) -> Result<Option<Self::Owned>, Self::Error>;
}

/// Defines the read side of an on-disk format for sorted runs.
///
/// A codec is a stateless, `Copy` configuration object that
/// knows how to create cursors for its format. Cursors hold whatever read-side
/// state the format needs, such as reusable decode buffers.
///
/// Writing is modeled by [`EncodeRun`] because a run format may be readable as
/// one owned item type while accepting several write-side item shapes.
///
/// The core crate provides no built-in codecs — implementations
/// live in domain crates (e.g., `spillover-bio` provides a
/// dryice-based codec) or in application code for simple cases.
pub trait Codec: Copy {
    /// The error type for encode/decode failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// A stateful cursor for decoding items from a sorted run.
    type Cursor<R: Read>: CodecCursor<Error = Self::Error>;

    /// Create a cursor that decodes items from `source`.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot be initialized.
    fn cursor<R: Read>(&self, source: R) -> Result<Self::Cursor<R>, Self::Error>;
}

/// Read sorted runs using an ordinary iterator of owned records.
///
/// Implementing this trait provides [`Codec`] and [`TakeOwned`] automatically.
/// Formats needing borrowed decoder storage may implement [`Codec`] directly.
pub trait DecodeRun: Copy {
    /// The record yielded by the reader.
    type Record;
    /// The error type for read failures.
    type Error: std::error::Error + Send + Sync + 'static;
    /// A reader over one sorted run.
    type Reader<R: Read>: Iterator<Item = Result<Self::Record, Self::Error>>;

    /// Create a reader for a sorted run.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot be initialized.
    fn reader<R: Read>(&self, source: R) -> Result<Self::Reader<R>, Self::Error>;
}

/// Cursor retaining the current record of an ordinary run reader.
///
/// Constructed by the blanket [`Codec`] implementation for [`DecodeRun`].
pub struct RetainedReader<I, T> {
    reader: I,
    current: Option<T>,
}

impl<F: DecodeRun> Codec for F {
    type Error = F::Error;
    type Cursor<R: Read> = RetainedReader<F::Reader<R>, F::Record>;

    fn cursor<R: Read>(&self, source: R) -> Result<Self::Cursor<R>, Self::Error> {
        let reader = self.reader(source)?;
        Ok(RetainedReader {
            reader,
            current: None,
        })
    }
}

impl<I, T, E> CodecCursor for RetainedReader<I, T>
where
    I: Iterator<Item = Result<T, E>>,
    E: std::error::Error + Send + Sync + 'static,
{
    type Record = T;
    type Error = E;

    fn advance(&mut self) -> Result<bool, Self::Error> {
        self.current = None;
        let decoded = self.reader.next().transpose()?;
        let positioned = decoded.is_some();
        self.current = decoded;
        Ok(positioned)
    }

    fn with_current<Output>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> Output,
    ) -> Result<Output, Self::Error> {
        let record = self
            .current
            .as_ref()
            .expect("current access requires a positioned reader");
        Ok(visit(record))
    }
}

impl<I, T, E> TakeOwned for RetainedReader<I, T>
where
    I: Iterator<Item = Result<T, E>>,
    E: std::error::Error + Send + Sync + 'static,
{
    type Owned = T;

    fn take_owned(&mut self) -> Result<Option<Self::Owned>, Self::Error> {
        Ok(self.current.take())
    }
}

/// Capability for encoding write-side items into an unkeyed sorted run.
///
/// Core owns temporary-run transport: opening files and applying the outer
/// buffers configured by [`MergeConfig`](crate::merge::MergeConfig). The codec
/// owns the run format and may keep format-specific writer state such as
/// compression blocks.
///
/// `I` is the write-side item shape. It may differ from the record representation
/// exposed by the codec's cursor.
pub trait EncodeRun<I: ?Sized>: Codec {
    /// A stateful writer for encoding `I` values into a sorted run.
    type Writer<W: Write>: CodecWriter<I, Error = Self::Error>;

    /// Create a writer that encodes items into `dest`.
    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W>;
}

/// A stateful writer that stores items alongside precomputed keys.
///
/// Created by [`EncodeKeyedRun::keyed_writer`]. Like [`CodecWriter`],
/// [`finish`](Self::finish) must be called after the last item.
pub trait KeyedCodecWriter<I: ?Sized, K> {
    /// The error type for write failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Write one item with its precomputed key.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding or writing fails.
    fn write_keyed(&mut self, record: crate::key::Keyed<&I, &K>) -> Result<(), Self::Error>;

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
/// [`with_current`](CodecCursor::with_current) for the merge winner and for
/// both records when stored keys tie. Distinct keys can be compared without
/// accessing full records; tied keys require full-record comparison.
pub trait KeyedCodecCursor<K>: CodecCursor {
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

/// Read-side extension for codecs that store a compact *record key*
/// alongside each record on disk.
///
/// The record key is a compact, owned representation derived from
/// the same data as the [`SortKey`](crate::key::SortKey), but
/// potentially in a different encoding (e.g., 2-bit packed
/// nucleotides vs. raw ASCII bytes). During the k-way merge, the
/// heap compares these small fixed-width record keys first. Full-record access
/// is needed for the winner and for both records when keys tie; codecs may defer
/// full deserialization until that access.
///
/// Because the record key and the sort key may use different
/// representations of the same underlying data, the user's
/// [`Compare`](crate::compare::Compare) implementation must
/// handle both types — sorting behaviour cannot be assumed to
/// transfer between encodings.
///
/// The merge engine selects between ordinary [`Codec`] cursors and keyed cursors
/// at compile time based on whether the user calls `.codec()` or
/// `.keyed_codec()` on the builder. Writing keyed runs is modeled separately by
/// [`EncodeKeyedRun`].
pub trait KeyedCodec: Codec {
    /// The compact record key stored alongside each record on
    /// disk for merge acceleration. This is distinct from the
    /// [`SortKey`](crate::key::SortKey), which is a transient,
    /// potentially borrowed value used during in-memory chunk
    /// sort.
    type Key;

    /// A stateful cursor that can retrieve keys and records
    /// independently.
    type KeyedCursor<R: Read>: KeyedCodecCursor<Self::Key, Error = Self::Error>;

    /// Create a keyed cursor over a byte source.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot be initialized.
    fn keyed_cursor<R: Read>(&self, source: R) -> Result<Self::KeyedCursor<R>, Self::Error>;
}

/// Legacy capability for deriving a keyed codec's stored key from a write-side item.
///
/// This is separate from [`KeyedCodec`] because the item written into a run need
/// not be the same representation that the codec materializes when reading the
/// run back. Allocation-conscious writers may implement it for borrowed records.
/// Complete-order sorting derives keys through [`crate::key::DeriveKey`] instead.
pub trait DeriveKey<I: ?Sized>: KeyedCodec {
    /// Derive the stored key for `item`.
    fn derive_key(&self, item: &I) -> Self::Key;
}

/// Capability for encoding write-side items into a keyed sorted run.
///
/// The writer receives an item paired with its already-derived stored key.
pub trait EncodeKeyedRun<I: ?Sized>: KeyedCodec {
    /// A stateful writer that encodes `I` values with stored keys.
    type KeyedWriter<W: Write>: KeyedCodecWriter<I, Self::Key, Error = Self::Error>;

    /// Create a keyed writer that encodes items with their keys into `dest`.
    fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W>;
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
    }

    impl<R: Read> Iterator for U64Reader<R> {
        type Item = std::io::Result<u64>;

        fn next(&mut self) -> Option<Self::Item> {
            let mut buf = [0u8; 8];
            match self.inner.read(&mut buf[..1]) {
                Ok(0) => None,
                Ok(_) => {
                    let result = self
                        .inner
                        .read_exact(&mut buf[1..])
                        .map(|()| u64::from_le_bytes(buf));
                    Some(result)
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
            Ok(U64Reader { inner: source })
        }
    }

    impl EncodeRun<u64> for U64Codec {
        type Writer<W: Write> = U64Writer<W>;

        fn writer<W: Write>(&self, dest: W) -> U64Writer<W> {
            U64Writer {
                inner: BufWriter::new(dest),
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

        let mut reader = U64Codec
            .cursor(std::io::Cursor::new(&buf))
            .expect("cursor should open");
        assert!(reader.advance().expect("advance should succeed"));
        let visited = reader
            .with_current(|item| *item)
            .expect("with_current should succeed");
        assert_eq!(visited, 42, "current representation should match");
        let item = reader
            .take_owned()
            .expect("take should succeed")
            .expect("current should exist");
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

        let mut reader = U64Codec
            .cursor(std::io::Cursor::new(&buf))
            .expect("cursor should open");
        let mut recovered = Vec::new();
        while reader.advance().expect("advance should succeed") {
            let recovered_record = reader
                .take_owned()
                .expect("take should succeed")
                .expect("current should exist");
            recovered.push(recovered_record);
        }

        assert_eq!(
            recovered, values,
            "all round-tripped values should match in order"
        );
    }

    #[test]
    fn codec_read_empty_returns_none() {
        let buf: Vec<u8> = Vec::new();
        let mut reader = U64Codec
            .cursor(std::io::Cursor::new(&buf))
            .expect("cursor should open");
        let result = reader.advance().expect("reading empty should not error");
        assert!(!result, "reading from an empty source should return false");
    }

    #[test]
    fn codec_read_truncated_returns_error() {
        let buf = vec![0u8; 3]; // less than 8 bytes
        let mut reader = U64Codec
            .cursor(std::io::Cursor::new(&buf))
            .expect("cursor should open");
        let result = reader.advance();
        assert!(
            result.is_err(),
            "reading a partial record should return an error, not None"
        );
    }
}
