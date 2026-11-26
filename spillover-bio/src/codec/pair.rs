//! Two physical dryice records form one logical pair.

use std::io::{Read, Write};

use dryice::{DryIceError, NameCodec, QualityCodec, RecordKey, SequenceCodec};
use spillover::codec::{
    Codec, CodecCursor, CodecWriter, EncodeKeyedRun, EncodeRun, KeyedCodec, KeyedCodecCursor,
    KeyedCodecWriter, TakeOwned,
};
use spillover::key::Keyed;

use super::{DryIceCodec, KeyedDryIceCodec, WriterAdapter};
use crate::record::{ReadPair, SeqRecord, SeqRecordParts};

/// Frames consecutive dryice records as pairs. Block sizes count physical reads.
///
/// The layout is for temporary sort runs, not a versioned archival pair format.
/// Keyed runs repeat the supplied pair key on both mates and check equality
/// when reading. This checks framing, not biological pairing.
/// Cursors eagerly retain both owned mates per open run in addition to the
/// underlying block buffers. That retained storage is outside the active-window budget.
#[derive(Clone, Copy)]
pub struct PairCodec<C>(C);

impl<C> PairCodec<C> {
    /// Wrap a configured ordinary or keyed dryice codec.
    #[must_use]
    pub fn new(codec: C) -> Self {
        Self(codec)
    }
}

/// A dryice failure or malformed pair in a temporary run.
#[derive(Debug)]
pub enum PairCodecError {
    /// An underlying dryice operation failed.
    DryIce(DryIceError),
    /// A physical record was not followed by its mate.
    TruncatedPair,
    /// The two physical records have different stored keys.
    MismatchedKeys,
    /// The underlying cursor did not provide a positioned record or key.
    MissingCurrent,
}

impl std::fmt::Display for PairCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DryIce(error) => write!(f, "DryIce: {error}"),
            Self::TruncatedPair => f.write_str("unpaired record at end of run"),
            Self::MismatchedKeys => f.write_str("mate keys differ"),
            Self::MissingCurrent => f.write_str("cursor has no current record or key"),
        }
    }
}

impl std::error::Error for PairCodecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DryIce(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DryIceError> for PairCodecError {
    fn from(error: DryIceError) -> Self {
        Self::DryIce(error)
    }
}

/// Eagerly decoded pair cursor.
#[doc(hidden)]
pub struct PairCursor<C> {
    inner: C,
    current: Option<ReadPair<SeqRecord>>,
    ended: bool,
}

impl<C> CodecCursor for PairCursor<C>
where
    C: CodecCursor<Error = DryIceError> + TakeOwned<Owned = SeqRecord>,
{
    type Record = ReadPair<SeqRecord>;
    type Error = PairCodecError;

    fn advance(&mut self) -> Result<bool, Self::Error> {
        self.current = None;
        if self.ended {
            return Ok(false);
        }
        // Until a complete pair has been read, any failure terminates framing.
        self.ended = true;
        if !self.inner.advance()? {
            return Ok(false);
        }
        let first = self
            .inner
            .take_owned()?
            .ok_or(PairCodecError::MissingCurrent)?;
        if !self.inner.advance()? {
            return Err(PairCodecError::TruncatedPair);
        }
        let second = self
            .inner
            .take_owned()?
            .ok_or(PairCodecError::MissingCurrent)?;
        self.current = Some(ReadPair::from_mates(first, second));
        self.ended = false;
        Ok(true)
    }

    fn with_current<T>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> T,
    ) -> Result<T, Self::Error> {
        let pair = self
            .current
            .as_ref()
            .ok_or(PairCodecError::MissingCurrent)?;
        Ok(visit(pair))
    }
}

impl<C> TakeOwned for PairCursor<C>
where
    C: CodecCursor<Error = DryIceError> + TakeOwned<Owned = SeqRecord>,
{
    type Owned = ReadPair<SeqRecord>;

    fn take_owned(&mut self) -> Result<Option<Self::Owned>, Self::Error> {
        Ok(self.current.take())
    }
}

/// Eagerly decoded pair cursor retaining the duplicated stored key.
#[doc(hidden)]
pub struct KeyedPairCursor<C, K> {
    inner: C,
    current: Option<ReadPair<SeqRecord>>,
    key: Option<K>,
    ended: bool,
}

impl<C, K> CodecCursor for KeyedPairCursor<C, K>
where
    C: KeyedCodecCursor<K, Error = DryIceError> + TakeOwned<Owned = SeqRecord>,
    K: PartialEq,
{
    type Record = ReadPair<SeqRecord>;
    type Error = PairCodecError;

    fn advance(&mut self) -> Result<bool, Self::Error> {
        self.current = None;
        self.key = None;
        if self.ended {
            return Ok(false);
        }
        self.ended = true;
        if !self.inner.advance()? {
            return Ok(false);
        }
        let key = self.inner.current_key()?;
        let first = self
            .inner
            .take_owned()?
            .ok_or(PairCodecError::MissingCurrent)?;
        if !self.inner.advance()? {
            return Err(PairCodecError::TruncatedPair);
        }
        let second_key = self.inner.current_key()?;
        if key != second_key {
            return Err(PairCodecError::MismatchedKeys);
        }
        let second = self
            .inner
            .take_owned()?
            .ok_or(PairCodecError::MissingCurrent)?;
        self.current = Some(ReadPair::from_mates(first, second));
        self.key = Some(key);
        self.ended = false;
        Ok(true)
    }

    fn with_current<T>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> T,
    ) -> Result<T, Self::Error> {
        let pair = self
            .current
            .as_ref()
            .ok_or(PairCodecError::MissingCurrent)?;
        Ok(visit(pair))
    }
}

impl<C, K> TakeOwned for KeyedPairCursor<C, K>
where
    C: KeyedCodecCursor<K, Error = DryIceError> + TakeOwned<Owned = SeqRecord>,
    K: PartialEq,
{
    type Owned = ReadPair<SeqRecord>;

    fn take_owned(&mut self) -> Result<Option<Self::Owned>, Self::Error> {
        self.key = None;
        Ok(self.current.take())
    }
}

impl<C, K> KeyedCodecCursor<K> for KeyedPairCursor<C, K>
where
    C: KeyedCodecCursor<K, Error = DryIceError> + TakeOwned<Owned = SeqRecord>,
    K: Clone + PartialEq,
{
    fn current_key(&self) -> Result<K, Self::Error> {
        self.key.clone().ok_or(PairCodecError::MissingCurrent)
    }
}

impl<S: SequenceCodec, Q: QualityCodec, N: NameCodec> Codec for PairCodec<DryIceCodec<S, Q, N>> {
    type Error = PairCodecError;
    type Cursor<R: Read> = PairCursor<<DryIceCodec<S, Q, N> as Codec>::Cursor<R>>;

    fn cursor<R: Read>(&self, source: R) -> Result<Self::Cursor<R>, Self::Error> {
        let inner = self.0.cursor(source)?;
        Ok(PairCursor {
            inner,
            current: None,
            ended: false,
        })
    }
}

impl<K: RecordKey + Clone + PartialEq, S: SequenceCodec, Q: QualityCodec, N: NameCodec> Codec
    for PairCodec<KeyedDryIceCodec<K, S, Q, N>>
{
    type Error = PairCodecError;
    type Cursor<R: Read> = KeyedPairCursor<<KeyedDryIceCodec<K, S, Q, N> as Codec>::Cursor<R>, K>;

    fn cursor<R: Read>(&self, source: R) -> Result<Self::Cursor<R>, Self::Error> {
        let inner = self.0.cursor(source)?;
        Ok(KeyedPairCursor {
            inner,
            current: None,
            key: None,
            ended: false,
        })
    }
}

impl<K: RecordKey + Clone + PartialEq, S: SequenceCodec, Q: QualityCodec, N: NameCodec> KeyedCodec
    for PairCodec<KeyedDryIceCodec<K, S, Q, N>>
{
    type Key = K;
    type KeyedCursor<R: Read> = <Self as Codec>::Cursor<R>;

    fn keyed_cursor<R: Read>(&self, source: R) -> Result<Self::KeyedCursor<R>, Self::Error> {
        self.cursor(source)
    }
}

/// Writer for ordinary pair runs.
#[doc(hidden)]
pub struct PairWriter<W>(W);

impl<W, R> CodecWriter<ReadPair<R>> for PairWriter<W>
where
    W: CodecWriter<R, Error = DryIceError>,
{
    type Error = PairCodecError;

    fn write(&mut self, pair: &ReadPair<R>) -> Result<(), Self::Error> {
        self.0.write(pair.r1())?;
        self.0.write(pair.r2())?;
        Ok(())
    }

    fn finish(self) -> Result<(), Self::Error> {
        self.0.finish()?;
        Ok(())
    }
}

impl<R: SeqRecordParts, S: SequenceCodec, Q: QualityCodec, N: NameCodec> EncodeRun<ReadPair<R>>
    for PairCodec<DryIceCodec<S, Q, N>>
{
    type Writer<W: Write> = PairWriter<<DryIceCodec<S, Q, N> as EncodeRun<R>>::Writer<W>>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        let inner = <DryIceCodec<S, Q, N> as EncodeRun<R>>::writer(&self.0, dest);
        PairWriter(inner)
    }
}

/// Writer for pair runs with the same supplied key on each physical record.
#[doc(hidden)]
pub struct KeyedPairWriter<W, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K>(
    WriterAdapter<W, S, Q, N, K>,
);

impl<W: Write, R: SeqRecordParts, K: RecordKey, S: SequenceCodec, Q: QualityCodec, N: NameCodec>
    KeyedCodecWriter<ReadPair<R>, K> for KeyedPairWriter<W, S, Q, N, K>
{
    type Error = PairCodecError;

    fn write_keyed(&mut self, entry: Keyed<&ReadPair<R>, &K>) -> Result<(), Self::Error> {
        let pair = *entry.item();
        let key = *entry.stored_key();
        self.0.write_record_with_key(pair.r1(), key)?;
        self.0.write_record_with_key(pair.r2(), key)?;
        Ok(())
    }

    fn finish(self) -> Result<(), Self::Error> {
        self.0.finish()?;
        Ok(())
    }
}

impl<
    R: SeqRecordParts,
    K: RecordKey + Clone + PartialEq,
    S: SequenceCodec,
    Q: QualityCodec,
    N: NameCodec,
> EncodeKeyedRun<ReadPair<R>> for PairCodec<KeyedDryIceCodec<K, S, Q, N>>
{
    type KeyedWriter<W: Write> = KeyedPairWriter<W, S, Q, N, K>;

    fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
        let inner =
            <KeyedDryIceCodec<K, S, Q, N> as EncodeKeyedRun<R>>::keyed_writer(&self.0, dest);
        KeyedPairWriter(inner)
    }
}
