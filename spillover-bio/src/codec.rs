//! Dryice-backed codecs for spillover's sort engine.
//!
//! Two codec types are provided, corresponding to spillover's
//! two merge paths:
//!
//! - [`DryIceCodec`] implements [`Codec`] for the unkeyed merge path
//!   (no record keys, full record deserialization during merge).
//! - [`KeyedDryIceCodec`] implements both [`Codec`] and
//!   [`KeyedCodec`] for the keyed merge path (precomputed keys
//!   stored alongside records, key-first merge comparison with
//!   full-record fallback when keys tie).
//!
//! Both are `Copy` configuration markers — stateful writers and
//! cursors are created by the merge engine as needed. Thin newtype
//! adapters bridge dryice's types to spillover's traits.

use std::{
    io::{Read, Write},
    marker::PhantomData,
};

use dryice::{
    DryIceError, DryIceReader, DryIceWriter, NameCodec, QualityCodec, RawAsciiCodec, RawNameCodec,
    RawQualityCodec, RecordKey, SeqRecordLike, SequenceCodec,
};
use spillover::codec::{
    Codec, CodecCursor, CodecWriter, DeriveKey, EncodeKeyedRun, EncodeRun, KeyedCodec,
    KeyedCodecCursor, KeyedCodecWriter, TakeOwned,
};

use crate::{
    record::{SeqRecord, SeqRecordParts},
    sort::RecordKeyer,
};

/// Compile-time marker for dryice codec type parameters.
/// Uses the fn pointer pattern so the params don't impose
/// Copy/Send/etc bounds on the containing struct.
type CodecMarker<S, Q, N> = PhantomData<fn() -> (S, Q, N)>;

/// Same as [`CodecMarker`], with an additional record keyer type.
type KeyedCodecMarker<S, Q, N, Kr> = PhantomData<fn() -> (S, Q, N, Kr)>;

// ── Base path codec ──────────────────────────────────────────

/// Dryice codec for the unkeyed merge path (no record keys).
///
/// Configure with builder-style methods to select dryice's
/// sequence, quality, and name codecs. Defaults to raw
/// (uncompressed) codecs.
///
/// ```ignore
/// use spillover_bio::codec::DryIceCodec;
/// use dryice::{TwoBitExactCodec, BinnedQualityCodec, SplitNameCodec};
///
/// let codec = DryIceCodec::new()
///     .sequence_codec::<TwoBitExactCodec>()
///     .quality_codec::<BinnedQualityCodec>()
///     .name_codec::<SplitNameCodec>();
/// ```
pub struct DryIceCodec<S = RawAsciiCodec, Q = RawQualityCodec, N = RawNameCodec> {
    target_block_records: usize,
    _codecs: CodecMarker<S, Q, N>,
}

// Manual Clone/Copy: the PhantomData<fn() -> (S, Q, N)> is always
// Copy regardless of whether S, Q, N are Copy, but the derive
// macro doesn't know that and adds unnecessary bounds.
impl<S, Q, N> Clone for DryIceCodec<S, Q, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S, Q, N> Copy for DryIceCodec<S, Q, N> {}

impl DryIceCodec {
    /// Create a codec with default dryice settings (raw codecs,
    /// 4096 records per block).
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_block_records: 4096,
            _codecs: PhantomData,
        }
    }
}

impl Default for DryIceCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, Q, N> DryIceCodec<S, Q, N> {
    /// Set the target number of records per dryice block.
    #[must_use]
    pub fn with_block_records(mut self, n: usize) -> Self {
        self.target_block_records = n;
        self
    }

    /// Change the sequence codec.
    #[must_use]
    pub fn sequence_codec<S2>(self) -> DryIceCodec<S2, Q, N> {
        DryIceCodec {
            target_block_records: self.target_block_records,
            _codecs: PhantomData,
        }
    }

    /// Change the quality codec.
    #[must_use]
    pub fn quality_codec<Q2>(self) -> DryIceCodec<S, Q2, N> {
        DryIceCodec {
            target_block_records: self.target_block_records,
            _codecs: PhantomData,
        }
    }

    /// Change the name codec.
    #[must_use]
    pub fn name_codec<N2>(self) -> DryIceCodec<S, Q, N2> {
        DryIceCodec {
            target_block_records: self.target_block_records,
            _codecs: PhantomData,
        }
    }

    /// Use 2-bit exact sequence encoding (lossless, ~4x smaller).
    #[must_use]
    pub fn two_bit_exact(self) -> DryIceCodec<dryice::TwoBitExactCodec, Q, N> {
        self.sequence_codec::<dryice::TwoBitExactCodec>()
    }

    /// Use 2-bit lossy-N sequence encoding (ambiguous bases → N).
    #[must_use]
    pub fn two_bit_lossy_n(self) -> DryIceCodec<dryice::TwoBitLossyNCodec, Q, N> {
        self.sequence_codec::<dryice::TwoBitLossyNCodec>()
    }

    /// Use Illumina-style 8-level quality binning (lossy).
    #[must_use]
    pub fn binned_quality(self) -> DryIceCodec<S, dryice::BinnedQualityCodec, N> {
        self.quality_codec::<dryice::BinnedQualityCodec>()
    }

    /// Omit quality scores entirely.
    #[must_use]
    pub fn omit_quality(self) -> DryIceCodec<S, dryice::OmittedQualityCodec, N> {
        self.quality_codec::<dryice::OmittedQualityCodec>()
    }

    /// Use split name encoding (id + description stored separately).
    #[must_use]
    pub fn split_names(self) -> DryIceCodec<S, Q, dryice::SplitNameCodec> {
        self.name_codec::<dryice::SplitNameCodec>()
    }

    /// Omit names entirely.
    #[must_use]
    pub fn omit_names(self) -> DryIceCodec<S, Q, dryice::OmittedNameCodec> {
        self.name_codec::<dryice::OmittedNameCodec>()
    }

    /// Transition to a keyed codec with the given record keyer.
    #[must_use]
    pub fn with_record_keyer<Kr: RecordKeyer>(self, keyer: Kr) -> KeyedDryIceCodec<Kr, S, Q, N> {
        KeyedDryIceCodec {
            target_block_records: self.target_block_records,
            keyer,
            _codecs: PhantomData,
        }
    }

    /// Transition to a keyed codec with a 38-byte packed sequence
    /// key covering 152 bases (full Illumina 150bp reads).
    #[must_use]
    pub fn with_illumina_key(self) -> KeyedDryIceCodec<crate::sort::SequenceOrder<38>, S, Q, N> {
        self.with_record_keyer(crate::sort::ILLUMINA_ORDER)
    }

    /// Transition to a keyed codec with a 64-byte packed sequence
    /// key covering 256 bases (full 250bp paired-end reads).
    #[must_use]
    pub fn with_paired_end_key(self) -> KeyedDryIceCodec<crate::sort::SequenceOrder<64>, S, Q, N> {
        self.with_record_keyer(crate::sort::PAIRED_END_ORDER)
    }

    /// Transition to a keyed codec with a 128-byte packed sequence
    /// key covering 512 bases (prefix for long reads).
    #[must_use]
    pub fn with_long_read_key(self) -> KeyedDryIceCodec<crate::sort::SequenceOrder<128>, S, Q, N> {
        self.with_record_keyer(crate::sort::LONG_READ_ORDER)
    }
}

// ── Keyed path codec ─────────────────────────────────────────

/// Dryice codec for the keyed merge path (with record keys).
///
/// Created from [`DryIceCodec::with_record_keyer`]. Implements
/// both [`Codec`] and [`KeyedCodec`] so it can be used with
/// either `.codec()` or `.keyed_codec()` on the spillover
/// builder.
pub struct KeyedDryIceCodec<Kr, S = RawAsciiCodec, Q = RawQualityCodec, N = RawNameCodec> {
    target_block_records: usize,
    keyer: Kr,
    _codecs: KeyedCodecMarker<S, Q, N, Kr>,
}

impl<Kr: Copy, S, Q, N> Clone for KeyedDryIceCodec<Kr, S, Q, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Kr: Copy, S, Q, N> Copy for KeyedDryIceCodec<Kr, S, Q, N> {}

impl<Kr, S, Q, N> KeyedDryIceCodec<Kr, S, Q, N> {
    /// Set the target number of records per dryice block.
    #[must_use]
    pub fn with_block_records(mut self, n: usize) -> Self {
        self.target_block_records = n;
        self
    }
}

// ── Writer adapters ──────────────────────────────────────────

struct SeqRecordPartsAdapter<'a, R: ?Sized>(&'a R);

impl<R: SeqRecordParts + ?Sized> SeqRecordLike for SeqRecordPartsAdapter<'_, R> {
    fn name(&self) -> &[u8] {
        self.0.name()
    }

    fn sequence(&self) -> &[u8] {
        self.0.sequence()
    }

    fn quality(&self) -> &[u8] {
        self.0.quality()
    }
}

/// Newtype wrapping an unkeyed [`DryIceWriter`] to implement
/// spillover's [`CodecWriter`].
pub struct UnkeyedWriterAdapter<W, S: SequenceCodec, Q: QualityCodec, N: NameCodec>(
    DryIceWriter<W, S, Q, N, dryice::NoRecordKey>,
);

impl<W: Write, S: SequenceCodec, Q: QualityCodec, N: NameCodec> UnkeyedWriterAdapter<W, S, Q, N> {
    fn finish(self) -> Result<(), DryIceError> {
        self.0.finish().map(|_| ())
    }
}

impl<W: Write, S: SequenceCodec, Q: QualityCodec, N: NameCodec, R: SeqRecordParts + ?Sized>
    CodecWriter<R> for UnkeyedWriterAdapter<W, S, Q, N>
{
    type Error = DryIceError;

    fn write(&mut self, item: &R) -> Result<(), DryIceError> {
        self.0.write_record(&SeqRecordPartsAdapter(item))
    }

    fn finish(self) -> Result<(), DryIceError> {
        Self::finish(self)
    }
}

/// Newtype wrapping a keyed [`DryIceWriter`] to implement both
/// [`CodecWriter`] and [`KeyedCodecWriter`].
pub struct KeyedWriterAdapter<W, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K>(
    DryIceWriter<W, S, Q, N, K>,
);

impl<W: Write, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K>
    KeyedWriterAdapter<W, S, Q, N, K>
{
    fn finish(self) -> Result<(), DryIceError> {
        self.0.finish().map(|_| ())
    }
}

impl<
    W: Write,
    S: SequenceCodec,
    Q: QualityCodec,
    N: NameCodec,
    K: RecordKey,
    R: SeqRecordParts + ?Sized,
> KeyedCodecWriter<R, K> for KeyedWriterAdapter<W, S, Q, N, K>
{
    type Error = DryIceError;

    fn write_keyed(&mut self, item: &R, key: &K) -> Result<(), DryIceError> {
        self.0
            .write_record_with_key(&SeqRecordPartsAdapter(item), key)
    }

    fn finish(self) -> Result<(), DryIceError> {
        Self::finish(self)
    }
}

// ── Reader adapters ──────────────────────────────────────────

/// Read-only access to the current record in native dryice decoding storage.
///
/// Cursors lend this facade through [`CodecCursor::with_current`]. It exposes
/// record data through [`SeqRecordParts`], without exposing cursor control.
/// The native reader is retained here once per cursor, not copied per record.
pub struct DecodedRecord<R, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K> {
    reader: DryIceReader<R, S, Q, N, K>,
}

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K> SeqRecordParts
    for DecodedRecord<R, S, Q, N, K>
{
    fn name(&self) -> &[u8] {
        self.reader.name()
    }

    fn sequence(&self) -> &[u8] {
        self.reader.sequence()
    }

    fn quality(&self) -> &[u8] {
        self.reader.quality()
    }
}

fn own_record<R: SeqRecordParts + ?Sized>(record: &R) -> SeqRecord {
    let name = record.name();
    let sequence = record.sequence();
    let quality = record.quality();
    SeqRecord::from_slices(name, sequence, quality)
}

/// Unkeyed dryice cursor with borrowed record access and optional owned output.
pub struct UnkeyedReaderAdapter<R, S: SequenceCodec, Q: QualityCodec, N: NameCodec>(
    DecodedRecord<R, S, Q, N, dryice::NoRecordKey>,
);

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec> UnkeyedReaderAdapter<R, S, Q, N> {
    fn open(source: R) -> Result<Self, DryIceError> {
        let reader = DryIceReader::with_codecs(source)?;
        let record = DecodedRecord { reader };
        Ok(Self(record))
    }
}

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec> CodecCursor
    for UnkeyedReaderAdapter<R, S, Q, N>
{
    type Record = DecodedRecord<R, S, Q, N, dryice::NoRecordKey>;
    type Error = DryIceError;

    fn advance(&mut self) -> Result<bool, DryIceError> {
        self.0.reader.next_record()
    }

    fn with_current<Output>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> Output,
    ) -> Result<Output, DryIceError> {
        Ok(visit(&self.0))
    }
}

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec> TakeOwned
    for UnkeyedReaderAdapter<R, S, Q, N>
{
    type Owned = SeqRecord;

    fn take_owned(&mut self) -> Result<Option<Self::Owned>, DryIceError> {
        let record = self.with_current(own_record)?;
        Ok(Some(record))
    }
}

/// Keyed dryice cursor with borrowed record access and optional owned output.
pub struct KeyedReaderAdapter<R, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K>(
    DecodedRecord<R, S, Q, N, K>,
);

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey>
    KeyedReaderAdapter<R, S, Q, N, K>
{
    fn open(source: R) -> Result<Self, DryIceError> {
        let reader = DryIceReader::open(source)?;
        let record = DecodedRecord { reader };
        Ok(Self(record))
    }
}

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey> CodecCursor
    for KeyedReaderAdapter<R, S, Q, N, K>
{
    type Record = DecodedRecord<R, S, Q, N, K>;
    type Error = DryIceError;

    fn advance(&mut self) -> Result<bool, DryIceError> {
        self.0.reader.next_record()
    }

    fn with_current<Output>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> Output,
    ) -> Result<Output, DryIceError> {
        Ok(visit(&self.0))
    }
}

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey> TakeOwned
    for KeyedReaderAdapter<R, S, Q, N, K>
{
    type Owned = SeqRecord;

    fn take_owned(&mut self) -> Result<Option<Self::Owned>, DryIceError> {
        let record = self.with_current(own_record)?;
        Ok(Some(record))
    }
}

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey + Clone>
    KeyedCodecCursor<K> for KeyedReaderAdapter<R, S, Q, N, K>
{
    fn current_key(&self) -> Result<K, DryIceError> {
        self.0.reader.record_key()
    }
}

// ── Codec impls ──────────────────────────────────────────────

impl<S: SequenceCodec, Q: QualityCodec, N: NameCodec> Codec for DryIceCodec<S, Q, N> {
    type Error = DryIceError;
    type Cursor<R: Read> = UnkeyedReaderAdapter<R, S, Q, N>;

    fn cursor<R: Read>(&self, source: R) -> Result<Self::Cursor<R>, Self::Error> {
        UnkeyedReaderAdapter::open(source)
    }
}

impl<R: SeqRecordParts + ?Sized, S: SequenceCodec, Q: QualityCodec, N: NameCodec> EncodeRun<R>
    for DryIceCodec<S, Q, N>
{
    type Writer<W: Write> = UnkeyedWriterAdapter<W, S, Q, N>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        UnkeyedWriterAdapter(
            DryIceWriter::builder()
                .inner(dest)
                .sequence_codec::<S>()
                .quality_codec::<Q>()
                .name_codec::<N>()
                .target_block_records(self.target_block_records)
                .build(),
        )
    }
}

impl<Kr: RecordKeyer, S: SequenceCodec, Q: QualityCodec, N: NameCodec> Codec
    for KeyedDryIceCodec<Kr, S, Q, N>
{
    type Error = DryIceError;
    type Cursor<R: Read> = KeyedReaderAdapter<R, S, Q, N, Kr::RecordKey>;

    fn cursor<R: Read>(&self, source: R) -> Result<Self::Cursor<R>, Self::Error> {
        KeyedReaderAdapter::open(source)
    }
}

impl<Kr: RecordKeyer, S: SequenceCodec, Q: QualityCodec, N: NameCodec> KeyedCodec
    for KeyedDryIceCodec<Kr, S, Q, N>
{
    type Key = Kr::RecordKey;
    type KeyedCursor<R: Read> = KeyedReaderAdapter<R, S, Q, N, Kr::RecordKey>;

    fn keyed_cursor<R: Read>(&self, source: R) -> Result<Self::KeyedCursor<R>, Self::Error> {
        KeyedReaderAdapter::open(source)
    }
}

impl<R: SeqRecordParts + ?Sized, Kr: RecordKeyer, S: SequenceCodec, Q: QualityCodec, N: NameCodec>
    EncodeKeyedRun<R> for KeyedDryIceCodec<Kr, S, Q, N>
{
    type KeyedWriter<W: Write> = KeyedWriterAdapter<W, S, Q, N, Kr::RecordKey>;

    fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
        KeyedWriterAdapter(
            DryIceWriter::builder()
                .inner(dest)
                .sequence_codec::<S>()
                .quality_codec::<Q>()
                .name_codec::<N>()
                .record_key::<Kr::RecordKey>()
                .target_block_records(self.target_block_records)
                .build(),
        )
    }
}

impl<Kr, S, Q, N, R> DeriveKey<R> for KeyedDryIceCodec<Kr, S, Q, N>
where
    Kr: RecordKeyer,
    S: SequenceCodec,
    Q: QualityCodec,
    N: NameCodec,
    R: SeqRecordParts + ?Sized,
{
    fn derive_key(&self, item: &R) -> Kr::RecordKey {
        self.keyer.record_key(item)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::{key::PackedSequenceKey, record::SeqRecordView};

    use super::*;

    fn test_records() -> Vec<SeqRecord> {
        vec![
            SeqRecord::new(b"r1", b"ACGT", b"!!!!"),
            SeqRecord::new(b"r2", b"TGCA", b"####"),
            SeqRecord::new(b"r3", b"AAAA", b"IIII"),
        ]
    }

    #[test]
    fn unkeyed_path_round_trips_records() {
        let codec = DryIceCodec::new();
        let records = test_records();

        let mut buf = Vec::new();
        let mut writer = <DryIceCodec as EncodeRun<SeqRecord>>::writer(&codec, &mut buf);
        for rec in &records {
            writer.write(rec).expect("write should succeed");
        }
        writer.finish().expect("finish should succeed");

        let source = Cursor::new(&buf);
        let mut reader = codec.cursor(source).expect("cursor should open");
        let mut recovered = Vec::new();
        while reader.advance().expect("advance should succeed") {
            reader
                .with_current(|record| {
                    let view =
                        SeqRecordView::new(record.name(), record.sequence(), record.quality());
                    assert_eq!(
                        view,
                        records[recovered.len()].as_view(),
                        "current view should borrow the positioned record"
                    );
                })
                .expect("with_current should succeed");
            let record = reader
                .take_owned()
                .expect("take_owned should succeed")
                .expect("current record should be available");
            recovered.push(record);
        }

        assert_eq!(recovered, records);
    }

    #[test]
    fn unkeyed_path_writes_record_views() {
        let codec = DryIceCodec::new();
        let records = test_records();

        let mut buf = Vec::new();
        let mut writer = <DryIceCodec as EncodeRun<SeqRecordView<'_>>>::writer(&codec, &mut buf);
        for rec in &records {
            writer.write(&rec.as_view()).expect("write should succeed");
        }
        writer.finish().expect("finish should succeed");

        let source = Cursor::new(&buf);
        let mut reader = codec.cursor(source).expect("cursor should open");
        let mut recovered = Vec::new();
        while reader.advance().expect("advance should succeed") {
            let record = reader
                .take_owned()
                .expect("take_owned should succeed")
                .expect("current record should be available");
            recovered.push(record);
        }

        assert_eq!(recovered, records);
    }

    #[test]
    fn keyed_path_round_trips_records_and_keys() {
        let codec = DryIceCodec::new().with_record_keyer(crate::sort::SequenceOrder::<2>);
        let records = test_records();
        let keys: Vec<_> = records
            .iter()
            .map(|r| PackedSequenceKey::<2>::from_sequence(r.sequence()))
            .collect();

        let mut buf = Vec::new();
        let mut writer = <KeyedDryIceCodec<crate::sort::SequenceOrder<2>> as EncodeKeyedRun<
            SeqRecord,
        >>::keyed_writer(&codec, &mut buf);
        for (rec, key) in records.iter().zip(keys.iter()) {
            writer
                .write_keyed(rec, key)
                .expect("write_keyed should succeed");
        }
        writer.finish().expect("finish should succeed");

        let source = Cursor::new(&buf);
        let mut reader = codec.keyed_cursor(source).expect("cursor should open");
        let mut recovered_keys = Vec::new();
        let mut recovered_records = Vec::new();
        while reader.advance().expect("advance should succeed") {
            recovered_keys.push(reader.current_key().expect("current_key should succeed"));
            let record = reader
                .take_owned()
                .expect("take_owned should succeed")
                .expect("current record should be available");
            recovered_records.push(record);
        }

        assert_eq!(recovered_records, records);
        assert_eq!(recovered_keys, keys);
    }

    #[test]
    fn keyed_path_writes_record_views() {
        let codec = DryIceCodec::new().with_record_keyer(crate::sort::SequenceOrder::<2>);
        let records = test_records();
        let keys: Vec<_> = records
            .iter()
            .map(|r| PackedSequenceKey::<2>::from_sequence(r.sequence()))
            .collect();

        let mut buf = Vec::new();
        let mut writer = <KeyedDryIceCodec<crate::sort::SequenceOrder<2>> as EncodeKeyedRun<
            SeqRecordView<'_>,
        >>::keyed_writer(&codec, &mut buf);
        for (rec, key) in records.iter().zip(keys.iter()) {
            writer
                .write_keyed(&rec.as_view(), key)
                .expect("write_keyed should succeed");
        }
        writer.finish().expect("finish should succeed");

        let source = Cursor::new(&buf);
        let mut reader = codec.keyed_cursor(source).expect("cursor should open");
        let mut recovered_records = Vec::new();
        while reader.advance().expect("advance should succeed") {
            let record = reader
                .take_owned()
                .expect("take_owned should succeed")
                .expect("current record should be available");
            recovered_records.push(record);
        }

        assert_eq!(recovered_records, records);
    }

    #[test]
    fn keyed_codec_derives_keys_from_record_views() {
        let codec = DryIceCodec::new().with_record_keyer(crate::sort::SequenceOrder::<2>);
        let record = SeqRecord::new(b"r1", b"ACGT", b"!!!!");

        let owned_key = codec.derive_key(&record);
        let view_key = codec.derive_key(&record.as_view());

        assert_eq!(owned_key, PackedSequenceKey::<2>::from_sequence(b"ACGT"));
        assert_eq!(view_key, owned_key);
    }

    #[test]
    fn empty_file_reads_nothing() {
        let codec = DryIceCodec::new();

        let mut buf = Vec::new();
        let writer = <DryIceCodec as EncodeRun<SeqRecord>>::writer(&codec, &mut buf);
        writer.finish().expect("finish empty should succeed");

        let source = Cursor::new(&buf);
        let mut reader = codec.cursor(source).expect("cursor should open");
        assert!(
            !reader.advance().expect("advance should succeed"),
            "empty file should yield no records"
        );
    }

    #[test]
    fn codec_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<DryIceCodec>();
    }

    #[test]
    fn keyed_codec_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<KeyedDryIceCodec<crate::sort::SequenceOrder<2>>>();
    }

    #[test]
    fn codec_builder_turbofish_free() {
        let _codec = DryIceCodec::new()
            .two_bit_exact()
            .binned_quality()
            .split_names()
            .with_block_records(2048);
    }

    #[test]
    fn keyed_codec_turbofish_free() {
        let _codec = DryIceCodec::new()
            .two_bit_exact()
            .binned_quality()
            .split_names()
            .with_illumina_key();
    }

    #[test]
    fn keyed_codec_with_compact_codecs() {
        let codec = DryIceCodec::new()
            .two_bit_exact()
            .binned_quality()
            .split_names()
            .with_illumina_key();

        let records = test_records();
        let keys: Vec<_> = records
            .iter()
            .map(|r| PackedSequenceKey::<38>::from_sequence(r.sequence()))
            .collect();

        let mut buf = Vec::new();
        let mut writer =
            <KeyedDryIceCodec<crate::sort::SequenceOrder<38>, _, _, _> as EncodeKeyedRun<
                SeqRecord,
            >>::keyed_writer(&codec, &mut buf);
        for (rec, key) in records.iter().zip(keys.iter()) {
            writer
                .write_keyed(rec, key)
                .expect("write_keyed should succeed");
        }
        writer.finish().expect("finish should succeed");

        let source = Cursor::new(&buf);
        let mut reader = codec.keyed_cursor(source).expect("cursor should open");
        let mut count = 0;
        while reader.advance().expect("advance") {
            let _key = reader.current_key().expect("current_key");
            let _record = reader
                .take_owned()
                .expect("take_owned should succeed")
                .expect("current record should be available");
            count += 1;
        }
        assert_eq!(count, records.len());
    }
}
