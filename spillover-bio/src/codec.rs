//! Dryice-backed codecs for spillover's sort engine.
//!
//! Two codec types are provided, corresponding to spillover's
//! two merge paths:
//!
//! - [`DryIceCodec`] implements [`Codec`] for the base merge path
//!   (no record keys, full record deserialization during merge).
//! - [`KeyedDryIceCodec`] implements both [`Codec`] and
//!   [`KeyedCodec`] for the keyed merge path (precomputed keys
//!   stored alongside records, key-first merge comparison with
//!   full-record fallback when keys tie).
//!
//! Both are `Copy` configuration markers — stateful writers and
//! readers are created by the merge engine as needed. Thin newtype
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
    Codec, CodecReader, CodecWriter, KeyedCodec, KeyedCodecReader, KeyedCodecWriter,
};

use crate::record::SeqRecord;

/// Compile-time marker for dryice codec type parameters.
/// Uses the fn pointer pattern so the params don't impose
/// Copy/Send/etc bounds on the containing struct.
type CodecMarker<S, Q, N> = PhantomData<fn() -> (S, Q, N)>;

/// Same as [`CodecMarker`], with an additional record key type.
type KeyedCodecMarker<S, Q, N, K> = PhantomData<fn() -> (S, Q, N, K)>;

// ── Base path codec ──────────────────────────────────────────

/// Dryice codec for the base merge path (no record keys).
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

    /// Transition to a keyed codec with the given record key type
    /// and a function to derive keys from records.
    #[must_use]
    pub fn with_record_key<K: RecordKey>(
        self,
        derive_key: fn(&SeqRecord) -> K,
    ) -> KeyedDryIceCodec<S, Q, N, K> {
        KeyedDryIceCodec {
            target_block_records: self.target_block_records,
            derive_key,
            _codecs: PhantomData,
        }
    }

    /// Transition to a keyed codec with a 38-byte packed sequence
    /// key covering 152 bases (full Illumina 150bp reads).
    #[must_use]
    pub fn with_illumina_key(self) -> KeyedDryIceCodec<S, Q, N, crate::key::IlluminaKey> {
        self.with_record_key(|rec: &SeqRecord| {
            crate::key::PackedSequenceKey::from_sequence(rec.sequence())
        })
    }

    /// Transition to a keyed codec with a 64-byte packed sequence
    /// key covering 256 bases (full 250bp paired-end reads).
    #[must_use]
    pub fn with_paired_end_key(self) -> KeyedDryIceCodec<S, Q, N, crate::key::PairedEndKey> {
        self.with_record_key(|rec: &SeqRecord| {
            crate::key::PackedSequenceKey::from_sequence(rec.sequence())
        })
    }

    /// Transition to a keyed codec with a 128-byte packed sequence
    /// key covering 512 bases (prefix for long reads).
    #[must_use]
    pub fn with_long_read_key(self) -> KeyedDryIceCodec<S, Q, N, crate::key::LongReadPrefixKey> {
        self.with_record_key(|rec: &SeqRecord| {
            crate::key::PackedSequenceKey::from_sequence(rec.sequence())
        })
    }
}

// ── Keyed path codec ─────────────────────────────────────────

/// Dryice codec for the keyed merge path (with record keys).
///
/// Created from [`DryIceCodec::with_record_key`]. Implements
/// both [`Codec`] and [`KeyedCodec`] so it can be used with
/// either `.codec()` or `.keyed_codec()` on the spillover
/// builder.
pub struct KeyedDryIceCodec<S = RawAsciiCodec, Q = RawQualityCodec, N = RawNameCodec, K = ()> {
    target_block_records: usize,
    derive_key: fn(&SeqRecord) -> K,
    _codecs: KeyedCodecMarker<S, Q, N, K>,
}

impl<S, Q, N, K> Clone for KeyedDryIceCodec<S, Q, N, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S, Q, N, K> Copy for KeyedDryIceCodec<S, Q, N, K> {}

impl<S, Q, N, K> KeyedDryIceCodec<S, Q, N, K> {
    /// Set the target number of records per dryice block.
    #[must_use]
    pub fn with_block_records(mut self, n: usize) -> Self {
        self.target_block_records = n;
        self
    }
}

// ── Writer adapters ──────────────────────────────────────────

/// Newtype wrapping an unkeyed [`DryIceWriter`] to implement
/// spillover's [`CodecWriter`].
pub struct UnkeyedWriterAdapter<W, S: SequenceCodec, Q: QualityCodec, N: NameCodec>(
    DryIceWriter<W, S, Q, N, dryice::NoRecordKey>,
);

impl<W: Write, S: SequenceCodec, Q: QualityCodec, N: NameCodec> CodecWriter<SeqRecord>
    for UnkeyedWriterAdapter<W, S, Q, N>
{
    type Error = DryIceError;

    fn write(&mut self, item: &SeqRecord) -> Result<(), DryIceError> {
        self.0.write_record(item)
    }

    fn finish(self) -> Result<(), DryIceError> {
        self.0.finish().map(|_| ())
    }
}

/// Newtype wrapping a keyed [`DryIceWriter`] to implement both
/// [`CodecWriter`] and [`KeyedCodecWriter`].
pub struct KeyedWriterAdapter<W, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K>(
    DryIceWriter<W, S, Q, N, K>,
);

impl<W: Write, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey> CodecWriter<SeqRecord>
    for KeyedWriterAdapter<W, S, Q, N, K>
{
    type Error = DryIceError;

    fn write(&mut self, _item: &SeqRecord) -> Result<(), DryIceError> {
        // KeyedCodec extends Codec, so this impl must exist. But
        // a keyed writer can't write records without keys — callers
        // should use .keyed_codec() on the builder, which routes
        // through KeyedCodecWriter::write_keyed instead.
        Err(DryIceError::InvalidWriterConfiguration(
            "keyed codec used without providing a key — use .keyed_codec() on the builder",
        ))
    }

    fn finish(self) -> Result<(), DryIceError> {
        self.0.finish().map(|_| ())
    }
}

impl<W: Write, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey>
    KeyedCodecWriter<SeqRecord, K> for KeyedWriterAdapter<W, S, Q, N, K>
{
    type Error = DryIceError;

    fn write_keyed(&mut self, item: &SeqRecord, key: &K) -> Result<(), DryIceError> {
        self.0.write_record_with_key(item, key)
    }

    fn finish(self) -> Result<(), DryIceError> {
        self.0.finish().map(|_| ())
    }
}

// ── Reader adapters ──────────────────────────────────────────

/// Newtype wrapping an unkeyed [`DryIceReader`] to implement
/// spillover's [`CodecReader`]. Yields owned [`SeqRecord`] values.
pub struct UnkeyedReaderAdapter<R, S: SequenceCodec, Q: QualityCodec, N: NameCodec>(
    DryIceReader<R, S, Q, N, dryice::NoRecordKey>,
);

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec> CodecReader<SeqRecord>
    for UnkeyedReaderAdapter<R, S, Q, N>
{
    type Error = DryIceError;

    fn read(&mut self) -> Result<Option<SeqRecord>, DryIceError> {
        if self.0.next_record()? {
            Ok(Some(SeqRecord::from_slices(
                self.0.name(),
                self.0.sequence(),
                self.0.quality(),
            )))
        } else {
            Ok(None)
        }
    }
}

/// Newtype wrapping a keyed [`DryIceReader`] to implement both
/// [`CodecReader`] and [`KeyedCodecReader`].
pub struct KeyedReaderAdapter<R, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K>(
    DryIceReader<R, S, Q, N, K>,
);

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey> CodecReader<SeqRecord>
    for KeyedReaderAdapter<R, S, Q, N, K>
{
    type Error = DryIceError;

    fn read(&mut self) -> Result<Option<SeqRecord>, DryIceError> {
        if self.0.next_record()? {
            Ok(Some(SeqRecord::from_slices(
                self.0.name(),
                self.0.sequence(),
                self.0.quality(),
            )))
        } else {
            Ok(None)
        }
    }
}

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey + Clone>
    KeyedCodecReader<SeqRecord, K> for KeyedReaderAdapter<R, S, Q, N, K>
{
    type Error = DryIceError;

    fn next_key(&mut self) -> Result<Option<K>, DryIceError> {
        if self.0.next_record()? {
            Ok(Some(self.0.record_key()?))
        } else {
            Ok(None)
        }
    }

    fn current_record(&mut self) -> Result<SeqRecord, DryIceError> {
        Ok(SeqRecord::from_slices(
            self.0.name(),
            self.0.sequence(),
            self.0.quality(),
        ))
    }
}

// ── Codec impls ──────────────────────────────────────────────

impl<S: SequenceCodec, Q: QualityCodec, N: NameCodec> Codec<SeqRecord> for DryIceCodec<S, Q, N> {
    type Error = DryIceError;
    type Writer<W: Write> = UnkeyedWriterAdapter<W, S, Q, N>;
    type Reader<R: Read> = UnkeyedReaderAdapter<R, S, Q, N>;

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

    fn reader<R: Read>(&self, source: R) -> Self::Reader<R> {
        UnkeyedReaderAdapter(
            DryIceReader::with_codecs::<S, Q, N>(source)
                .expect("dryice file header should be valid"),
        )
    }
}

impl<S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey + Clone> Codec<SeqRecord>
    for KeyedDryIceCodec<S, Q, N, K>
{
    type Error = DryIceError;
    type Writer<W: Write> = KeyedWriterAdapter<W, S, Q, N, K>;
    type Reader<R: Read> = KeyedReaderAdapter<R, S, Q, N, K>;

    fn writer<W: Write>(&self, dest: W) -> Self::Writer<W> {
        KeyedWriterAdapter(
            DryIceWriter::builder()
                .inner(dest)
                .sequence_codec::<S>()
                .quality_codec::<Q>()
                .name_codec::<N>()
                .record_key::<K>()
                .target_block_records(self.target_block_records)
                .build(),
        )
    }

    fn reader<R: Read>(&self, source: R) -> Self::Reader<R> {
        KeyedReaderAdapter(
            DryIceReader::open::<S, Q, N, K>(source).expect("dryice file header should be valid"),
        )
    }
}

impl<S: SequenceCodec, Q: QualityCodec, N: NameCodec, K: RecordKey + Clone> KeyedCodec<SeqRecord>
    for KeyedDryIceCodec<S, Q, N, K>
{
    type Key = K;
    type KeyedWriter<W: Write> = KeyedWriterAdapter<W, S, Q, N, K>;
    type KeyedReader<R: Read> = KeyedReaderAdapter<R, S, Q, N, K>;

    fn derive_key(&self, item: &SeqRecord) -> K {
        (self.derive_key)(item)
    }

    fn keyed_writer<W: Write>(&self, dest: W) -> Self::KeyedWriter<W> {
        KeyedWriterAdapter(
            DryIceWriter::builder()
                .inner(dest)
                .sequence_codec::<S>()
                .quality_codec::<Q>()
                .name_codec::<N>()
                .record_key::<K>()
                .target_block_records(self.target_block_records)
                .build(),
        )
    }

    fn keyed_reader<R: Read>(&self, source: R) -> Self::KeyedReader<R> {
        KeyedReaderAdapter(
            DryIceReader::open::<S, Q, N, K>(source).expect("dryice file header should be valid"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_records() -> Vec<SeqRecord> {
        vec![
            SeqRecord::new(b"r1", b"ACGT", b"!!!!"),
            SeqRecord::new(b"r2", b"TGCA", b"####"),
            SeqRecord::new(b"r3", b"AAAA", b"IIII"),
        ]
    }

    #[test]
    fn base_path_round_trips_records() {
        let codec = DryIceCodec::new();
        let records = test_records();

        let mut buf = Vec::new();
        let mut writer = codec.writer(&mut buf);
        for rec in &records {
            writer.write(rec).expect("write should succeed");
        }
        CodecWriter::finish(writer).expect("finish should succeed");

        let mut reader = codec.reader(std::io::Cursor::new(&buf));
        let mut recovered = Vec::new();
        while let Some(rec) = reader.read().expect("read should succeed") {
            recovered.push(rec);
        }

        assert_eq!(recovered, records);
    }

    #[test]
    fn keyed_path_round_trips_records_and_keys() {
        use crate::key::PackedSequenceKey;

        let codec = DryIceCodec::new().with_record_key(|rec: &SeqRecord| {
            PackedSequenceKey::<2>::from_sequence(rec.sequence())
        });
        let records = test_records();
        let keys: Vec<_> = records
            .iter()
            .map(|r| PackedSequenceKey::<2>::from_sequence(r.sequence()))
            .collect();

        let mut buf = Vec::new();
        let mut writer = codec.keyed_writer(&mut buf);
        for (rec, key) in records.iter().zip(keys.iter()) {
            writer
                .write_keyed(rec, key)
                .expect("write_keyed should succeed");
        }
        KeyedCodecWriter::finish(writer).expect("finish should succeed");

        let mut reader = codec.keyed_reader(std::io::Cursor::new(&buf));
        let mut recovered_keys = Vec::new();
        let mut recovered_records = Vec::new();
        while let Some(key) = reader.next_key().expect("next_key should succeed") {
            recovered_keys.push(key);
            recovered_records.push(
                reader
                    .current_record()
                    .expect("current_record should succeed"),
            );
        }

        assert_eq!(recovered_records, records);
        assert_eq!(recovered_keys, keys);
    }

    #[test]
    fn empty_file_reads_nothing() {
        let codec = DryIceCodec::new();

        let mut buf = Vec::new();
        let writer = codec.writer(&mut buf);
        writer.finish().expect("finish empty should succeed");

        let mut reader = codec.reader(std::io::Cursor::new(&buf));
        assert!(
            reader.read().expect("read should succeed").is_none(),
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
        use crate::key::PackedSequenceKey;

        fn assert_copy<T: Copy>() {}
        assert_copy::<
            KeyedDryIceCodec<RawAsciiCodec, RawQualityCodec, RawNameCodec, PackedSequenceKey<2>>,
        >();
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
        use crate::key::PackedSequenceKey;

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
        let mut writer = codec.keyed_writer(&mut buf);
        for (rec, key) in records.iter().zip(keys.iter()) {
            writer
                .write_keyed(rec, key)
                .expect("write_keyed should succeed");
        }
        KeyedCodecWriter::finish(writer).expect("finish should succeed");

        let mut reader = codec.keyed_reader(std::io::Cursor::new(&buf));
        let mut count = 0;
        while let Some(_key) = reader.next_key().expect("next_key") {
            let _rec = reader.current_record().expect("current_record");
            count += 1;
        }
        assert_eq!(count, records.len());
    }
}
