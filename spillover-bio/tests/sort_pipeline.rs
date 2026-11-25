//! End-to-end integration tests for the spillover-bio sort pipeline.

use std::{
    cell::Cell,
    cmp::Ordering,
    convert::Infallible,
    io::{self, Read, Write},
    num::NonZeroUsize,
    ops::Add,
};

use dryice::SeqRecordLike;
use spillover::SortedItemsError;
use spillover::chunk::{
    ChunkSorter, EstimateSortMemory, Sequential, SortMemory, SortMemorySummary, SummarizeSortInput,
};
use spillover::codec::{
    Codec, CodecCursor, CodecWriter, DecodeRun, DeriveKey, EncodeKeyedRun, EncodeRun, KeyedCodec,
    KeyedCodecCursor, KeyedCodecWriter, TakeOwned,
};
use spillover::compare::{Compare, Natural};
use spillover::dedup::Dedup;
use spillover::key::{Owned, SortKey};
use spillover::merge::{MergeConfig, MergeError, RunMerger};
use spillover::sorter::{ByteLimit, VisitSortedItems};
use spillover_bio::{
    codec::DryIceCodec,
    radix::RadixThenRefine,
    record::{SeqRecord, SeqRecordArena, SeqRecordParts, SeqRecordView},
    sort::{
        Builder, ILLUMINA_ORDER, LENGTH_ORDER, NAME_ORDER, OrderCodecPolicy, RawRecords,
        RecordDedup, Reverse, SeqRecordSink, SequenceOrder,
    },
};
use thiserror::Error;

struct Number(u8);

struct BorrowedNumberOrder<'a>(&'a Cell<bool>);

impl Compare<Number> for BorrowedNumberOrder<'_> {
    fn compare(&self, left: &Number, right: &Number) -> Ordering {
        let ordering = left.0.cmp(&right.0);
        if self.0.get() {
            ordering.reverse()
        } else {
            ordering
        }
    }
}

struct ComparisonSort;

impl<T, Cmp: Compare<T> + ?Sized> ChunkSorter<T, Cmp> for ComparisonSort {
    fn sort(&self, items: &mut [T], cmp: &Cmp) {
        items.sort_unstable_by(|left, right| cmp.compare(left, right));
    }
}

#[derive(Clone, Copy)]
struct NumberCodec;

struct NumberReader<R>(R);

struct NumberWriter<W>(W);

impl<R: Read> Iterator for NumberReader<R> {
    type Item = io::Result<Number>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut byte = [0];
        match self.0.read(&mut byte) {
            Ok(0) => None,
            Ok(_) => Some(Ok(Number(byte[0]))),
            Err(error) => Some(Err(error)),
        }
    }
}

impl DecodeRun for NumberCodec {
    type Record = Number;
    type Error = io::Error;
    type Reader<R: Read> = NumberReader<R>;

    fn reader<R: Read>(&self, source: R) -> io::Result<Self::Reader<R>> {
        Ok(NumberReader(source))
    }
}

impl EncodeRun<Number> for NumberCodec {
    type Writer<W: Write> = NumberWriter<W>;

    fn writer<W: Write>(&self, destination: W) -> Self::Writer<W> {
        NumberWriter(destination)
    }
}

impl<W: Write> CodecWriter<Number> for NumberWriter<W> {
    type Error = io::Error;

    fn write(&mut self, item: &Number) -> io::Result<()> {
        self.0.write_all(&[item.0])
    }

    fn finish(mut self) -> io::Result<()> {
        self.0.flush()
    }
}

#[test]
fn unkeyed_item_spills_keep_locally_borrowed_whole_order() {
    let descending = Cell::new(true);
    let order = BorrowedNumberOrder(&descending);
    let mut sorter = spillover::sorter::Builder::new()
        .order(order)
        .codec(NumberCodec)
        .chunk_sort(ComparisonSort)
        .max_buffer_items(2)
        .build();

    for value in [1, 3, 2] {
        sorter.push(Number(value)).expect("push");
    }

    let output = sorter.finish().expect("finish");
    let values: Vec<u8> = output.map(|record| record.expect("decode").0).collect();
    assert_eq!(values, [3, 2, 1]);
}

#[test]
fn ordinary_decode_run_moves_non_clone_records_across_spills() {
    let sort_key = Owned(|record: &Number| record.0);
    let mut sorter = spillover::sorter::Builder::new()
        .key(sort_key)
        .codec(NumberCodec)
        .max_buffer_items(1)
        .build();

    for value in [3, 1, 2] {
        let record = Number(value);
        sorter.push(record).expect("push");
    }

    let output_stream = sorter.finish().expect("finish");
    let values: Vec<u8> = output_stream.map(|item| item.expect("decode").0).collect();
    assert_eq!(values, [1, 2, 3]);
}

#[derive(Clone, Copy)]
struct ByteCodec;

#[derive(Clone, Copy)]
struct ByteKey;

struct ByteCursor<R> {
    source: R,
    current: Vec<u8>,
}

struct ByteWriter<W>(W);

impl Codec for ByteCodec {
    type Error = io::Error;
    type Cursor<R: Read> = ByteCursor<R>;

    fn cursor<R: Read>(&self, source: R) -> io::Result<Self::Cursor<R>> {
        Ok(ByteCursor {
            source,
            current: Vec::new(),
        })
    }
}

impl<R: Read> CodecCursor for ByteCursor<R> {
    type Record = [u8];
    type Error = io::Error;

    fn advance(&mut self) -> io::Result<bool> {
        self.current.clear();
        let mut length_prefix = [0_u8; 1];
        if self.source.read(&mut length_prefix)? == 0 {
            return Ok(false);
        }

        let record_length = usize::from(length_prefix[0]);
        self.current.resize(record_length, 0);
        self.source.read_exact(&mut self.current)?;
        Ok(true)
    }

    fn with_current<Output>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> Output,
    ) -> io::Result<Output> {
        Ok(visit(self.current.as_slice()))
    }
}

impl<I: ?Sized + AsRef<[u8]>> EncodeRun<I> for ByteCodec {
    type Writer<W: Write> = ByteWriter<W>;

    fn writer<W: Write>(&self, destination: W) -> Self::Writer<W> {
        ByteWriter(destination)
    }
}

impl<I: ?Sized + AsRef<[u8]>, W: Write> CodecWriter<I> for ByteWriter<W> {
    type Error = io::Error;

    fn write(&mut self, item: &I) -> io::Result<()> {
        let bytes = item.as_ref();
        let record_length = u8::try_from(bytes.len())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        let length_prefix = [record_length];
        self.0.write_all(&length_prefix)?;
        self.0.write_all(bytes)
    }

    fn finish(mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl<I: ?Sized + AsRef<[u8]>> SortKey<I> for ByteKey {
    type Key<'a>
        = &'a [u8]
    where
        I: 'a;

    fn key<'a>(&self, item: &'a I) -> Self::Key<'a> {
        item.as_ref()
    }
}

#[test]
fn advanced_unsized_records_support_borrowed_output_across_spills() {
    let mut sorter = spillover::sorter::Builder::new()
        .key(ByteKey)
        .codec(ByteCodec)
        .max_buffer_items(1)
        .build();

    for record in [vec![3], vec![1, 2], vec![1]] {
        sorter.push(record).expect("push");
    }

    let output_stream = sorter.finish().expect("finish");
    let mut observed = Vec::new();
    output_stream
        .visit_items(|record: &[u8]| -> Result<(), Infallible> {
            let copied_bytes = record.to_vec();
            observed.push(copied_bytes);
            Ok(())
        })
        .expect("borrowed output");

    assert_eq!(observed, vec![vec![1], vec![1, 2], vec![3]]);
}

struct ScoredRecord {
    identity: u8,
    score: u8,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct StoredScore(u8);

struct ScoreOrder;

impl Compare<ScoredRecord> for ScoreOrder {
    fn compare(&self, left: &ScoredRecord, right: &ScoredRecord) -> Ordering {
        left.score
            .cmp(&right.score)
            .then_with(|| left.identity.cmp(&right.identity))
    }
}

impl spillover::key::KeyedOrder for ScoreOrder {
    type StoredKey = StoredScore;

    fn compare_keys(&self, left: &StoredScore, right: &StoredScore) -> Ordering {
        left.cmp(right)
    }
}

impl spillover::key::DeriveKey<ScoredRecord> for ScoreOrder {
    fn derive_key(&self, item: &ScoredRecord) -> StoredScore {
        StoredScore(item.score)
    }
}

#[derive(Clone, Copy)]
struct ScoreCodec;

struct ScoreCursor<R> {
    source: R,
    current: Option<(ScoredRecord, u8)>,
}

struct ScoreWriter<W>(W);

#[derive(Debug, Error)]
enum ScoreCodecError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("unknown record identity {0}")]
    UnknownIdentity(u8),
    #[error("record {identity} requires original key {expected}, received {supplied}")]
    KeyMismatch {
        identity: u8,
        expected: u8,
        supplied: u8,
    },
}

impl Codec for ScoreCodec {
    type Error = ScoreCodecError;
    type Cursor<R: Read> = ScoreCursor<R>;

    fn cursor<R: Read>(&self, source: R) -> Result<Self::Cursor<R>, Self::Error> {
        Ok(ScoreCursor {
            source,
            current: None,
        })
    }
}

impl KeyedCodec for ScoreCodec {
    type Key = StoredScore;
    type KeyedCursor<R: Read> = ScoreCursor<R>;

    fn keyed_cursor<R: Read>(&self, source: R) -> Result<Self::KeyedCursor<R>, Self::Error> {
        self.cursor(source)
    }
}

impl<R: Read> CodecCursor for ScoreCursor<R> {
    type Record = ScoredRecord;
    type Error = ScoreCodecError;

    fn advance(&mut self) -> Result<bool, Self::Error> {
        self.current = None;
        let mut encoded = [0_u8; 2];
        if self.source.read(&mut encoded[..1])? == 0 {
            return Ok(false);
        }
        self.source.read_exact(&mut encoded[1..])?;

        let [identity, stored_score] = encoded;
        let record = ScoredRecord { identity, score: 0 };
        self.current = Some((record, stored_score));
        Ok(true)
    }

    fn with_current<Output>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> Output,
    ) -> Result<Output, Self::Error> {
        let (record, _) = self
            .current
            .as_ref()
            .expect("current access requires a positioned cursor");
        Ok(visit(record))
    }
}

impl<R: Read> KeyedCodecCursor<StoredScore> for ScoreCursor<R> {
    fn current_key(&self) -> Result<StoredScore, Self::Error> {
        let (_, stored_score) = self
            .current
            .as_ref()
            .expect("key access requires a positioned cursor");
        Ok(StoredScore(*stored_score))
    }
}

impl EncodeKeyedRun<ScoredRecord> for ScoreCodec {
    type KeyedWriter<W: Write> = ScoreWriter<W>;

    fn keyed_writer<W: Write>(&self, destination: W) -> Self::KeyedWriter<W> {
        ScoreWriter(destination)
    }
}

impl<W: Write> KeyedCodecWriter<ScoredRecord, StoredScore> for ScoreWriter<W> {
    type Error = ScoreCodecError;

    fn write_keyed(
        &mut self,
        entry: spillover::key::Keyed<&ScoredRecord, &StoredScore>,
    ) -> Result<(), Self::Error> {
        let item = *entry.item();
        let key = *entry.stored_key();
        let expected = match item.identity {
            b'a' => 9,
            b'b' => 8,
            b'c' => 7,
            b'd' => 6,
            b'e' => 5,
            b'f' => 4,
            b'g' => 3,
            b'h' => 2,
            b'i' => 1,
            identity => return Err(ScoreCodecError::UnknownIdentity(identity)),
        };
        if key.0 != expected {
            return Err(ScoreCodecError::KeyMismatch {
                identity: item.identity,
                expected,
                supplied: key.0,
            });
        }

        let encoded = [item.identity, key.0];
        self.0.write_all(&encoded)?;
        Ok(())
    }

    fn finish(mut self) -> Result<(), Self::Error> {
        self.0.flush()?;
        Ok(())
    }
}

#[test]
fn keyed_merge_preserves_original_keys_across_multiple_spill_generations() {
    let mut merge_config = MergeConfig::default();
    let max_fan_in = NonZeroUsize::new(2).expect("two is nonzero");
    merge_config.max_fan_in = max_fan_in;
    let mut sorter = spillover::sorter::Builder::new()
        .order(ScoreOrder)
        .keyed_codec(ScoreCodec)
        .merge_config(merge_config)
        .max_buffer_items(1)
        .build();

    for (identity, score) in [
        (b'a', 9),
        (b'b', 8),
        (b'c', 7),
        (b'd', 6),
        (b'e', 5),
        (b'f', 4),
        (b'g', 3),
        (b'h', 2),
        (b'i', 1),
    ] {
        let record = ScoredRecord { identity, score };
        sorter.push(record).expect("push with original key");
    }

    let output_stream = sorter.finish().expect("merge with original keys");
    let mut observed = Vec::new();
    output_stream
        .visit_items(|record: &ScoredRecord| -> Result<(), Infallible> {
            observed.push(record.identity);
            Ok(())
        })
        .expect("borrowed output");

    assert_eq!(
        observed,
        [b'i', b'h', b'g', b'f', b'e', b'd', b'c', b'b', b'a']
    );
}

#[derive(Clone, Copy)]
struct RejectTwoCodec;

struct RejectTwoReader<R>(R);

struct RejectTwoWriter<W>(W);

#[derive(Debug, Error)]
enum NumberDecodeError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("rejected encoded byte {0}")]
    RejectedByte(u8),
}

impl<R: Read> Iterator for RejectTwoReader<R> {
    type Item = Result<Number, NumberDecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut byte = [0_u8; 1];
        match self.0.read(&mut byte) {
            Ok(0) => None,
            Ok(_) if byte[0] == 2 => Some(Err(NumberDecodeError::RejectedByte(2))),
            Ok(_) => Some(Ok(Number(byte[0]))),
            Err(error) => Some(Err(error.into())),
        }
    }
}

impl DecodeRun for RejectTwoCodec {
    type Record = Number;
    type Error = NumberDecodeError;
    type Reader<R: Read> = RejectTwoReader<R>;

    fn reader<R: Read>(&self, source: R) -> Result<Self::Reader<R>, Self::Error> {
        Ok(RejectTwoReader(source))
    }
}

impl EncodeRun<Number> for RejectTwoCodec {
    type Writer<W: Write> = RejectTwoWriter<W>;

    fn writer<W: Write>(&self, destination: W) -> Self::Writer<W> {
        RejectTwoWriter(destination)
    }
}

impl<W: Write> CodecWriter<Number> for RejectTwoWriter<W> {
    type Error = NumberDecodeError;

    fn write(&mut self, item: &Number) -> Result<(), Self::Error> {
        let encoded = [item.0];
        self.0.write_all(&encoded)?;
        Ok(())
    }

    fn finish(mut self) -> Result<(), Self::Error> {
        self.0.flush()?;
        Ok(())
    }
}

#[test]
fn owned_delivery_precedes_refill_error_and_fuses() {
    let sort_key = Owned(|record: &Number| record.0);
    let mut sorter = spillover::sorter::Builder::new()
        .key(sort_key)
        .codec(RejectTwoCodec)
        .max_buffer_items(3)
        .build();

    let deliverable_record = Number(1);
    sorter.push(deliverable_record).expect("push one");
    let rejected_record = Number(2);
    sorter.push(rejected_record).expect("push two");
    let trailing_record = Number(3);
    sorter.push(trailing_record).expect("push three");
    let mut output_stream = sorter.finish().expect("finish");

    let first = output_stream.next();
    assert!(matches!(first, Some(Ok(Number(1)))));

    let failure = output_stream.next();
    assert!(matches!(
        failure,
        Some(Err(MergeError::Codec(NumberDecodeError::RejectedByte(2))))
    ));
    assert!(output_stream.next().is_none());
}

#[derive(Clone, Copy)]
struct MissingCurrentCodec;

struct MissingCurrentCursor<R> {
    reader: NumberReader<R>,
    current: Option<Number>,
}

impl Codec for MissingCurrentCodec {
    type Error = io::Error;
    type Cursor<R: Read> = MissingCurrentCursor<R>;

    fn cursor<R: Read>(&self, source: R) -> io::Result<Self::Cursor<R>> {
        let reader = NumberReader(source);
        Ok(MissingCurrentCursor {
            reader,
            current: None,
        })
    }
}

impl EncodeRun<Number> for MissingCurrentCodec {
    type Writer<W: Write> = NumberWriter<W>;

    fn writer<W: Write>(&self, destination: W) -> Self::Writer<W> {
        NumberWriter(destination)
    }
}

impl<R: Read> CodecCursor for MissingCurrentCursor<R> {
    type Record = Number;
    type Error = io::Error;

    fn advance(&mut self) -> io::Result<bool> {
        self.current = None;
        let decoded = self.reader.next().transpose()?;
        let positioned = decoded.is_some();
        self.current = decoded;
        Ok(positioned)
    }

    fn with_current<Output>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> Output,
    ) -> io::Result<Output> {
        let record = self
            .current
            .as_ref()
            .expect("current access requires a positioned cursor");
        Ok(visit(record))
    }
}

impl<R: Read> TakeOwned for MissingCurrentCursor<R> {
    type Owned = Number;

    fn take_owned(&mut self) -> io::Result<Option<Self::Owned>> {
        let record = self.current.take().filter(|record| record.0 != 1);
        Ok(record)
    }
}

#[test]
fn selected_missing_current_is_terminal() {
    let sort_key = Owned(|record: &Number| record.0);
    let mut sorter = spillover::sorter::Builder::new()
        .key(sort_key)
        .codec(MissingCurrentCodec)
        .max_buffer_items(2)
        .build();

    let selected_record = Number(1);
    sorter.push(selected_record).expect("push one");
    let later_record = Number(2);
    sorter.push(later_record).expect("push two");
    let mut output_stream = sorter.finish().expect("finish");

    let failure = output_stream.next();
    assert!(matches!(failure, Some(Err(MergeError::MissingCurrent))));
    assert!(output_stream.next().is_none());
}

#[derive(Clone, Copy)]
struct DistinctProjectionCodec;

struct KeyedNumberCursor<R> {
    source: R,
    current: Option<(Number, u8)>,
}

struct KeyedNumberWriter<W>(W);

impl Codec for DistinctProjectionCodec {
    type Error = io::Error;
    type Cursor<R: Read> = ByteCursor<R>;

    fn cursor<R: Read>(&self, source: R) -> io::Result<Self::Cursor<R>> {
        ByteCodec.cursor(source)
    }
}

impl KeyedCodec for DistinctProjectionCodec {
    type Key = u8;
    type KeyedCursor<R: Read> = KeyedNumberCursor<R>;

    fn keyed_cursor<R: Read>(&self, source: R) -> io::Result<Self::KeyedCursor<R>> {
        Ok(KeyedNumberCursor {
            source,
            current: None,
        })
    }
}

impl<R: Read> CodecCursor for KeyedNumberCursor<R> {
    type Record = Number;
    type Error = io::Error;

    fn advance(&mut self) -> io::Result<bool> {
        self.current = None;
        let mut encoded = [0_u8; 2];
        if self.source.read(&mut encoded[..1])? == 0 {
            return Ok(false);
        }
        self.source.read_exact(&mut encoded[1..])?;

        let [value, stored_key] = encoded;
        let record = Number(value);
        self.current = Some((record, stored_key));
        Ok(true)
    }

    fn with_current<Output>(
        &mut self,
        visit: impl FnOnce(&Self::Record) -> Output,
    ) -> io::Result<Output> {
        let (record, _) = self
            .current
            .as_ref()
            .expect("current access requires a positioned cursor");
        Ok(visit(record))
    }
}

impl<R: Read> KeyedCodecCursor<u8> for KeyedNumberCursor<R> {
    fn current_key(&self) -> io::Result<u8> {
        let (_, stored_key) = self
            .current
            .as_ref()
            .expect("key access requires a positioned cursor");
        Ok(*stored_key)
    }
}

impl DeriveKey<Number> for DistinctProjectionCodec {
    fn derive_key(&self, item: &Number) -> u8 {
        item.0 / 10
    }
}

impl EncodeKeyedRun<Number> for DistinctProjectionCodec {
    type KeyedWriter<W: Write> = KeyedNumberWriter<W>;

    fn keyed_writer<W: Write>(&self, destination: W) -> Self::KeyedWriter<W> {
        KeyedNumberWriter(destination)
    }
}

impl<W: Write> KeyedCodecWriter<Number, u8> for KeyedNumberWriter<W> {
    type Error = io::Error;

    fn write_keyed(&mut self, entry: spillover::key::Keyed<&Number, &u8>) -> io::Result<()> {
        let item = *entry.item();
        let key = *entry.stored_key();
        let encoded = [item.0, *key];
        self.0.write_all(&encoded)
    }

    fn finish(mut self) -> io::Result<()> {
        self.0.flush()
    }
}

#[test]
fn keyed_builder_uses_distinct_keyed_record_projection_across_spills() {
    let sort_key = Owned(|record: &Number| record.0);
    let mut sorter = spillover::sorter::Builder::new()
        .key(sort_key)
        .keyed_codec(DistinctProjectionCodec)
        .max_buffer_items(1)
        .build();

    for value in [31, 12, 11, 22] {
        let record = Number(value);
        sorter.push(record).expect("push");
    }

    let output_stream = sorter.finish().expect("finish");
    let mut observed = Vec::new();
    output_stream
        .visit_items(|record: &Number| -> Result<(), Infallible> {
            observed.push(record.0);
            Ok(())
        })
        .expect("borrowed output");

    assert_eq!(observed, [11, 12, 22, 31]);
}

struct ForwardTraversal;

struct ForwardedSource<S>(S);

struct Stop(u8);

impl<S> Dedup<S> for ForwardTraversal {
    type Deduped = ForwardedSource<S>;

    fn dedup(self, source: S) -> Self::Deduped {
        ForwardedSource(source)
    }
}

impl<S: VisitSortedItems> VisitSortedItems for ForwardedSource<S> {
    type Record = S::Record;
    type Error = S::Error;

    fn visit_items<F, FE>(self, callback: F) -> Result<(), SortedItemsError<Self::Error, FE>>
    where
        F: FnMut(&Self::Record) -> Result<(), FE>,
    {
        self.0.visit_items(callback)
    }
}

#[test]
fn downstream_dedup_adapter_forwards_typed_sink_error() {
    let sort_key = Owned(|record: &Number| record.0);
    let mut sorter = spillover::sorter::Builder::new()
        .key(sort_key)
        .codec(NumberCodec)
        .dedup(ForwardTraversal)
        .max_buffer_items(1)
        .build();

    for value in [3, 1, 2] {
        let record = Number(value);
        sorter.push(record).expect("push");
    }

    let output_stream = sorter.finish().expect("finish");
    let mut visited = Vec::new();
    let result = output_stream.items().try_for_each(|record| {
        visited.push(record.0);
        if record.0 == 2 {
            return Err(Stop(42));
        }
        Ok(())
    });

    assert_eq!(visited, [1, 2]);
    assert!(matches!(result, Err(SortedItemsError::Sink(Stop(42)))));
}

#[test]
fn downstream_dedup_adapter_forwards_typed_source_error() {
    let sort_key = Owned(|record: &Number| record.0);
    let mut sorter = spillover::sorter::Builder::new()
        .key(sort_key)
        .codec(RejectTwoCodec)
        .dedup(ForwardTraversal)
        .max_buffer_items(3)
        .build();

    for value in [1, 2, 3] {
        let record = Number(value);
        sorter.push(record).expect("push");
    }

    let output_stream = sorter.finish().expect("finish");
    let mut visited = Vec::new();
    let result = output_stream
        .items()
        .try_for_each(|record| -> Result<(), Infallible> {
            visited.push(record.0);
            Ok(())
        });

    assert_eq!(visited, [1]);
    assert!(matches!(
        result,
        Err(SortedItemsError::Source(MergeError::Codec(
            NumberDecodeError::RejectedByte(2)
        )))
    ));
}

struct BorrowedPriorityOrder<'a> {
    priority: &'a [u8],
}

impl Compare<Number> for BorrowedPriorityOrder<'_> {
    fn compare(&self, left: &Number, right: &Number) -> Ordering {
        let left_position = self.priority.iter().position(|&value| value == left.0);
        let right_position = self.priority.iter().position(|&value| value == right.0);

        match (left_position, right_position) {
            (Some(left_rank), Some(right_rank)) => left_rank.cmp(&right_rank),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => left.0.cmp(&right.0),
        }
    }
}

#[test]
fn borrowed_non_clone_order_survives_intermediate_and_terminal_merge() {
    let priority = [2, 3, 1];
    let comparator = BorrowedPriorityOrder {
        priority: &priority,
    };
    let max_fan_in = NonZeroUsize::new(2).expect("two is nonzero");
    let mut config = MergeConfig::default();
    config.max_fan_in = max_fan_in;
    let merger = RunMerger::new(NumberCodec, comparator, config);
    let mut runs = Vec::new();

    for value in [1, 2, 3] {
        let records = [Number(value)];
        let run = merger.spill_sorted(records).expect("spill singleton");
        runs.push(run);
    }

    let output_stream = merger.merge(runs).expect("merge");
    let observed: Vec<u8> = output_stream
        .map(|item| {
            let record = item.expect("decode");
            record.0
        })
        .collect();

    assert_eq!(observed, [2, 3, 1]);
}

#[derive(Clone, Copy)]
struct AdmissionCodec;

struct AdmissionReader<R>(R);

struct AdmissionWriter<W>(W);

#[derive(Debug, Error)]
enum AdmissionCodecError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("rejected record one")]
    RejectedOne,
    #[error("rejected record zero")]
    RejectedZero,
}

impl<R: Read> Iterator for AdmissionReader<R> {
    type Item = Result<u64, AdmissionCodecError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut encoded = [0_u8; 8];
        match self.0.read(&mut encoded[..1]) {
            Ok(0) => return None,
            Ok(_) => {}
            Err(error) => return Some(Err(error.into())),
        }
        if let Err(error) = self.0.read_exact(&mut encoded[1..]) {
            return Some(Err(error.into()));
        }

        let record = u64::from_le_bytes(encoded);
        Some(Ok(record))
    }
}

impl DecodeRun for AdmissionCodec {
    type Record = u64;
    type Error = AdmissionCodecError;
    type Reader<R: Read> = AdmissionReader<R>;

    fn reader<R: Read>(&self, source: R) -> Result<Self::Reader<R>, Self::Error> {
        Ok(AdmissionReader(source))
    }
}

impl EncodeRun<u64> for AdmissionCodec {
    type Writer<W: Write> = AdmissionWriter<W>;

    fn writer<W: Write>(&self, destination: W) -> Self::Writer<W> {
        AdmissionWriter(destination)
    }
}

impl<W: Write> CodecWriter<u64> for AdmissionWriter<W> {
    type Error = AdmissionCodecError;

    fn write(&mut self, item: &u64) -> Result<(), Self::Error> {
        match *item {
            1 => return Err(AdmissionCodecError::RejectedOne),
            0 => return Err(AdmissionCodecError::RejectedZero),
            _ => {}
        }

        let encoded = item.to_le_bytes();
        self.0.write_all(&encoded)?;
        Ok(())
    }

    fn finish(mut self) -> Result<(), Self::Error> {
        self.0.flush()?;
        Ok(())
    }
}

#[test]
fn unkeyed_byte_budget_preflushes_old_window_before_insertion() {
    let mut merge_config = MergeConfig::default();
    merge_config.write_buffer_bytes = 1009;
    let sort_key = Owned(|record: &u64| *record);
    let mut sorter = spillover::sorter::Builder::new()
        .key(sort_key)
        .codec(AdmissionCodec)
        .chunk_sort(Sequential)
        .merge_config(merge_config)
        .fixed_size_budget(1024)
        .build();

    // Modeled spill bytes: 1009 + 8 = 1017 for the old window, but
    // 1009 + 16 = 1025 with the incoming record. Growth-only overlap is
    // below 1024 and does not overlap the spill transport.
    let old_record = 1_u64;
    sorter.push(old_record).expect("first push should succeed");
    let incoming_record = 0_u64;
    let result = sorter.push(incoming_record);

    assert!(
        matches!(
            &result,
            Err(MergeError::Codec(AdmissionCodecError::RejectedOne))
        ),
        "expected old-window preflush to reject one, got {result:?}"
    );
}

#[test]
fn from_inputs_budget_includes_allocating_summary_envelope() {
    struct Summary(Box<[u8; 128]>);

    impl Add<&Summary> for &Summary {
        type Output = Summary;

        fn add(self, other: &Summary) -> Summary {
            let mut combined = Box::new([0; 128]);
            combined[0] = self.0[0].saturating_add(other.0[0]);
            Summary(combined)
        }
    }

    struct Algorithm;

    impl ChunkSorter<u64, Natural> for Algorithm {
        fn sort(&self, items: &mut [u64], order: &Natural) {
            Sequential.sort(items, order);
        }
    }

    impl SortMemorySummary for Algorithm {
        type Summary = Summary;

        fn summary(&self) -> Summary {
            Summary(Box::new([0; 128]))
        }

        fn summary_allowance(&self, _: Option<&Summary>) -> (usize, usize) {
            (128, 0)
        }
    }

    impl SummarizeSortInput<u64> for Algorithm {
        fn summarize(&self, _: &u64) -> Summary {
            Summary(Box::new([0; 128]))
        }

        fn summary_allowance_for(&self, _: &Summary, _: &u64) -> (usize, usize) {
            (128, 0)
        }
    }

    impl EstimateSortMemory<u64, Summary> for Algorithm {
        fn sort_memory(&self, _: usize, _: &Summary) -> SortMemory {
            SortMemory::NONE
        }
    }

    // One slot plus 3H is 392 > 388. The joined replacement and two
    // overlapping summaries are 24 + 256 = 280, below the budget.
    let mut merge_config = MergeConfig::default();
    merge_config.write_buffer_bytes = 1;
    let mut sorter = spillover::sorter::Builder::new()
        .order(Natural)
        .codec(AdmissionCodec)
        .chunk_sort(Algorithm)
        .merge_config(merge_config)
        .memory_budget(ByteLimit::new(388).from_inputs())
        .build();

    sorter.push(1_u64).expect("first input fits");
    let result = sorter.push(0_u64);
    assert!(
        matches!(
            &result,
            Err(MergeError::Codec(AdmissionCodecError::RejectedOne))
        ),
        "expected the old record to be rejected, got {result:?}"
    );
}

struct WitnessSummary(Box<[u8; 32]>);

impl Add<&WitnessSummary> for &WitnessSummary {
    type Output = WitnessSummary;

    fn add(self, other: &WitnessSummary) -> WitnessSummary {
        let mut combined = Box::new([0; 32]);
        combined[0] = self.0[0].saturating_add(other.0[0]).min(63);
        WitnessSummary(combined)
    }
}

struct WitnessOrder;

impl Compare<u64> for WitnessOrder {
    fn compare(&self, left: &u64, right: &u64) -> Ordering {
        left.cmp(right)
    }
}

impl spillover::key::KeyedOrder for WitnessOrder {
    type StoredKey = Vec<u8>;

    fn compare_keys(&self, left: &Vec<u8>, right: &Vec<u8>) -> Ordering {
        left.cmp(right)
    }
}

impl spillover::key::DeriveKey<u64> for WitnessOrder {
    fn derive_key(&self, item: &u64) -> Vec<u8> {
        let mut key = vec![0; 256];
        key[0] = u8::try_from((*item).min(63)).expect("capped weight fits");
        key
    }
}

impl spillover::key::EstimateKeyMemory<u64> for WitnessOrder {
    fn estimate_key_memory(&self, _: &u64) -> spillover::key::KeyMemory {
        spillover::key::KeyMemory::retained_heap(256)
    }
}

struct WitnessSort;

impl<C: Compare<spillover::key::Keyed<usize, Vec<u8>>> + ?Sized>
    ChunkSorter<spillover::key::Keyed<usize, Vec<u8>>, C> for WitnessSort
{
    fn sort(&self, items: &mut [spillover::key::Keyed<usize, Vec<u8>>], cmp: &C) {
        let weight = items.iter().fold(0_u8, |total, entry| {
            total.saturating_add(entry.stored_key()[0]).min(63)
        });
        let scratch = if weight >= 63 {
            vec![0_u8; 600]
        } else {
            Vec::new()
        };
        std::hint::black_box(&scratch);
        Sequential.sort(items, cmp);
        std::hint::black_box(scratch);
    }
}

impl SortMemorySummary for WitnessSort {
    type Summary = WitnessSummary;

    fn summary(&self) -> WitnessSummary {
        WitnessSummary(Box::new([0; 32]))
    }

    fn summary_allowance(&self, _: Option<&WitnessSummary>) -> (usize, usize) {
        (32, 0)
    }
}

impl SummarizeSortInput<u64> for WitnessSort {
    fn summarize(&self, input: &u64) -> WitnessSummary {
        let mut summary = Box::new([0; 32]);
        summary[0] = u8::try_from((*input).min(63)).expect("capped weight fits");
        WitnessSummary(summary)
    }

    fn summary_allowance_for(&self, _: &WitnessSummary, _: &u64) -> (usize, usize) {
        (32, 0)
    }
}

impl EstimateSortMemory<spillover::key::Keyed<usize, Vec<u8>>, WitnessSummary> for WitnessSort {
    fn sort_memory(&self, _: usize, summary: &WitnessSummary) -> SortMemory {
        let bytes = if summary.0[0] >= 63 { 600 } else { 0 };
        SortMemory::temporary(bytes)
    }
}

#[derive(Debug, Error)]
enum WitnessError {
    #[error("old rejected")]
    OldRejected,
    #[error("new rejected")]
    NewRejected,
    #[error("unexpected input {0}")]
    UnexpectedInput(u64),
    #[error("unexpected codec operation")]
    UnexpectedOperation,
}

#[derive(Clone, Copy)]
struct WitnessCodec;
struct WitnessCursor<R>(std::marker::PhantomData<R>);
struct WitnessWriter<W>(std::marker::PhantomData<W>);

impl Codec for WitnessCodec {
    type Error = WitnessError;
    type Cursor<R: Read> = WitnessCursor<R>;

    fn cursor<R: Read>(&self, _: R) -> Result<Self::Cursor<R>, Self::Error> {
        Err(WitnessError::UnexpectedOperation)
    }
}

impl KeyedCodec for WitnessCodec {
    type Key = Vec<u8>;
    type KeyedCursor<R: Read> = WitnessCursor<R>;

    fn keyed_cursor<R: Read>(&self, _: R) -> Result<Self::KeyedCursor<R>, Self::Error> {
        Err(WitnessError::UnexpectedOperation)
    }
}

impl<R: Read> CodecCursor for WitnessCursor<R> {
    type Record = u64;
    type Error = WitnessError;

    fn advance(&mut self) -> Result<bool, Self::Error> {
        Err(WitnessError::UnexpectedOperation)
    }

    fn with_current<Output>(
        &mut self,
        _: impl FnOnce(&u64) -> Output,
    ) -> Result<Output, Self::Error> {
        Err(WitnessError::UnexpectedOperation)
    }
}

impl<R: Read> KeyedCodecCursor<Vec<u8>> for WitnessCursor<R> {
    fn current_key(&self) -> Result<Vec<u8>, Self::Error> {
        Err(WitnessError::UnexpectedOperation)
    }
}

impl EncodeKeyedRun<u64> for WitnessCodec {
    type KeyedWriter<W: Write> = WitnessWriter<W>;

    fn keyed_writer<W: Write>(&self, _: W) -> Self::KeyedWriter<W> {
        WitnessWriter(std::marker::PhantomData)
    }
}

impl<W: Write> KeyedCodecWriter<u64, Vec<u8>> for WitnessWriter<W> {
    type Error = WitnessError;

    fn write_keyed(
        &mut self,
        entry: spillover::key::Keyed<&u64, &Vec<u8>>,
    ) -> Result<(), Self::Error> {
        match **entry.item() {
            32 => Err(WitnessError::OldRejected),
            31 => Err(WitnessError::NewRejected),
            value => Err(WitnessError::UnexpectedInput(value)),
        }
    }

    fn finish(self) -> Result<(), Self::Error> {
        Err(WitnessError::UnexpectedOperation)
    }
}

#[test]
fn keyed_from_inputs_preflushes_old_window_with_heap_and_scratch() {
    // The two key heaps and sort scratch alone require 1112 bytes.
    // On 64-bit layouts the singleton needs 329 bytes; omitting key heap
    // or sort scratch from the pair leaves 712 or 625 bytes, respectively.
    let mut merge_config = MergeConfig::default();
    merge_config.write_buffer_bytes = 1;
    let mut sorter = spillover::sorter::Builder::new()
        .order(WitnessOrder)
        .keyed_codec(WitnessCodec)
        .chunk_sort(WitnessSort)
        .merge_config(merge_config)
        .memory_budget(ByteLimit::new(1100).from_inputs())
        .build();

    sorter.push(32_u64).expect("first input fits");
    let result = sorter.push(31_u64);
    assert!(
        matches!(result, Err(MergeError::Codec(WitnessError::OldRejected))),
        "expected old-window rejection, got {result:?}"
    );
}

fn make_record(name: &[u8], seq: &[u8], qual: &[u8]) -> SeqRecord {
    SeqRecord::new(name, seq, qual)
}

struct DownstreamNameOrder {
    descending: Box<bool>,
}

impl<R: SeqRecordParts> Compare<R> for DownstreamNameOrder {
    fn compare(&self, left: &R, right: &R) -> Ordering {
        let ordering = left.name().cmp(right.name());
        if *self.descending {
            ordering.reverse()
        } else {
            ordering
        }
    }
}

impl OrderCodecPolicy for DownstreamNameOrder {
    type Policy = RawRecords;
}

#[test]
fn downstream_non_copy_order_and_chunk_sorter_work_with_owned_and_arena_storage() {
    let mut sorter = Builder::new()
        .sort_by_unkeyed(DownstreamNameOrder {
            descending: Box::new(true),
        })
        .chunk_sort(ComparisonSort)
        .max_buffer_items(2)
        .build();
    for name in [b"b", b"a", b"c", b"d", b"e"] {
        sorter.push(make_record(name, b"A", b"!")).expect("push");
    }
    let mut sink = VecSink::default();
    sorter
        .finish()
        .expect("finish")
        .items()
        .try_for_each(|record| sink.write_record(record))
        .expect("borrowed owned-buffer output");
    let names: Vec<&[u8]> = sink.records.iter().map(SeqRecord::name).collect();
    assert_eq!(names, [b"e", b"d", b"c", b"b", b"a"]);

    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_unkeyed(DownstreamNameOrder {
            descending: Box::new(false),
        })
        .chunk_sort(ComparisonSort)
        .arena(&mut arena)
        .max_buffer_items(2)
        .build();
    for name in [b"b", b"a", b"c", b"d", b"e"] {
        let record = SeqRecordView::new(name, b"A", b"!");
        sorter.push(&record).expect("push borrowed view");
    }
    let mut sink = VecSink::default();
    sorter
        .finish()
        .expect("finish")
        .items()
        .try_for_each(|record| sink.write_record(record))
        .expect("borrowed arena-buffer output");
    assert!(arena.is_empty(), "finish must release arena records");
    let names: Vec<&[u8]> = sink.records.iter().map(SeqRecord::name).collect();
    assert_eq!(names, [b"a", b"b", b"c", b"d", b"e"]);
}

#[test]
fn downstream_chunk_sorter_obeys_keyed_full_record_tiebreaks_across_spills() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .chunk_sort(ComparisonSort)
        .max_buffer_items(2)
        .build();
    for record in [
        make_record(b"c", b"C", b"!"),
        make_record(b"n", b"N", b"!"),
        make_record(b"a-high", b"A", b"#"),
        make_record(b"a-low", b"A", b"!"),
    ] {
        sorter.push(record).expect("push");
    }
    let records: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|record| record.expect("decode"))
        .collect();
    assert_eq!(
        records,
        [
            make_record(b"a-low", b"A", b"!"),
            make_record(b"a-high", b"A", b"#"),
            make_record(b"n", b"N", b"!"),
            make_record(b"c", b"C", b"!"),
        ]
    );
}

#[test]
fn owned_unkeyed_budget_includes_spill_writer() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(manifest.is_file());
    let writer_bytes = 1009_usize;
    let record_size = std::mem::size_of::<SeqRecord>();
    assert!(record_size > 0);
    let two_records = record_size.checked_mul(2).expect("record sizes should fit");
    assert!(two_records < writer_bytes + 1);
    let budget = writer_bytes
        .checked_add(512)
        .and_then(|bytes| bytes.checked_add(record_size))
        .and_then(|bytes| bytes.checked_add(1))
        .expect("budget should fit");
    let mut merge_config = MergeConfig::default();
    merge_config.temp_dir = Some(manifest);
    merge_config.write_buffer_bytes = writer_bytes;
    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .chunk_sort(Sequential)
        .merge_config(merge_config)
        .measured_budget(budget)
        .build();

    let sequence = [b'A'; 127];
    let quality = [b'!'; 127];
    let first = SeqRecord::new(b"r1", sequence.as_slice(), quality.as_slice());
    let second = SeqRecord::new(b"r2", sequence.as_slice(), quality.as_slice());
    sorter.push(first).expect("first push should succeed");
    let result = sorter.push(second);
    assert!(
        matches!(result, Err(MergeError::Io(_))),
        "second push should attempt a spill, got {result:?}"
    );
}

#[derive(Default)]
struct VecSink {
    records: Vec<SeqRecord>,
}

impl SeqRecordSink for VecSink {
    type Error = std::convert::Infallible;

    fn write_record<R: SeqRecordParts + ?Sized>(&mut self, record: &R) -> Result<(), Self::Error> {
        self.records.push(SeqRecord::from_slices(
            record.name(),
            record.sequence(),
            record.quality(),
        ));

        Ok(())
    }
}

struct FailingSink;

impl SeqRecordSink for FailingSink {
    type Error = std::io::Error;

    fn write_record<R: SeqRecordParts + ?Sized>(&mut self, _record: &R) -> Result<(), Self::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "destination closed",
        ))
    }
}

#[test]
fn unsorted_records_come_out_sorted_by_sequence() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    let records = vec![
        make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
        make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
        make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
    ];

    for rec in records {
        sorter.push(rec).expect("push should succeed");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|r| r.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn owned_keyed_sorter_accepts_explicitly_owned_record_views() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    let records = [
        make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
        make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
        make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
    ];

    for rec in &records {
        let view = rec.as_view();
        let owned_record = view.into();
        sorter.push(owned_record).expect("push should succeed");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|r| r.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn owned_unkeyed_sorter_accepts_explicitly_owned_record_views() {
    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    let records = [
        make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
        make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
        make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
    ];

    for rec in &records {
        let view = rec.as_view();
        let owned_record = view.into();
        sorter.push(owned_record).expect("push should succeed");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|r| r.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn arena_keyed_sorter_sorts_borrowed_record_views() {
    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .arena(&mut arena)
        .max_buffer_items(2)
        .build();

    let records = [
        make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
        make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
        make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
    ];

    for rec in &records {
        sorter.push(&rec.as_view()).expect("push should succeed");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|r| r.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn arena_keyed_radix_chunk_preserves_record_identity() {
    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .arena(&mut arena)
        .max_buffer_items(65)
        .build();

    for quality in (b'!'..=b'1').rev() {
        for sequence in [b"TTTT", b"GGGG", b"CCCC", b"AAAA"] {
            let name = [sequence[0], quality];
            let qualities = [quality; 4];
            let record = SeqRecordView::new(&name, sequence, &qualities);
            sorter.push(&record).expect("push should succeed");
        }
    }

    let mut output = sorter.finish().expect("finish should succeed");
    for sequence in [b"AAAA", b"CCCC", b"GGGG", b"TTTT"] {
        for quality in b'!'..=b'1' {
            let expected_name = [sequence[0], quality];
            let expected_quality = [quality; 4];
            let record = output
                .next()
                .expect("record should exist")
                .expect("record should decode");
            assert_eq!(record.name(), expected_name);
            assert_eq!(record.sequence(), sequence);
            assert_eq!(record.quality(), expected_quality);
        }
    }
    assert!(output.next().is_none(), "output should be exhausted");
}

#[test]
fn arena_keyed_default_radix_spills_follow_complete_sequence_order_above_fallback() {
    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .arena(&mut arena)
        .max_buffer_items(65)
        .build();

    for (name, sequence) in [(b"c", b"C"), (b"n", b"N")] {
        let record = SeqRecordView::new(name, sequence, b"I");
        sorter.push(&record).expect("push should succeed");
    }
    for quality in (b'!'..=b'_').rev() {
        let name = [b'A', quality];
        let qualities = [quality];
        let record = SeqRecordView::new(&name, b"A", &qualities);
        sorter.push(&record).expect("push should succeed");
    }
    let name = [b'A', b'`'];
    let qualities = [b'`'];
    let record = SeqRecordView::new(&name, b"A", &qualities);
    sorter.push(&record).expect("push should succeed");

    let mut output = sorter.finish().expect("finish should succeed");
    for quality in b'!'..=b'`' {
        let record = output.next().expect("record should exist").expect("decode");
        assert_eq!(record.name(), [b'A', quality]);
        assert_eq!(record.sequence(), b"A");
        assert_eq!(record.quality(), [quality]);
    }
    for (name, sequence) in [(b"n", b"N"), (b"c", b"C")] {
        let record = output.next().expect("record should exist").expect("decode");
        assert_eq!(record.name(), name);
        assert_eq!(record.sequence(), sequence);
        assert_eq!(record.quality(), b"I");
    }
    assert!(output.next().is_none(), "output should be exhausted");
}

#[test]
fn arena_unkeyed_sort_preserves_all_empty_record() {
    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .arena(&mut arena)
        .max_buffer_items(1)
        .build();

    let record = SeqRecordView::new(b"", b"", b"");
    sorter.push(&record).expect("push should succeed");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|record| record.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name(), b"");
    assert_eq!(results[0].sequence(), b"");
    assert_eq!(results[0].quality(), b"");
}

#[test]
fn arena_unkeyed_sorter_sorts_borrowed_record_views() {
    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .arena(&mut arena)
        .max_buffer_items(2)
        .build();

    let records = [
        make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
        make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
        make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
    ];

    for rec in &records {
        sorter.push(&rec.as_view()).expect("push should succeed");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|r| r.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn arena_keyed_measured_ignores_retained_payload_after_clear() {
    let mut arena = SeqRecordArena::with_capacity(8192);
    let capacity = arena.capacity();
    assert!(capacity >= 8192);

    let mut merge_config = MergeConfig::default();
    merge_config.write_buffer_bytes = 0;
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .sort_with_sequential()
        .codec(DryIceCodec::new())
        .merge_config(merge_config)
        .arena(&mut arena)
        .measured_budget(1)
        .build();
    let seed = SeqRecordView::new(b"seed", b"A", b"!");
    sorter.push(&seed).expect("seed push should succeed");
    drop(sorter);

    assert_eq!(arena.byte_len(), 0);
    assert_eq!(arena.capacity(), capacity);

    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(manifest.is_file());
    let mut merge_config = MergeConfig::default();
    merge_config.temp_dir = Some(manifest);
    merge_config.write_buffer_bytes = 0;
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .sort_with_sequential()
        .codec(DryIceCodec::new())
        .merge_config(merge_config)
        .arena(&mut arena)
        .measured_budget(capacity / 2)
        .build();
    let record = SeqRecordView::new(b"s", b"A", b"!");
    sorter.push(&record).expect("tiny push should succeed");
    drop(sorter);

    assert_eq!(arena.byte_len(), 3);
}

#[test]
fn arena_keyed_measured_admits_oversized_before_failed_commit() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(manifest.is_file());
    let mut merge_config = MergeConfig::default();
    merge_config.temp_dir = Some(manifest);
    merge_config.write_buffer_bytes = 0;

    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .sort_with_sequential()
        .codec(DryIceCodec::new())
        .merge_config(merge_config)
        .arena(&mut arena)
        .measured_budget(1)
        .build();

    let record = SeqRecordView::new(b"oversized", b"ACGT", b"!!!!");
    let result = sorter.push(&record);
    assert!(
        matches!(result, Err(MergeError::Io(_))),
        "push should fail during commit, got {result:?}"
    );

    drop(sorter);
    assert_eq!(arena.byte_len(), 17);
}

#[test]
fn arena_keyed_measured_preflushes_before_payload_replacement() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(manifest.is_file());
    let mut merge_config = MergeConfig::default();
    merge_config.temp_dir = Some(manifest);
    merge_config.write_buffer_bytes = 64;

    let mut arena = SeqRecordArena::with_capacity(4096);
    let capacity = arena.capacity();
    assert!(capacity >= 4096);
    let budget = capacity.checked_mul(2).expect("budget should fit");
    let name_length = capacity.checked_add(1).expect("name length should fit");
    let name = vec![b'n'; name_length];

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .chunk_sort(Sequential)
        .merge_config(merge_config)
        .arena(&mut arena)
        .measured_budget(budget)
        .build();

    let first = SeqRecordView::new(b"a", b"AC", b"!!");
    sorter.push(&first).expect("first push should succeed");

    let second = SeqRecordView::new(&name, b"A", b"!");
    let second_result = sorter.push(&second);
    assert!(
        matches!(second_result, Err(MergeError::Io(_))),
        "second push should fail during preflush, got {second_result:?}"
    );

    drop(sorter);
    assert_eq!(arena.byte_len(), 5);
}

#[test]
fn arena_unkeyed_measured_preflushes_before_payload_replacement() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(manifest.is_file());
    let mut merge_config = MergeConfig::default();
    merge_config.temp_dir = Some(manifest);
    merge_config.write_buffer_bytes = 64;

    let mut arena = SeqRecordArena::with_capacity(4096);
    let capacity = arena.capacity();
    assert!(capacity >= 4096);
    let budget = capacity.checked_mul(2).expect("budget should fit");
    let name_length = capacity.checked_add(1).expect("name length should fit");
    let name = vec![b'n'; name_length];

    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .chunk_sort(Sequential)
        .merge_config(merge_config)
        .arena(&mut arena)
        .measured_budget(budget)
        .build();

    let first = SeqRecordView::new(b"a", b"AC", b"!!");
    sorter.push(&first).expect("first push should succeed");

    let second = SeqRecordView::new(&name, b"A", b"!");
    let second_result = sorter.push(&second);
    assert!(
        matches!(second_result, Err(MergeError::Io(_))),
        "second push should fail during preflush, got {second_result:?}"
    );

    drop(sorter);
    assert_eq!(arena.byte_len(), 5);
}

#[test]
fn arena_unkeyed_item_retry_does_not_admit_on_failed_spill() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(manifest.is_file());
    let mut merge_config = MergeConfig::default();
    merge_config.temp_dir = Some(manifest);
    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .merge_config(merge_config)
        .arena(&mut arena)
        .max_buffer_items(1)
        .build();

    let first = SeqRecordView::new(b"a", b"AC", b"!!");
    let first_result = sorter.push(&first);
    assert!(matches!(first_result, Err(MergeError::Io(_))));

    let second = SeqRecordView::new(b"second", b"G", b"#");
    let second_result = sorter.push(&second);
    assert!(matches!(second_result, Err(MergeError::Io(_))));

    drop(sorter);
    assert_eq!(arena.byte_len(), 5);
}

#[test]
fn terminal_build_clears_preexisting_arena_records() {
    let mut arena = SeqRecordArena::new();
    let order = ILLUMINA_ORDER.unkeyed();
    let codec = DryIceCodec::new();
    let mut seeding_sorter = Builder::new()
        .sort_by_unkeyed(order)
        .codec(codec)
        .arena(&mut arena)
        .max_buffer_items(2)
        .build();

    let old_record = SeqRecordView::new(b"old", b"A", b"I");
    seeding_sorter
        .push(&old_record)
        .expect("seed should succeed");
    drop(seeding_sorter);
    assert_eq!(arena.byte_len(), 5, "fixture must retain the old record");

    let fresh_sorter = Builder::new()
        .sort_by_unkeyed(order)
        .codec(codec)
        .arena(&mut arena)
        .max_buffer_items(2)
        .build();
    drop(fresh_sorter);

    assert_eq!(arena.byte_len(), 0);
}

#[test]
fn arena_sorter_clears_arena_after_finish() {
    let mut arena = SeqRecordArena::new();
    {
        let mut sorter = Builder::new()
            .sort_by_illumina()
            .codec(DryIceCodec::new())
            .arena(&mut arena)
            .max_buffer_items(10)
            .build();

        sorter
            .push(&SeqRecordView::new(b"r1", b"AAAAAAAA", b"!!!!!!!!"))
            .expect("push should succeed");

        let results: Vec<SeqRecord> = sorter
            .finish()
            .expect("finish should succeed")
            .map(|r| r.expect("each record should decode"))
            .collect();

        assert_eq!(results.len(), 1);
    }

    assert!(arena.is_empty());
}

#[test]
fn sorted_record_stream_writes_to_custom_sink() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    sorter
        .push(make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"))
        .expect("push");

    let mut sink = VecSink::default();
    sorter
        .finish()
        .expect("finish should succeed")
        .items()
        .try_for_each(|record| sink.write_record(record))
        .expect("borrowed traversal should succeed");

    assert_eq!(sink.records.len(), 3);
    assert_eq!(sink.records[0].sequence(), b"AAAAAAAA");
    assert_eq!(sink.records[1].sequence(), b"CCCCCCCC");
    assert_eq!(sink.records[2].sequence(), b"TTTTTTTT");
}

#[test]
fn mixed_output_preserves_name_history() {
    let mut sorter = Builder::new()
        .sort_by_name()
        .dedup_by_name()
        .codec(DryIceCodec::new())
        .max_buffer_items(1)
        .build();

    let records = [
        make_record(b"a", b"AAAA", b"!!!!"),
        make_record(b"a", b"AAAA", b"!!!!"),
        make_record(b"b", b"CCCC", b"####"),
    ];
    for record in records {
        sorter.push(record).expect("push should succeed");
    }

    let mut stream = sorter.finish().expect("finish should succeed");
    let first = stream
        .next()
        .expect("first record should exist")
        .expect("first record should decode");
    assert_eq!(first.name(), b"a");
    assert_eq!(first.sequence(), b"AAAA");
    assert_eq!(first.quality(), b"!!!!");

    let mut sink = VecSink::default();
    stream
        .items()
        .try_for_each(|record| sink.write_record(record))
        .expect("borrowed traversal should succeed");

    assert_eq!(sink.records.len(), 1);
    assert_eq!(sink.records[0].name(), b"b");
    assert_eq!(sink.records[0].sequence(), b"CCCC");
    assert_eq!(sink.records[0].quality(), b"####");
}

#[test]
fn mixed_output_preserves_adjacent_sequence_history_without_global_dedup() {
    let mut sorter = Builder::new()
        .sort_by_name()
        .dedup_by_sequence()
        .max_buffer_items(1)
        .build();
    for record in [
        make_record(b"d", b"A", b"$"),
        make_record(b"c", b"C", b"#"),
        make_record(b"b", b"A", b"!"),
        make_record(b"a", b"A", b"!"),
    ] {
        sorter.push(record).expect("push");
    }

    let mut output = sorter.finish().expect("finish");
    let first = output.next().expect("first record").expect("decode");
    assert_eq!(first.name(), b"a");
    assert_eq!(first.sequence(), b"A");
    assert_eq!(first.quality(), b"!");

    let mut sink = VecSink::default();
    output
        .items()
        .try_for_each(|record| sink.write_record(record))
        .expect("borrowed traversal");

    // Dedup compares adjacent records: the A after C must survive.
    assert_eq!(
        sink.records,
        [make_record(b"c", b"C", b"#"), make_record(b"d", b"A", b"$")]
    );
}

struct BioEvents(std::vec::IntoIter<Result<SeqRecord, &'static str>>);

impl Iterator for BioEvents {
    type Item = Result<SeqRecord, &'static str>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl VisitSortedItems for BioEvents {
    type Record = SeqRecord;
    type Error = &'static str;

    fn visit_items<F, FE>(self, mut callback: F) -> Result<(), SortedItemsError<Self::Error, FE>>
    where
        F: FnMut(&Self::Record) -> Result<(), FE>,
    {
        for event in self {
            let record = event.map_err(SortedItemsError::Source)?;
            callback(&record).map_err(SortedItemsError::Sink)?;
        }
        Ok(())
    }
}

#[test]
fn bio_dedup_source_error_fuses_owned_and_borrowed_delivery() {
    let events = vec![
        Ok(make_record(b"a", b"A", b"!")),
        Err("source failed"),
        Ok(make_record(b"c", b"C", b"!")),
    ];
    let source = BioEvents(events.into_iter());
    let mut output = RecordDedup::Sequence.dedup(source);
    assert_eq!(
        output.next().expect("first event").expect("record"),
        make_record(b"a", b"A", b"!")
    );
    assert!(matches!(output.next(), Some(Err("source failed"))));
    assert!(output.next().is_none(), "source errors must fuse iteration");

    let mut visited = 0;
    output
        .visit_items(|_| -> Result<(), Infallible> {
            visited += 1;
            Ok(())
        })
        .expect("fused borrowed traversal");
    assert_eq!(visited, 0, "borrowed traversal must not resume the source");
}

#[test]
fn bio_dedup_borrowed_delivery_preserves_source_and_sink_errors() {
    let events = vec![
        Ok(make_record(b"a", b"A", b"!")),
        Err("source failed"),
        Ok(make_record(b"c", b"C", b"!")),
    ];
    let source = BioEvents(events.into_iter());
    let output = RecordDedup::Sequence.dedup(source);
    let mut visited = Vec::new();
    let result = output.visit_items(|record| -> Result<(), Infallible> {
        visited.push(record.sequence().to_vec());
        Ok(())
    });
    assert_eq!(visited, [b"A"]);
    assert!(matches!(
        result,
        Err(SortedItemsError::Source("source failed"))
    ));

    let events = vec![
        Ok(make_record(b"a", b"A", b"!")),
        Ok(make_record(b"c", b"C", b"!")),
    ];
    let source = BioEvents(events.into_iter());
    let output = RecordDedup::Sequence.dedup(source);
    let mut visited = 0;
    let result = output.visit_items(|_| {
        visited += 1;
        Err::<(), _>(Stop(42))
    });
    assert_eq!(visited, 1, "sink failure must stop delivery immediately");
    assert!(matches!(result, Err(SortedItemsError::Sink(Stop(42)))));
}

#[test]
fn sorted_record_stream_applies_dedup_when_writing_to_sink() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .dedup_by_sequence()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    sorter
        .push(make_record(b"r1", b"AAAAAAAA", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"CCCCCCCC", b"!!!!!!!!"))
        .expect("push");

    let mut sink = VecSink::default();
    sorter
        .finish()
        .expect("finish should succeed")
        .items()
        .try_for_each(|record| sink.write_record(record))
        .expect("borrowed traversal should succeed");

    assert_eq!(sink.records.len(), 2);
    assert_eq!(sink.records[0].sequence(), b"AAAAAAAA");
    assert_eq!(sink.records[1].sequence(), b"CCCCCCCC");
}

#[test]
fn sorted_record_stream_wraps_sink_errors() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    sorter
        .push(make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");

    let mut sink = FailingSink;
    let err = sorter
        .finish()
        .expect("finish should succeed")
        .items()
        .try_for_each(|record| sink.write_record(record))
        .expect_err("sink error should be wrapped");

    assert!(
        matches!(err, SortedItemsError::Sink(error) if error.kind() == io::ErrorKind::BrokenPipe)
    );
}

#[test]
fn sorted_record_stream_writes_to_dryice_writer() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    sorter
        .push(make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"))
        .expect("push");

    let mut output = Vec::new();
    let mut writer = dryice::DryIceWriter::builder().inner(&mut output).build();
    sorter
        .finish()
        .expect("finish should succeed")
        .items()
        .try_for_each(|record| SeqRecordSink::write_record(&mut writer, record))
        .expect("borrowed traversal should succeed");
    writer.finish().expect("writer should finish");

    let mut reader = dryice::DryIceReader::new(output.as_slice()).expect("reader should open");
    let mut sequences = Vec::new();
    while reader.next_record().expect("reader should advance") {
        sequences.push(reader.sequence().to_vec());
    }

    assert_eq!(sequences, [b"AAAAAAAA", b"CCCCCCCC", b"TTTTTTTT"]);
}

#[test]
fn sort_with_disk_spilling() {
    // Buffer holds 3 records, input has 10 → multiple flushes to disk.
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(3)
        .build();

    let bases = [b'A', b'C', b'G', b'T'];
    for i in (0..10).rev() {
        let seq: Vec<u8> = std::iter::repeat_n(bases[i % 4], 8).collect();
        let qual = vec![b'!'; 8];
        sorter
            .push(make_record(format!("r{i}").as_bytes(), &seq, &qual))
            .expect("push should succeed");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|r| r.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 10);
    // Verify sorted order
    for window in results.windows(2) {
        assert!(
            window[0].sequence() <= window[1].sequence(),
            "output should be sorted: {:?} should come before {:?}",
            std::str::from_utf8(window[0].sequence()),
            std::str::from_utf8(window[1].sequence()),
        );
    }
}

#[test]
fn owned_keyed_spills_follow_complete_sequence_order() {
    let mut sorter = Builder::new()
        .sort_by(ILLUMINA_ORDER)
        .chunk_sort(Sequential)
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    // C and N must share the first initial run.
    let records = [
        SeqRecord::new(b"c", b"C", b"I"),
        SeqRecord::new(b"n", b"N", b"I"),
        SeqRecord::new(b"a", b"A", b"I"),
    ];
    for record in records {
        sorter.push(record).expect("push should succeed");
    }

    let output = sorter.finish().expect("finish should succeed");
    let sequences: Vec<Vec<u8>> = output
        .map(|result| {
            let record = result.expect("record should decode");
            record.sequence().to_vec()
        })
        .collect();

    assert_eq!(sequences, [b"A", b"N", b"C"]);
}

#[test]
fn owned_keyed_default_radix_spills_follow_complete_sequence_order_below_fallback() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    let records = [
        SeqRecord::new(b"c", b"C", b"I"),
        SeqRecord::new(b"n", b"N", b"I"),
        SeqRecord::new(b"a", b"A", b"I"),
    ];
    for record in records {
        sorter.push(record).expect("push should succeed");
    }

    let mut output = sorter.finish().expect("finish should succeed");
    for (name, sequence, quality) in [(b"a", b"A", b"I"), (b"n", b"N", b"I"), (b"c", b"C", b"I")] {
        let record = output.next().expect("record should exist").expect("decode");
        assert_eq!(record.name(), name);
        assert_eq!(record.sequence(), sequence);
        assert_eq!(record.quality(), quality);
    }
    assert!(output.next().is_none());
}

#[test]
fn quality_tiebreaker_orders_identical_sequences() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"r1", b"ACGTACGT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"ACGTACGT", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"ACGTACGT", b"########"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 3);
    // All same sequence, ordered by quality: ! < # < I in ASCII
    assert_eq!(results[0].quality(), b"!!!!!!!!");
    assert_eq!(results[1].quality(), b"########");
    assert_eq!(results[2].quality(), b"IIIIIIII");
}

#[test]
fn keyed_merge_preserves_quality_tiebreak_across_run_boundaries() {
    // Buffer holds 2 records, so this input spills into multiple runs.
    // All sequences are identical; merge must use full-record fallback
    // when packed keys tie to preserve quality ordering.
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    sorter
        .push(make_record(b"r1", b"ACGTACGT", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"ACGTACGT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"ACGTACGT", b"########"))
        .expect("push");
    sorter
        .push(make_record(b"r4", b"ACGTACGT", b"JJJJJJJJ"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 4);
    assert_eq!(results[0].quality(), b"!!!!!!!!");
    assert_eq!(results[1].quality(), b"########");
    assert_eq!(results[2].quality(), b"IIIIIIII");
    assert_eq!(results[3].quality(), b"JJJJJJJJ");
}

#[test]
fn sort_by_name_orders_lexicographically() {
    let mut input = vec![
        make_record(b"charlie", b"ACGT", b"!!!!"),
        make_record(b"alice", b"TGCA", b"!!!!"),
        make_record(b"bob", b"GGGG", b"!!!!"),
        make_record(b"read_010", b"AAAA", b"!!!!"),
        make_record(b"read_002", b"CCCC", b"!!!!"),
        make_record(b"read_001", b"TTTT", b"!!!!"),
    ];

    let mut sorter = Builder::new()
        .sort_by_name()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    input.sort_by(|a, b| NAME_ORDER.compare(a, b));

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(got.name(), expected.name(), "name mismatch at position {i}");
    }
}

#[test]
fn reverse_sequence_order_produces_descending_output() {
    let mut input = vec![
        make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
        make_record(b"r2", b"TTTTTTTT", b"!!!!!!!!"),
        make_record(b"r3", b"CCCCCCCC", b"!!!!!!!!"),
        make_record(b"r4", b"ACGTACGT", b"IIIIIIII"),
        make_record(b"r5", b"ACGTACGT", b"########"),
    ];

    let mut sorter = Builder::new()
        .sort_by(Reverse(ILLUMINA_ORDER))
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    input.sort_by(|a, b| Reverse(ILLUMINA_ORDER).compare(a, b));

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i}"
        );
        assert_eq!(
            got.quality(),
            expected.quality(),
            "quality mismatch at position {i}"
        );
    }
}

#[test]
fn sort_by_length_orders_shortest_first() {
    let input = vec![
        make_record(b"len12_a", b"ACGTACGTACGT", b"!!!!!!!!!!!!"),
        make_record(b"len4_a", b"ACGT", b"!!!!"),
        make_record(b"len8", b"ACGTACGT", b"!!!!!!!!"),
        make_record(b"len1", b"A", b"!"),
        make_record(b"len4_b", b"TGCA", b"!!!!"),
        make_record(b"len12_b", b"TGCATGCATGCA", b"!!!!!!!!!!!!"),
    ];

    let mut sorter = Builder::new()
        .sort_by(LENGTH_ORDER)
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    let result_lengths: Vec<_> = results.iter().map(|r| r.sequence().len()).collect();
    let mut expected_lengths: Vec<_> = input.iter().map(|r| r.sequence().len()).collect();
    expected_lengths.sort_unstable();

    assert_eq!(result_lengths, expected_lengths);
}

#[test]
fn dedup_removes_duplicate_sequences() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .dedup_by_sequence()
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"r1", b"ACGTACGT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"ACGTACGT", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r4", b"TTTTTTTT", b"########"))
        .expect("push");
    sorter
        .push(make_record(b"r5", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(
        results.len(),
        3,
        "duplicates should be collapsed to one per unique sequence"
    );
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"ACGTACGT");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn empty_input_produces_empty_output() {
    let sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert!(results.is_empty());
}

#[test]
fn large_dataset_matches_reference_sort() {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut input: Vec<SeqRecord> = (0..500)
        .map(|i| {
            let seq: Vec<u8> = (0..16).map(|j| bases[(i * 7 + j * 13) % 4]).collect();
            let qual: Vec<u8> = (0..16)
                .map(|j| b'!' + u8::try_from((i + j) % 40).expect("fits"))
                .collect();
            make_record(format!("r{i:04}").as_bytes(), &seq, &qual)
        })
        .collect();

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(50)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    input.sort_by(|a, b| ILLUMINA_ORDER.compare(a, b));

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i}"
        );
        assert_eq!(
            got.quality(),
            expected.quality(),
            "quality mismatch at position {i}"
        );
    }
}

#[test]
fn spilling_with_many_flushes_preserves_all_records() {
    // Buffer holds 2 records, input has 20 → 10 flushes.
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    let bases = [b'A', b'C', b'G', b'T'];
    for i in 0..20 {
        let seq: Vec<u8> = (0..8).map(|j| bases[(i * 3 + j * 7) % 4]).collect();
        let qual = vec![b'!'; 8];
        sorter
            .push(make_record(format!("r{i:02}").as_bytes(), &seq, &qual))
            .expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(
        results.len(),
        20,
        "all records should survive multiple flushes and merge"
    );

    for window in results.windows(2) {
        assert!(
            (window[0].sequence(), window[0].quality())
                <= (window[1].sequence(), window[1].quality()),
            "output should be sorted"
        );
    }
}

#[test]
fn compact_codecs_round_trip_correctly() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(
            DryIceCodec::new()
                .two_bit_exact()
                .binned_quality()
                .split_names(),
        )
        .max_buffer_items(100)
        .build();

    let records = vec![
        make_record(
            b"instrument:run:flowcell 1:N:0:ATCACG",
            b"TTTTTTTTTTTTTTTT",
            b"IIIIIIIIIIIIIIII",
        ),
        make_record(
            b"instrument:run:flowcell 2:N:0:ATCACG",
            b"AAAAAAAAAAAAAAAA",
            b"!!!!!!!!!!!!!!!!",
        ),
        make_record(
            b"instrument:run:flowcell 3:N:0:ATCACG",
            b"CCCCCCCCCCCCCCCC",
            b"################",
        ),
    ];

    for rec in records {
        sorter.push(rec).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"AAAAAAAAAAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCCCCCCCCCC");
    assert_eq!(results[2].sequence(), b"TTTTTTTTTTTTTTTT");
}

#[test]
fn realistic_150bp_reads_sort_correctly() {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut input: Vec<SeqRecord> = (0..100)
        .map(|i| {
            let seq: Vec<u8> = (0..150)
                .map(|j| bases[(i * 7 + j * 13 + j / 10) % 4])
                .collect();
            let qual: Vec<u8> = (0..150)
                .map(|j| b'!' + u8::try_from((i + j * 3) % 40).expect("fits"))
                .collect();
            make_record(format!("read_{i:04}").as_bytes(), &seq, &qual)
        })
        .collect();

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(20)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    input.sort_by(|a, b| {
        a.sequence()
            .cmp(b.sequence())
            .then_with(|| a.quality().cmp(b.quality()))
    });

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i}"
        );
        assert_eq!(
            got.quality(),
            expected.quality(),
            "quality mismatch at position {i}"
        );
    }
}

#[test]
fn radix_sort_through_pipeline() {
    let bases = [b'A', b'C', b'G', b'T'];
    let mut input: Vec<SeqRecord> = (0..200)
        .map(|i| {
            let seq: Vec<u8> = (0..16).map(|j| bases[(i * 7 + j * 13) % 4]).collect();
            let qual: Vec<u8> = (0..16)
                .map(|j| b'!' + u8::try_from((i + j) % 40).expect("fits"))
                .collect();
            make_record(format!("r{i:03}").as_bytes(), &seq, &qual)
        })
        .collect();

    let mut sorter = Builder::new()
        .sort_by(SequenceOrder::<4>)
        .codec(DryIceCodec::new())
        .chunk_sort(RadixThenRefine::<4>)
        .max_buffer_items(50)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    input.sort_by(|a, b| SequenceOrder::<4>.compare(a, b));

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i} with radix sort"
        );
    }
}

#[test]
fn sequences_with_ambiguous_bases_sort_correctly() {
    let mut input = vec![
        make_record(b"n", b"NNNNNNNN", b"IIIIIIII"),
        make_record(b"a", b"AAAAAAAA", b"!!!!!!!!"),
        make_record(b"t", b"TTTTTTTT", b"!!!!!!!!"),
        make_record(b"c", b"CCCCCCCC", b"########"),
    ];

    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    input.sort_by(|a, b| ILLUMINA_ORDER.compare(a, b));

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i}"
        );
        assert_eq!(
            got.quality(),
            expected.quality(),
            "quality mismatch at position {i}"
        );
    }
}

#[test]
fn single_record_round_trips() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"only_record", b"ACGTACGT", b"IIIIIIII"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name(), b"only_record");
    assert_eq!(results[0].sequence(), b"ACGTACGT");
    assert_eq!(results[0].quality(), b"IIIIIIII");
}

#[test]
fn all_identical_records_sort_and_preserve_count() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(3)
        .build();

    for i in 0..10 {
        sorter
            .push(make_record(
                format!("r{i}").as_bytes(),
                b"ACGTACGT",
                b"!!!!!!!!",
            ))
            .expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(
        results.len(),
        10,
        "identical records should all be preserved without dedup"
    );
}

#[test]
fn dedup_with_spilling_removes_cross_chunk_duplicates() {
    // Buffer holds 3 records. Duplicates span chunk boundaries.
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .dedup_by_sequence()
        .max_buffer_items(3)
        .build();

    // Push records with duplicates that will end up in different chunks
    for seq in [
        b"AAAAAAAA",
        b"AAAAAAAA",
        b"CCCCCCCC",
        b"CCCCCCCC",
        b"CCCCCCCC",
        b"GGGGGGGG",
        b"TTTTTTTT",
        b"TTTTTTTT",
    ] {
        sorter
            .push(make_record(b"r", seq.as_slice(), &[b'!'; 8]))
            .expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(
        results.len(),
        4,
        "dedup should collapse duplicates even across chunk boundaries"
    );
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"GGGGGGGG");
    assert_eq!(results[3].sequence(), b"TTTTTTTT");
}

#[test]
fn unkeyed_sort_preserves_exact_sequence_and_quality_ordering() {
    // The unkeyed path uses full (sequence, quality) comparison
    // during merge, so sequence+quality ordering is exact across
    // run boundaries.
    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .max_buffer_items(3)
        .build();

    sorter
        .push(make_record(b"r1", b"ACGTACGT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"ACGTACGT", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"ACGTACGT", b"########"))
        .expect("push");
    sorter
        .push(make_record(b"r4", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r5", b"TTTTTTTT", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 5);
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    // ACGT group: quality ordering preserved across runs
    assert_eq!(results[1].sequence(), b"ACGTACGT");
    assert_eq!(results[1].quality(), b"!!!!!!!!");
    assert_eq!(results[2].sequence(), b"ACGTACGT");
    assert_eq!(results[2].quality(), b"########");
    assert_eq!(results[3].sequence(), b"ACGTACGT");
    assert_eq!(results[3].quality(), b"IIIIIIII");
    assert_eq!(results[4].sequence(), b"TTTTTTTT");
}

#[test]
fn exact_sequence_sort_uses_raw_sequence_order() {
    let mut sorter = Builder::new()
        .sort_by_sequence_exact()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    sorter.push(make_record(b"n", b"N", b"!")).expect("push");
    sorter.push(make_record(b"c", b"C", b"!")).expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results[0].sequence(), b"C");
    assert_eq!(results[1].sequence(), b"N");
}

#[test]
fn exact_sequence_sort_matches_reference_across_runs() {
    let mut input = vec![
        make_record(b"n", b"N", b"!"),
        make_record(b"c", b"C", b"!"),
        make_record(b"a_long", b"AAAA", b"####"),
        make_record(b"a_short", b"A", b"I"),
        make_record(b"same_hi", b"ACGT", b"IIII"),
        make_record(b"same_lo", b"ACGT", b"!!!!"),
    ];

    let mut sorter = Builder::new()
        .sort_by_sequence_exact()
        .codec(DryIceCodec::new())
        .max_buffer_items(2)
        .build();

    for rec in &input {
        sorter.push(rec.clone()).expect("push");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    input.sort_by(|a, b| {
        a.sequence()
            .cmp(b.sequence())
            .then_with(|| a.quality().cmp(b.quality()))
    });

    assert_eq!(results.len(), input.len());
    for (i, (got, expected)) in results.iter().zip(input.iter()).enumerate() {
        assert_eq!(
            got.sequence(),
            expected.sequence(),
            "sequence mismatch at position {i}"
        );
        assert_eq!(
            got.quality(),
            expected.quality(),
            "quality mismatch at position {i}"
        );
    }
}

#[test]
fn unkeyed_sort_with_variable_length_sequences() {
    // The unkeyed path doesn't have the packed key padding
    // ambiguity, so variable-length sequences sort correctly.
    let mut sorter = Builder::new()
        .sort_by_unkeyed(ILLUMINA_ORDER.unkeyed())
        .codec(DryIceCodec::new())
        .max_buffer_items(3)
        .build();

    sorter.push(make_record(b"r1", b"TA", b"!!")).expect("push");
    sorter.push(make_record(b"r2", b"T", b"!")).expect("push");
    sorter
        .push(make_record(b"r3", b"TAA", b"!!!"))
        .expect("push");
    sorter.push(make_record(b"r4", b"A", b"!")).expect("push");
    sorter
        .push(make_record(b"r5", b"AAAA", b"!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 5);
    // Lexicographic: A < AAAA < T < TA < TAA
    assert_eq!(results[0].sequence(), b"A");
    assert_eq!(results[1].sequence(), b"AAAA");
    assert_eq!(results[2].sequence(), b"T");
    assert_eq!(results[3].sequence(), b"TA");
    assert_eq!(results[4].sequence(), b"TAA");
}

#[test]
fn sorter_is_send() {
    fn assert_send<T: Send>(_: &T) {}

    let sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new())
        .max_buffer_items(100)
        .build();

    assert_send(&sorter);
}

mod proptests {
    use proptest::prelude::*;

    use super::*;

    fn arb_fixed_len_sequence(len: usize) -> impl Strategy<Value = Vec<u8>> {
        proptest::collection::vec(
            proptest::sample::select(vec![b'A', b'C', b'G', b'T']),
            len..=len,
        )
    }

    fn arb_quality(len: usize) -> impl Strategy<Value = Vec<u8>> {
        proptest::collection::vec(b'!'..=b'I', len..=len)
    }

    /// Generate records with uniform sequence length. Packed key
    /// ordering is exact when all sequences are the same length
    /// (no padding ambiguity with trailing A's).
    fn arb_record(seq_len: usize) -> impl Strategy<Value = SeqRecord> {
        (arb_fixed_len_sequence(seq_len), arb_quality(seq_len))
            .prop_map(|(seq, qual)| SeqRecord::new(b"r", seq, qual))
    }

    proptest! {
        #[test]
        fn output_is_sorted_for_any_input(
            records in proptest::collection::vec(arb_record(32), 1..100),
            buffer_size in 2usize..20,
        ) {
            let mut sorter = Builder::new()
                .sort_by_illumina()
                .codec(DryIceCodec::new())
                .max_buffer_items(buffer_size)
                .build();

            for rec in &records {
                sorter.push(rec.clone()).expect("push");
            }

            let results: Vec<SeqRecord> = sorter
                .finish()
                .expect("finish")
                .map(|r| r.expect("decode"))
                .collect();

            prop_assert_eq!(results.len(), records.len());

            // Keyed merge preserves full sequence+quality ordering,
            // using fallback full-record comparison when keys tie.
            for window in results.windows(2) {
                prop_assert!(
                    (window[0].sequence(), window[0].quality())
                        <= (window[1].sequence(), window[1].quality()),
                    "output must be sorted by sequence then quality"
                );
            }
        }

        #[test]
        fn output_preserves_all_records(
            records in proptest::collection::vec(arb_record(32), 1..100),
            buffer_size in 2usize..20,
        ) {
            let mut sorter = Builder::new()
                .sort_by_illumina()
                .codec(DryIceCodec::new())
                .max_buffer_items(buffer_size)
                .build();

            for rec in &records {
                sorter.push(rec.clone()).expect("push");
            }

            let mut results: Vec<SeqRecord> = sorter
                .finish()
                .expect("finish")
                .map(|r| r.expect("decode"))
                .collect();

            let mut expected = records;
            expected.sort_by(|a, b| {
                a.sequence()
                    .cmp(b.sequence())
                    .then_with(|| a.quality().cmp(b.quality()))
            });

            prop_assert_eq!(results.len(), expected.len());

            // Sort both by (seq, qual, name) for comparison since
            // records with equal (seq, qual) may be in any relative order.
            let key = |r: &SeqRecord| {
                (r.sequence().to_vec(), r.quality().to_vec(), r.name().to_vec())
            };
            results.sort_by_key(|r| key(r));
            expected.sort_by_key(|r| key(r));

            for (got, exp) in results.iter().zip(expected.iter()) {
                prop_assert_eq!(got.sequence(), exp.sequence());
                prop_assert_eq!(got.quality(), exp.quality());
            }
        }

        #[test]
        fn name_sort_output_is_sorted(
            records in proptest::collection::vec(
                (proptest::collection::vec(b'a'..=b'z', 1..20), arb_fixed_len_sequence(16))
                    .prop_flat_map(|(name, seq)| {
                        let len = seq.len();
                        (Just(name), Just(seq), arb_quality(len))
                    })
                    .prop_map(|(name, seq, qual)| SeqRecord::new(name, seq, qual)),
                1..50,
            ),
            buffer_size in 2usize..10,
        ) {
            let mut sorter = Builder::new()
                .sort_by_name()
                .codec(DryIceCodec::new())
                .max_buffer_items(buffer_size)
                .build();

            for rec in &records {
                sorter.push(rec.clone()).expect("push");
            }

            let results: Vec<SeqRecord> = sorter
                .finish()
                .expect("finish")
                .map(|r| r.expect("decode"))
                .collect();

            prop_assert_eq!(results.len(), records.len());

            for window in results.windows(2) {
                prop_assert!(
                    window[0].name() <= window[1].name(),
                    "output must be sorted by name"
                );
            }
        }
    }
}

// ── Minimal builder (defaults) ──────────────────────────────

#[test]
fn minimal_builder_sorts_with_all_defaults() {
    let mut sorter = Builder::new().sort_by_illumina().build();

    let records = vec![
        make_record(b"r3", b"TTTTTTTT", b"!!!!!!!!"),
        make_record(b"r1", b"AAAAAAAA", b"!!!!!!!!"),
        make_record(b"r2", b"CCCCCCCC", b"!!!!!!!!"),
    ];

    for rec in records {
        sorter.push(rec).expect("push should succeed");
    }

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish should succeed")
        .map(|r| r.expect("each record should decode"))
        .collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
    assert_eq!(results[2].sequence(), b"TTTTTTTT");
}

#[test]
fn minimal_builder_with_explicit_codec_and_default_budget() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .codec(DryIceCodec::new().two_bit_exact())
        .build();

    sorter
        .push(make_record(b"r1", b"CCCCCCCC", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
}

#[test]
fn minimal_builder_with_dedup_convenience() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .dedup_by_sequence()
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"r1", b"AAAAAAAA", b"IIIIIIII"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r3", b"CCCCCCCC", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results.len(), 2, "duplicate sequence should be removed");
    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"CCCCCCCC");
}

#[test]
fn sort_with_sequential_override() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .sort_with_sequential()
        .max_buffer_items(100)
        .build();

    sorter
        .push(make_record(b"r1", b"TTTTTTTT", b"!!!!!!!!"))
        .expect("push");
    sorter
        .push(make_record(b"r2", b"AAAAAAAA", b"!!!!!!!!"))
        .expect("push");

    let results: Vec<SeqRecord> = sorter
        .finish()
        .expect("finish")
        .map(|r| r.expect("decode"))
        .collect();

    assert_eq!(results[0].sequence(), b"AAAAAAAA");
    assert_eq!(results[1].sequence(), b"TTTTTTTT");
}
