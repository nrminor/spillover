use std::cell::Cell;
use std::cmp::Ordering;
use std::error::Error;
use std::io::{self, Read, Write};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use dryice::{DryIceError, RecordKey};
use spillover::codec::{
    Codec, CodecCursor, CodecWriter, EncodeKeyedRun, EncodeRun, KeyedCodec, KeyedCodecCursor,
    KeyedCodecWriter, TakeOwned,
};
use spillover::key::{DeriveKey, Keyed, KeyedOrder};
use spillover_bio::codec::{DryIceCodec, PairCodec, PairCodecError};
use spillover_bio::key::PackedSequenceKey;
use spillover_bio::record::{ReadPair, SeqRecord};
use spillover_bio::sort::SequenceOrder;

struct SuppliedOrder(Cell<usize>);

struct FailingReader<'a> {
    bytes: &'a [u8],
    position: usize,
    fail_at: usize,
    failed: bool,
    calls: Rc<Cell<usize>>,
}

impl Read for FailingReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.calls.set(self.calls.get() + 1);
        if self.position == self.fail_at && !self.failed {
            self.failed = true;
            return Err(io::Error::other("injected read failure"));
        }
        let limit = if self.failed {
            self.bytes.len()
        } else {
            self.fail_at
        };
        let count = output.len().min(limit - self.position);
        output[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
        self.position += count;
        Ok(count)
    }
}

struct FailingWriter {
    bytes: Rc<std::cell::RefCell<Vec<u8>>>,
    fail_at: usize,
    armed: Rc<Cell<bool>>,
}

impl Write for FailingWriter {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        let mut bytes = self.bytes.borrow_mut();
        if self.armed.get() && bytes.len() >= self.fail_at {
            return Err(io::Error::other("injected write failure"));
        }
        let count = if self.armed.get() {
            input.len().min(self.fail_at - bytes.len())
        } else {
            input.len()
        };
        bytes.extend_from_slice(&input[..count]);
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn assert_io_source(error: &PairCodecError, message: &str) {
    assert!(matches!(error, PairCodecError::DryIce(DryIceError::Io(_))));
    let dryice = error.source().expect("dryice source");
    assert!(dryice.is::<DryIceError>());
    let io = dryice.source().expect("I/O source");
    assert!(io.is::<io::Error>());
    assert_eq!(io.to_string(), message);
}

static KEY_CLONES: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CountingKey(u8);

impl Clone for CountingKey {
    fn clone(&self) -> Self {
        KEY_CLONES.fetch_add(1, AtomicOrdering::Relaxed);
        Self(self.0)
    }
}

impl RecordKey for CountingKey {
    const WIDTH: u16 = 1;
    const TYPE_TAG: [u8; 16] = *b"pair-test-key-v1";

    fn encode_into(&self, out: &mut [u8]) {
        out[0] = self.0;
    }

    fn decode_from(bytes: &[u8]) -> Result<Self, DryIceError> {
        let [value] = bytes else {
            return Err(DryIceError::InvalidRecordKeyEncoding {
                message: "invalid counting key length",
            });
        };
        Ok(Self(*value))
    }
}

struct CountingOrder;

impl KeyedOrder for CountingOrder {
    type StoredKey = CountingKey;

    fn compare_keys(&self, left: &CountingKey, right: &CountingKey) -> Ordering {
        left.cmp(right)
    }
}

impl DeriveKey<ReadPair<SeqRecord>> for CountingOrder {
    fn derive_key(&self, _: &ReadPair<SeqRecord>) -> CountingKey {
        CountingKey(42)
    }
}

impl KeyedOrder for SuppliedOrder {
    type StoredKey = PackedSequenceKey<2>;

    fn compare_keys(&self, left: &Self::StoredKey, right: &Self::StoredKey) -> Ordering {
        left.cmp(right)
    }
}

impl DeriveKey<ReadPair<SeqRecord>> for SuppliedOrder {
    fn derive_key(&self, _: &ReadPair<SeqRecord>) -> Self::StoredKey {
        self.0.set(self.0.get() + 1);
        PackedSequenceKey::from_sequence(b"TTTT")
    }
}

fn pair(name: &[u8]) -> ReadPair<SeqRecord> {
    let first = SeqRecord::new(name, b"A", b"!");
    let second = SeqRecord::new(name, b"C", b"#");
    ReadPair::from_mates(first, second)
}

#[test]
fn ordinary_roundtrip_crosses_odd_blocks() -> Result<(), Box<dyn std::error::Error>> {
    let codec = PairCodec::new(DryIceCodec::new().with_block_records(3));
    let pairs = [pair(b"a"), pair(b"b"), pair(b"c")];
    let mut bytes = Vec::new();
    let mut writer = EncodeRun::<ReadPair<SeqRecord>>::writer(&codec, &mut bytes);
    for pair in &pairs {
        writer.write(pair)?;
    }
    CodecWriter::<ReadPair<SeqRecord>>::finish(writer)?;
    let mut cursor = codec.cursor(bytes.as_slice())?;
    for expected in &pairs {
        assert!(cursor.advance()?);
        cursor.with_current(|actual| {
            assert_eq!(actual.r1(), expected.r1());
            assert_eq!(actual.r2(), expected.r2());
        })?;
        let actual = cursor.take_owned()?.expect("positioned pair");
        assert_eq!(actual.r1(), expected.r1());
        assert_eq!(actual.r2(), expected.r2());
    }
    assert!(!cursor.advance()?);
    Ok(())
}

#[test]
fn keyed_roundtrip_preserves_supplied_key_without_deriving_during_write()
-> Result<(), Box<dyn std::error::Error>> {
    let base = DryIceCodec::new().with_block_records(3).with_record_key();
    let codec = PairCodec::new(base);
    let order = SuppliedOrder(Cell::new(0));
    let pairs = [pair(b"a"), pair(b"b")];
    let entries: Vec<_> = pairs
        .iter()
        .map(|pair| Keyed::derive(pair, &order))
        .collect();
    assert_eq!(order.0.get(), 2);
    let mut bytes = Vec::new();
    let mut writer = EncodeKeyedRun::<ReadPair<SeqRecord>>::keyed_writer(&codec, &mut bytes);
    for entry in &entries {
        writer.write_keyed(entry.borrowed())?;
    }
    KeyedCodecWriter::<ReadPair<SeqRecord>, _>::finish(writer)?;
    assert_eq!(order.0.get(), 2);
    let mut cursor = codec.keyed_cursor(bytes.as_slice())?;
    for entry in &entries {
        assert!(cursor.advance()?);
        assert_eq!(&cursor.current_key()?, entry.stored_key());
        let actual = cursor.take_owned()?.expect("positioned pair");
        assert_eq!(actual.r1(), entry.item().r1());
        assert_eq!(actual.r2(), entry.item().r2());
    }
    assert!(!cursor.advance()?);
    Ok(())
}

#[test]
fn orphan_is_a_typed_error_and_cursor_is_fused() -> Result<(), Box<dyn std::error::Error>> {
    let base = DryIceCodec::new().with_block_records(3);
    let mut bytes = Vec::new();
    let mut writer = EncodeRun::<SeqRecord>::writer(&base, &mut bytes);
    writer.write(&SeqRecord::new(b"orphan", b"A", b"!"))?;
    CodecWriter::<SeqRecord>::finish(writer)?;
    let mut cursor = PairCodec::new(base).cursor(bytes.as_slice())?;
    assert!(matches!(
        cursor.advance(),
        Err(PairCodecError::TruncatedPair)
    ));
    assert!(!cursor.advance()?);
    Ok(())
}

#[test]
fn keyed_orphan_is_a_typed_error_and_cursor_is_fused() -> Result<(), Box<dyn std::error::Error>> {
    let base = DryIceCodec::new().with_record_key();
    let record = SeqRecord::new(b"orphan", b"A", b"!");
    let entry = Keyed::derive(&record, &SequenceOrder::<2>);
    let mut bytes = Vec::new();
    let mut writer = EncodeKeyedRun::<SeqRecord>::keyed_writer(&base, &mut bytes);
    writer.write_keyed(entry.borrowed())?;
    KeyedCodecWriter::<SeqRecord, _>::finish(writer)?;
    let mut cursor = PairCodec::new(base).keyed_cursor(bytes.as_slice())?;
    assert!(matches!(
        cursor.advance(),
        Err(PairCodecError::TruncatedPair)
    ));
    assert!(!cursor.advance()?);
    Ok(())
}

#[test]
fn mismatched_keys_are_typed_and_do_not_reframe_later_records()
-> Result<(), Box<dyn std::error::Error>> {
    let base = DryIceCodec::new().with_record_key();
    let records = [
        SeqRecord::new(b"a", b"A", b"!"),
        SeqRecord::new(b"b", b"C", b"!"),
        SeqRecord::new(b"c", b"A", b"!"),
        SeqRecord::new(b"d", b"A", b"!"),
    ];
    let order = SequenceOrder::<2>;
    let mut bytes = Vec::new();
    let mut writer = EncodeKeyedRun::<SeqRecord>::keyed_writer(&base, &mut bytes);
    for record in &records {
        let entry = Keyed::derive(record, &order);
        writer.write_keyed(entry.borrowed())?;
    }
    KeyedCodecWriter::<SeqRecord, _>::finish(writer)?;
    let mut cursor = PairCodec::new(base).cursor(bytes.as_slice())?;
    assert!(matches!(
        cursor.advance(),
        Err(PairCodecError::MismatchedKeys)
    ));
    assert!(!cursor.advance()?);
    Ok(())
}

#[test]
fn read_failure_on_second_mate_fuses_both_cursor_types() -> Result<(), Box<dyn std::error::Error>> {
    let ordinary = DryIceCodec::new().with_block_records(1);
    let keyed = ordinary.with_record_key();
    let pairs = [pair(b"a"), pair(b"b")];
    let order = SuppliedOrder(Cell::new(0));

    let mut first_ordinary = Vec::new();
    let mut writer = EncodeRun::<SeqRecord>::writer(&ordinary, &mut first_ordinary);
    writer.write(pairs[0].r1())?;
    CodecWriter::<SeqRecord>::finish(writer)?;
    let mut ordinary_bytes = Vec::new();
    let codec = PairCodec::new(ordinary);
    let mut writer = EncodeRun::<ReadPair<SeqRecord>>::writer(&codec, &mut ordinary_bytes);
    for pair in &pairs {
        writer.write(pair)?;
    }
    CodecWriter::<ReadPair<SeqRecord>>::finish(writer)?;
    let calls = Rc::new(Cell::new(0));
    let reader = FailingReader {
        bytes: &ordinary_bytes,
        position: 0,
        fail_at: first_ordinary.len(),
        failed: false,
        calls: Rc::clone(&calls),
    };
    let mut cursor = codec.cursor(reader)?;
    let error = cursor.advance().expect_err("R2 read must fail");
    assert_io_source(&error, "injected read failure");
    let calls_at_failure = calls.get();
    assert!(!cursor.advance()?);
    assert_eq!(calls.get(), calls_at_failure);
    assert!(cursor.take_owned()?.is_none());

    let first_key = Keyed::derive(&pairs[0], &order);
    let record_key = *first_key.stored_key();
    let single_order = SequenceOrder::<2>;
    // The boundary must use the same key width, not necessarily the same key bytes.
    let first_record = Keyed::derive(pairs[0].r1(), &single_order);
    let mut first_keyed = Vec::new();
    let mut writer = EncodeKeyedRun::<SeqRecord>::keyed_writer(&keyed, &mut first_keyed);
    writer.write_keyed(first_record.borrowed())?;
    KeyedCodecWriter::<SeqRecord, _>::finish(writer)?;
    let mut keyed_bytes = Vec::new();
    let codec = PairCodec::new(keyed);
    let mut writer = EncodeKeyedRun::<ReadPair<SeqRecord>>::keyed_writer(&codec, &mut keyed_bytes);
    for pair in &pairs {
        let entry = Keyed::derive(pair, &order);
        writer.write_keyed(entry.borrowed())?;
    }
    KeyedCodecWriter::<ReadPair<SeqRecord>, _>::finish(writer)?;
    assert_ne!(record_key, *first_record.stored_key());
    let calls = Rc::new(Cell::new(0));
    let reader = FailingReader {
        bytes: &keyed_bytes,
        position: 0,
        fail_at: first_keyed.len(),
        failed: false,
        calls: Rc::clone(&calls),
    };
    let mut cursor = codec.keyed_cursor(reader)?;
    let error = cursor.advance().expect_err("keyed R2 read must fail");
    assert_io_source(&error, "injected read failure");
    let calls_at_failure = calls.get();
    assert!(!cursor.advance()?);
    assert_eq!(calls.get(), calls_at_failure);
    assert!(cursor.take_owned()?.is_none());
    Ok(())
}

#[test]
fn write_failure_on_second_mate_preserves_underlying_io_error()
-> Result<(), Box<dyn std::error::Error>> {
    let base = DryIceCodec::new().with_block_records(1);
    let item = pair(b"a");
    let mut first = Vec::new();
    let mut writer = EncodeRun::<SeqRecord>::writer(&base, &mut first);
    writer.write(item.r1())?;
    CodecWriter::<SeqRecord>::finish(writer)?;
    let bytes = Rc::new(std::cell::RefCell::new(Vec::new()));
    let dest = FailingWriter {
        bytes: Rc::clone(&bytes),
        fail_at: first.len(),
        armed: Rc::new(Cell::new(true)),
    };
    let codec = PairCodec::new(base);
    let mut writer = EncodeRun::<ReadPair<SeqRecord>>::writer(&codec, dest);
    let error = writer.write(&item).expect_err("R2 write must fail");
    assert_io_source(&error, "injected write failure");
    assert_eq!(*bytes.borrow(), first);

    let keyed = base.with_record_key();
    let first_record = Keyed::derive(item.r1(), &SequenceOrder::<2>);
    let mut first = Vec::new();
    let mut writer = EncodeKeyedRun::<SeqRecord>::keyed_writer(&keyed, &mut first);
    writer.write_keyed(first_record.borrowed())?;
    KeyedCodecWriter::<SeqRecord, _>::finish(writer)?;
    let bytes = Rc::new(std::cell::RefCell::new(Vec::new()));
    let dest = FailingWriter {
        bytes: Rc::clone(&bytes),
        fail_at: first.len(),
        armed: Rc::new(Cell::new(true)),
    };
    let codec = PairCodec::new(keyed);
    let entry = Keyed::derive(&item, &SuppliedOrder(Cell::new(0)));
    let mut writer = EncodeKeyedRun::<ReadPair<SeqRecord>>::keyed_writer(&codec, dest);
    let error = writer
        .write_keyed(entry.borrowed())
        .expect_err("keyed R2 write must fail");
    assert_io_source(&error, "injected write failure");
    assert_eq!(bytes.borrow().len(), first.len());
    let written = bytes.borrow();
    let mut cursor = keyed.keyed_cursor(written.as_slice())?;
    assert!(cursor.advance()?);
    assert_eq!(&cursor.current_key()?, entry.stored_key());
    assert_eq!(cursor.take_owned()?.as_ref(), Some(item.r1()));
    Ok(())
}

#[test]
fn finish_failures_preserve_underlying_io_error() -> Result<(), Box<dyn std::error::Error>> {
    let base = DryIceCodec::new().with_block_records(3);
    let item = pair(b"a");
    let armed = Rc::new(Cell::new(false));
    let dest = FailingWriter {
        bytes: Rc::new(std::cell::RefCell::new(Vec::new())),
        fail_at: 0,
        armed: Rc::clone(&armed),
    };
    let codec = PairCodec::new(base);
    let mut writer = EncodeRun::<ReadPair<SeqRecord>>::writer(&codec, dest);
    writer.write(&item)?;
    armed.set(true);
    let error = CodecWriter::<ReadPair<SeqRecord>>::finish(writer).expect_err("finish must fail");
    assert_io_source(&error, "injected write failure");

    let armed = Rc::new(Cell::new(false));
    let dest = FailingWriter {
        bytes: Rc::new(std::cell::RefCell::new(Vec::new())),
        fail_at: 0,
        armed: Rc::clone(&armed),
    };
    let codec = PairCodec::new(base.with_record_key());
    let entry = Keyed::derive(&item, &SuppliedOrder(Cell::new(0)));
    let mut writer = EncodeKeyedRun::<ReadPair<SeqRecord>>::keyed_writer(&codec, dest);
    writer.write_keyed(entry.borrowed())?;
    armed.set(true);
    let error = KeyedCodecWriter::<ReadPair<SeqRecord>, _>::finish(writer)
        .expect_err("keyed finish must fail");
    assert_io_source(&error, "injected write failure");
    Ok(())
}

#[test]
fn keyed_writer_never_clones_supplied_key() -> Result<(), Box<dyn std::error::Error>> {
    let codec = PairCodec::new(DryIceCodec::new().with_block_records(1).with_record_key());
    let item = pair(b"a");
    let entry = Keyed::derive(&item, &CountingOrder);
    let mut bytes = Vec::new();
    let mut writer = EncodeKeyedRun::<ReadPair<SeqRecord>>::keyed_writer(&codec, &mut bytes);
    KEY_CLONES.store(0, AtomicOrdering::Relaxed);
    writer.write_keyed(entry.borrowed())?;
    KeyedCodecWriter::<ReadPair<SeqRecord>, _>::finish(writer)?;
    assert_eq!(KEY_CLONES.load(AtomicOrdering::Relaxed), 0);
    let mut cursor = codec.keyed_cursor(bytes.as_slice())?;
    assert!(cursor.advance()?);
    assert_eq!(cursor.current_key()?, CountingKey(42));
    Ok(())
}
