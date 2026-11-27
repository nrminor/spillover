use std::{cmp::Ordering, convert::Infallible, io::Read};

use dryice::{NameCodec, QualityCodec, SequenceCodec};
use spillover::{SortedItemsError, compare::Compare};
use spillover_bio::{
    codec::{DecodedRecord, DryIceCodec},
    record::{ReadPair, SeqRecord, SeqRecordArena, SeqRecordParts},
    sort::{
        Builder as ExternalBuilder, CompatibleWith, ILLUMINA_ORDER, ILLUMINA_PAIR_ORDER,
        LENGTH_ORDER, LONG_READ_ORDER, NAME_ORDER, PAIR_NAME_ORDER, PAIR_SEQUENCE_EXACT_ORDER,
        PAIRED_END_ORDER, Reverse, UnkeyedSequenceOrder,
        in_memory::{Builder, InMemory},
    },
};

fn read(name: &str, sequence: &str) -> SeqRecord {
    SeqRecord::new(name, sequence, "!")
}

fn fixture() -> Vec<SeqRecord> {
    vec![
        read("c", "C"),
        read("a", "A"),
        read("b", "A"),
        read("d", "G"),
    ]
}

#[test]
fn collection_sorting_and_compaction_preserve_allocations() {
    let configuration = Builder::new().sort_by_name();
    let mut reads = fixture();
    configuration.sort_slice(&mut reads);
    configuration.sort_slice(&mut reads);
    assert_eq!(reads[0].name(), b"a");
    let pointer = reads.as_ptr();
    let capacity = reads.capacity();
    let payloads: Vec<_> = reads.iter().map(|r| r.sequence().as_ptr()).collect();
    Builder::new()
        .sort_by_illumina()
        .dedup_by_sequence()
        .sort_vec(&mut reads);
    assert_eq!(reads.len(), 3);
    assert_eq!(reads.as_ptr(), pointer);
    assert_eq!(reads.capacity(), capacity);
    assert!(
        reads
            .iter()
            .all(|r| payloads.contains(&r.sequence().as_ptr()))
    );
}

#[test]
fn transferred_vec_moves_original_records() {
    let mut reads = Vec::with_capacity(10);
    reads.push(read("b", "C"));
    let original = reads[0].sequence().as_ptr();
    let mut sorter = Builder::new().sort_by_name().build_from(reads);
    sorter.push(read("a", "A"));
    let mut output = sorter.finish();
    assert_eq!(
        output.next().expect("first").expect("infallible").name(),
        b"a"
    );
    assert_eq!(
        output
            .next()
            .expect("second")
            .expect("infallible")
            .sequence()
            .as_ptr(),
        original
    );
}

#[test]
fn owned_and_arena_output_share_dedup_history_across_traversal() {
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .dedup_by_sequence()
        .build();
    for record in fixture() {
        sorter.push(record);
    }
    let mut output = sorter.finish();
    assert_eq!(
        output
            .next()
            .expect("first")
            .expect("infallible")
            .sequence(),
        b"A"
    );
    let mut rest = Vec::new();
    output
        .items()
        .try_for_each(|r| {
            rest.push(r.sequence().to_vec());
            Ok::<(), Infallible>(())
        })
        .expect("visit");
    assert_eq!(rest, [b"C".to_vec(), b"G".to_vec()]);

    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .dedup_by_sequence()
        .arena(&mut arena)
        .build();
    for record in fixture() {
        sorter.push(&record);
    }
    let mut output = sorter.finish();
    let first = output.next().expect("first").expect("infallible");
    let mut rest = Vec::new();
    output
        .items()
        .try_for_each(|r| {
            rest.push(r.sequence().to_vec());
            Ok::<(), Infallible>(())
        })
        .expect("visit");
    assert_eq!(rest, [b"C".to_vec(), b"G".to_vec()]);
    assert_eq!(first.sequence(), b"A");
}

#[test]
fn arena_reset_retains_capacity_and_sink_failure_is_classified() {
    let mut arena = SeqRecordArena::with_capacity(1024);
    let mut sorter = Builder::new().sort_by_name().arena(&mut arena).build();
    sorter.push(&read("a", "A"));
    sorter.push(&read("b", "C"));
    let result = sorter
        .finish()
        .items()
        .try_for_each(|_| Err::<(), _>("stopped"));
    assert!(matches!(result, Err(SortedItemsError::Sink("stopped"))));
    let capacity = arena.capacity();
    let next = Builder::new().sort_by_name().arena(&mut arena).build();
    assert_eq!(next.finish().count(), 0);
    assert!(arena.is_empty());
    assert_eq!(arena.capacity(), capacity);
}

#[test]
fn explicit_arena_materialization_copies_only_accepted_records() {
    let mut arena = SeqRecordArena::new();
    let mut sorter = Builder::new()
        .sort_by_illumina()
        .dedup_by_sequence()
        .arena(&mut arena)
        .build();
    for record in fixture() {
        sorter.push(&record);
    }
    let mut owned = Vec::new();
    sorter
        .finish()
        .items()
        .try_for_each(|view| {
            let record = SeqRecord::from_slices(view.name(), view.sequence(), view.quality());
            assert_ne!(record.sequence().as_ptr(), view.sequence().as_ptr());
            owned.push(record);
            Ok::<(), Infallible>(())
        })
        .expect("visit");
    assert_eq!(owned.len(), 3);
    drop(
        Builder::new()
            .sort_by_name()
            .arena(&mut arena)
            .build()
            .finish(),
    );
    assert_eq!(owned[0].sequence(), b"A");
}

struct ByLength;
impl<R: SeqRecordParts> Compare<R> for ByLength {
    fn compare(&self, a: &R, b: &R) -> Ordering {
        a.sequence().len().cmp(&b.sequence().len())
    }
}
impl CompatibleWith<InMemory> for ByLength {
    type Input = SeqRecord;
}

struct ConfigurationNameOrder;

impl Compare<SeqRecord> for ConfigurationNameOrder {
    fn compare(&self, left: &SeqRecord, right: &SeqRecord) -> Ordering {
        NAME_ORDER.compare(left, right)
    }
}

impl<R: Read, S: SequenceCodec, Q: QualityCodec, N: NameCodec, K>
    Compare<DecodedRecord<R, S, Q, N, K>> for ConfigurationNameOrder
{
    fn compare(
        &self,
        left: &DecodedRecord<R, S, Q, N, K>,
        right: &DecodedRecord<R, S, Q, N, K>,
    ) -> Ordering {
        NAME_ORDER.compare(left, right)
    }
}

impl Compare<ReadPair<SeqRecord>> for ConfigurationNameOrder {
    fn compare(&self, left: &ReadPair<SeqRecord>, right: &ReadPair<SeqRecord>) -> Ordering {
        PAIR_NAME_ORDER.compare(left, right)
    }
}

impl CompatibleWith<DryIceCodec> for ConfigurationNameOrder {
    type Input = SeqRecord;
}

impl CompatibleWith<InMemory> for ConfigurationNameOrder {
    type Input = ReadPair<SeqRecord>;
}

#[test]
fn same_order_infers_single_external_reads_and_intact_memory_pairs() {
    // These independent builds have no later pushes to supply an input type.
    let _external = ExternalBuilder::new()
        .sort_by_unkeyed(ConfigurationNameOrder)
        .codec(DryIceCodec::new())
        .max_buffer_items(1)
        .build();
    let _memory = Builder::new().sort_by(ConfigurationNameOrder).build();

    let mut external = ExternalBuilder::new()
        .sort_by_unkeyed(ConfigurationNameOrder)
        .codec(DryIceCodec::new())
        .max_buffer_items(1)
        .build();
    external.push(read("c", "G")).expect("push c");
    external.push(read("a", "T")).expect("push a");
    external.push(read("b", "A")).expect("push b");
    let reads: Vec<_> = external
        .finish()
        .expect("finish external sort")
        .collect::<Result<_, _>>()
        .expect("decode reads");
    assert_eq!(reads, [read("a", "T"), read("b", "A"), read("c", "G")]);

    let mut memory = Builder::new().sort_by(ConfigurationNameOrder).build();
    memory.push(ReadPair::from_mates(
        SeqRecord::new("b/1", "AC", "!#"),
        SeqRecord::new("b/2", "TG", "$%"),
    ));
    memory.push(ReadPair::from_mates(
        SeqRecord::new("a/1", "GT", "&'"),
        SeqRecord::new("a/2", "CA", "()"),
    ));
    let pairs: Vec<_> = memory
        .finish()
        .collect::<Result<_, _>>()
        .expect("infallible memory sort");
    assert_eq!(
        pairs,
        [
            ReadPair::from_mates(
                SeqRecord::new("a/1", "GT", "&'"),
                SeqRecord::new("a/2", "CA", "()"),
            ),
            ReadPair::from_mates(
                SeqRecord::new("b/1", "AC", "!#"),
                SeqRecord::new("b/2", "TG", "$%"),
            ),
        ]
    );
}

struct Tagged {
    record: SeqRecord,
    annotation: String,
}
struct ByTag;
impl Compare<Tagged> for ByTag {
    fn compare(&self, a: &Tagged, b: &Tagged) -> Ordering {
        a.annotation.cmp(&b.annotation)
    }
}

fn build_generic_memory_order<O>(order: O)
where
    O: CompatibleWith<InMemory, Input = SeqRecord> + Compare<SeqRecord>,
{
    let _sorter = Builder::new().sort_by(order).build();
}

#[test]
fn empty_builds_infer_inputs_for_custom_and_wrapped_orders() {
    let _single = Builder::new().sort_by_name().build();
    let _pair = Builder::new().sort_by_illumina_pairs().build();
    let _custom = Builder::new().sort_by(ByLength).build();
    let _nested_reverse = Builder::new().sort_by(Reverse(Reverse(ByLength))).build();
    build_generic_memory_order(ByLength);
    build_generic_memory_order(Reverse(Reverse(ByLength)));
    let mut arena = SeqRecordArena::new();
    let _arena = Builder::new().sort_by(ByLength).arena(&mut arena).build();
}

#[test]
fn extensions_and_non_clone_metadata_need_no_codec() {
    let mut sorter = Builder::new().sort_by(ByLength).build();
    sorter.push(read("long", "AAAA"));
    sorter.push(read("short", "A"));
    assert_eq!(
        sorter
            .finish()
            .next()
            .expect("first")
            .expect("infallible")
            .name(),
        b"short"
    );
    let mut items = vec![
        Tagged {
            record: read("b", "C"),
            annotation: "z".into(),
        },
        Tagged {
            record: read("a", "A"),
            annotation: "a".into(),
        },
    ];
    Builder::new().sort_by(ByTag).sort_vec(&mut items);
    assert_eq!(items[0].record.name(), b"a");
    assert_eq!(items[1].annotation, "z");
    Builder::new().sort_by(ByTag).sort_slice(&mut items);
    let _sorter = Builder::new().sort_by(ByTag).build_from(items);
}

#[test]
fn all_single_presets_and_radix_sort_larger_inputs() {
    let reads: Vec<_> = (0..1024)
        .rev()
        .map(|i| {
            let name = format!("read-{i:04}");
            let sequence: String = (0..16)
                .map(|shift| b"ACGT"[(i >> (shift * 2)) & 3] as char)
                .collect();
            read(&name, &sequence)
        })
        .collect();
    macro_rules! check {
        ($builder:expr, $order:expr) => {{
            let mut items = reads.clone();
            $builder.sort_slice(&mut items);
            assert!(
                items
                    .windows(2)
                    .all(|w| $order.compare(&w[0], &w[1]) != Ordering::Greater)
            );
        }};
    }
    check!(Builder::new().sort_by_illumina(), ILLUMINA_ORDER);
    check!(Builder::new().sort_by_paired_end(), PAIRED_END_ORDER);
    check!(Builder::new().sort_by_long_read(), LONG_READ_ORDER);
    check!(
        Builder::new().sort_by_sequence_exact(),
        UnkeyedSequenceOrder
    );
    check!(Builder::new().sort_by_name(), NAME_ORDER);
    check!(Builder::new().sort_by_length(), LENGTH_ORDER);
    check!(
        Builder::new().sort_by_illumina().sort_with_sequential(),
        ILLUMINA_ORDER
    );
    #[cfg(feature = "rayon")]
    check!(
        Builder::new().sort_by_illumina().sort_with_parallel(),
        ILLUMINA_ORDER
    );
}

#[test]
fn pair_presets_preserve_whole_pairs() {
    let pairs = vec![
        ReadPair::from_mates(read("a/1", "A"), read("a/2", "T")),
        ReadPair::from_mates(read("b/1", "A"), read("b/2", "G")),
    ];
    macro_rules! check {
        ($builder:expr, $order:expr) => {{
            let mut sorter = $builder.build();
            for pair in pairs.clone() {
                sorter.push(pair);
            }
            let result: Vec<_> = sorter
                .finish()
                .collect::<Result<_, _>>()
                .expect("infallible");
            assert_eq!(result.len(), 2);
            assert_ne!($order.compare(&result[0], &result[1]), Ordering::Greater);
            assert_eq!(result[0].r1().name()[0], result[0].r2().name()[0]);
        }};
    }
    check!(Builder::new().sort_by_illumina_pairs(), ILLUMINA_PAIR_ORDER);
    check!(
        Builder::new().sort_by_pair_sequence_exact(),
        PAIR_SEQUENCE_EXACT_ORDER
    );
    check!(Builder::new().sort_by_pair_name(), PAIR_NAME_ORDER);
}
