use std::{cmp::Ordering, convert::Infallible, error::Error, num::NonZeroUsize};

use spillover::{
    compare::Compare,
    merge::{MergeConfig, MergeError},
};
use spillover_bio::{
    codec::DryIceCodec,
    record::{ReadPair, SeqRecord, SeqRecordParts},
    sort::{
        Builder, CompatibleWith, ILLUMINA_250_PAIR_ORDER, ILLUMINA_PAIR_ORDER, PAIR_LENGTHS_ORDER,
        PAIR_NAME_ORDER, PAIR_QUALITY_ORDER, PAIR_SEQUENCE_EXACT_ORDER, Reverse,
    },
};

type TestResult = Result<(), Box<dyn Error>>;
type Fields = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

fn fields<R: SeqRecordParts>(pair: &ReadPair<R>) -> Fields {
    (
        pair.r1().name().to_vec(),
        pair.r1().sequence().to_vec(),
        pair.r1().quality().to_vec(),
        pair.r2().name().to_vec(),
        pair.r2().sequence().to_vec(),
        pair.r2().quality().to_vec(),
    )
}

fn input_pairs() -> Vec<ReadPair<SeqRecord>> {
    let prefix = "A".repeat(152);
    let long_first = format!("{prefix}C");
    let long_second = format!("{prefix}G");
    let long_quality = "!".repeat(153);
    let inputs = [
        ("z/1", "AC", "zz", "z/2", "TT", "!!"),
        ("a/1", "AC", "!!", "a/2", "AA", "zz"),
        ("b/1", "AC", "!!", "b/2", "AC", "!!"),
        ("c/1", "ac", "!!", "c/2", "AA", "!!"),
        (
            "d/1",
            long_first.as_str(),
            long_quality.as_str(),
            "d/2",
            "T",
            "!",
        ),
        (
            "e/1",
            long_second.as_str(),
            long_quality.as_str(),
            "e/2",
            "A",
            "!",
        ),
        ("a/1", "AC", "!!", "a/2", "AA", "zz"),
        ("f/1", "N", "!", "f/2", "A", "!"),
    ];
    inputs
        .into_iter()
        .map(|(name1, sequence1, quality1, name2, sequence2, quality2)| {
            let r1 = SeqRecord::new(name1, sequence1, quality1);
            let r2 = SeqRecord::new(name2, sequence2, quality2);
            ReadPair::from_mates(r1, r2)
        })
        .collect()
}

fn verify_output<O: Compare<ReadPair<SeqRecord>>>(
    input: &[ReadPair<SeqRecord>],
    output: &[ReadPair<SeqRecord>],
    order: &O,
) {
    let mut original_fields: Vec<_> = input.iter().map(fields).collect();
    let mut output_fields: Vec<_> = output.iter().map(fields).collect();
    original_fields.sort_unstable();
    output_fields.sort_unstable();
    assert_eq!(original_fields, output_fields);
    for adjacent in output.windows(2) {
        assert_ne!(order.compare(&adjacent[0], &adjacent[1]), Ordering::Greater);
    }
}

#[derive(Clone, Copy)]
struct PairTotalLength;

impl<R: SeqRecordParts> Compare<ReadPair<R>> for PairTotalLength {
    fn compare(&self, left: &ReadPair<R>, right: &ReadPair<R>) -> Ordering {
        let left_length = left.r1().sequence().len() + left.r2().sequence().len();
        let right_length = right.r1().sequence().len() + right.r2().sequence().len();
        left_length.cmp(&right_length)
    }
}

impl CompatibleWith<DryIceCodec> for PairTotalLength {
    type Input = ReadPair<SeqRecord>;
}

#[test]
fn built_in_orders_survive_multi_pass_merge() -> TestResult {
    let input = input_pairs();
    let mut merge_config = MergeConfig::default();
    merge_config.max_fan_in = NonZeroUsize::new(2).expect("nonzero fan-in");

    macro_rules! check_order {
        ($builder:expr, $order:expr) => {{
            let mut sorter = $builder
                .max_buffer_items(1)
                .merge_config(merge_config.clone())
                .build();
            for pair in &input {
                sorter.push(pair.clone())?;
            }
            let output = sorter.finish()?;
            let collected: Result<Vec<_>, _> = output.collect();
            let collected = collected?;
            verify_output(&input, &collected, &$order);
        }};
    }

    check_order!(Builder::new().sort_by_illumina_pairs(), ILLUMINA_PAIR_ORDER);
    check_order!(
        Builder::new().sort_by(ILLUMINA_250_PAIR_ORDER),
        ILLUMINA_250_PAIR_ORDER
    );
    check_order!(
        Builder::new().sort_by_pair_sequence_exact(),
        PAIR_SEQUENCE_EXACT_ORDER
    );
    check_order!(Builder::new().sort_by_pair_name(), PAIR_NAME_ORDER);
    check_order!(
        Builder::new().sort_by_unkeyed(PAIR_QUALITY_ORDER),
        PAIR_QUALITY_ORDER
    );
    check_order!(
        Builder::new().sort_by_unkeyed(PAIR_LENGTHS_ORDER),
        PAIR_LENGTHS_ORDER
    );
    check_order!(
        Builder::new().sort_by_unkeyed(PairTotalLength),
        PairTotalLength
    );
    let reversed = Reverse(ILLUMINA_PAIR_ORDER);
    check_order!(Builder::new().sort_by(reversed), reversed);
    let reversed_twice = Reverse(Reverse(ILLUMINA_PAIR_ORDER));
    check_order!(Builder::new().sort_by(reversed_twice), reversed_twice);
    Ok(())
}

#[test]
fn odd_physical_blocks_and_mixed_output_keep_whole_pairs() -> TestResult {
    let input = input_pairs();
    let codec = DryIceCodec::new().with_block_records(3);
    let mut sorter = Builder::new()
        .sort_by_illumina_pairs()
        .codec(codec)
        .max_buffer_items(2)
        .build();
    for pair in &input {
        sorter.push(pair.clone())?;
    }
    let mut output = sorter.finish()?;
    let first = output.next().expect("nonempty output")?;
    let mut collected = vec![first];
    output.items().try_for_each(|pair| {
        collected.push(pair.clone());
        Ok::<(), Infallible>(())
    })?;
    verify_output(&input, &collected, &ILLUMINA_PAIR_ORDER);
    Ok(())
}

#[test]
fn measured_budget_supports_keyed_and_unkeyed_pairs() -> TestResult {
    let input = input_pairs();
    let mut keyed = Builder::new()
        .sort_by_illumina_pairs()
        .measured_budget(2048)
        .build();
    let mut unkeyed = Builder::new()
        .sort_by_pair_sequence_exact()
        .measured_budget(2048)
        .build();
    for pair in &input {
        keyed.push(pair.clone())?;
        unkeyed.push(pair.clone())?;
    }
    let keyed_output = keyed.finish()?;
    let keyed_collected: Result<Vec<_>, _> = keyed_output.collect();
    let keyed_collected = keyed_collected?;
    verify_output(&input, &keyed_collected, &ILLUMINA_PAIR_ORDER);
    let unkeyed_output = unkeyed.finish()?;
    let unkeyed_collected: Result<Vec<_>, _> = unkeyed_output.collect();
    let unkeyed_collected = unkeyed_collected?;
    verify_output(&input, &unkeyed_collected, &PAIR_SEQUENCE_EXACT_ORDER);
    Ok(())
}

fn pair_with_name_lengths(r1_length: usize, r2_length: usize) -> ReadPair<SeqRecord> {
    let r1_name = vec![b'a'; r1_length];
    let r2_name = vec![b'b'; r2_length];
    let r1 = SeqRecord::new(r1_name, b"A", b"!");
    let r2 = SeqRecord::new(r2_name, b"T", b"!");
    ReadPair::from_mates(r1, r2)
}

#[test]
fn measured_pair_admission_counts_both_asymmetric_mate_heaps() -> TestResult {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(manifest.is_file());
    let mut merge_config = MergeConfig::default();
    merge_config.temp_dir = Some(manifest);
    merge_config.write_buffer_bytes = 0;

    // Names do not enlarge the Illumina key. Each pair fits alone, but two
    // exceed 1 MiB only when BOTH mate payloads are counted. Omitting either
    // mate leaves at most 800 KiB, comfortably below the spill threshold.
    macro_rules! check_admission {
        ($builder:expr) => {{
            for (r1_length, r2_length) in [(200 * 1024, 400 * 1024), (400 * 1024, 200 * 1024)] {
                let mut sorter = $builder
                    .sort_with_sequential()
                    .merge_config(merge_config.clone())
                    .measured_budget(1024 * 1024)
                    .build();
                sorter.push(pair_with_name_lengths(r1_length, r2_length))?;
                let result = sorter.push(pair_with_name_lengths(r1_length, r2_length));
                assert!(
                    matches!(result, Err(MergeError::Io(_))),
                    "two asymmetric pairs must trigger a spill: R1={r1_length}, R2={r2_length}, result={result:?}"
                );
            }
            for (r1_length, r2_length) in [(1, 1), (1, 400 * 1024), (400 * 1024, 1)] {
                let mut control = $builder
                    .sort_with_sequential()
                    .merge_config(merge_config.clone())
                    .measured_budget(1024 * 1024)
                    .build();
                control.push(pair_with_name_lengths(r1_length, r2_length))?;
                control.push(pair_with_name_lengths(r1_length, r2_length))?;
            }
        }};
    }

    check_admission!(Builder::new().sort_by_illumina_pairs());
    check_admission!(Builder::new().sort_by_pair_sequence_exact());
    Ok(())
}

#[test]
fn measured_pairs_allow_one_oversized_item_and_spill_it() -> TestResult {
    let input = pair_with_name_lengths(1, 2 * 1024 * 1024);
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    assert!(manifest.is_file());
    let mut failing_config = MergeConfig::default();
    failing_config.temp_dir = Some(manifest);
    failing_config.write_buffer_bytes = 0;

    macro_rules! check_oversized {
        ($builder:expr, $order:expr) => {{
            let mut failing = $builder
                .sort_with_sequential()
                .merge_config(failing_config.clone())
                .measured_budget(1024 * 1024)
                .build();
            let result = failing.push(input.clone());
            assert!(
                matches!(result, Err(MergeError::Io(_))),
                "oversized first pair must attempt a spill, got {result:?}"
            );

            let mut valid_config = MergeConfig::default();
            valid_config.write_buffer_bytes = 0;
            let mut sorter = $builder
                .sort_with_sequential()
                .merge_config(valid_config)
                .measured_budget(1024 * 1024)
                .build();
            sorter.push(input.clone())?;
            let output = sorter.finish()?;
            let collected: Result<Vec<_>, _> = output.collect();
            let collected = collected?;
            verify_output(std::slice::from_ref(&input), &collected, &$order);
        }};
    }

    check_oversized!(Builder::new().sort_by_illumina_pairs(), ILLUMINA_PAIR_ORDER);
    check_oversized!(
        Builder::new().sort_by_pair_sequence_exact(),
        PAIR_SEQUENCE_EXACT_ORDER
    );
    Ok(())
}

struct ExactEncodedLength;

impl<R: SeqRecordParts + ?Sized> Compare<R> for ExactEncodedLength {
    fn compare(&self, left: &R, right: &R) -> Ordering {
        left.sequence().len().cmp(&right.sequence().len())
    }
}

// Deliberately no raw-codec grant: explicit configuration must stand alone.
impl CompatibleWith<DryIceCodec<dryice::TwoBitExactCodec>> for ExactEncodedLength {
    type Input = SeqRecord;
}

fn build_generic_pair_order<O>(order: O)
where
    O: CompatibleWith<DryIceCodec, Input = ReadPair<SeqRecord>> + Compare<ReadPair<SeqRecord>>,
{
    let _sorter = Builder::new()
        .sort_by_unkeyed(order)
        .max_buffer_items(2)
        .build();
}

#[test]
fn standalone_build_infers_input_without_push_or_output_annotations() {
    let _single_keyed = Builder::new().sort_by_illumina().build();
    let _single_unkeyed = Builder::new().sort_by_sequence_exact().build();
    let _pair_keyed = Builder::new().sort_by_illumina_pairs().build();
    let _pair_unkeyed = Builder::new().sort_by_pair_sequence_exact().build();
    let _custom_pair = Builder::new().sort_by_unkeyed(PairTotalLength).build();
    let _reverse_pair = Builder::new().sort_by(Reverse(ILLUMINA_PAIR_ORDER)).build();
    let _reverse_custom = Builder::new()
        .sort_by_unkeyed(Reverse(PairTotalLength))
        .build();
    let _nested_reverse = Builder::new()
        .sort_by_unkeyed(Reverse(Reverse(PairTotalLength)))
        .build();
    let _explicit_only = Builder::new()
        .sort_by_unkeyed(ExactEncodedLength)
        .codec(DryIceCodec::new().two_bit_exact())
        .build();
    let _reverse_explicit_only = Builder::new()
        .sort_by_unkeyed(Reverse(ExactEncodedLength))
        .codec(DryIceCodec::new().two_bit_exact())
        .build();
    build_generic_pair_order(PairTotalLength);
    build_generic_pair_order(Reverse(PairTotalLength));
    build_generic_pair_order(Reverse(Reverse(PairTotalLength)));
}
