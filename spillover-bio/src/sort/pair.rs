//! Orders for whole read pairs.

use std::cmp::Ordering;

use spillover::{
    compare::Compare,
    key::{DeriveKey, EstimateKeyMemory, KeyMemory, KeyedOrder},
};

use crate::{
    key::PackedSequenceKey,
    record::{ReadPair, SeqRecordParts},
};

/// Compare both sequences, both qualities, then both full names.
#[derive(Debug, Clone, Copy, Default)]
pub struct PairSequenceExactOrder;

/// Compare a packed R1 sequence key, then the complete exact pair order.
///
/// Packing folds case, maps ambiguous bases, pads and truncates; consequently
/// this order is not necessarily the same as raw-byte sequence ordering.
#[derive(Debug, Clone, Copy, Default)]
pub struct PairSequenceOrder<const N: usize>;

/// Compare R1 and R2 names, without additional tiebreakers.
#[derive(Debug, Clone, Copy, Default)]
pub struct PairNameOrder;

/// Compare R1 and R2 quality bytes, without additional tiebreakers.
#[derive(Debug, Clone, Copy, Default)]
pub struct PairQualityOrder;

/// Compare R1 and R2 sequence lengths, without additional tiebreakers.
#[derive(Debug, Clone, Copy, Default)]
pub struct PairLengthsOrder;

/// Pair sequence order with a 38-byte packed R1 key.
pub const ILLUMINA_PAIR_ORDER: PairSequenceOrder<38> = PairSequenceOrder;
/// Pair sequence order with a 64-byte packed R1 key.
pub const ILLUMINA_250_PAIR_ORDER: PairSequenceOrder<64> = PairSequenceOrder;
/// Exact pair sequence order.
pub const PAIR_SEQUENCE_EXACT_ORDER: PairSequenceExactOrder = PairSequenceExactOrder;
/// Pair name order.
pub const PAIR_NAME_ORDER: PairNameOrder = PairNameOrder;
/// Pair quality order.
pub const PAIR_QUALITY_ORDER: PairQualityOrder = PairQualityOrder;
/// Pair length order.
pub const PAIR_LENGTHS_ORDER: PairLengthsOrder = PairLengthsOrder;

impl<R: SeqRecordParts> Compare<ReadPair<R>> for PairSequenceExactOrder {
    fn compare(&self, left: &ReadPair<R>, right: &ReadPair<R>) -> Ordering {
        let left_fields = (
            left.r1().sequence(),
            left.r2().sequence(),
            left.r1().quality(),
            left.r2().quality(),
            left.r1().name(),
            left.r2().name(),
        );
        let right_fields = (
            right.r1().sequence(),
            right.r2().sequence(),
            right.r1().quality(),
            right.r2().quality(),
            right.r1().name(),
            right.r2().name(),
        );
        left_fields.cmp(&right_fields)
    }
}

impl<R: SeqRecordParts> Compare<ReadPair<R>> for PairNameOrder {
    fn compare(&self, left: &ReadPair<R>, right: &ReadPair<R>) -> Ordering {
        let left_names = (left.r1().name(), left.r2().name());
        let right_names = (right.r1().name(), right.r2().name());
        left_names.cmp(&right_names)
    }
}

impl<R: SeqRecordParts> Compare<ReadPair<R>> for PairQualityOrder {
    fn compare(&self, left: &ReadPair<R>, right: &ReadPair<R>) -> Ordering {
        let left_qualities = (left.r1().quality(), left.r2().quality());
        let right_qualities = (right.r1().quality(), right.r2().quality());
        left_qualities.cmp(&right_qualities)
    }
}

impl<R: SeqRecordParts> Compare<ReadPair<R>> for PairLengthsOrder {
    fn compare(&self, left: &ReadPair<R>, right: &ReadPair<R>) -> Ordering {
        let left_first_length = left.r1().sequence().len();
        let left_second_length = left.r2().sequence().len();
        let right_first_length = right.r1().sequence().len();
        let right_second_length = right.r2().sequence().len();
        let left_lengths = (left_first_length, left_second_length);
        let right_lengths = (right_first_length, right_second_length);
        left_lengths.cmp(&right_lengths)
    }
}

impl<const N: usize> KeyedOrder for PairSequenceOrder<N> {
    type StoredKey = PackedSequenceKey<N>;

    fn compare_keys(&self, left: &Self::StoredKey, right: &Self::StoredKey) -> Ordering {
        left.cmp(right)
    }
}

impl<const N: usize, R: SeqRecordParts> DeriveKey<ReadPair<R>> for PairSequenceOrder<N> {
    fn derive_key(&self, pair: &ReadPair<R>) -> Self::StoredKey {
        PackedSequenceKey::from_sequence(pair.r1().sequence())
    }
}

impl<const N: usize, R: SeqRecordParts> Compare<ReadPair<R>> for PairSequenceOrder<N> {
    fn compare(&self, left: &ReadPair<R>, right: &ReadPair<R>) -> Ordering {
        let left_key = self.derive_key(left);
        let right_key = self.derive_key(right);
        let primary = self.compare_keys(&left_key, &right_key);
        primary.then_with(|| PAIR_SEQUENCE_EXACT_ORDER.compare(left, right))
    }
}

impl<const N: usize, R: SeqRecordParts> EstimateKeyMemory<ReadPair<R>> for PairSequenceOrder<N> {
    fn estimate_key_memory(&self, _: &ReadPair<R>) -> KeyMemory {
        KeyMemory::NONE
    }
}
