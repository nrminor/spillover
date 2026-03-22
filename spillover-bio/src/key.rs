//! Packed sequence keys for dryice record key storage.
//!
//! [`PackedSequenceKey`] is a const-generic fixed-width key that
//! stores 2-bit-packed nucleotide sequence data. It implements
//! dryice's [`RecordKey`] trait, so it can be stored alongside
//! records in dryice files and compared during the merge phase
//! without deserializing full records.
//!
//! 2-bit packing via [`bitnuc`] gives 4 bases per byte. The
//! packed representation preserves lexicographic order — byte-level
//! comparison gives the same ordering as base-level comparison.
//!
//! Type aliases are provided for common read lengths:
//! - [`IlluminaKey`]: 38 bytes = 152 bases (covers 150bp reads)
//! - [`PairedEndKey`]: 64 bytes = 256 bases (covers 250bp reads)
//! - [`LongReadPrefixKey`]: 128 bytes = 512 bases (prefix for long reads)

use dryice::{DryIceError, RecordKey};

/// Build a stable 16-byte type tag encoding the key width.
///
/// The tag format is `spill:seq2b:NNNN` where NNNN is the
/// zero-padded decimal width in bytes. Key widths above 9999
/// are not supported (and would be pathological).
#[allow(clippy::cast_possible_truncation)]
const fn make_tag(n: usize) -> [u8; 16] {
    // Truncation is safe: we're extracting decimal digits (0-9),
    // which always fit in u8.
    let mut tag = *b"spill:seq2b:0000";
    tag[12] = b'0' + ((n / 1000) % 10) as u8;
    tag[13] = b'0' + ((n / 100) % 10) as u8;
    tag[14] = b'0' + ((n / 10) % 10) as u8;
    tag[15] = b'0' + (n % 10) as u8;
    tag
}

/// A fixed-width 2-bit-packed sequence key for merge acceleration.
///
/// Stores the first `N` bytes of 2-bit-packed sequence data,
/// covering `N * 4` bases. Implements [`Ord`] via lexicographic
/// byte comparison, which preserves the nucleotide ordering
/// (A=00 < C=01 < G=10 < T=11).
///
/// For reads shorter than `N * 4` bases, the remaining bytes
/// are zero-padded, which sorts short reads before longer ones
/// that share the same prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PackedSequenceKey<const N: usize>(pub [u8; N]);

impl<const N: usize> PackedSequenceKey<N> {
    /// The number of nucleotide bases this key can represent.
    pub const BASES: usize = N * 4;

    /// Pack the first [`BASES`](Self::BASES) bases of a sequence
    /// into a key. Bases beyond the key's capacity are truncated.
    /// Sequences shorter than the capacity are zero-padded.
    ///
    /// Ambiguous bases (anything other than A, C, G, T) are
    /// mapped to A (0b00) in the packed representation. The full
    /// sequence with ambiguities preserved is stored in the
    /// record payload for tiebreaking.
    #[must_use]
    pub fn from_sequence(sequence: &[u8]) -> Self {
        let mut key = [0u8; N];
        let bases_to_pack = sequence.len().min(Self::BASES);

        for (i, &base) in sequence[..bases_to_pack].iter().enumerate() {
            let bits = match base {
                b'C' | b'c' => 0b01,
                b'G' | b'g' => 0b10,
                b'T' | b't' => 0b11,
                // A and all ambiguous bases map to 0b00
                _ => 0b00,
            };
            let byte_idx = i / 4;
            let bit_offset = 6 - (i % 4) * 2;
            key[byte_idx] |= bits << bit_offset;
        }

        Self(key)
    }
}

// Allow: N is a key width in bytes; realistic values are well
// within u16 range (max ~128 for long-read prefixes).
#[allow(clippy::cast_possible_truncation)]
impl<const N: usize> RecordKey for PackedSequenceKey<N> {
    const WIDTH: u16 = N as u16;
    const TYPE_TAG: [u8; 16] = make_tag(N);

    fn encode_into(&self, out: &mut [u8]) {
        debug_assert_eq!(out.len(), N);
        out.copy_from_slice(&self.0);
    }

    fn decode_from(bytes: &[u8]) -> Result<Self, DryIceError> {
        let arr: [u8; N] = bytes
            .try_into()
            .map_err(|_| DryIceError::InvalidRecordKeyEncoding {
                message: "packed sequence key length mismatch",
            })?;
        Ok(Self(arr))
    }
}

/// 38-byte key covering 152 bases — full Illumina 150bp reads.
pub type IlluminaKey = PackedSequenceKey<38>;

/// 64-byte key covering 256 bases — full 250bp paired-end reads.
pub type PairedEndKey = PackedSequenceKey<64>;

/// 128-byte key covering 512 bases — prefix for long reads.
pub type LongReadPrefixKey = PackedSequenceKey<128>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_simple_sequence() {
        let key = PackedSequenceKey::<2>::from_sequence(b"ACGTACGT");
        // A=00, C=01, G=10, T=11 → byte 0: 00_01_10_11 = 0x1B
        // A=00, C=01, G=10, T=11 → byte 1: 00_01_10_11 = 0x1B
        assert_eq!(key.0, [0x1B, 0x1B]);
    }

    #[test]
    fn pack_preserves_lexicographic_order() {
        let key_a = PackedSequenceKey::<2>::from_sequence(b"AAAAAAAA");
        let key_c = PackedSequenceKey::<2>::from_sequence(b"CCCCCCCC");
        let key_g = PackedSequenceKey::<2>::from_sequence(b"GGGGGGGG");
        let key_t = PackedSequenceKey::<2>::from_sequence(b"TTTTTTTT");

        assert!(key_a < key_c);
        assert!(key_c < key_g);
        assert!(key_g < key_t);
    }

    #[test]
    fn pack_short_sequence_zero_pads() {
        let key = PackedSequenceKey::<4>::from_sequence(b"AC");
        // A=00 at bits 7:6, C=01 at bits 5:4 → 0b00_01_00_00 = 0x10
        // remaining 3 bytes = 0
        assert_eq!(key.0[0], 0x10);
        assert_eq!(key.0[1], 0);
        assert_eq!(key.0[2], 0);
        assert_eq!(key.0[3], 0);
    }

    #[test]
    fn short_sequence_sorts_before_longer_with_same_prefix() {
        let short = PackedSequenceKey::<4>::from_sequence(b"AC");
        let long = PackedSequenceKey::<4>::from_sequence(b"ACGTACGTACGTACGT");
        assert!(
            short < long,
            "zero-padded short sequence should sort before longer one"
        );
    }

    #[test]
    fn pack_handles_lowercase() {
        let upper = PackedSequenceKey::<2>::from_sequence(b"ACGTACGT");
        let lower = PackedSequenceKey::<2>::from_sequence(b"acgtacgt");
        assert_eq!(upper, lower, "case should not affect packing");
    }

    #[test]
    fn pack_maps_ambiguous_to_a() {
        let with_n = PackedSequenceKey::<1>::from_sequence(b"NCGT");
        let with_a = PackedSequenceKey::<1>::from_sequence(b"ACGT");
        assert_eq!(
            with_n, with_a,
            "ambiguous bases should map to A in the packed key"
        );
    }

    #[test]
    fn record_key_round_trips() {
        let key = PackedSequenceKey::<8>::from_sequence(b"ACGTACGTACGTACGTACGTACGTACGTACGT");
        let mut buf = vec![0u8; 8];
        key.encode_into(&mut buf);
        let decoded = PackedSequenceKey::<8>::decode_from(&buf).expect("decode should succeed");
        assert_eq!(key, decoded);
    }

    #[test]
    fn type_tag_encodes_width() {
        assert_eq!(&PackedSequenceKey::<38>::TYPE_TAG, b"spill:seq2b:0038");
        assert_eq!(&PackedSequenceKey::<64>::TYPE_TAG, b"spill:seq2b:0064");
        assert_eq!(&PackedSequenceKey::<128>::TYPE_TAG, b"spill:seq2b:0128");
    }

    #[test]
    fn illumina_key_covers_150bp() {
        assert_eq!(IlluminaKey::BASES, 152, "38 bytes × 4 = 152 bases");
    }

    #[test]
    fn paired_end_key_covers_256bp() {
        assert_eq!(PairedEndKey::BASES, 256);
    }

    #[test]
    fn long_read_prefix_key_covers_512bp() {
        assert_eq!(LongReadPrefixKey::BASES, 512);
    }
}
