//! Genomics-opinionated disk-spilling sort for FASTQ/FASTA records.
//!
//! `spillover-bio` builds on the generic [`spillover`] crate to
//! provide a ready-to-use external sorter for sequence records. It
//! supplies genomics-specific sort keys, sequence-focused dedup
//! strategies on sorted output, and uses the `dryice` format for
//! temporary on-disk storage.
//! Use [`Builder`] for disk-spilling sorting and [`InMemoryBuilder`] when
//! the input fits in memory. Extension protocols remain in their modules.

pub mod codec;
pub mod key;
pub mod radix;
pub mod record;
pub mod sort;

#[doc(inline)]
pub use codec::{DryIceCodec, PairCodecError};
#[doc(inline)]
pub use record::{ReadPair, SeqRecord, SeqRecordArena, SeqRecordParts, SeqRecordView};
#[doc(inline)]
pub use sort::in_memory::Builder as InMemoryBuilder;
#[doc(inline)]
pub use sort::{Builder, SeqRecordSink};
