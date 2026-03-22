//! Genomics-opinionated disk-spilling sort for FASTQ/FASTA records.
//!
//! `spillover-bio` builds on the generic [`spillover`] crate to
//! provide a ready-to-use external sorter for sequence records. It
//! supplies bloom filter-based deduplication, genomics-specific sort
//! keys, and uses the `dryice` format for temporary on-disk storage.

pub mod codec;
pub mod key;
pub mod record;
