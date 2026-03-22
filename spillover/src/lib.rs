//! Generic, disk-spilling external sort with pluggable deduplication.
//!
//! `spillover` provides the core machinery for sorting datasets that
//! exceed available memory by flushing sorted runs to temporary files
//! on disk and merging them back via a k-way merge. It is deliberately
//! unopinionated about the data being sorted, the sort key, the
//! deduplication strategy, and the on-disk serialization format.
//!
//! Domain-specific crates (like `spillover-bio` for genomics) inject
//! their own implementations of these traits to build a complete
//! sorting pipeline tailored to their data types and workflows.

pub mod chunk;
pub mod codec;
pub mod compare;
pub mod dedup;
mod error;
pub mod key;
pub mod merge;
pub mod sorter;

pub use error::SpilloverError;
pub use get_size2::GetSize;
