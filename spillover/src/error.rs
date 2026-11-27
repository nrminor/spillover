//! Error types for the spillover crate.
//!
//! Sorter and merge operations return [`MergeError`](crate::merge::MergeError),
//! preserving codec errors in its `Codec` variant. Borrowed traversal
//! distinguishes source and sink failures with [`SortedItemsError`].

/// Error returned while visiting finalized sorted output through
/// [`SortedItems`](crate::SortedItems).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SortedItemsError<E, F> {
    /// The sorted output source failed to produce the next item.
    #[error("sorted output source failed: {0}")]
    Source(E),

    /// The item sink rejected or failed to process an item.
    #[error("sorted item sink failed: {0}")]
    Sink(F),
}
