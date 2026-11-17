//! Error types for the spillover crate.
//!
//! Sorter and merge operations return [`MergeError`](crate::merge::MergeError),
//! preserving codec errors in its `Codec` variant. Borrowed traversal
//! distinguishes source and sink failures with [`SortedItemsError`].
//! [`SpilloverError`] is a separate error enum, not the sorter's return type.

/// Errors originating from spillover's own operations.
///
/// This covers the failure modes inherent to disk-spilling
/// external sort: I/O errors when writing or reading temporary
/// files, and detection of corrupt (truncated) run files.
///
/// Sorter and merge APIs use [`MergeError`](crate::merge::MergeError) instead,
/// including its variant for concrete codec errors.
///
/// ```
/// use spillover::SpilloverError;
///
/// let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
/// let spill_err = SpilloverError::from(io_err);
/// assert!(matches!(spill_err, SpilloverError::Io(_)));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SpilloverError {
    /// An I/O error occurred during flush, merge, or temp file
    /// operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A temporary run file ended with a partial record, indicating
    /// truncation or on-disk corruption of a file the sorter itself
    /// wrote.
    #[error("temporary run file ended with a partial record")]
    TruncatedRun,
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_converts_via_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = SpilloverError::from(io_err);
        assert!(
            matches!(err, SpilloverError::Io(_)),
            "From<io::Error> should produce SpilloverError::Io"
        );
    }

    #[test]
    fn truncated_run_displays_correctly() {
        let err = SpilloverError::TruncatedRun;
        assert!(
            err.to_string().contains("partial record"),
            "TruncatedRun display should mention partial record"
        );
    }

    #[test]
    fn error_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SpilloverError>();
    }

    #[test]
    fn error_source_chains_for_io() {
        use std::error::Error;

        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let err = SpilloverError::from(io_err);
        assert!(
            err.source().is_some(),
            "SpilloverError::Io should have a source error"
        );
    }
}
