//! Error types for the spillover crate.
//!
//! [`SpilloverError`] covers failures that originate within
//! spillover itself: I/O errors during flush and merge operations,
//! and corrupt temporary files. Errors from user-provided trait
//! implementations (`Codec`, `Dedup`) remain as their concrete
//! types — spillover does not wrap or erase them. Users compose
//! these with `SpilloverError` in whatever way suits their
//! application, whether via a custom enum with `#[from]`, a
//! boxed trait object, or any other pattern.

/// Errors originating from spillover's own operations.
///
/// This covers the failure modes inherent to disk-spilling
/// external sort: I/O errors when writing or reading temporary
/// files, and detection of corrupt (truncated) run files.
///
/// Errors from user-provided `Codec` and `Dedup` implementations
/// are not included here — they stay as their concrete associated
/// types. Users who want a unified error type for their pipeline
/// can wrap `SpilloverError` alongside their codec/dedup errors
/// in their own enum.
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
