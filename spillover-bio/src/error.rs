//! Error types for spillover-bio.

/// Error returned while draining a finalized sorted record stream.
///
/// `RE` is the error type produced while reading sorted records from the stream.
/// `WE` is the error type produced by the destination
/// [`SeqRecordSink`](crate::sort::SeqRecordSink).
#[derive(Debug, thiserror::Error)]
pub enum SortedRecordStreamError<RE, WE>
where
    RE: std::error::Error + 'static,
    WE: std::error::Error + 'static,
{
    /// The sorted stream failed while producing the next record.
    #[error("sorted record stream failed: {0}")]
    Source(#[source] RE),

    /// The destination failed while accepting a sorted record.
    #[error("sorted record destination failed: {0}")]
    Sink(#[source] WE),
}
