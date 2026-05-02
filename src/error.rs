//! Error type for the `egemma` crate.

#[cfg(feature = "inference")]
use std::path::PathBuf;
use thiserror::Error;

/// All errors surfaced from the public API.
///
/// `#[non_exhaustive]` so that adding variants in a future minor
/// release isn't a breaking change for `match` arms — downstream
/// callers must include a wildcard (`_ => ...`) branch.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
  /// ORT-backed graph load failure. Gated on the `inference` feature
  /// because `ort::Error` doesn't exist when the feature is off.
  #[cfg(feature = "inference")]
  #[error("failed to load ONNX graph at {path}: {source}")]
  LoadGraph {
    /// Path that was passed to `commit_from_file`.
    path: PathBuf,
    /// Underlying `ort` error from the session-builder pipeline.
    source: ort::Error,
  },

  /// Required ONNX output tensor was not present in the session output map.
  /// Indicates an unexpected re-export or a corrupted graph.
  #[error("required ONNX output `{name}` was missing from session run")]
  MissingOnnxOutput {
    /// Name of the missing output (e.g. `"sentence_embedding"`).
    name: &'static str,
  },

  /// Tokenizer load OR runtime use failure. Covers `Tokenizer::from_file`
  /// errors at construction, `<pad>`-token contract violations during
  /// configuration, `encode_batch` failures during inference, and any
  /// uneven-row anomalies surfaced from the tokenizers crate.
  #[error("tokenizer error: {0}")]
  Tokenizer(String),

  /// ORT returned a tensor whose rank wasn't 2 (we expect
  /// `[batch, EMBED_DIM]`).
  #[error("unexpected output rank: expected 2, got {rank} with shape {shape:?}")]
  OutputRank {
    /// Number of dimensions in the returned tensor.
    rank: usize,
    /// Full shape vector for diagnostics.
    shape: Vec<i64>,
  },

  /// Session-level shape contract violation: a required outlet was
  /// missing, had the wrong rank, had a static dim where we needed
  /// a dynamic one, or had a static dim that didn't match expectations.
  #[error("session shape mismatch on `{input}`: expected {expected}, got {got:?}")]
  SessionShapeMismatch {
    /// Outlet name that didn't satisfy the contract.
    input: &'static str,
    /// Human-readable expectation message.
    expected: &'static str,
    /// Actual shape from the session metadata.
    got: Vec<i64>,
  },

  /// Session contract violation that isn't a shape mismatch — wrong
  /// element type, missing outlet, or non-tensor outlet. Carries the
  /// actual `TensorElementType` so users debugging a bad re-export
  /// see the dtype, not a shape vector that doesn't apply. Gated on
  /// `feature = "inference"` because the `got` field is an `ort` type.
  #[cfg(feature = "inference")]
  #[error("session contract mismatch on `{input}`: expected {expected}, got {got:?}")]
  SessionContractMismatch {
    /// Outlet name that didn't satisfy the contract.
    input: &'static str,
    /// Human-readable expectation message.
    expected: &'static str,
    /// Actual tensor element type from the session metadata.
    got: ort::value::TensorElementType,
  },

  /// `Embedding` constructed from a `Vec<f32>` whose length didn't
  /// equal [`crate::Embedding::EMBED_DIM`] (768).
  #[error("embedding dimension mismatch: expected {expected}, got {got}")]
  EmbeddingDim {
    /// Required dim (always 768 in 0.1.0).
    expected: usize,
    /// Caller-supplied dim.
    got: usize,
  },

  /// `Embedding::try_from(Vec<f32>)` rejected an input whose
  /// `||v||₂` was outside `[1 - ε, 1 + ε]`. The encoder path
  /// normalizes raw model output unconditionally — this variant
  /// only fires for caller-supplied vectors that should already be
  /// unit-norm (e.g. deserialized from a vector store).
  #[error("embedding is not unit-norm (got ||v||₂ = {norm}, tolerance ε = {epsilon})")]
  NotNormalized {
    /// Computed L2 norm of the input vector.
    norm: f32,
    /// Tolerance window the norm had to fall inside.
    epsilon: f32,
  },

  /// An empty string was passed to [`crate::TextEncoder::embed`] or
  /// appeared inside the slice given to
  /// [`crate::TextEncoder::embed_batch`].
  #[error("text input is empty")]
  EmptyText,

  /// The slice passed to [`crate::TextEncoder::embed_batch`] exceeded
  /// `BatchOptions::max_batch_size`.
  #[error("batch size {got} exceeds maximum {max}")]
  BatchTooLarge {
    /// Number of inputs in the call.
    got: usize,
    /// Configured upper bound.
    max: usize,
  },

  /// `BatchOptions::batch_size` was outside the legal range
  /// `1..=max_batch_size` at encoder construction.
  #[error("invalid batch_size {batch_size}: must be in 1..={max_batch_size}")]
  InvalidBatchSize {
    /// The supplied (rejected) batch size.
    batch_size: usize,
    /// The configured upper bound.
    max_batch_size: usize,
  },

  /// `BatchOptions::max_seq_len` was zero at encoder construction.
  /// Tokenizer truncation requires `max_length > 0`; a zero-length
  /// budget is meaningless. Caught alongside `InvalidBatchSize` so
  /// shape-of-options errors stay together rather than leaking out
  /// as opaque tokenizer-config errors.
  #[error("invalid max_seq_len 0: must be > 0")]
  InvalidMaxSeqLen,

  /// Batched-failure envelope: wraps the underlying error with the
  /// position of the offending input. See
  /// [`crate::TextEncoder::embed_batch`] for the indexing
  /// granularity (row-precise vs chunk-level).
  #[error("batch index {index}: {source}")]
  Batch {
    /// Zero-based index into the input slice.
    index: usize,
    /// Underlying error.
    source: Box<Error>,
  },

  /// ORT runtime error pass-through. Gated on the `inference` feature
  /// because `ort::Error` doesn't exist when the feature is off.
  #[cfg(feature = "inference")]
  #[error(transparent)]
  Ort(#[from] ort::Error),

  /// Filesystem / I/O error pass-through (e.g. when reading a model
  /// file).
  #[error(transparent)]
  Io(#[from] std::io::Error),
}

/// Crate-local `Result` alias parameterized on the [`Error`](enum@Error)
/// enum. Disambiguated because `thiserror::Error` (the derive macro) is
/// also in scope here.
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn empty_text_displays_message() {
    assert_eq!(Error::EmptyText.to_string(), "text input is empty");
  }

  #[test]
  fn batch_wraps_inner_error() {
    let inner = Error::EmptyText;
    let wrapped = Error::Batch {
      index: 3,
      source: Box::new(inner),
    };
    assert_eq!(wrapped.to_string(), "batch index 3: text input is empty");
  }

  #[test]
  fn embedding_dim_mismatch_shows_expected_and_got() {
    let err = Error::EmbeddingDim {
      expected: 768,
      got: 512,
    };
    assert_eq!(
      err.to_string(),
      "embedding dimension mismatch: expected 768, got 512"
    );
  }
}
