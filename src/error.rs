//! Error type for the `egemma` crate.

#[cfg(feature = "inference")]
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
  /// ORT-backed graph load failure. Gated on the `inference` feature
  /// because `ort::Error` doesn't exist when the feature is off.
  #[cfg(feature = "inference")]
  #[error("failed to load ONNX graph at {path}: {source}")]
  LoadGraph { path: PathBuf, source: ort::Error },

  /// Required ONNX output tensor was not present in the session output map.
  /// Indicates an unexpected re-export or a corrupted graph.
  #[error("required ONNX output `{name}` was missing from session run")]
  MissingOnnxOutput { name: &'static str },

  #[error("tokenizer load failed: {0}")]
  Tokenizer(String),

  #[error("unexpected output rank: expected 2, got {rank} with shape {shape:?}")]
  OutputRank { rank: usize, shape: Vec<i64> },

  #[error("session shape mismatch on `{input}`: expected {expected}, got {got:?}")]
  SessionShapeMismatch {
    input: &'static str,
    expected: &'static str,
    got: Vec<i64>,
  },

  #[error("embedding dimension mismatch: expected {expected}, got {got}")]
  EmbeddingDim { expected: usize, got: usize },

  #[error("embedding is not unit-norm (got ||v||₂ = {norm}, tolerance ε = {epsilon})")]
  NotNormalized { norm: f32, epsilon: f32 },

  #[error("text input is empty")]
  EmptyText,

  #[error("batch size {got} exceeds maximum {max}")]
  BatchTooLarge { got: usize, max: usize },

  /// `BatchOptions::batch_size` was outside the legal range
  /// `1..=max_batch_size` at encoder construction.
  #[error("invalid batch_size {batch_size}: must be in 1..={max_batch_size}")]
  InvalidBatchSize {
    batch_size: usize,
    max_batch_size: usize,
  },

  #[error("batch index {index}: {source}")]
  Batch { index: usize, source: Box<Error> },

  /// ORT runtime error pass-through. Gated on the `inference` feature
  /// because `ort::Error` doesn't exist when the feature is off.
  #[cfg(feature = "inference")]
  #[error(transparent)]
  Ort(#[from] ort::Error),

  #[error(transparent)]
  Io(#[from] std::io::Error),
}

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
