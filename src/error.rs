//! Error type for the `egemma` crate.

#[cfg(feature = "inference")]
use std::path::PathBuf;
use thiserror::Error;

/// Which phase of the MLX backend produced an [`Error::Mlx`]: reading/validating
/// the checkpoint config, loading weights, or the inference forward pass. Lets
/// callers branch (e.g. retry `Runtime`, fail-fast `Config`) even though the
/// underlying `mlxrs::Error` text is opaque.
#[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxErrorKind {
  /// Config read / parse / `Gemma3Config::validate`.
  Config,
  /// Weight load / `sanitize` / `from_weights` / pooling-config read.
  Load,
  /// Inference: tensor build, `encode_text`, eval, or output-shape extraction.
  Runtime,
}

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

  /// A specific [`crate::options::Backend`] was requested via
  /// [`crate::Options::with_backend`] but cannot be honored — e.g.
  /// [`Backend::Mlx`](crate::options::Backend::Mlx) on a non-Apple-Silicon
  /// target, or when the directory holds no checkpoint for the requested
  /// backend.
  #[error("requested backend {requested:?} is unavailable: {reason}")]
  BackendUnavailable {
    /// The backend the caller asked for.
    requested: crate::options::Backend,
    /// Why it could not be honored.
    reason: String,
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

  /// Error from the MLX (`mlxrs`) inference backend — checkpoint load,
  /// tokenization, or the bidirectional backbone / Dense-head forward pass.
  /// Compiled only on the Apple-Silicon target (the only place the backend
  /// exists). The `mlxrs::Error` is captured as its `Display` string so this
  /// crate's public `Error` does not leak the `mlxrs` type into its API.
  #[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
  #[error("mlx backend {kind:?} error: {message}")]
  Mlx {
    /// Which phase of the backend failed.
    kind: MlxErrorKind,
    /// Human-readable description of the MLX backend failure.
    message: String,
  },

  /// `Vec::try_reserve_exact` returned an error — the global allocator could
  /// not satisfy a text-batch scratch request on the MLX path. Surfaced as a
  /// typed error rather than a process abort. `requested_bytes` helps callers
  /// tell whether they hit a cap they chose versus system memory pressure.
  ///
  /// `cause` is named (not `source`) because `TryReserveError` does not
  /// implement `std::error::Error` on stable Rust today, so its `Display` is
  /// captured as a string. Compiled only on the Apple-Silicon target (the only
  /// place the MLX backend exists).
  #[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
  #[error("failed to allocate {requested_bytes} bytes for `{which}` scratch buffer: {cause}")]
  AllocationFailed {
    /// Buffer the allocator was asked to reserve.
    which: &'static str,
    /// Number of bytes that were requested.
    requested_bytes: usize,
    /// `Display` representation of the underlying `TryReserveError`.
    cause: String,
  },
}

#[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
impl Error {
  /// Build an [`Error::Mlx`] from a static reason string.
  pub(crate) fn mlx(kind: MlxErrorKind, reason: &'static str) -> Self {
    Error::Mlx {
      kind,
      message: reason.to_string(),
    }
  }

  /// Build an [`Error::Mlx`] from an owned reason string.
  pub(crate) fn mlx_owned(kind: MlxErrorKind, reason: String) -> Self {
    Error::Mlx {
      kind,
      message: reason,
    }
  }

  /// Convert an `mlxrs::Error` into [`Error::Mlx`], capturing its `Display`.
  pub(crate) fn from_mlx(kind: MlxErrorKind, source: mlxrs::Error) -> Self {
    Error::Mlx {
      kind,
      message: source.to_string(),
    }
  }
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

  #[test]
  fn backend_unavailable_displays_requested_and_reason() {
    let e = Error::BackendUnavailable {
      requested: crate::options::Backend::Mlx,
      reason: "not apple silicon".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("Mlx"), "got {msg:?}");
    assert!(msg.contains("not apple silicon"), "got {msg:?}");
  }

  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  #[test]
  fn mlx_error_displays_kind_and_message() {
    let e = Error::Mlx {
      kind: MlxErrorKind::Runtime,
      message: "boom".to_string(),
    };
    assert_eq!(e.to_string(), "mlx backend Runtime error: boom");
  }
}
