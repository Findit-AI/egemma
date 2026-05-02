//! Session, batch, and threading options for [`crate::TextEncoder`].
//!
//! `GraphOptimizationLevel` and `Options::optimization_level` are
//! re-exported / present only with `feature = "inference"` — they
//! reach into `ort` types that don't exist on wasm builds.
//!
//! `serde::{Serialize, Deserialize}` derives on `Options`,
//! `BatchOptions`, and `ThreadOptions` are gated on `feature = "serde"`
//! so consumers who don't need config (de)serialization don't pay the
//! serde compile-time cost.

#[cfg(feature = "inference")]
pub use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// `optimization_level`'s `serialize` / `deserialize` adapters depend on
// both `inference` (for the `GraphOptimizationLevel` type itself) and
// `serde` (for the trait machinery). `Options::optimization_level`
// references this module only under the same conjunction.
#[cfg(all(feature = "inference", feature = "serde"))]
mod optimization_level {
  use super::GraphOptimizationLevel;
  use serde::*;

  #[derive(
    Debug, Default, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize,
  )]
  #[serde(rename_all = "snake_case")]
  enum OptimizationLevel {
    Disable,
    #[default]
    Level1,
    Level2,
    Level3,
    All,
  }

  impl From<GraphOptimizationLevel> for OptimizationLevel {
    #[inline]
    fn from(value: GraphOptimizationLevel) -> Self {
      match value {
        GraphOptimizationLevel::Disable => Self::Disable,
        GraphOptimizationLevel::Level1 => Self::Level1,
        GraphOptimizationLevel::Level2 => Self::Level2,
        GraphOptimizationLevel::Level3 => Self::Level3,
        GraphOptimizationLevel::All => Self::All,
      }
    }
  }

  impl From<OptimizationLevel> for GraphOptimizationLevel {
    #[inline]
    fn from(value: OptimizationLevel) -> Self {
      match value {
        OptimizationLevel::Disable => Self::Disable,
        OptimizationLevel::Level1 => Self::Level1,
        OptimizationLevel::Level2 => Self::Level2,
        OptimizationLevel::Level3 => Self::Level3,
        OptimizationLevel::All => Self::All,
      }
    }
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn serialize<S>(level: &GraphOptimizationLevel, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    OptimizationLevel::from(*level).serialize(serializer)
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn deserialize<'de, D>(deserializer: D) -> Result<GraphOptimizationLevel, D::Error>
  where
    D: Deserializer<'de>,
  {
    OptimizationLevel::deserialize(deserializer).map(Into::into)
  }

  // Must stay in lock-step with `Options::new()` so that deserializing a
  // config that omits `optimization_level` yields the same baseline level
  // a normal `Options::default()` would.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn default() -> GraphOptimizationLevel {
    GraphOptimizationLevel::Level1
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_max_seq_len() -> usize {
  2048
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_batch_size() -> usize {
  8
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_max_batch_size() -> usize {
  1024
}

/// Sequence-length and batching policy for [`crate::TextEncoder`].
/// Validated at encoder construction; see
/// [`Self::with_max_seq_len`] / [`Self::with_batch_size`] /
/// [`Self::with_max_batch_size`] for the tunables.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BatchOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_max_seq_len"))]
  max_seq_len: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_batch_size"))]
  batch_size: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_max_batch_size"))]
  max_batch_size: usize,
}

impl BatchOptions {
  /// Construct a `BatchOptions` with the crate defaults
  /// (`max_seq_len = 2048`, `batch_size = 8`, `max_batch_size = 1024`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      max_seq_len: default_max_seq_len(),
      batch_size: default_batch_size(),
      max_batch_size: default_max_batch_size(),
    }
  }

  /// Maximum number of tokens per input. Long inputs are truncated to
  /// this length by the tokenizer. Defaults to 2048 — `embedding-gemma`'s
  /// trained context window.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_seq_len(&self) -> usize {
    self.max_seq_len
  }

  /// Inputs per ORT inference call (chunk size).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn batch_size(&self) -> usize {
    self.batch_size
  }

  /// Hard upper bound on `texts.len()` accepted by `embed_batch`.
  /// Inputs above this are rejected with `Error::BatchTooLarge`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_batch_size(&self) -> usize {
    self.max_batch_size
  }

  /// Returns a copy with [`Self::max_seq_len`] replaced.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_seq_len(mut self, n: usize) -> Self {
    self.max_seq_len = n;
    self
  }

  /// Returns a copy with [`Self::batch_size`] replaced.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_batch_size(mut self, n: usize) -> Self {
    self.batch_size = n;
    self
  }

  /// Returns a copy with [`Self::max_batch_size`] replaced.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_batch_size(mut self, n: usize) -> Self {
    self.max_batch_size = n;
    self
  }

  /// In-place setter for [`Self::max_seq_len`]; returns `&mut self`
  /// so calls can chain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_seq_len(&mut self, n: usize) -> &mut Self {
    self.max_seq_len = n;
    self
  }

  /// In-place setter for [`Self::batch_size`]; returns `&mut self`
  /// so calls can chain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_batch_size(&mut self, n: usize) -> &mut Self {
    self.batch_size = n;
    self
  }

  /// In-place setter for [`Self::max_batch_size`]; returns `&mut self`
  /// so calls can chain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_batch_size(&mut self, n: usize) -> &mut Self {
    self.max_batch_size = n;
    self
  }

  /// Reject:
  /// - `batch_size == 0` (the silent `.max(1)` coercion footgun)
  /// - `batch_size > max_batch_size` (config error: wastes scratch
  ///   memory and never produces a chunk that large in practice)
  /// - `max_seq_len == 0` (tokenizer truncation requires `max_length > 0`;
  ///   a zero-length budget is meaningless and would otherwise leak out
  ///   as an opaque tokenizer-config error from `configure_tokenizer`)
  #[cfg_attr(not(any(feature = "inference", test)), allow(dead_code))]
  pub(crate) fn validate(&self) -> Result<(), crate::Error> {
    if self.batch_size == 0 || self.batch_size > self.max_batch_size {
      return Err(crate::Error::InvalidBatchSize {
        batch_size: self.batch_size,
        max_batch_size: self.max_batch_size,
      });
    }
    if self.max_seq_len == 0 {
      return Err(crate::Error::InvalidMaxSeqLen);
    }
    Ok(())
  }
}

impl Default for BatchOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_intra_threads() -> usize {
  1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_inter_threads() -> usize {
  1
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_parallel_execution() -> bool {
  false
}

/// ORT thread-pool configuration. Maps onto ORT session-builder
/// settings (`with_intra_threads` / `with_inter_threads` /
/// `with_parallel_execution`). All defaults are `1` / `false`,
/// matching ORT's CPU-friendly low-contention setup; tune up for
/// high-throughput offline batches.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThreadOptions {
  #[cfg_attr(feature = "serde", serde(default = "default_intra_threads"))]
  intra_threads: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_inter_threads"))]
  inter_threads: usize,
  #[cfg_attr(feature = "serde", serde(default = "default_parallel_execution"))]
  parallel_execution: bool,
}

impl ThreadOptions {
  /// Construct with crate defaults (1 intra-op thread, 1 inter-op
  /// thread, parallel execution off).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      intra_threads: default_intra_threads(),
      inter_threads: default_inter_threads(),
      parallel_execution: default_parallel_execution(),
    }
  }

  /// Intra-op thread count — ORT's per-operator parallelism.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn intra_threads(&self) -> usize {
    self.intra_threads
  }

  /// Inter-op thread count — ORT's between-operator parallelism.
  /// Only meaningful when [`Self::parallel_execution`] is `true`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn inter_threads(&self) -> usize {
    self.inter_threads
  }

  /// Whether ORT runs independent operators concurrently. Most
  /// embedding workloads don't benefit; off by default.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn parallel_execution(&self) -> bool {
    self.parallel_execution
  }

  /// Returns a copy with [`Self::intra_threads`] replaced.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_intra_threads(mut self, n: usize) -> Self {
    self.intra_threads = n;
    self
  }

  /// Returns a copy with [`Self::inter_threads`] replaced.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_inter_threads(mut self, n: usize) -> Self {
    self.inter_threads = n;
    self
  }

  /// Returns a copy with [`Self::parallel_execution`] replaced.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_parallel_execution(mut self, p: bool) -> Self {
    self.parallel_execution = p;
    self
  }

  /// In-place setter for [`Self::intra_threads`]; returns `&mut self`
  /// so calls can chain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_intra_threads(&mut self, n: usize) -> &mut Self {
    self.intra_threads = n;
    self
  }

  /// In-place setter for [`Self::inter_threads`]; returns `&mut self`
  /// so calls can chain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_inter_threads(&mut self, n: usize) -> &mut Self {
    self.inter_threads = n;
    self
  }

  /// In-place setter for [`Self::parallel_execution`]; returns
  /// `&mut self` so calls can chain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_parallel_execution(&mut self, p: bool) -> &mut Self {
    self.parallel_execution = p;
    self
  }
}

impl Default for ThreadOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

/// Top-level configuration passed to [`crate::TextEncoder`]
/// constructors. Bundles ORT graph-optimization level
/// (gated on `feature = "inference"`), [`BatchOptions`], and
/// [`ThreadOptions`].
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  #[cfg(feature = "inference")]
  #[cfg_attr(
    feature = "serde",
    serde(with = "optimization_level", default = "optimization_level::default")
  )]
  optimization_level: GraphOptimizationLevel,
  #[cfg_attr(feature = "serde", serde(default))]
  batch: BatchOptions,
  #[cfg_attr(feature = "serde", serde(default))]
  threads: ThreadOptions,
}

impl Options {
  /// Construct with crate defaults
  /// (`Level1` optimization, default `BatchOptions`, default
  /// `ThreadOptions`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      #[cfg(feature = "inference")]
      optimization_level: GraphOptimizationLevel::Level1,
      batch: BatchOptions::new(),
      threads: ThreadOptions::new(),
    }
  }

  /// ORT graph-optimization level applied at session-build time.
  #[cfg(feature = "inference")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level
  }

  /// The nested [`BatchOptions`] for sequence-length and chunking
  /// policy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn batch(&self) -> BatchOptions {
    self.batch
  }

  /// The nested [`ThreadOptions`] for ORT thread-pool tuning.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threads(&self) -> ThreadOptions {
    self.threads
  }

  /// Returns a copy with [`Self::optimization_level`] replaced.
  #[cfg(feature = "inference")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_optimization_level(mut self, l: GraphOptimizationLevel) -> Self {
    self.optimization_level = l;
    self
  }

  /// Returns a copy with [`Self::batch`] replaced.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_batch(mut self, b: BatchOptions) -> Self {
    self.batch = b;
    self
  }

  /// Returns a copy with [`Self::threads`] replaced.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_threads(mut self, t: ThreadOptions) -> Self {
    self.threads = t;
    self
  }

  /// In-place setter for [`Self::optimization_level`]; returns
  /// `&mut self` so calls can chain.
  #[cfg(feature = "inference")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_optimization_level(&mut self, l: GraphOptimizationLevel) -> &mut Self {
    self.optimization_level = l;
    self
  }

  /// In-place setter for [`Self::batch`]; returns `&mut self` so
  /// calls can chain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_batch(&mut self, b: BatchOptions) -> &mut Self {
    self.batch = b;
    self
  }

  /// In-place setter for [`Self::threads`]; returns `&mut self` so
  /// calls can chain.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_threads(&mut self, t: ThreadOptions) -> &mut Self {
    self.threads = t;
    self
  }
}

impl Default for Options {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[cfg(feature = "inference")]
  #[test]
  fn defaults_match_spec() {
    let o = Options::default();
    assert_eq!(o.optimization_level(), GraphOptimizationLevel::Level1);
    assert_eq!(o.batch().max_seq_len(), 2048);
    assert_eq!(o.batch().batch_size(), 8);
    assert_eq!(o.batch().max_batch_size(), 1024);
    assert_eq!(o.threads().intra_threads(), 1);
    assert_eq!(o.threads().inter_threads(), 1);
    assert!(!o.threads().parallel_execution());
  }

  #[cfg(feature = "inference")]
  #[test]
  fn builder_chains_compose() {
    let o = Options::default()
      .with_optimization_level(GraphOptimizationLevel::Level3)
      .with_batch(BatchOptions::default().with_batch_size(32))
      .with_threads(ThreadOptions::default().with_intra_threads(4));

    assert_eq!(o.optimization_level(), GraphOptimizationLevel::Level3);
    assert_eq!(o.batch().batch_size(), 32);
    assert_eq!(o.threads().intra_threads(), 4);
  }

  #[test]
  fn options_is_copy() {
    fn _require_copy<T: Copy>() {}
    _require_copy::<Options>();
    _require_copy::<BatchOptions>();
    _require_copy::<ThreadOptions>();
  }

  #[test]
  fn validate_rejects_zero_batch_size() {
    let bad = BatchOptions::default().with_batch_size(0);
    match bad.validate() {
      Err(crate::Error::InvalidBatchSize {
        batch_size: 0,
        max_batch_size: 1024,
      }) => {}
      other => panic!("expected InvalidBatchSize {{ 0, 1024 }}, got {other:?}"),
    }
  }

  #[test]
  fn validate_rejects_batch_size_above_max() {
    let bad = BatchOptions::default()
      .with_batch_size(2048)
      .with_max_batch_size(1024);
    match bad.validate() {
      Err(crate::Error::InvalidBatchSize {
        batch_size: 2048,
        max_batch_size: 1024,
      }) => {}
      other => panic!("expected InvalidBatchSize {{ 2048, 1024 }}, got {other:?}"),
    }
  }

  #[test]
  fn validate_accepts_default() {
    BatchOptions::default()
      .validate()
      .expect("default BatchOptions must validate (8 / 1024)");
  }

  #[cfg(all(feature = "inference", feature = "serde"))]
  #[test]
  fn deserializing_empty_object_equals_default() {
    let from_empty: Options = serde_json::from_str("{}").expect("empty options");
    let dflt = Options::default();
    assert_eq!(from_empty.optimization_level(), dflt.optimization_level());
    assert_eq!(from_empty.batch().max_seq_len(), dflt.batch().max_seq_len());
    assert_eq!(from_empty.batch().batch_size(), dflt.batch().batch_size());
    assert_eq!(
      from_empty.batch().max_batch_size(),
      dflt.batch().max_batch_size()
    );
  }
}
