//! Sliding-window long-input support (the `windowing` feature). Splits text
//! longer than the model's window into overlapping chunks, each embedded
//! through the normal per-text path; `embed_pooled` combines them.

use core::ops::Range;

/// How to cut a long text into windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowStrategy {
  /// Exact token-count windows, cut at token boundaries via tokenizer byte
  /// offsets. Deterministic; may split mid-sentence.
  FixedToken,
  /// Token-budgeted windows that prefer sentence/word boundaries (via
  /// `text-splitter`). Better-formed chunks; window token counts vary.
  Semantic,
}

/// Windowing configuration for [`crate::TextEncoder::embed_windows`] /
/// [`crate::TextEncoder::embed_pooled`].
#[derive(Clone, Copy, Debug)]
pub struct WindowOptions {
  strategy: WindowStrategy,
  /// Max real tokens per window. `None` → the encoder's `max_seq_len` minus the
  /// tokenizer's special-token overhead (so a re-embedded chunk never truncates).
  size: Option<usize>,
  /// Token overlap between consecutive windows (0 = no overlap).
  overlap: usize,
}

impl WindowOptions {
  /// Construct with a strategy, default size (encoder `max_seq_len`), no overlap.
  pub const fn new(strategy: WindowStrategy) -> Self {
    Self {
      strategy,
      size: None,
      overlap: 0,
    }
  }

  /// The cut strategy.
  pub const fn strategy(&self) -> WindowStrategy {
    self.strategy
  }

  /// The configured max tokens per window, if overridden.
  pub const fn size(&self) -> Option<usize> {
    self.size
  }

  /// The token overlap between windows.
  pub const fn overlap(&self) -> usize {
    self.overlap
  }

  /// Set an explicit window size (max real tokens per window).
  pub const fn with_size(mut self, n: usize) -> Self {
    self.size = Some(n);
    self
  }

  /// Set the token overlap between consecutive windows.
  pub const fn with_overlap(mut self, n: usize) -> Self {
    self.overlap = n;
    self
  }
}

/// One window's embedding plus where it came from in the original text.
#[derive(Clone, Debug)]
pub struct WindowEmbedding {
  /// Byte range of this window within the original `text`.
  pub byte_span: Range<usize>,
  /// The window's L2-normalized embedding.
  pub embedding: crate::Embedding,
}
