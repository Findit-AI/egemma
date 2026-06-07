//! Sliding-window long-input support (the `windowing` feature). Splits text
//! longer than the model's window into overlapping chunks, each embedded
//! through the normal per-text path; `embed_pooled` combines them.

use core::ops::Range;

use tokenizers::Tokenizer;

use crate::error::{Error, Result};

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

/// Windowing configuration for [`crate::TextEncoder`]'s `embed_windows` /
/// `embed_pooled`.
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

/// Split `text` into `(byte_span, chunk_text)` windows per `opts`. `tokenizer`
/// MUST have truncation disabled (the caller clones + clears it). `max_seq_len`
/// is the encoder's per-input cap; the effective per-window token budget is
/// `opts.size().unwrap_or(max_seq_len).saturating_sub(special_reserve)` so a
/// re-embedded chunk (which re-adds `special_reserve` special tokens) stays
/// within `max_seq_len`.
pub(crate) fn split_windows(
  tokenizer: &Tokenizer,
  text: &str,
  opts: &WindowOptions,
  max_seq_len: usize,
  special_reserve: usize,
) -> Result<Vec<(Range<usize>, String)>> {
  let budget = opts
    .size()
    .unwrap_or(max_seq_len)
    .saturating_sub(special_reserve)
    .max(1);
  match opts.strategy() {
    WindowStrategy::FixedToken => fixed_token(tokenizer, text, budget, opts.overlap()),
    WindowStrategy::Semantic => semantic(tokenizer, text, budget, opts.overlap()),
  }
}

/// Exact token-count windows using tokenizer byte offsets.
///
/// Encodes the full text, then slides a window of `size` tokens stepping by
/// `stride = (size - overlap).max(1)`. The byte range for each window is taken
/// directly from `Encoding::get_offsets()`. Degenerate `(0, 0)` offset pairs
/// (which some tokenizers emit for special tokens) are skipped via the
/// `byte_end > byte_start` guard.
fn fixed_token(
  tokenizer: &Tokenizer,
  text: &str,
  size: usize,
  overlap: usize,
) -> Result<Vec<(Range<usize>, String)>> {
  let enc = tokenizer
    .encode(text, false)
    .map_err(|e| Error::Tokenizer(e.to_string()))?;
  let offsets = enc.get_offsets(); // byte start/end per token
  if offsets.is_empty() {
    return Ok(Vec::new());
  }
  let stride = size.saturating_sub(overlap).max(1);
  let n = offsets.len();
  let mut out = Vec::new();
  let mut start = 0usize;
  loop {
    let end = (start + size).min(n); // token window [start, end)
    let byte_start = offsets[start].0;
    let byte_end = offsets[end - 1].1;
    // Skip degenerate (0, 0) offsets that some tokenizers emit for special
    // tokens added at boundaries.
    if byte_end > byte_start {
      out.push((byte_start..byte_end, text[byte_start..byte_end].to_string()));
    }
    // Break only after advancing, so the window at the current `start` (which
    // may begin on the final token) is always emitted first. `stride >= 1`
    // guarantees termination.
    start += stride;
    if start >= n {
      break;
    }
  }
  Ok(out)
}

/// Token-budgeted semantic windows via `text-splitter`.
///
/// Uses `ChunkConfig::new(size).with_sizer(tokenizer).with_overlap(overlap)?.with_trim(false)`,
/// then `TextSplitter::new(cfg).chunk_char_indices(text)` which yields
/// `ChunkCharIndex { byte_offset, char_offset, chunk }`.  The byte span is
/// `byte_offset .. byte_offset + chunk.len()` (valid because `chunk` is a
/// subslice of `text`).
fn semantic(
  tokenizer: &Tokenizer,
  text: &str,
  size: usize,
  overlap: usize,
) -> Result<Vec<(Range<usize>, String)>> {
  use text_splitter::{ChunkConfig, TextSplitter};
  let cfg = ChunkConfig::new(size)
    .with_sizer(tokenizer)
    .with_overlap(overlap)
    .map_err(|e| Error::Tokenizer(format!("text-splitter overlap: {e}")))?
    .with_trim(false);
  let splitter = TextSplitter::new(cfg);
  let mut out = Vec::new();
  for ci in splitter.chunk_char_indices(text) {
    let byte_start = ci.byte_offset;
    let chunk = ci.chunk;
    out.push((byte_start..byte_start + chunk.len(), chunk.to_string()));
  }
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;
  use tokenizers::{
    Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
  };

  /// Build a tiny whitespace `WordLevel` tokenizer with vocab `a`..`j` + `[UNK]`.
  fn test_tokenizer() -> Tokenizer {
    let vocab = [
      ("a", 0u32),
      ("b", 1),
      ("c", 2),
      ("d", 3),
      ("e", 4),
      ("f", 5),
      ("g", 6),
      ("h", 7),
      ("i", 8),
      ("j", 9),
      ("[UNK]", 10),
    ]
    .into_iter()
    .map(|(w, id)| (w.to_string(), id))
    .collect();
    let model = WordLevel::builder()
      .vocab(vocab)
      .unk_token("[UNK]".to_string())
      .build()
      .unwrap();
    let mut tok = Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace));
    tok
  }

  #[test]
  fn fixed_token_windows_have_expected_token_counts_and_spans() {
    let tok = test_tokenizer();
    let text = "a b c d e f g h i j"; // 10 tokens
    let opts = WindowOptions::new(WindowStrategy::FixedToken)
      .with_size(4)
      .with_overlap(1);
    let w = split_windows(&tok, text, &opts, 4096, 0).unwrap();
    // stride = 4-1 = 3 → starts 0,3,6,9 → 4 windows (last is "j")
    assert_eq!(w.len(), 4);
    // first window spans tokens 0..4 = "a b c d"
    assert_eq!(&text[w[0].0.clone()], "a b c d");
    // last window is the tail token "j"
    assert_eq!(&text[w[3].0.clone()], "j");
  }

  #[test]
  fn fixed_token_single_window_when_under_size() {
    let tok = test_tokenizer();
    let text = "a b c";
    let opts = WindowOptions::new(WindowStrategy::FixedToken)
      .with_size(8)
      .with_overlap(0);
    let w = split_windows(&tok, text, &opts, 4096, 0).unwrap();
    assert_eq!(w.len(), 1);
    assert_eq!(&text[w[0].0.clone()], "a b c");
  }

  #[test]
  fn semantic_windows_cover_text_within_budget() {
    let tok = test_tokenizer();
    let text = "a b c d e f g h i j";
    let opts = WindowOptions::new(WindowStrategy::Semantic)
      .with_size(4)
      .with_overlap(0);
    let w = split_windows(&tok, text, &opts, 4096, 0).unwrap();
    assert!(!w.is_empty());
    // every chunk re-tokenizes to <= size tokens
    for (span, chunk) in &w {
      assert!(!chunk.is_empty());
      assert!(span.end <= text.len());
      let n = tok.encode(chunk.as_str(), false).unwrap().get_ids().len();
      assert!(n <= 4, "chunk '{chunk}' had {n} tokens");
    }
  }

  #[test]
  fn fixed_token_consecutive_windows_share_overlap() {
    let tok = test_tokenizer();
    let text = "a b c d e f g h i j";
    let opts = WindowOptions::new(WindowStrategy::FixedToken)
      .with_size(4)
      .with_overlap(1);
    let w = split_windows(&tok, text, &opts, 4096, 0).unwrap();
    // size 4, overlap 1, stride 3: window 1 begins on token 3 ("d") — the last
    // token of window 0 ("a b c d") — so consecutive windows share one token.
    assert_eq!(&text[w[0].0.clone()], "a b c d");
    assert_eq!(&text[w[1].0.clone()], "d e f g");
  }

  #[test]
  fn split_windows_empty_text_is_empty() {
    let tok = test_tokenizer();
    let opts = WindowOptions::new(WindowStrategy::FixedToken).with_size(4);
    let w = split_windows(&tok, "", &opts, 4096, 0).unwrap();
    assert!(w.is_empty());
  }

  #[test]
  fn fixed_token_overlap_ge_size_terminates() {
    let tok = test_tokenizer();
    let text = "a b c d e"; // 5 tokens
    // overlap >= size → stride clamps to 1; must terminate, one window per start.
    let opts = WindowOptions::new(WindowStrategy::FixedToken)
      .with_size(3)
      .with_overlap(5);
    let w = split_windows(&tok, text, &opts, 4096, 0).unwrap();
    assert_eq!(w.len(), 5);
    assert_eq!(&text[w[0].0.clone()], "a b c");
  }
}
