//! Sliding-window long-input support (the `windowing` feature). Splits text
//! longer than the model's window into overlapping byte-exact fixed-token
//! windows, each embedded through the normal per-text path; `embed_pooled`
//! combines them.

use core::ops::Range;

use tokenizers::Tokenizer;

use crate::error::{Error, Result};

/// A windowing result tuple: `(byte_span, window_ids, content_token_count)`. The
/// ids include the framing specials; the count is the real content tokens
/// (specials excluded), carried for `embed_pooled` weighting.
pub(crate) type IdWindow = (Range<usize>, Vec<u32>, usize);

/// Windowing configuration for [`crate::TextEncoder`]'s `embed_windows` /
/// `embed_pooled`.
///
/// Windows are exact token-count slices that embed the original encoding's token
/// IDs **verbatim** — the full text is encoded once (with special tokens), the
/// content tokens are tiled by stride, and each window re-attaches the encoding's
/// leading/trailing specials (e.g. BOS). No re-tokenization, so a window's byte
/// span describes exactly the tokens embedded; deterministic and byte-exact, but
/// may split mid-sentence.
#[derive(Clone, Copy, Debug)]
pub struct WindowOptions {
  /// Max real tokens per window. `None` → the encoder's `max_seq_len` minus the
  /// tokenizer's special-token overhead (so a re-embedded chunk never truncates).
  size: Option<usize>,
  /// Token overlap between consecutive windows (0 = no overlap).
  overlap: usize,
}

impl Default for WindowOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl WindowOptions {
  /// Construct with the default size (fills the encoder window) and no overlap.
  pub const fn new() -> Self {
    Self {
      size: None,
      overlap: 0,
    }
  }

  /// The configured max tokens per window, if overridden.
  pub const fn size(&self) -> Option<usize> {
    self.size
  }

  /// The token overlap between windows.
  pub const fn overlap(&self) -> usize {
    self.overlap
  }

  /// Set an explicit window size (max real tokens per window). A size of `0` is
  /// clamped to a 1-token budget rather than rejected.
  pub const fn with_size(mut self, n: usize) -> Self {
    self.size = Some(n);
    self
  }

  /// Set the token overlap between consecutive windows. An overlap `>=` the
  /// window size clamps the step to 1 token per window (maximum overlap).
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

/// Per-window REAL-token budget from the requested [`WindowOptions::size`] and
/// the encoder limits, enforcing the documented `size` contract: the budget is
/// the requested size (`None` → fill the window) clamped to a 1-token minimum,
/// then capped so a window's full id vector (`budget + reserve` tokens) never
/// exceeds `max_seq_len`.
///
/// Returns [`Error::Tokenizer`] only when the special-token `reserve` alone
/// fills `max_seq_len`, leaving no room for even one content token.
fn window_budget(opts: &WindowOptions, max_seq_len: usize, reserve: usize) -> Result<usize> {
  let ceiling = max_seq_len
    .checked_sub(reserve)
    .filter(|&b| b > 0)
    .ok_or_else(|| {
      Error::Tokenizer(format!(
        "window budget is zero: special-token reserve {reserve} fills max_seq_len {max_seq_len}"
      ))
    })?;
  // `size` is the max REAL tokens per window: `None` fills the window; an
  // explicit `0` clamps to 1 (never rejected); any size is capped at `ceiling`
  // so the re-framed window (content + specials) stays within `max_seq_len`.
  Ok(match opts.size() {
    None => ceiling,
    Some(n) => n.max(1).min(ceiling),
  })
}

/// Exact token-count windows that embed the encoding's token IDs **verbatim** —
/// byte-exact (no re-tokenization, so no silent truncation and no coverage gap).
///
/// The full `text` is encoded **once with special tokens**, so the model's
/// leading/trailing specials (e.g. BOS) are captured exactly as the per-text
/// path would produce them. The encoding's `special_tokens_mask` locates the
/// content span `[c0, c1]` (`mask == 0`); the tokens before `c0` are the leading
/// specials and those after `c1` the trailing specials, re-attached unchanged to
/// every window. Returns `Ok(Vec::new())` when the text has no content token.
///
/// The per-window real-token budget is [`WindowOptions::size`] (default fills the
/// window), sized by [`window_budget`] so each window's full id vector
/// (`leading + content slice + trailing`) is `<= max_seq_len`. Returns
/// [`Error::Tokenizer`] when the special-token reserve alone fills `max_seq_len`.
/// The content tokens tile by `stride = (budget - overlap).max(1)`, stopping as
/// soon as a window reaches the content end so the last window is not a redundant
/// suffix of the previous one. Each window's byte span comes from the ORIGINAL
/// encoding's offsets at the sliced content indices, so it describes exactly the
/// tokens embedded. Each returned tuple is `(byte_span, window_ids,
/// content_token_count)`, where `content_token_count` is the window's real
/// content-token count (the framing specials excluded) used for pooled weighting.
///
/// At most `max_windows` windows are produced; a longer input returns
/// [`Error::BatchTooLarge`] before any window ids are allocated. The full text is
/// encoded once (allocation proportional to the input — inherent to windowing by
/// tokens); the cap bounds the amplified per-window id allocation, not that pass.
pub(crate) fn fixed_token_id_windows(
  tokenizer: &Tokenizer,
  text: &str,
  opts: &WindowOptions,
  max_seq_len: usize,
  max_windows: usize,
) -> Result<Vec<IdWindow>> {
  // Encode WITH special tokens so the leading/trailing specials (BOS, etc.) are
  // captured verbatim and embedded exactly as the per-text path would.
  let enc = tokenizer
    .encode(text, true)
    .map_err(|e| Error::Tokenizer(e.to_string()))?;
  let ids = enc.get_ids();
  let offsets = enc.get_offsets();
  let mask = enc.get_special_tokens_mask(); // 1 = special, 0 = content

  // Locate the content span [c0, c1] (the first/last non-special token). With no
  // content token there is nothing to embed.
  let Some(c0) = mask.iter().position(|&m| m == 0) else {
    return Ok(Vec::new());
  };
  let c1 = mask
    .iter()
    .rposition(|&m| m == 0)
    .expect("a content token exists since c0 was found");

  let leading: Vec<u32> = ids[..c0].to_vec();
  let trailing: Vec<u32> = ids[c1 + 1..].to_vec();
  let reserve = leading.len() + trailing.len();
  let content_len = c1 - c0 + 1;

  // Real-token budget per the `WindowOptions::size` contract (default fills the
  // window, 0 clamps to 1), capped so `budget + reserve <= max_seq_len`.
  let budget = window_budget(opts, max_seq_len, reserve)?;

  let stride = budget.saturating_sub(opts.overlap()).max(1);
  // Window count, computed before any allocation: one window per stride until a
  // window's end reaches `content_len`. Reject an over-cap count up front (no
  // window ids allocated yet) so windowing cannot OOM ahead of `embed_batch`.
  let count = if content_len <= budget {
    1
  } else {
    (content_len - budget).div_ceil(stride) + 1
  };
  if count > max_windows {
    return Err(Error::BatchTooLarge {
      got: count,
      max: max_windows,
    });
  }

  let mut out = Vec::with_capacity(count);
  let mut start = 0usize;
  loop {
    let end = start.saturating_add(budget).min(content_len); // content [start, end)
    // The sliced content tokens, framed by the constant leading/trailing
    // specials. Exactly `reserve + (end - start)` ids, which is `<= max_seq_len`.
    let window_ids: Vec<u32> = leading
      .iter()
      .chain(&ids[c0 + start..c0 + end])
      .chain(&trailing)
      .copied()
      .collect();
    // Byte span from the ORIGINAL encoding's offsets at the sliced content
    // indices, so it describes exactly the tokens embedded. Skip a degenerate
    // (0, 0)-style span some tokenizers emit.
    let byte_start = offsets[c0 + start].0;
    let byte_end = offsets[c0 + end - 1].1;
    if byte_end > byte_start {
      // `end - start` is the window's real content-token count (the framing
      // specials excluded), carried out for `embed_pooled` weighting.
      out.push((byte_start..byte_end, window_ids, end - start));
    }
    // Stop once a window reaches the content end: advancing further would emit a
    // window wholly contained in this one (a redundant, over-weighted tail).
    if end >= content_len {
      break;
    }
    start += stride;
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

  // The `test_tokenizer` is a WordLevel tokenizer with NO post-processor, so
  // `encode(text, true) == encode(text, false)`: the special-tokens mask is all
  // zeros, `reserve == 0`, and each window's id vector is exactly its content
  // tokens. These tests therefore exercise pure content windowing — and assert
  // each window's `Vec<u32>` token count alongside its byte span, the property
  // the embed path relies on (it embeds these ids verbatim).

  #[test]
  fn fixed_token_windows_have_expected_token_counts_and_spans() {
    let tok = test_tokenizer();
    let text = "a b c d e f g h i j"; // 10 tokens
    let opts = WindowOptions::new().with_size(4).with_overlap(1);
    let w = fixed_token_id_windows(&tok, text, &opts, 4096, 1024).unwrap();
    // stride 3 → starts 0,3,6; window [6,10) reaches the end, so we stop with no
    // redundant [9,10) tail → 3 windows.
    assert_eq!(w.len(), 3);
    assert_eq!(&text[w[0].0.clone()], "a b c d");
    assert_eq!(&text[w[2].0.clone()], "g h i j");
    // 4 content tokens per window, no specials → 4 ids each.
    assert_eq!(w[0].1, vec![0, 1, 2, 3]); // a b c d
    assert_eq!(w[2].1, vec![6, 7, 8, 9]); // g h i j
  }

  #[test]
  fn fixed_token_single_window_when_under_size() {
    let tok = test_tokenizer();
    let text = "a b c";
    let opts = WindowOptions::new().with_size(8).with_overlap(0);
    let w = fixed_token_id_windows(&tok, text, &opts, 4096, 1024).unwrap();
    assert_eq!(w.len(), 1);
    assert_eq!(&text[w[0].0.clone()], "a b c");
    assert_eq!(w[0].1, vec![0, 1, 2]);
  }

  #[test]
  fn fixed_token_consecutive_windows_share_overlap() {
    let tok = test_tokenizer();
    let text = "a b c d e f g h i j";
    let opts = WindowOptions::new().with_size(4).with_overlap(1);
    let w = fixed_token_id_windows(&tok, text, &opts, 4096, 1024).unwrap();
    // size 4, overlap 1, stride 3: window 1 begins on token 3 ("d") — the last
    // token of window 0 ("a b c d") — so consecutive windows share one token.
    assert_eq!(&text[w[0].0.clone()], "a b c d");
    assert_eq!(&text[w[1].0.clone()], "d e f g");
    // The shared "d" (id 3) is the last id of window 0 and the first of window 1.
    assert_eq!(w[0].1, vec![0, 1, 2, 3]);
    assert_eq!(w[1].1, vec![3, 4, 5, 6]);
  }

  #[test]
  fn fixed_token_empty_text_is_empty() {
    let tok = test_tokenizer();
    let opts = WindowOptions::new().with_size(4);
    let w = fixed_token_id_windows(&tok, "", &opts, 4096, 1024).unwrap();
    assert!(w.is_empty());
  }

  #[test]
  fn fixed_token_overlap_ge_size_terminates() {
    let tok = test_tokenizer();
    let text = "a b c d e"; // 5 tokens
    // overlap >= size → stride clamps to 1; windows [0,3),[1,4),[2,5); [2,5)
    // reaches the end → 3 windows, terminates.
    let opts = WindowOptions::new().with_size(3).with_overlap(5);
    let w = fixed_token_id_windows(&tok, text, &opts, 4096, 1024).unwrap();
    assert_eq!(w.len(), 3);
    assert_eq!(&text[w[0].0.clone()], "a b c");
    assert_eq!(w[0].1, vec![0, 1, 2]);
  }

  #[test]
  fn size_is_clamped_to_max_seq_len() {
    let tok = test_tokenizer();
    let text = "a b c d e"; // 5 tokens
    // Requested size far exceeds max_seq_len (4) → clamped to 4, so every window's
    // id vector holds at most the cap, with no silent truncation.
    let opts = WindowOptions::new().with_size(10_000);
    let w = fixed_token_id_windows(&tok, text, &opts, 4, 1024).unwrap();
    for (_, ids, _) in &w {
      assert!(ids.len() <= 4, "window has {} ids (cap 4)", ids.len());
    }
  }

  #[test]
  fn zero_budget_is_rejected() {
    let tok = test_tokenizer();
    let opts = WindowOptions::new();
    // max_seq_len 0 → cap 0 → zero real-token budget → Error::Tokenizer. (This
    // tokenizer adds no specials, so the reserve is 0; the cap itself is the
    // zero. The reserve-fills-the-cap branch shares this error path.)
    let err = fixed_token_id_windows(&tok, "a b", &opts, 0, 1024).unwrap_err();
    assert!(matches!(err, Error::Tokenizer(_)), "got {err:?}");
  }

  #[test]
  fn too_many_windows_is_batch_too_large() {
    let tok = test_tokenizer();
    let text = "a b c d e f g h i j"; // 10 tokens
    // size 2, no overlap → 5 windows, but max_windows = 2 → rejected before alloc.
    let opts = WindowOptions::new().with_size(2);
    let err = fixed_token_id_windows(&tok, text, &opts, 4096, 2).unwrap_err();
    assert!(
      matches!(err, Error::BatchTooLarge { max: 2, .. }),
      "got {err:?}"
    );
  }

  #[test]
  fn huge_size_does_not_overflow() {
    let tok = test_tokenizer();
    // with_size(usize::MAX) is clamped to max_seq_len; no `start + budget` overflow.
    let opts = WindowOptions::new().with_size(usize::MAX);
    let w = fixed_token_id_windows(&tok, "a b c", &opts, 4096, 1024).unwrap();
    assert_eq!(w.len(), 1);
    assert_eq!(w[0].1, vec![0, 1, 2]);
  }

  #[test]
  fn fixed_token_windows_fit_budget() {
    let tok = test_tokenizer();
    let text = "a b c d e f g h i j";
    let opts = WindowOptions::new().with_size(3).with_overlap(1);
    let w = fixed_token_id_windows(&tok, text, &opts, 4096, 1024).unwrap();
    // Each window's id vector holds at most the budget — byte-exact by
    // construction (the content ids are sliced from the one encoding, never
    // re-tokenized), so the embed path cannot silently truncate it.
    for (_, ids, _) in &w {
      assert!(ids.len() <= 3, "window has {} ids (budget 3)", ids.len());
    }
  }

  /// A tokenizer that adds REAL specials (`[CLS] … [SEP]` via `BertProcessing`,
  /// so `reserve == 2` per single sequence). The plain `test_tokenizer` has no
  /// post-processor (all-zero special mask, `reserve == 0`), so it cannot reach
  /// the `size`-vs-reserve interaction. These tests use this tokenizer to pin that
  /// `size` counts REAL content tokens, independent of the special-token reserve.
  fn test_tokenizer_with_specials() -> Tokenizer {
    use tokenizers::processors::bert::BertProcessing;
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
      ("[CLS]", 11),
      ("[SEP]", 12),
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
    tok.with_post_processor(Some(BertProcessing::new(
      ("[SEP]".to_string(), 12),
      ("[CLS]".to_string(), 11),
    )));
    tok
  }

  #[test]
  fn fixed_token_size_is_real_content_tokens_not_total_ids() {
    let tok = test_tokenizer_with_specials();
    let text = "a b c d e f g h i j"; // 10 content tokens
    // size = 4 REAL tokens, reserve = 2 (CLS/SEP). Each window must hold 4
    // CONTENT tokens framed by the 2 specials → 6 ids total. A `budget = size -
    // reserve` semantics would have given only 2 content tokens per window.
    let opts = WindowOptions::new().with_size(4).with_overlap(0);
    let w = fixed_token_id_windows(&tok, text, &opts, 4096, 1024).unwrap();
    // [CLS] a b c d [SEP] — 4 content tokens, byte span covers exactly them.
    assert_eq!(w[0].1, vec![11, 0, 1, 2, 3, 12]);
    assert_eq!(&text[w[0].0.clone()], "a b c d");
    // The carried content-token count (used for pooled weighting) excludes the
    // framing specials: the full window holds 6 ids but counts as 4 content
    // tokens, and the short final window (i j) counts as 2 — the property that
    // keeps `embed_pooled` from over-weighting short tails by their specials.
    assert_eq!(w[0].2, 4, "first window: 4 content tokens, not 6 ids");
    assert_eq!(
      w.last().unwrap().2,
      2,
      "final window: 2 content tokens (i j)"
    );
    for (_, ids, n) in &w {
      // 4 content + 2 specials; the last window may carry fewer content tokens.
      assert!(ids.len() <= 6, "window has {} ids (cap 6)", ids.len());
      // The content count never includes the 2 framing specials.
      assert_eq!(
        *n,
        ids.len() - 2,
        "content count must exclude the 2 specials"
      );
    }
  }

  #[test]
  fn fixed_token_size_one_with_specials_is_one_content_token() {
    let tok = test_tokenizer_with_specials();
    let text = "a b c"; // 3 content tokens
    // size = 1 REAL token, reserve = 2. A `size - reserve` semantics would have
    // underflowed to a spurious zero-budget error. Here: one content token per
    // window, framed by the specials.
    let opts = WindowOptions::new().with_size(1);
    let w = fixed_token_id_windows(&tok, text, &opts, 4096, 1024).unwrap();
    assert_eq!(w.len(), 3, "stride 1 → one window per content token");
    assert_eq!(w[0].1, vec![11, 0, 12]); // [CLS] a [SEP]
    assert_eq!(w[1].1, vec![11, 1, 12]); // [CLS] b [SEP]
    assert_eq!(w[2].1, vec![11, 2, 12]); // [CLS] c [SEP]
  }

  #[test]
  fn fixed_token_size_zero_with_specials_clamps_to_one_content_token() {
    let tok = test_tokenizer_with_specials();
    let text = "a b";
    // `with_size(0)` is documented to clamp to a 1-token budget rather than
    // error — even though reserve = 2 here.
    let opts = WindowOptions::new().with_size(0);
    let w = fixed_token_id_windows(&tok, text, &opts, 4096, 1024).unwrap();
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].1, vec![11, 0, 12]);
    assert_eq!(w[1].1, vec![11, 1, 12]);
  }

  #[test]
  fn fixed_token_reserve_filling_max_seq_len_is_rejected() {
    let tok = test_tokenizer_with_specials();
    // max_seq_len = 2 with reserve = 2 → no room for even one content token, so
    // the budget is genuinely zero and rejected (the only error condition now).
    let opts = WindowOptions::new().with_size(4);
    let err = fixed_token_id_windows(&tok, "a b c", &opts, 2, 1024).unwrap_err();
    assert!(matches!(err, Error::Tokenizer(_)), "got {err:?}");
  }
}
