//! Text encoder for `embedding-gemma`.

use std::path::Path;

use tokenizers::{
  PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
  TruncationParams, TruncationStrategy,
};

use crate::{
  embedding::Embedding,
  error::{Error, Result},
  options::Options,
};

const EMBED_DIM: usize = Embedding::EMBED_DIM;
const PAD_TOKEN: &str = "<pad>";

/// `embedding-gemma` text-tower inference. Owns one `ort::Session` and one
/// `tokenizers::Tokenizer`.
///
/// `TextEncoder: Send + !Sync` — `ort::Session` is `!Sync`. Workers wanting
/// parallelism instantiate one `TextEncoder` per thread, or share one behind
/// a `Mutex<TextEncoder>`.
pub struct TextEncoder {
  session: ort::session::Session,
  tokenizer: Tokenizer,
  opts: Options,
}

impl TextEncoder {
  /// **Not available on wasm32.** `ort 2.0.0-rc.12` cfg-gates
  /// `commit_from_file` out of wasm32 builds. On wasm callers must
  /// construct the `ort::session::Session` via the wasm-specific async
  /// APIs and pass it to [`Self::from_ort_session`].
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_files(graph: &Path, tokenizer: &Path) -> Result<Self> {
    Self::from_files_with_options(graph, tokenizer, Options::default())
  }

  /// Same wasm32 caveat as [`Self::from_files`].
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_files_with_options(graph: &Path, tokenizer: &Path, opts: Options) -> Result<Self> {
    let session = crate::session::build_session(graph, opts)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| Error::Tokenizer(e.to_string()))?;
    // `configure_tokenizer` runs inside `from_ort_session_with_options`,
    // so we don't apply it here.
    Self::from_ort_session_with_options(session, tokenizer, opts)
  }

  /// Construct from a caller-built `ort::Session` and `Tokenizer`,
  /// using the crate-default [`Options`]. Equivalent to calling
  /// [`Self::from_ort_session_with_options`] with `Options::default()`.
  /// On wasm32 this is the supported entry point because
  /// `ort 2.0.0-rc.12` cfg-gates `commit_from_file` out of wasm
  /// builds — wasm callers must build the `ort::Session` themselves
  /// (e.g. via the wasm-specific async APIs) and pass it in.
  pub fn from_ort_session(session: ort::session::Session, tokenizer: Tokenizer) -> Result<Self> {
    Self::from_ort_session_with_options(session, tokenizer, Options::default())
  }

  /// Construct from a caller-built `ort::Session` and `Tokenizer` with
  /// custom [`Options`]. Public so wasm32 callers (who can't use
  /// [`Self::from_files_with_options`] because `ort 2.0.0-rc.12`
  /// cfg-gates `commit_from_file` out of wasm builds) can still tune
  /// `max_seq_len`, `batch_size`, and `max_batch_size`.
  pub fn from_ort_session_with_options(
    session: ort::session::Session,
    tokenizer: Tokenizer,
    opts: Options,
  ) -> Result<Self> {
    validate_text_session(&session)?;
    opts.batch().validate()?;
    let tokenizer = configure_tokenizer(tokenizer, opts.batch().max_seq_len())?;
    Ok(Self {
      session,
      tokenizer,
      opts,
    })
  }

  /// Encode a single string and return its 768-dim L2-normalized
  /// [`Embedding`]. Empty input is rejected with [`Error::EmptyText`].
  /// For multiple inputs, prefer [`Self::embed_batch`] — it amortizes
  /// the per-call ORT overhead across the batch.
  pub fn embed(&mut self, text: &str) -> Result<Embedding> {
    if text.is_empty() {
      return Err(Error::EmptyText);
    }
    let mut out = self.embed_batch(&[text])?;
    Ok(out.remove(0))
  }

  /// Returns `Ok(vec![])` for an empty input slice (no ORT call).
  /// Returns `Error::BatchTooLarge` when `texts.len() > opts.batch.max_batch_size`.
  /// Internally chunks `texts` into groups of size `BatchOptions::batch_size`
  /// and runs one ORT inference per chunk; the returned `Vec` preserves
  /// input order and has the same length as `texts` on success.
  ///
  /// **Failure semantics.** Aborts on the first failing chunk and returns
  /// `Error::Batch { index, source }`. The wrapped `index` granularity
  /// depends on where the failure originated:
  ///
  /// - **Row-precise** (`index = base + offending_row`) for failures
  ///   that pin to a specific input: empty-text guard, per-row
  ///   tokenizer-output length mismatch, and per-row embedding
  ///   normalization failures (`Error::NotNormalized` from
  ///   `from_model_output`).
  /// - **Chunk-level** (`index = base`, the chunk's first input
  ///   position) for failures that don't pin to a single row:
  ///   `tokenizer.encode_batch` failures, ORT tensor-build / `run` /
  ///   output-extract errors, output-rank or output-shape mismatches.
  ///   Inspect `source` to disambiguate.
  ///
  /// Already-computed embeddings from earlier chunks are dropped.
  pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }
    let max = self.opts.batch().max_batch_size();
    if texts.len() > max {
      return Err(Error::BatchTooLarge {
        got: texts.len(),
        max,
      });
    }
    if let Some((index, _)) = texts.iter().enumerate().find(|(_, t)| t.is_empty()) {
      return Err(Error::Batch {
        index,
        source: Box::new(Error::EmptyText),
      });
    }
    let chunk = self.opts.batch().batch_size();
    let mut out = Vec::with_capacity(texts.len());
    for (chunk_idx, group) in texts.chunks(chunk).enumerate() {
      let base_index = chunk_idx * chunk;
      let chunk_emb = embed_chunk(&mut self.session, &self.tokenizer, group, base_index)?;
      out.extend(chunk_emb);
    }
    Ok(out)
  }

  /// Run a single throwaway inference to amortize first-call ORT
  /// graph compilation. Useful when latency-sensitive code wants to
  /// pay the warm-up cost up-front rather than on the first user
  /// request.
  pub fn warmup(&mut self) -> Result<()> {
    let _ = self.embed("warmup")?;
    Ok(())
  }
}

fn embed_chunk(
  session: &mut ort::session::Session,
  tokenizer: &Tokenizer,
  group: &[&str],
  base_index: usize,
) -> Result<Vec<Embedding>> {
  let encodings = tokenizer
    .encode_batch(group.to_vec(), true)
    .map_err(|e| Error::Batch {
      index: base_index,
      source: Box::new(Error::Tokenizer(e.to_string())),
    })?;

  let batch = group.len();
  // BatchLongest pads every encoding in the chunk to the same length.
  let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);
  if seq_len == 0 {
    return Err(Error::Batch {
      index: base_index,
      source: Box::new(Error::EmptyText),
    });
  }

  let mut input_ids = Vec::with_capacity(batch * seq_len);
  let mut attention_mask = Vec::with_capacity(batch * seq_len);
  for (i, enc) in encodings.iter().enumerate() {
    let ids = enc.get_ids();
    let mask = enc.get_attention_mask();
    if ids.len() != seq_len || mask.len() != seq_len {
      return Err(Error::Batch {
        index: base_index + i,
        source: Box::new(Error::Tokenizer(format!(
          "tokenizer produced uneven row {} (ids={}, mask={}, expected {})",
          i,
          ids.len(),
          mask.len(),
          seq_len
        ))),
      });
    }
    input_ids.extend(ids.iter().map(|&u| u as i64));
    attention_mask.extend(mask.iter().map(|&u| u as i64));
  }

  run_session(
    session,
    &input_ids,
    &attention_mask,
    batch,
    seq_len,
    base_index,
  )
}

fn run_session(
  session: &mut ort::session::Session,
  input_ids: &[i64],
  attention_mask: &[i64],
  batch: usize,
  seq_len: usize,
  base_index: usize,
) -> Result<Vec<Embedding>> {
  use ort::value::TensorRef;

  // Wrap chunk-level errors (tensor build, ORT run, output extraction,
  // shape validation) with `Error::Batch { index: base_index }` so the
  // caller can identify which chunk failed even when the failure
  // doesn't pin to a specific row. Per-row normalization failures get
  // a precise `base_index + i` further down. This mirrors siglip2's
  // text_enc batch-failure semantics — a documented contract that
  // `embed_batch` reports failures via `Error::Batch`.
  let wrap_chunk = |source: Error| Error::Batch {
    index: base_index,
    source: Box::new(source),
  };

  let shape: [usize; 2] = [batch, seq_len];
  let ids_val =
    TensorRef::from_array_view((shape, input_ids)).map_err(|e| wrap_chunk(Error::Ort(e)))?;
  let mask_val =
    TensorRef::from_array_view((shape, attention_mask)).map_err(|e| wrap_chunk(Error::Ort(e)))?;

  let outputs = session
    .run(ort::inputs![
      "input_ids" => ids_val,
      "attention_mask" => mask_val,
    ])
    .map_err(|e| wrap_chunk(Error::Ort(e)))?;

  let out = outputs.get("sentence_embedding").ok_or_else(|| {
    wrap_chunk(Error::MissingOnnxOutput {
      name: "sentence_embedding",
    })
  })?;
  let (shape, data) = out
    .try_extract_tensor::<f32>()
    .map_err(|e| wrap_chunk(Error::Ort(e)))?;

  if shape.len() != 2 {
    return Err(wrap_chunk(Error::OutputRank {
      rank: shape.len(),
      shape: shape.to_vec(),
    }));
  }
  if shape[0] != batch as i64 || shape[1] != EMBED_DIM as i64 {
    return Err(wrap_chunk(Error::SessionShapeMismatch {
      input: "sentence_embedding",
      expected: "[batch, 768]",
      got: shape.to_vec(),
    }));
  }

  embeddings_from_chunk(data, batch, base_index)
}

/// Convert a flat `[batch * EMBED_DIM]` model-output buffer into
/// `batch` `Embedding`s, wrapping per-row normalization failures as
/// `Error::Batch { index: base_index + i, source }` so callers can
/// quarantine the offending row. Pulled out of `run_session` so the
/// indexed wrapping is unit-testable without an ORT session.
fn embeddings_from_chunk(data: &[f32], batch: usize, base_index: usize) -> Result<Vec<Embedding>> {
  debug_assert_eq!(data.len(), batch * EMBED_DIM);
  let mut embeddings = Vec::with_capacity(batch);
  for i in 0..batch {
    let row = &data[i * EMBED_DIM..(i + 1) * EMBED_DIM];
    let emb = Embedding::from_model_output(row).map_err(|source| Error::Batch {
      index: base_index + i,
      source: Box::new(source),
    })?;
    embeddings.push(emb);
  }
  Ok(embeddings)
}

fn validate_text_session(session: &ort::session::Session) -> Result<()> {
  use ort::value::TensorElementType;

  let inputs = session.inputs();
  let outputs = session.outputs();

  // Both inputs are `[batch, seq]` with dynamic batch and dynamic seq.
  check_outlet(inputs, "input_ids", TensorElementType::Int64, &[-1, -1])?;
  check_outlet(
    inputs,
    "attention_mask",
    TensorElementType::Int64,
    &[-1, -1],
  )?;
  // Output is `[batch, EMBED_DIM]` with dynamic batch.
  check_outlet(
    outputs,
    "sentence_embedding",
    TensorElementType::Float32,
    &[-1, EMBED_DIM as i64],
  )?;
  Ok(())
}

/// Verify an `Outlet` exists with the expected dtype and shape.
///
/// `expected_shape` semantics:
///
/// - `-1` means **the graph MUST declare this axis dynamic**. A static
///   dim there is rejected. `embed_chunk` sends batches of
///   `[group.len(), BatchLongest seq_len]` where neither dim is known
///   at session-build time, so a graph baking in `[1, 2048]` or
///   `[8, 512]` would fail at first `Session::run`.
/// - any other value is an **exact match** requirement. The graph may
///   either match exactly or declare the axis dynamic (`-1`); both
///   work at runtime.
fn check_outlet(
  outlets: &[ort::value::Outlet],
  name: &'static str,
  expected_dtype: ort::value::TensorElementType,
  expected_shape: &[i64],
) -> Result<()> {
  use ort::value::ValueType;

  let outlet = outlets
    .iter()
    .find(|o| o.name() == name)
    .ok_or(Error::SessionShapeMismatch {
      input: name,
      expected: "outlet present in session",
      got: vec![],
    })?;

  match outlet.dtype() {
    ValueType::Tensor { ty, shape, .. } => {
      if *ty != expected_dtype {
        // Use `SessionContractMismatch` so the actual dtype shows up
        // in the message — `SessionShapeMismatch.got: Vec<i64>` would
        // either be the shape (irrelevant for a dtype error) or empty.
        return Err(Error::SessionContractMismatch {
          input: name,
          expected: "matching tensor dtype",
          got: *ty,
        });
      }
      let actual: &[i64] = shape;
      if actual.len() != expected_shape.len() {
        return Err(Error::SessionShapeMismatch {
          input: name,
          expected: "matching tensor rank",
          got: actual.to_vec(),
        });
      }
      for (i, &want) in expected_shape.iter().enumerate() {
        let act = actual[i];
        if want == -1 {
          // We require this axis to be dynamic. A graph baking in
          // a concrete dim here would load successfully under the
          // old wildcard semantics and only fail at `Session::run`
          // when `embed_chunk` sends a different size.
          if act != -1 {
            return Err(Error::SessionShapeMismatch {
              input: name,
              expected: "dynamic axis (graph declares -1; static-shape \
                         exports incompatible with chunked APIs)",
              got: actual.to_vec(),
            });
          }
        } else {
          // Concrete dim required. Graph may match exactly or declare
          // the axis dynamic — both work at runtime.
          if act != -1 && act != want {
            return Err(Error::SessionShapeMismatch {
              input: name,
              expected: "matching static dim",
              got: actual.to_vec(),
            });
          }
        }
      }
      Ok(())
    }
    _ => Err(Error::SessionShapeMismatch {
      input: name,
      expected: "tensor",
      got: vec![],
    }),
  }
}

fn configure_tokenizer(mut tokenizer: Tokenizer, max_seq_len: usize) -> Result<Tokenizer> {
  let pad_id = tokenizer
    .token_to_id(PAD_TOKEN)
    .ok_or_else(|| Error::Tokenizer(format!("loaded tokenizer has no `{PAD_TOKEN}` token")))?;

  // Pad to the longest input in each batch. The model's `attention_mask`
  // input lets us mask out the padding tokens cleanly, so we don't need
  // a fixed sequence length — every chunk pads to its own longest row.
  tokenizer.with_padding(Some(PaddingParams {
    strategy: PaddingStrategy::BatchLongest,
    direction: PaddingDirection::Right,
    pad_id,
    pad_token: PAD_TOKEN.to_string(),
    pad_type_id: 0,
    pad_to_multiple_of: None,
  }));

  // Truncate long inputs to `max_seq_len`. `with_truncation` returns
  // `Result<&mut Self>` and only fails when `stride > effective_max_length`;
  // with `stride = 0` and `max_length > 0` this is infallible.
  if max_seq_len == 0 {
    return Err(Error::Tokenizer(
      "max_seq_len must be > 0 (BatchOptions::with_max_seq_len)".to_string(),
    ));
  }
  tokenizer
    .with_truncation(Some(TruncationParams {
      direction: TruncationDirection::Right,
      max_length: max_seq_len,
      strategy: TruncationStrategy::LongestFirst,
      stride: 0,
    }))
    .map_err(|e| Error::Tokenizer(e.to_string()))?;
  Ok(tokenizer)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn pad_token_constant_matches_gemma_vocab() {
    // The `embedding-gemma` tokenizer.json has `<pad>` at id 0; this
    // pin keeps the constant in sync with the assumption used by
    // `from_files_with_options`.
    assert_eq!(PAD_TOKEN, "<pad>");
  }

  #[test]
  fn embed_dim_constant_matches_embedding_module() {
    assert_eq!(EMBED_DIM, 768);
  }

  /// `embed_batch` documents that failures surface as
  /// `Error::Batch { index, source }` carrying the offending zero-based
  /// index. Pin: a degenerate row (here, all-zero → `NotNormalized`)
  /// in the middle of a batch must have its position preserved across
  /// the `embeddings_from_chunk` boundary instead of bubbling up as a
  /// bare `NotNormalized`.
  #[test]
  fn embeddings_from_chunk_wraps_row_error_with_index() {
    // 3 rows × 768. Rows 0 and 2 are unit vectors (normalize fine);
    // row 1 is all-zero, which `from_model_output` rejects as
    // `Error::NotNormalized` — the row we want to surface.
    let mut data = vec![0.0f32; 3 * EMBED_DIM];
    data[0] = 1.0;
    data[2 * EMBED_DIM] = 1.0;

    let err = embeddings_from_chunk(&data, 3, 100).expect_err("row 1 must fail");
    match err {
      Error::Batch { index, source } => {
        assert_eq!(index, 101, "expected base_index + 1, got {index}");
        match *source {
          Error::NotNormalized { norm, .. } => assert_eq!(norm, 0.0),
          other => panic!("expected NotNormalized inside Batch, got {other}"),
        }
      }
      other => panic!("expected Error::Batch, got {other}"),
    }
  }

  /// Sibling check: when every row is well-formed,
  /// `embeddings_from_chunk` returns the full batch with no wrapping.
  #[test]
  fn embeddings_from_chunk_succeeds_for_clean_batch() {
    let mut data = vec![0.0f32; 2 * EMBED_DIM];
    data[0] = 1.0;
    data[EMBED_DIM] = 1.0;
    let out = embeddings_from_chunk(&data, 2, 0).expect("clean batch must succeed");
    assert_eq!(out.len(), 2);
    for e in &out {
      assert_eq!(e.dim(), EMBED_DIM);
      let cos = e.try_cosine(e).expect("happy path");
      assert!((cos - 1.0).abs() < 1e-5);
    }
  }
}
