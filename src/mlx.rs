//! MLX (mlxrs) inference backend — Apple-Silicon only.
//!
//! This is the macOS/arm64 alternative to the default `ort` (ONNX Runtime)
//! inference path. It is compiled unconditionally on `aarch64-apple-darwin`
//! (and nowhere else), because `mlxrs` binds the Metal-backed MLX C++ runtime
//! through `mlx-c` FFI and has no other target. There is no `mlx` Cargo feature
//! — the backend is selected automatically by platform (see Cargo.toml).
//!
//! # Design
//!
//! [`crate::TextEncoder`] holds an internal backend enum (`Backend::Ort` vs
//! `Backend::Mlx`). The ONNX path is untouched; the MLX path is reached through
//! the platform auto-routing in [`crate::TextEncoder::from_dir`] (which probes
//! the checkpoint directory and picks MLX when an MLX checkpoint is present),
//! never a user-facing backend knob. Both backends expose the **same** public
//! API and return the same [`crate::Embedding`] (768-dim, L2-normalized).
//!
//! # Weight source
//!
//! The MLX backend consumes an **MLX-format checkpoint** (a `config.json` + a
//! weight file + a `tokenizer.json`), not the ONNX graph. The canonical
//! checkpoint is `google/embeddinggemma-300m` re-exported to MLX weights (e.g.
//! an `mlx-community` mirror). The loader reads `config.json`, then delegates
//! weight discovery to [`mlxrs::io::load_weights_from_dir`], which auto-detects
//! a sharded `model.safetensors.index.json`, a single `model.safetensors`, a
//! `*.gguf`, or a `*.npz` — runs the model's `sanitize` key-remap, and builds
//! the Gemma3 sentence-encoder via `mlxrs`. A quantized export (packed
//! `.weight` / `.scales` / `.biases` triples + a `quantization` config block)
//! loads through the same path: `mlxrs`'s `EmbeddingGemmaModel::from_weights`
//! auto-detects each layer's quantization by the presence of its `.scales`
//! sibling.
//!
//! The explicit-format constructors ([`MlxModel::from_safetensors`], and — under
//! the matching feature — `from_npz` / `from_gguf`) take a **weight file path**
//! directly and read the sibling `config.json` (+ optional `1_Pooling`) from the
//! file's parent directory, for callers who already know the format and location.
//!
//! The model dimensions + quantization scheme are always read from
//! `config.json`; the gguf path is a **weight load seam only** — its embedded
//! metadata is NOT mapped to a config, so a gguf checkpoint still requires a
//! `config.json` alongside it.
//!
//! # Text path
//!
//! EmbeddingGemma is a mean-pooling sentence-encoder: the MLX path tokenizes
//! with special tokens, **right-pads** each row to the batch maximum with the
//! Gemma `<pad>` id, builds the matching `0/1` attention mask, runs the
//! bidirectional backbone → mean-pool → Dense projection → L2-normalize via
//! `encode_text`, and reads the pooled embedding row. This mirrors `mlxrs`'s
//! own `TextEmbedder` contract for EmbeddingGemma
//! ([`Padding::DynamicRightPad`]) — no fixed sequence length, no per-text
//! truncation cap (the consuming application bounds oversized prompts; the
//! library faithfully tokenizes whatever it is handed).

use std::{collections::HashMap, path::Path, rc::Rc};

use mlxrs::embeddings::{
  config::{StPoolingConfig, pooling_from_st_config_path},
  embeddinggemma::{EmbeddingGemmaModel, config::Gemma3Config, sanitize},
};
use tokenizers::Tokenizer;

use crate::{
  embedding::Embedding,
  error::{Error, Result},
};

/// The standard config file name inside an MLX checkpoint directory.
const CONFIG_FILE: &str = "config.json";

/// The per-layer quantization marker `mlxrs` (and mlx-embeddings) use: a
/// quantized `nn.Linear` / `nn.Embedding` stores its packed weight alongside a
/// sibling `<prefix>.scales` tensor. Its presence ANYWHERE in the loaded weight
/// map is the signal that the checkpoint is (at least partly) quantized — the
/// same convention `mlxrs`'s `EmbeddingGemmaModel::from_weights` resolves per
/// layer when it auto-detects each layer's quantization.
const QUANT_SCALES_SUFFIX: &str = ".scales";

/// Discriminate a dense from a quantized MLX checkpoint by the `mlxrs`
/// convention: a quantized checkpoint carries at least one `<layer>.scales`
/// sibling tensor (see [`QUANT_SCALES_SUFFIX`]); a dense one carries none.
///
/// `mlxrs`'s `EmbeddingGemmaModel::from_weights` is a **unified** load entry
/// point — it auto-detects each layer's quantization from the same `.scales`
/// signal and resolves the scheme from the config's `quantization` block — so
/// the wrapper does not branch the load on this. The discriminator is retained
/// for parity with the sibling SigLIP2 backend and as a cheap, model-free
/// classification the unit tests pin; it inspects only the map's KEYS.
fn weights_are_quantized(weights: &HashMap<String, mlxrs::Array>) -> bool {
  weights.keys().any(|k| k.ends_with(QUANT_SCALES_SUFFIX))
}

/// The directory the sibling `config.json` (+ optional `1_Pooling`) is read from
/// when an explicit-format constructor is handed a **weight file path**: the
/// file's parent directory, or the current directory (`.`) when `weights` is a
/// bare filename with no parent component (so `from_safetensors("model.safetensors")`
/// reads `./config.json`, not a config at the filesystem root).
pub(crate) fn weights_parent(weights: &Path) -> &Path {
  weights
    .parent()
    .filter(|p| !p.as_os_str().is_empty())
    .unwrap_or_else(|| Path::new("."))
}

/// A loaded MLX EmbeddingGemma sentence-encoder. Shared (`Rc`) so it can be
/// cloned cheaply if reused.
///
/// `mlxrs`'s `EmbeddingGemmaModel::encode_text` takes `&self`, so this is
/// immutable after construction. `Rc` (not `Arc`) because
/// `EmbeddingGemmaModel` is `!Send + !Sync` — it holds MLX device-array handles
/// that are not safe to move or share across threads — so an MLX-backed encoder
/// is single-threaded (`!Send`) by construction, unlike the `ort`-backed `Send`
/// encoder.
#[derive(Clone)]
pub(crate) struct MlxModel {
  model: Rc<EmbeddingGemmaModel>,
  /// The Gemma `<pad>` token id the right-padding writes into pad cells. Read
  /// from the model's text-encoding contract at construction, so the wrapper
  /// does not hard-code a pad id ungrounded in `mlxrs`'s contract.
  pad_token_id: u32,
  /// Hard cap on a single batch's length, mirroring the ORT path's
  /// [`crate::BatchOptions::max_batch_size`]. The `from_mlx_dir` constructor
  /// takes no [`crate::Options`], so this is the crate-default cap
  /// ([`BatchOptions::default`](crate::BatchOptions)'s `1024`); the text batch
  /// path rejects oversized batches with [`Error::BatchTooLarge`] BEFORE
  /// allocating, exactly as the ORT path does.
  max_batch_size: usize,
  /// Micro-batch chunk size for the text path, mirroring the ORT path's
  /// [`crate::BatchOptions::batch_size`]. `embed_text_batch` splits a request
  /// into chunks of this many rows and runs one `encode_text` forward per
  /// chunk (each dynamically right-padded to its own max length), so a within-
  /// cap batch never materializes as a single oversized MLX/Metal graph. The
  /// constructor takes no [`crate::Options`], so this is the crate-default.
  batch_size: usize,
}

impl MlxModel {
  /// Load a model from an MLX checkpoint **directory** containing `config.json`,
  /// a weight set, and optionally a `1_Pooling/config.json` (the matryoshka
  /// output dimension + mean strategy).
  ///
  /// Weight discovery is delegated to [`mlxrs::io::load_weights_from_dir`], which
  /// auto-detects a sharded `model.safetensors.index.json`, a single
  /// `model.safetensors`, a `*.gguf`, or a `*.npz` via the centralized `mlxrs`
  /// loader. For an exact known weight file path use [`Self::from_safetensors`]
  /// (or the feature-gated `from_npz` / `from_gguf`).
  pub(crate) fn from_dir(dir: &Path) -> Result<Self> {
    Self::construct(dir, || {
      mlxrs::io::load_weights_from_dir(dir).map_err(Error::from_mlx)
    })
  }

  /// Load a model from an **exact** `model.safetensors` file path. The
  /// `config.json` (and optional `1_Pooling/config.json`) are read from the
  /// weight file's parent directory (see [`weights_parent`]).
  pub(crate) fn from_safetensors(weights: &Path) -> Result<Self> {
    Self::construct(weights_parent(weights), || {
      mlxrs::io::load_safetensors(weights).map_err(Error::from_mlx)
    })
  }

  /// Load a model from an **exact** `*.npz` file path. The `config.json` (and
  /// optional `1_Pooling/config.json`) are read from the weight file's parent
  /// directory (see [`weights_parent`]).
  #[cfg(feature = "npz")]
  pub(crate) fn from_npz(weights: &Path) -> Result<Self> {
    Self::construct(weights_parent(weights), || {
      mlxrs::io::load_npz(weights).map_err(Error::from_mlx)
    })
  }

  /// Load a model from an **exact** `*.gguf` file path. The `config.json` (and
  /// optional `1_Pooling/config.json`) are read from the weight file's parent
  /// directory (see [`weights_parent`]); the gguf's embedded metadata is NOT
  /// mapped to a config, so a sibling `config.json` is still required.
  #[cfg(feature = "gguf")]
  pub(crate) fn from_gguf(weights: &Path) -> Result<Self> {
    Self::construct(weights_parent(weights), || {
      mlxrs::io::load_gguf(weights)
        .map(|(w, _meta)| w)
        .map_err(Error::from_mlx)
    })
  }

  /// Shared construction body for every MLX constructor: read + validate the
  /// `config.json` from `dir`, then load the weights via `load`, `sanitize`,
  /// read the optional pooling config, and build the Gemma3 sentence-encoder.
  ///
  /// `dir` is the directory the `config.json` + `1_Pooling/config.json` live in
  /// (the checkpoint dir for [`Self::from_dir`], the weight file's parent for the
  /// explicit-format constructors); `load` supplies the raw (pre-`sanitize`)
  /// weight map. The config read + full [`Gemma3Config::validate`] run BEFORE
  /// `load`, so a malformed config fails fast and never touches the weight file.
  fn construct(
    dir: &Path,
    load: impl FnOnce() -> Result<HashMap<String, mlxrs::Array>>,
  ) -> Result<Self> {
    let config_path = dir.join(CONFIG_FILE);

    let config_json = std::fs::read_to_string(&config_path)?;
    let config = Gemma3Config::from_json(&config_json).map_err(Error::from_mlx)?;

    // Run the FULL `Gemma3Config::validate` (it pins `model_type` and requires
    // every dimension / count — `hidden_size`, `vocab_size`, the layer / head
    // counts, the grouped-query split, the finite-positive RoPE / RMSNorm /
    // scale floats — structurally valid) BEFORE any weight is loaded, so a
    // malformed config fails fast with a typed error and never touches the
    // (expensive) weight file. NO upper cap beyond `mlxrs`'s own is imposed (the
    // checkpoint author owns the model dimensions; this is a library, not
    // DoS-hardened).
    config.validate().map_err(Error::from_mlx)?;

    let raw = load()?;
    let weights = sanitize(raw).map_err(Error::from_mlx)?;

    // An MLX EmbeddingGemma checkpoint may be a QUANTIZED safetensors (an
    // `mlx-community` 8-bit export whose projections + token embedding carry
    // per-layer `.scales` / `.biases`), not dense f32. `mlxrs`'s
    // `from_weights` is a UNIFIED entry point: it auto-detects each layer's
    // quantization from the `.scales` signal and resolves the per-layer scheme
    // from the config's `quantization` block, so the dense and quantized
    // checkpoints both load through this one call. (The `.scales` discriminator
    // is exposed as `weights_are_quantized` for parity with the SigLIP2 backend
    // + the unit tests, but the load does not branch on it here.)
    let _ = weights_are_quantized(&weights);

    // Read the optional `1_Pooling/config.json` (the matryoshka dimension + the
    // mean strategy). A genuinely absent file is `None` — `mlxrs`'s
    // `from_weights` then uses the deployment default (mean + normalize + full
    // dimension), exactly as the `mlxrs` load factory does. Only attempt the
    // parse when the file is actually present so an absent pooling config is not
    // surfaced as a load error.
    let pooling: Option<StPoolingConfig> = if dir.join("1_Pooling").join("config.json").is_file() {
      Some(pooling_from_st_config_path(dir).map_err(Error::from_mlx)?)
    } else {
      None
    };

    let model = EmbeddingGemmaModel::from_weights(config, weights, pooling.as_ref())
      .map_err(Error::from_mlx)?;

    // The Gemma `<pad>` id the model's dynamic-right-pad text encoding writes
    // into pad cells — read from the model's own contract, not hard-coded.
    let pad_token_id = model.text_encoding_pad_token_id();

    Ok(Self {
      model: Rc::new(model),
      pad_token_id,
      // The MLX constructor takes no `Options`, so adopt the crate-default
      // `max_batch_size` cap and `batch_size` micro-batch the ORT path uses by
      // default — same contract.
      max_batch_size: crate::options::BatchOptions::default().max_batch_size(),
      batch_size: crate::options::BatchOptions::default().batch_size(),
    })
  }

  /// The hard per-batch cap this model adopted at construction (the crate-
  /// default `max_batch_size`). Read by [`crate::text_enc::TextEncoder::embed_batch`]
  /// so the outer max-batch guard can reject an oversized batch before the
  /// per-item empty scan, symmetric with the ORT path.
  pub(crate) fn max_batch_size(&self) -> usize {
    self.max_batch_size
  }

  /// Encode a batch of text strings. Each string is tokenized with special
  /// tokens, then right-padded to the batch maximum with the Gemma `<pad>` id
  /// under EmbeddingGemma's dynamic-right-pad contract (matching `mlxrs`'s
  /// `Padding::DynamicRightPad`), the matching `0/1` attention mask is built,
  /// and the batch is run through the bidirectional backbone + mean-pool +
  /// Dense head in one call.
  pub(crate) fn embed_text_batch(
    &self,
    tokenizer: &Tokenizer,
    texts: &[&str],
  ) -> Result<Vec<Embedding>> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }
    // Reject oversized batches BEFORE allocating, with the same typed error
    // the ORT path returns.
    if texts.len() > self.max_batch_size {
      return Err(Error::BatchTooLarge {
        got: texts.len(),
        max: self.max_batch_size,
      });
    }

    // Honor the crate's `batch_size` micro-batch contract the ORT path uses:
    // split the (already cap-bounded) request into `batch_size`-row chunks and
    // run one `encode_text` forward per chunk, each dynamically right-padded to
    // its OWN max length, appending rows in order. A within-cap batch therefore
    // never executes as a single oversized MLX/Metal graph.
    let mut out = Vec::with_capacity(texts.len());
    for (chunk_idx, group) in texts.chunks(self.batch_size).enumerate() {
      // Mirror the ORT path's indexed batch-error contract (`Error::Batch
      // { index, source }`, see `OrtTextEncoder::embed_batch`): wrap each
      // chunk-level failure with the chunk's base input index, and a row-level
      // embedding conversion failure with `base + row`, so a caller can
      // quarantine the offending input regardless of backend.
      let base = chunk_idx * self.batch_size;
      let TextBatch {
        input_ids,
        attention_mask,
        batch,
        seq_len,
      } = build_text_batch(tokenizer, group, self.pad_token_id).map_err(|e| Error::Batch {
        index: base,
        source: Box::new(e),
      })?;

      let input_ids =
        mlxrs::Array::from_slice::<i32>(&input_ids, &(batch, seq_len)).map_err(|e| {
          Error::Batch {
            index: base,
            source: Box::new(Error::from_mlx(e)),
          }
        })?;
      let attention_mask = mlxrs::Array::from_slice::<f32>(&attention_mask, &(batch, seq_len))
        .map_err(|e| Error::Batch {
          index: base,
          source: Box::new(Error::from_mlx(e)),
        })?;

      let pooled = self
        .model
        .encode_text(&input_ids, &attention_mask)
        .map_err(|e| Error::Batch {
          index: base,
          source: Box::new(Error::from_mlx(e)),
        })?;
      for (row_idx, row) in eval_rows(&pooled, batch)
        .map_err(|e| Error::Batch {
          index: base,
          source: Box::new(e),
        })?
        .into_iter()
        .enumerate()
      {
        out.push(embedding_from_row(row).map_err(|e| Error::Batch {
          index: base + row_idx,
          source: Box::new(e),
        })?);
      }
    }
    Ok(out)
  }
}

/// A small extension that surfaces the model's dynamic-right-pad `pad_token_id`
/// from its [`mlxrs::embeddings::TextEmbedder`] contract, so the wrapper grounds
/// the pad id it writes into pad cells in `mlxrs`'s own text-encoding contract
/// rather than re-deriving it.
trait TextEncodingPadId {
  fn text_encoding_pad_token_id(&self) -> u32;
}

impl TextEncodingPadId for EmbeddingGemmaModel {
  fn text_encoding_pad_token_id(&self) -> u32 {
    use mlxrs::embeddings::{Padding, TextEmbedder};
    match self.text_encoding().padding {
      Padding::DynamicRightPad { pad_token_id } => pad_token_id,
      // EmbeddingGemma's `text_encoding` always declares `DynamicRightPad`; the
      // other variant is never produced. Fall back to the Gemma `<pad>` id (0)
      // rather than panic on the impossible branch.
      _ => 0,
    }
  }
}

/// The flat `(batch * seq_len)` row-major `input_ids` (`i32`) + `attention_mask`
/// (`f32`) the MLX text path feeds `encode_text`, plus the `(batch, seq_len)`
/// geometry.
struct TextBatch {
  input_ids: Vec<i32>,
  attention_mask: Vec<f32>,
  batch: usize,
  seq_len: usize,
}

/// Tokenize `texts` and build the flat `(batch * seq_len)` row-major `i32`
/// `input_ids` + `f32` `attention_mask` matrices under EmbeddingGemma's
/// dynamic-right-pad contract: every row is right-padded to the batch maximum
/// real length with `pad_token_id`, the mask is `1.0` over real tokens and
/// `0.0` over pad cells.
///
/// `tokenizer` MUST have its built-in padding disabled (see
/// [`crate::text_enc::prepare_mlx_tokenizer`]) so `encode_batch` returns each
/// row's real ids (with the post-processor's BOS); the right-padding to the
/// batch maximum happens here. Factored out of [`MlxModel::embed_text_batch`]
/// so the construction is unit-testable without the GPU model.
///
/// `batch * seq_len` is computed with `checked_mul` and the buffers reserved
/// fallibly, so a pathological sequence length or batch surfaces a typed error,
/// never a panic / abort. The caller is responsible for the `max_batch_size`
/// cap (it is enforced in `embed_text_batch` before this call).
fn build_text_batch(tokenizer: &Tokenizer, texts: &[&str], pad_token_id: u32) -> Result<TextBatch> {
  let encodings = tokenizer
    .encode_batch(texts.to_vec(), true)
    .map_err(|e| Error::Tokenizer(e.to_string()))?;

  // The padded sequence length is the batch's longest real row. With the
  // tokenizer's built-in padding disabled, each encoding is at its natural
  // length, so the maximum is the dynamic-right-pad target. An empty `texts`
  // is handled by the caller (returns early); here `encodings` is non-empty.
  let seq_len = encodings
    .iter()
    .map(|e| e.get_ids().len())
    .max()
    .unwrap_or(0);
  let batch = texts.len();

  let total = batch.checked_mul(seq_len).ok_or_else(|| {
    Error::mlx_owned(format!("batch {batch} * seq_len {seq_len} overflows usize"))
  })?;

  let mut input_ids: Vec<i32> = Vec::new();
  input_ids
    .try_reserve_exact(total)
    .map_err(|e| Error::AllocationFailed {
      which: "mlx text input_ids",
      requested_bytes: total.saturating_mul(std::mem::size_of::<i32>()),
      cause: e.to_string(),
    })?;
  let mut attention_mask: Vec<f32> = Vec::new();
  attention_mask
    .try_reserve_exact(total)
    .map_err(|e| Error::AllocationFailed {
      which: "mlx text attention_mask",
      requested_bytes: total.saturating_mul(std::mem::size_of::<f32>()),
      cause: e.to_string(),
    })?;

  let pad = checked_id(pad_token_id)?;
  for enc in &encodings {
    let ids = enc.get_ids();
    for &id in ids {
      input_ids.push(checked_id(id)?);
      attention_mask.push(1.0);
    }
    for _ in ids.len()..seq_len {
      input_ids.push(pad);
      attention_mask.push(0.0);
    }
  }

  Ok(TextBatch {
    input_ids,
    attention_mask,
    batch,
    seq_len,
  })
}

/// Convert a `tokenizers` `u32` id to the `i32` MLX consumes, rejecting any id
/// above `i32::MAX` (which an `as` cast would wrap to a negative gather index)
/// with a typed [`Error::Mlx`] naming the offending id.
fn checked_id(id: u32) -> Result<i32> {
  i32::try_from(id).map_err(|_| {
    Error::mlx_owned(format!(
      "tokenizer id {id} exceeds i32::MAX for MLX input_ids"
    ))
  })
}

/// Evaluate a `(rows, dim)` MLX array and split it into `rows` owned `Vec<f32>`
/// rows. The array is cast to f32 (a no-op for an already-f32 embedding; the
/// needed cast for an f16/bf16/quantized-checkpoint embedding) before `eval` so
/// the model's tensors are never mutated.
fn eval_rows(arr: &mlxrs::Array, rows: usize) -> Result<Vec<Vec<f32>>> {
  let shape = arr.shape();
  if shape.len() != 2 {
    return Err(Error::mlx_owned(format!(
      "expected a rank-2 (rows, dim) embedding tensor, got shape {shape:?}"
    )));
  }
  if shape[0] != rows {
    return Err(Error::mlx_owned(format!(
      "expected {rows} embedding rows, got {}",
      shape[0]
    )));
  }
  let dim = shape[1];
  if dim == 0 {
    return Err(Error::mlx("embedding tensor has a zero-width dimension"));
  }
  // Cast to f32 before the host copy: a half-precision (f16/bf16) or quantized
  // MLX checkpoint yields an embedding in its activation dtype (the tower
  // preserves it through `l2_normalize`), and `to_vec::<f32>` is dtype-strict.
  // `astype` is a no-op for an already-f32 embedding and produces a NEW array,
  // so the model's tensors are never mutated (the property the prior
  // `try_clone` had).
  let mut owned = arr.astype(mlxrs::Dtype::F32).map_err(Error::from_mlx)?;
  owned.eval().map_err(Error::from_mlx)?;
  let flat = owned.to_vec::<f32>().map_err(Error::from_mlx)?;
  Ok(flat.chunks_exact(dim).map(<[f32]>::to_vec).collect())
}

/// Wrap one model-output row into an [`Embedding`].
///
/// `mlxrs`'s `encode_text` L2-normalizes its output, so the row is unit-norm to
/// f32 ULP; [`Embedding::from_model_output`] validates the dim (768) and
/// renormalizes (snapping tiny f32 drift), the same validated path the ONNX
/// backend's rows take.
fn embedding_from_row(row: Vec<f32>) -> Result<Embedding> {
  Embedding::from_model_output(&row)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A weight map whose key set mirrors a DENSE checkpoint (only `.weight`
  /// siblings, no `.scales`) is classified dense. Pins the dense side of the
  /// dense-vs-quantized discriminator without the GPU model or a real
  /// checkpoint.
  #[test]
  fn discriminator_classifies_dense_weight_map() {
    let mut weights: HashMap<String, mlxrs::Array> = HashMap::new();
    let dummy = mlxrs::Array::from_slice::<f32>(&[0.0], &(1usize,)).expect("1-elem array");
    weights.insert(
      "model.layers.0.self_attn.q_proj.weight".to_string(),
      dummy.try_clone().expect("clone"),
    );
    weights.insert("model.embed_tokens.weight".to_string(), dummy);
    assert!(
      !weights_are_quantized(&weights),
      "a `.weight`-only map must be classified dense"
    );
  }

  /// A weight map carrying a single `<layer>.scales` sibling (the 8-bit
  /// quantized export's packed projection) is classified quantized. A `.scales`
  /// ANYWHERE in the map flips the discriminator.
  #[test]
  fn discriminator_classifies_quantized_weight_map() {
    let mut weights: HashMap<String, mlxrs::Array> = HashMap::new();
    let dummy = mlxrs::Array::from_slice::<f32>(&[0.0], &(1usize,)).expect("1-elem array");
    weights.insert(
      "dense.0.weight".to_string(),
      dummy.try_clone().expect("clone"),
    );
    weights.insert(
      "dense.0.scales".to_string(),
      dummy.try_clone().expect("clone"),
    );
    weights.insert("dense.0.biases".to_string(), dummy);
    assert!(
      weights_are_quantized(&weights),
      "a map with any `.scales` sibling must be classified quantized"
    );
  }

  /// The boundary id `i32::MAX` is in range and converts cleanly; the first id
  /// above it is rejected with a typed `Error::Mlx` naming the offending id,
  /// rather than wrapping (via an `as` cast) to a negative MLX gather index.
  #[test]
  fn checked_id_accepts_max_i32_and_rejects_overflow() {
    assert_eq!(checked_id(i32::MAX as u32).expect("in range"), i32::MAX);
    let bad = (i32::MAX as u32) + 1;
    match checked_id(bad) {
      Err(Error::Mlx(msg)) => assert!(
        msg.contains(&bad.to_string()),
        "error must name the offending id {bad}, got {msg:?}"
      ),
      other => panic!("expected Error::Mlx naming the id, got {other:?}"),
    }
  }

  /// A `config.json` whose `hidden_size` is non-positive is rejected at
  /// construction with a typed [`Error::Mlx`], BEFORE any weight is loaded — a
  /// zero `hidden_size` would otherwise build a zero-width pooled / Dense-head
  /// tensor. The guard runs on the parsed config, so it surfaces without a real
  /// `model.safetensors` on disk. NB: no UPPER cap is asserted — a large
  /// positive `hidden_size` is the checkpoint author's dimension and stays
  /// accepted (this is a library, not DoS-hardened).
  #[test]
  fn from_dir_rejects_nonpositive_hidden_size() {
    let dir = std::env::temp_dir().join(format!(
      "egemma_mlx_cfg_hidden_{}_{:?}",
      std::process::id(),
      std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp config dir");
    std::fs::write(dir.join(CONFIG_FILE), br#"{"hidden_size": 0}"#).expect("write config.json");
    let result = MlxModel::from_dir(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    // `MlxModel` is not `Debug` (it holds an `Rc`-backed device model), so use
    // `.err().expect(...)` rather than formatting the whole `Result`.
    let err = result
      .err()
      .expect("a zero hidden_size must be rejected at construction");
    match err {
      Error::Mlx(msg) => assert!(
        msg.contains("hidden_size"),
        "expected an Error::Mlx naming hidden_size, got {msg:?}"
      ),
      other => panic!("expected Error::Mlx for a zero hidden_size, got {other}"),
    }
  }

  /// A `config.json` that PARSES but fails the full [`Gemma3Config::validate`]
  /// on a NON-`hidden_size` field — here a wrong `model_type` (validate pins it
  /// to `"gemma3_text"`) — is rejected with a typed [`Error::Mlx`] BEFORE any
  /// weight is loaded. The temp dir holds ONLY the malformed `config.json` (no
  /// readable weight file), so a `from_dir` that nonetheless errors proves the
  /// full config validation runs ahead of the (expensive) weight load, not after
  /// it in `from_weights`.
  #[test]
  fn from_dir_rejects_invalid_config_before_weight_load() {
    let dir = std::env::temp_dir().join(format!(
      "egemma_mlx_cfg_modeltype_{}_{:?}",
      std::process::id(),
      std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp config dir");
    // Valid JSON with a valid positive `hidden_size`, but a `model_type` the
    // validator rejects — so the failure is on a field OTHER than `hidden_size`,
    // and no weight file exists in the dir.
    std::fs::write(
      dir.join(CONFIG_FILE),
      br#"{"model_type": "not_gemma3", "hidden_size": 768}"#,
    )
    .expect("write config.json");
    let result = MlxModel::from_dir(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    let err = result
      .err()
      .expect("an invalid model_type must be rejected at construction");
    assert!(
      matches!(err, Error::Mlx(_)),
      "expected Error::Mlx for an invalid config, got {err}"
    );
  }

  /// A quantized / fp16 MLX checkpoint yields an embedding in f16; the strict
  /// `to_vec::<f32>` would fail without the astype cast. Build an f16 `(2, 2)`
  /// array and assert it extracts to the right f32 rows.
  #[test]
  fn eval_rows_casts_half_precision_embedding_to_f32() {
    let dense = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
    let half = dense.astype(mlxrs::Dtype::F16).unwrap();
    let rows = eval_rows(&half, 2).unwrap();
    assert_eq!(rows, vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]]);
  }

  /// [`weights_parent`] returns the file's parent directory for a path with a
  /// directory component, and the current directory (`.`) — never the filesystem
  /// root — for a bare filename, so an explicit-format constructor handed
  /// `"model.safetensors"` reads `./config.json`.
  #[test]
  fn weights_parent_resolves_parent_else_current_dir() {
    assert_eq!(
      weights_parent(Path::new("/ckpt/model.safetensors")),
      Path::new("/ckpt")
    );
    assert_eq!(
      weights_parent(Path::new("ckpt/model.npz")),
      Path::new("ckpt")
    );
    // A bare filename has an empty parent; it must map to `.`, not `""`.
    assert_eq!(
      weights_parent(Path::new("model.safetensors")),
      Path::new(".")
    );
  }
}
