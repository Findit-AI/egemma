//! End-to-end integration test against the released `embedding-gemma`
//! ONNX export.
//!
//! Requires the model files at runtime; gated behind the `EGEMMA_MODEL_DIR`
//! env var so `cargo test` works without the assets present. Set it to a
//! directory containing `model.onnx` (with its `.onnx_data` sidecar)
//! and a `tokenizer.json`.
//!
//! `model.onnx` is the canonical fp32 export from
//! `onnx-community/embeddinggemma-300m-ONNX`. The model card flags
//! fp16 as an unsupported activation dtype for this graph; we don't
//! auto-discover `model_fp16.onnx`. If you have only the fp16 file
//! locally and have validated it on your tokenizer/quality bar,
//! point at it explicitly via `EGEMMA_MODEL_FILE=model_fp16.onnx`.
//!
//! ```bash
//! EGEMMA_MODEL_DIR=/path/to/embedding-gemma cargo test --test integration
//! ```
//!
//! # CI contract — read this before assuming "tests passed = inference works"
//!
//! GitHub Actions does **not** set `EGEMMA_MODEL_DIR`. When unset, every
//! test in this file emits a `[INTEGRATION-SKIP]` banner and returns
//! `Ok(())` without loading a model. CI therefore reports them as
//! `ok` even though no `ort::Session::run` ever happened. This is a
//! deliberate trade-off: the alternative — pulling ~600 MB of model
//! assets per CI run, or maintaining a synthetic ONNX fixture — costs
//! more than it catches, because the structural risks (input/output
//! name drift, dtype drift, dim drift) are already enforced at
//! construction time by `validate_text_session` (see `src/text_enc.rs`),
//! and the unit tests pin the constant assumptions (`PAD_TOKEN`,
//! `EMBED_DIM`).
//!
//! **Developer responsibility.** Before merging changes that touch
//! `src/text_enc.rs`, `src/session.rs`, `src/simd/`, `src/embedding.rs`,
//! or this file, run:
//!
//! ```bash
//! EGEMMA_MODEL_DIR=/path/to/embedding-gemma cargo test --test integration
//! ```
//!
//! and confirm 4/4 pass. Grep CI logs for `[INTEGRATION-SKIP]` to
//! verify whether a given run actually exercised this path.

#![cfg(feature = "inference")]

use std::path::PathBuf;

use egemma::{Embedding, TextEncoder};

fn model_dir() -> Option<PathBuf> {
  std::env::var_os("EGEMMA_MODEL_DIR").map(PathBuf::from)
}

fn discover_graph(dir: &std::path::Path) -> PathBuf {
  if let Some(name) = std::env::var_os("EGEMMA_MODEL_FILE") {
    return dir.join(name);
  }
  let canonical = dir.join("model.onnx");
  if canonical.exists() {
    return canonical;
  }
  panic!(
    "no `model.onnx` found in {}; set `EGEMMA_MODEL_FILE` to point at \
     a different filename (the upstream model card flags fp16 as an \
     unsupported activation dtype for this graph, so `model_fp16.onnx` \
     is not auto-discovered — pass it explicitly only if you've \
     validated it for your workload)",
    dir.display()
  );
}

/// Centralizes the skip-or-load decision so every test prints the same
/// `[INTEGRATION-SKIP]` banner — searchable in CI logs to distinguish
/// "real run with assertions" from "skipped, env var unset" runs.
/// Returns `None` when `EGEMMA_MODEL_DIR` is unset; the caller should
/// `return` immediately.
fn try_load_encoder(test_name: &str) -> Option<TextEncoder> {
  if model_dir().is_none() {
    eprintln!(
      "[INTEGRATION-SKIP] {test_name}: EGEMMA_MODEL_DIR unset — skipping. \
       Run locally with `EGEMMA_MODEL_DIR=/path/to/embedding-gemma cargo test \
       --test integration` before merging inference-path changes."
    );
    return None;
  }
  let dir = model_dir().expect("model_dir checked above");
  let graph = discover_graph(&dir);
  let tokenizer = dir.join("tokenizer.json");
  Some(TextEncoder::from_files(&graph, &tokenizer).expect("loading encoder must succeed"))
}

#[test]
fn embed_single_returns_unit_norm_vector() {
  let Some(mut encoder) = try_load_encoder("embed_single_returns_unit_norm_vector") else {
    return;
  };
  let e = encoder
    .embed("hello world")
    .expect("single embed must succeed");
  assert_eq!(e.dim(), Embedding::EMBED_DIM);
  let cos = e.try_cosine(&e).expect("self-cosine on valid embedding");
  assert!(
    (cos - 1.0).abs() < 1e-4,
    "self-cosine should be 1.0; got {cos}"
  );
}

#[test]
fn embed_batch_preserves_order_and_self_cosine() {
  let Some(mut encoder) = try_load_encoder("embed_batch_preserves_order_and_self_cosine") else {
    return;
  };
  let prompts = ["alpha", "the quick brown fox", "lorem ipsum dolor sit amet"];
  let embeddings = encoder.embed_batch(&prompts).expect("batch embed");
  assert_eq!(embeddings.len(), prompts.len());
  for e in &embeddings {
    assert_eq!(e.dim(), Embedding::EMBED_DIM);
    let cos = e.try_cosine(e).expect("self-cosine on valid embedding");
    assert!((cos - 1.0).abs() < 1e-4);
  }
  let single = encoder
    .embed("alpha")
    .expect("single embed for parity check");
  let parity = single
    .try_cosine(&embeddings[0])
    .expect("parity cosine on valid pair");
  assert!(
    (parity - 1.0).abs() < 1e-3,
    "single vs batched embedding for the same prompt should match"
  );
}

#[test]
fn related_prompts_more_similar_than_unrelated() {
  let Some(mut encoder) = try_load_encoder("related_prompts_more_similar_than_unrelated") else {
    return;
  };
  let v = encoder
    .embed_batch(&[
      "task: search result | query: how do birds fly?",
      "Birds use lift generated by their wings to fly.",
      "The price of bananas in Tokyo is rising.",
    ])
    .expect("batch embed");

  let related = v[0]
    .try_cosine(&v[1])
    .expect("related cosine on valid pair");
  let unrelated = v[0]
    .try_cosine(&v[2])
    .expect("unrelated cosine on valid pair");
  assert!(
    related > unrelated,
    "expected related > unrelated; got related={related}, unrelated={unrelated}"
  );
}

#[test]
fn empty_text_rejected() {
  let Some(mut encoder) = try_load_encoder("empty_text_rejected") else {
    return;
  };
  let err = encoder.embed("").expect_err("empty text must error");
  assert!(matches!(err, egemma::Error::EmptyText));
}
