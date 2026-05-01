//! Embed a few sentences with `embedding-gemma` and print pairwise cosine
//! similarity. Run from the crate root with the canonical fp32 export
//! from `onnx-community/embeddinggemma-300m-ONNX`:
//!
//! ```bash
//! cargo run --example embed_text -- \
//!     /path/to/model.onnx /path/to/tokenizer.json
//! ```
//!
//! The upstream model card flags fp16 as an unsupported activation
//! dtype for this graph; pass `model_fp16.onnx` only if you've
//! validated quality for your specific workload.

use std::{env, path::PathBuf, process::ExitCode};

use egemma::TextEncoder;

fn main() -> ExitCode {
  let mut args = env::args().skip(1);
  let graph: PathBuf = match args.next() {
    Some(p) => p.into(),
    None => {
      eprintln!("usage: embed_text <model.onnx> <tokenizer.json>");
      return ExitCode::from(2);
    }
  };
  let tokenizer: PathBuf = match args.next() {
    Some(p) => p.into(),
    None => {
      eprintln!("usage: embed_text <model.onnx> <tokenizer.json>");
      return ExitCode::from(2);
    }
  };

  let mut encoder = match TextEncoder::from_files(&graph, &tokenizer) {
    Ok(e) => e,
    Err(err) => {
      eprintln!("failed to load encoder: {err}");
      return ExitCode::FAILURE;
    }
  };

  let prompts = [
    "task: search result | query: how do I build a Rust ONNX inference library?",
    "Rust crates that wrap ONNX Runtime for embedding generation.",
    "Today's weather forecast for Singapore.",
  ];

  let embeddings = match encoder.embed_batch(&prompts) {
    Ok(v) => v,
    Err(err) => {
      eprintln!("embed failed: {err}");
      return ExitCode::FAILURE;
    }
  };

  for (i, e) in embeddings.iter().enumerate() {
    println!("[{i}] {:?}", &e.as_slice()[..6]);
  }
  println!();
  for i in 0..embeddings.len() {
    for j in (i + 1)..embeddings.len() {
      match embeddings[i].try_cosine(&embeddings[j]) {
        Ok(cos) => println!("cos({i}, {j}) = {cos:.4}"),
        Err(err) => {
          eprintln!("cos({i}, {j}) failed: {err}");
          return ExitCode::FAILURE;
        }
      }
    }
  }
  ExitCode::SUCCESS
}
