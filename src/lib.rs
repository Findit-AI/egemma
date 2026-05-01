//! EmbeddingGemma inference library — produces 768-dim L2-normalized
//! sentence embeddings from Google's `embedding-gemma` ONNX export.
//!
//! Mirrors the `siglip2` text-tower API surface: a [`TextEncoder`] with
//! `from_files` / `from_files_with_options` / `from_ort_session`
//! constructors, plus `embed`, `embed_batch`, and `warmup`.
//!
//! # Target / feature contract
//!
//! The `inference` feature is **on by default** and is **native-only**.
//! It pulls `ort` (ONNX Runtime FFI) and `tokenizers` (which transitively
//! depends on C-only libraries like `onig_sys`); neither builds on
//! `wasm32-*` today. Building wasm with default features therefore fails
//! deep in `getrandom` / `onig_sys` before this crate's code is reached.
//!
//! **Wasm consumers must opt out:**
//!
//! ```bash
//! cargo check --target wasm32-unknown-unknown --no-default-features
//! ```
//!
//! Without `inference`, the public surface is the [`Embedding`] type,
//! [`Options`] / [`BatchOptions`] / [`ThreadOptions`], and the
//! [`Error`] enum — useful when inference itself happens elsewhere
//! (a server, a different runtime) and only the value types and
//! similarity primitive need to be present.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes)]

pub mod embedding;
pub mod error;
pub mod options;
#[cfg(feature = "inference")]
pub(crate) mod session;
pub(crate) mod simd;
#[cfg(feature = "inference")]
pub mod text_enc;

pub use embedding::Embedding;
pub use error::{Error, Result};
#[cfg(feature = "inference")]
pub use options::GraphOptimizationLevel;
pub use options::{BatchOptions, Options, ThreadOptions};
#[cfg(feature = "inference")]
pub use text_enc::TextEncoder;
