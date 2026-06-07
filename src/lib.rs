#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes, missing_docs)]

// Checkpoint-directory layout constants + the Apple-Silicon backend probe the
// auto-routing `TextEncoder::from_dir` constructor shares. Crate-internal;
// compiled wherever a `from_dir` constructor exists (the ONNX `from_dir` path
// needs only `inference` on a non-wasm host; the MLX probe inside is further
// target-gated).
#[cfg(all(feature = "inference", not(target_arch = "wasm32")))]
pub(crate) mod backend_select;
pub mod embedding;
pub mod error;
// MLX (`mlxrs`) inference backend — Apple Silicon only. Crate-internal: reached
// through the platform auto-routing in `TextEncoder::from_dir` (there is no
// public MLX entry point and no `mlx` feature — the backend is chosen by
// platform, see Cargo.toml). Compiled on `aarch64-apple-darwin` whenever the
// inference surface is built — it reuses the `tokenizers` text path and backs
// the `inference`-gated `TextEncoder`, so an `--no-default-features` ONNX-free
// build pulls in neither it nor `mlxrs`'s runtime use. `mlxrs` binds the MLX
// C++ runtime and has no other target, so the backend exists nowhere else.
#[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
mod mlx;
pub mod options;

#[cfg(feature = "inference")]
pub(crate) mod session;
pub(crate) mod simd;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub mod text_enc;

pub use embedding::Embedding;
pub use error::{Error, Result};
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use options::GraphOptimizationLevel;
pub use options::{Backend, BatchOptions, Options, ThreadOptions};

#[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
#[cfg_attr(docsrs, doc(cfg(all(target_os = "macos", target_arch = "aarch64"))))]
pub use error::MlxErrorKind;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use text_enc::TextEncoder;
