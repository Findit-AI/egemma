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
// MLX (`mlxrs`) inference backend — Apple Silicon only, opt-in via the `mlx`
// feature. Crate-internal: reached through the platform auto-routing in
// `TextEncoder::from_dir` (there is no public MLX entry point — the public
// surface is `TextEncoder` plus the explicit MLX constructors, see Cargo.toml).
// Compiled only on `aarch64-apple-darwin` with `mlx` on — it reuses the
// `tokenizers` text path and backs `TextEncoder`, so a default (or
// `--no-default-features`) build pulls in neither it nor `mlxrs`. `mlxrs` binds
// the MLX C++ runtime and has no other target, so the backend exists nowhere else.
#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
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

#[cfg(feature = "windowing")]
#[cfg_attr(docsrs, doc(cfg(feature = "windowing")))]
pub mod window;
#[cfg(feature = "windowing")]
#[cfg_attr(docsrs, doc(cfg(feature = "windowing")))]
pub use window::{WindowEmbedding, WindowOptions};

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[cfg_attr(
  docsrs,
  doc(cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))
)]
pub use error::MlxErrorKind;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use text_enc::TextEncoder;
