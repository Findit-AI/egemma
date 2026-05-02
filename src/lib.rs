#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes, missing_docs)]

pub mod embedding;
pub mod error;
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
pub use options::{BatchOptions, Options, ThreadOptions};
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use text_enc::TextEncoder;
