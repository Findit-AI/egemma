# Changelog

All notable changes to `egemma` will be documented in this file. The
format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0]

### Changed (breaking)

- `Error::Mlx(String)` is now `Error::Mlx { kind: MlxErrorKind, message: String }`.
- MLX backend now truncates inputs longer than `max_seq_len` (default 2048);
  previously the MLX path did not truncate. Use the upcoming windowing API for
  full long-input coverage.
- `Embedding` is now runtime-dimensioned: `Embedding::EMBED_DIM` is removed
  (use `Embedding::dim()`; `Embedding::DEFAULT_DIM` documents the 768 base).
  `from_model_output` / `TryFrom<Vec<f32>>` accept any non-empty unit-norm
  length, enabling Matryoshka (128/256/512) checkpoints end-to-end.

### Added

- `Backend` selector (`Auto`/`Onnx`/`Mlx`) on `Options` via `with_backend`.
- `TextEncoder::from_dir_with_options`, `from_safetensors_with_options`, and
  (feature-gated) `from_npz_with_options` / `from_gguf_with_options`.
- `Error::BackendUnavailable` for an unsatisfiable forced backend.
- `MlxErrorKind { Config, Load, Runtime }` tags on MLX failures.
- `Embedding::to_matryoshka(dim)` — prefix-truncate + renormalize a 768-d
  embedding to a shorter Matryoshka dimension.
- `Embedding::DEFAULT_DIM` — documentation constant for the 768 base-export
  dimension (replaces the removed `EMBED_DIM`).

## [0.1.0]

Initial release. See `Cargo.toml` for the public surface.
