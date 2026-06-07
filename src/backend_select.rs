//! Checkpoint-directory layout constants + the Apple-Silicon backend probe.
//!
//! [`crate::TextEncoder::from_dir`] picks its inference backend automatically;
//! [`crate::TextEncoder::from_dir_with_options`] accepts an explicit
//! [`crate::options::Backend`] override. This module owns the small amount of shared
//! logic that decision needs: the canonical file names a checkpoint directory
//! holds, and (on `aarch64-apple-darwin`) the probe that decides whether a
//! directory is an MLX checkpoint worth routing to the `mlxrs` Metal backend.
//!
//! Routing contract:
//! - On **Apple Silicon**, `from_dir` prefers MLX when [`prefer_mlx`] is `true`
//!   (an MLX `config.json` is present, a weight set in any ENABLED format is
//!   present — a sharded `model.safetensors.index.json` or a single
//!   `model.safetensors` always, a `*.npz` only under the `npz` feature, a
//!   `*.gguf` only under the `gguf` feature — AND **none** of the ONNX graph(s)
//!   the calling constructor needs is in the directory) and falls back to ONNX
//!   otherwise. Routing is **per-constructor**: each `from_dir`
//!   passes the graph(s) *it* loads, so the probe checks the graph the caller
//!   actually needs — [`crate::TextEncoder::from_dir`] passes [`TEXT_ONNX`]. The
//!   ONNX graph disambiguates: `config.json` + `model.safetensors` are also the
//!   standard HuggingFace source-asset names, so a directory that ships those HF
//!   sources alongside the `*.onnx` graph the caller needs is an ONNX checkpoint
//!   for that constructor and must route to ONNX — the presence of the required
//!   graph is the signal that wins, because that file is the thing the ONNX
//!   backend actually loads (the MLX backend never reads it).
//! - On **every other platform**, only the ONNX backend is compiled, so
//!   `from_dir` loads the ONNX graph unconditionally and this probe is unused.

/// The EmbeddingGemma text-encoder ONNX graph file name inside a checkpoint
/// directory (the canonical fp32 optimum export of
/// `google/embeddinggemma-300m`). Its `.onnx_data` external-weights sidecar,
/// when present, is auto-discovered by ORT alongside it.
pub(crate) const TEXT_ONNX: &str = "model.onnx";

/// The MLX-format config file name (the `mlxrs` checkpoint marker, paired with a
/// weight file). Mirrors `crate::mlx`'s `CONFIG_FILE`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const MLX_CONFIG: &str = "config.json";

/// The MLX-format safetensors weights file name — the always-available baseline
/// weight format. Its presence (with [`MLX_CONFIG`]) is one signal `from_dir`
/// routes to the MLX backend on Apple Silicon.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const MLX_SAFETENSORS: &str = "model.safetensors";

/// The legacy single-file safetensors weights name. Some older MLX checkpoints
/// ship their weights as `weights.safetensors` rather than `model.safetensors`;
/// `mlxrs::io::load_weights_from_dir` accepts it as a fallback tier, so its
/// presence (with [`MLX_CONFIG`]) also routes to the MLX backend.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const MLX_SAFETENSORS_LEGACY: &str = "weights.safetensors";

/// The sharded-checkpoint index file name. A multi-shard safetensors export
/// (`model-00001-of-0000N.safetensors` + …) ships a `model.safetensors.index.json`
/// weight map instead of a single `model.safetensors`; `mlxrs::io::load_weights_from_dir`
/// loads it, so its presence (with [`MLX_CONFIG`]) also routes to MLX.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const MLX_SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

/// Report whether `dir` holds an MLX weight set in any ENABLED format:
/// a sharded `model.safetensors.index.json` always; a single `model.safetensors`
/// (or the legacy `weights.safetensors`) always; a `*.npz` only under the `npz`
/// feature; a `*.gguf` only under the `gguf` feature. Mirrors the formats
/// [`mlxrs::io::load_weights_from_dir`] loads, so routing and loading agree on
/// which checkpoints count. A dir with only `model.npz` therefore routes to MLX
/// iff `npz` is on.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn has_mlx_weights(dir: &std::path::Path) -> bool {
  if dir.join(MLX_SAFETENSORS_INDEX).is_file() {
    return true;
  }
  if dir.join(MLX_SAFETENSORS).is_file() {
    return true;
  }
  if dir.join(MLX_SAFETENSORS_LEGACY).is_file() {
    return true;
  }
  #[cfg(feature = "npz")]
  if has_extension(dir, "npz") {
    return true;
  }
  #[cfg(feature = "gguf")]
  if has_extension(dir, "gguf") {
    return true;
  }
  false
}

/// Whether `dir` contains at least one file with the given `extension`. Only
/// referenced from the `npz`/`gguf` arms of [`has_mlx_weights`], so it is
/// `cfg`-elided on a default (safetensors-only) build.
#[cfg(all(
  target_os = "macos",
  target_arch = "aarch64",
  any(feature = "npz", feature = "gguf")
))]
fn has_extension(dir: &std::path::Path, extension: &str) -> bool {
  let Ok(entries) = std::fs::read_dir(dir) else {
    return false;
  };
  entries.flatten().any(|entry| {
    let path = entry.path();
    path.extension().and_then(|e| e.to_str()) == Some(extension) && path.is_file()
  })
}

/// The backend [`route`] selected for a checkpoint directory.
#[cfg(all(feature = "inference", not(target_arch = "wasm32")))]
pub(crate) enum Routed {
  /// Load via ONNX Runtime.
  Onnx,
  /// Load via the MLX backend.
  Mlx,
}

/// Decide which backend to load for `dir`, honoring an explicit
/// [`crate::options::Backend`]. `Auto` uses the [`prefer_mlx`] probe; `Onnx`
/// forces ONNX; `Mlx` forces MLX and errors with
/// [`crate::Error::BackendUnavailable`] when the directory holds no MLX
/// checkpoint.
#[cfg(all(feature = "inference", not(target_arch = "wasm32"), target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn route(
  dir: &std::path::Path,
  backend: crate::options::Backend,
  required_onnx: &[&str],
) -> Result<Routed, crate::Error> {
  use crate::options::Backend;
  match backend {
    Backend::Auto => Ok(if prefer_mlx(dir, required_onnx) {
      Routed::Mlx
    } else {
      Routed::Onnx
    }),
    Backend::Onnx => Ok(Routed::Onnx),
    Backend::Mlx => {
      if dir.join(MLX_CONFIG).is_file() && has_mlx_weights(dir) {
        Ok(Routed::Mlx)
      } else {
        Err(crate::Error::BackendUnavailable {
          requested: Backend::Mlx,
          reason: format!(
            "no MLX checkpoint (config.json + a weight file) in {}",
            dir.display()
          ),
        })
      }
    }
  }
}

/// Off Apple Silicon only the ONNX backend exists, so `Auto`/`Onnx` route to
/// ONNX and `Mlx` is unavailable.
#[cfg(all(
  feature = "inference",
  not(target_arch = "wasm32"),
  not(all(target_os = "macos", target_arch = "aarch64"))
))]
pub(crate) fn route(
  _dir: &std::path::Path,
  backend: crate::options::Backend,
  _required_onnx: &[&str],
) -> Result<Routed, crate::Error> {
  use crate::options::Backend;
  match backend {
    Backend::Auto | Backend::Onnx => Ok(Routed::Onnx),
    Backend::Mlx => Err(crate::Error::BackendUnavailable {
      requested: Backend::Mlx,
      reason: "the MLX backend is only available on aarch64-apple-darwin".to_string(),
    }),
  }
}

/// Probe `dir` and report whether the MLX backend should load it for the calling
/// constructor: `true` iff it contains an MLX `config.json` and a weight file in
/// any enabled format (see [`has_mlx_weights`]) AND **none** of `required_onnx`
/// (the ONNX graph file name(s) the caller loads) is a file in `dir`. This is
/// the checkpoint-format detection the auto-routing `from_dir` constructor uses
/// on Apple Silicon.
///
/// The required ONNX graph is the disambiguator: `config.json` +
/// `model.safetensors` are also the standard HuggingFace source-asset names, so
/// a directory that ships those HF sources next to an `*.onnx` graph the caller
/// needs would otherwise be misrouted to MLX (and `from_dir` would then fail with
/// no ONNX fallback). Because the MLX backend never reads the `*.onnx` graph
/// while the ONNX backend does, the presence of a required graph means "this is
/// an ONNX checkpoint for this constructor" and routes to ONNX. A directory with
/// a required `*.onnx` graph but no MLX weights likewise returns `false`.
///
/// Pure filesystem existence checks, no I/O of the files themselves: the
/// constructor that wins does the real load (and surfaces a typed error if the
/// chosen checkpoint is malformed), so this stays a cheap, side-effect-free
/// dispatch decision.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn prefer_mlx(dir: &std::path::Path, required_onnx: &[&str]) -> bool {
  dir.join(MLX_CONFIG).is_file()
    && has_mlx_weights(dir)
    && !required_onnx.iter().any(|onnx| dir.join(onnx).is_file())
}

#[cfg(all(test, feature = "inference", not(target_arch = "wasm32")))]
mod route_tests {
  use super::*;
  use crate::options::Backend;

  #[test]
  fn route_onnx_is_always_onnx() {
    let tmp = std::env::temp_dir().join(format!("egemma_route_onnx_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir");
    // Force ONNX must route ONNX regardless of contents.
    let r = route(&tmp, Backend::Onnx, &[TEXT_ONNX]).expect("onnx route ok");
    assert!(matches!(r, Routed::Onnx));
    let _ = std::fs::remove_dir_all(&tmp);
  }

  #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
  #[test]
  fn route_mlx_unavailable_off_apple_silicon() {
    let tmp = std::env::temp_dir().join(format!("egemma_route_mlx_off_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir");
    let err = route(&tmp, Backend::Mlx, &[TEXT_ONNX])
      .err()
      .expect("forcing Mlx off Apple Silicon must error");
    assert!(matches!(err, crate::Error::BackendUnavailable { .. }));
    let _ = std::fs::remove_dir_all(&tmp);
  }

  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  #[test]
  fn route_mlx_unavailable_when_no_mlx_checkpoint() {
    let tmp = std::env::temp_dir().join(format!("egemma_route_mlx_empty_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir");
    // Empty dir: forcing Mlx must error (no config.json + weights).
    let err = route(&tmp, Backend::Mlx, &[TEXT_ONNX])
      .err()
      .expect("forcing Mlx with no checkpoint must error");
    assert!(matches!(err, crate::Error::BackendUnavailable { .. }));
    let _ = std::fs::remove_dir_all(&tmp);
  }
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests {
  use super::*;

  /// A directory holding BOTH an MLX `config.json` and `model.safetensors`, with
  /// the required ONNX graph absent, is an MLX checkpoint — `prefer_mlx` routes
  /// it to the MLX backend.
  #[test]
  fn prefer_mlx_true_when_mlx_weights_present() {
    let tmp = std::env::temp_dir().join(format!("egemma_mlx_probe_mlx_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join(MLX_SAFETENSORS), b"\0").expect("write model.safetensors");
    assert!(
      prefer_mlx(&tmp, &[TEXT_ONNX]),
      "config.json + model.safetensors present (no ONNX graph) must select MLX"
    );
    let _ = std::fs::remove_dir_all(&tmp);
  }

  /// A SHARDED MLX checkpoint — `config.json` + `model.safetensors.index.json`
  /// (the weight map for a multi-shard export) with NO single `model.safetensors`
  /// and no ONNX graph — is an MLX checkpoint: `prefer_mlx` routes it to MLX,
  /// because `mlxrs::io::load_weights_from_dir` loads the sharded layout via the
  /// index.
  #[test]
  fn prefer_mlx_true_for_sharded_index_only() {
    let tmp = std::env::temp_dir().join(format!("egemma_mlx_probe_shard_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join(MLX_SAFETENSORS_INDEX), b"{}").expect("write index.json");
    let routed = prefer_mlx(&tmp, &[TEXT_ONNX]);
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
      routed,
      "config.json + model.safetensors.index.json (sharded, no ONNX graph) must select MLX"
    );
  }

  /// A LEGACY single-file MLX checkpoint — `config.json` + `weights.safetensors`
  /// (the older single-file name) with NO `model.safetensors` and no ONNX graph
  /// — is an MLX checkpoint: `prefer_mlx` routes it to MLX, because
  /// `mlxrs::io::load_weights_from_dir` accepts `weights.safetensors` as a
  /// fallback tier.
  #[test]
  fn prefer_mlx_true_for_legacy_weights_safetensors() {
    let tmp = std::env::temp_dir().join(format!("egemma_mlx_probe_legacy_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join(MLX_SAFETENSORS_LEGACY), b"\0").expect("write weights.safetensors");
    let routed = prefer_mlx(&tmp, &[TEXT_ONNX]);
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
      routed,
      "config.json + weights.safetensors (legacy single-file, no ONNX graph) must select MLX"
    );
  }

  /// An ONNX-only directory (the `model.onnx` graph, no MLX `model.safetensors`)
  /// is NOT an MLX checkpoint — `prefer_mlx` is `false`, so `from_dir` routes
  /// to ONNX. A bare `config.json` without the weights is likewise not enough.
  #[test]
  fn prefer_mlx_false_for_onnx_only_dir() {
    let tmp = std::env::temp_dir().join(format!("egemma_mlx_probe_onnx_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(TEXT_ONNX), b"\0").expect("write model.onnx");
    // A config.json alone (no model.safetensors) must still not select MLX.
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    assert!(
      !prefer_mlx(&tmp, &[TEXT_ONNX]),
      "the ONNX graph + a lone config.json (no model.safetensors) must NOT select MLX"
    );
    let _ = std::fs::remove_dir_all(&tmp);
  }

  /// When the required ONNX graph is present, the directory is an ONNX
  /// checkpoint and routes to ONNX — even if it also carries `config.json` +
  /// `model.safetensors` (which double as the standard HuggingFace source-asset
  /// names). The ONNX graph is the disambiguator, so `prefer_mlx` is `false`.
  #[test]
  fn prefer_mlx_false_when_onnx_graph_present_alongside_mlx_weights() {
    let tmp = std::env::temp_dir().join(format!("egemma_mlx_probe_both_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join(MLX_SAFETENSORS), b"\0").expect("write model.safetensors");
    std::fs::write(tmp.join(TEXT_ONNX), b"\0").expect("write model.onnx");
    assert!(
      !prefer_mlx(&tmp, &[TEXT_ONNX]),
      "the ONNX graph disambiguates: a dir carrying it is an ONNX checkpoint \
       and must route to ONNX, not MLX"
    );
    let _ = std::fs::remove_dir_all(&tmp);
  }

  /// A dir with only `config.json` + `model.npz` (no safetensors, no ONNX graph)
  /// routes to MLX **iff** the `npz` feature is on — the routing widens to the
  /// same formats the loader accepts. Without `npz`, the `.npz` is not a
  /// recognized weight file and `prefer_mlx` is `false`.
  #[test]
  fn prefer_mlx_npz_only_routes_iff_npz_feature() {
    let tmp = std::env::temp_dir().join(format!("egemma_mlx_probe_npz_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join("model.npz"), b"\0").expect("write model.npz");
    let routed = prefer_mlx(&tmp, &[TEXT_ONNX]);
    let _ = std::fs::remove_dir_all(&tmp);
    assert_eq!(
      routed,
      cfg!(feature = "npz"),
      "a config.json + model.npz dir must route to MLX iff the npz feature is on"
    );
  }

  /// Same contract for gguf: a `config.json` + `model.gguf` dir routes to MLX
  /// iff the `gguf` feature is on.
  #[test]
  fn prefer_mlx_gguf_only_routes_iff_gguf_feature() {
    let tmp = std::env::temp_dir().join(format!("egemma_mlx_probe_gguf_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir tmp");
    std::fs::write(tmp.join(MLX_CONFIG), b"{}").expect("write config.json");
    std::fs::write(tmp.join("model.gguf"), b"\0").expect("write model.gguf");
    let routed = prefer_mlx(&tmp, &[TEXT_ONNX]);
    let _ = std::fs::remove_dir_all(&tmp);
    assert_eq!(
      routed,
      cfg!(feature = "gguf"),
      "a config.json + model.gguf dir must route to MLX iff the gguf feature is on"
    );
  }
}
