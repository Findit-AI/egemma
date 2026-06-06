//! Checkpoint-directory layout constants + the Apple-Silicon backend probe.
//!
//! [`crate::TextEncoder::from_dir`] picks its inference backend automatically —
//! there is no user-facing knob. This module owns the small amount of shared
//! logic that decision needs: the canonical file names a checkpoint directory
//! holds, and (on `aarch64-apple-darwin`) the probe that decides whether a
//! directory is an MLX checkpoint worth routing to the `mlxrs` Metal backend.
//!
//! Routing contract:
//! - On **Apple Silicon**, `from_dir` prefers MLX when [`prefer_mlx`] is `true`
//!   (an MLX `config.json` is present, a weight file in any ENABLED format is
//!   present — `model.safetensors` always, a `*.npz` only under the `npz`
//!   feature, a `*.gguf` only under the `gguf` feature — AND **none** of the
//!   ONNX graph(s) the calling constructor needs is in the directory) and falls
//!   back to ONNX otherwise. Routing is **per-constructor**: each `from_dir`
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
/// routes to the MLX backend on Apple Silicon. Mirrors `crate::mlx`'s
/// `SAFETENSORS_FILE`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) const MLX_SAFETENSORS: &str = "model.safetensors";

/// Report whether `dir` holds an MLX weight file in any ENABLED format:
/// `model.safetensors` always; a `*.npz` only under the `npz` feature; a
/// `*.gguf` only under the `gguf` feature. Mirrors the priority the loader's
/// `load_weights` detector probes, so routing and loading agree on which formats
/// count. A dir with only `model.npz` therefore routes to MLX iff `npz` is on.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn has_mlx_weights(dir: &std::path::Path) -> bool {
  if dir.join(MLX_SAFETENSORS).is_file() {
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
