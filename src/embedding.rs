//! `Embedding` — L2-normalized, runtime-dimensioned sentence embedding.

use std::sync::Arc;

use crate::error::{Error, Result};

/// L2-normalized sentence embedding. Length is runtime-determined (see
/// [`Self::dim`]); 768 for the base export, smaller for Matryoshka checkpoints
/// or after [`Self::to_matryoshka`].
///
/// `Embedding` deliberately does **not** implement `Serialize` or `Deserialize`.
/// An auto-derived `Deserialize` would bypass the L2-norm invariant
/// that `TryFrom<Vec<f32>>` exists to enforce. Round-trip via the inner
/// representation:
///
/// ```ignore
/// // Serialize via the inner slice (`&[f32]: Serialize`):
/// let json = serde_json::to_string(embedding.as_slice())?;
///
/// // Deserialize via the validated path:
/// let v: Vec<f32> = serde_json::from_str(&json)?;
/// let embedding  = Embedding::try_from(v)?;  // validates L2-norm (any non-zero length)
/// ```
#[derive(Clone, Debug)]
pub struct Embedding(Arc<[f32]>);

impl Embedding {
  /// The dimension of the canonical EmbeddingGemma base export. `Embedding` no
  /// longer enforces this — its dimension is runtime-determined (see
  /// [`Self::dim`]) to support Matryoshka checkpoints (128/256/512) — but the
  /// value is kept as documentation of the common case.
  pub const DEFAULT_DIM: usize = 768;

  /// L2-norm tolerance for the unit-norm invariant.
  pub const NORM_EPSILON: f32 = 5e-4;

  /// Number of `f32` lanes in the embedding. Runtime-determined; 768 for
  /// any `Embedding` produced from the base export, smaller for Matryoshka
  /// checkpoints or after [`Self::to_matryoshka`].
  pub fn dim(&self) -> usize {
    self.0.len()
  }

  /// Borrowed view of the underlying `f32` data. Cheap (no copy) and
  /// the standard input for downstream similarity / vector-store code
  /// that wants a `&[f32]`.
  pub fn as_slice(&self) -> &[f32] {
    &self.0
  }

  /// Returns the inner `Arc<[f32]>`. O(1) — atomic refcount only, no
  /// data copy. Callers who need a fresh `Vec<f32>` can write
  /// `embedding.into_inner().to_vec()` so the allocation is explicit.
  pub fn into_inner(self) -> Arc<[f32]> {
    self.0
  }

  /// Cosine similarity. Both operands must be unit-norm; valid because every
  /// `Embedding` in this crate is L2-normalized at construction.
  ///
  /// Returns [`crate::Error::EmbeddingDim`] when `self.dim() != other.dim()`
  /// or when `self.dim() == 0`.
  ///
  /// Internally dispatches through the crate-private SIMD layer — picks
  /// NEON on aarch64, AVX2+FMA on x86_64 (when the runtime CPU
  /// advertises both), or a four-accumulator scalar fallback on every
  /// other target.
  pub fn try_cosine(&self, other: &Embedding) -> Result<f32> {
    if self.dim() != other.dim() {
      return Err(Error::EmbeddingDim {
        expected: self.dim(),
        got: other.dim(),
      });
    }
    if self.dim() == 0 {
      return Err(Error::EmbeddingDim {
        expected: 1,
        got: 0,
      });
    }
    Ok(crate::simd::dot(self.as_slice(), other.as_slice()))
  }

  /// Normalize an arbitrary-length finite vector to unit L2 norm, rejecting a
  /// zero/non-finite norm. Not feature-gated (used by both the inference output
  /// path and the always-available `to_matryoshka`).
  fn normalized(data: &[f32]) -> Result<Self> {
    let norm = crate::simd::dot(data, data).sqrt();
    if !norm.is_finite() || norm == 0.0 {
      return Err(Error::NotNormalized {
        norm,
        epsilon: Self::NORM_EPSILON,
      });
    }
    let factor = 1.0 / norm;
    Ok(Self(data.iter().map(|&x| x * factor).collect()))
  }

  /// Crate-internal: build an `Embedding` from raw model output. The
  /// `embedding-gemma` ONNX export emits `sentence_embedding` that may
  /// or may not be L2-normalized depending on the optimum-export pipeline
  /// — we re-normalize unconditionally so downstream cosine code is
  /// always operating on unit-norm vectors. Rejection only happens for
  /// all-zero output (degenerate model state), or non-finite components.
  ///
  /// The `TryFrom<Vec<f32>>` path keeps the strict near-unit-norm check
  /// — that's for *caller-supplied* embeddings (e.g., deserialized from
  /// a vector store) which should already be unit-norm; silent renorm
  /// there would mask data corruption.
  #[cfg(feature = "inference")]
  pub(crate) fn from_model_output(data: &[f32]) -> Result<Self> {
    Self::normalized(data)
  }

  /// Prefix-truncate this embedding to `dim` dimensions and L2-renormalize —
  /// the supported way to get a 512/256/128-d Matryoshka vector from a 768-d
  /// base embedding (EmbeddingGemma is MRL-trained). Returns
  /// [`Error::EmbeddingDim`] if `dim` is 0 or greater than [`Self::dim`].
  pub fn to_matryoshka(&self, dim: usize) -> Result<Embedding> {
    if dim == 0 || dim > self.dim() {
      return Err(Error::EmbeddingDim {
        expected: self.dim(),
        got: dim,
      });
    }
    // A prefix of a unit vector isn't unit-norm; renormalize via the
    // private normalizing helper (which normalizes unconditionally).
    Self::normalized(&self.as_slice()[..dim])
  }
}

impl TryFrom<Vec<f32>> for Embedding {
  type Error = Error;

  /// Validates L2-norm (`Error::NotNormalized`, tolerance `NORM_EPSILON`).
  /// This path is for **caller-supplied** embeddings — typically deserialized
  /// from a vector store — that should already be unit-norm; we reject (rather
  /// than silently renormalize) so corruption can't slip through. Any non-empty
  /// length is accepted as long as the norm is near 1.0.
  ///
  /// Vectors whose `||v||₂` is within `NORM_EPSILON` of 1.0 are
  /// snapped to exactly 1.0 (in-place renorm preserves the cosine
  /// invariant under tiny f32 drift).
  fn try_from(mut v: Vec<f32>) -> Result<Self> {
    let norm_sq = crate::simd::dot(&v, &v);
    let norm = norm_sq.sqrt();
    if !norm.is_finite() || (norm - 1.0).abs() > Self::NORM_EPSILON {
      return Err(Error::NotNormalized {
        norm,
        epsilon: Self::NORM_EPSILON,
      });
    }
    let factor = 1.0 / norm;
    for x in &mut v {
      *x *= factor;
    }
    Ok(Self(v.into()))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn unit_vec(dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[0] = 1.0;
    v
  }

  #[test]
  fn try_from_accepts_unit_norm_768() {
    let v = unit_vec(768);
    let e = Embedding::try_from(v).expect("unit-norm 768-dim should succeed");
    assert_eq!(e.dim(), 768);
    let cos = e.try_cosine(&e).expect("happy path");
    assert!((cos - 1.0).abs() < 1e-5);
  }

  #[test]
  fn try_from_rejects_non_unit_norm() {
    let v = vec![0.5f32; 768];
    let err = Embedding::try_from(v).unwrap_err();
    match err {
      Error::NotNormalized { .. } => {}
      _ => panic!("expected NotNormalized, got {err}"),
    }
  }

  #[cfg(feature = "inference")]
  #[test]
  fn from_model_output_normalizes_arbitrary_norm() {
    let v = vec![1.0f32; 768];
    let e = Embedding::from_model_output(&v).expect("arbitrary-norm output must be normalized");
    let cos = e.try_cosine(&e).expect("happy path");
    assert!(
      (cos - 1.0).abs() < 1e-5,
      "post-norm cosine should be 1.0; got {cos}"
    );
    assert!((e.as_slice()[0] - (1.0 / (768.0_f32).sqrt())).abs() < 1e-6);
  }

  #[cfg(feature = "inference")]
  #[test]
  fn from_model_output_rejects_zero_norm() {
    let v = vec![0.0f32; 768];
    let err = Embedding::from_model_output(&v).unwrap_err();
    match err {
      Error::NotNormalized { norm, .. } => assert_eq!(norm, 0.0),
      _ => panic!("expected NotNormalized for zero output, got {err}"),
    }
  }

  #[cfg(feature = "inference")]
  #[test]
  fn from_model_output_rejects_nan_component() {
    let mut v = vec![0.5f32; 768];
    v[100] = f32::NAN;
    let err = Embedding::from_model_output(&v).unwrap_err();
    match err {
      Error::NotNormalized { norm, .. } => assert!(norm.is_nan()),
      _ => panic!("expected NotNormalized for NaN, got {err}"),
    }
  }

  #[test]
  fn try_from_renormalizes_within_tolerance() {
    let mut v = unit_vec(768);
    v[1] = Embedding::NORM_EPSILON / 2.0;
    let e = Embedding::try_from(v).expect("near-unit norm should be accepted");
    let dot = e.try_cosine(&e).expect("happy path");
    assert!(
      (dot - 1.0).abs() < 1e-5,
      "renormalized cosine should be 1.0; got {dot}"
    );
  }

  /// `try_cosine` must surface dim mismatches as `Error::EmbeddingDim`
  /// rather than panicking. Pins the contract that callers who want a
  /// panic-free surface can rely on it never panicking on dim differences.
  #[test]
  fn try_cosine_returns_dim_error_on_mismatch() {
    let a = Embedding(vec![1.0f32, 0.0].into());
    let b = Embedding(vec![1.0f32, 0.0, 0.0].into());
    let err = a
      .try_cosine(&b)
      .expect_err("dim mismatch must surface as Err");
    match err {
      Error::EmbeddingDim { expected, got } => {
        assert_eq!(expected, 2, "lhs dim");
        assert_eq!(got, 3, "rhs dim");
      }
      other => panic!("expected Error::EmbeddingDim, got {other}"),
    }
  }

  /// With runtime dimensions, two equal-length-4 vectors are valid for
  /// cosine — the old `try_into::<&[f32; 768]>` rejection no longer applies.
  #[test]
  fn try_cosine_works_for_small_equal_dim() {
    let a = Embedding(vec![1.0f32, 0.0, 0.0, 0.0].into());
    let b = Embedding(vec![0.0f32, 1.0, 0.0, 0.0].into());
    assert_eq!(a.try_cosine(&b).expect("equal-dim cosine ok"), 0.0);
  }

  /// Happy path: when both operands are valid 768-d unit vectors,
  /// `try_cosine` returns `Ok(_)` close to 1.0 for the self-pair.
  #[test]
  fn try_cosine_self_unit_pair() {
    let v = unit_vec(768);
    let e = Embedding::try_from(v).expect("unit-norm 768-d should succeed");
    let cos = e.try_cosine(&e).expect("happy path must be Ok");
    assert!((cos - 1.0).abs() < 1e-5);
  }

  /// `into_inner` exposes the storage `Arc<[f32]>` cheaply (no copy),
  /// and the inner slice round-trips through the renormalization
  /// performed by `try_from`. Replaces the old `into_vec_round_trips`
  /// test which exercised an API that was removed in favor of the
  /// allocation-free `into_inner`.
  #[test]
  fn into_inner_exposes_arc_unchanged() {
    let v = unit_vec(768);
    let e = Embedding::try_from(v).expect("unit-norm 768-d should succeed");
    let arc = e.into_inner();
    assert_eq!(arc.len(), 768);
    assert!((arc[0] - 1.0).abs() < 1e-6);
  }

  #[test]
  fn embedding_is_send_sync() {
    fn _req<T: Send + Sync>() {}
    _req::<Embedding>();
  }

  #[cfg(feature = "inference")]
  #[test]
  fn from_model_output_accepts_non_768_dims() {
    for len in [128usize, 256, 512, 768] {
      let v = unit_vec(len);
      let e = Embedding::from_model_output(&v).expect("any unit-norm len ok");
      assert_eq!(e.dim(), len);
      assert!((e.try_cosine(&e).unwrap() - 1.0).abs() < 1e-4);
    }
  }

  #[test]
  fn try_from_accepts_non_768_unit_norm() {
    let e = Embedding::try_from(unit_vec(256)).expect("256-d unit-norm ok");
    assert_eq!(e.dim(), 256);
  }

  #[cfg(feature = "inference")]
  #[test]
  fn from_model_output_rejects_empty() {
    let err = Embedding::from_model_output(&[]).unwrap_err();
    assert!(matches!(err, Error::NotNormalized { .. }));
  }

  #[cfg(feature = "inference")]
  #[test]
  fn to_matryoshka_truncates_and_renormalizes() {
    let e = Embedding::from_model_output(&unit_vec(768)).unwrap();
    let m = e.to_matryoshka(256).expect("truncate to 256");
    assert_eq!(m.dim(), 256);
    assert!((m.try_cosine(&m).unwrap() - 1.0).abs() < 1e-4);
  }

  #[cfg(feature = "inference")]
  #[test]
  fn to_matryoshka_rejects_zero_and_too_large() {
    let e = Embedding::from_model_output(&unit_vec(768)).unwrap();
    assert!(e.to_matryoshka(0).is_err());
    assert!(e.to_matryoshka(769).is_err());
  }

  #[test]
  fn to_matryoshka_works_without_inference() {
    // `to_matryoshka` is always-available (not inference-gated); build via
    // `try_from` so this path is covered under `--no-default-features`.
    let e = Embedding::try_from(unit_vec(768)).expect("768-d unit-norm");
    let m = e.to_matryoshka(256).expect("truncate to 256");
    assert_eq!(m.dim(), 256);
    assert!((m.try_cosine(&m).unwrap() - 1.0).abs() < 1e-4);
    assert!(e.to_matryoshka(0).is_err());
    assert!(e.to_matryoshka(769).is_err());
  }
}
