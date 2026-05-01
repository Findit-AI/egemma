//! `Embedding` — L2-normalized 768-dim sentence embedding.

use std::sync::Arc;

use crate::error::{Error, Result};

/// L2-normalized embedding. Length is `EMBED_DIM` (768) in 0.1.0.
///
/// `Embedding` deliberately does **not** implement `Serialize` or `Deserialize`.
/// An auto-derived `Deserialize` would bypass the dim and L2-norm invariants
/// that `TryFrom<Vec<f32>>` exists to enforce. Round-trip via the inner
/// representation:
///
/// ```ignore
/// // Serialize via the inner slice (`&[f32]: Serialize`):
/// let json = serde_json::to_string(embedding.as_slice())?;
///
/// // Deserialize via the validated path:
/// let v: Vec<f32> = serde_json::from_str(&json)?;
/// let embedding  = Embedding::try_from(v)?;  // validates dim + L2-norm
/// ```
#[derive(Clone, Debug)]
pub struct Embedding(Arc<[f32]>);

impl Embedding {
  /// 0.1.0 supports only the 768-dim base export.
  pub const EMBED_DIM: usize = 768;

  /// L2-norm tolerance for the unit-norm invariant.
  pub const NORM_EPSILON: f32 = 5e-4;

  pub fn dim(&self) -> usize {
    self.0.len()
  }

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
  /// or when either operand's dim doesn't equal [`Self::EMBED_DIM`]. In
  /// 0.1.0 every public constructor (`try_from`, `from_model_output` via
  /// `TextEncoder`) produces a 768-d `Embedding`, so the error path is
  /// only reachable in-crate; the check is forward-compatibility for
  /// variable-dim embeddings and a guard against future internal misuse.
  ///
  /// Internally dispatches through [`crate::simd::dot_768`] — picks NEON
  /// on aarch64, AVX2+FMA on x86_64 (when the runtime CPU advertises
  /// both), or a four-accumulator scalar fallback on every other target.
  pub fn try_cosine(&self, other: &Embedding) -> Result<f32> {
    if self.dim() != other.dim() {
      return Err(Error::EmbeddingDim {
        expected: self.dim(),
        got: other.dim(),
      });
    }
    let a: &[f32; Self::EMBED_DIM] =
      self
        .as_slice()
        .try_into()
        .map_err(|_| Error::EmbeddingDim {
          expected: Self::EMBED_DIM,
          got: self.dim(),
        })?;
    let b: &[f32; Self::EMBED_DIM] =
      other
        .as_slice()
        .try_into()
        .map_err(|_| Error::EmbeddingDim {
          expected: Self::EMBED_DIM,
          got: other.dim(),
        })?;
    Ok(crate::simd::dot_768(a, b))
  }

  /// Crate-internal: build an `Embedding` from raw model output. The
  /// `embedding-gemma` ONNX export emits `sentence_embedding` that may
  /// or may not be L2-normalized depending on the optimum-export pipeline
  /// — we re-normalize unconditionally so downstream cosine code is
  /// always operating on unit-norm vectors. Rejection only happens for
  /// dim mismatch, all-zero output (degenerate model state), or
  /// non-finite components.
  ///
  /// The `TryFrom<Vec<f32>>` path keeps the strict near-unit-norm check
  /// — that's for *caller-supplied* embeddings (e.g., deserialized from
  /// a vector store) which should already be unit-norm; silent renorm
  /// there would mask data corruption.
  #[cfg(feature = "inference")]
  pub(crate) fn from_model_output(data: &[f32]) -> Result<Self> {
    let arr: &[f32; Self::EMBED_DIM] = data.try_into().map_err(|_| Error::EmbeddingDim {
      expected: Self::EMBED_DIM,
      got: data.len(),
    })?;
    let norm_sq = crate::simd::dot_768(arr, arr);
    let norm = norm_sq.sqrt();
    if !norm.is_finite() || norm == 0.0 {
      return Err(Error::NotNormalized {
        norm,
        epsilon: Self::NORM_EPSILON,
      });
    }
    let factor = 1.0 / norm;
    let arc: Arc<[f32]> = data.iter().map(|&x| x * factor).collect();
    Ok(Self(arc))
  }
}

impl TryFrom<Vec<f32>> for Embedding {
  type Error = Error;

  /// Validates dim (`Error::EmbeddingDim`) and L2-norm
  /// (`Error::NotNormalized`, tolerance `NORM_EPSILON`). This path is for
  /// **caller-supplied** embeddings — typically deserialized from a
  /// vector store — that should already be unit-norm; we reject (rather
  /// than silently renormalize) so corruption can't slip through.
  ///
  /// Vectors whose `||v||₂` is within `NORM_EPSILON` of 1.0 are
  /// snapped to exactly 1.0 (in-place renorm preserves the cosine
  /// invariant under tiny f32 drift).
  fn try_from(mut v: Vec<f32>) -> Result<Self> {
    let norm_sq = {
      let arr: &[f32; Self::EMBED_DIM] =
        v.as_slice().try_into().map_err(|_| Error::EmbeddingDim {
          expected: Self::EMBED_DIM,
          got: v.len(),
        })?;
      crate::simd::dot_768(arr, arr)
    };
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
  fn try_from_rejects_wrong_dim() {
    let v = vec![0.0; 100];
    let err = Embedding::try_from(v).unwrap_err();
    match err {
      Error::EmbeddingDim { expected, got } => {
        assert_eq!(expected, 768);
        assert_eq!(got, 100);
      }
      _ => panic!("expected EmbeddingDim, got {err}"),
    }
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

  /// The SIMD boundary takes `&[f32; 768]`, so a wrong-length slice
  /// can never reach the unsafe kernels — `from_model_output` rejects
  /// it at the conversion site with `Error::EmbeddingDim`. This test
  /// pins the rejection path so a future refactor that re-loosens the
  /// signature back to `&[f32]` would surface as a unit-test failure.
  #[cfg(feature = "inference")]
  #[test]
  fn from_model_output_rejects_wrong_dim() {
    let v = vec![0.5f32; 100];
    let err = Embedding::from_model_output(&v).unwrap_err();
    match err {
      Error::EmbeddingDim { expected, got } => {
        assert_eq!(expected, 768);
        assert_eq!(got, 100);
      }
      _ => panic!("expected EmbeddingDim, got {err}"),
    }
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

  /// `try_cosine` must also reject same-dim-but-non-768 pairs (the
  /// `try_into::<&[f32; EMBED_DIM]>` failure path inside the kernel
  /// boundary). This pair has matching dims (both 4), so the
  /// dim-equality check passes, but the typed-array conversion still
  /// fails — and `try_cosine` translates that into `EmbeddingDim`
  /// rather than panicking.
  #[test]
  fn try_cosine_returns_dim_error_when_both_wrong_size() {
    let a = Embedding(vec![1.0f32, 0.0, 0.0, 0.0].into());
    let b = Embedding(vec![0.0f32, 1.0, 0.0, 0.0].into());
    let err = a
      .try_cosine(&b)
      .expect_err("non-EMBED_DIM operands must error");
    match err {
      Error::EmbeddingDim { expected, got } => {
        assert_eq!(expected, Embedding::EMBED_DIM);
        assert_eq!(got, 4);
      }
      other => panic!("expected Error::EmbeddingDim, got {other}"),
    }
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
}
