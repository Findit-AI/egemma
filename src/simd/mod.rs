//! Crate-internal SIMD primitives. Only one operation is hot enough to
//! be worth hand-vectorizing: the 768-element f32 dot product
//! ([`Embedding::try_cosine`], `||v||²` during normalization). Pointwise
//! scales and integer widenings auto-vectorize under `-O3`, so they
//! stay in scalar form.
//!
//! Backends:
//! - `scalar` — always compiled, reference implementation.
//! - `neon` — aarch64 NEON + FMA. NEON is baseline on aarch64, so the
//!   dispatcher invokes it unconditionally on that target.
//! - `x86` — x86_64 AVX2 + FMA. Selected at runtime when
//!   `is_x86_feature_detected!("avx2")` and `…("fma")` both succeed.
//!
//! Numerical contract: SIMD backends are not byte-identical to scalar
//! (different summation order changes f32 rounding) but agree within
//! `1e-3` absolute for `dot_768` on unit-norm 768-d vectors. Tests in
//! each backend module enforce this.
//!
//! Safety boundary: `dot_768` takes `&[f32; 768]` rather than `&[f32]`.
//! The unsafe per-arch kernels read exactly 768 elements via raw
//! pointer offsets, and the type-level length invariant is what makes
//! that read sound. A `&[f32]`-typed parameter would only be checked
//! by `debug_assert!`, which is stripped in release — the type-level
//! version eliminates that release-mode footgun by construction.
//!
//! Miri escape hatch: every per-arch dispatcher short-circuits to
//! scalar under `cfg!(miri)`. Miri cannot evaluate target-specific
//! LLVM intrinsics (`vfmaq_f32`, `_mm256_fmadd_ps`, …) and would
//! abort with "unsupported operation: can't call foreign function"
//! the moment a normal test went through `Embedding::try_cosine`.
//! Routing through scalar lets the Miri matrix exercise the same
//! call sites as native CI — and validate the *unsafe-free* path —
//! without ever entering the SIMD backends. The per-arch backend
//! tests that call the unsafe kernels directly are gated with
//! `#[cfg(not(miri))]` for the same reason.

pub(crate) mod scalar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

/// Dispatch to the best available 768-d f32 dot product. The fixed-size
/// array parameter is the safety contract: the unsafe per-arch backends
/// rely on exactly 768 elements being readable.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn dot_768(a: &[f32; 768], b: &[f32; 768]) -> f32 {
  dot_768_dispatch(a, b)
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn dot_768_dispatch(a: &[f32; 768], b: &[f32; 768]) -> f32 {
  // Miri can't evaluate `vfmaq_f32` / `vld1q_f32` and would abort with
  // "unsupported operation: can't call foreign function" — see the
  // module-level docstring. Route through scalar so Miri-driven jobs
  // still exercise `Embedding::try_cosine` and the surrounding logic.
  if cfg!(miri) {
    return scalar::dot_768(a, b);
  }
  // SAFETY: NEON is a baseline aarch64 feature — every aarch64 CPU has
  // it. The 768-element precondition is encoded in the parameter type.
  unsafe { neon::dot_768(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn dot_768_dispatch(a: &[f32; 768], b: &[f32; 768]) -> f32 {
  // Same Miri reasoning as the aarch64 path — bypass the AVX2+FMA
  // intrinsics under Miri.
  if cfg!(miri) {
    return scalar::dot_768(a, b);
  }
  // `is_x86_feature_detected!` caches its result behind an atomic, so
  // the per-call cost is a relaxed load + branch.
  if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
    // SAFETY: feature detection above; 768-element precondition encoded
    // in the parameter type.
    unsafe { x86::dot_768_avx2_fma(a, b) }
  } else {
    scalar::dot_768(a, b)
  }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn dot_768_dispatch(a: &[f32; 768], b: &[f32; 768]) -> f32 {
  scalar::dot_768(a, b)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn fixture() -> (Box<[f32; 768]>, Box<[f32; 768]>) {
    let a: Box<[f32; 768]> = (0..768)
      .map(|i| ((i as f32) * 0.013).sin())
      .collect::<Vec<_>>()
      .into_boxed_slice()
      .try_into()
      .unwrap();
    let b: Box<[f32; 768]> = (0..768)
      .map(|i| ((i as f32) * 0.017).cos())
      .collect::<Vec<_>>()
      .into_boxed_slice()
      .try_into()
      .unwrap();
    (a, b)
  }

  #[test]
  fn dispatch_agrees_with_scalar_within_tolerance() {
    let (a, b) = fixture();
    let s = scalar::dot_768(&a, &b);
    let d = dot_768(&a, &b);
    assert!(
      (s - d).abs() < 1e-3,
      "dispatch dot ({d}) disagrees with scalar ({s})"
    );
  }

  #[test]
  fn dispatch_zero_for_orthogonal_axes() {
    // e_0 vs e_1 → exactly 0 in both scalar and SIMD (no rounding).
    let mut a = Box::new([0.0f32; 768]);
    let mut b = Box::new([0.0f32; 768]);
    a[0] = 1.0;
    b[1] = 1.0;
    assert_eq!(dot_768(&a, &b), 0.0);
  }

  /// Short slices can never reach the SIMD boundary: the parameter
  /// type `&[f32; 768]` rejects them at compile time. This test
  /// documents the conversion-site failure mode that callers see when
  /// they pass a wrong-length slice — the SIMD backends never get a
  /// chance to read OOB.
  #[test]
  fn short_slice_cannot_be_converted_to_768_array() {
    let v = vec![0.0f32; 100];
    let arr: Result<&[f32; 768], _> = v.as_slice().try_into();
    assert!(
      arr.is_err(),
      "100-element slice must not convert to [f32; 768]"
    );
  }
}
