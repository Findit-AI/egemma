//! x86_64 AVX2 + FMA backend. Selected by the dispatcher when both
//! `is_x86_feature_detected!("avx2")` and `…("fma")` succeed at
//! runtime. Carries `#[target_feature(enable = "avx2,fma")]` so the
//! intrinsics execute in an explicitly AVX2+FMA context.

use core::arch::x86_64::*;

/// 768-element f32 dot product using AVX2 256-bit registers + FMA.
///
/// 768 = 8 × 96 → 24 outer iterations × 4 independent FMA chains
/// (4 × 8 = 32 elements per iteration) = 96 FMAs total.
///
/// # Safety
///
/// AVX2 + FMA must be present (dispatcher-verified via
/// `is_x86_feature_detected!`). The 768-element length precondition is
/// encoded in the parameter type (`&[f32; 768]`), not asserted at
/// runtime — this is what makes the raw-pointer reads sound in release
/// builds where `debug_assert!`s would have been stripped.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_768_avx2_fma(a: &[f32; 768], b: &[f32; 768]) -> f32 {
  let mut acc0 = _mm256_setzero_ps();
  let mut acc1 = _mm256_setzero_ps();
  let mut acc2 = _mm256_setzero_ps();
  let mut acc3 = _mm256_setzero_ps();

  let pa = a.as_ptr();
  let pb = b.as_ptr();

  // 768 / 32 = 24 outer iterations, each loading 4 × 8-lane vectors
  // per operand into 4 parallel accumulators.
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: `i + 32 ≤ 768` each iteration; pa/pb point to fixed-size
    // arrays of length 768 by the parameter type. AVX2+FMA loads/FMAs
    // are sound under `#[target_feature(enable = "avx2,fma")]`.
    unsafe {
      let a0 = _mm256_loadu_ps(pa.add(i));
      let a1 = _mm256_loadu_ps(pa.add(i + 8));
      let a2 = _mm256_loadu_ps(pa.add(i + 16));
      let a3 = _mm256_loadu_ps(pa.add(i + 24));
      let b0 = _mm256_loadu_ps(pb.add(i));
      let b1 = _mm256_loadu_ps(pb.add(i + 8));
      let b2 = _mm256_loadu_ps(pb.add(i + 16));
      let b3 = _mm256_loadu_ps(pb.add(i + 24));
      acc0 = _mm256_fmadd_ps(a0, b0, acc0);
      acc1 = _mm256_fmadd_ps(a1, b1, acc1);
      acc2 = _mm256_fmadd_ps(a2, b2, acc2);
      acc3 = _mm256_fmadd_ps(a3, b3, acc3);
    }
    i += 32;
  }

  // Reduce: 4 vectors → 1 vector → scalar. These intrinsics are
  // pure lane/arithmetic ops (no memory access), so they are safe to
  // call inside an `unsafe fn` body without an inner `unsafe { ... }`
  // wrapper — the `target_feature(enable = "avx2,fma")` scope already
  // guarantees the SIMD context they need.
  let s01 = _mm256_add_ps(acc0, acc1);
  let s23 = _mm256_add_ps(acc2, acc3);
  let s = _mm256_add_ps(s01, s23);
  let lo = _mm256_castps256_ps128(s);
  let hi = _mm256_extractf128_ps(s, 1);
  let sum128 = _mm_add_ps(lo, hi);
  // sum128 = [a, b, c, d]; want a + b + c + d.
  let shuf = _mm_movehdup_ps(sum128); // [b, b, d, d]
  let sums = _mm_add_ps(sum128, shuf); // [a+b, _, c+d, _]
  let shuf2 = _mm_movehl_ps(sums, sums); // [c+d, …]
  let total = _mm_add_ss(sums, shuf2);
  _mm_cvtss_f32(total)
}

// Direct call to the unsafe AVX2+FMA kernel. Miri can't evaluate
// `_mm256_loadu_ps` / `_mm256_fmadd_ps`; under `cfg(miri)` the
// dispatcher routes through scalar, so the public API is still
// covered.
#[cfg(all(test, not(miri)))]
mod tests {
  use super::*;

  /// Returns `true` if AVX2+FMA are both detected on the current
  /// host; otherwise prints a `[SIMD-SKIP]` banner and returns
  /// `false`. Skip-not-panic: the dispatcher supports non-AVX2
  /// x86_64 via the scalar fallback (see `simd::dot_768_dispatch`),
  /// so panicking here would fail `cargo test` on a configuration
  /// the library handles. CI Linux x86_64 runners have AVX2+FMA,
  /// which is where the kernel coverage actually fires; the banner
  /// is grep-able to verify in CI logs.
  fn avx2_fma_available() -> bool {
    let ok =
      std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    if !ok {
      eprintln!(
        "[SIMD-SKIP] AVX2/FMA unavailable on this x86_64 host — direct kernel tests skipped. \
         The dispatcher's scalar fallback handles this configuration; CI Linux x86_64 runners \
         exercise the AVX2 kernel separately."
      );
    }
    ok
  }

  fn boxed_array(f: impl Fn(usize) -> f32) -> Box<[f32; 768]> {
    (0..768)
      .map(f)
      .collect::<Vec<_>>()
      .into_boxed_slice()
      .try_into()
      .expect("768 elements")
  }

  /// Compares the AVX2+FMA kernel against the scalar reference on
  /// the trigonometric fixture used by the cross-backend agreement
  /// test in `simd::tests`.
  #[test]
  fn agrees_with_scalar_within_tolerance() {
    if !avx2_fma_available() {
      return;
    }
    let a = boxed_array(|i| ((i as f32) * 0.013).sin());
    let b = boxed_array(|i| ((i as f32) * 0.017).cos());
    let s = crate::simd::scalar::dot_768(&a, &b);
    // SAFETY: AVX2+FMA asserted above; type-encoded length.
    let v = unsafe { dot_768_avx2_fma(&a, &b) };
    assert!((s - v).abs() < 1e-3, "avx2+fma ({v}) vs scalar ({s})",);
  }

  /// Orthogonal axis vectors → exact 0 dot product. Pin: SIMD
  /// summation order can't introduce drift on inputs that are
  /// identically zero outside one slot, so this checks the kernel's
  /// "no spurious accumulation" property bit-exactly (no tolerance).
  #[test]
  fn orthogonal_axes_dot_to_exact_zero() {
    if !avx2_fma_available() {
      return;
    }
    let mut a = Box::new([0.0f32; 768]);
    let mut b = Box::new([0.0f32; 768]);
    a[0] = 1.0;
    b[1] = 1.0;
    // SAFETY: AVX2+FMA asserted; type-encoded length.
    let v = unsafe { dot_768_avx2_fma(&a, &b) };
    assert_eq!(v, 0.0, "orthogonal e0·e1 must be exactly 0; got {v}");
  }

  /// Self-dot of a unit-norm vector → exactly 1.0. Same bit-exact
  /// reasoning as `orthogonal_axes_dot_to_exact_zero`: only one slot
  /// contributes, no FP error from summation ordering.
  #[test]
  fn unit_vector_self_dot_is_one() {
    if !avx2_fma_available() {
      return;
    }
    let mut a = Box::new([0.0f32; 768]);
    a[123] = 1.0;
    // SAFETY: AVX2+FMA asserted; type-encoded length.
    let v = unsafe { dot_768_avx2_fma(&a, &a) };
    assert_eq!(v, 1.0, "unit-vector self-dot must be exactly 1.0; got {v}");
  }

  /// Constant-vector dot product: 768 × c × d. Catches FMA
  /// accumulation bugs across the four chains (any chain
  /// missing or double-counted lanes shows up here).
  #[test]
  fn constant_vectors_match_known_sum() {
    if !avx2_fma_available() {
      return;
    }
    let a = Box::new([0.5f32; 768]);
    let b = Box::new([0.25f32; 768]);
    // 768 * 0.5 * 0.25 = 96.0
    // SAFETY: AVX2+FMA asserted; type-encoded length.
    let v = unsafe { dot_768_avx2_fma(&a, &b) };
    assert!(
      (v - 96.0).abs() < 1e-4,
      "expected 96.0 from 768·0.5·0.25; got {v}",
    );
  }

  /// Alternating-sign fixture: catches a class of reduction bugs
  /// where a chain accidentally swaps subtract/add or where signs
  /// drop during horizontal reduce. Also widens the tolerance check
  /// past the all-positive trigonometric case.
  #[test]
  fn alternating_sign_agrees_with_scalar() {
    if !avx2_fma_available() {
      return;
    }
    let a = boxed_array(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
    let b = boxed_array(|i| if i % 3 == 0 { 1.0 } else { -1.0 });
    let s = crate::simd::scalar::dot_768(&a, &b);
    // SAFETY: AVX2+FMA asserted; type-encoded length.
    let v = unsafe { dot_768_avx2_fma(&a, &b) };
    assert!(
      (s - v).abs() < 1e-3,
      "alternating-sign avx2 ({v}) vs scalar ({s})",
    );
  }
}
