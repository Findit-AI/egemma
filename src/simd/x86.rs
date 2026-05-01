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

  #[test]
  fn agrees_with_scalar_within_tolerance() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
      eprintln!("skipping: AVX2/FMA not available on this host");
      return;
    }
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
    let s = crate::simd::scalar::dot_768(&a, &b);
    // SAFETY: AVX2+FMA detected above; type-encoded length.
    let v = unsafe { dot_768_avx2_fma(&a, &b) };
    assert!((s - v).abs() < 1e-3, "avx2+fma ({v}) vs scalar ({s})");
  }
}
