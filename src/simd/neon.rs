//! aarch64 NEON backend — selected unconditionally on aarch64 (NEON is
//! a baseline feature). The kernel carries
//! `#[target_feature(enable = "neon")]` so its intrinsics execute in
//! an explicitly NEON-enabled context rather than one merely inherited
//! from the aarch64 target's default features.

use core::arch::aarch64::*;

/// 768-element f32 dot product using NEON FMA.
///
/// Four parallel 4-lane accumulators (16 lanes total). Each
/// `vfmaq_f32` multiplies 4 f32s and adds into the accumulator, fully
/// pipelined across the four chains.
///
/// # Safety
///
/// NEON must be available — guaranteed on aarch64 by the ISA, but we
/// keep the `target_feature` annotation so the call site is explicitly
/// typed as a NEON context. The 768-element length precondition is
/// encoded in the parameter type (`&[f32; 768]`), not asserted at
/// runtime — this is what makes the raw-pointer reads sound in
/// release builds where `debug_assert!`s would have been stripped.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot_768(a: &[f32; 768], b: &[f32; 768]) -> f32 {
  let mut acc0 = vdupq_n_f32(0.0);
  let mut acc1 = vdupq_n_f32(0.0);
  let mut acc2 = vdupq_n_f32(0.0);
  let mut acc3 = vdupq_n_f32(0.0);

  let pa = a.as_ptr();
  let pb = b.as_ptr();

  // 768 / 16 = 48 outer iterations: 4 × 4-lane FMAs across 4
  // independent dependency chains = 192 FMAs total.
  let mut i = 0usize;
  while i < 768 {
    // SAFETY: `i + 16 ≤ 768` each iteration; pa/pb point to fixed-size
    // arrays of length 768 by the parameter type. NEON loads/FMAs are
    // sound under `#[target_feature(enable = "neon")]`.
    unsafe {
      let a0 = vld1q_f32(pa.add(i));
      let a1 = vld1q_f32(pa.add(i + 4));
      let a2 = vld1q_f32(pa.add(i + 8));
      let a3 = vld1q_f32(pa.add(i + 12));
      let b0 = vld1q_f32(pb.add(i));
      let b1 = vld1q_f32(pb.add(i + 4));
      let b2 = vld1q_f32(pb.add(i + 8));
      let b3 = vld1q_f32(pb.add(i + 12));
      acc0 = vfmaq_f32(acc0, a0, b0);
      acc1 = vfmaq_f32(acc1, a1, b1);
      acc2 = vfmaq_f32(acc2, a2, b2);
      acc3 = vfmaq_f32(acc3, a3, b3);
    }
    i += 16;
  }

  // Pairwise reduce 4 vectors → 1 vector → scalar.
  let s01 = vaddq_f32(acc0, acc1);
  let s23 = vaddq_f32(acc2, acc3);
  let s = vaddq_f32(s01, s23);
  vaddvq_f32(s)
}

// Direct calls to the unsafe NEON kernel. Miri can't evaluate the
// `vfmaq_f32` / `vld1q_f32` intrinsics; the same coverage of the
// public API (via `Embedding::cosine`) under Miri is provided by the
// scalar fallback the dispatcher routes to under `cfg!(miri)`.
#[cfg(all(test, not(miri)))]
mod tests {
  use super::*;

  #[test]
  fn agrees_with_scalar_within_tolerance() {
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
    // SAFETY: NEON is baseline on aarch64; length is type-encoded.
    let n = unsafe { dot_768(&a, &b) };
    assert!((s - n).abs() < 1e-3, "neon ({n}) vs scalar ({s})");
  }

  #[test]
  fn unit_vector_self_dot_is_one() {
    let mut a = Box::new([0.0f32; 768]);
    a[42] = 1.0;
    // SAFETY: NEON baseline; type-encoded length.
    let got = unsafe { dot_768(&a, &a) };
    assert_eq!(got, 1.0);
  }
}
