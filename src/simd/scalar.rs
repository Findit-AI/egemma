//! Always-compiled scalar reference implementation. Acts as the
//! fallback backend on targets without a SIMD path and as the
//! agreement baseline for the per-arch backends' tests.

/// Four-accumulator scalar dot product. The four independent reduction
/// chains let the compiler overlap multiplies/adds across iterations
/// even without SIMD intrinsics.
///
/// `#[allow(dead_code)]`: on aarch64 the dispatcher always picks NEON
/// (a baseline ISA feature), so this baseline is unreachable from
/// non-test builds on that target — but we keep it compiled as the
/// agreement reference for the per-arch backends' tests and as the
/// fallback path on non-aarch64 / non-x86_64 targets.
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn dot_768(a: &[f32; 768], b: &[f32; 768]) -> f32 {
  let mut acc = [0.0_f32; 4];
  let mut i = 0;
  while i < 768 {
    acc[0] += a[i] * b[i];
    acc[1] += a[i + 1] * b[i + 1];
    acc[2] += a[i + 2] * b[i + 2];
    acc[3] += a[i + 3] * b[i + 3];
    i += 4;
  }
  acc[0] + acc[1] + acc[2] + acc[3]
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn dot_orthogonal_unit_vectors_is_zero() {
    let mut a = Box::new([0.0f32; 768]);
    let mut b = Box::new([0.0f32; 768]);
    a[0] = 1.0;
    b[1] = 1.0;
    assert_eq!(dot_768(&a, &b), 0.0);
  }

  #[test]
  fn dot_self_unit_vector_is_one() {
    let mut a = Box::new([0.0f32; 768]);
    a[100] = 1.0;
    assert_eq!(dot_768(&a, &a), 1.0);
  }

  #[test]
  fn dot_constant_vectors_matches_known_sum() {
    let a = Box::new([0.5f32; 768]);
    let b = Box::new([0.25f32; 768]);
    // 768 × 0.5 × 0.25 = 96.0
    let got = dot_768(&a, &b);
    assert!((got - 96.0).abs() < 1e-4, "expected ≈96.0, got {got}");
  }
}
