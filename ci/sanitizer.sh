#!/bin/bash
set -ex

export ASAN_OPTIONS="detect_odr_violation=0 detect_leaks=0"

TARGET="x86_64-unknown-linux-gnu"

# Sanitizer feature set: matches the coverage job. `--all-features` is
# unsafe here — it would activate the opt-in execution-provider features
# (`cuda`, `tensorrt`, `directml`, `rocm`, `coreml`), each of which
# requires the corresponding vendor SDK to compile. Stock GitHub
# runners don't have any of them, so `--all-features` would fail in
# `ort-sys`'s build script before the unsafe SIMD code is ever
# instrumented. EP-specific sanitizer coverage belongs on separately
# provisioned runners.
FEATURES="inference,serde"

# Run address sanitizer
RUSTFLAGS="-Z sanitizer=address" \
cargo test --tests --target "$TARGET" --no-default-features --features "$FEATURES"

# Run leak sanitizer
RUSTFLAGS="-Z sanitizer=leak" \
cargo test --tests --target "$TARGET" --no-default-features --features "$FEATURES"

# Run memory sanitizer (requires -Zbuild-std for instrumented std)
RUSTFLAGS="-Z sanitizer=memory" \
cargo -Zbuild-std test --tests --target "$TARGET" --no-default-features --features "$FEATURES"

# Run thread sanitizer (requires -Zbuild-std for instrumented std)
RUSTFLAGS="-Z sanitizer=thread" \
cargo -Zbuild-std test --tests --target "$TARGET" --no-default-features --features "$FEATURES"
