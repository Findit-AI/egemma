#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Error: TARGET is not provided"
  exit 1
fi

TARGET="$1"

# Install cross-compilation toolchain on Linux
if [ "$(uname)" = "Linux" ]; then
  case "$TARGET" in
    aarch64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-aarch64-linux-gnu
      ;;
    i686-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-multilib
      ;;
    powerpc64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-powerpc64-linux-gnu
      ;;
    s390x-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-s390x-linux-gnu
      ;;
    riscv64gc-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-riscv64-linux-gnu
      ;;
  esac
fi

rustup toolchain install nightly --component miri
rustup override set nightly
cargo miri setup

export MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-disable-isolation -Zmiri-symbolic-alignment-check -Zmiri-tree-borrows"

# Miri can't evaluate the FFI in `ort` / `tokenizers` (the `inference`
# default-feature dependencies), and most of the matrix targets
# (powerpc64, s390x, riscv64, i686) have no ort prebuilds. With
# `--no-default-features` Miri exercises the embedding + options + simd
# subset it actually validates: the SIMD dispatcher routes through the
# scalar fallback under `cfg!(miri)`, so the unsafe NEON / AVX2 kernel
# boundaries are indirectly covered through `Embedding::try_cosine`
# without ever entering the platform intrinsics Miri can't model.
cargo miri test --all-targets --no-default-features --target "$TARGET"
