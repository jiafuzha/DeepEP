#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISHMEM_DIR="${ISHMEM_DIR:-/opt/intel/ishmem}"
BUILD_DIR="$ROOT_DIR/build/tests/ut"
OUT="$BUILD_DIR/xpu_ishmem_mapped_api_ut"

mkdir -p "$BUILD_DIR"

if [ -f "$ISHMEM_DIR/lib/pkgconfig/ishmem.pc" ]; then
    export PKG_CONFIG_PATH="$ISHMEM_DIR/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

ISHMEM_CFLAGS="$(pkg-config --cflags ishmem)"
ISHMEM_LIBS="$(pkg-config --libs ishmem)"

icpx -std=c++20 -O2 -fsycl -fsycl-rdc -fsycl-targets=spir64 \
    $ISHMEM_CFLAGS \
    "$ROOT_DIR/tests/ut/xpu_ishmem_mapped_api_ut.cpp" \
    -o "$OUT" \
    -lze_loader $ISHMEM_LIBS

echo "$OUT"
