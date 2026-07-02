#!/usr/bin/env bash
# Build the iSHMEM auto-NIC-selection check binary (nic_pcie_check).
# Mirrors the icpx/iSHMEM link recipe of the coherence UT build script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/nic_pcie_check.cpp"
OUT="$HERE/nic_pcie_check"

ISHMEM_DIR="${ISHMEM_DIR:-/opt/intel/ishmem}"
if [ -f "$ISHMEM_DIR/lib/pkgconfig/ishmem.pc" ]; then
    export PKG_CONFIG_PATH="$ISHMEM_DIR/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

ISHMEM_CFLAGS="$(pkg-config --cflags ishmem)"
ISHMEM_LIBS="$(pkg-config --libs ishmem)"

icpx -std=c++20 -O2 -fsycl -fsycl-rdc -fsycl-targets=spir64 \
    $ISHMEM_CFLAGS \
    "$SRC" \
    -o "$OUT" \
    -lze_loader $ISHMEM_LIBS -lhwloc

echo "$OUT"
