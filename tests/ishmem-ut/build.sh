#!/bin/bash
# Build the standalone iSHMEM device-PUT reproducer (ll_put_repro.cpp).
#
# Selects the iSHMEM install via ISHMEM_DIR so you can build against either the
# known-good or the suspect iSHMEM and compare:
#   ISHMEM_DIR=/root/jiafuzha/code-repo/zjf2012/ishmem_ibgda/build/_install  (good)
#   ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install                    (broken)
#
# Mirrors the SYCL flags DeepEP's setup.py uses for the XPU extension so the
# device code is generated/linked the same way.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ISHMEM_DIR:=/root/jiafuzha/code-repo/zjf2012/ishmem_ibgda/build/_install}"

# Source oneAPI BEFORE enabling strict mode: setvars.sh references unbound vars
# and returns non-zero, which would trip `set -euo pipefail`.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true

set -euo pipefail

if [ ! -f "$ISHMEM_DIR/lib/libishmem.a" ]; then
    echo "ERROR: libishmem.a not found under ISHMEM_DIR=$ISHMEM_DIR" >&2
    exit 1
fi

export PKG_CONFIG_PATH="$ISHMEM_DIR/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
ISHMEM_CFLAGS="$(pkg-config --cflags ishmem)"
ISHMEM_LIBS="$(pkg-config --libs ishmem)"

OUT="$SCRIPT_DIR/ll_put_repro"
SRC="$SCRIPT_DIR/ll_put_repro.cpp"

echo "===== Building ll_put_repro against ISHMEM_DIR=$ISHMEM_DIR ====="
set -x
icpx -O3 -std=c++17 -fsycl -fsycl-rdc -fsycl-targets=spir64 \
    $ISHMEM_CFLAGS \
    "$SRC" -o "$OUT" \
    $ISHMEM_LIBS -lze_loader -lhwloc -libverbs
set +x
echo "===== Built $OUT ====="
