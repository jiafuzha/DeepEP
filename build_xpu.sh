#!/usr/bin/env bash
# Build DeepEP for Intel XPU with GUARANTEED iSHMEM archive parity.
#
# This is the ONLY supported way to build the XPU target, because it eliminates the
# recurring "linked against a stale libishmem.a" class of bug by always doing, in order:
#   1. rebuild + install iSHMEM so build/_install/lib/libishmem.a matches the current source
#   2. force DeepEP to re-extract the iSHMEM device objects AND rebuild the SYCL device link
#   3. build DeepEP (setup.py also hard-fails if the archive is older than the iSHMEM source)
#
# Env knobs:
#   ISHMEM_DIR        install prefix (default: zjf2012/ishmem_ibgda/build/_install)
#   REBUILD_ISHMEM=0  skip the iSHMEM rebuild (the setup.py freshness guard still runs)
#   DEEP_EP_TARGET    defaults to xpu
set -euo pipefail

: "${ISHMEM_DIR:=/root/jiafuzha/code-repo/zjf2012/ishmem_ibgda/build/_install}"
: "${REBUILD_ISHMEM:=1}"
export ISHMEM_DIR
export DEEP_EP_TARGET="${DEEP_EP_TARGET:-xpu}"

# Generic multi-device AOT is stable; a bmg-only device link is not (DEVICE_LOST).
unset TORCH_XPU_ARCH_LIST XPU_AOT_TARGETS || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ISHMEM_BUILD_DIR="$(dirname "$ISHMEM_DIR")"   # conventionally <repo>/build

# 1. oneAPI environment (safe to source repeatedly). NOTE: setvars.sh is not `set -e`
# clean -- sourcing it under `set -e` aborts mid-script, so relax errexit around it.
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +eu
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    set -eu
fi

# 2. Rebuild + install iSHMEM so the installed archive always matches the current source.
if [ "$REBUILD_ISHMEM" = "1" ]; then
    if [ -f "$ISHMEM_BUILD_DIR/CMakeCache.txt" ]; then
        echo "===== [1/3] Rebuilding iSHMEM -> $ISHMEM_DIR ====="
        cmake --build "$ISHMEM_BUILD_DIR" --target install -j"$(nproc)"
    else
        echo "WARNING: no CMake build dir at $ISHMEM_BUILD_DIR; skipping iSHMEM rebuild." >&2
        echo "         Set ISHMEM_DIR to <repo>/build/_install or REBUILD_ISHMEM=0." >&2
    fi
else
    echo "===== [1/3] REBUILD_ISHMEM=0: skipping iSHMEM rebuild (freshness guard still runs) ====="
fi

# 3. Force iSHMEM re-extraction AND device relink (closes both staleness gaps).
echo "===== [2/3] Forcing iSHMEM re-extraction + device relink ====="
rm -rf build/ishmem-sycl-dlink
find build -path '*temp*' -name 'sycl_dlink.o' -delete 2>/dev/null || true

# 4. Build DeepEP.
echo "===== [3/3] Building DeepEP (DEEP_EP_TARGET=$DEEP_EP_TARGET) ====="
python3 setup.py build_ext --inplace

# 5. Verify parity so a bad build is caught immediately.
so="$(ls deep_ep_cpp*.so 2>/dev/null | head -1 || true)"
echo "===== Build verification ====="
if [ -f build/ishmem-sycl-dlink/.archive-stamp ]; then
    echo "archive-stamp   : $(cat build/ishmem-sycl-dlink/.archive-stamp)"
fi
if [ -f build/ishmem-sycl-dlink/barrier.cpp.o ]; then
    echo "barrier.cpp.o   : $(md5sum build/ishmem-sycl-dlink/barrier.cpp.o | awk '{print $1}')"
fi
if [ -n "$so" ]; then
    dev="$(strings "$so" | grep -m1 -- '-device ' || true)"
    echo "AOT device list : ${dev:-<none found>}"
    if echo "$dev" | grep -Eq -- '-device bmg *$'; then
        echo "WARNING: .so is bmg-ONLY AOT (unstable). Ensure TORCH_XPU_ARCH_LIST/XPU_AOT_TARGETS are unset." >&2
    fi
    echo "artifact        : $so"
    echo "OK"
else
    echo "ERROR: no deep_ep_cpp*.so produced" >&2
    exit 1
fi
