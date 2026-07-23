#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# FAITHFUL reproducer driver: NamedBarrier vs the REAL iSHMEM device library,
# device-linked with -fsycl-rdc exactly like DeepEP's setup.py.
#
# Must be run inside the DeepEP build container (oneAPI 2025.3 + the iSHMEM
# archive that DeepEP links), e.g.:
#
#   docker run --rm --device /dev/dri \
#     -v $PWD/tests/repro-ishmem-named-barrier:/repro \
#     -v /root/jiafuzha/ishmem_ibgda:/root/jiafuzha/ishmem_ibgda \
#     deepep_jiafuzha bash -lc \
#     'source /opt/intel/oneapi/setvars.sh --force; cd /repro; ./run_ishmem_real.sh'
#
# Runs three configurations:
#   [A] control    : named barrier only, NO iSHMEM RDMA path        -> expect PASS
#   [B] repro      : named barrier + real iSHMEM IBGDA device path   -> expect FAIL
#                    (vISA: "More than 1 kernel attribute defined NBarrierCnt")
#   [C] workaround : same as [B] + IGC_FunctionControl=1 force-inline-> expect PASS
#                    (compiles, but ~+32% latency in the real LL kernel)
#
# The NBarrierCnt error is a JIT-compile error (surfaces at first kernel launch,
# before the kernel body runs); its exact text is written by IGC to
# `kernel.errors.txt` in the working directory, which this script prints.
# -----------------------------------------------------------------------------
set -u
cd "$(dirname "$0")"
SRC="$(pwd)/repro_ishmem_real.cpp"

CXX=${CXX:-icpx}
ISHMEM_DIR=${ISHMEM_DIR:-/root/jiafuzha/ishmem_ibgda/build/_install}
export PKG_CONFIG_PATH="${ISHMEM_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

if ! pkg-config --exists ishmem; then
  echo "ERROR: pkg-config cannot find 'ishmem' (ISHMEM_DIR=$ISHMEM_DIR)."
  echo "Run this INSIDE the DeepEP build container with the iSHMEM install mounted."
  exit 2
fi
ISHMEM_CFLAGS=$(pkg-config --cflags ishmem)
ISHMEM_LIBS=$(pkg-config --libs ishmem)

# -fsycl-rdc + spir64: JIT, same device-link shape DeepEP uses (setup.py:110).
COMMON=(-std=c++17 -O2 -fsycl -fsycl-rdc -fsycl-targets=spir64)

# Returns: 0 = built + ran clean, 2 = reproduced NBarrierCnt vISA failure, 3 = other build fail.
build_run() {
  local label="$1" def="$2" envp="$3"
  local rundir; rundir=$(mktemp -d)   # kernel.errors.txt lands here
  local bin="$rundir/repro_ishmem_${label}"
  echo "======================================================================"
  echo "[$label] building ($def) ${envp:+env=$envp}"
  # shellcheck disable=SC2086
  "$CXX" "${COMMON[@]}" $ISHMEM_CFLAGS $def "$SRC" -o "$bin" $ISHMEM_LIBS > "$rundir/build.log" 2>&1
  if [[ $? -ne 0 ]]; then
    echo "[$label] BUILD FAILED:"; tail -8 "$rundir/build.log" | sed 's/^/    /'
    rm -rf "$rundir"; return 3
  fi
  echo "[$label] BUILD OK; running (JIT compiles at first launch)..."
  # shellcheck disable=SC2086
  ( cd "$rundir" && timeout 90 env ONEAPI_DEVICE_SELECTOR=level_zero:gpu $envp "$bin" ) \
      > "$rundir/run.log" 2>&1
  local rc=$?
  sed 's/^/    /' "$rundir/run.log"
  local errfile; errfile=$(find "$rundir" -name kernel.errors.txt 2>/dev/null | head -1)
  if [[ -n "$errfile" ]] && grep -qi NBarrierCnt "$errfile"; then
    echo "[$label] --- IGC kernel.errors.txt ---"
    grep -iE "CISA routine|NBarrierCnt" "$errfile" | head -2 | sed 's/^/    /'
    echo "[$label] REPRODUCED: 'More than 1 kernel attribute defined NBarrierCnt'."
    rm -rf "$rundir"; return 2
  fi
  if grep -qi "parsing vISA inline assembly failed" "$rundir/run.log"; then
    echo "[$label] REPRODUCED: vISA parse failure at JIT (NBarrierCnt class)."
    rm -rf "$rundir"; return 2
  fi
  echo "[$label] run rc=$rc"
  rm -rf "$rundir"; return $rc
}

echo "### [A] control: named barrier only (expect PASS)"
build_run A "-DUSE_ISHMEM=0" ""; ra=$?
echo
echo "### [B] repro: named barrier + REAL iSHMEM IBGDA path (expect NBarrierCnt FAIL)"
build_run B "-DUSE_ISHMEM=1" ""; rb=$?
echo
echo "### [C] workaround: [B] + IGC_FunctionControl=1 force-inline (expect PASS)"
build_run C "-DUSE_ISHMEM=1" "IGC_FunctionControl=1"; rc_=$?

echo
echo "======================================================================"
echo "SUMMARY"
echo "  [A] control    : $([[ $ra -eq 0 ]] && echo 'PASS (named barrier alone is fine)' || echo 'rc='$ra)"
echo "  [B] repro      : $([[ $rb -eq 2 ]] && echo 'REPRODUCED (NBarrierCnt vISA failure)' || echo 'NOT reproduced (rc='$rb')')"
echo "  [C] workaround : $([[ $rc_ -eq 0 ]] && echo 'PASS (force-inline compiles + runs)' || echo 'rc='$rc_)"
echo "======================================================================"
# Success iff: named barrier alone is fine, iSHMEM+barrier reproduces, force-inline works around.
[[ $ra -eq 0 && $rb -eq 2 && $rc_ -eq 0 ]] && exit 0 || exit 1
