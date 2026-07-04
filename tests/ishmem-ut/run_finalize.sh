#!/usr/bin/env bash
# ===========================================================================
# Launch the standalone iSHMEM TEARDOWN-ABORT reproducer (finalize_repro)
# across the two real nodes b70-hq-1 and b70-hq-2.  Run this FROM b70-hq-1.
#
# Reproduces the DeepEP internode teardown crash seen AFTER all correctness
# checks pass:
#     [teardown] all done, exiting cleanly
#     terminate called without an active exception
#     ... KILLED BY SIGNAL: 6 (Aborted)
#
# Root cause: iSHMEM's IBGDA host-proxy progress loop runs in a global
# std::thread (proxy_thread, proxy.cpp). It is join()ed ONLY by
# ishmem_finalize() -> ishmemi_proxy_fini(). DeepEP's internode::finalize()
# skips ishmem_finalize() by default, so at process exit the global
# std::thread destructor runs while the thread is still joinable, calling
# std::terminate() -> SIGABRT.
#
# Modes (REPRO_FINALIZE):
#   REPRO_FINALIZE=0  (default)  no ishmem_finalize() -> EXPECT abort (rc=134),
#                                mirrors DeepEP default -> REPRODUCES the crash.
#   REPRO_FINALIZE=1             call ishmem_finalize() (joins proxy_thread) ->
#                                EXPECT clean exit rc=0 (the fix), unless
#                                finalize itself hangs.
#
# Usage:
#   ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install \
#       bash tests/ishmem-ut/build.sh finalize_repro
#   bash tests/ishmem-ut/run_finalize.sh                    # default: reproduce
#   REPRO_FINALIZE=1 bash tests/ishmem-ut/run_finalize.sh   # fix (expect rc=0)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

NODE0=b70-hq-1
NODE1=b70-hq-2

BIN="$SCRIPT_DIR/finalize_repro"
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB
NUM_PROCESSES="${NUM_PROCESSES:-2}"   # ranks per node (= ppn); 2x2 = 4 PEs
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
REPRO_FINALIZE="${REPRO_FINALIZE:-0}"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build it first:" >&2
    echo "  ISHMEM_DIR=<path> bash tests/ishmem-ut/build.sh finalize_repro" >&2
    exit 1
fi

verify_rdma() {
    echo "===== Verifying RDMA accessibility ====="
    if ! ibv_devinfo -d mlx5_4 2>/dev/null | grep -q PORT_ACTIVE; then
        echo "FAIL: $NODE0 mlx5_4 port not active"; return 1
    fi
    if ! ssh "$NODE1" 'ibv_devinfo -d mlx5_4 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: $NODE1 mlx5_4 port not active"; return 1
    fi
    echo "Both nodes report PORT_ACTIVE on mlx5_4."
    echo "===== RDMA accessibility check done ====="
}

clean_ipc_state() {
    echo "===== Cleaning leaked IPC state (/dev/shm PSM3/iSHMEM sems) ====="
    rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/ishmem* 2>/dev/null || true
    ssh "$NODE1" 'rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/ishmem* 2>/dev/null; true' 2>/dev/null || true
}

sync_to_node1() {
    echo "===== Syncing repro to $NODE1 ====="
    ssh "$NODE1" "mkdir -p $SCRIPT_DIR"
    scp "$SCRIPT_DIR/node_wrapper.sh" "$NODE1:$SCRIPT_DIR/node_wrapper.sh" >/dev/null
    scp "$BIN" "$NODE1:$BIN" >/dev/null
    echo "Sync done."
}

run_test() {
    sync_to_node1
    verify_rdma || { echo "RDMA accessibility check failed; aborting." >&2; return 1; }
    clean_ipc_state

    local TOTAL_RANKS=$((NUM_PROCESSES * 2))
    echo "===== RUN finalize_repro REPRO_FINALIZE=$REPRO_FINALIZE (2 nodes x ${NUM_PROCESSES} ranks = $TOTAL_RANKS PEs) ====="

    local WRAPPER_PATH="$SCRIPT_DIR/node_wrapper.sh"
    chmod +x "$WRAPPER_PATH"
    cd "$DEEP_EP_DIR"

    set +e
    timeout "$TIMEOUT_SEC" mpirun \
        -n "$TOTAL_RANKS" -ppn "$NUM_PROCESSES" \
        -hosts "$NODE0,$NODE1" \
        -genv ISHMEM_IB_ENABLE_IBGDA 1 \
        -genv ISHMEM_IBGDA_DIRECT_DOORBELL 1 \
        -genv ISHMEM_ENABLE_GPU_IPC 0 \
        -genv ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP 0 \
        -genv ISHMEM_SYMMETRIC_SIZE "$ISHMEM_SYMMETRIC_SIZE" \
        -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1 \
        -genv ISHMEM_IBGDA_QPS_PER_PE 1 \
        -genv ISHMEM_IBGDA_DB_BATCH_SIZE 0 \
        -genv ISHMEM_IBGDA_BAR_BACKEND igub \
        -genv I_MPI_FABRICS shm:ofi \
        -genv FI_PROVIDER tcp \
        -genv ISHMEM_DEBUG "${ISHMEM_DEBUG:-0}" \
        -genv REPRO_FINALIZE "$REPRO_FINALIZE" \
        -launcher ssh \
        "$WRAPPER_PATH" \
        "$BIN"
    local rc=$?
    set -e

    # Interpret the result. SIGABRT under mpirun surfaces as rc=134 (128+6) or,
    # depending on the launcher, a generic non-zero with "Aborted"/SIGNAL 6 in
    # the output above. rc=0 means the proxy thread was joined cleanly.
    echo "----- finalize_repro exit rc=$rc -----"
    if [ "$REPRO_FINALIZE" = "0" ]; then
        if [ $rc -ne 0 ]; then
            echo "===== REPRODUCED: teardown abort with REPRO_FINALIZE=0 (rc=$rc). ====="
            echo "      Look for 'terminate called without an active exception' + SIGNAL 6 above."
        else
            echo "===== NOT reproduced: clean rc=0 even without ishmem_finalize() ====="
        fi
    else
        if [ $rc -eq 0 ]; then
            echo "===== FIX CONFIRMED: ishmem_finalize() joined proxy_thread, clean rc=0. ====="
        else
            echo "===== ishmem_finalize() path did NOT exit cleanly (rc=$rc): finalize hang/abort. ====="
        fi
    fi
    return $rc
}

run_test
