#!/usr/bin/env bash
# ===========================================================================
# Launch the standalone iSHMEM low-latency DATA-CORRECTNESS reproducer
# (ll_combine_repro) across the two real nodes b70-hq-1 and b70-hq-2. Run FROM
# b70-hq-1.
#
# Reproduces the DeepEP low-latency TOKEN-LOSS regression that appears with the
# *newer* iSHMEM-IBGDA build (ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install):
# native_flush no longer aborts (see ll_quiet_repro for that earlier crash), but
# the LL test now fails a correctness assertion instead --
#
#     [rank0]: AssertionError: 42 != 56   (received tokens != expected)
#     [rank1]: AssertionError: Error: diff=0.49...  (payload bytes wrong)
#
# i.e. some device-initiated ishmem_putmem_nbi transfers are SILENTLY DROPPED or
# deliver a WRONG per-channel count. This UT mirrors the real LL dispatch traffic
# (per-(source,expert) VARIABLE counts, payload-put + 4-byte count-flag-put, one
# work-item per channel over the single per-PE QP) and then verifies BOTH the
# delivered count flag AND every payload row. No DeepEP / torch involved.
#
# Modes (REPRO_SYNC):
#   quiet   (default) device ishmemx_quiet_work_group  == DEEP_EP_LL_FLAG_PROGRESS=1
#   barrier           device ishmemx_barrier_all_work_group == LL default path
#
# EXPECT:
#   Fixed  ISHMEM -> every PE prints "[llcount] ... PASS", rc=0.
#   Broken ISHMEM -> at least one PE prints "... FAIL first_bad_slot=le=..,src=..,
#                    got=X exp=Y" (token loss/corruption), rc!=0.
#
# Usage:
#   ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install \
#       bash tests/ishmem-ut/build.sh ll_combine_repro
#   bash tests/ishmem-ut/run_ll_count.sh                    # default REPRO_SYNC=quiet
#   REPRO_SYNC=barrier bash tests/ishmem-ut/run_ll_count.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

NODE0=b70-hq-1
NODE1=b70-hq-2

BIN="$SCRIPT_DIR/ll_combine_repro"
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB
NUM_PROCESSES="${NUM_PROCESSES:-2}"   # ranks per node (= ppn); 2x2 = 4 PEs
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
REPRO_SYNC="${REPRO_SYNC:-quiet}"
REPRO_MAX_TOKENS="${REPRO_MAX_TOKENS:-64}"
REPRO_LOCAL_EXPERTS="${REPRO_LOCAL_EXPERTS:-2}"
REPRO_RECV_ACQ="${REPRO_RECV_ACQ:-1}"
REPRO_CLEAR="${REPRO_CLEAR:-cached}"
REPRO_ROW_INTS="${REPRO_ROW_INTS:-3584}"
REPRO_ITERS="${REPRO_ITERS:-4}"
REPRO_MAX_PUT_KB="${REPRO_MAX_PUT_KB:-0}"
REPRO_RESET_BARRIER="${REPRO_RESET_BARRIER:-0}"
REPRO_SKEW_US="${REPRO_SKEW_US:-0}"
REPRO_VERIFY="${REPRO_VERIFY:-perslot}"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build it first:" >&2
    echo "  ISHMEM_DIR=<path> bash tests/ishmem-ut/build.sh ll_combine_repro" >&2
    exit 1
fi

verify_rdma() {
    echo "===== Verifying RDMA accessibility ====="
    if ! ibv_devinfo -d mlx5_4 2>/dev/null | grep -q PORT_ACTIVE; then
        echo "FAIL: $NODE0 mlx5_4 port not active"; return 1
    fi
    if ! ssh -n "$NODE1" 'ibv_devinfo -d mlx5_4 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: $NODE1 mlx5_4 port not active"; return 1
    fi
    echo "Both nodes report PORT_ACTIVE on mlx5_4."
    echo "===== RDMA accessibility check done ====="
}

clean_ipc_state() {
    echo "===== Cleaning leaked IPC state (/dev/shm PSM3/iSHMEM sems) ====="
    rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/*ishmem* 2>/dev/null || true
    ssh -n "$NODE1" 'rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/*ishmem* 2>/dev/null; true' 2>/dev/null || true
}

sync_to_node1() {
    echo "===== Syncing repro to $NODE1 ====="
    ssh -n "$NODE1" "mkdir -p $SCRIPT_DIR"
    scp -q "$SCRIPT_DIR/node_wrapper.sh" "$NODE1:$SCRIPT_DIR/node_wrapper.sh"
    scp -q "$BIN" "$NODE1:$BIN"
    echo "Sync done."
}

run_test() {
    sync_to_node1
    verify_rdma || { echo "RDMA accessibility check failed; aborting." >&2; return 1; }
    clean_ipc_state

    local TOTAL_RANKS=$((NUM_PROCESSES * 2))
    echo "===== RUN ll_combine_repro REPRO_SYNC=$REPRO_SYNC (2 nodes x ${NUM_PROCESSES} ranks = $TOTAL_RANKS PEs) ====="

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
        -genv REPRO_SYNC "$REPRO_SYNC" \
        -genv REPRO_MAX_TOKENS "$REPRO_MAX_TOKENS" \
        -genv REPRO_LOCAL_EXPERTS "$REPRO_LOCAL_EXPERTS" \
        -genv REPRO_ROW_INTS "$REPRO_ROW_INTS" \
        -genv REPRO_ITERS "$REPRO_ITERS" \
        -genv REPRO_MAX_PUT_KB "$REPRO_MAX_PUT_KB" \
        -genv REPRO_RESET_BARRIER "$REPRO_RESET_BARRIER" \
        -genv REPRO_SKEW_US "$REPRO_SKEW_US" \
        -genv REPRO_VERIFY "$REPRO_VERIFY" \
        -genv REPRO_RECV_ACQ "$REPRO_RECV_ACQ" \
        -genv REPRO_CLEAR "$REPRO_CLEAR" \
        -launcher ssh \
        "$WRAPPER_PATH" \
        "$BIN"
    local rc=$?
    set -e

    echo "----- ll_combine_repro exit rc=$rc -----"
    if [ $rc -ne 0 ]; then
        echo "===== REPRODUCED: LL token-loss/correctness regression (rc=$rc). ====="
        echo "      Look for '... FAIL first_bad_slot=... got=X exp=Y' above."
    else
        echo "===== CLEAN: every device NBI put delivered, all PEs PASS (fixed ISHMEM). ====="
    fi
    return $rc
}

run_test
