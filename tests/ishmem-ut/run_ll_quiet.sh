#!/usr/bin/env bash
# ===========================================================================
# Launch the standalone iSHMEM IBGDA native_flush REGRESSION reproducer
# (ll_quiet_repro) across the two real nodes b70-hq-1 and b70-hq-2. Run FROM
# b70-hq-1.
#
# Reproduces the DeepEP low-latency abort that appeared AFTER the upstream
# iSHMEM-IBGDA fix (which repaired the internode-NORMAL hang):
#
#     [IBGDA native_flush CQE_ERROR] wc[0].status=9   (IBV_WC_REM_OP_ERR)
#     IBGDA native_flush: completion failed status=9
#     ERROR: poll -- native_flush failed in quiet_flag path        (quiet mode)
#     ibgda native_flush failed for ordering op barrier_all        (barrier mode)
#     Abort(-1) ... MPI_Abort(MPI_COMM_WORLD,-1)
#
# The failing primitive is IBGDA native_flush, invoked to drain this PE's
# outbound RDMA send queue after device-initiated ishmem_putmem_nbi -- exactly
# what LL dispatch/combine do. No DeepEP / torch involved.
#
# Modes (REPRO_SYNC):
#   quiet   (default) device ishmemx_quiet_work_group  == DEEP_EP_LL_FLAG_PROGRESS=1
#                     -> "native_flush failed in quiet_flag path"
#   barrier           device ishmemx_barrier_all_work_group == LL default path
#                     -> "native_flush failed for ordering op barrier_all"
#   host_quiet        host ishmem_quiet()   (isolates device vs host origin)
#   host_barrier      host ishmem_barrier_all()
#
# EXPECT (this ISHMEM build): abort with native_flush status=9 (rc!=0).
# A FIXED ISHMEM build prints "[llquiet] ... PASS" on every PE and exits 0.
#
# Usage:
#   ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install \
#       bash tests/ishmem-ut/build.sh ll_quiet_repro
#   bash tests/ishmem-ut/run_ll_quiet.sh                    # default REPRO_SYNC=quiet
#   REPRO_SYNC=barrier bash tests/ishmem-ut/run_ll_quiet.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

NODE0=b70-hq-1
NODE1=b70-hq-2

BIN="$SCRIPT_DIR/ll_quiet_repro"
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB
NUM_PROCESSES="${NUM_PROCESSES:-2}"   # ranks per node (= ppn); 2x2 = 4 PEs
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
REPRO_SYNC="${REPRO_SYNC:-quiet}"
REPRO_PAYLOAD_INTS="${REPRO_PAYLOAD_INTS:-4096}"
REPRO_ITERS="${REPRO_ITERS:-1}"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build it first:" >&2
    echo "  ISHMEM_DIR=<path> bash tests/ishmem-ut/build.sh ll_quiet_repro" >&2
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
    echo "===== RUN ll_quiet_repro REPRO_SYNC=$REPRO_SYNC (2 nodes x ${NUM_PROCESSES} ranks = $TOTAL_RANKS PEs) ====="

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
        -genv REPRO_PAYLOAD_INTS "$REPRO_PAYLOAD_INTS" \
        -genv REPRO_ITERS "$REPRO_ITERS" \
        -launcher ssh \
        "$WRAPPER_PATH" \
        "$BIN"
    local rc=$?
    set -e

    echo "----- ll_quiet_repro exit rc=$rc -----"
    if [ $rc -ne 0 ]; then
        echo "===== REPRODUCED: IBGDA native_flush regression (rc=$rc). ====="
        echo "      Look for 'IBGDA native_flush ... status=9' above."
    else
        echo "===== CLEAN: native_flush drained OK, all PEs PASS (fixed ISHMEM). ====="
    fi
    return $rc
}

run_test
