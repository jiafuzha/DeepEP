#!/usr/bin/env bash
# ===========================================================================
# Launch the standalone iSHMEM device-PUT reproducer (ll_put_repro) across the
# two real nodes b70-hq-1 and b70-hq-2.  Run this script FROM b70-hq-1.
#
# This isolates the exact iSHMEM primitives the DeepEP low-latency dispatch
# kernel relies on (device-initiated ishmem_putmem_nbi + a device barrier) so a
# broken iSHMEM shows up as "delivered=1/N" (only the self slot) instead of the
# full DeepEP token-count assertion. See ll_put_repro.cpp for the mapping.
#
# It reuses the SAME transport env as tests/real-2node-ll/run.sh (IBGDA on,
# GPU_IPC off, one QP per PE, tcp bootstrap) and the SAME per-rank node_wrapper
# (ZE_AFFINITY_MASK=4,5, mlx5_4/mlx5_5, dynamic FI_VERBS_IFACE).
#
# Usage:
#   # 1. Build against whichever iSHMEM you want to test:
#   ISHMEM_DIR=/root/jiafuzha/code-repo/zjf2012/ishmem_ibgda/build/_install \
#       bash tests/ishmem-ut/build.sh          # known-good
#   # or
#   ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install \
#       bash tests/ishmem-ut/build.sh          # suspect / broken
#   # 2. Run it:
#   bash tests/ishmem-ut/run.sh
#
#   Knobs: REPRO_BARRIER=device|host  REPRO_PAYLOAD_INTS=4096  REPRO_ITERS=1
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

NODE0=b70-hq-1
NODE1=b70-hq-2

BIN="$SCRIPT_DIR/ll_put_repro"
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB
NUM_PROCESSES="${NUM_PROCESSES:-2}"   # ranks per node (= ppn); 2x2 = 4 PEs
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"

REPRO_BARRIER="${REPRO_BARRIER:-device}"
REPRO_PAYLOAD_INTS="${REPRO_PAYLOAD_INTS:-4096}"
REPRO_ITERS="${REPRO_ITERS:-3}"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build it first:" >&2
    echo "  ISHMEM_DIR=<path> bash tests/ishmem-ut/build.sh" >&2
    exit 1
fi

# --- Verify RDMA accessibility on both nodes (mlx5_4 smoke test). ---
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

# --- Clean leaked IPC state from previous SIGKILL'd runs. ---
clean_ipc_state() {
    echo "===== Cleaning leaked IPC state (/dev/shm PSM3/iSHMEM sems) ====="
    rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/ishmem* 2>/dev/null || true
    ssh "$NODE1" 'rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/ishmem* 2>/dev/null; true' 2>/dev/null || true
}

# --- Sync wrapper + freshly-built binary to NODE1. ---
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
    echo "===== RUN ll_put_repro (2 nodes x ${NUM_PROCESSES} ranks = $TOTAL_RANKS PEs) ====="

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
        -genv REPRO_BARRIER "$REPRO_BARRIER" \
        -genv REPRO_PAYLOAD_INTS "$REPRO_PAYLOAD_INTS" \
        -genv REPRO_ITERS "$REPRO_ITERS" \
        -launcher ssh \
        "$WRAPPER_PATH" \
        "$BIN"
    local rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        echo "===== PASS ll_put_repro (iSHMEM delivered all remote slots) ====="
    else
        echo "===== FAIL ll_put_repro (rc=$rc; remote token/slot delivery broken) ====="
    fi
    return $rc
}

run_test
