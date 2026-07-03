#!/usr/bin/env bash
# ===========================================================================
# Launch the standalone iSHMEM NORMAL-path reproducer (normal_putmem_repro)
# across the two real nodes b70-hq-1 and b70-hq-2.  Run this FROM b70-hq-1.
#
# This isolates the primitives the DeepEP internode NORMAL dispatch relies on
# but the low-latency path does NOT: device-side BLOCKING ishmem_putmem and the
# single-work-item DEVICE-WIDE ishmem_barrier_all() (internode.cpp:2510-2541),
# both completing while the host is parked in queue.wait().
#
# A broken iSHMEM (blocking / host-proxy-progress path) shows up here as a HANG
# in the `putmem` / `barrier` modes, while the LL-style `nbi_wg` control mode
# still completes -- matching the real test: real-2node-ll PASSES but
# real-2node (normal) HANGS at the first dispatch.
#
# Usage:
#   # 1. Build (against whichever iSHMEM you want to test):
#   ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install \
#       bash tests/ishmem-ut/build.sh normal_putmem_repro
#   # 2. Run:
#   bash tests/ishmem-ut/run_normal.sh                  # default REPRO_MODE=full
#   REPRO_MODE=exact   bash tests/ishmem-ut/run_normal.sh  # no-warmup control (PASS)
#   REPRO_MODE=barrier bash tests/ishmem-ut/run_normal.sh
#   REPRO_MODE=nbi_wg  bash tests/ishmem-ut/run_normal.sh   # LL-style control (PASS)
#
#   Knobs: REPRO_MODE=full|exact|putmem|barrier|nbi_wg  REPRO_STAGE_TIMEOUT=30
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

NODE0=b70-hq-1
NODE1=b70-hq-2

BIN="$SCRIPT_DIR/normal_putmem_repro"
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB
NUM_PROCESSES="${NUM_PROCESSES:-2}"   # ranks per node (= ppn); 2x2 = 4 PEs
# Overall mpirun timeout: must exceed the in-binary per-stage watchdog so the
# watchdog (not mpirun) reports the hang first, but still bounds a full stall.
REPRO_STAGE_TIMEOUT="${REPRO_STAGE_TIMEOUT:-30}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
REPRO_MODE="${REPRO_MODE:-full}"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Build it first:" >&2
    echo "  ISHMEM_DIR=<path> bash tests/ishmem-ut/build.sh normal_putmem_repro" >&2
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
    echo "===== RUN normal_putmem_repro mode=$REPRO_MODE (2 nodes x ${NUM_PROCESSES} ranks = $TOTAL_RANKS PEs) ====="

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
        -genv REPRO_MODE "$REPRO_MODE" \
        -genv REPRO_STAGE_TIMEOUT "$REPRO_STAGE_TIMEOUT" \
        -launcher ssh \
        "$WRAPPER_PATH" \
        "$BIN"
    local rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        echo "===== PASS normal_putmem_repro mode=$REPRO_MODE (primitives completed) ====="
    else
        echo "===== FAIL normal_putmem_repro mode=$REPRO_MODE (rc=$rc; blocking putmem / device barrier hung) ====="
    fi
    return $rc
}

run_test
