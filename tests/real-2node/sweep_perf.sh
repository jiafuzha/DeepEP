#!/usr/bin/env bash
# ===========================================================================
# Perf sweep of test_internode.py across token sizes.
# Uses DEEP_EP_PERF=1 DEEP_EP_MIN=1 to get clean latency/bandwidth numbers
# in a single run per token size without the full correctness sweep.
#
# Token sizes: 32, 64, 128, 256, 512, 1024
# Fixed: hidden=7168, topk=2, experts=8, processes-per-node=2, total-ranks=4
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NODE0=b70-hq-1
NODE1=b70-hq-2
ISHMEM_DIR="${ISHMEM_DIR:-/root/jiafuzha/ishmem_ibgda/build/_install}"

TOKEN_SIZES=(32 64 128 256 512 1024)
HIDDEN=7168
NUM_TOPK=2
NUM_EXPERTS=8
NUM_PROCESSES=2

# Helper: round up to next power of 2
next_pow2() {
    local n=$1
    if (( n <= 0 )); then echo 1; return; fi
    n=$(( n - 1 ))
    n=$(( n | n >> 1 ))
    n=$(( n | n >> 2 ))
    n=$(( n | n >> 4 ))
    n=$(( n | n >> 8 ))
    n=$(( n | n >> 16 ))
    echo $(( n + 1 ))
}

# ---------------------------------------------------------------------------
# Sync and build check
# ---------------------------------------------------------------------------
echo "===== Syncing scripts to $NODE1 ====="
ssh "$NODE1" "mkdir -p $(dirname "$SCRIPT_DIR") && mkdir -p $SCRIPT_DIR"
scp "$SCRIPT_DIR/run.sh" "$SCRIPT_DIR/node_wrapper.sh" "$NODE1:$SCRIPT_DIR/" >/dev/null
scp "$DEEP_EP_DIR/tests/test_internode.py" "$DEEP_EP_DIR/tests/utils.py" "$NODE1:$DEEP_EP_DIR/tests/" >/dev/null
echo "Sync done."

# ---------------------------------------------------------------------------
# Clean IPC state before starting
# ---------------------------------------------------------------------------
clean_ipc() {
    rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/ishmem* 2>/dev/null || true
    rm -f /dev/shm/*oneccl* /dev/shm/*ccl_* /dev/shm/sem.*ccl* /dev/shm/gloo* /dev/shm/sem.gloo* 2>/dev/null || true
    rm -f /tmp/deep_ep_xpu_ipc_*.sock 2>/dev/null || true
    ssh "$NODE1" 'rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/ishmem* /dev/shm/*oneccl* /dev/shm/*ccl_* /dev/shm/sem.*ccl* /dev/shm/gloo* /dev/shm/sem.gloo* /tmp/deep_ep_xpu_ipc_*.sock 2>/dev/null; true' 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Run perf for a single token size
# ---------------------------------------------------------------------------
run_perf() {
    local NUM_TOKENS=$1

    # Calculate buffer sizes (same scaling as sweep_tokens.sh)
    local PER=$(( HIDDEN * 2 ))
    local NVL_RAW=$(( NUM_TOKENS * PER * 12 ))
    local NVL_BYTES=$(next_pow2 "$NVL_RAW")
    local RDMA_RAW=$(( NUM_TOKENS * PER * 6 ))
    local RDMA_BYTES=$(next_pow2 "$RDMA_RAW")
    if (( NVL_BYTES < 134217728 )); then NVL_BYTES=134217728; fi
    if (( RDMA_BYTES < 67108864 )); then RDMA_BYTES=67108864; fi
    local SHM_RAW=$(( RDMA_BYTES * 8 ))
    local SHM_BYTES=$(next_pow2 "$SHM_RAW")
    if (( SHM_BYTES < 268435456 )); then SHM_BYTES=268435456; fi

    echo ""
    echo "================================================================"
    echo "PERF: tokens=$NUM_TOKENS  hidden=$HIDDEN  topk=$NUM_TOPK  experts=$NUM_EXPERTS"
    echo "  NVL=$(numfmt --to=iec $NVL_BYTES)  RDMA=$(numfmt --to=iec $RDMA_BYTES)  SHM=$(numfmt --to=iec $SHM_BYTES)"
    echo "================================================================"

    clean_ipc

    local WRAPPER_PATH="$SCRIPT_DIR/node_wrapper.sh"
    chmod +x "$WRAPPER_PATH"

    cd "$DEEP_EP_DIR"

    set +e
    timeout 600 mpirun \
        -n 4 -ppn "$NUM_PROCESSES" \
        -hosts "$NODE0,$NODE1" \
        -genv ISHMEM_IB_ENABLE_IBGDA 1 \
        -genv ISHMEM_IBGDA_DIRECT_DOORBELL 1 \
        -genv ISHMEM_ENABLE_GPU_IPC 0 \
        -genv ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP 0 \
        -genv ISHMEM_SYMMETRIC_SIZE "$SHM_BYTES" \
        -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1 \
        -genv ISHMEM_IBGDA_DB_BATCH_SIZE 0 \
        -genv ISHMEM_IBGDA_BAR_BACKEND igub \
        -genv I_MPI_FABRICS shm:ofi \
        -genv FI_PROVIDER tcp \
        -genv ISHMEM_DEBUG 0 \
        -genv DEEP_EP_NVL_RANKS "$NUM_PROCESSES" \
        -genv DEEP_EP_NVL_BYTES "$NVL_BYTES" \
        -genv DEEP_EP_RDMA_BYTES "$RDMA_BYTES" \
        -genv DEEP_EP_PERF 1 \
        -genv DEEP_EP_MIN 1 \
        -genv MASTER_ADDR "$NODE0" \
        -genv MASTER_PORT 29500 \
        -genv WORLD_SIZE 2 \
        -genv TORCH_DISTRIBUTED_DEBUG OFF \
        -launcher ssh \
        "$WRAPPER_PATH" \
        python3 -u tests/test_internode.py \
            --num-processes "$NUM_PROCESSES" \
            --num-tokens "$NUM_TOKENS" \
            --hidden "$HIDDEN" \
            --num-topk "$NUM_TOPK" \
            --num-experts "$NUM_EXPERTS" 2>&1
    local rc=$?
    set -e

    if [ $rc -eq 0 ]; then
        echo "  PERF RESULT: PASS (tokens=$NUM_TOKENS)"
    else
        echo "  PERF RESULT: FAIL (tokens=$NUM_TOKENS, rc=$rc)"
    fi

    # Extract PERF lines from output
    echo ""
    return $rc
}

# ---------------------------------------------------------------------------
# Main: run perf for each token size
# ---------------------------------------------------------------------------
echo "============================================================"
echo "DeepEP internode PERF sweep (DEEP_EP_PERF=1 DEEP_EP_MIN=1)"
echo "============================================================"

PERF_LOG="$SCRIPT_DIR/perf_results.log"
> "$PERF_LOG"

PASS_COUNT=0
FAIL_COUNT=0
declare -A RESULTS

for NUM_TOKENS in "${TOKEN_SIZES[@]}"; do
    TMP_LOG=$(mktemp)
    set +e
    run_perf "$NUM_TOKENS" > "$TMP_LOG" 2>&1
    RC=$?
    set -e

    # Extract [PERF] lines
    grep "\[PERF\]" "$TMP_LOG" | tee -a "$PERF_LOG"

    # Also capture teardown and result
    grep -E "PERF RESULT:|teardown|PASS|FAIL" "$TMP_LOG" | tee -a "$PERF_LOG"

    if [ $RC -eq 0 ]; then
        RESULTS[$NUM_TOKENS]="PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        RESULTS[$NUM_TOKENS]="FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    rm -f "$TMP_LOG"
    echo ""
done

echo ""
echo "============================================================"
echo "PERF SWEEP SUMMARY"
echo "============================================================"
for NUM_TOKENS in "${TOKEN_SIZES[@]}"; do
    printf "  tokens=%-6s -> %s\n" "$NUM_TOKENS" "${RESULTS[$NUM_TOKENS]}"
done
echo "------------------------------------------------------------"
echo "  PASS: $PASS_COUNT / ${#TOKEN_SIZES[@]}"
echo "  FAIL: $FAIL_COUNT / ${#TOKEN_SIZES[@]}"
echo "============================================================"
echo "Full perf log: $PERF_LOG"
echo "============================================================"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
exit 0