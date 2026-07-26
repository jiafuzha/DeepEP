#!/usr/bin/env bash
# ===========================================================================
# Sweep test_internode.py with varying token counts (32 -> 1024)
# on the 2-node XPU cluster, scaling buffer sizes appropriately.
#
# Token sizes: 32, 64, 128, 256, 512, 1024
# Fixed: hidden=7168, topk=2, experts=8, processes-per-node=2, total-ranks=4
#
# Buffer sizing logic:
#   Per_token_bytes = hidden * 2 (BF16) * overhead_factor
#   NVL:   num_tokens * hidden * 2 * 4 (local ranks + scratch) rounded up to power-of-2
#   RDMA:  num_tokens * hidden * 2 * 2 (send+recv) rounded up to power-of-2
#   SHM:   max(NVL * 2, RDMA * num_ranks) rounded up to power-of-2
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NODE0=b70-hq-1
NODE1=b70-hq-2
ISHMEM_DIR="/root/jiafuzha/ishmem_ibgda/build/_install"

# Token sizes to sweep: 32, 64, 128, 256, 512, 1024
TOKEN_SIZES=(32 64 128 256 512 1024)

# Fixed params
HIDDEN=7168
NUM_TOPK=2
NUM_EXPERTS=8
NUM_PROCESSES=2  # per node
# TOTAL_RANKS = NUM_PROCESSES * 2 = 4

# Output log directory
LOG_DIR="$SCRIPT_DIR/sweep_logs"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

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

echo "============================================================"
echo "DeepEP internode token sweep"
echo "  Hidden: $HIDDEN"
echo "  TopK:   $NUM_TOPK"
echo "  Experts: $NUM_EXPERTS"
echo "  Procs/node: $NUM_PROCESSES  (total ranks: 4)"
echo "  Token sizes: ${TOKEN_SIZES[*]}"
echo "============================================================"

PASS_COUNT=0
FAIL_COUNT=0
declare -A RESULTS

for NUM_TOKENS in "${TOKEN_SIZES[@]}"; do
    # Calculate buffer sizes
    # Scaling factors based on token count:
    #   NVL: 12x for multiple local ranks + IPC scratch + safety margin
    #   RDMA: 10x for send/recv staging across ranks + queue buffer (FP8 path needs ~1.4x more)
    #   SHM: 8x clamped RDMA for all 4 ranks + headroom
    PER_TOKEN_BYTES=$(( HIDDEN * 2 ))  # BF16 = 2 bytes per element
    NVL_RAW=$(( NUM_TOKENS * PER_TOKEN_BYTES * 12 ))
    NVL_BYTES=$(next_pow2 "$NVL_RAW")
    RDMA_RAW=$(( NUM_TOKENS * PER_TOKEN_BYTES * 10 ))
    RDMA_BYTES=$(next_pow2 "$RDMA_RAW")

    # Ensure minimums before computing SHM
    if (( NVL_BYTES < 134217728 )); then NVL_BYTES=134217728; fi   # min 128 MiB
    if (( RDMA_BYTES < 67108864 )); then RDMA_BYTES=67108864; fi    # min 64 MiB

    # Symmetric heap: cover RDMA buffer after clamping for all 4 ranks + headroom
    SHM_RAW=$(( RDMA_BYTES * 8 ))
    SHM_BYTES=$(next_pow2 "$SHM_RAW")
    if (( SHM_BYTES < 268435456 )); then SHM_BYTES=268435456; fi    # min 256 MiB

    echo ""
    echo "------------------------------------------------------------"
    echo "TOKENS=$NUM_TOKENS  NVL=$(numfmt --to=iec $NVL_BYTES)  RDMA=$(numfmt --to=iec $RDMA_BYTES)  SHM=$(numfmt --to=iec $SHM_BYTES)"
    echo "------------------------------------------------------------"

    LOG_FILE="$LOG_DIR/tokens_${NUM_TOKENS}.log"

    set +e
    ISHMEM_DIR=$ISHMEM_DIR \
    DEEP_EP_PERF=1 \
    NUM_PROCESSES=$NUM_PROCESSES \
    NUM_TOKENS=$NUM_TOKENS \
    HIDDEN=$HIDDEN \
    NUM_TOPK=$NUM_TOPK \
    NUM_EXPERTS=$NUM_EXPERTS \
    DEEP_EP_NVL_BYTES=$NVL_BYTES \
    DEEP_EP_RDMA_BYTES=$RDMA_BYTES \
    ISHMEM_SYMMETRIC_SIZE=$SHM_BYTES \
    SKIP_NIC_CHECK=1 \
        bash "$SCRIPT_DIR/run.sh" 2>&1 | tee "$LOG_FILE"
    RC=$?
    set -e

    # Extract PERF lines
    echo ""
    echo "  --- PERF ---"
    grep "\[PERF\]" "$LOG_FILE" | head -4 || echo "  (no PERF lines)"

    if [ $RC -eq 0 ]; then
        echo "  RESULT: PASS (tokens=$NUM_TOKENS)"
        RESULTS[$NUM_TOKENS]="PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  RESULT: FAIL (tokens=$NUM_TOKENS, rc=$RC)"
        RESULTS[$NUM_TOKENS]="FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    # Let GPU memory settle between runs: iSHMEM symmetric heap
    # deallocation is asynchronous; running the next test immediately
    # can cause OOM or data corruption (check_x rows not uniform).
    sleep 5
done

echo ""
echo "============================================================"
echo "SWEEP SUMMARY"
echo "============================================================"
for NUM_TOKENS in "${TOKEN_SIZES[@]}"; do
    printf "  tokens=%-6s -> %s\n" "$NUM_TOKENS" "${RESULTS[$NUM_TOKENS]}"
done
echo "------------------------------------------------------------"
echo "  PASS: $PASS_COUNT / ${#TOKEN_SIZES[@]}"
echo "  FAIL: $FAIL_COUNT / ${#TOKEN_SIZES[@]}"
echo "============================================================"
echo "Logs saved to: $LOG_DIR"
echo "============================================================"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
exit 0