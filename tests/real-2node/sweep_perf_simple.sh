#!/usr/bin/env bash
# Simple perf sweep: run each token size with DEEP_EP_PERF=1, capture [PERF] lines.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

next_pow2() {
    local n=$1
    if (( n <= 0 )); then echo 1; return; fi
    n=$(( n - 1 ))
    n=$(( n | n >> 1 )); n=$(( n | n >> 2 )); n=$(( n | n >> 4 ))
    n=$(( n | n >> 8 )); n=$(( n | n >> 16 ))
    echo $(( n + 1 ))
}

calc_buf() {
    local NT=$1 PER=$((7168 * 2))
    local NVL=$(next_pow2 $(( NT * PER * 12 ))); (( NVL < 134217728 )) && NVL=134217728
    local RDMA=$(next_pow2 $(( NT * PER * 6 ))); (( RDMA < 67108864 )) && RDMA=67108864
    local SHM=$(next_pow2 $(( RDMA * 8 ))); (( SHM < 268435456 )) && SHM=268435456
    echo "$NVL $RDMA $SHM"
}

for NT in 32 64 128 256 512 1024; do
    read NVL RDMA SHM <<< $(calc_buf $NT)
    echo ""
    echo "=== PERF tokens=$NT NVL=$(numfmt --to=iec $NVL) RDMA=$(numfmt --to=iec $RDMA) SHM=$(numfmt --to=iec $SHM) ==="

    rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/ishmem* 2>/dev/null || true
    ssh b70-hq-2 'rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/ishmem* 2>/dev/null; true' 2>/dev/null || true

    ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install \
    SKIP_NIC_CHECK=1 \
    DEEP_EP_PERF=1 \
    DEEP_EP_MIN=1 \
    DEEP_EP_NVL_BYTES=$NVL \
    DEEP_EP_RDMA_BYTES=$RDMA \
    ISHMEM_SYMMETRIC_SIZE=$SHM \
    NUM_PROCESSES=2 \
    NUM_TOKENS=$NT \
    HIDDEN=7168 \
    NUM_TOPK=2 \
    NUM_EXPERTS=8 \
        bash "$SCRIPT_DIR/run.sh" 2>&1 | grep -E "\[PERF\]|=====.PASS|=====.FAIL|KILLED BY" || true

    RC=${PIPESTATUS[0]}
    if [ "$RC" -eq 0 ]; then
        echo "  => PASS"
    else
        echo "  => FAIL (rc=$RC)"
    fi
done

echo ""
echo "=== PERF SWEEP COMPLETE ==="
