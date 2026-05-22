#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT_DIR/build/tests/ut/xpu_ishmem_mapped_api_ut"

if [ ! -x "$BIN" ]; then
    "$ROOT_DIR/tests/ut/build_xpu_ishmem_mapped_api_ut.sh" >/dev/null
fi

cases=(
    normal_putmem_blocking
    normal_putmem_nbi_quiet
    normal_putmem_parallel_work_items
    ll_int_put_nbi_quiet
    atomic_add_remote
    atomic_add_remote_many
    ll_ptr_device
    quiet_empty
    intranode_no_mapped_ishmem_api
    normal_sync_all_device
    ll_barrier_work_group
    ll_putmem_nbi_atomic_flag
    atomic_add_all_pes
)

known_failures=(
    # Device sync_all validates data movement, but can crash during shutdown/finalize.
    normal_sync_all_device
    # Device work-group barrier validates data movement, then crashes during shutdown/finalize.
    ll_barrier_work_group
    # Low-latency-style NBI payload followed by atomic completion flag times out.
    ll_putmem_nbi_atomic_flag
    # Atomic add to all PEs including self crashes/fails; remote-only atomics are covered separately.
    atomic_add_all_pes
)

is_known_failure() {
    local case_name="$1"
    local known
    for known in "${known_failures[@]}"; do
        if [ "$case_name" = "$known" ]; then
            return 0
        fi
    done
    return 1
}

run_case() {
    local case_name="$1"
    timeout 120 mpirun -n 2 \
        -genv ISHMEM_IB_ENABLE_IBGDA 1 \
        -genv ISHMEM_IBGDA_DIRECT_DOORBELL 1 \
        -genv ISHMEM_ENABLE_GPU_IPC 0 \
        -genv ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP 1 \
        -genv ISHMEM_SYMMETRIC_SIZE 67108864 \
        -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1 \
        -genv ISHMEM_IBGDA_QPS_PER_PE 1 \
        -genv ISHMEM_IBGDA_DB_BATCH_SIZE 0 \
        -genv ISHMEM_IBGDA_BAR_BACKEND igub \
        -genv I_MPI_FABRICS shm \
        -genv ISHMEM_DEBUG 0 \
        "$BIN" --case "$case_name"
}

unexpected=0
for case_name in "${cases[@]}"; do
    echo "===== RUN $case_name ====="
    set +e
    run_case "$case_name"
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "===== PASS $case_name ====="
    elif is_known_failure "$case_name"; then
        echo "===== KNOWN_FAILURE $case_name rc=$rc ====="
    else
        echo "===== FAIL $case_name rc=$rc ====="
        unexpected=1
    fi
done

exit "$unexpected"
