#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$ROOT_DIR/build/tests/ut/xpu_ishmem_mapped_api_ut"

if [ ! -x "$BIN" ]; then
    "$ROOT_DIR/tests/ut/build_xpu_ishmem_mapped_api_ut.sh" >/dev/null
fi


cases=(
    init_attr_uniqueid
    normal_putmem_blocking
    normal_putmem_nbi_quiet
    normal_putmem_parallel_work_items
    ll_int_put_nbi_quiet
    atomic_add_remote
    atomic_add_remote_many
    ll_ptr_device
    team_split_sync_destroy
    device_barrier_all
    quiet_empty
    intranode_no_mapped_ishmem_api
    normal_sync_all_device
    ll_barrier_work_group
    ll_putmem_nbi_atomic_flag
    atomic_add_all_pes
    combine_payload_work_group_putmem_atomic_tail
    combine_payload_work_group_putmem_atomic_tail_separate_kernels
    work_group_sideband_putmem_nbi_atomic_tail_repeat
    work_group_sideband_putmem_nbi_split_atomic_tail_repeat
    multi_channel_nbi_quiet_atomic
    dispatch_then_combine_blocking_put
    multi_channel_nbi_quiet_atomic_split
    l3_snoop_verify
    # --- Scalar device API hang/stability tests ---
    scalar_putmem_single_task
    scalar_putmem_nbi_quiet_single_task
    scalar_int_put_single_task
    scalar_quiet_after_nbi_single_task
    scalar_fence_single_task
    scalar_barrier_all_device
    scalar_sync_all_device
    scalar_putmem_parallel_for
    scalar_putmem_nd_range_no_wg_api
    scalar_get_single_task
    scalar_putmem_large_single_task
    scalar_p_single_task
    scalar_g_single_task
    scalar_atomic_fetch_add_single_task
    scalar_quiet_no_preceding_put
    scalar_put_multi_pe_single_task
    scalar_nbi_quiet_repeat_single_task
)

known_failures=(
    # Team/barrier/sync/fence APIs that can hang or segfault during finalize (flaky)
    team_split_sync_destroy
    device_barrier_all
    normal_sync_all_device
    ll_barrier_work_group
    scalar_sync_all_device
    scalar_fence_single_task
    scalar_barrier_all_device
    # Scalar iSHMEM device APIs that still hang with IBGDA direct-doorbell transport.
    # get-side APIs hang: ishmem_get, ishmem_short_g
    scalar_get_single_task
    scalar_g_single_task
    # ishmem_short_p hangs in single_task
    scalar_p_single_task
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
    timeout 120 mpirun -n 4 \
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
        -genv MASTER_ADDR 127.0.0.1 \
        -genv ISHMEM_DEBUG 0 \
        -genv BIN $BIN \
        -genv case_name $case_name \
        bash -c '
        # export ZE_AFFINITY_MASK=4,5,6,7
       "$BIN" --case "$case_name"
       '
        
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
