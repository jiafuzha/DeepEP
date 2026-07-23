#!/usr/bin/env bash
# ===========================================================================
# Launch 2-node DeepEP internode NORMAL-path test (NVL + RDMA) on real nodes
# b70-hq-1 and b70-hq-2.  Run this script FROM b70-hq-1.
#
# Both nodes use the same devices:
#   b70-hq-1 (node 0): GPUs 4,5 + NICs mlx5_4,mlx5_5
#   b70-hq-2 (node 1): GPUs 4,5 + NICs mlx5_4,mlx5_5
#
# With ppn=2 the rank topology is:
#   rank 0,1 -> b70-hq-1  local_rank 0,1 -> GPU 4,5 + mlx5_4,mlx5_5
#   rank 2,3 -> b70-hq-2  local_rank 0,1 -> GPU 4,5 + mlx5_4,mlx5_5
#
# test_internode.py detects the MPI launcher via MPI_LOCALRANKID and calls
# test_loop() directly (no self-relaunch). init_dist() derives the global rank
# from RANK (node rank, set by node_wrapper.sh) and WORLD_SIZE=2.
# DEEP_EP_NVL_RANKS=NUM_PROCESSES tells the test there are NUM_PROCESSES local
# (NVL) ranks per node, so num_nodes = num_ranks / nvl_ranks = 2.
#
# Prerequisites:
#   - Passwordless SSH from b70-hq-1 to b70-hq-2 (standard ~/.ssh/id_rsa).
#   - ISHMEM_DIR defaults to /root/jiafuzha/code-repo/zjf2012/ishmem_ibgda/build/_install.
#   - Intel oneAPI on both nodes; DeepEP built with XCCL-enabled PyTorch.
#
# Usage:
#   ./run.sh
#   SKIP_NIC_CHECK=1 NUM_PROCESSES=2 NUM_TOKENS=32 ./run.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

NODE0=b70-hq-1
NODE1=b70-hq-2

TEST_SCRIPT="${TEST_SCRIPT:-tests/test_internode.py}"
ISHMEM_DIR="${ISHMEM_DIR:-/root/jiafuzha/ishmem_ibgda/build/_install}"

# DeepEP buffer sizes
DEEP_EP_NVL_BYTES="${DEEP_EP_NVL_BYTES:-134217728}"   # 128 MiB
DEEP_EP_RDMA_BYTES="${DEEP_EP_RDMA_BYTES:-67108864}"  # 64 MiB
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB

MASTER_PORT="${MASTER_PORT:-29500}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"   # ranks per node (= ppn = nvl_ranks)
NUM_TOKENS="${NUM_TOKENS:-32}"
HIDDEN="${HIDDEN:-1024}"
NUM_TOPK="${NUM_TOPK:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-360}"

# --- Verify RDMA accessibility on both nodes ---
verify_rdma() {
    echo "===== Verifying RDMA accessibility ====="

    # Check local node (NODE0) directly; SSH to NODE1
    local devs0 devs1
    devs0=$(ibv_devices 2>/dev/null | awk 'NR>2 {print $1}' | tr '\n' ',' || true)
    devs1=$(ssh "$NODE1" 'ibv_devices 2>/dev/null | awk "NR>2 {print \$1}" | tr "\n" ","' 2>/dev/null || true)
    if [ -z "$devs0" ]; then
        echo "FAIL: $NODE0 has no IB devices visible"
        return 1
    fi
    echo "$NODE0 IB devices: $devs0"
    if [ -z "$devs1" ]; then
        echo "FAIL: $NODE1 has no IB devices visible"
        return 1
    fi
    echo "$NODE1 IB devices: $devs1"

    if ! ibv_devinfo -d mlx5_4 2>/dev/null | grep -q PORT_ACTIVE; then
        echo "FAIL: $NODE0 mlx5_4 port not active"
        return 1
    fi
    if ! ssh "$NODE1" 'ibv_devinfo -d mlx5_4 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: $NODE1 mlx5_4 port not active"
        return 1
    fi
    echo "Both nodes report PORT_ACTIVE on mlx5_4."

    # Quick RoCE/RDMA loopback smoke test (NODE1 connects to NODE0)
    pkill -f ibv_rc_pingpong 2>/dev/null || true
    ibv_rc_pingpong -d mlx5_4 -g 3 -n 1 >/tmp/rdma_srv.log 2>&1 &
    sleep 1
    if ssh "$NODE1" "timeout 10 ibv_rc_pingpong -d mlx5_4 -g 3 -n 1 $NODE0 >/tmp/rdma_cli.log 2>&1"; then
        echo "RoCE ibv_rc_pingpong mlx5_4($NODE1)<->mlx5_4($NODE0) SUCCEEDED."
    else
        echo "WARNING: ibv_rc_pingpong did not succeed."
        echo "         (Test still proceeds; iSHMEM uses physical NICs via /dev/infiniband.)"
        tail -n 5 /tmp/rdma_srv.log 2>/dev/null || true
        ssh "$NODE1" 'tail -n 5 /tmp/rdma_cli.log 2>/dev/null' || true
    fi
    echo "===== RDMA accessibility check done ====="
}

# --- Clean leaked IPC state from previous SIGKILL'd runs ---
clean_ipc_state() {
    echo "===== Cleaning leaked IPC state (/dev/shm PSM3/CCL sems) ====="
    _clean() {
        rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* 2>/dev/null || true
        rm -f /dev/shm/sem.ishmem* /dev/shm/ishmem* 2>/dev/null || true
        rm -f /dev/shm/*oneccl* /dev/shm/*ccl_* /dev/shm/sem.*ccl* 2>/dev/null || true
        rm -f /dev/shm/gloo* /dev/shm/sem.gloo* 2>/dev/null || true
        rm -f /tmp/deep_ep_xpu_ipc_*.sock 2>/dev/null || true
    }
    _clean || true
    ssh "$NODE1" 'rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/ishmem* /dev/shm/*oneccl* /dev/shm/*ccl_* /dev/shm/sem.*ccl* /dev/shm/gloo* /dev/shm/sem.gloo* /tmp/deep_ep_xpu_ipc_*.sock 2>/dev/null; true' 2>/dev/null || true
}

# --- Ensure MASTER_PORT is free on both nodes ---
ensure_port_free() {
    local port="$MASTER_PORT"
    echo "===== Ensuring MASTER_PORT $port free on both nodes ====="
    _free_port() {
        local p="$1"
        local attempt holders pids
        for attempt in 1 2 3; do
            holders=$( (ss -lntpH 2>/dev/null || netstat -lntp 2>/dev/null | tail -n +3) \
                       | awk -v p=":${p}" '$4 ~ p { print }' )
            if [ -z "$holders" ]; then echo "port $p is free"; return 0; fi
            pids=$(echo "$holders" | grep -oE '[0-9]+' | sort -u)
            [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
            sleep 2
        done
        echo "FAIL: port $p still in use" >&2; return 1
    }
    _free_port "$port" || return 1
    ssh "$NODE1" "
        port=$port
        for attempt in 1 2 3; do
            holders=\$( (ss -lntpH 2>/dev/null || netstat -lntp 2>/dev/null | tail -n +3) \
                       | awk -v p=\":$port\" '\$4 ~ p { print }' )
            if [ -z \"\$holders\" ]; then echo \"port $port is free\"; exit 0; fi
            pids=\$(echo \"\$holders\" | grep -oE '[0-9]+' | sort -u)
            [ -n \"\$pids\" ] && kill -9 \$pids 2>/dev/null || true
            sleep 2
        done
        echo 'FAIL: port $port still in use' >&2; exit 1
    " || return 1
}

# --- Sync scripts and nic_pcie_check binary to NODE1 so MPI can find node_wrapper.sh ---
sync_to_node1() {
    echo "===== Syncing scripts to $NODE1 ====="
    ssh "$NODE1" "mkdir -p $(dirname "$SCRIPT_DIR") && mkdir -p $SCRIPT_DIR"
    for f in run.sh node_wrapper.sh verify_nic_selection.sh; do
        scp "$SCRIPT_DIR/$f" "$NODE1:$SCRIPT_DIR/$f" >/dev/null
    done
    local nic_bin="$DEEP_EP_DIR/tests/docker-2node-ll/nic_pcie_check"
    if [ -f "$nic_bin" ]; then
        ssh "$NODE1" "mkdir -p $(dirname "$nic_bin")"
        scp "$nic_bin" "$NODE1:$nic_bin" >/dev/null
    fi
    echo "Sync done."
}

# --- Gate: verify iSHMEM auto GPU->NIC selection is same-PCIe-switch per rank ---
verify_nic_selection() {
    DEEP_EP_DIR="$DEEP_EP_DIR" \
    ISHMEM_DIR="$ISHMEM_DIR" \
    NUM_PROCESSES="$NUM_PROCESSES" \
    ISHMEM_SYMMETRIC_SIZE="$ISHMEM_SYMMETRIC_SIZE" \
        bash "$SCRIPT_DIR/verify_nic_selection.sh"
}

# --- Run DeepEP internode normal-path test ---
run_test() {
    sync_to_node1
    verify_rdma || { echo "RDMA accessibility check failed; aborting." >&2; return 1; }
    # NOTE: verify_nic_selection runs nic_pcie_check under 2-node mpirun; skip with
    # SKIP_NIC_CHECK=1 if that binary crashes on your cluster before iSHMEM init.
    if [ "${SKIP_NIC_CHECK:-0}" != "1" ]; then
        verify_nic_selection || {
            echo "iSHMEM auto NIC selection check FAILED; aborting." >&2
            echo "Re-run with SKIP_NIC_CHECK=1 to bypass this gate." >&2
            return 1
        }
    fi
    clean_ipc_state
    ensure_port_free || { echo "MASTER_PORT cleanup failed; aborting." >&2; return 1; }

    local TOTAL_RANKS=$((NUM_PROCESSES * 2))
    echo "===== RUN $TEST_SCRIPT (2 nodes x ${NUM_PROCESSES} ranks = $TOTAL_RANKS total) ====="

    local WRAPPER_PATH="$SCRIPT_DIR/node_wrapper.sh"
    chmod +x "$WRAPPER_PATH"

    cd "$DEEP_EP_DIR"

    # Optional env vars: only forward when NON-EMPTY. Several code-side gates use
    # `getenv(...) != nullptr`, so forwarding an EMPTY value (mpirun -genv NAME "")
    # would enable them (e.g. DEEP_EP_DBG_* per-stage tracing with queue.wait(),
    # which serializes every kernel boundary and destroys perf). Build the -genv
    # list dynamically and skip any var whose value is empty.
    OPT_GENV=()
    _add_opt_genv() { if [ -n "${2:-}" ]; then OPT_GENV+=( -genv "$1" "$2" ); fi; }
    _add_opt_genv DEEP_EP_INTERNODE_PAR_GATHER "${DEEP_EP_INTERNODE_PAR_GATHER:-}"
    _add_opt_genv DEEP_EP_INTERNODE_BLOCKING_PUT "${DEEP_EP_INTERNODE_BLOCKING_PUT:-}"
    _add_opt_genv DEEP_EP_INTERNODE_POLL_CAP   "${DEEP_EP_INTERNODE_POLL_CAP:-}"
    _add_opt_genv DEEP_EP_PERF                 "${DEEP_EP_PERF:-}"
    _add_opt_genv DEEP_EP_PERF_TOKENS          "${DEEP_EP_PERF_TOKENS:-}"
    _add_opt_genv DEEP_EP_MIN                  "${DEEP_EP_MIN:-}"
    _add_opt_genv DEEP_EP_DBG_DISPATCH         "${DEEP_EP_DBG_DISPATCH:-}"
    _add_opt_genv DEEP_EP_DBG_COMBINE          "${DEEP_EP_DBG_COMBINE:-}"
    _add_opt_genv DEEP_EP_TIME_WARMUP          "${DEEP_EP_TIME_WARMUP:-}"
    _add_opt_genv DEEP_EP_XPU_ISHMEM_FINALIZE  "${DEEP_EP_XPU_ISHMEM_FINALIZE:-}"
    # Normal internode auto-provisions ISHMEM_IBGDA_QPS_PER_PE = clamp_pow2(num_channels)
    # inside deep_ep/buffer.py (CUDA-faithful multi-QP striping). Only forward an
    # explicit user/env override here; forwarding a forced "1" would pin single-QP and
    # defeat the auto-provisioning (mpirun -genv would win over buffer.py's setdefault).
    _add_opt_genv ISHMEM_IBGDA_QPS_PER_PE      "${ISHMEM_IBGDA_QPS_PER_PE:-}"

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
        -genv ISHMEM_IBGDA_DB_BATCH_SIZE "${ISHMEM_IBGDA_DB_BATCH_SIZE:-0}" \
        -genv ISHMEM_IBGDA_BAR_BACKEND igub \
        -genv I_MPI_FABRICS shm:ofi \
        -genv FI_PROVIDER tcp \
        -genv ISHMEM_DEBUG "${ISHMEM_DEBUG:-0}" \
        "${OPT_GENV[@]}" \
        -genv DEEP_EP_NVL_RANKS "$NUM_PROCESSES" \
        -genv DEEP_EP_NVL_BYTES "$DEEP_EP_NVL_BYTES" \
        -genv DEEP_EP_RDMA_BYTES "$DEEP_EP_RDMA_BYTES" \
        -genv MASTER_ADDR "$NODE0" \
        -genv MASTER_PORT "$MASTER_PORT" \
        -genv WORLD_SIZE 2 \
        -genv TORCH_DISTRIBUTED_DEBUG OFF \
        -launcher ssh \
        "$WRAPPER_PATH" \
        python3 -u "$TEST_SCRIPT" \
            --num-processes "$NUM_PROCESSES" \
            --num-tokens "$NUM_TOKENS" \
            --hidden "$HIDDEN" \
            --num-topk "$NUM_TOPK" \
            --num-experts "$NUM_EXPERTS"
    local rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        echo "===== PASS $TEST_SCRIPT ====="
    else
        echo "===== FAIL $TEST_SCRIPT (rc=$rc) ====="
    fi
    return $rc
}

run_test
