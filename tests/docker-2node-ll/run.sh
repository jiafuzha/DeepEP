#!/usr/bin/env bash
# ===========================================================================
# Launch 2-node DeepEP internode LOW-LATENCY test using Docker containers.
#
# Each container is a separate "node" with its own hostname and bridge IP:
#   deepep-ll-node0: 172.31.1.10, GPUs 4,5, NICs mlx5_4,mlx5_5
#   deepep-ll-node1: 172.31.1.11, GPUs 6,7, NICs mlx5_6,mlx5_7
#
# mpirun is launched INSIDE deepep-ll-node0 (via docker exec); it spawns ranks
# on deepep-ll-node1 via SSH using the bridge IP. With ppn=2 the topology is:
#   - rank 0,1 on node0 -> local_rank 0,1 -> GPU 4,5 + NIC mlx5_4,mlx5_5
#   - rank 2,3 on node1 -> local_rank 0,1 -> GPU 6,7 + NIC mlx5_6,mlx5_7
#
# test_low_latency.py runs under the MPI launcher (DEEP_EP_TEST_LOW_LATENCY_NO_MPIRUN=1
# so it does NOT self-relaunch mpirun). Its multi-node launcher path reads
# MPI_LOCALRANKID as the per-node local rank and num_processes as the per-node
# rank count; init_dist() derives the global rank from RANK (node rank, set by
# node_wrapper.sh) and WORLD_SIZE (number of nodes = 2). This simulates a
# 2-node x 2-GPU low-latency deployment with --disable-nvlink (RDMA-only).
#
# Usage:
#   ./run.sh              # run default test_low_latency.py with 4 ranks
#   ./run.sh --up         # just start containers
#   ./run.sh --down       # stop containers
#   ./run.sh --shell node0|node1   # interactive shell into a node
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEEP_EP_DIR="/data/jiafuzha/code-repo/zjf2012/DeepEP"
TEST_SCRIPT="${TEST_SCRIPT:-tests/test_low_latency.py}"
SSH_DIR="/tmp/deepep-docker-ssh"

# DeepEP buffer sizes (defaults work for 2-node x 2-GPU layout)
DEEP_EP_NVL_BYTES="${DEEP_EP_NVL_BYTES:-134217728}"   # 128 MiB
DEEP_EP_RDMA_BYTES="${DEEP_EP_RDMA_BYTES:-67108864}"  # 64 MiB
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB

MASTER_PORT="${MASTER_PORT:-29500}"

# Test parameters (small by default for fast iteration).
# Low-latency defaults mirror the validated single-node loopback config.
NUM_PROCESSES="${NUM_PROCESSES:-2}"   # num local ranks per node (= ppn)
NUM_TOKENS="${NUM_TOKENS:-32}"
HIDDEN="${HIDDEN:-7168}"
NUM_TOPK="${NUM_TOPK:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-360}"

# --- Helpers ---
setup_ssh_keys() {
    if [ ! -f "$SSH_DIR/id_rsa" ]; then
        mkdir -p "$SSH_DIR"
        ssh-keygen -t rsa -N "" -f "$SSH_DIR/id_rsa" -q
        echo "Generated SSH keys in $SSH_DIR"
    fi
    cat > "$SSH_DIR/config" << 'EOF'
Host deepep-ll-node0
    HostName 172.31.1.10
    Port 22
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host deepep-ll-node1
    HostName 172.31.1.11
    Port 22
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
}

# --- Mutual exclusion: this low-latency sim and the normal-path sim
#     (tests/docker-2node) use the SAME physical XPU devices (4,5,6,7) and
#     NICs (mlx5_4..7). Running both at once makes them fight over the GPUs/NICs.
#     Remove the normal-path peer containers before bringing ours up. ---
PEER_CONTAINERS=("deepep-node0" "deepep-node1")
stop_peer_containers() {
    local removed=0
    for c in "${PEER_CONTAINERS[@]}"; do
        if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
            echo "Removing peer container $c (normal-path sim) to avoid XPU/NIC contention..."
            docker rm -f "$c" >/dev/null 2>&1 || true
            removed=1
        fi
    done
    [ "$removed" = "1" ] && sleep 1 || true
    return 0
}

up() {
    stop_peer_containers
    setup_ssh_keys
    docker compose up -d 2>&1
    sleep 3

    # Wait for SSH ready inside node0
    for i in 1 2 3 4 5; do
        if docker exec deepep-ll-node0 ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
              -o UserKnownHostsFile=/dev/null -i /root/.ssh/id_rsa \
              deepep-ll-node1 true 2>/dev/null; then
            echo "Containers ready: deepep-ll-node0 (172.31.1.10), deepep-ll-node1 (172.31.1.11)"
            return 0
        fi
        sleep 2
    done
    echo "WARNING: SSH from node0 to node1 not ready"
}

down() {
    docker compose down 2>/dev/null || true
}

ensure_up() {
    stop_peer_containers
    if ! docker ps --format '{{.Names}}' | grep -q deepep-ll-node0; then
        echo "Starting containers..."
        up
    fi
}

# --- Verify RDMA accessibility between the two containers (RoCE over physical NICs) ---
verify_rdma() {
    echo "===== Verifying RDMA accessibility ====="

    # 1. ibv devices visible in each container
    for c in deepep-ll-node0 deepep-ll-node1; do
        local devs
        devs=$(docker exec "$c" bash -lc 'ibv_devices 2>/dev/null | awk "NR>2 {print \$1}" | tr "\n" "," ' 2>/dev/null)
        if [ -z "$devs" ]; then
            echo "FAIL: $c has no IB devices visible (ibv_devices empty)"
            return 1
        fi
        echo "$c IB devices: $devs"
    done

    # 2. node0 PORT_ACTIVE check on mlx5_4
    if ! docker exec deepep-ll-node0 bash -lc 'ibv_devinfo -d mlx5_4 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: deepep-ll-node0 mlx5_4 port not active"
        return 1
    fi
    if ! docker exec deepep-ll-node1 bash -lc 'ibv_devinfo -d mlx5_6 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: deepep-ll-node1 mlx5_6 port not active"
        return 1
    fi
    echo "Both nodes report PORT_ACTIVE on assigned NICs."

    # 3. Bridge network reachability (control-plane). Use bash TCP since `ping` may be missing.
    if ! docker exec deepep-ll-node0 bash -lc 'timeout 3 bash -c ">/dev/tcp/deepep-ll-node1/22" 2>/dev/null'; then
        echo "FAIL: deepep-ll-node0 cannot reach deepep-ll-node1:22 over bridge network"
        return 1
    fi
    echo "Bridge control-plane reachable (deepep-ll-node1:22 from deepep-ll-node0)."

    # 4. Quick RoCE/RDMA loopback smoke between the two containers using ibv_rc_pingpong
    #    Uses the physical RoCE NICs (mlx5_4 on node0, mlx5_6 on node1), not the bridge IP.
    docker exec -d deepep-ll-node0 bash -lc 'pkill -f ibv_rc_pingpong 2>/dev/null; ibv_rc_pingpong -d mlx5_4 -g 3 -n 1 >/tmp/rdma_srv.log 2>&1' || true
    sleep 1
    if docker exec deepep-ll-node1 bash -lc 'timeout 10 ibv_rc_pingpong -d mlx5_6 -g 3 -n 1 deepep-ll-node0 >/tmp/rdma_cli.log 2>&1'; then
        echo "RoCE ibv_rc_pingpong mlx5_6<->mlx5_4 SUCCEEDED."
    else
        echo "WARNING: ibv_rc_pingpong over the bridge hostname did not succeed."
        echo "         (Test still proceeds; iSHMEM uses physical NICs via /dev/infiniband)."
        docker exec deepep-ll-node0 bash -lc 'tail -n 5 /tmp/rdma_srv.log 2>/dev/null' || true
        docker exec deepep-ll-node1 bash -lc 'tail -n 5 /tmp/rdma_cli.log 2>/dev/null' || true
    fi
    echo "===== RDMA accessibility check done ====="
    return 0
}

# --- Clean leaked IPC state (PSM3/oneCCL/gloo named semaphores + shm) left in
#     /dev/shm by a previously SIGKILL'd run. PSM3 locks a named POSIX semaphore
#     (sem.psm3_nic_affinity_shm_rw_mutex.*) during init; if a process is killed
#     while holding it (e.g. `timeout mpirun` on a hang/DEVICE_LOST), the next
#     run blocks FOREVER acquiring the same named semaphore -> intermittent
#     bootstrap/all_gather hang. With `ipc: host` the shared /dev/shm makes the
#     leak persist across runs and poison BOTH nodes, so we must scrub it first.
clean_ipc_state() {
    echo "===== Cleaning leaked IPC state (/dev/shm PSM3/CCL sems) ====="
    for c in deepep-ll-node0 deepep-ll-node1; do
        docker exec "$c" bash -lc '
            rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* 2>/dev/null || true
            rm -f /dev/shm/sem.ishmem* /dev/shm/ishmem* 2>/dev/null || true
            rm -f /dev/shm/*oneccl* /dev/shm/*ccl_* /dev/shm/sem.*ccl* 2>/dev/null || true
            rm -f /dev/shm/gloo* /dev/shm/sem.gloo* 2>/dev/null || true
            rm -f /tmp/deep_ep_xpu_ipc_*.sock 2>/dev/null || true
        ' 2>/dev/null || true
    done
    return 0
}

# --- Ensure MASTER_PORT is free on all nodes (xccl rendezvous). Kill only
#     processes holding the specific port (don't blindly kill all python). ---
ensure_port_free() {
    local port="$MASTER_PORT"
    echo "===== Ensuring MASTER_PORT $port free on both nodes ====="
    for c in deepep-ll-node0 deepep-ll-node1; do
        docker exec "$c" bash -lc "
            port=$port
            for attempt in 1 2 3; do
                holders=\$( (ss -lntpH 2>/dev/null || netstat -lntp 2>/dev/null | tail -n +3) \
                           | awk -v p=\":\$port\$\" '\$4 ~ p { print }' )
                if [ -z \"\$holders\" ]; then
                    echo \"[$c] port \$port is free\"
                    exit 0
                fi
                echo \"[$c] port \$port held by:\"
                echo \"\$holders\"
                pids=\$(echo \"\$holders\" | grep -oE 'pid=[0-9]+|users:\\(\\(\"[^\"]+\",pid=[0-9]+|[0-9]+/' \
                       | grep -oE '[0-9]+' | sort -u)
                if [ -n \"\$pids\" ]; then
                    echo \"[$c] killing pids holding port \$port: \$pids\"
                    kill -9 \$pids 2>/dev/null || true
                fi
                sleep 2
            done
            echo \"[$c] FAIL: port \$port still in use after 3 attempts\" >&2
            exit 1
        " || return 1
    done
    return 0
}

# --- Gate: verify iSHMEM auto GPU->NIC selection is same-PCIe-switch per rank ---
verify_nic_selection() {
    DEEP_EP_DIR="$DEEP_EP_DIR" \
    ISHMEM_DIR="${ISHMEM_DIR:-/root/.copilot/session-state/d757e418-b21f-4f96-8d86-d872b34e7e42/files/ishmem-2026-shim}" \
    NUM_PROCESSES="$NUM_PROCESSES" \
    ISHMEM_SYMMETRIC_SIZE="$ISHMEM_SYMMETRIC_SIZE" \
        bash "$SCRIPT_DIR/verify_nic_selection.sh"
}

# --- Run DeepEP internode test: launch mpirun INSIDE node0; spawn rank on node1 via SSH ---
run_test() {
    ensure_up
    verify_rdma || { echo "RDMA accessibility check failed; aborting test." >&2; return 1; }
    verify_nic_selection || {
        echo "iSHMEM auto NIC selection check FAILED (a rank's NIC is not under" >&2
        echo "the same PCIe switch as its GPU); aborting test." >&2
        return 1
    }
    clean_ipc_state
    ensure_port_free || { echo "MASTER_PORT cleanup failed; aborting test." >&2; return 1; }

    echo "===== RUN $TEST_SCRIPT (2 nodes x ${NUM_PROCESSES} ranks = $((NUM_PROCESSES * 2)) total) ====="

    local WRAPPER_PATH="$DEEP_EP_DIR/tests/docker-2node-ll/node_wrapper.sh"
    docker exec deepep-ll-node0 chmod +x "$WRAPPER_PATH" 2>/dev/null || true
    docker exec deepep-ll-node1 chmod +x "$WRAPPER_PATH" 2>/dev/null || true

    local TOTAL_RANKS=$((NUM_PROCESSES * 2))

    set +e
    docker exec \
        -e ISHMEM_DEBUG="${ISHMEM_DEBUG:-0}" \
        -e DEEP_EP_DBG_DISPATCH="${DEEP_EP_DBG_DISPATCH:-}" \
        -e DEEP_EP_DBG_COMBINE="${DEEP_EP_DBG_COMBINE:-}" \
        -e DEEP_EP_TIME_WARMUP="${DEEP_EP_TIME_WARMUP:-}" \
        deepep-ll-node0 \
        bash -lc "
            source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
            eval \"\$(conda shell.bash hook 2>/dev/null)\"
            conda activate jiafuzha_deepep 2>/dev/null || true
            cd $DEEP_EP_DIR

            timeout $TIMEOUT_SEC mpirun \
                -n $TOTAL_RANKS -ppn $NUM_PROCESSES \
                -hosts deepep-ll-node0,deepep-ll-node1 \
                -genv ISHMEM_IB_ENABLE_IBGDA 1 \
                -genv ISHMEM_IBGDA_DIRECT_DOORBELL 1 \
                -genv ISHMEM_ENABLE_GPU_IPC 0 \
                -genv ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP 0 \
                -genv ISHMEM_SYMMETRIC_SIZE $ISHMEM_SYMMETRIC_SIZE \
                -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1 \
                -genv ISHMEM_IBGDA_QPS_PER_PE 1 \
                -genv ISHMEM_IBGDA_DB_BATCH_SIZE 0 \
                -genv ISHMEM_IBGDA_BAR_BACKEND igub \
                -genv I_MPI_FABRICS shm:ofi \
                -genv FI_PROVIDER tcp \
                -genv ISHMEM_DEBUG \"\${ISHMEM_DEBUG:-0}\" \
                -genv DEEP_EP_DBG_DISPATCH \"\${DEEP_EP_DBG_DISPATCH:-}\" \
                -genv DEEP_EP_DBG_COMBINE \"\${DEEP_EP_DBG_COMBINE:-}\" \
                -genv DEEP_EP_TIME_WARMUP \"\${DEEP_EP_TIME_WARMUP:-}\" \
                -genv DEEP_EP_TEST_LOW_LATENCY_NO_MPIRUN 1 \
                -genv DEEP_EP_LL_FLAG_PROGRESS \"\${DEEP_EP_LL_FLAG_PROGRESS:-0}\" \
                -genv DEEP_EP_LL_FLAG_LSC \"\${DEEP_EP_LL_FLAG_LSC:-0}\" \
                -genv DEEP_EP_LL_FLAG_SENDER_FENCE \"\${DEEP_EP_LL_FLAG_SENDER_FENCE:-1}\" \
                -genv DEEP_EP_LL_POLL_CAP \"\${DEEP_EP_LL_POLL_CAP:-50000000}\" \
                -genv DEEP_EP_NVL_RANKS $NUM_PROCESSES \
                -genv DEEP_EP_NVL_BYTES $DEEP_EP_NVL_BYTES \
                -genv DEEP_EP_RDMA_BYTES $DEEP_EP_RDMA_BYTES \
                -genv MASTER_ADDR deepep-ll-node0 \
                -genv MASTER_PORT $MASTER_PORT \
                -genv WORLD_SIZE 2 \
                \
                \
                -genv TORCH_DISTRIBUTED_DEBUG OFF \
                -launcher ssh \
                -bootstrap-exec-args '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /root/.ssh/id_rsa' \
                $WRAPPER_PATH \
                python -u $TEST_SCRIPT \
                    --num-processes $NUM_PROCESSES \
                    --num-tokens $NUM_TOKENS \
                    --hidden $HIDDEN \
                    --num-topk $NUM_TOPK \
                    --num-experts $NUM_EXPERTS \
                    --disable-nvlink
        "
    local rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        echo "===== PASS $TEST_SCRIPT ====="
    else
        echo "===== FAIL $TEST_SCRIPT (rc=$rc) ====="
    fi
    return $rc
}

# --- Main ---
case "${1:-}" in
    --up)
        up
        ;;
    --down)
        down
        ;;
    --shell)
        ensure_up
        target="${2:-deepep-ll-node0}"
        case "$target" in
            node0|deepep-ll-node0) target=deepep-ll-node0 ;;
            node1|deepep-ll-node1) target=deepep-ll-node1 ;;
        esac
        docker exec -it "$target" bash
        ;;
    *)
        run_test
        ;;
esac
