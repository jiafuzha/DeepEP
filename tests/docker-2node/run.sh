#!/usr/bin/env bash
# ===========================================================================
# Launch 2-node DeepEP internode (NVL+RDMA) test using Docker containers.
#
# Each container is a separate "node" with its own hostname and bridge IP:
#   deepep-node0: 172.31.0.10, GPUs 4,5, NICs mlx5_4,mlx5_5
#   deepep-node1: 172.31.0.11, GPUs 6,7, NICs mlx5_6,mlx5_7
#
# mpirun is launched INSIDE deepep-node0 (via docker exec); it spawns ranks
# on deepep-node1 via SSH using the bridge IP. With ppn=2 the topology is:
#   - rank 0,1 on node0 -> local_rank 0,1 -> GPU 4,5 + NIC mlx5_4,mlx5_5
#   - rank 2,3 on node1 -> local_rank 0,1 -> GPU 6,7 + NIC mlx5_6,mlx5_7
#
# DeepEP test_internode.py reads MPI_LOCALRANKID and uses it as `local_rank`.
# Since DEEP_EP_NVL_RANKS=2, num_local_ranks=2 simulates 2 nodes x 2 GPUs.
#
# Usage:
#   ./run.sh              # run default test_internode.py with 4 ranks
#   ./run.sh --up         # just start containers
#   ./run.sh --down       # stop containers
#   ./run.sh --shell node0|node1   # interactive shell into a node
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEEP_EP_DIR="/data/jiafuzha/code-repo/zjf2012/DeepEP"
TEST_SCRIPT="${TEST_SCRIPT:-tests/test_internode.py}"
SSH_DIR="/tmp/deepep-docker-ssh"

# DeepEP buffer sizes (defaults work for 2-node x 2-GPU layout)
DEEP_EP_NVL_BYTES="${DEEP_EP_NVL_BYTES:-134217728}"   # 128 MiB
DEEP_EP_RDMA_BYTES="${DEEP_EP_RDMA_BYTES:-67108864}"  # 64 MiB
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB

# Test parameters (small by default for fast iteration)
NUM_PROCESSES="${NUM_PROCESSES:-2}"   # num local ranks (= ppn = nvl_ranks)
NUM_TOKENS="${NUM_TOKENS:-32}"
HIDDEN="${HIDDEN:-1024}"
NUM_TOPK="${NUM_TOPK:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"

# --- Helpers ---
setup_ssh_keys() {
    if [ ! -f "$SSH_DIR/id_rsa" ]; then
        mkdir -p "$SSH_DIR"
        ssh-keygen -t rsa -N "" -f "$SSH_DIR/id_rsa" -q
        echo "Generated SSH keys in $SSH_DIR"
    fi
    cat > "$SSH_DIR/config" << 'EOF'
Host deepep-node0
    HostName 172.31.0.10
    Port 22
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host deepep-node1
    HostName 172.31.0.11
    Port 22
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
}

up() {
    setup_ssh_keys
    docker compose up -d 2>&1
    sleep 3

    # Wait for SSH ready inside node0
    for i in 1 2 3 4 5; do
        if docker exec deepep-node0 ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
              -o UserKnownHostsFile=/dev/null -i /root/.ssh/id_rsa \
              deepep-node1 true 2>/dev/null; then
            echo "Containers ready: deepep-node0 (172.31.0.10), deepep-node1 (172.31.0.11)"
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
    if ! docker ps --format '{{.Names}}' | grep -q deepep-node0; then
        echo "Starting containers..."
        up
    fi
}

# --- Verify RDMA accessibility between the two containers (RoCE over physical NICs) ---
verify_rdma() {
    echo "===== Verifying RDMA accessibility ====="

    # 1. ibv devices visible in each container
    for c in deepep-node0 deepep-node1; do
        local devs
        devs=$(docker exec "$c" bash -lc 'ibv_devices 2>/dev/null | awk "NR>2 {print \$1}" | tr "\n" "," ' 2>/dev/null)
        if [ -z "$devs" ]; then
            echo "FAIL: $c has no IB devices visible (ibv_devices empty)"
            return 1
        fi
        echo "$c IB devices: $devs"
    done

    # 2. node0 PORT_ACTIVE check on mlx5_4
    if ! docker exec deepep-node0 bash -lc 'ibv_devinfo -d mlx5_4 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: deepep-node0 mlx5_4 port not active"
        return 1
    fi
    if ! docker exec deepep-node1 bash -lc 'ibv_devinfo -d mlx5_6 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: deepep-node1 mlx5_6 port not active"
        return 1
    fi
    echo "Both nodes report PORT_ACTIVE on assigned NICs."

    # 3. Bridge network reachability (control-plane). Use bash TCP since `ping` may be missing.
    if ! docker exec deepep-node0 bash -lc 'timeout 3 bash -c ">/dev/tcp/deepep-node1/22" 2>/dev/null'; then
        echo "FAIL: deepep-node0 cannot reach deepep-node1:22 over bridge network"
        return 1
    fi
    echo "Bridge control-plane reachable (deepep-node1:22 from deepep-node0)."

    # 4. Quick RoCE/RDMA loopback smoke between the two containers using ibv_rc_pingpong
    #    Uses the physical RoCE NICs (mlx5_4 on node0, mlx5_6 on node1), not the bridge IP.
    docker exec -d deepep-node0 bash -lc 'pkill -f ibv_rc_pingpong 2>/dev/null; ibv_rc_pingpong -d mlx5_4 -g 3 -n 1 >/tmp/rdma_srv.log 2>&1' || true
    sleep 1
    if docker exec deepep-node1 bash -lc 'timeout 10 ibv_rc_pingpong -d mlx5_6 -g 3 -n 1 deepep-node0 >/tmp/rdma_cli.log 2>&1'; then
        echo "RoCE ibv_rc_pingpong mlx5_6<->mlx5_4 SUCCEEDED."
    else
        echo "WARNING: ibv_rc_pingpong over the bridge hostname did not succeed."
        echo "         (Test still proceeds; iSHMEM uses physical NICs via /dev/infiniband)."
        docker exec deepep-node0 bash -lc 'tail -n 5 /tmp/rdma_srv.log 2>/dev/null' || true
        docker exec deepep-node1 bash -lc 'tail -n 5 /tmp/rdma_cli.log 2>/dev/null' || true
    fi
    echo "===== RDMA accessibility check done ====="
    return 0
}

# --- Run DeepEP internode test: launch mpirun INSIDE node0; spawn rank on node1 via SSH ---
run_test() {
    ensure_up
    verify_rdma || { echo "RDMA accessibility check failed; aborting test." >&2; return 1; }

    echo "===== RUN $TEST_SCRIPT (2 nodes x ${NUM_PROCESSES} ranks = $((NUM_PROCESSES * 2)) total) ====="

    local WRAPPER_PATH="$DEEP_EP_DIR/tests/docker-2node/node_wrapper.sh"
    docker exec deepep-node0 chmod +x "$WRAPPER_PATH" 2>/dev/null || true
    docker exec deepep-node1 chmod +x "$WRAPPER_PATH" 2>/dev/null || true

    local TOTAL_RANKS=$((NUM_PROCESSES * 2))

    set +e
    docker exec \
        -e ISHMEM_DEBUG="${ISHMEM_DEBUG:-0}" \
        deepep-node0 \
        bash -lc "
            source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
            eval \"\$(conda shell.bash hook 2>/dev/null)\"
            conda activate jiafuzha_deepep 2>/dev/null || true
            cd $DEEP_EP_DIR

            timeout $TIMEOUT_SEC mpirun \
                -n $TOTAL_RANKS -ppn $NUM_PROCESSES \
                -hosts deepep-node0,deepep-node1 \
                -genv ISHMEM_IB_ENABLE_IBGDA 1 \
                -genv ISHMEM_IBGDA_DIRECT_DOORBELL 1 \
                -genv ISHMEM_ENABLE_GPU_IPC 0 \
                -genv ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP 1 \
                -genv ISHMEM_SYMMETRIC_SIZE $ISHMEM_SYMMETRIC_SIZE \
                -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1 \
                -genv ISHMEM_IBGDA_QPS_PER_PE 1 \
                -genv ISHMEM_IBGDA_DB_BATCH_SIZE 0 \
                -genv ISHMEM_IBGDA_BAR_BACKEND igub \
                -genv I_MPI_FABRICS shm:ofi \
                -genv FI_PROVIDER tcp \
                -genv ISHMEM_DEBUG \"\${ISHMEM_DEBUG:-0}\" \
                -genv DEEP_EP_NVL_RANKS $NUM_PROCESSES \
                -genv DEEP_EP_NVL_BYTES $DEEP_EP_NVL_BYTES \
                -genv DEEP_EP_RDMA_BYTES $DEEP_EP_RDMA_BYTES \
                -genv MASTER_ADDR deepep-node0 \
                -genv MASTER_PORT 29500 \
                -genv WORLD_SIZE 2 \
                -launcher ssh \
                -bootstrap-exec-args '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /root/.ssh/id_rsa' \
                $WRAPPER_PATH \
                python -u $TEST_SCRIPT \
                    --num-processes $NUM_PROCESSES \
                    --num-tokens $NUM_TOKENS \
                    --hidden $HIDDEN \
                    --num-topk $NUM_TOPK \
                    --num-experts $NUM_EXPERTS
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
        target="${2:-deepep-node0}"
        case "$target" in
            node0|deepep-node0) target=deepep-node0 ;;
            node1|deepep-node1) target=deepep-node1 ;;
        esac
        docker exec -it "$target" bash
        ;;
    *)
        run_test
        ;;
esac
