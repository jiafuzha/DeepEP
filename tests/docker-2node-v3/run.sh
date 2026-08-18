#!/usr/bin/env bash
# ===========================================================================
# Launch 2-node DeepEP internode (NVL+RDMA, NORMAL path) test  -- v3
#
# Same design as docker-2node-v2 but targets the HIGHER 4-GPU/NIC slice
# (GPUs 4-7, NICs mlx5_4..7) on 8-GPU hosts, so it can coexist with v2
# (GPUs 0-3) without hardware contention.
#
# Each container is a separate "node" with its own hostname and SSH port on the
# host network namespace:
#   deepep-v3-node0: 127.0.0.1:2330, GPUs 4,5, NICs mlx5_4,mlx5_5
#   deepep-v3-node1: 127.0.0.1:2331, GPUs 6,7, NICs mlx5_6,mlx5_7
#
# mpirun is launched INSIDE deepep-v3-node0 (via docker exec) and spawns ranks
# on deepep-v3-node1 via SSH. With ppn=2 the topology is:
#   - rank 0,1 on node0 -> local_rank 0,1 -> GPU 4,5 + NIC mlx5_4,mlx5_5
#   - rank 2,3 on node1 -> local_rank 0,1 -> GPU 6,7 + NIC mlx5_6,mlx5_7
#
# Usage:
#   ./run.sh              # run default tests/test_internode.py with 4 ranks
#   ./run.sh --up         # just start containers
#   ./run.sh --down       # stop containers
#   ./run.sh --shell node0|node1   # interactive shell into a node
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEEP_EP_DIR="/root/jiafuzha/code-repo/zjf2012/DeepEP"
TEST_SCRIPT="${TEST_SCRIPT:-tests/test_internode.py}"
SSH_DIR="/tmp/deepep-docker-ssh"
ISHMEM_DIR="${ISHMEM_DIR:-/root/jiafuzha/code-repo/ishmem_ibgda/build/_install}"

NODE0_CONTAINER="deepep-v3-node0"
NODE1_CONTAINER="deepep-v3-node1"
NODE0_PORT=2330
NODE1_PORT=2331

# igub_vmem BAR-bridge driver (loaded on host so containers stay lib/module-free).
DEEP_EP_IGUB_KO="${DEEP_EP_IGUB_KO:-/root/jiafuzha/code-repo/intel_gpu_uar_bridge/driver/igub_vmem_drv.ko}"

# DeepEP buffer sizes (defaults work for 2-node x 2-GPU layout)
DEEP_EP_NVL_BYTES="${DEEP_EP_NVL_BYTES:-134217728}"   # 128 MiB
DEEP_EP_RDMA_BYTES="${DEEP_EP_RDMA_BYTES:-67108864}"  # 64 MiB
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB

MASTER_PORT="${MASTER_PORT:-29500}"

# Test parameters (small by default for fast iteration)
NUM_PROCESSES="${NUM_PROCESSES:-2}"   # num local ranks (= ppn = nvl_ranks)
NUM_TOKENS="${NUM_TOKENS:-32}"
HIDDEN="${HIDDEN:-1024}"
NUM_TOPK="${NUM_TOPK:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-360}"

FI_PROVIDER_VAL="${FI_PROVIDER:-tcp}"

setup_ssh_keys() {
    if [ ! -f "$SSH_DIR/id_rsa" ]; then
        mkdir -p "$SSH_DIR"
        ssh-keygen -t rsa -N "" -f "$SSH_DIR/id_rsa" -q
        echo "Generated SSH keys in $SSH_DIR"
    fi
    cat > "$SSH_DIR/config" << EOF
Host $NODE0_CONTAINER
    HostName 127.0.0.1
    Port $NODE0_PORT
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host $NODE1_CONTAINER
    HostName 127.0.0.1
    Port $NODE1_PORT
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
}

# --- Load the igub_vmem BAR-bridge driver on the HOST (idempotent) ---
ensure_igub_driver() {
    if lsmod 2>/dev/null | grep -q '^igub_vmem_drv'; then
        return 0
    fi
    if [ -f "$DEEP_EP_IGUB_KO" ]; then
        echo "Loading igub_vmem driver on host: $DEEP_EP_IGUB_KO"
        insmod "$DEEP_EP_IGUB_KO" 2>/dev/null || echo "  insmod failed (already loaded or insufficient privilege?)" >&2
    else
        echo "  igub module $DEEP_EP_IGUB_KO not found; skipping (container may lack /dev/igub_vmem)" >&2
    fi
    [ -e /dev/igub_vmem ] || echo "  WARNING: /dev/igub_vmem not present on host" >&2
}

# --- Mutual exclusion: all 2-node sims share the same physical XPUs and NICs.
#     The v3 harness uses GPUs 4-7 + NICs mlx5_4..7, which are independent of
#     the v2 harness (GPUs 0-3 + NICs mlx5_0..3). Still remove ALL peer sim
#     containers in case another v3 or LL instance is already running on our slice.
#     NOTE: we do NOT remove v2 containers — they use a different GPU/NIC slice. ---
PEER_CONTAINERS=("deepep-v3-node0" "deepep-v3-node1" "deepep-ll-v3-node0" "deepep-ll-v3-node1")
stop_peer_containers() {
    local removed=0
    for c in "${PEER_CONTAINERS[@]}"; do
        if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
            echo "Removing peer container $c to avoid XPU/NIC contention..."
            docker rm -f "$c" >/dev/null 2>&1 || true
            removed=1
        fi
    done
    [ "$removed" = "1" ] && sleep 1 || true
    return 0
}

up() {
    stop_peer_containers
    ensure_igub_driver
    setup_ssh_keys
    if [ "${DEEP_EP_FORCE_BUILD:-0}" = "1" ] || ! docker image inspect "${IMAGE_NAME:-deepep_jiafuzha}" >/dev/null 2>&1; then
        docker compose up -d --build 2>&1
    else
        docker compose up -d 2>&1
    fi
    sleep 3

    for i in 1 2 3 4 5; do
        if docker exec "$NODE0_CONTAINER" ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
              -o UserKnownHostsFile=/dev/null -i /root/.ssh/id_rsa \
              "$NODE1_CONTAINER" true 2>/dev/null; then
            echo "Containers ready: $NODE0_CONTAINER (127.0.0.1:$NODE0_PORT), $NODE1_CONTAINER (127.0.0.1:$NODE1_PORT)"
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
    ensure_igub_driver
    if ! docker ps --format '{{.Names}}' | grep -q "$NODE0_CONTAINER"; then
        echo "Starting containers..."
        up
    fi
}

# --- Verify RDMA accessibility between the two containers (RoCE over physical NICs) ---
verify_rdma() {
    echo "===== Verifying RDMA accessibility ====="

    for c in "$NODE0_CONTAINER" "$NODE1_CONTAINER"; do
        local devs
        devs=$(docker exec "$c" bash -lc 'ibv_devices 2>/dev/null | awk "NR>2 {print \$1}" | tr "\n" "," ' 2>/dev/null)
        if [ -z "$devs" ]; then
            echo "FAIL: $c has no IB devices visible (ibv_devices empty)"
            return 1
        fi
        echo "$c IB devices: $devs"
    done

    local n0_nic n1_nic
    n0_nic=$(docker exec "$NODE0_CONTAINER" bash -lc 'echo ${NODE_MLX5_HCAS%%,*}')
    n1_nic=$(docker exec "$NODE1_CONTAINER" bash -lc 'echo ${NODE_MLX5_HCAS%%,*}')
    if ! docker exec "$NODE0_CONTAINER" bash -lc "ibv_devinfo -d $n0_nic 2>/dev/null | grep -q PORT_ACTIVE"; then
        echo "FAIL: $NODE0_CONTAINER $n0_nic port not active"
        return 1
    fi
    if ! docker exec "$NODE1_CONTAINER" bash -lc "ibv_devinfo -d $n1_nic 2>/dev/null | grep -q PORT_ACTIVE"; then
        echo "FAIL: $NODE1_CONTAINER $n1_nic port not active"
        return 1
    fi
    echo "Both nodes report PORT_ACTIVE on assigned NICs ($n0_nic / $n1_nic)."

    if ! docker exec "$NODE0_CONTAINER" bash -lc "timeout 3 bash -c \">/dev/tcp/$NODE1_CONTAINER/$NODE1_PORT\" 2>/dev/null"; then
        echo "FAIL: $NODE0_CONTAINER cannot reach $NODE1_CONTAINER:$NODE1_PORT over host network"
        return 1
    fi
    echo "Control-plane reachable ($NODE1_CONTAINER:$NODE1_PORT from $NODE0_CONTAINER)."

    docker exec -d "$NODE0_CONTAINER" bash -lc "pkill -f ibv_rc_pingpong 2>/dev/null; ibv_rc_pingpong -d $n0_nic -g 3 -n 1 >/tmp/rdma_srv.log 2>&1" || true
    sleep 1
    if docker exec "$NODE1_CONTAINER" bash -lc "timeout 10 ibv_rc_pingpong -d $n1_nic -g 3 -n 1 $NODE0_CONTAINER >/tmp/rdma_cli.log 2>&1"; then
        echo "RoCE ibv_rc_pingpong $n1_nic<->$n0_nic SUCCEEDED."
    else
        echo "WARNING: ibv_rc_pingpong over the host hostname did not succeed."
        docker exec "$NODE0_CONTAINER" bash -lc 'tail -n 5 /tmp/rdma_srv.log 2>/dev/null' || true
        docker exec "$NODE1_CONTAINER" bash -lc 'tail -n 5 /tmp/rdma_cli.log 2>/dev/null' || true
    fi
    echo "===== RDMA accessibility check done ====="
    return 0
}

# --- Clean leaked IPC state ---
clean_ipc_state() {
    echo "===== Cleaning leaked IPC state (/dev/shm PSM3/CCL sems) ====="
    for c in "$NODE0_CONTAINER" "$NODE1_CONTAINER"; do
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

# --- Ensure MASTER_PORT is free on both nodes ---
ensure_port_free() {
    local port="$MASTER_PORT"
    echo "===== Ensuring MASTER_PORT $port free on both nodes ====="
    for c in "$NODE0_CONTAINER" "$NODE1_CONTAINER"; do
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

# --- Sync torch editable metadata into both containers ---
TORCH_META_STASH="/root/jiafuzha/.deepep-v3-torch-meta"
sync_torch_metadata() {
    local stash="$TORCH_META_STASH"
    docker exec "$NODE0_CONTAINER" test -d "$stash" 2>/dev/null || {
        echo "NOTE: torch metadata stash $stash not found; skipping metadata sync." >&2
        return 0
    }
    local want
    want=$(docker exec "$NODE0_CONTAINER" bash -lc "ls -d $stash/torch-*.dist-info 2>/dev/null | xargs -r -n1 basename" | head -1)
    [ -n "$want" ] || { echo "NOTE: no torch dist-info in stash; skipping." >&2; return 0; }
    for c in "$NODE0_CONTAINER" "$NODE1_CONTAINER"; do
        docker exec "$c" bash -lc '
            set -e
            SP=/usr/local/lib/python3.12/dist-packages
            STASH='"$stash"'
            WANT='"$want"'
            if [ ! -d "$SP/$WANT" ]; then
                rm -rf "$SP"/torch-*.dist-info "$SP"/__editable__.torch-*.pth "$SP"/__editable___torch_*_finder.py
                cp -a "$STASH"/. "$SP"/
            fi
        ' || echo "WARNING: torch metadata sync failed on $c" >&2
    done
}

# --- Run DeepEP internode test ---
run_test() {
    ensure_up
    sync_torch_metadata
    verify_rdma || { echo "RDMA accessibility check failed; aborting test." >&2; return 1; }
    clean_ipc_state
    ensure_port_free || { echo "MASTER_PORT cleanup failed; aborting test." >&2; return 1; }

    echo "===== RUN $TEST_SCRIPT (2 nodes x ${NUM_PROCESSES} ranks = $((NUM_PROCESSES * 2)) total) ====="

    local WRAPPER_PATH="$DEEP_EP_DIR/tests/docker-2node-v3/node_wrapper.sh"
    docker exec "$NODE0_CONTAINER" chmod +x "$WRAPPER_PATH" 2>/dev/null || true
    docker exec "$NODE1_CONTAINER" chmod +x "$WRAPPER_PATH" 2>/dev/null || true

    local TOTAL_RANKS=$((NUM_PROCESSES * 2))

    set +e
    docker exec \
        -e ISHMEM_DEBUG="${ISHMEM_DEBUG:-0}" \
        -e ISHMEM_DIR="$ISHMEM_DIR" \
        -e DEEP_EP_DBG_DISPATCH="${DEEP_EP_DBG_DISPATCH:-}" \
        -e DEEP_EP_DBG_DROP="${DEEP_EP_DBG_DROP:-}" \
        -e DEEP_EP_MIN="${DEEP_EP_MIN:-}" \
        -e DEEP_EP_DBG_COMBINE="${DEEP_EP_DBG_COMBINE:-}" \
        -e DEEP_EP_PERF="${DEEP_EP_PERF:-}" \
        -e DEEP_EP_TIME_WARMUP="${DEEP_EP_TIME_WARMUP:-}" \
        -e DEEP_EP_SKIP_WARMUP="${DEEP_EP_SKIP_WARMUP:-}" \
        "$NODE0_CONTAINER" \
        bash -lc "
            source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
            export ISHMEM_DIR=$ISHMEM_DIR
            export LD_LIBRARY_PATH=\$ISHMEM_DIR/lib:\${LD_LIBRARY_PATH:-}
            cd $DEEP_EP_DIR

            timeout $TIMEOUT_SEC mpirun \
                -n $TOTAL_RANKS -ppn $NUM_PROCESSES \
                -hosts $NODE0_CONTAINER,$NODE1_CONTAINER \
                -genv ISHMEM_IB_ENABLE_IBGDA 1 \
                -genv ISHMEM_IBGDA_DIRECT_DOORBELL 1 \
                -genv ISHMEM_ENABLE_GPU_IPC 0 \
                -genv ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP 0 \
                -genv ISHMEM_SYMMETRIC_SIZE $ISHMEM_SYMMETRIC_SIZE \
                -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1 \
                -genv ISHMEM_IBGDA_QPS_PER_PE 1 \
                -genv ISHMEM_IBGDA_DB_BATCH_SIZE 0 \
                -genv ISHMEM_IBGDA_QUIET_SKIP_DRAIN \${ISHMEM_IBGDA_QUIET_SKIP_DRAIN:-1} \
                -genv ISHMEM_IBGDA_BAR_BACKEND igub \
                -genv I_MPI_FABRICS shm:ofi \
                -genv FI_PROVIDER '$FI_PROVIDER_VAL' \
                -genv ISHMEM_DIR $ISHMEM_DIR \
                -genv ISHMEM_DEBUG \"\${ISHMEM_DEBUG:-0}\" \
                -genv DEEP_EP_DBG_DISPATCH \"\${DEEP_EP_DBG_DISPATCH:-}\" \
                -genv DEEP_EP_DBG_DROP \"\${DEEP_EP_DBG_DROP:-}\" \
                -genv DEEP_EP_MIN \"\${DEEP_EP_MIN:-}\" \
                -genv DEEP_EP_DBG_COMBINE \"\${DEEP_EP_DBG_COMBINE:-}\" \
                -genv DEEP_EP_PERF \"\${DEEP_EP_PERF:-}\" \
                -genv DEEP_EP_TIME_WARMUP \"\${DEEP_EP_TIME_WARMUP:-}\" \
                -genv DEEP_EP_SKIP_WARMUP \"\${DEEP_EP_SKIP_WARMUP:-}\" \
                -genv DEEP_EP_NVL_RANKS $NUM_PROCESSES \
                -genv DEEP_EP_NVL_BYTES $DEEP_EP_NVL_BYTES \
                -genv DEEP_EP_RDMA_BYTES $DEEP_EP_RDMA_BYTES \
                -genv MASTER_ADDR $NODE0_CONTAINER \
                -genv MASTER_PORT $MASTER_PORT \
                -genv WORLD_SIZE 2 \
                -genv TORCH_DISTRIBUTED_DEBUG OFF \
                -launcher ssh \
                -bootstrap-exec-args '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /root/.ssh/id_rsa' \
                $WRAPPER_PATH \
                python3 -u $TEST_SCRIPT \
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
        target="${2:-$NODE0_CONTAINER}"
        case "$target" in
            node0|"$NODE0_CONTAINER") target="$NODE0_CONTAINER" ;;
            node1|"$NODE1_CONTAINER") target="$NODE1_CONTAINER" ;;
        esac
        docker exec -it "$target" bash
        ;;
    *)
        run_test
        ;;
esac