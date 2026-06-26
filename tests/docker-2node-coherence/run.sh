#!/usr/bin/env bash
# ===========================================================================
# Launch 2-node iSHMEM COHERENCE experiment using Docker containers.
#
# Purpose: empirically test whether the GPU can read a flag/payload written by
# an EXTERNAL NIC RDMA write using system-scope acquire (atomic_fence / atomic_ref)
# vs uncached loads, when GPU and NIC are pinned to the SAME PCIe domain. This
# revisits the claim that "system-scope acquire triggers DEVICE_LOST / only
# uc_load works". It runs the ishmem_ibgda UT `xpu_ishmem_mapped_api_ut` cases:
#   ext_flag_cached / ext_flag_fence_sys / ext_flag_aref_sys / ext_flag_ucload
#
# Each container is a separate "node" with its own hostname and bridge IP:
#   deepep-coh-node0: 172.31.2.10, GPUs 4,5, NICs mlx5_4,mlx5_5
#   deepep-coh-node1: 172.31.2.11, GPUs 6,7, NICs mlx5_6,mlx5_7
#
# SAME-PCIe-DOMAIN pinning (node_wrapper.sh sets ISHMEM_IBGDA_NIC explicitly):
#   switch [a8-b4]: GPU4(ac)+GPU5(b0)+NIC b2 -> mlx5_4, mlx5_5
#   switch [b8-c5]: GPU6(bd)+GPU7(c1)+NIC ba -> mlx5_6, mlx5_7
#   => rank local L on node N uses GPU(BASE+L) <-> mlx5_(BASE+L) (same switch).
#
# Usage:
#   ./run.sh              # build + run all four ext_flag_* coherence cases
#   CASES="ext_flag_aref_sys" ./run.sh   # run a single case
#   ./run.sh --up         # just start containers
#   ./run.sh --down       # stop containers
#   ./run.sh --shell node0|node1   # interactive shell into a node
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEEP_EP_DIR="/data/jiafuzha/code-repo/zjf2012/DeepEP"
SSH_DIR="/tmp/deepep-coh-docker-ssh"

# DeepEP buffer sizes (defaults work for 2-node x 2-GPU layout)
DEEP_EP_NVL_BYTES="${DEEP_EP_NVL_BYTES:-134217728}"   # 128 MiB
DEEP_EP_RDMA_BYTES="${DEEP_EP_RDMA_BYTES:-67108864}"  # 64 MiB
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"  # 256 MiB

MASTER_PORT="${MASTER_PORT:-29520}"

# --- iSHMEM coherence UT (built from the mounted ishmem_ibgda repo) ---
ISHMEM_REPO="${ISHMEM_REPO:-/data/jiafuzha/ishmem_ibgda}"
ISHMEM_DIR="${ISHMEM_DIR:-/root/.copilot/session-state/d757e418-b21f-4f96-8d86-d872b34e7e42/files/ishmem-2026-shim}"
UT_BIN="$ISHMEM_REPO/build/test/unit/deepep/xpu_ishmem_mapped_api_ut"
UT_BUILD="$ISHMEM_REPO/test/unit/deepep/build_xpu_ishmem_mapped_api_ut.sh"
# Cases to run (space-separated). The four ext_flag_* cases are the experiment.
CASES="${CASES:-ext_flag_ucload ext_flag_aref_sys ext_flag_fence_sys}"
UT_NUM_ELEMS="${UT_NUM_ELEMS:-64}"
UT_PEER_MODE="${UT_PEER_MODE:-all2all}"
UT_SPIN_CAP="${UT_SPIN_CAP:-200000000}"

# Test parameters (small by default for fast iteration)
NUM_PROCESSES="${NUM_PROCESSES:-2}"   # num local ranks (= ppn)
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"

# --- Helpers ---
setup_ssh_keys() {
    if [ ! -f "$SSH_DIR/id_rsa" ]; then
        mkdir -p "$SSH_DIR"
        ssh-keygen -t rsa -N "" -f "$SSH_DIR/id_rsa" -q
        echo "Generated SSH keys in $SSH_DIR"
    fi
    cat > "$SSH_DIR/config" << 'EOF'
Host deepep-coh-node0
    HostName 172.31.2.10
    Port 22
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host deepep-coh-node1
    HostName 172.31.2.11
    Port 22
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
}

# --- Mutual exclusion: this coherence sim and the other 2-node sims
#     (tests/docker-2node, tests/docker-2node-ll) use the SAME physical XPU
#     devices (4,5,6,7) and NICs (mlx5_4..7). Running concurrently makes them
#     fight over the GPUs/NICs. Remove BOTH peer sims' containers first. ---
PEER_CONTAINERS=("deepep-node0" "deepep-node1" "deepep-ll-node0" "deepep-ll-node1")
stop_peer_containers() {
    local removed=0
    for c in "${PEER_CONTAINERS[@]}"; do
        if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
            echo "Removing peer container $c (low-latency sim) to avoid XPU/NIC contention..."
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
        if docker exec deepep-coh-node0 ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
              -o UserKnownHostsFile=/dev/null -i /root/.ssh/id_rsa \
              deepep-coh-node1 true 2>/dev/null; then
            echo "Containers ready: deepep-coh-node0 (172.31.2.10), deepep-coh-node1 (172.31.2.11)"
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
    if ! docker ps --format '{{.Names}}' | grep -q deepep-coh-node0; then
        echo "Starting containers..."
        up
    fi
}

# --- Verify RDMA accessibility between the two containers (RoCE over physical NICs) ---
verify_rdma() {
    echo "===== Verifying RDMA accessibility ====="

    # 1. ibv devices visible in each container
    for c in deepep-coh-node0 deepep-coh-node1; do
        local devs
        devs=$(docker exec "$c" bash -lc 'ibv_devices 2>/dev/null | awk "NR>2 {print \$1}" | tr "\n" "," ' 2>/dev/null)
        if [ -z "$devs" ]; then
            echo "FAIL: $c has no IB devices visible (ibv_devices empty)"
            return 1
        fi
        echo "$c IB devices: $devs"
    done

    # 2. node0 PORT_ACTIVE check on mlx5_4
    if ! docker exec deepep-coh-node0 bash -lc 'ibv_devinfo -d mlx5_4 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: deepep-coh-node0 mlx5_4 port not active"
        return 1
    fi
    if ! docker exec deepep-coh-node1 bash -lc 'ibv_devinfo -d mlx5_6 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: deepep-coh-node1 mlx5_6 port not active"
        return 1
    fi
    echo "Both nodes report PORT_ACTIVE on assigned NICs."

    # 3. Bridge network reachability (control-plane). Use bash TCP since `ping` may be missing.
    if ! docker exec deepep-coh-node0 bash -lc 'timeout 3 bash -c ">/dev/tcp/deepep-coh-node1/22" 2>/dev/null'; then
        echo "FAIL: deepep-coh-node0 cannot reach deepep-coh-node1:22 over bridge network"
        return 1
    fi
    echo "Bridge control-plane reachable (deepep-coh-node1:22 from deepep-coh-node0)."

    # 4. Quick RoCE/RDMA loopback smoke between the two containers using ibv_rc_pingpong
    #    Uses the physical RoCE NICs (mlx5_4 on node0, mlx5_6 on node1), not the bridge IP.
    docker exec -d deepep-coh-node0 bash -lc 'pkill -f ibv_rc_pingpong 2>/dev/null; ibv_rc_pingpong -d mlx5_4 -g 3 -n 1 >/tmp/rdma_srv.log 2>&1' || true
    sleep 1
    if docker exec deepep-coh-node1 bash -lc 'timeout 10 ibv_rc_pingpong -d mlx5_6 -g 3 -n 1 deepep-coh-node0 >/tmp/rdma_cli.log 2>&1'; then
        echo "RoCE ibv_rc_pingpong mlx5_6<->mlx5_4 SUCCEEDED."
    else
        echo "WARNING: ibv_rc_pingpong over the bridge hostname did not succeed."
        echo "         (Test still proceeds; iSHMEM uses physical NICs via /dev/infiniband)."
        docker exec deepep-coh-node0 bash -lc 'tail -n 5 /tmp/rdma_srv.log 2>/dev/null' || true
        docker exec deepep-coh-node1 bash -lc 'tail -n 5 /tmp/rdma_cli.log 2>/dev/null' || true
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
    for c in deepep-coh-node0 deepep-coh-node1; do
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
    for c in deepep-coh-node0 deepep-coh-node1; do
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

# --- Build the ishmem coherence UT inside node0 (repo is bind-mounted) ---
build_ut() {
    echo "===== Building iSHMEM coherence UT ($UT_BIN) ====="
    docker exec deepep-coh-node0 bash -lc "
        source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
        eval \"\$(conda shell.bash hook 2>/dev/null)\"
        conda activate jiafuzha_deepep 2>/dev/null || true
        export ISHMEM_DIR=$ISHMEM_DIR
        export PKG_CONFIG_PATH=$ISHMEM_DIR/lib/pkgconfig:\${PKG_CONFIG_PATH:-}
        bash $UT_BUILD
    " || { echo "UT build failed" >&2; return 1; }
    return 0
}

# --- Run the iSHMEM coherence UT: launch mpirun INSIDE node0; spawn rank on node1 via SSH ---
run_test() {
    ensure_up
    verify_rdma || { echo "RDMA accessibility check failed; aborting test." >&2; return 1; }
    clean_ipc_state
    ensure_port_free || { echo "MASTER_PORT cleanup failed; aborting test." >&2; return 1; }
    build_ut || return 1

    local WRAPPER_PATH="$DEEP_EP_DIR/tests/docker-2node-coherence/node_wrapper.sh"
    docker exec deepep-coh-node0 chmod +x "$WRAPPER_PATH" 2>/dev/null || true
    docker exec deepep-coh-node1 chmod +x "$WRAPPER_PATH" 2>/dev/null || true

    local TOTAL_RANKS=$((NUM_PROCESSES * 2))
    local overall=0

    for CASE in $CASES; do
        echo
        echo "===== RUN coherence UT case=$CASE (2 nodes x ${NUM_PROCESSES} ranks = $TOTAL_RANKS total) ====="
        clean_ipc_state
        ensure_port_free || { echo "port cleanup failed" >&2; overall=1; continue; }
        set +e
        docker exec \
            -e ISHMEM_DEBUG="${ISHMEM_DEBUG:-0}" \
            deepep-coh-node0 \
            bash -lc "
                source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
                eval \"\$(conda shell.bash hook 2>/dev/null)\"
                conda activate jiafuzha_deepep 2>/dev/null || true
                cd $DEEP_EP_DIR

                timeout $TIMEOUT_SEC mpirun \
                    -n $TOTAL_RANKS -ppn $NUM_PROCESSES \
                    -hosts deepep-coh-node0,deepep-coh-node1 \
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
                    -genv MASTER_ADDR deepep-coh-node0 \
                    -genv MASTER_PORT $MASTER_PORT \
                    -genv WORLD_SIZE 2 \
                    -genv UT_PEER_MODE $UT_PEER_MODE \
                    -genv UT_SPIN_CAP $UT_SPIN_CAP \
                    -launcher ssh \
                    -bootstrap-exec-args '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i /root/.ssh/id_rsa' \
                    $WRAPPER_PATH \
                    $UT_BIN --case $CASE --num-elems $UT_NUM_ELEMS
            "
        local rc=$?
        set -e
        if [ $rc -eq 0 ]; then
            echo "===== PASS case=$CASE ====="
        else
            echo "===== FAIL case=$CASE (rc=$rc) ====="
            overall=1
        fi
    done
    return $overall
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
        target="${2:-deepep-coh-node0}"
        case "$target" in
            node0|deepep-coh-node0) target=deepep-coh-node0 ;;
            node1|deepep-coh-node1) target=deepep-coh-node1 ;;
        esac
        docker exec -it "$target" bash
        ;;
    *)
        run_test
        ;;
esac
