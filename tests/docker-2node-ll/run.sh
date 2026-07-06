#!/usr/bin/env bash
# ===========================================================================
# Launch 2-node DeepEP internode LOW-LATENCY test using Docker containers.
#
# for smc26
# Each container is a separate "node" with its own hostname and bridge IP:
#   deepep-ll-node0: 172.31.1.10, GPUs 0,1, NICs mlx5_0,mlx5_1
#   deepep-ll-node1: 172.31.1.11, GPUs 2,3, NICs mlx5_2,mlx5_3
# for 140
# Each container is a separate "node" with its own hostname and SSH port on
# the host network namespace:
#   deepep-ll-node0: 127.0.0.1:2310, GPUs 4,5, NICs mlx5_4,mlx5_5
#   deepep-ll-node1: 127.0.0.1:2311, GPUs 6,7, NICs mlx5_6,mlx5_7

#
# mpirun is launched INSIDE deepep-ll-node0 (via docker exec); it spawns ranks
# on deepep-ll-node1 via SSH using the bridge IP. With ppn=2 the topology is:
#   - rank 0,1 on node0 -> local_rank 0,1 -> GPU 0,1 + NIC mlx5_0,mlx5_1
#   - rank 2,3 on node1 -> local_rank 0,1 -> GPU 2,3 + NIC mlx5_2,mlx5_3
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

DEEP_EP_DIR="/root/jiafuzha/code-repo/zjf2012/DeepEP"
TEST_SCRIPT="${TEST_SCRIPT:-tests/test_low_latency.py}"
SSH_DIR="/tmp/deepep-docker-ssh"
ISHMEM_DIR="${ISHMEM_DIR:-/root/jiafuzha/ishmem_ibgda/build/_install}"

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

# Optional clean-env reset before each test run. On this BMG + mlx5 stack a
# DEVICE_LOST / init-hang in one run leaves NIC/QP + GPU page-table state wedged,
# which cascades into subsequent runs (the flag path is especially sensitive).
# Reloading the igub_vmem BAR-bridge driver fully resets that state. Opt in with
# DEEP_EP_LL_RESET_DRIVER=1 (host must allow rmmod/insmod). Override the module
# path via DEEP_EP_IGUB_KO and the settle time via DEEP_EP_IGUB_SETTLE_SEC.
DEEP_EP_LL_RESET_DRIVER="${DEEP_EP_LL_RESET_DRIVER:-0}"
DEEP_EP_IGUB_KO="${DEEP_EP_IGUB_KO:-/root/jiafuzha/code-repo/intel_gpu_uar_bridge/driver/igub_vmem_drv.ko}"
DEEP_EP_IGUB_SETTLE_SEC="${DEEP_EP_IGUB_SETTLE_SEC:-40}"

# --- Helpers ---
init_ib_device_config() {
    read -r -a NODE0_IB_DEVICES_ARR <<< "${NODE0_IB_DEVICES:-mlx5_4 mlx5_5}"
    read -r -a NODE1_IB_DEVICES_ARR <<< "${NODE1_IB_DEVICES:-mlx5_6 mlx5_7}"
    NODE0_IB_DEVICES_STR="${NODE0_IB_DEVICES:-mlx5_4 mlx5_5}"
    NODE1_IB_DEVICES_STR="${NODE1_IB_DEVICES:-mlx5_6 mlx5_7}"
    if [ "${#NODE0_IB_DEVICES_ARR[@]}" -eq 0 ] || [ "${#NODE1_IB_DEVICES_ARR[@]}" -eq 0 ]; then
        echo "NODE0_IB_DEVICES and NODE1_IB_DEVICES must each contain at least one IB device" >&2
        exit 1
    fi
}

init_ib_device_config
NODE0_PRIMARY_IB_DEVICE="${NODE0_IB_DEVICES_ARR[0]}"
NODE1_PRIMARY_IB_DEVICE="${NODE1_IB_DEVICES_ARR[0]}"

setup_ssh_keys() {
    if [ ! -f "$SSH_DIR/id_rsa" ]; then
        mkdir -p "$SSH_DIR"
        ssh-keygen -t rsa -N "" -f "$SSH_DIR/id_rsa" -q
        echo "Generated SSH keys in $SSH_DIR"
    fi
    cat > "$SSH_DIR/config" << 'EOF'
Host deepep-ll-node0
    HostName 127.0.0.1
    Port 2310
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null

Host deepep-ll-node1
    HostName 127.0.0.1
    Port 2311
    User root
    IdentityFile /root/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
}

# --- Mutual exclusion: this low-latency sim and the normal-path sim
#     (tests/docker-2node) use the SAME physical XPU devices (0,1,2,3) and
#     NICs (mlx5_0..3). Running both at once makes them fight over the GPUs/NICs.
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
            echo "Containers ready: deepep-ll-node0 (127.0.0.1:2310), deepep-ll-node1 (127.0.0.1:2311)"
            return 0
        fi
        sleep 2
    done
    echo "WARNING: SSH from node0 to node1 not ready"
}

down() {
    docker compose down 2>/dev/null || true
}

# Reload the igub_vmem BAR-bridge driver to fully reset wedged NIC/QP/GPU state.
# Brings containers down first (must release the driver refcount), reloads, then
# lets the driver settle. Callers should bring containers back up afterwards.
reset_igub_driver() {
    echo "===== DEEP_EP_LL_RESET_DRIVER=1: reloading igub_vmem driver for a clean env ====="
    down
    sleep 3
    local rc
    rc="$(cat /sys/module/igub_vmem_drv/refcnt 2>/dev/null)"
    if [ "${rc:-0}" != "0" ] && [ -n "$rc" ]; then
        echo "  igub_vmem_drv refcnt=$rc (busy); skipping reload" >&2
        return 0
    fi
    if lsmod | grep -q '^igub_vmem_drv'; then
        rmmod igub_vmem_drv 2>/dev/null && echo "  rmmod igub_vmem_drv ok" \
            || { echo "  rmmod FAILED (continuing without reset)" >&2; return 0; }
    fi
    if [ -f "$DEEP_EP_IGUB_KO" ]; then
        insmod "$DEEP_EP_IGUB_KO" 2>/dev/null && echo "  insmod $DEEP_EP_IGUB_KO ok" \
            || echo "  insmod FAILED" >&2
    else
        echo "  module $DEEP_EP_IGUB_KO not found; skipping insmod" >&2
    fi
    echo "  settling ${DEEP_EP_IGUB_SETTLE_SEC}s..."
    sleep "$DEEP_EP_IGUB_SETTLE_SEC"
    local gpus
    gpus="$(timeout 20 sycl-ls 2>/dev/null | grep -c 'level_zero.*gpu')"
    echo "  post-reset: GPUs=$gpus refcnt=$(cat /sys/module/igub_vmem_drv/refcnt 2>/dev/null)"
}

ensure_up() {
    stop_peer_containers
    if [ "$DEEP_EP_LL_RESET_DRIVER" = "1" ]; then
        reset_igub_driver
        up
        return 0
    fi
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

    # for smc26
    # 2. node0 PORT_ACTIVE check on mlx5_0
    if ! docker exec deepep-ll-node0 bash -lc 'ibv_devinfo -d mlx5_0 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: deepep-ll-node0 mlx5_0 port not active"
        return 1
    fi
    if ! docker exec deepep-ll-node1 bash -lc 'ibv_devinfo -d mlx5_2 2>/dev/null | grep -q PORT_ACTIVE'; then
        echo "FAIL: deepep-ll-node1 mlx5_2 port not active"
    # for 140
    # 2. Verify the assigned IB ports are ACTIVE. Use sysfs instead of
    #    ibv_devinfo here because the container bind-mounts host RDMA userspace
    #    libs, and ibv_devinfo can fail with libibverbs ABI mismatches even when
    #    the device itself is visible and active.
    #if ! docker exec deepep-ll-node0 bash -lc "grep -q 'ACTIVE' /sys/class/infiniband/$NODE0_PRIMARY_IB_DEVICE/ports/1/state"; then
    #    echo "FAIL: deepep-ll-node0 $NODE0_PRIMARY_IB_DEVICE port not active"
    #    return 1
    #fi
    #if ! docker exec deepep-ll-node1 bash -lc "grep -q 'ACTIVE' /sys/class/infiniband/$NODE1_PRIMARY_IB_DEVICE/ports/1/state"; then
    #    echo "FAIL: deepep-ll-node1 $NODE1_PRIMARY_IB_DEVICE port not active"
        return 1
    fi
    echo "Both nodes report PORT_ACTIVE on assigned NICs."

    # 3. Bridge network reachability (control-plane). Use bash TCP since `ping` may be missing.
    if ! docker exec deepep-ll-node0 bash -lc 'timeout 3 bash -c ">/dev/tcp/deepep-ll-node1/2311" 2>/dev/null'; then
        echo "FAIL: deepep-ll-node0 cannot reach deepep-ll-node1:2311 over host network"
        return 1
    fi
    echo "Control-plane reachable (deepep-ll-node1:2311 from deepep-ll-node0)."

    # 4. Quick RoCE/RDMA loopback smoke between the two containers using ibv_rc_pingpong
    # for smc26
    #    Uses the physical RoCE NICs (mlx5_0 on node0, mlx5_2 on node1), not the bridge IP.
    docker exec -d deepep-ll-node0 bash -lc 'pkill -f ibv_rc_pingpong 2>/dev/null; ibv_rc_pingpong -d mlx5_0 -g 3 -n 1 >/tmp/rdma_srv.log 2>&1' || true
    sleep 1
    if docker exec deepep-ll-node1 bash -lc 'timeout 10 ibv_rc_pingpong -d mlx5_2 -g 3 -n 1 deepep-ll-node0 >/tmp/rdma_cli.log 2>&1'; then
        echo "RoCE ibv_rc_pingpong mlx5_2<->mlx5_0 SUCCEEDED."
    # for 140
    #    Uses the physical RoCE NICs (mlx5_4 on node0, mlx5_6 on node1), not the bridge IP.
    #docker exec -d deepep-ll-node0 bash -lc "pkill -f ibv_rc_pingpong 2>/dev/null; ibv_rc_pingpong -d $NODE0_PRIMARY_IB_DEVICE -g 3 -n 1 >/tmp/rdma_srv.log 2>&1" || true
    #sleep 1
    #if docker exec deepep-ll-node1 bash -lc "timeout 10 ibv_rc_pingpong -d $NODE1_PRIMARY_IB_DEVICE -g 3 -n 1 deepep-ll-node0 >/tmp/rdma_cli.log 2>&1"; then
    #    echo "RoCE ibv_rc_pingpong $NODE1_PRIMARY_IB_DEVICE<->$NODE0_PRIMARY_IB_DEVICE SUCCEEDED."
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
    # for smc26
    ISHMEM_DIR="${ISHMEM_DIR:-/root/jiafuzha/code-repo/ishmem_ibgda/build/_install}" \
    # for 140
    #ISHMEM_DIR="$ISHMEM_DIR" \
    NUM_PROCESSES="$NUM_PROCESSES" \
    NODE0_IB_DEVICES="$NODE0_IB_DEVICES_STR" \
    NODE1_IB_DEVICES="$NODE1_IB_DEVICES_STR" \
    ISHMEM_SYMMETRIC_SIZE="$ISHMEM_SYMMETRIC_SIZE" \
        bash "$SCRIPT_DIR/verify_nic_selection.sh"
}

# GPU->NIC PCIe affinity is a STATIC topology property: once it verifies for a
# given container set it cannot change until the containers are recreated. The
# gate spawns its OWN full ishmem_init/IBGDA-provisioning mpirun, which on this
# stack intermittently hangs (rc=124 in QP provisioning) on a cold container or
# right after a prior DEVICE_LOST -- a FALSE failure that needlessly aborts an
# otherwise-good test run. So: (1) retry the transient gate a few times, and
# (2) cache the PASS in a stamp file inside node0 for this container lifetime
# (a fresh `docker compose up` starts with an empty /tmp, auto-invalidating it).
NIC_CHECK_STAMP="/tmp/.deepep_ll_nic_check_ok"
verify_nic_selection_cached() {
    if docker exec deepep-ll-node0 test -f "$NIC_CHECK_STAMP" 2>/dev/null; then
        echo "iSHMEM NIC selection: cached PASS for this container lifetime; skipping re-check." >&2
        return 0
    fi
    local attempts="${NIC_CHECK_ATTEMPTS:-3}"
    local a
    for a in $(seq 1 "$attempts"); do
        if verify_nic_selection; then
            docker exec deepep-ll-node0 bash -lc "touch $NIC_CHECK_STAMP" 2>/dev/null || true
            return 0
        fi
        echo "iSHMEM NIC selection attempt $a/$attempts failed (transient IBGDA init hang); retrying..." >&2
        # Clear stale gate procs left inside both containers by the timed-out mpirun.
        for c in deepep-ll-node0 deepep-ll-node1; do
            docker exec "$c" bash -lc 'pkill -9 -f "nic_pcie_check|pmi_proxy|hydra|mpiexec" 2>/dev/null || true' 2>/dev/null || true
        done
        sleep 2
    done
    return 1
}

# --- Run DeepEP internode test: launch mpirun INSIDE node0; spawn rank on node1 via SSH ---
run_test() {
    ensure_up
    verify_rdma || { echo "RDMA accessibility check failed; aborting test." >&2; return 1; }
    if [ -n "${SKIP_NIC_CHECK:-}" ]; then
        echo "SKIP_NIC_CHECK set: skipping iSHMEM auto NIC selection pre-flight." >&2
    else
        verify_nic_selection_cached || {
            echo "iSHMEM auto NIC selection check FAILED (a rank's NIC is not under" >&2
            echo "the same PCIe switch as its GPU); aborting test." >&2
            return 1
        }
    fi
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
        -e ISHMEM_DIR="$ISHMEM_DIR" \
        -e NODE0_IB_DEVICES="$NODE0_IB_DEVICES_STR" \
        -e NODE1_IB_DEVICES="$NODE1_IB_DEVICES_STR" \
        -e DEEP_EP_DBG_DISPATCH="${DEEP_EP_DBG_DISPATCH:-}" \
        -e DEEP_EP_DBG_COMBINE="${DEEP_EP_DBG_COMBINE:-}" \
        -e DEEP_EP_TIME_WARMUP="${DEEP_EP_TIME_WARMUP:-}" \
        -e DEEP_EP_SKIP_WARMUP="${DEEP_EP_SKIP_WARMUP:-}" \
        deepep-ll-node0 \
        bash -lc "
            source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
            eval \"\$(conda shell.bash hook 2>/dev/null)\"
            conda activate jiafuzha_deepep 2>/dev/null || true
            export ISHMEM_DIR=$ISHMEM_DIR
            export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}:\$ISHMEM_DIR/lib
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
                -genv ISHMEM_DIR $ISHMEM_DIR \
                -genv NODE0_IB_DEVICES '$NODE0_IB_DEVICES_STR' \
                -genv NODE1_IB_DEVICES '$NODE1_IB_DEVICES_STR' \
                -genv ISHMEM_DEBUG \"\${ISHMEM_DEBUG:-0}\" \
                -genv DEEP_EP_DBG_DISPATCH \"\${DEEP_EP_DBG_DISPATCH:-}\" \
                -genv DEEP_EP_DBG_COMBINE \"\${DEEP_EP_DBG_COMBINE:-}\" \
                -genv DEEP_EP_TIME_WARMUP \"\${DEEP_EP_TIME_WARMUP:-}\" \
                -genv DEEP_EP_SKIP_WARMUP \"\${DEEP_EP_SKIP_WARMUP:-}\" \
                -genv DEEP_EP_TEST_LOW_LATENCY_NO_MPIRUN 1 \
                -genv DEEP_EP_LL_FLAG_PROGRESS \"\${DEEP_EP_LL_FLAG_PROGRESS:-0}\" \
                -genv DEEP_EP_LL_FLAG_LSC \"\${DEEP_EP_LL_FLAG_LSC:-0}\" \
                -genv DEEP_EP_LL_FLAG_SENDER_FENCE \"\${DEEP_EP_LL_FLAG_SENDER_FENCE:-1}\" \
                -genv DEEP_EP_LL_FLAG_RECV_ACQ \"\${DEEP_EP_LL_FLAG_RECV_ACQ:-1}\" \
                -genv DEEP_EP_LL_POLL_CAP \"\${DEEP_EP_LL_POLL_CAP:-1000000}\" \
                -genv DEEP_EP_NVL_RANKS $NUM_PROCESSES \
                -genv DEEP_EP_NVL_BYTES $DEEP_EP_NVL_BYTES \
                -genv DEEP_EP_RDMA_BYTES $DEEP_EP_RDMA_BYTES \
                -genv MASTER_ADDR deepep-ll-node0 \
                -genv MASTER_PORT $MASTER_PORT \
                -genv WORLD_SIZE 2 \
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
