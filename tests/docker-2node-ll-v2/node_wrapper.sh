#!/bin/bash
# Per-rank wrapper executed inside each MPI rank (docker-2node-ll-v2, LOW-LATENCY).
#
# Recreated for the isolated `deepep_jiafuzha` image: sources ONLY the in-image
# oneAPI (no host conda / no host oneAPI mount) and derives per-rank GPU/iface
# from environment set by docker-compose.
#
# Compose sets per node (defaults = smc26):
#   node0: ZE_AFFINITY_MASK=0,1  NODE_MLX5_HCAS=mlx5_0,mlx5_1  NODE_IB_IFACES=ens1006f0np0,ens1006f1np1  SIM_NODE_RANK=0
#   node1: ZE_AFFINITY_MASK=2,3  NODE_MLX5_HCAS=mlx5_2,mlx5_3  NODE_IB_IFACES=ens2005f0np0,ens2005f1np1  SIM_NODE_RANK=1
set -uo pipefail

# Restore per-node identity (SIM_NODE_RANK / ZE_AFFINITY_MASK / NODE_MLX5_HCAS /
# NODE_IB_IFACES). The docker-compose per-container `environment:` is NOT
# inherited by mpirun's SSH-launched ranks, so PID1 dumps it to this file.
if [ -f /run/deepep_node_env ]; then
    set +u
    source /run/deepep_node_env
    set -u
fi

# --- In-image oneAPI only (no conda) ---
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +u
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    set -u
fi
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ISHMEM_DIR=${ISHMEM_DIR:-/root/jiafuzha/code-repo/ishmem_ibgda/build/_install}
export LD_LIBRARY_PATH=${ISHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}
if [ -f /usr/lib/x86_64-linux-gnu/libhwloc.so ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libhwloc.so${LD_PRELOAD:+ $LD_PRELOAD}"
fi

# Each container has its own hostname / IPC namespace, so MPI assigns
# MPI_LOCALRANKID = 0..PPN-1 within each container correctly.
LOCAL_RANK=${MPI_LOCALRANKID:-0}
export LOCAL_RANK

# init_dist (tests/utils.py) reads RANK as the *node* rank (0..WORLD_SIZE-1);
# WORLD_SIZE (set by run.sh) is the number of nodes.
export RANK=${RANK:-${SIM_NODE_RANK:-0}}

# --- Per-rank iface selection from compose environment ---
IFS=',' read -r -a NODE_IFACES <<< "${NODE_IB_IFACES:-}"
if [ "$LOCAL_RANK" -lt "${#NODE_IFACES[@]}" ]; then
    export FI_VERBS_IFACE="${NODE_IFACES[$LOCAL_RANK]}"
fi

# NOTE: ISHMEM_IBGDA_NIC is intentionally NOT set here. iSHMEM auto-selects the
# NIC closest to the chosen GPU by PCIe topology, which is correct in this 2-node
# simulation since each node only exposes its own GPUs (0,1 or 2,3). The
# auto-selection is independently verified before each test run by
# verify_nic_selection.sh / nic_pcie_check (run.sh), which asserts the
# auto-picked NIC shares the GPU's PCIe switch.

# DeepEP needs PYTHONPATH to find the in-tree deep_ep package.
export PYTHONPATH=/root/jiafuzha/code-repo/zjf2012/DeepEP:${PYTHONPATH:-}

# DeepEP / iSHMEM needs MASTER_ADDR (rank-0 node hostname).
export MASTER_ADDR=${MASTER_ADDR:-deepep-ll-v2-node0}

echo "[$(hostname) lr=$LOCAL_RANK gr=${PMI_RANK:-?}] ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-unset} IFACE=${FI_VERBS_IFACE:-unset}" >&2

# --- FAULT-MODE VM (root-cause fix for the LL DEVICE_LOST) ------------------
# ROOT CAUSE (proven by DEBUG xe CONFIG_DRM_XE_DEBUG_VM + clean A/B):
#   The DeepEP process VM is a preempt-fence (long-running) VM because IBGDA
#   runs a persistent long-running poll/quiet exec queue on it. Any BO
#   bind/rebind/eviction/migration on that VM during the run triggers xe's
#   preempt_rebind_work_func, which must SUSPEND the busy-spinning IBGDA LR
#   queue and migrate BOs via the bcs copy engine. Under contention that
#   returns -16 (EBUSY) -> xe calls xe_vm_kill(vm) -> resets ALL exec queues on
#   the VM (incl. bcs) -> "engine_class=bcs" reset -> UR_RESULT_ERROR_DEVICE_LOST.
# FIX: run the DeepEP XPU process on a FAULT-MODE VM (recoverable page faults).
#   Fault-mode binds pages on demand via the pagefault handler and NEVER runs a
#   preempt-rebind (never suspends the LR IBGDA queue) -> the kill can't happen.
#   Verified 100% pass (8/8 no-reset; OFF/ON/OFF/ON clean-reset A/B all flip) at
#   FULL performance (~1171 us @ H7168, == the good baseline). This is standard
#   GPU on-demand paging (NOT a CPU proxy). Opt out with DEEP_EP_XPU_FAULT_MODE=0.
if [ "${DEEP_EP_XPU_FAULT_MODE:-1}" != "0" ]; then
    export NEOReadDebugKeys=1
    export EnableRecoverablePageFaults=1
fi

# --- L0/UR/NEO debug instrumentation (Stage 1, no rebuild) ---
# Enabled by touching .l0debug in this dir (bind-mounted, visible in both
# containers). Captures the exact failing ze/UR call + faulting VA per rank.
_DBGDIR=/root/jiafuzha/code-repo/zjf2012/DeepEP/tests/docker-2node-ll-v2
if [ -f "${_DBGDIR}/.l0debug" ]; then
    # Light, low-overhead instrumentation so the timing race still reproduces.
    # NEO prints the faulting VA/context on a GPU page fault instead of an opaque
    # DEVICE_LOST; recoverable faults keep the context alive long enough to report.
    export NEOReadDebugKeys=1
    export PrintDebugMessages=1
    if [ -f "${_DBGDIR}/.no_recoverable" ]; then
        export EnableRecoverablePageFaults=0
    else
        export EnableRecoverablePageFaults=1
    fi
    export PrintDeviceAndDriverInfo=0
    export ZE_ENABLE_VALIDATION_LAYER="${DEEP_EP_L0_VALLAYER:-0}"
    export ZE_ENABLE_PARAMETER_VALIDATION="${DEEP_EP_L0_VALLAYER:-0}"
    [ -n "${DEEP_EP_L0_URTRACE:-}" ] && export UR_L0_DEBUG=-1 && export ZE_DEBUG=4
    _GR=${PMI_RANK:-${RANK:-0}}_${LOCAL_RANK}
    mkdir -p "${_DBGDIR}/debuglogs" 2>/dev/null || true
    echo "[node_wrapper] L0 DEBUG ENABLED, rank=$_GR -> debuglogs/l0_${_GR}.log" >&2
    ulimit -c unlimited
    exec "$@" 2> >(tee "${_DBGDIR}/debuglogs/l0_${_GR}.log" >&2)
fi

ulimit -c unlimited
exec "$@"
