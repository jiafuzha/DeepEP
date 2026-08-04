#!/bin/bash
# Per-rank wrapper executed inside each MPI rank (docker-2node-v3, NORMAL path).
#
# Same as v2 node_wrapper.sh but for GPU 4-7 + NIC mlx5_4-7 slice.
#
# Compose sets per node:
#   node0: ZE_AFFINITY_MASK=4,5  NODE_MLX5_HCAS=mlx5_4,mlx5_5  NODE_IB_IFACES=ens910f0np0,ens910f1np1  SIM_NODE_RANK=0
#   node1: ZE_AFFINITY_MASK=6,7  NODE_MLX5_HCAS=mlx5_6,mlx5_7  NODE_IB_IFACES=ens911f0np0,ens911f1np1  SIM_NODE_RANK=1
#
# Within a node, MPI_LOCALRANKID (0..PPN-1) selects the local GPU/NIC/iface.
set -uo pipefail

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

LOCAL_RANK=${MPI_LOCALRANKID:-0}
export LOCAL_RANK

export RANK=${RANK:-${SIM_NODE_RANK:-0}}

# --- Per-rank GPU/NIC/iface selection from compose environment ---
IFS=',' read -r -a NODE_HCAS <<< "${NODE_MLX5_HCAS:-}"
IFS=',' read -r -a NODE_IFACES <<< "${NODE_IB_IFACES:-}"

if [ "${#NODE_HCAS[@]}" -eq 0 ]; then
    echo "node_wrapper: NODE_MLX5_HCAS is empty; cannot pin IBGDA NIC" >&2
    exit 1
fi
if [ "$LOCAL_RANK" -ge "${#NODE_HCAS[@]}" ]; then
    echo "node_wrapper: LOCAL_RANK=$LOCAL_RANK exceeds NODE_MLX5_HCAS=${NODE_MLX5_HCAS}" >&2
    exit 1
fi

export ISHMEM_IBGDA_NIC="${NODE_HCAS[$LOCAL_RANK]}"
if [ "${#NODE_IFACES[@]}" -gt "$LOCAL_RANK" ]; then
    export FI_VERBS_IFACE="${NODE_IFACES[$LOCAL_RANK]}"
fi

export PYTHONPATH=/root/jiafuzha/code-repo/zjf2012/DeepEP:${PYTHONPATH:-}
export MASTER_ADDR=${MASTER_ADDR:-deepep-v3-node0}

echo "[$(hostname) lr=$LOCAL_RANK gr=${PMI_RANK:-?}] ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-unset} NIC=$ISHMEM_IBGDA_NIC IFACE=${FI_VERBS_IFACE:-unset}" >&2

# --- FAULT-MODE VM ---
if [ "${DEEP_EP_XPU_FAULT_MODE:-1}" != "0" ]; then
    export NEOReadDebugKeys=1
    export EnableRecoverablePageFaults=1
fi

ulimit -c unlimited
exec "$@"