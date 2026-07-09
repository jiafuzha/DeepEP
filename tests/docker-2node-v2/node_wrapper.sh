#!/bin/bash
# Per-rank wrapper executed inside each MPI rank (docker-2node-v2, NORMAL path).
#
# Recreated for the isolated `deepep_jiafuzha` image: sources ONLY the in-image
# oneAPI (no host conda / no host oneAPI mount) and derives the per-rank GPU and
# IBGDA NIC from environment set by docker-compose, instead of hostname-hardcoded
# device tables.
#
# Compose sets per node (defaults = smc26):
#   node0: ZE_AFFINITY_MASK=0,1  NODE_MLX5_HCAS=mlx5_0,mlx5_1  NODE_IB_IFACES=ens1006f0np0,ens1006f1np1  SIM_NODE_RANK=0
#   node1: ZE_AFFINITY_MASK=2,3  NODE_MLX5_HCAS=mlx5_2,mlx5_3  NODE_IB_IFACES=ens2005f0np0,ens2005f1np1  SIM_NODE_RANK=1
#
# Within a node, MPI_LOCALRANKID (0..PPN-1) selects the local GPU/NIC/iface.
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

# In the host-network 2-node simulation the full host NIC topology is visible to
# every rank, so pin the IBGDA NIC explicitly per local rank to match the known
# GPU<->NIC pairing (node0 lr0->GPU0<->mlx5_0, lr1->GPU1<->mlx5_1; node1 GPU2/3).
export ISHMEM_IBGDA_NIC="${NODE_HCAS[$LOCAL_RANK]}"
if [ "${#NODE_IFACES[@]}" -gt "$LOCAL_RANK" ]; then
    export FI_VERBS_IFACE="${NODE_IFACES[$LOCAL_RANK]}"
fi

# DeepEP needs PYTHONPATH to find the in-tree deep_ep package.
export PYTHONPATH=/root/jiafuzha/code-repo/zjf2012/DeepEP:${PYTHONPATH:-}

# DeepEP / iSHMEM needs MASTER_ADDR (rank-0 node hostname).
export MASTER_ADDR=${MASTER_ADDR:-deepep-v2-node0}

echo "[$(hostname) lr=$LOCAL_RANK gr=${PMI_RANK:-?}] ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-unset} NIC=$ISHMEM_IBGDA_NIC IFACE=${FI_VERBS_IFACE:-unset}" >&2

ulimit -c unlimited
exec "$@"
