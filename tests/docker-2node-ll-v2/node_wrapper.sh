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

ulimit -c unlimited
exec "$@"
