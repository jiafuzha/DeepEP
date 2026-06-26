#!/bin/bash
# Per-rank wrapper executed inside each MPI rank.
#
# Sets up oneAPI/conda env, ZE_AFFINITY_MASK and ISHMEM_IBGDA_NIC based on
# container hostname (node identity) and MPI_LOCALRANKID (rank within node).
#
# Hardware topology (NUMA 1):
#   GPU 4 (0000:ac:00.0) <-> mlx5_4 (0000:b2:00.0)
#   GPU 5 (0000:b0:00.0) <-> mlx5_5 (0000:b2:00.1)
#   GPU 6 (0000:bd:00.0) <-> mlx5_6 (0000:ba:00.0)
#   GPU 7 (0000:c1:00.0) <-> mlx5_7 (0000:ba:00.1)
#
# Node 0 (deepep-coh-node0): GPUs 4,5 + NICs mlx5_4,mlx5_5
# Node 1 (deepep-coh-node1): GPUs 6,7 + NICs mlx5_6,mlx5_7

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
# Use miniforge3 for conda env
if [ -f /root/miniforge3/etc/profile.d/conda.sh ]; then
    source /root/miniforge3/etc/profile.d/conda.sh
fi
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null
conda activate jiafuzha_deepep 2>/dev/null
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

# Each container has its own hostname / network namespace, so MPI assigns
# MPI_LOCALRANKID = 0..PPN-1 within each container correctly.
LOCAL_RANK=${MPI_LOCALRANKID:-0}
export LOCAL_RANK

# Detect node identity from container hostname (set by docker-compose).
HOSTNAME_VAL=$(hostname)
case "$HOSTNAME_VAL" in
    *node1*|*node-1*)
        IFACES=(ens5008f0np0 ens5008f1np1)
        export ZE_AFFINITY_MASK=6,7
        NODE_RANK=1
        ;;
    *)
        IFACES=(ens4013f0np0 ens4013f1np1)
        export ZE_AFFINITY_MASK=4,5
        NODE_RANK=0
        ;;
esac

# init_dist (tests/utils.py) reads RANK as the *node* rank (0..WORLD_SIZE-1).
# WORLD_SIZE is set by run.sh to the number of nodes.
export RANK=${RANK:-$NODE_RANK}

# EXPLICIT same-PCIe-domain GPU<->NIC pinning (coherence experiment).
# Unlike the normal sim, we do NOT rely on iSHMEM NIC auto-selection. We pin
# ISHMEM_IBGDA_NIC so each rank's GPU and NIC sit under the SAME PCIe switch:
#   switch [a8-b4]: GPU4 (ac) + GPU5 (b0) + NIC b2 -> mlx5_4, mlx5_5
#   switch [b8-c5]: GPU6 (bd) + GPU7 (c1) + NIC ba -> mlx5_6, mlx5_7
# => GPU i <-> mlx5_i is same-domain. With ZE_AFFINITY_MASK={4,5}|{6,7} and
#    resolve_device_index = LOCAL_RANK % num_devices, local rank L on node N
#    uses GPU (BASE+L) and must use NIC mlx5_(BASE+L), BASE = 4 (node0) | 6 (node1).
NIC_BASE=$(( NODE_RANK == 1 ? 6 : 4 ))
export ISHMEM_IBGDA_NIC="mlx5_$(( NIC_BASE + LOCAL_RANK ))"
export FI_VERBS_IFACE=${IFACES[$LOCAL_RANK]}

# DeepEP needs PYTHONPATH to find the in-tree deep_ep package
export PYTHONPATH=/data/jiafuzha/code-repo/zjf2012/DeepEP:${PYTHONPATH:-}

# DeepEP / iSHMEM needs MASTER_ADDR (rank-0 node hostname)
export MASTER_ADDR=${MASTER_ADDR:-deepep-coh-node0}

echo "[$(hostname) lr=$LOCAL_RANK gr=${PMI_RANK:-?}] ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK NIC=$ISHMEM_IBGDA_NIC IFACE=$FI_VERBS_IFACE" >&2

ulimit -c unlimited
exec "$@"
