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
# Node 0 (deepep-node0): GPUs 4,5 + NICs mlx5_4,mlx5_5
# Node 1 (deepep-node1): GPUs 6,7 + NICs mlx5_6,mlx5_7

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
# Use miniforge3 for conda env
if [ -f /root/miniforge3/etc/profile.d/conda.sh ]; then
    source /root/miniforge3/etc/profile.d/conda.sh
fi
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null
conda activate jiafuzha_deepep 2>/dev/null
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ISHMEM_DIR=${ISHMEM_DIR:-/root/jiafuzha/ishmem_ibgda/build/_install}
export ISHMEM_IBGDA_QUIET_SKIP_DRAIN=${ISHMEM_IBGDA_QUIET_SKIP_DRAIN:-1}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${ISHMEM_DIR}/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libhwloc.so${LD_PRELOAD:+ $LD_PRELOAD}"

resolve_iface_from_ibdev() {
    local ibdev="$1"
    local net_dir="/sys/class/infiniband/$ibdev/device/net"
    if [ ! -d "$net_dir" ]; then
        echo "Missing sysfs netdev mapping for IB device $ibdev" >&2
        return 1
    fi
    local iface
    iface=$(ls "$net_dir" 2>/dev/null | head -n 1)
    if [ -z "$iface" ]; then
        echo "No netdev found for IB device $ibdev" >&2
        return 1
    fi
    printf '%s\n' "$iface"
}

# Each container has its own hostname / network namespace, so MPI assigns
# MPI_LOCALRANKID = 0..PPN-1 within each container correctly.
LOCAL_RANK=${MPI_LOCALRANKID:-0}
export LOCAL_RANK

# Detect node identity from container hostname (set by docker-compose).
HOSTNAME_VAL=$(hostname)
case "$HOSTNAME_VAL" in
    *node1*|*node-1*)
# for smc26
        IFACES=(ens2005f0np0 ens2005f1np1)
        IBDEVS=(mlx5_2 mlx5_3)
        export ZE_AFFINITY_MASK=2,3
        NODE_RANK=1
        ;;
    *)
        IFACES=(ens1006f0np0 ens1006f1np1)
        IBDEVS=(mlx5_0 mlx5_1)
        export ZE_AFFINITY_MASK=0,1
# for 140
#        NODE_IB_DEVICES_STR=${NODE1_IB_DEVICES:-mlx5_6 mlx5_7}
#        export ZE_AFFINITY_MASK=6,7
#        NODE_RANK=1
#        ;;
#    *)
#        NODE_IB_DEVICES_STR=${NODE0_IB_DEVICES:-mlx5_4 mlx5_5}
#        export ZE_AFFINITY_MASK=4,5
        NODE_RANK=0
        ;;
esac
read -r -a NODE_IB_DEVICES <<< "$NODE_IB_DEVICES_STR"
# init_dist (tests/utils.py) reads RANK as the *node* rank (0..WORLD_SIZE-1).
# WORLD_SIZE is set by run.sh to the number of nodes.
export RANK=${RANK:-$NODE_RANK}

# In the host-network 2-node simulation the full host NIC topology is visible
# to every rank. Pin the IBGDA NIC explicitly per local rank to match the
# known GPU<->NIC pairing used by the lower-level docker dispatch reproducer:
#   node0 lr0->GPU4<->mlx5_4, lr1->GPU5<->mlx5_5
#   node1 lr0->GPU6<->mlx5_6, lr1->GPU7<->mlx5_7
if [ "$LOCAL_RANK" -ge "${#IFACES[@]}" ]; then
    echo "LOCAL_RANK=$LOCAL_RANK exceeds configured IFACES: ${IFACES[*]}" >&2
    exit 1
fi
IB_DEVICE=${IBDEVS[$LOCAL_RANK]}
export ISHMEM_IBGDA_NIC="$IB_DEVICE"
export FI_VERBS_IFACE=${IFACES[$LOCAL_RANK]}

# DeepEP needs PYTHONPATH to find the in-tree deep_ep package
export PYTHONPATH=/root/jiafuzha/code-repo/zjf2012/DeepEP:${PYTHONPATH:-}

# DeepEP / iSHMEM needs MASTER_ADDR (rank-0 node hostname)
export MASTER_ADDR=${MASTER_ADDR:-deepep-node0}

echo "[$(hostname) lr=$LOCAL_RANK gr=${PMI_RANK:-?}] ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK NIC=$ISHMEM_IBGDA_NIC IFACE=$FI_VERBS_IFACE" >&2

ulimit -c unlimited
exec "$@"
