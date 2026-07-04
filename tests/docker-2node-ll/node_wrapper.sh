#!/bin/bash
# Per-rank wrapper executed inside each MPI rank.
#
# Sets up oneAPI/conda env, ZE_AFFINITY_MASK and ISHMEM_IBGDA_NIC based on
# container hostname (node identity) and MPI_LOCALRANKID (rank within node).
#
# Hardware topology (NUMA 0):
#   GPU 0 (0000:1f:00.0) <-> mlx5_0 (0000:25:00.0)
#   GPU 1 (0000:23:00.0) <-> mlx5_1 (0000:25:00.1)
#   GPU 2 (0000:42:00.0) <-> mlx5_2 (0000:48:00.0)
#   GPU 3 (0000:46:00.0) <-> mlx5_3 (0000:48:00.1)
#
# Node 0 (deepep-ll-node0): GPUs 0,1 + NICs mlx5_0,mlx5_1
# Node 1 (deepep-ll-node1): GPUs 2,3 + NICs mlx5_2,mlx5_3

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
# Use miniforge3 for conda env
if [ -f /root/miniforge3/etc/profile.d/conda.sh ]; then
    source /root/miniforge3/etc/profile.d/conda.sh
fi
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null
conda activate jiafuzha_deepep 2>/dev/null
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ISHMEM_DIR=${ISHMEM_DIR:-/root/jiafuzha/ishmem_ibgda/build/_install}
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
# for smc
        IFACES=(ens2005f0np0 ens2005f1np1)
        export ZE_AFFINITY_MASK=2,3
        NODE_RANK=1
        ;;
    *)
        IFACES=(ens1006f0np0 ens1006f1np1)
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

# NOTE: ISHMEM_IBGDA_NIC is intentionally NOT set here. iSHMEM auto-selects the
# NIC closest to the chosen GPU by PCIe topology, which is correct in this
# 2-node simulation since each node only exposes its own GPUs (0,1 or 2,3).
# The auto-selection is independently verified before each test run by
# verify_nic_selection.sh / nic_pcie_check (run.sh), which asserts the
# auto-picked NIC shares the GPU's PCIe switch.
if [ "$LOCAL_RANK" -ge "${#NODE_IB_DEVICES[@]}" ]; then
    echo "LOCAL_RANK=$LOCAL_RANK exceeds configured IB devices: $NODE_IB_DEVICES_STR" >&2
    exit 1
fi
IB_DEVICE=${NODE_IB_DEVICES[$LOCAL_RANK]}
export FI_VERBS_IFACE=$(resolve_iface_from_ibdev "$IB_DEVICE")

# DeepEP needs PYTHONPATH to find the in-tree deep_ep package
export PYTHONPATH=/root/jiafuzha/code-repo/zjf2012/DeepEP:${PYTHONPATH:-}

# DeepEP / iSHMEM needs MASTER_ADDR (rank-0 node hostname)
export MASTER_ADDR=${MASTER_ADDR:-deepep-ll-node0}

echo "[$(hostname) lr=$LOCAL_RANK gr=${PMI_RANK:-?}] ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK IB_DEVICE=$IB_DEVICE IFACE=$FI_VERBS_IFACE" >&2

ulimit -c unlimited
exec "$@"
