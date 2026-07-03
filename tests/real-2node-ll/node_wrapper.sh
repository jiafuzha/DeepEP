#!/bin/bash
# Per-rank wrapper executed inside each MPI rank on real nodes b70-hq-1 / b70-hq-2.
#
# Sets up oneAPI env (including oneCCL/XCCL), ZE_AFFINITY_MASK and FI_VERBS_IFACE
# based on actual hostname (node identity) and MPI_LOCALRANKID (rank within node).
#
# Both nodes use the same hardware topology:
#   GPU 4 (0000:ac:00.0) <-> mlx5_4 (0000:b2:00.0)  [ens4013f0np0]
#   GPU 5 (0000:b0:00.0) <-> mlx5_5 (0000:b2:00.1)  [ens4013f1np1]
#
# Both nodes share the same path: /root/jiafuzha/code-repo/zjf2012/DeepEP

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
# deep_ep_cpp.so was linked without -lhwloc; preload it to satisfy the missing symbol.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhwloc.so${LD_PRELOAD:+:$LD_PRELOAD}

LOCAL_RANK=${MPI_LOCALRANKID:-0}
export LOCAL_RANK

# Detect node identity from hostname to set NODE_RANK.
HOSTNAME_VAL=$(hostname)
case "$HOSTNAME_VAL" in
    *hq-2*|*hq2*)
        NODE_RANK=1
        ;;
    *)
        NODE_RANK=0
        ;;
esac

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=4,5

# Resolve the verbs interface dynamically from the selected mlx5 device.
# local_rank 0 -> mlx5_4, local_rank 1 -> mlx5_5. The netdev name differs per
# node (e.g. ens911f0np0 on b70-hq-1, ens910f0np0 on b70-hq-2), so never
# hardcode it; read it from sysfs instead.
NIC_DEVS=(mlx5_4 mlx5_5)
SEL_NIC=${NIC_DEVS[$LOCAL_RANK]}
SEL_IFACE=$(ls /sys/class/infiniband/${SEL_NIC}/device/net/ 2>/dev/null | head -n1)
if [ -n "$SEL_IFACE" ]; then
    export FI_VERBS_IFACE=$SEL_IFACE
fi

# init_dist (tests/utils.py) reads RANK as the *node* rank (0..WORLD_SIZE-1).
export RANK=${RANK:-$NODE_RANK}

export PYTHONPATH=/root/jiafuzha/code-repo/zjf2012/DeepEP:${PYTHONPATH:-}

export MASTER_ADDR=${MASTER_ADDR:-b70-hq-1}

echo "[$(hostname) lr=$LOCAL_RANK gr=${PMI_RANK:-?}] ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK NIC=$SEL_NIC IFACE=$FI_VERBS_IFACE" >&2

ulimit -c unlimited
exec "$@"
