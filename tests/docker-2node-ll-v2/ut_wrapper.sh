#!/bin/bash
# Per-rank wrapper for the ibgda_doorbell UT across the 2-container LL sim.
# Pins each MPI rank to exactly ONE GPU + its matching NIC (unlike the DeepEP
# node_wrapper which exposes both node GPUs and lets DeepEP pick by local rank).
#   node0 (SIM_NODE_RANK=0): lr0->GPU0/mlx5_0  lr1->GPU1/mlx5_1
#   node1 (SIM_NODE_RANK=1): lr0->GPU2/mlx5_2  lr1->GPU3/mlx5_3
set -uo pipefail

if [ -f /run/deepep_node_env ]; then set +u; source /run/deepep_node_env; set -u; fi
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
fi

export ISHMEM_DIR=${ISHMEM_DIR:-/root/jiafuzha/ishmem_ibgda/build/_install}
export LD_LIBRARY_PATH=${ISHMEM_DIR}/lib:${LD_LIBRARY_PATH:-}
if [ -f /usr/lib/x86_64-linux-gnu/libhwloc.so ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libhwloc.so${LD_PRELOAD:+ $LD_PRELOAD}"
fi

LR=${MPI_LOCALRANKID:-0}
SNR=${SIM_NODE_RANK:-0}
GPU=$(( SNR * 2 + LR ))
export ZE_AFFINITY_MASK=$GPU
export ISHMEM_IBGDA_NIC=mlx5_${GPU}
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

echo "[$(hostname) lr=$LR snr=$SNR] ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK NIC=$ISHMEM_IBGDA_NIC" >&2
ulimit -c unlimited
exec "$@"
