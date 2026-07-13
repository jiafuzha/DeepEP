#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

required_env=(
  SIM_NODE_NAME
  SIM_NODE_RANK
  XPU_RENDER_NODES
  RDMA_MLX5_HCAS
  MASTER_ADDR
  MASTER_PORT
)

for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
done

expected_render_count="${EXPECTED_RENDER_PER_NODE:-4}"
expected_mlx5_count="${EXPECTED_MLX5_PER_NODE:-4}"

validate_cmd=(
  python3
  "$SCRIPT_DIR/pcie_topology.py"
  validate-selection
  --render-nodes "$XPU_RENDER_NODES"
  --mlx5-hcas "$RDMA_MLX5_HCAS"
  --expected-render-count "$expected_render_count"
  --expected-mlx5-count "$expected_mlx5_count"
)

if [[ -n "${EXPECTED_PCIE_SWITCH:-}" ]]; then
  validate_cmd+=(--expected-switch "$EXPECTED_PCIE_SWITCH")
fi

"${validate_cmd[@]}"

if [[ -z "${ZE_AFFINITY_MASK:-}" ]]; then
  echo "ZE_AFFINITY_MASK is not set; this container will see every XPU exposed by /dev/dri." >&2
  exit 1
fi

if [[ -z "${UCX_NET_DEVICES:-}" ]]; then
  echo "UCX_NET_DEVICES is not set; RDMA traffic would not be pinned to the selected mlx5 HCAs." >&2
  exit 1
fi

echo "Preflight passed for ${SIM_NODE_NAME} (rank ${SIM_NODE_RANK})"
echo "  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
echo "  UCX_NET_DEVICES=${UCX_NET_DEVICES}"
echo "  MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
