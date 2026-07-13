#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
ENV_FILE="${1:-$REPO_DIR/.env.compose}"
COMPOSE_FILE="$REPO_DIR/docker-compose.yml"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  echo "Run $SCRIPT_DIR/up_sim_nodes.sh --discover-only first." >&2
  exit 1
fi

compose_cmd=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")

running_services=$("${compose_cmd[@]}" ps --services --status running)
for service in node0 node1; do
  if ! grep -qx "$service" <<<"$running_services"; then
    echo "Service $service is not running. Start the topology with $SCRIPT_DIR/up_sim_nodes.sh first." >&2
    exit 1
  fi
done

torchrun_cmd='
set -euo pipefail
cd /root/jiafuzha
if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is not installed in the container. Build/install the PyTorch XPU tree first." >&2
  exit 1
fi
exec torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${SIM_NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${DIST_SCRIPT}"
'

node1_log=$(mktemp)
cleanup() {
  rm -f "$node1_log"
}
trap cleanup EXIT

"${compose_cmd[@]}" exec -T node1 bash -lc "$torchrun_cmd" >"$node1_log" 2>&1 &
node1_pid=$!

set +e
"${compose_cmd[@]}" exec -T node0 bash -lc "$torchrun_cmd"
node0_status=$?
wait "$node1_pid"
node1_status=$?
set -e

if ((node1_status != 0)); then
  cat "$node1_log" >&2
fi

if ((node0_status != 0 || node1_status != 0)); then
  exit 1
fi

cat "$node1_log"
