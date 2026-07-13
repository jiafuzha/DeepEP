#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
ENV_FILE="$REPO_DIR/.env.compose"
SKIP_BUILD=0
DISCOVER_ONLY=0

while (($#)); do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --discover-only)
      DISCOVER_ONLY=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--env-file PATH] [--skip-build] [--discover-only]" >&2
      exit 1
      ;;
  esac
done

python3 "$SCRIPT_DIR/pcie_topology.py" discover-env --output "$ENV_FILE"

if ((DISCOVER_ONLY)); then
  exit 0
fi

compose_cmd=(docker compose --env-file "$ENV_FILE" -f "$REPO_DIR/docker-compose.yml")

if ((SKIP_BUILD)); then
  "${compose_cmd[@]}" up -d
else
  "${compose_cmd[@]}" up -d --build
fi
