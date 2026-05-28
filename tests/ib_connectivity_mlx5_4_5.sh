#!/usr/bin/env bash
set -euo pipefail

DEV_A="mlx5_5"
DEV_B="mlx5_6"
PORT_A="${PORT_A:-1}"
PORT_B="${PORT_B:-1}"
ITERS="${ITERS:-1000}"
SIZE="${SIZE:-4096}"
TIMEOUT_SEC="${TIMEOUT_SEC:-20}"
PERFTEST_PORT="${PERFTEST_PORT:-18515}"

usage() {
    cat <<'EOF'
Test IB connectivity between mlx5_4 and mlx5_5.

Usage:
  tests/ib_connectivity_mlx5_4_5.sh check
      Show device mapping and link state for mlx5_4/mlx5_5.

  tests/ib_connectivity_mlx5_4_5.sh local [server_ip]
      Run ib_write_bw with server pinned to mlx5_4 and client pinned to mlx5_5.
      If server_ip is omitted, 127.0.0.1 is used.

  tests/ib_connectivity_mlx5_4_5.sh server
      Start server on mlx5_4 (for two-node test).

  tests/ib_connectivity_mlx5_4_5.sh client <server_ip>
      Start client on mlx5_5 to connect to remote server.

Environment overrides:
    PORT_A=1 PORT_B=1 ITERS=1000 SIZE=4096 TIMEOUT_SEC=20 PERFTEST_PORT=18515

Requirements:
  - perftest (ib_write_bw)
  - ibstat, ibdev2netdev
EOF
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: missing command '$cmd'" >&2
        exit 1
    fi
}

print_dev_info() {
    local dev="$1"
    local port="$2"

    echo "===== $dev port $port ====="
    ibstat "$dev" 2>/dev/null || {
        echo "ERROR: ibstat failed for $dev" >&2
        return 1
    }
    echo
}

check_link_active() {
    local dev="$1"
    local port="$2"
    local line

    line="$(ibstat "$dev" 2>/dev/null | awk -v p="$port" '
        $1=="Port" && $2==p":" {in_port=1; next}
        in_port && $1=="State:" {print $2; exit}
    ')"

    if [[ "$line" != "Active" ]]; then
        echo "ERROR: $dev port $port is not Active (current: ${line:-unknown})" >&2
        return 1
    fi
    echo "OK: $dev port $port state is Active"
}

do_check() {
    require_cmd ibstat
    require_cmd ibdev2netdev

    echo "=== ibdev2netdev mapping ==="
    ibdev2netdev
    echo

    print_dev_info "$DEV_A" "$PORT_A"
    print_dev_info "$DEV_B" "$PORT_B"

    check_link_active "$DEV_A" "$PORT_A"
    check_link_active "$DEV_B" "$PORT_B"
}

is_port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :$port )" 2>/dev/null | awk 'NR>1{found=1} END{exit(found?0:1)}'
        return $?
    fi
    return 1
}

check_perftest_port_available() {
    if is_port_in_use "$PERFTEST_PORT"; then
        echo "ERROR: TCP port $PERFTEST_PORT is already in use; set PERFTEST_PORT to a free value" >&2
        return 1
    fi
    return 0
}

run_server() {
    require_cmd ib_write_bw
    check_perftest_port_available
    echo "Starting server on $DEV_A port $PORT_A"
    ib_write_bw -d "$DEV_A" -i "$PORT_A" -x 3 -p "$PERFTEST_PORT" -s "$SIZE" -n "$ITERS"
}

run_client() {
    local server_ip="$1"
    require_cmd ib_write_bw
    echo "Starting client on $DEV_B port $PORT_B -> server $server_ip"
    ib_write_bw -d "$DEV_B" -i "$PORT_B" -x 3 -p "$PERFTEST_PORT" -s "$SIZE" -n "$ITERS" "$server_ip"
}

run_local() {
    local server_ip="${1:-127.0.0.1}"
    local server_pid=""
    local server_log="/tmp/ib_bw_server_${DEV_A}_p${PERFTEST_PORT}.log"
    local client_log="/tmp/ib_bw_client_${DEV_B}_p${PERFTEST_PORT}.log"
    local attempt
    local connected=0

    require_cmd ib_write_bw
    check_perftest_port_available

    echo "Launching local server on $DEV_A port $PORT_A"
    ib_write_bw -d "$DEV_A" -i "$PORT_A" -x 3 -p "$PERFTEST_PORT" -s "$SIZE" -n "$ITERS" >"$server_log" 2>&1 &
    server_pid=$!

    cleanup() {
        if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
            kill "$server_pid" >/dev/null 2>&1 || true
        fi
    }
    trap cleanup EXIT

    for attempt in $(seq 1 "$TIMEOUT_SEC"); do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            echo "ERROR: ib_write_bw server terminated before accepting client" >&2
            cat "$server_log" >&2 || true
            return 1
        fi

        if ib_write_bw -d "$DEV_B" -i "$PORT_B" -x 3 -p "$PERFTEST_PORT" -s "$SIZE" -n "$ITERS" "$server_ip" >"$client_log" 2>&1; then
            connected=1
            break
        fi

        sleep 1
    done

    if [[ "$connected" -ne 1 ]]; then
        echo "ERROR: client could not connect to server after ${TIMEOUT_SEC}s" >&2
        echo "---- client log ----" >&2
        cat "$client_log" >&2 || true
        echo "---- server log ----" >&2
        cat "$server_log" >&2 || true
        return 1
    fi

    wait "$server_pid" || true
    trap - EXIT

    echo
    echo "Server log: $server_log"
    echo "Client log: $client_log"
}

main() {
    if [[ "$#" -lt 1 ]]; then
        usage
        exit 1
    fi

    case "$1" in
    check)
        do_check
        ;;
    server)
        do_check
        run_server
        ;;
    client)
        if [[ "$#" -ne 2 ]]; then
            echo "ERROR: client mode requires <server_ip>" >&2
            usage
            exit 1
        fi
        do_check
        run_client "$2"
        ;;
    local)
        do_check
        if [[ "$#" -eq 2 ]]; then
            run_local "$2"
        else
            run_local
        fi
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "ERROR: unknown mode '$1'" >&2
        usage
        exit 1
        ;;
    esac
}

main "$@"