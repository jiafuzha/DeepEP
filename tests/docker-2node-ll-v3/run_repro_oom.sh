#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# run_repro_oom.sh — thin wrapper around the LL docker runner (run.sh) that
# launches the MINIMAL OOM reproducer tests/repro_ll_oom.py instead of the full
# test_low_latency.py, forwarding the isolation knobs.  Same 2-node sim, same
# per-rank GPU/NIC pinning, same iSHMEM/IBGDA env matrix.
#
# Usage:
#   ./run_repro_oom.sh                                  # 1 run, REPRO_MODE=full
#   LOOPS=10 REPRO_MODE=full ./run_repro_oom.sh         # 10 warm back-to-back
#   REPRO_MODE=alloc_churn_only ITERS=400 ./run_repro_oom.sh
#   REPRO_MODE=dispatch_only ./run_repro_oom.sh
#   REPRO_MODE=no_ishmem ./run_repro_oom.sh
#
# Knobs (env, forwarded to the ranks by run.sh):
#   REPRO_MODE  full | dispatch_only | alloc_churn_only | no_ishmem   (default full)
#   ITERS       tight-loop iterations                                 (default 200)
#   HIDDEN NUM_TOKENS NUM_EXPERTS NUM_TOPK   (passed as CLI to the script)
#   DUAL_STREAM FP8_ALTERNATE STOP_ON_OOM HEARTBEAT
#   LOOPS       number of warm back-to-back process launches          (default 1)
#   ISHMEM_IBGDA_STATS_DIR  (set to capture pre-doorbell IBGDA stats at failure)
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TEST_SCRIPT="tests/repro_ll_oom.py"
export REPRO_MODE="${REPRO_MODE:-full}"
export ITERS="${ITERS:-200}"
export DUAL_STREAM="${DUAL_STREAM:-1}"
export FP8_ALTERNATE="${FP8_ALTERNATE:-1}"
export STOP_ON_OOM="${STOP_ON_OOM:-1}"
export HEARTBEAT="${HEARTBEAT:-20}"
export CHURN_MB="${CHURN_MB:-0}"
export CHURN_VARY="${CHURN_VARY:-1}"
export EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-0}"
export GUARD_ALLOC="${GUARD_ALLOC:-0}"

# LL problem size (small tokens, real hidden by default)
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export NUM_TOKENS="${NUM_TOKENS:-32}"
export HIDDEN="${HIDDEN:-7168}"
export NUM_TOPK="${NUM_TOPK:-2}"
export NUM_EXPERTS="${NUM_EXPERTS:-8}"

# default: capture IBGDA stats so failures show pre-doorbell timing
export ISHMEM_IBGDA_STATS_DIR="${ISHMEM_IBGDA_STATS_DIR:-$SELF_DIR/_repro_stats}"
mkdir -p "$ISHMEM_IBGDA_STATS_DIR" 2>/dev/null || true

LOOPS="${LOOPS:-1}"
LOGDIR="${LOGDIR:-/tmp/repro_oom}"
mkdir -p "$LOGDIR"
pass=0; oom=0; other=0
for l in $(seq 1 "$LOOPS"); do
    echo "########## REPRO LOOP $l/$LOOPS  MODE=$REPRO_MODE ITERS=$ITERS HIDDEN=$HIDDEN ##########"
    log="$LOGDIR/${REPRO_MODE}_run${l}.log"
    ( cd "$SELF_DIR" && ./run.sh ) > "$log" 2>&1
    if   grep -q 'OOM(39) at iter=' "$log"; then oom=$((oom+1));  verdict="OOM(39)"
    elif grep -q 'DEVICE ERR at iter=\|REPRODUCED (exit=2' "$log"; then other=$((other+1)); verdict="OTHER-ERR"
    elif grep -q 'did not reproduce' "$log"; then pass=$((pass+1)); verdict="no-repro"
    else other=$((other+1)); verdict="UNKNOWN(see $log)"; fi
    # surface the key lines
    grep -E 'RESULT:|OOM\(39\)|DEVICE ERR|failing tensor|XPU VRAM|IBGDA stats|bcs|reset|VM worker|iter=[0-9]+ OK' "$log" | tail -8
    echo "---- LOOP $l verdict=$verdict ----"
    sleep 2
done
echo ""
echo "===== REPRO SUMMARY ($LOOPS runs, MODE=$REPRO_MODE): pass/no-repro=$pass  OOM(39)=$oom  other=$other ====="
[ $oom -gt 0 ] && exit 39 || exit 0
