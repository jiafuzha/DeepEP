#!/usr/bin/env bash
# Run one LL point for a given token count + db_batch, print a parseable result line.
# Usage: dbbatch_sweep.sh <tokens> <db_batch>
set -uo pipefail
cd "$(dirname "$0")"

T="$1"; DB="$2"
case "$T" in
  32|64|128) SZ=536870912;  POLL=""; RWGS="" ;;
  512)       SZ=1073741824; POLL=""; RWGS="" ;;
  1024)      SZ=1717986918; POLL=""; RWGS="" ;;
  2048)      SZ=3221225472; POLL=500000000; RWGS=2048 ;;
  4096)      SZ=6442450944; POLL=500000000; RWGS=2048 ;;
  *) echo "bad tokens $T" >&2; exit 2 ;;
esac

LOG="/tmp/dbb_${T}_${DB}.log"
rm -f /dev/shm/*ishmem* /tmp/deep_ep_xpu_ipc_*.sock 2>/dev/null

env \
  ISHMEM_IBGDA_DB_BATCH_SIZE="$DB" \
  DEEP_EP_LL_RESET_DRIVER=1 \
  ${POLL:+DEEP_EP_LL_POLL_CAP=$POLL} \
  ${RWGS:+DEEP_EP_LL_REDUCE_WGS=$RWGS} \
  ISHMEM_SYMMETRIC_SIZE="$SZ" \
  NUM_PROCESSES=2 NUM_TOKENS="$T" HIDDEN=7168 NUM_TOPK=2 NUM_EXPERTS=8 \
  TIMEOUT_SEC=320 \
  ./run.sh > "$LOG" 2>&1
rc=$?

avg=$(grep -oE 'avg_t=[0-9.]+ us' "$LOG" | head -1 | grep -oE '[0-9.]+')
mn=$(grep -oE 'min_t=[0-9.]+ us'  "$LOG" | head -1 | grep -oE '[0-9.]+')
mx=$(grep -oE 'max_t=[0-9.]+ us'  "$LOG" | head -1 | grep -oE '[0-9.]+')
pass=$(grep -cE 'PASS tests/test_low_latency.py' "$LOG")
res="FAIL"; [ "$pass" -ge 1 ] && res="PASS"
echo "RESULT tokens=$T db_batch=$DB rc=$rc result=$res avg=${avg:-NA} min=${mn:-NA} max=${mx:-NA} log=$LOG"
