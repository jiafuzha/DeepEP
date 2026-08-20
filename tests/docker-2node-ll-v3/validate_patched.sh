#!/usr/bin/env bash
# ===========================================================================
# POST-REBOOT validation of design-(b) on the PATCHED true-UC stack.
# Run AFTER rebooting into the patched xe (xe.ko.zst.PATCHED-NEEDS_UC).
#
#   cd /root/jiafuzha/code-repo/zjf2012/DeepEP/tests/docker-2node-ll-v3
#   ./validate_patched.sh
#
# Steps: verify patched xe loaded -> load igub -> bring up containers (patched
# libze auto-mounted via docker-compose.override.yml) -> verify container libze
# is the patched build -> run design-(b) A/B (coop=1 vs coop=0) at 32 tokens
# with reset-per-run -> classify any failure (compute-wedge ccs vs doorbell).
# ===========================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"

KVER=6.19.14-061914-generic
XE_INST=/lib/modules/$KVER/kernel/drivers/gpu/drm/xe/xe.ko.zst
PATCHED_MD5=d43e9582dcf4f99084c71977ed69ad4f     # xe.ko.zst.PATCHED-NEEDS_UC
STOCK_MD5=5dd6675c9f80b5f25f7f97ca307edc09       # xe.ko.zst.stock-backup
PATCHED_LIBZE_MD5=659f4abe48c059953a48b81b76c99d6e
IGUB_KO=/root/jiafuzha/code-repo/intel_gpu_uar_bridge/driver/igub_vmem_drv.ko

echo "===== [1/5] verify booted kernel + patched xe module ====="
echo "uname -r: $(uname -r)"
cur=$(md5sum "$XE_INST" 2>/dev/null | awk '{print $1}')
echo "installed xe.ko.zst md5: $cur"
if [ "$cur" = "$PATCHED_MD5" ]; then echo "  -> PATCHED xe is installed (loaded at boot). OK"
elif [ "$cur" = "$STOCK_MD5" ]; then echo "  !! STOCK xe installed -- patched staging was reverted. ABORT"; exit 2
else echo "  !! UNKNOWN xe md5. ABORT"; exit 2; fi

echo "===== [2/5] load igub_vmem driver ====="
if ! lsmod | grep -q '^igub_vmem_drv'; then
  insmod "$IGUB_KO" && echo "  insmod ok" || { echo "  insmod FAILED"; exit 2; }
  sleep 20
else echo "  already loaded"; fi
gpus=$(sycl-ls 2>/dev/null | grep -c 'level_zero.*gpu' || true)
echo "  level_zero GPUs visible on host: $gpus"

echo "===== [3/5] bring up containers (patched libze auto-mounted) ====="
if [ ! -f docker-compose.override.yml ]; then echo "  !! override missing -- patched libze NOT mounted. ABORT"; exit 2; fi
DEEP_EP_LL_RESET_DRIVER=0 ./run.sh --up >/dev/null 2>&1 || true
sleep 3
cmd5=$(docker exec deepep-ll-v3-node0 bash -lc 'md5sum /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.36300' 2>/dev/null | awk '{print $1}')
echo "  container libze md5: $cmd5"
if [ "$cmd5" = "$PATCHED_LIBZE_MD5" ]; then echo "  -> PATCHED libze mounted in container. OK"
else echo "  !! container libze is NOT the patched build. ABORT"; exit 2; fi

echo "===== [4/5] design-(b) A/B on true-UC stack (32 tok, reset-per-run) ====="
run_one() {
  local coop=$1 tag=$2
  local t0=$(date +%s)
  out=$(DEEP_EP_LL_RESET_DRIVER=1 DEEP_EP_LL_FLAG_PROGRESS=1 DEEP_EP_LL_COOP_PUT=$coop \
        NUM_TOKENS=32 TIMEOUT_SEC=360 timeout 520 ./run.sh 2>&1)
  local rc=$? t1=$(date +%s)
  local res perf; if echo "$out" | grep -q "===== PASS"; then res=PASS; else res="FAIL(rc=$rc)"; fi
  perf=$(echo "$out" | grep -oE "avg_t=[0-9.]+ us" | head -1)
  echo "[$tag] coop=$coop -> $res $perf  (${t1}-${t0}=$((t1-t0))s)"
  # classify failure signature from kernel log
  if [ "$res" != "PASS" ]; then
    echo "  --- dmesg signature ---"
    journalctl -k --since "5 min ago" 2>/dev/null | grep -iE "engine_class=|VM worker error|pat_index 3|NEEDS_UC|Schedule disable|reset" | tail -8 | sed 's/^/    /'
  fi
}
for i in 1 2 3; do run_one 1 "patched-coop32-$i"; done
for i in 1 2 3; do run_one 0 "patched-base32-$i"; done

echo "===== [5/5] UC-binding evidence (patch printk) ====="
journalctl -k -b 2>/dev/null | grep -iE "NEEDS_UC|pat_index 3|force.*UC|peer.*MMIO" | tail -10 || echo "  (no NEEDS_UC printk found this boot)"
echo "===== DONE. If runs hang in local torch ops (ccs wedge) BEFORE dispatch -> true-UC compute-wedge dead-end reproduced."
echo "      To REVERT to stock: cp /root/jiafuzha/code-repo/xe.ko.zst.stock-backup $XE_INST && depmod -a $KVER &&"
echo "      rm docker-compose.override.yml && reboot."
