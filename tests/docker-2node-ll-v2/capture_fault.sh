#!/usr/bin/env bash
# Post-reboot: reproduce the LL fault on the DEBUG xe and capture the faulting VA.
# The DEBUG xe (CONFIG_DRM_XE_DEBUG_VM) prints "ASID" + "Faulted Address" + VM_BIND
# trace on the illegal access. We run with recoverable page faults OFF (normal), so
# the illegal access actually FAULTS (fault-mode would hide it).
set -uo pipefail
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SD"
IGUB=/root/jiafuzha/code-repo/intel_gpu_uar_bridge/driver/igub_vmem_drv.ko
KREL=6.19.14-061914-generic

echo "=== 1. verify DEBUG xe is live ==="
LIVE=$(md5sum /lib/modules/$KREL/kernel/drivers/gpu/drm/xe/xe.ko.zst 2>/dev/null | cut -d' ' -f1)
echo "boot xe.ko.zst md5=$LIVE (expect 01faaa0adda600c225cccf3a6d2527f7 = DEBUG)"
if [ "$LIVE" != "01faaa0adda600c225cccf3a6d2527f7" ]; then
  echo "!! DEBUG xe not staged as boot module -- run stage_xe_debug.sh activate + reboot"; exit 1
fi
# DEBUG build exposes GuC-log + pagefault debugfs and dynamic_debug pr_debug callsites.
XE_DDBG=$(grep -c "drivers/gpu/drm/xe" /sys/kernel/debug/dynamic_debug/control 2>/dev/null || echo 0)
echo "xe dynamic_debug callsites available: $XE_DDBG (DEBUG build should be >0)"

echo "=== 2. ensure igub loaded ==="
lsmod | grep -q igub_vmem || { insmod "$IGUB" && sleep 20; }
lsmod | grep igub_vmem || { echo "!! igub not loaded"; exit 1; }
echo "GPUs: $(sycl-ls 2>/dev/null | grep -c 'level_zero.*gpu')"

echo "=== 3. enable xe VM/pagefault pr_debug (DEBUG build has callsites) ==="
DDC=/sys/kernel/debug/dynamic_debug/control
for f in xe_vm xe_gt_pagefault xe_bo xe_guc_submit xe_exec_queue; do
  echo "file drivers/gpu/drm/xe/${f}.c +p" > "$DDC" 2>/dev/null && echo "  +p $f" || echo "  (skip $f)"
done
echo 3 > /sys/module/xe/parameters/guc_log_level 2>/dev/null || true

echo "=== 4. clean state + reproduce (recoverable faults OFF so it FAULTS) ==="
rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/*ishmem* /tmp/deep_ep_xpu_ipc_*.sock 2>/dev/null
touch .l0debug .no_recoverable   # NEO page-fault prints + recoverable OFF
rm -rf debuglogs_fault; mkdir -p debuglogs_fault
dmesg -C 2>/dev/null || true
# Loop a few times; stop at first FAIL and snapshot dmesg.
for i in 1 2 3 4 5; do
  rm -f /dev/shm/*ishmem* /tmp/deep_ep_xpu_ipc_*.sock 2>/dev/null
  DEEP_EP_LL_RESET_DRIVER=1 TIMEOUT_SEC=220 ./run.sh > "debuglogs_fault/run$i.log" 2>&1
  if grep -q "===== PASS" "debuglogs_fault/run$i.log"; then echo "run$i PASS"; else
    echo "run$i FAIL -> capturing dmesg"; break; fi
done
dmesg > debuglogs_fault/dmesg_full.txt 2>/dev/null

echo "=== 5. FAULTING VA + VM_BIND trace ==="
grep -inE "Faulted Address|ASID|VM worker|engine_class=bcs|pagefault|page fault|access type|xe_vm|xe_bo.*bind|VM_BIND|asid=|fault" \
    debuglogs_fault/dmesg_full.txt | tail -60
echo "=== full dmesg saved to debuglogs_fault/dmesg_full.txt ; per-rank NEO logs in debuglogs/ ==="
rm -f .l0debug .no_recoverable
