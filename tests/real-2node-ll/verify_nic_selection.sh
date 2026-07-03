#!/usr/bin/env bash
# verify_nic_selection.sh
#
# Verifies that iSHMEM's automatic GPU->NIC affinity selection binds each rank's
# GPU to a NIC under the SAME PCIe switch. Run BEFORE the LL test (see run.sh).
# No ISHMEM_IBGDA_NIC pinning is used — the goal is to validate the auto-selection.
#
# Run FROM b70-hq-1 (mpirun spawns ranks on b70-hq-2 via SSH).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_EP_DIR="${DEEP_EP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ISHMEM_DIR="${ISHMEM_DIR:-/root/jiafuzha/code-repo/zjf2012/ishmem_ibgda/build/_install}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MASTER_PORT="${NIC_CHECK_MASTER_PORT:-29540}"
ISHMEM_SYMMETRIC_SIZE="${ISHMEM_SYMMETRIC_SIZE:-268435456}"
TIMEOUT_SEC="${NIC_CHECK_TIMEOUT_SEC:-120}"

# Re-use the nic_pcie_check source from the docker-based sim directory.
DOCKER_SIM_DIR="$DEEP_EP_DIR/tests/docker-2node-ll"
CHECK_DIR="$DOCKER_SIM_DIR"
CHECK_BIN="$DOCKER_SIM_DIR/nic_pcie_check"
WRAPPER_PATH="$SCRIPT_DIR/node_wrapper.sh"
TOTAL_RANKS=$((NUM_PROCESSES * 2))

echo "===== Verifying iSHMEM auto GPU->NIC selection (same-PCIe-switch) ====="

# --- 1. Build nic_pcie_check locally on b70-hq-1 ---
(
    set +eu
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    export ISHMEM_DIR="$ISHMEM_DIR"
    export PKG_CONFIG_PATH="$ISHMEM_DIR/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    chmod +x "$CHECK_DIR/build_nic_pcie_check.sh" "$WRAPPER_PATH" 2>/dev/null
    set -e
    bash "$CHECK_DIR/build_nic_pcie_check.sh"
) || { echo "FAIL: nic_pcie_check build failed" >&2; exit 1; }

# --- 2. Run it under mpirun, capture the binding summary ---
RAW_LOG="$(mktemp)"
trap 'rm -f "$RAW_LOG"' EXIT

(
    set +eu
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    cd "$DEEP_EP_DIR"
    timeout "$TIMEOUT_SEC" mpirun \
        -n "$TOTAL_RANKS" -ppn "$NUM_PROCESSES" \
        -hosts b70-hq-1,b70-hq-2 \
        -genv ISHMEM_IB_ENABLE_IBGDA 1 \
        -genv ISHMEM_IBGDA_DIRECT_DOORBELL 1 \
        -genv ISHMEM_ENABLE_GPU_IPC 0 \
        -genv ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP 0 \
        -genv ISHMEM_SYMMETRIC_SIZE "$ISHMEM_SYMMETRIC_SIZE" \
        -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1 \
        -genv ISHMEM_IBGDA_QPS_PER_PE 1 \
        -genv ISHMEM_IBGDA_DB_BATCH_SIZE 0 \
        -genv ISHMEM_IBGDA_BAR_BACKEND igub \
        -genv I_MPI_FABRICS shm:ofi \
        -genv FI_PROVIDER tcp \
        -genv ISHMEM_DEBUG 1 \
        -genv MASTER_ADDR b70-hq-1 \
        -genv MASTER_PORT "$MASTER_PORT" \
        -genv WORLD_SIZE 2 \
        -launcher ssh \
        "$WRAPPER_PATH" \
        "$CHECK_BIN"
) >"$RAW_LOG" 2>&1
RC=$?

echo "--- iSHMEM binding summary (selected NIC per PE) ---"
grep -E "selected_nic=|NIC_PCIE_CHECK_INIT" "$RAW_LOG" || true

if ! grep -q "selected_nic=" "$RAW_LOG"; then
    echo "FAIL: no iSHMEM GPU-NIC binding summary found (init failed? rc=$RC)" >&2
    echo "----- captured output (tail) -----" >&2
    tail -30 "$RAW_LOG" >&2
    exit 1
fi

# --- 3+4. Parse summary + assert same PCIe switch via host sysfs ---
python3 - "$RAW_LOG" <<'PYEOF'
import os, re, sys

raw = open(sys.argv[1], errors="replace").read()
pat = re.compile(
    r"pe=(\d+).*?ze_mask=(\S+).*?expected_nic=(\S+).*?selected_nic=(\S+).*?"
    r"nic_bdf=(\S+).*?gpu_bdf=(\S+)")
rows = {}
for m in pat.finditer(raw):
    pe = int(m.group(1))
    rows[pe] = dict(pe=pe, ze=m.group(2), expected=m.group(3),
                    nic=m.group(4), nic_bdf=m.group(5).lower(), gpu_bdf=m.group(6).lower())

if not rows:
    print("FAIL: could not parse any GPU-NIC binding rows", file=sys.stderr)
    sys.exit(1)

def realpath_components(bdf):
    p = "/sys/bus/pci/devices/%s" % bdf
    try:
        rp = os.path.realpath(p)
    except OSError:
        return None
    if not os.path.exists(rp):
        return None
    return rp.split("/")

BRIDGE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$")

def same_switch(gpu_bdf, nic_bdf):
    a = realpath_components(gpu_bdf)
    b = realpath_components(nic_bdf)
    if a is None or b is None:
        return None, "unresolved-sysfs"
    common = []
    for x, y in zip(a, b):
        if x != y:
            break
        common.append(x)
    if not common:
        return False, "no-common-path"
    last = common[-1]
    if BRIDGE.match(last):
        return True, "switch=%s" % last
    return False, "diverge-at=%s" % last

print("%-4s %-8s %-10s %-16s %-16s %-7s %s" %
      ("pe", "ze_mask", "nic", "nic_bdf", "gpu_bdf", "result", "detail"))
all_ok = True
for pe in sorted(rows):
    r = rows[pe]
    if r["nic_bdf"] in ("unknown", "") or r["gpu_bdf"] in ("unknown", ""):
        ok, detail = False, "missing-bdf"
    else:
        ok, detail = same_switch(r["gpu_bdf"], r["nic_bdf"])
    verdict = "PASS" if ok else ("WARN" if ok is None else "FAIL")
    if ok is not True:
        all_ok = False
    print("%-4d %-8s %-10s %-16s %-16s %-7s %s" %
          (pe, r["ze"], r["nic"], r["nic_bdf"], r["gpu_bdf"], verdict, detail))

if all_ok:
    print("RESULT: PASS -- every rank's auto-selected NIC shares its GPU's PCIe switch")
    sys.exit(0)
else:
    print("RESULT: FAIL -- at least one rank's auto-selected NIC is NOT same-switch")
    sys.exit(1)
PYEOF
PYRC=$?

if [ $PYRC -ne 0 ]; then
    echo "===== NIC selection check FAILED =====" >&2
    exit 1
fi
echo "===== NIC selection check PASSED ====="
