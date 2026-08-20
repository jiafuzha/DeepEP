# DeepEP LL v2: Build & Smoke Test Guide

## Overview

This guide documents how to build iSHMEM/IBGDA and DeepEP XPU extension, then run a
low-latency (LL) internode smoke test using the `docker-2node-ll-v2` harness.

**Key insight**: iSHMEM and DeepEP **must** be built with the **same oneAPI compiler
version** used at runtime. The container image ships oneAPI 2025.3, so everything must
be built inside the container. Building on the host (which may have a different oneAPI
version, e.g., 2026.0) produces bitcode that causes `llvm-link: error: linked module
is broken!` at the SYCL device-link step.

---

## Prerequisites

- Docker with the `deepep_jiafuzha` image (self-contained: oneAPI 2025.3, RDMA/verbs,
  level-zero, system Python 3.12, PyTorch 2.14.0a0 XPU).
- `/root/jiafuzha` is shared between host and containers via bind-mount.
- `igub_vmem_drv.ko` loaded on the host (`/dev/igub_vmem` must exist).
- Clean IPC state: no stale PSM3/oneCCL named semaphores or shared memory segments.

---

## Step 1: Start the Containers

```bash
cd /root/jiafuzha/code-repo/zjf2012/DeepEP/tests/docker-2node-ll-v2

# Clean stale IPC state first (on host)
rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/*ishmem* \
      /tmp/deep_ep_xpu_ipc_*.sock

# Bring up both containers
DEEP_EP_FORCE_BUILD=0 bash run.sh --up
```

This starts two containers:
| Container | Hostname | SSH Port | GPUs | NICs |
|-----------|----------|----------|------|------|
| `deepep-ll-v2-node0` | `deepep-ll-v2-node0` | 127.0.0.1:2330 | 0,1 | mlx5_0,mlx5_1 |
| `deepep-ll-v2-node1` | `deepep-ll-v2-node1` | 127.0.0.1:2331 | 2,3 | mlx5_2,mlx5_3 |

Verify they are running:
```bash
docker ps --format '{{.Names}}' | grep deepep-ll-v2
```

---

## Step 2: Build iSHMEM/IBGDA (inside `node0` container)

The iSHMEM static archive (`libishmem.a`) is device-linked into the DeepEP `.so`, so
its bitcode **must match the container's oneAPI 2025.3 compiler** (`icx`/`icpx` 2025.3.3).

```bash
docker exec deepep-ll-v2-node0 bash -lc "
set -e
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
cd /root/jiafuzha/ishmem_ibgda
rm -rf build
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DENABLE_MPI=ON \
  -DENABLE_IBGDA=1 \
  -DENABLE_HWLOC=1 -DENABLE_AOT_COMPILATION=FALSE \
  -DENABLE_OPENSHMEM=OFF \
  -DISHMEM_DEFAULT_RUNTIME=MPI \
  -DCMAKE_INSTALL_PREFIX=/root/jiafuzha/ishmem_ibgda/build/_install
make -j\$(nproc)
make install
echo 'iSHMEM build: OK'
"
```

**Verify** the build produces the expected output:
```bash
ls -la /root/jiafuzha/ishmem_ibgda/build/_install/lib/libishmem.a
# Expected: ~29 MiB static archive
```

---

## Step 3: Build DeepEP XPU Extension (inside `node0` container)

```bash
docker exec deepep-ll-v2-node0 bash -lc "
set -e
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export PIP_BREAK_SYSTEM_PACKAGES=1
export DEEP_EP_TARGET=xpu
export ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install
unset TORCH_XPU_ARCH_LIST XPU_AOT_TARGETS
cd /root/jiafuzha/code-repo/zjf2012/DeepEP
rm -rf build/ishmem-sycl-dlink build
python3 setup.py build_ext --inplace
echo 'DeepEP build: OK'
"
```

### Post-build Verification

```bash
docker exec deepep-ll-v2-node0 bash -lc "
# 1. Barrier .cpp.o md5 (known-good: 1e8b7aec...; known-bad: a04bc5dd...)
md5sum /root/jiafuzha/code-repo/zjf2012/DeepEP/build/ishmem-sycl-dlink/barrier.cpp.o

# 2. Device-link target list (MUST be generic multi-device, NOT 'bmg' only)
strings /root/jiafuzha/code-repo/zjf2012/DeepEP/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so \
  | grep -m1 -- '-device '
# Expected: -device pvc,bmg,arl-h,mtl-h,lnl-m,ptl-h,ptl-u

# 3. Archive stamp (which libishmem.a was linked)
cat /root/jiafuzha/code-repo/zjf2012/DeepEP/build/ishmem-sycl-dlink/.archive-stamp

# 4. Import test (should succeed without libsycl errors)
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
python3 -c 'import deep_ep; print(\"deep_ep OK\")'
"
```

### Build `nic_pcie_check` (for NIC selection gating)

```bash
docker exec deepep-ll-v2-node0 bash -lc "
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install
export LD_LIBRARY_PATH=\$ISHMEM_DIR/lib:\$LD_LIBRARY_PATH
cd /root/jiafuzha/code-repo/zjf2012/DeepEP/tests/docker-2node-ll-v2
bash build_nic_pcie_check.sh
"
```

---

## Step 4: Run the LL Smoke Test

```bash
cd /root/jiafuzha/code-repo/zjf2012/DeepEP/tests/docker-2node-ll-v2

NUM_PROCESSES=2 \
NUM_TOKENS=32 \
HIDDEN=7168 \
NUM_TOPK=2 \
NUM_EXPERTS=8 \
ISHMEM_IBGDA_DB_BATCH_SIZE=8 \
SKIP_NIC_CHECK=1 \
bash run.sh
```

### Test Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_PROCESSES` | 2 | Local ranks per node (total ranks = 4) |
| `NUM_TOKENS` | 32 | Number of tokens |
| `HIDDEN` | 7168 | Hidden dimension |
| `NUM_TOPK` | 2 | Top-K experts per token |
| `NUM_EXPERTS` | 8 | Total experts |
| `ISHMEM_IBGDA_DB_BATCH_SIZE` | 8 | Doorbell batch size |
| `SKIP_NIC_CHECK` | 1 | Skip pre-flight NIC PCIe check (optional, for faster iteration) |

### Key Ranks Topology

```
Rank 0: node0, local_rank=0 → GPU 0 + NIC mlx5_0  (iface ens1006f0np0)
Rank 1: node0, local_rank=1 → GPU 1 + NIC mlx5_1  (iface ens1006f1np1)
Rank 2: node1, local_rank=0 → GPU 2 + NIC mlx5_2  (iface ens2005f0np0)
Rank 3: node1, local_rank=1 → GPU 3 + NIC mlx5_3  (iface ens2005f1np1)
```

### Expected Results (H7168, 32 tokens, 8 experts)

All 4 ranks should report **~685 µs** average dispatch+combine latency with
**1.7–1.8 GB/s** bandwidth, clean quiesce, and orderly exit:

```
[rank 0] Dispatch + combine bandwidth: 1.71 GB/s, avg_t=685.56 us
[rank 1] Dispatch + combine bandwidth: 1.71 GB/s, avg_t=685.63 us
[rank 2] Dispatch + combine bandwidth: 1.81 GB/s, avg_t=685.62 us
[rank 3] Dispatch + combine bandwidth: 1.74 GB/s, avg_t=685.72 us
===== PASS tests/test_low_latency.py =====
```

A result of **~32 ms/iter** (orders of magnitude slower) indicates the wrong iSHMEM
barrier archive was linked — rebuild iSHMEM inside the container.

---

## Step 5: Tear Down

```bash
cd /root/jiafuzha/code-repo/zjf2012/DeepEP/tests/docker-2node-ll-v2
bash run.sh --down

# Clean IPC state
rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/*ishmem* \
      /tmp/deep_ep_xpu_ipc_*.sock
```

---

## Environment Variables Reference

### Build-time
| Variable | Value | Purpose |
|----------|-------|---------|
| `DEEP_EP_TARGET` | `xpu` | Target XPU/SYCL backend |
| `ISHMEM_DIR` | `/root/jiafuzha/ishmem_ibgda/build/_install` | iSHMEM install path |
| `TORCH_XPU_ARCH_LIST` | **unset** | Force generic multi-device AOT |
| `XPU_AOT_TARGETS` | **unset** | Force generic multi-device AOT |

### Runtime (set via `mpirun -genv` or `docker exec -e`)
| Variable | Smoke Test Value | Purpose |
|----------|-----------------|---------|
| `ISHMEM_IB_ENABLE_IBGDA` | `1` | Enable GPU-initiated RDMA |
| `ISHMEM_IBGDA_DIRECT_DOORBELL` | `1` | Direct doorbell path |
| `ISHMEM_IBGDA_BAR_BACKEND` | `igub` | igub_vmem BAR bridge |
| `ISHMEM_IBGDA_QPS_PER_PE` | auto (1) | QPs per PE |
| `ISHMEM_IBGDA_DB_BATCH_SIZE` | `8` | Doorbell batch size |
| `ISHMEM_IBGDA_DB_MODE` | `0` | Doorbell mode |
| `ISHMEM_ENABLE_GPU_IPC` | `0` | Disable GPU IPC in 2-node sim |
| `ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP` | `0` | Host-heap accessibility |
| `ISHMEM_SYMMETRIC_SIZE` | `268435456` | 256 MiB symmetric heap |
| `DEEP_EP_LL_FLAG_SENDER_FENCE` | `1` | Sender-side fence |
| `DEEP_EP_LL_FLAG_RECV_ACQ` | `1` | Receiver acquire |
| `DEEP_EP_LL_FLAG_LSC` | `0` | LSC hint mode |
| `DEEP_EP_LL_POLL_CAP` | `50000000` | Poll iteration cap |
| `DEEP_EP_NVL_BYTES` | `134217728` | 128 MiB NVL buffer |
| `DEEP_EP_RDMA_BYTES` | `67108864` | 64 MiB RDMA buffer |

---

## Troubleshooting

### `llvm-link: error: linked module is broken!` / `Intrinsic has incorrect argument type!`
**Cause**: iSHMEM was built with a different oneAPI compiler version than the one
used to build DeepEP. The container has oneAPI 2025.3 — build iSHMEM **inside the
container**, not on the host.

### `ImportError: libsycl.so.9: cannot open shared object file`
**Cause**: PyTorch in `/root/jiafuzha/code-repo/pytorch` was built with a different
oneAPI version (e.g., 2026.0 which provides `libsycl.so.9`). The container has oneAPI
2025.3 which provides `libsycl.so.8`. Rebuild PyTorch with the container's oneAPI.

### ~32 ms/iter LL latency (instead of ~685 µs)
**Cause**: Wrong iSHMEM barrier archive was device-linked. Verify with `md5sum
build/ishmem-sycl-dlink/barrier.cpp.o` — known-bad is `a04bc5dd...`, known-good is
`1e8b7aec...` (or `21e03ac7...`). Rebuild iSHMEM inside the container.

### `DEVICE_LOST` / GPU wedge
**Cause**: Accumulated HW/NIC/QP state from a prior failed run. Bring containers down,
reload `igub_vmem_drv`, clean `/dev/shm`, and retry. For persistent issues, a full
host reboot may be needed.

### bmg-only AOT (`-device bmg`)
**Cause**: `TORCH_XPU_ARCH_LIST=bmg` was set. Always `unset TORCH_XPU_ARCH_LIST
XPU_AOT_TARGETS` before building for generic multi-device AOT stability.
Verify with: `strings deep_ep_cpp*.so | grep -m1 -- '-device '`