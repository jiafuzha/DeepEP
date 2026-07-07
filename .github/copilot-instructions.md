# DeepEP Copilot instructions

## Build, test, and lint commands

- Build the CUDA/PyTorch extension: `NVSHMEM_DIR=/path/to/nvshmem python setup.py build`. If `NVSHMEM_DIR` is omitted and the `nvidia.nvshmem` Python package is unavailable, setup defines `DISABLE_NVSHMEM` and internode/low-latency features are not built.
- Install locally: `NVSHMEM_DIR=/path/to/nvshmem python setup.py install`, or `bash install.sh` to build a wheel and install `dist/*.whl`.
- Useful build environment variables from `setup.py`: `TORCH_CUDA_ARCH_LIST` (defaults to `9.0`, or `8.0` with `DISABLE_SM90_FEATURES=1`), `DISABLE_SM90_FEATURES=1`, `DISABLE_AGGRESSIVE_PTX_INSTRS=1`, and `TOPK_IDX_BITS=32|64`.
- Format and lint changed files: `bash format.sh`. Format/lint the whole repo: `bash format.sh --all`. This runs YAPF and Ruff for Python and runs `clang-format` for C/CUDA files when available.
- Direct Python lint: `ruff check .`. Formatting uses YAPF settings in `pyproject.toml`; C/CUDA formatting uses `.clang-format`.
- Tests are standalone distributed CUDA scripts:
  - Intranode: `python tests/test_intranode.py --num-processes 8`
  - Internode: run on each node with `WORLD_SIZE=<nodes> RANK=<node_rank> MASTER_ADDR=<rank0-host> MASTER_PORT=8361 python tests/test_internode.py --num-processes 8`
  - Low latency: `python tests/test_low_latency.py --num-processes 8`
- To run one smaller test script while iterating, lower the script arguments, for example: `python tests/test_intranode.py --num-processes 2 --num-tokens 128 --hidden 2048 --num-topk 2 --num-experts 4`. Low-latency equivalent: `python tests/test_low_latency.py --num-processes 2 --num-tokens 16 --hidden 2048 --num-topk 2 --num-experts 4`.

## High-level architecture

- `deep_ep` is the Python facade. `deep_ep.Buffer` owns the public dispatch/combine APIs, default tuning configs, CUDA event overlap wrapper usage, and the rank-to-rank handle exchange before calling the C++ extension.
- `deep_ep_cpp` is a PyTorch CUDA extension built in `setup.py`. `csrc/deep_ep.cpp` provides the pybind layer and runtime buffer implementation; `csrc/deep_ep.hpp`, `csrc/config.hpp`, and `csrc/event.hpp` define the runtime, `Config`, and CUDA event wrappers exposed to Python.
- Kernel entry points are declared in `csrc/kernels/api.cuh`. Implementations are split by communication mode: `layout.cu` computes token routing metadata, `intranode.cu` handles NVLink/CUDA IPC paths, `internode.cu` handles RDMA plus NVLink forwarding, `internode_ll.cu` handles low-latency RDMA kernels, and `runtime.cu` wraps NVSHMEM initialization/barriers.
- Normal dispatch flow is: Python computes or receives top-k metadata, `Buffer.get_dispatch_layout()` builds per-rank/per-expert routing tensors, `Buffer.dispatch()` sends tokens and returns a handle tuple, and `Buffer.combine()` reuses that handle to reduce tokens back.
- Memory domains are explicit. Intranode communication uses CUDA IPC-accessible NVLink buffers. Internode and low-latency paths use NVSHMEM/IBGDA RDMA buffers. Low-latency mode has its own RDMA layout, optional receive hooks, and mask/shrink APIs.
- `EventOverlap` wraps C++ `EventHandle` so Python callers can pass `previous_event`, set `async_finish=True`, and decide whether allocations belong to the communication stream.

## Key conventions

- `NUM_MAX_NVL_PEERS` is 8. Global ranks are partitioned as `rdma_rank = rank / 8` and `nvl_rank = rank % 8`; internode tests assume 8 local ranks and more than one node.
- Preserve dispatch handle tuple layouts. `combine()`, cached dispatch, tests, and low-latency APIs unpack these tuples by position.
- Use `deep_ep.topk_idx_t` for top-k indices; it is selected by `TOPK_IDX_BITS` at build time. A top-k index of `-1` means no expert selection.
- Normal BF16 inputs are tensors shaped `[num_tokens, hidden]`. FP8 dispatch inputs are `(x_fp8, x_scales)` tuples, with per-128-channel scaling helpers in `tests/utils.py`.
- Hidden sizes are compile-time switch cases in `csrc/kernels/launch.cuh`; adding a model dimension usually requires adding a `SWITCH_HIDDEN` case and validating all affected kernels.
- High-throughput kernels use an even SM count (`Buffer.set_num_sms`) and `Config` chunk sizes. Default config maps live in `deep_ep/buffer.py`; tests also act as tuning scripts and print candidate configs.
- Low-latency buffers should be allocated with `low_latency_mode=True` and `num_qps_per_rank=num_experts // group.size()` for the common path. Call `clean_low_latency_buffer()` before low-latency kernels if the RDMA buffer may be dirty from other communication.
- CUDA kernel code relies on inline PTX memory operations and fences in `csrc/kernels/utils.cuh`; preserve acquire/release scopes and the `DISABLE_AGGRESSIVE_PTX_INSTRS` fallback when changing these paths.
- SM90-specific launch/TMA/FP8 behavior is guarded by `DISABLE_SM90_FEATURES`. Keep the fallback launch path in `csrc/kernels/launch.cuh` working for A100/CUDA 11 builds.
- `format.sh` temporarily rewrites `#pragma unroll` before running `clang-format`; do not replace this with a plain clang-format command if preserving existing formatting behavior matters.
- For CUDA-to-SYCL/XPU migration work, use the repository agent in `.github/agents/cuda-to-sycl-xpu.agent.md` and the migration skills under `.github/skills/` before manually converting PTX, NVSHMEM, or memory-ordering code.
- For CUDA IPC / symmetric-memory migration to Intel Level Zero IPC, use `.github/agents/cuda-ipc-symmetric-memory-level-zero.agent.md`.
- For iSHMEM/IBGDA XPU performance tuning and `DEVICE_LOST` debugging on Intel BMG (LL/internode perf gaps, iSHMEM static-archive parity, multi-device AOT stability, IPC read/write stability, env-var matrix, the 2-node docker harness), use `.github/agents/ishmem-ibgda-xpu-perf-debug.agent.md` and the "XPU/SYCL performance tuning & DEVICE_LOST debugging" section below.
- Two migration skills are available under `.github/skills/`: `memory-semantics-and-ptx-assembly-converter` (PTX fences, bar.sync, mbarrier, cp.async, atomics → SYCL) and `nvshmem-ibgda-to-ishmem` (NVSHMEM/IBGDA API → iSHMEM). Invoke these skills before making manual PTX or NVSHMEM edits.
- `nvshmem_ishmem_api_mapping.txt` at the repo root documents the full NVSHMEM/IBGDA → iSHMEM API symbol mapping for kernel migration; consult it when renaming NVSHMEM calls.
- In-progress XPU/SYCL migration lives in `csrc/xpu/`; files there are SYCL counterparts of the CUDA kernels in `csrc/kernels/` (e.g. `intranode.cpp`, `internode.cpp`, `layout.cpp`). `xpu_kernels.hpp` and `xpu_runtime.hpp` are the XPU-side equivalents of `api.cuh` and `runtime.cu`.

## XPU/SYCL performance tuning & DEVICE_LOST debugging (iSHMEM/IBGDA on Intel BMG)

These are hard-won lessons from tuning the XPU internode/low-latency (LL) paths.
Read them before any XPU perf work or when chasing `UR_RESULT_ERROR_DEVICE_LOST`.
For iSHMEM/IBGDA specifics, also use `.github/agents/ishmem-ibgda-xpu-perf-debug.agent.md`.

### Top pitfall: iSHMEM archive parity (root cause of BOTH the LL perf gap AND DEVICE_LOST)
- iSHMEM (`libishmem.a`) is **statically linked** into `deep_ep_cpp*.so` (merged into the
  SYCL device link). Only the **build-time** `ISHMEM_DIR` determines the barrier/RDMA
  implementation baked into the `.so`; the runtime `ISHMEM_DIR`/`LD_LIBRARY_PATH` does NOT
  change it (there is no iSHMEM `.so`).
- Building against the WRONG/older `libishmem.a` gives a slow/broken `ishmemx_barrier_all_work_group`
  that either times out (~32.768 ms/iter → ~29× LL slowdown, test still "passes") or crashes
  the device (`DEVICE_LOST`). Both symptoms have the SAME cause. Fixing the archive fixed LL
  from ~32751 µs back to ~1178 µs (matching the good baseline ~1138 µs @ H7168).
- **Always build against the SAME iSHMEM repo the proven-good target uses.** Do NOT use a
  stray/session-shim `libishmem.a`. Verify parity before trusting any perf/DEVICE_LOST result:
  - `cat build/ishmem-sycl-dlink/.archive-stamp`  → `<archive_path>:<mtime_ns>:<size>`
  - `md5sum build/ishmem-sycl-dlink/{barrier,ibgda,nbi,proxy,rma,memory_ordering}.cpp.o` and
    compare against a known-good build. Known-good barrier md5 `1e8b7aec…`; slow/broken `a04bc5dd…`.
  - To force re-extraction after changing archives: `rm -rf build/ishmem-sycl-dlink`.

### Force generic multi-device AOT (bmg-only AOT is unstable → DEVICE_LOST)
- **`unset TORCH_XPU_ARCH_LIST XPU_AOT_TARGETS` before building.** A bmg-ONLY AOT device link
  is unstable and frequently causes `DEVICE_LOST` at runtime. The generic multi-device AOT list
  is stable. Confirm the `.so` device link is generic, not bmg-only:
  `strings deep_ep_cpp*.so | grep -m1 -- '-device '` → expect
  `-device pvc,bmg,arl-h,mtl-h,lnl-m,ptl-h,ptl-u` (NOT `-device bmg`).
- `TORCH_XPU_ARCH_LIST=bmg` forces torch to inject `-Xs -device bmg` into the `deep_ep_cpp`
  SyclExtension device link → bmg-only miscompile. If a downstream repo has a separate plain
  Extension that legitimately needs `XPU_AOT_TARGETS=bmg` (e.g. DeepSymm's `deep_symm_cpp`),
  keep it ONLY for that extension and still drop `TORCH_XPU_ARCH_LIST` for `deep_ep_cpp`.

### IPC remote-access stability
- **Remote device READ via IPC is UNSTABLE; remote device WRITE via IPC is STABLE.** Prefer
  push/write-based communication patterns (producer writes into the remote peer's buffer) over
  pull/read-based ones (consumer reads from a remote peer's buffer) in intranode/IPC XPU paths.
  A remote-read hotspot is a prime suspect for intermittent `DEVICE_LOST`/corruption; convert it
  to a remote write plus a local read where possible.

### DEVICE_LOST is often accumulated HW/NIC/QP wedge, not a code bug
- On the BMG + mlx5 stack, a `DEVICE_LOST` or init-hang leaves NIC/QP + GPU page-table state
  wedged; it then cascades into the next runs (LL flag path is especially sensitive), often
  within ~2 runs. Before concluding a change caused `DEVICE_LOST`, **reset and reproduce on
  clean HW**, and run an A/B against a known-good build/repo on the same clean HW.
- Driver reset (helps, does not always fully clear): `rmmod igub_vmem_drv; sleep 4;
  insmod /root/jiafuzha/code-repo/intel_gpu_uar_bridge/driver/igub_vmem_drv.ko; sleep 20`;
  verify `sycl-ls | grep -c 'level_zero.*gpu'` equals the expected GPU count. Only a full reboot
  reliably clears a bad wedge; after reboot the `igub` driver must be loaded manually.
- Clean stale IPC/shm state before every run: `docker rm -f <containers>;
  rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/*ishmem* /tmp/deep_ep_xpu_ipc_*.sock`.
- `ishmem_finalize` is problematic and is intentionally NOT called by DeepEP; keep it skipped in
  any downstream (e.g. DeepSymm) too.

### Correct XPU build command (multi-device AOT, explicit iSHMEM archive)
```
source /opt/intel/oneapi/setvars.sh --force && conda activate <env>
export DEEP_EP_TARGET=xpu ISHMEM_DIR=/root/jiafuzha/code-repo/ishmem_ibgda/build/_install
unset TORCH_XPU_ARCH_LIST XPU_AOT_TARGETS
rm -rf build/ishmem-sycl-dlink            # force iSHMEM re-extraction when switching archives
python setup.py build_ext --inplace
```
Then verify BOTH: barrier `.cpp.o` md5 matches known-good, and `-device` list is multi-device.

### iSHMEM env vars (BMG + IBGDA/igub, validated LL/internode config)
Set as `mpirun -genv <VAR> <val>` (or `-e` for docker). Values below are the validated defaults.

| Variable | Value | Purpose |
| --- | --- | --- |
| `ISHMEM_IB_ENABLE_IBGDA` | `1` | Enable IBGDA (GPU-initiated RDMA). |
| `ISHMEM_IBGDA_DIRECT_DOORBELL` | `1` | Direct doorbell path (low latency). |
| `ISHMEM_IBGDA_BAR_BACKEND` | `igub` | BAR backend = intel_gpu_uar_bridge (`igub_vmem_drv.ko`). |
| `ISHMEM_IBGDA_QPS_PER_PE` | `1` | RC QPs per PE (LL common path). |
| `ISHMEM_IBGDA_DB_BATCH_SIZE` | `0` | Doorbell batch size. |
| `ISHMEM_ENABLE_GPU_IPC` | `0` | Disable iSHMEM GPU IPC on the 2-node sim. |
| `ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP` | `0` | Host-heap accessibility (LL default 0). |
| `ISHMEM_SYMMETRIC_SIZE` | `268435456` (256 MiB) | Symmetric heap size. |
| `ISHMEM_IBGDA_NIC` | per-rank `mlx5_N` | Pin each rank to the NIC under its GPU's PCIe switch. |
| `ISHMEM_DIR` | `.../ishmem_ibgda/build/_install` | Build-time archive selection (see parity pitfall). |
| `ISHMEM_HOST_ROOT` | derived from `ISHMEM_DIR` | Repo root the harness bind-mounts into containers. |
| `ISHMEM_DEBUG` | `0` (set `1` to debug) | iSHMEM verbose init/QP logging. |
| `ZE_ENABLE_PCI_ID_DEVICE_ORDER` | `1` | Stable PCI-ordered device enumeration. |
| `ZE_AFFINITY_MASK` | per-rank GPU id(s) | Pin each rank to its GPU (same PCIe switch as its NIC). |
| `I_MPI_FABRICS` | `shm:ofi` (or `shm`) | MPI fabric for the bootstrap. |
| `FI_PROVIDER` | as needed | OFI provider selection. |

DeepEP LL runtime tuning flags: `DEEP_EP_LL_FLAG_SENDER_FENCE=1`, `DEEP_EP_LL_FLAG_RECV_ACQ=1`,
`DEEP_EP_LL_FLAG_LSC=0`, `DEEP_EP_LL_FLAG_PROGRESS=0`, `DEEP_EP_LL_POLL_CAP=1000000` (a too-large
poll cap masks wedge/hang instead of failing fast). Buffer sizes: `DEEP_EP_NVL_BYTES`,
`DEEP_EP_RDMA_BYTES`. Driver reset opt-in: `DEEP_EP_LL_RESET_DRIVER=1`.

### docker 2-node harness: switching iSHMEM repos needs a bind-mount, not just ISHMEM_DIR
- `nic_pcie_check` is compiled INSIDE the container via `pkg-config ishmem`. If `ISHMEM_DIR`
  points to a repo the container does not bind-mount, the build fails with
  `Package ishmem was not found`. `run.sh` therefore derives `ISHMEM_HOST_ROOT` from `ISHMEM_DIR`
  (strips `/build/_install`), exports it, and `docker-compose.yml` bind-mounts
  `${ISHMEM_HOST_ROOT}` at the same host path in-container. Set `ISHMEM_DIR` and it just works.

### Validation baselines & configs
- LL (`tests/docker-2node-ll`): `NUM_PROCESSES=2 NUM_TOKENS=32 HIDDEN=7168 NUM_TOPK=2
  NUM_EXPERTS=8`. Good baseline ~1138 µs @ H7168 (~687 µs @ H2048). A ~32 ms/iter result means
  the wrong/slow iSHMEM barrier archive.
- Normal internode (`tests/docker-2node`): `NUM_PROCESSES=2 NUM_TOKENS=32 HIDDEN=1024 NUM_TOPK=2
  NUM_EXPERTS=8`, expect `===== PASS tests/test_internode.py =====`.
- Always establish an A/B against a known-good build/repo on the SAME clean HW before attributing
  a perf regression or `DEVICE_LOST` to a code change (containers share GPUs/NICs → cannot run two
  2-node sims at once).
