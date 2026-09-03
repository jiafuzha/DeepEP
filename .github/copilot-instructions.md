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
- For merging/porting DeepEP XPU/SYCL changes into the DeepSymm repo, use `.github/agents/deepep-to-deepsymm-merge.agent.md` (encodes the known merge deltas: iSHMEM archive parity, generic multi-device AOT, `nvl`→`pcie` rename, `ishmem_finalize` skip, `torch.where` OOM, `HIDDEN` default, LL `POLL_CAP`, and the `ISHMEM_DIR` bind-mount).
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

### LL DEVICE_LOST from lost IBGDA doorbells — UAR WC/PAT root cause (2026-07)
- The residual LL `DEVICE_LOST` (wedge at iter ~2-6, dmesg `xe_guc_exec_queue_lr_cleanup` +
  "Schedule disable failed to respond" + `VM worker error: -16`) is a **lost GPU→NIC doorbell**:
  the UAR MMIO store never egresses as a PCIe Memory-Write TLP, so the NIC CQ never advances and
  the GPU quiet/poll loop spins past the GuC hang-check watchdog → GT reset → `DEVICE_LOST`.
- **iSHMEM-side fix (done, in `src/ibgda_device_impl.h`):** (1) ring the UAR doorbell and write
  SND_DBR with an **uncached LSC store** (`__builtin_intel_sycl_ptr_annotation(...,
  "sycl-cache-write-hint", 0x7)` → `store.ugm.uc.uc`) via new helpers
  `ishmemi_ibgda_uc_store32/64`, and (2) use a **system-scope** fence
  (`atomic_fence(seq_cst, memory_scope::system)` for the doorbell; `release/system` for the
  WQE/SND_DBR publish) — the old "system scope → coherency-tracker DEVICE_LOST" comment is
  OBSOLETE (disproven by commit `45dd3b2c` UT `ext_flag_fence_sys`/`ext_flag_aref_sys`). This cut
  the failure rate from wedge-by-iter-2 to ~8/10, but does NOT reach 10/10 by itself.
- **Why iSHMEM alone is insufficient — the memory TYPE (PAT), not just cache hints:** cache-write
  hints bypass L1/L3, but if the imported UAR page's **PTE PAT is WB/WC** the store can still be
  write-combined at the memory-controller level. iSHMEM already imports the UAR with
  `ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED` (`src/ibgda.cpp` `map_uar_to_gpu_va_*`), but that flag is
  honored end-to-end ONLY with the two OS patches under
  `/root/jiafuzha/code-repo/intel_gpu_uar_bridge/patches/`:
  `0001-fix-drm-honor-uncacheable-flag-on-shared-dma-buf-imp.patch` (compute-runtime/NEO: select
  `CachePolicy::uncached` when `flags.uncacheable`) and
  `0001-drm-xe-force-XE_CACHE_NONE-on-peer-to-peer-MMIO-dma-.patch` (kernel xe: force UC PAT on
  P2P MMIO dma-buf imports). **Check they are applied before blaming iSHMEM code:** kernel
  `uname -r` must be the patched xe build (a stock `*-generic` mainline kernel has NO NEEDS_UC);
  `dpkg -l | grep libze-intel-gpu1` being a stock distro package (e.g. `25.48.36300.8-0`) means
  NEO patch #2 is NOT applied and `BIAS_UNCACHED` is silently dropped → UAR bound WB/WC → residual
  doorbell loss. Re-apply/rebuild the patched kernel + compute-runtime to restore UC and reach
  10/10 (the in-code "session 647f1a24" note documents this working state).
- **DEAD-END UPDATE (2026-07-08): with the patched xe (true-UC PAT[3]) LOADED, the LL hang is a GPU
  COMPUTE (ccs) wedge, NOT a lost doorbell — the doorbell path is never reached.** On freshly-reset
  igub HW, iSHMEM init + `Buffer::sync` complete, then the test hangs in the first *local* XPU
  tensor ops (`torch.ones/randn/topk`) before any dispatch; GPU counters show
  `stats_doorbell_writes=0 / post_put_nbi_calls=0`, WQ all-zero, CQ all `0xFF`, and dmesg shows
  `engine_class=ccs` reset + `VM worker error: -62`. The identical torch ops run fine STANDALONE
  (no ishmem init) on the same clean HW, so the UC P2P-MMIO import wedges compute once bound into the
  process VM. This is mode-independent (see env-selectable `ISHMEM_IBGDA_DB_MODE` 0..5 +
  `DEEP_EP_SYNC_DBG=1` tracing) and NOT fixable by iSHMEM doorbell tuning. Because the patched xe
  forces UC on ALL `sg_dma_is_bus_address` P2P imports regardless of userspace flags, WB/WC cannot be
  restored from userspace — reverting to the prior WB/WC 8/10 path needs a REBOOT to stock xe (revert
  kernel patch #1); that A/B is a user decision. Do NOT keep tuning the doorbell for the true-UC stack.

### Build/toolchain parity: build iSHMEM + DeepEP with the SAME oneAPI the runtime uses
- The DeepEP `.so` device-links the iSHMEM `.cpp.o` bitcode; a **compiler-version mismatch** yields
  `llvm-link: error: linked module is broken!`. If the host oneAPI was upgraded (e.g. to 2026.0
  where MKL ships `libmkl_intel_lp64.so.2` → `.so.3`, breaking torch import) but the docker sims
  run oneAPI 2025.3, **build iSHMEM AND DeepEP INSIDE the container** (which has 2025.3 + a torch
  with matching MKL): `docker exec deepep-ll-v2-node0 bash -lc '... _build_ishmem.sh ...; python3
  setup.py build_ext --inplace'`. `_build_ishmem.sh` must `source setvars.sh --force` (plain
  `source` returns 3 when oneAPI is already in-env, and `set -e` aborts the build silently).

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

### Internode-NORMAL (HT) silent hang = under-provisioned buffers (2026-09)
- `tests/docker-2node-v2` at `HIDDEN=7168` used to wedge in the first fused dispatch (rc=124, `ccs`
  reset, `xe_guc_exec_queue_lr_cleanup`) — **indistinguishable from a HW wedge**. Cause: harness
  buffer defaults (128 MiB NVL / 64 MiB RDMA / 256 MiB symmetric) only fit `HIDDEN=1024`.
  Defaults are now **512 MiB NVL / 512 MiB RDMA / 2 GiB symmetric**.
- UNDER-sizing hangs silently; OVER-sizing fails instantly with `ishmem_align failed for N bytes`.
  **Fast bisect: drop to `HIDDEN=1024` — if it passes, it is sizing, not code or HW.**
- The `buffer.py` HT QP clamp (`min(num_qps, 16)`, lockstep with the `[1,16]` kernel clamps) is
  correct hygiene but was NOT the hang cause (`QPS_PER_PE=32` runs fine).

### `timeout` orphan cascade — how to fabricate a fake hang
- `timeout N` kills only the OUTER `mpirun`; the in-container `mpiexec.hydra` + 4 ranks survive and
  hold GPU contexts/QPs/heap, so every later run hangs. **Gate before AND after every run:** zero
  live `test_internode.py`, no deepep containers, no `/dev/shm/*ishmem*`, no
  `/tmp/deep_ep_xpu_ipc_*.sock`. A non-zero post-count is a failed run to reap, not a datapoint.
  Reap by explicit PID only. Strictly one sim, one repo, at a time.

### GPU off the PCI bus: parent-bridge remove+rescan (no reboot)
- 3-of-4 GPUs + `Failed to resize BAR2 to 32768M (-EINVAL)` + `pci resource is not valid`:
  device-level remove/rescan and `resource2_resize` all fail. **Remove the GPU's parent PCIe bridge**
  (`echo 1 > /sys/bus/pci/devices/0000:1e:01.0/remove`) → `sleep 8` → `echo 1 > /sys/bus/pci/rescan`
  → `sleep 30`. BAR2 returns at 32 G and `xe` re-probes. Recurs; re-check before every campaign.

### Fused-kernel knobs C and G: REVERTED to default OFF — they hang at ≥1024 tokens
- `DEEP_EP_FUSED_DROP_FENCE` (C) + `DEEP_EP_FUSED_FLAG_NBI_AMO` (G) (`csrc/xpu/internode.cpp`).
  Measured **−7% isolated dispatch** as a pair and were briefly shipped ON — but that A/B ran
  **only at `num_tokens=32`**. A token sweep at E=384/topk=6/H=7168 then showed 32/64/128/256 PASS
  but **1024 HANGS**, while the same shape PASSES with both knobs `0`. The hang drives a GuC
  watchdog → `ccs` reset and twice knocked a GPU off the PCI bus. **Both are now default OFF; set
  either to `1` to opt in.**
- Mechanism: the flag AMOs are **flow control**, and the two sites are NOT equivalent. The tail AMO
  (`internode_dispatch_fused.inc:794`) rides `channel_id % qps_per_pe` — the same QP as the payload
  put, so it really is RC-ordered and C's premise holds. The **head-credit** AMO (`:1034`, combine
  `:1169`) rides `(channel_id + num_channels) % qps_per_pe` — a **different** QP with no fence of
  its own, and the blocking AMO's CQE poll was its ONLY completion guarantee. Below ~256 tokens the
  recv buffer never fills so credit is never awaited and the gap is invisible; at ≥1024 the buffer
  wraps, credit becomes load-bearing, and a posted-but-never-completed credit update leaves the
  sender spinning forever. Re-enabling needs a completion/`quiet` or a same-QP mapping for the
  head-credit path, **plus a full sweep to 4096**.
- **GOLDEN RULE:** a perf tuning validated at ONE shape is NOT validated. Anything touching flow
  control, credit, or completion must be **swept across the token range before shipping** — the
  failure mode is a hang that is indistinguishable from a hardware fault.
- The fences at `internode_notify_fused.inc:237` and `:286` are lone transport fences and must stay
  unconditional.
- Perf noise on this rig is **±10%** and bimodal — use interleaved paired A/B and prefer the
  isolated-dispatch number (<1% within-arm spread) over round-trip.

### Internode-normal token sweep baseline (knobs OFF, E=384, topk=6, H=7168, µs)
| tokens | 32 | 64 | 128 | 256 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dispatch(iso) | 961.7 | 1105.2 | 1312.2 | 1687.9 | 3355.0 | 5356.8 | 9786.3 |
| combine(iso) | 947.9 | 1006.4 | 1532.7 | 1772.1 | 3076.1 | 4464.0 | 6972.2 |
| round-trip | 1431.5 | 1716.5 | 2466.4 | 3039.1 | 5877.0 | 9294.1 | 16290.1 |

7/7 PASS. Latency-bound below ~1024 tokens (~1.4 ms floor), throughput-bound above. Buffers: 512M
NVL / 512M RDMA / 2G symmetric up to 1024; 1G/1G/3G for 2048–4096. A 4 GiB heap aborts in NEO
(`drm_neo.cpp:265`) — over-sizing is not free.

### Validation baselines & configs
- LL (`tests/docker-2node-ll`): `NUM_PROCESSES=2 NUM_TOKENS=32 HIDDEN=7168 NUM_TOPK=2
  NUM_EXPERTS=8`. Good baseline ~1138 µs @ H7168 (~687 µs @ H2048). A ~32 ms/iter result means
  the wrong/slow iSHMEM barrier archive.
- Normal internode (`tests/docker-2node`): `NUM_PROCESSES=2 NUM_TOKENS=32 HIDDEN=1024 NUM_TOPK=2
  NUM_EXPERTS=8`, expect `===== PASS tests/test_internode.py =====`.
- Always establish an A/B against a known-good build/repo on the SAME clean HW before attributing
  a perf regression or `DEVICE_LOST` to a code change (containers share GPUs/NICs → cannot run two
  2-node sims at once).
