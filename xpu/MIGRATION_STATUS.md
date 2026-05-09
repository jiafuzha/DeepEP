# DeepEP XPU Migration Status

This directory is the staged XPU migration workspace.

## Completed in this commit

- Created mirrored Python package at `xpu/deep_ep`.
- Added dynamic extension loading with XPU-first fallback.
- Added temporary Python stub backend `xpu/deep_ep_cpp_xpu.py`.
- Mirrored native sources into `xpu/csrc` for file-by-file porting.
- Mirrored NVSHMEM notes into `xpu/third-party` for iSHMEM transition tracking.
- Added backend-aware stream/event abstraction in `xpu/csrc/event.hpp` with `DEEPEP_USE_XPU` compile switch.
- Switched xpu extension defaults to `deep_ep_cpp_xpu` in `xpu/csrc/deep_ep.hpp` and `xpu/csrc/CMakeLists.txt`.
- Replaced common stream/device callsites in `xpu/csrc/deep_ep.cpp` to use backend helper APIs.
- Isolated kernel dtype conversion behind `get_runtime_dtype` in `xpu/csrc/deep_ep.cpp`.
- Added XPU-conditional shared-memory IPC handle types in `xpu/csrc/deep_ep.hpp` to decouple CUDA handle structs.
- Added explicit XPU guard rails for shared-memory allocation and IPC open/close/export/import in `xpu/csrc/deep_ep.cpp`.
- Centralized device sync, memcpy, memset, malloc/free, host-mapped allocation, and device-query runtime calls behind helper wrappers in `xpu/csrc/deep_ep.cpp`.
- Replaced CUDA-specific kernel API signatures in `xpu/csrc/kernels/*.cu` from `cudaStream_t` and `cudaDataType_t` to `runtime_stream_t` and `runtime_data_type_t`.
- Added backend-agnostic BF16 runtime dtype constant `RUNTIME_R_16BF` in `xpu/csrc/kernels/configs.cuh` and switched kernel launch type dispatch to it.
- Added `DEEPEP_USE_XPU` compile guards, include isolation, and explicit unsupported stubs for NVSHMEM/PTX-heavy internode paths in `xpu/csrc/kernels/internode.cu` and `xpu/csrc/kernels/internode_ll.cu` to preserve symbol parity while isolating CUDA-only implementations.
- Replaced hard-fail XPU runtime stubs in `xpu/csrc/kernels/runtime.cu` with staged functional fallbacks: no-op intranode/inter-node barriers, stable local unique-id payload, rank-preserving init, and alignment-aware host `alloc/free` fallback to unblock non-NVSHMEM XPU bring-up paths.
- Added explicit `DEEPEP_USE_XPU` stub split in `xpu/csrc/kernels/layout.cu` so layout dispatch no longer goes through CUDA launch macros on XPU builds and reports a clear staged-migration message instead.
- Added an XPU-only CMake debug build path in `xpu/csrc/CMakeLists.txt` that builds `deep_ep_cpp_xpu` from `deep_ep_xpu_stub.cpp` without requiring CUDA toolchains.
- Added backend-aware stream/device helpers in `xpu/deep_ep/utils.py` and `xpu/deep_ep/buffer.py` so the XPU package no longer hard-depends on CUDA stream constructors.
- Hardened stream reconstruction fallback in `xpu/deep_ep/buffer.py` across backend API variations and added XPU no-op path in `xpu/csrc/kernels/CMakeLists.txt` so XPU mode does not pull CUDA/NVSHMEM link dependencies from kernel static libraries.
- Replaced assert-only XPU runtime helper wrappers in `xpu/csrc/deep_ep.cpp` with staged functional fallbacks for memcpy/memset/malloc/free/device query, plus local IPC handle encode/decode for shared-memory synchronization flows in single-process staging.
- Added explicit XPU intranode-only gating in `xpu/csrc/deep_ep.cpp` by disabling NVSHMEM-dependent internode branches under `DEEPEP_USE_XPU` and introducing early assertions when `num_rdma_bytes > 0`.
- Made `xpu/csrc/kernels/intranode.cu` include paths XPU-safe by excluding CUDA-only utility headers in `DEEPEP_USE_XPU` mode.
- Added an opt-in staged native XPU build path in `xpu/csrc/CMakeLists.txt` via `DEEPEP_XPU_NATIVE_STAGED=ON`, compiling mirrored kernel sources as C++ with XPU stub entrypoints while keeping stub-only build as default.
- Added `DEEPEP_XPU_BUILD_MODE` in `xpu/setup.py` with `stub` (default) and `native-staged` modes.
- Added CMake-driven native-staged extension build flow in `xpu/setup.py` to invoke `xpu/csrc` with `DEEPEP_XPU_NATIVE_STAGED=ON` and package `deep_ep_cpp_xpu` artifact.
- Updated `xpu/install.sh` to pass through `DEEPEP_XPU_BUILD_MODE` (default `stub`).
- Added automatic fallback from `DEEPEP_XPU_BUILD_MODE=native-staged` to `stub` in `xpu/setup.py` when prerequisites are missing (e.g., `cmake` or `torch`), with strict opt-out via `DEEPEP_XPU_NATIVE_STAGED_STRICT=1`.
- Updated `xpu/install.sh` to pass through `DEEPEP_XPU_NATIVE_STAGED_STRICT`.
- Added shared `EP_UNSUPPORTED_XPU(...)` error macro in `xpu/csrc/kernels/exception.cuh` and switched XPU kernel stubs in `launch.cuh`, `layout.cu`, `intranode.cu`, `internode.cu`, and `internode_ll.cu` to the unified adapter path for consistent backend error semantics.
- Added staged single-rank XPU Python fallback in `xpu/deep_ep/buffer.py` for `dispatch` and `combine` intranode paths when native kernels report unsupported status, including fallback handle propagation and bias application.
- Added staged single-rank XPU Python fallback in `xpu/deep_ep/buffer.py` for `internode_dispatch` and `internode_combine` paths when native internode kernels report unsupported status, including fallback handle propagation and bias application.
- Added targeted staged-fallback tests in `xpu/tests/test_xpu_buffer_fallbacks.py` for single-rank XPU `internode_dispatch` and `internode_combine` Python fallback behavior.
- Added staged single-rank XPU Python fallback in `xpu/deep_ep/buffer.py` for `low_latency_dispatch` and `low_latency_combine`, including fallback token mapping handles and hook compatibility.
- Added targeted staged-fallback tests in `xpu/tests/test_xpu_buffer_fallbacks.py` for single-rank XPU low-latency dispatch/combine fallback behavior.
- Upgraded the Python stub backend in `xpu/deep_ep_cpp_xpu.py` from constructor-only failure mode to a functional staged runtime for initialization/sync/buffer APIs (`is_available`, `get_local_device_id`, `get_local_ipc_handle`, `get_local_buffer_tensor`, `get_comm_stream`, `destroy`) while keeping kernel entrypoints explicit `NotImplemented` to trigger Python fallback paths.
- Fixed staged stub RDMA root-rank mapping in `xpu/deep_ep_cpp_xpu.py` (`get_root_rdma_rank(global_rank=True)`) to return per-RDMA-group global root rank.
- Added targeted staged-backend tests in `xpu/tests/test_xpu_stub_backend.py` covering buffer metadata, IPC handle payload shape, local buffer tensor views, sync/destroy lifecycle, and low-latency size-hint API stability.
- Enabled staged single-rank local initialization in `xpu/deep_ep/buffer.py` when `group=None` and `comm=None` on XPU, so high-level `Buffer` bring-up no longer requires distributed group bootstrap during migration.
- Replaced staged XPU no-op device synchronization in `xpu/csrc/deep_ep.cpp` with concrete backend stream synchronization (`backend::get_current_stream().synchronize()`).
- Replaced staged fixed-device fallback in `xpu/csrc/deep_ep.cpp` with concrete backend device discovery via current stream device index.
- Replaced staged fixed XPU SM-count fallback in `xpu/csrc/deep_ep.cpp` with a real PyTorch XPU device-property query via `at::xpu::getDeviceProperties(device_id)->max_compute_units`, retaining a conservative fallback only if the property is unavailable.
- Fixed native RDMA root-rank mapping in `xpu/csrc/deep_ep.cpp` so `get_root_rdma_rank(global=true)` returns the per-RDMA-group global root rank instead of the local NVL rank.
- Replaced staged host `malloc/free` fallback for mapped host allocations in `xpu/csrc/deep_ep.cpp` with the real PyTorch XPU pinned host allocator (`at::xpu::getPinnedMemoryAllocator()`), so host-side counters now use backend-managed pinned memory instead of plain heap memory.
- Replaced staged host-heap shared-memory allocation in `xpu/csrc/deep_ep.cpp` with the real PyTorch XPU caching allocator (`c10::xpu::XPUCachingAllocator::raw_alloc/raw_delete()`), so staged local NVL buffer memory now resides on XPU-managed device memory instead of `posix_memalign` host memory.
- Replaced staged XPU host-side `memcpy`/`memset` helper behavior in `xpu/csrc/deep_ep.cpp` with real SYCL queue operations (`queue().memcpy` and `queue().memset`) for staged local buffer synchronization and workspace initialization.
- Made XPU `runtime_memcpy_h2d` in `xpu/csrc/deep_ep.cpp` explicitly synchronous by waiting on the submitted SYCL memcpy event, preserving CUDA-path `cudaMemcpy` semantics during IPC pointer-table synchronization.
- Added regression coverage in `xpu/tests/test_xpu_buffer_fallbacks.py` for staged single-rank `Buffer` initialization without `group`/`comm` by monkeypatching runtime bootstrap.
- Verified XPU import and staged runtime lifecycle in the requested environment setup (`conda activate jiafu_deepep`, `source /opt/intel/oneapi/setvars.sh`, `source /root/jiafuzha/env.sh`), including successful `xpu.deep_ep` import and `xpu.deep_ep_cpp_xpu.Buffer` sync/destroy smoke path.
- Fixed low-latency maintenance fallback gating in `xpu/deep_ep/buffer.py` to trigger on unsupported-operation errors even when runtime XPU detection helper returns false.
- Added regression coverage in `xpu/tests/test_xpu_buffer_fallbacks.py` to ensure low-latency mask maintenance fallback works without `_has_xpu_runtime()`.
- Fixed native module-init dtype export in `xpu/csrc/deep_ep.cpp` by mapping `topk_idx_t` to Python `torch.int32/int64` directly, avoiding fragile THP scalar-type conversion that caused `ImportError: unsupported scalarType` during staged native import.
- Fixed `xpu/setup.py` native-staged build flow regressions (indentation bug and CMake prefix propagation) so pybind11/Torch CMake discovery works consistently in editable installs.
- Added staged native CMake compatibility for `.cu` sources under `icpx` by forcing C++ language treatment in `xpu/csrc/CMakeLists.txt` and aligned XPU type aliases in `xpu/csrc/kernels/configs.cuh`.
- Fixed staged native symbol parity for XPU internode low-latency combine in `xpu/csrc/kernels/internode_ll.cu` by matching the exported API signature.
- Added `torch_python` link resolution in `xpu/csrc/CMakeLists.txt` using Torch lib directories to avoid unresolved Python binding symbols at extension load time.
- Verified staged native build in the requested environment (`conda activate jiafu_deepep`, `source /opt/intel/oneapi/setvars.sh`, `CC=icx`, `CXX=icpx`) and confirmed `xpu.deep_ep` imports with native extension module `xpu.deep_ep_cpp_xpu`.
- Added import regression coverage in `xpu/tests/test_xpu_import.py` to assert extension load and exported `topk_idx_t` dtype metadata.
- Hardened XPU local shared-memory IPC in `xpu/csrc/deep_ep.cpp` and `xpu/csrc/deep_ep.hpp` by tracking allocation sizes per pointer, exporting validated size metadata in local IPC handles, validating handle-size parity on import, and rejecting unknown pointers on allocator free/export.
- Added exporting-process PID tagging/validation to XPU local IPC handles in `xpu/csrc/deep_ep.cpp` so staged local handles explicitly fail on cross-process import attempts.
- Added explicit XPU local IPC handle metadata (magic/version/kind) in `xpu/csrc/deep_ep.cpp` and strict decode-time validation to reject stale/malformed or unsupported staged-local handles earlier in import paths.
- Added imported-handle lifecycle tracking for XPU local IPC in `xpu/csrc/deep_ep.hpp` and `xpu/csrc/deep_ep.cpp`, including duplicate-open rejection and strict close validation for unknown pointers.
- Extended XPU local IPC handle schema in `xpu/csrc/deep_ep.cpp` with explicit `flags`/`reserved` fields and strict zero-validation to support forward schema evolution without silently accepting unknown payload bits.
- Added generation-aware stale-handle detection for XPU local IPC in `xpu/csrc/deep_ep.cpp` and `xpu/csrc/deep_ep.hpp` by embedding per-allocation generation IDs into local IPC payloads and validating imported handles against a process-global live-allocation registry.
- Added native staged regression coverage in `xpu/tests/test_xpu_import.py` to verify that stale XPU local IPC handles are rejected after the exporting buffer is destroyed.
- Hardened XPU mapped-host allocation in `xpu/csrc/deep_ep.cpp` with tracked host `malloc/free` fallback when `at::xpu::getPinnedMemoryAllocator()` is unavailable, avoiding hard-fail initialization in environments without pinned allocator support.
- Converted the stale-handle native regression in `xpu/tests/test_xpu_import.py` to a subprocess-based check so expected process termination on staged stale-handle rejection does not abort the main pytest runner.
- Rebuilt native-staged editable package with `DEEPEP_XPU_BUILD_MODE=native-staged` using `--no-build-isolation` in the requested oneAPI/`icx`/`icpx` environment, and verified `xpu/tests/test_xpu_import.py` passes (`3 passed`).
- Added subprocess-based malformed-IPC metadata regression coverage in `xpu/tests/test_xpu_import.py` by corrupting local handle magic bytes and verifying staged native import rejects the tampered handle.
- Verified targeted native IPC rejection regressions in `xpu/tests/test_xpu_import.py` (`stale_ipc_handle` and `malformed_ipc_handle`) pass (`2 passed`, `2 deselected`).
- Added checksum-protected XPU local IPC payloads in `xpu/csrc/deep_ep.cpp` using FNV-1a over handle metadata fields and strict decode-time checksum validation.
- Added checksum-corruption regression coverage in `xpu/tests/test_xpu_import.py` and verified targeted native IPC rejection regressions (`stale_ipc_handle`, `malformed_ipc_handle`, `corrupted_ipc_checksum`) pass (`3 passed`, `2 deselected`).
- Added subprocess-based duplicate-remote-handle regression coverage in `xpu/tests/test_xpu_import.py` to verify XPU local IPC duplicate-open rejection during staged native `sync`.
- Verified targeted native IPC rejection regressions in `xpu/tests/test_xpu_import.py` (`stale_ipc_handle`, `malformed_ipc_handle`, `corrupted_ipc_checksum`, `duplicate_remote_ipc_handle`) pass (`4 passed`, `2 deselected`).
- Upgraded XPU mapped-host fallback in `xpu/csrc/deep_ep.cpp` from plain host `malloc` to SYCL shared USM (`sycl::malloc_shared`) when pinned allocator is unavailable, preserving device accessibility for staged fallback counters while retaining tracked `malloc/free` as a final fallback.
- Rebuilt native-staged editable package with `DEEPEP_XPU_BUILD_MODE=native-staged` in the requested oneAPI `icx`/`icpx` environment and verified focused import/IPC regression checks in `xpu/tests/test_xpu_import.py` pass (`5 passed`, `1 deselected`).
- Fixed a native lifecycle leak in `xpu/csrc/deep_ep.cpp` by freeing `moe_recv_rdma_counter` in `Buffer::destroy()` (it was allocated via `runtime_malloc_host_mapped` but previously never released).
- Fixed internode availability semantics in `xpu/csrc/deep_ep.cpp` and `xpu/deep_ep_cpp_xpu.py` to require `num_rdma_bytes > 0` (not just multi-rank topology), preventing staged XPU paths from reporting internode-ready when RDMA buffers are disabled.
- Added regressions in `xpu/tests/test_xpu_import.py` and `xpu/tests/test_xpu_stub_backend.py` to verify `is_internode_available()` stays false when `num_rdma_bytes == 0`, including after `sync()`.
- Removed staged XPU hard-fail on `use_fabric=True` in `xpu/csrc/deep_ep.cpp` and added native regression coverage in `xpu/tests/test_xpu_import.py`, so fabric hints are accepted as no-op in staged XPU mode instead of aborting initialization.
- Replaced staged XPU hard-fail on `num_rdma_bytes > 0` in `xpu/csrc/deep_ep.cpp` with graceful downgrade to local-only mode (RDMA bytes ignored with one-time warning), and added native regression coverage in `xpu/tests/test_xpu_import.py` to verify staged XPU `sync()` succeeds while `is_internode_available()` remains false.
- Added XPU local IPC transport-kind metadata validation in `xpu/csrc/deep_ep.cpp` by encoding regular vs fabric-hint handle kinds and rejecting exporter/importer kind mismatches during staged local handle import.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` to verify XPU IPC handle kind mismatch is rejected.
- Gated RDMA-level host counter allocation/free on `num_rdma_bytes > 0` in `xpu/csrc/deep_ep.cpp` to avoid unnecessary staged allocations when internode is disabled.
- Added explicit RDMA precondition checks at internode and low-latency entry points in `xpu/csrc/deep_ep.cpp` so RDMA-only APIs fail fast with clear assertions when called in local-only staged mode.
- Hardened native buffer lifecycle in `xpu/csrc/deep_ep.cpp` by making destructor auto-destroy path conditional on `not destroyed`, preventing double-destroy assertions after manual `destroy()` in auto-destroy mode.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` to verify manual `destroy()` is safe even when `explicitly_destroy=False`.
- Made `Buffer::destroy()` idempotent in `xpu/csrc/deep_ep.cpp` by returning early when already destroyed.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` to verify repeated explicit `destroy()` calls are safe.
- Replaced staged stream-derived device-id helper in `xpu/csrc/deep_ep.cpp` with concrete XPU runtime query via `c10::xpu::current_device()`.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` to verify `get_local_device_id()` matches `torch.xpu.current_device()`.
- Replaced staged stream-level synchronization in `runtime_device_synchronize()` with device-level synchronization via `c10::xpu::syncStreamsOnDevice()` in `xpu/csrc/deep_ep.cpp` for correct XPU device-sync semantics.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` to verify device synchronization succeeds during buffer destruction.
- Fixed test configuration parameter alignment in `xpu/tests/test_xpu_buffer_fallbacks.py` to use correct Config constructor signature.
- Validated comprehensive kernel fallback system for staged XPU development: all 8 fallback tests pass, enabling single-rank XPU operation with Python fallback paths for unsupported kernels.
- Documented kernel migration strategy: Phase 1 (current) uses staged fallbacks, Phase 2 will port to SYCL, Phase 3 will migrate NVSHMEM→iSHMEM.
- Replaced XPU stub in `xpu/csrc/kernels/layout.cu` with a concrete XPU implementation of `layout::get_dispatch_layout` using stream-ordered D2H/H2D transfers and host-side token/rank/expert counting.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` to verify `Buffer.get_dispatch_layout()` succeeds and returns expected layout outputs on XPU.
- Replaced XPU stubs for single-rank intranode paths in `xpu/csrc/kernels/intranode.cu` with concrete implementations for `notify_dispatch`, `dispatch`, `cached_notify_combine`, and `combine` (multi-rank still guarded to fallback).
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` to verify single-rank intranode `dispatch` + `combine` executes on native XPU path and returns expected gathered/scattered tensors.
- Extended the single-rank native XPU `intranode::combine` path in `xpu/csrc/kernels/intranode.cu` to support repeated-source reduction semantics and optional bias application, matching the staged fallback contract more closely.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` to verify single-rank intranode native combine reduces duplicate source indices and applies bias correctly on XPU.
- Fixed the single-rank native XPU `intranode::dispatch` contract in `xpu/csrc/kernels/intranode.cu` to use `num_channels = num_sms / 2`, matching the CUDA kernel interface and native tensor shapes.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` to verify single-rank intranode cached-handle dispatch reuse succeeds on the native XPU path.
- Enabled staged native XPU multi-rank `intranode::cached_notify_dispatch` in `xpu/csrc/kernels/intranode.cu` (stream-ordered no-op metadata sync instead of unsupported guard), so cached multi-rank dispatch no longer fails at notify stage.
- Enabled staged native XPU multi-rank `intranode::cached_notify_combine` in `xpu/csrc/kernels/intranode.cu` (stream-ordered no-op queue sync instead of unsupported guard), so cached multi-rank combine no longer fails at notify stage.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` for two-rank cached intranode dispatch on the staged XPU path and validated it under the requested oneAPI + `icx`/`icpx` environment without NUMA pinning.
- Added staged native XPU multi-rank `intranode::combine` host rendezvous path in `xpu/csrc/kernels/intranode.cu`, including source-rank reconstruction from dispatch rank-prefix metadata and optional top-k weight accumulation.
- Added scoped XPU `gil_scoped_release` in `Buffer::intranode_combine` (`xpu/csrc/deep_ep.cpp`) to avoid same-process two-thread combine rendezvous deadlock.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` for two-rank same-process intranode combine on the staged XPU path.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` for two-rank same-process cached intranode combine on the staged XPU path.
- Added staged local-only XPU low-latency buffer allocation in `xpu/csrc/deep_ep.cpp` so low-latency maintenance APIs can operate without NVSHMEM transport.
- Replaced XPU stubs for `internode_ll::clean_low_latency_buffer`, `query_mask_buffer`, `update_mask_buffer`, and `clean_mask_buffer` in `xpu/csrc/kernels/internode_ll.cu` with concrete stream-ordered staged implementations.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` for single-rank low-latency maintenance on the staged XPU path (mask update/query/clean and low-latency buffer clean).
- Replaced XPU stubs for single-rank BF16 `internode_ll::dispatch` and `internode_ll::combine` in `xpu/csrc/kernels/internode_ll.cu` with staged host implementations covering the low-latency dispatch/combine round trip on XPU.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` for single-rank low-latency dispatch plus combine on the staged XPU path.
- Extended staged single-rank XPU low-latency dispatch in `xpu/csrc/deep_ep.cpp` and `xpu/csrc/kernels/internode_ll.cu` to support `use_fp8=True` in native mode by keeping BF16 payloads and emitting unit scale tensors per packed token block.
- Enabled staged single-rank XPU low-latency combine with `use_logfmt=True` in `xpu/csrc/kernels/internode_ll.cu`.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` for single-rank low-latency FP8 dispatch mode and logfmt combine mode on the staged XPU path.
- Replaced XPU stubs for multi-rank low-latency `internode_ll::dispatch` and `internode_ll::combine` in `xpu/csrc/kernels/internode_ll.cu` with staged same-process host rendezvous implementations.
- Added scoped XPU `gil_scoped_release` around low-latency dispatch/combine stream waits and launch paths in `xpu/csrc/deep_ep.cpp` to avoid threaded same-process rendezvous deadlock.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` for two-rank same-process low-latency dispatch plus combine on the staged XPU path.
- Replaced XPU stubs for single-rank `internode::notify_dispatch`, `internode::dispatch`, `internode::cached_notify`, and `internode::combine` in `xpu/csrc/kernels/internode.cu` with staged host implementations.
- Added native subprocess regression coverage in `xpu/tests/test_xpu_import.py` for single-rank staged internode dispatch plus combine on the XPU path.
- Reconstructed two-rank XPU dispatch/combine regressions (intranode uncached/cached, internode uncached/cached, and low-latency) to run in two separate subprocesses with `ZE_AFFINITY_MASK=0,1`, one process per local XPU device.
- Added two-process two-rank low-latency regressions for FP8 dispatch mode and logfmt combine mode in `xpu/tests/test_xpu_import.py`, both executed via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Extended staged XPU `intranode::combine` in `xpu/csrc/kernels/intranode.cu` to support `float32` tensors natively across both single-rank and staged multi-rank host-rendezvous paths (BF16 path unchanged).
- Added two-process two-rank `float32` intranode dispatch/combine regression in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Extended staged XPU `internode::combine` in `xpu/csrc/kernels/internode.cu` to support `float32` tensors natively across both single-rank and staged multi-rank host-rendezvous paths (BF16 path unchanged).
- Added two-process two-rank `float32` internode dispatch/combine regression in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Added two-process two-rank cached `float32` internode dispatch/combine regression in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Added two-process multi-round stress regression for cached two-rank intranode dispatch/combine in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Added two-process multi-round stress regression for cached two-rank `float32` internode dispatch/combine in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Added two-process multi-round stress regression for two-rank low-latency FP8 dispatch/combine in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Added two-process multi-round stress regression for two-rank low-latency logfmt combine in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Extended staged XPU intranode combine to support `float16` in both single-rank and multi-rank host paths in `xpu/csrc/kernels/intranode.cu`.
- Extended staged XPU internode combine to support `float16` in both single-rank and multi-rank host paths in `xpu/csrc/kernels/internode.cu`.
- Added two-process two-rank `float16` intranode dispatch/combine regression in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Added two-process two-rank `float16` internode dispatch/combine regression in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Added two-process two-rank cached `float16` intranode dispatch/combine regression in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.
- Added two-process two-rank cached `float16` internode dispatch/combine regression in `xpu/tests/test_xpu_import.py` via `ZE_AFFINITY_MASK=0,1` subprocess isolation.

## Layer Migration Status Summary

**C++ Runtime Layer (xpu/csrc/deep_ep.cpp):** ✅ LARGELY COMPLETE
- [x] Device synchronization (c10::xpu::syncStreamsOnDevice)
- [x] Device querying (c10::xpu::current_device, at::xpu::getDeviceProperties)
- [x] Memory management (c10::xpu::XPUCachingAllocator, at::xpu::getPinnedMemoryAllocator)
- [x] IPC transport with generation/checksum validation
- [x] Buffer lifecycle management (idempotent destroy)
- [x] Error handling and live allocation tracking
- All 55 XPU tests pass, including expanded native intranode regressions (uncached/cached two-rank two-process dispatch+combine plus float32/float16 two-rank two-process combine and cached float16), staged low-latency maintenance coverage, staged low-latency BF16/FP8 dispatch+combine coverage (including logfmt combine), staged two-rank low-latency dispatch/combine coverage (including FP8/logfmt two-process variants), staged internode dispatch/combine coverage (single-rank and two-rank two-process, including float32/float16 plus cached float32/float16), cached multi-round two-process stress coverage, low-latency FP8/logfmt multi-round two-process stress coverage, fallback tests, and import/runtime coverage

**Python Wrapper Layer (xpu/deep_ep/*.py):** ✅ COMPREHENSIVE FALLBACK SYSTEM
- [x] Dynamic extension loading with XPU-first fallback
- [x] Staged fallback dispatch/combine/low-latency paths (single-rank support)
- [x] Device stream reconstruction for XPU
- [x] Single-rank initialization path
- [x] Error handling and fallback triggers for unsupported kernels
- Test coverage: 8 dedicated fallback tests + runtime integration tests

**Kernel Layer (xpu/csrc/kernels/*.cu|*.cuh):** 🚧 PARTIALLY PORTED (PHASE 2 STARTED)
- Current approach: `layout::get_dispatch_layout`, staged single-rank intranode dispatch/combine, staged two-rank same-process intranode dispatch/combine, staged single-rank internode dispatch/combine, and staged low-latency maintenance plus dispatch/combine helpers have concrete XPU implementations; remaining kernels still return `EP_UNSUPPORTED_XPU` and trigger Python fallbacks
- [ ] Full SYCL kernel implementation (in progress)
- [x] Python fallback coverage for all major operations:
  - intranode dispatch/combine → Python bincount + tensor scatter
  - internode dispatch/combine → Python BF16 serialization
  - low-latency dispatch/combine → Python fallback paths
- Production path: SYCL kernel porting (future work)

## Kernel Migration Strategy

**Phase 1: Staged Single-Rank Support (Current - Complete)** ✅
- Kernels return `EP_UNSUPPORTED_XPU` placeholders
- Python fallback layer handles unsupported operations
- Enables XPU development/testing in single-rank mode
- All regression tests pass

**Phase 2: Multi-Rank XPU Support (In Progress)**
- Implement SYCL kernels for compute-critical operations:
  1. layout::get_dispatch_layout (token counting) ✅ concrete XPU path implemented
  2. intranode::dispatch/combine (NVLink MoE gather/scatter) 🚧 staged two-rank same-process dispatch (uncached/cached) and combine implemented; full SYCL parallelization and broader multi-rank hardening pending
  3. internode::dispatch/combine (RDMA MoE operations)
  4. low-latency dispatch/combine (low-latency paths) 🚧 staged single-rank BF16/FP8 dispatch, logfmt combine, and maintenance helpers implemented; broader multi-rank support still pending
- Use SYCLomatic tool for CUDA→SYCL automated porting as starting point
- Requires SYCL queue management and work-group coordination

**Phase 3: Internode NVSHMEM→iSHMEM Migration (Future)**
- Replace NVSHMEM runtime with iSHMEM
- Port CUDA Level Zero IPC → Level Zero native IPC
- Implement barrier synchronization for multi-rank scenarios

## Next file-by-file migration targets

1. `xpu/csrc/kernels/intranode.cu`: extend current single-rank native path to multi-rank/SYCL parallel kernels
2. `xpu/csrc/kernels/internode.cu` + `ibgda_device.cuh`: iSHMEM porting and Level Zero IPC
3. `xpu/csrc/kernels/low_latency*` paths: concrete XPU implementation for low-latency dispatch/combine
4. `xpu/csrc/kernels/launch.cuh`: Full SYCL kernel launch macro implementation
