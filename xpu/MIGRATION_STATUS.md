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

## Next file-by-file migration targets

1. `xpu/csrc/deep_ep.cpp`: implement XPU memory management and IPC backend by replacing helper-wrapper CUDA internals with XPU/Level Zero implementations.
2. `xpu/csrc/kernels/*.cu|*.cuh`: migrate kernels to SYCL and remove PTX inline assembly.
3. `xpu/csrc/kernels/internode*.cu` and `ibgda_device.cuh`: replace NVSHMEM/CUDA IPC paths with iSHMEM and Level Zero IPC.
4. `xpu/tests`: port distributed tests to xpu backend and ccl backend.
5. `xpu/csrc/kernels/launch.cuh`: replace staged XPU launch adapter stub with real SYCL/XPU kernel dispatch.
