# DeepEP Copilot Instructions

## Build, test, and lint

- Build the CUDA/PyTorch extension in-place for development with `NVSHMEM_DIR=/path/to/nvshmem python setup.py build`. If `NVSHMEM_DIR` is unset and `nvidia.nvshmem` is not installed, the extension still builds, but internode and low-latency features are compiled out.
- Install a wheel with `bash install.sh` or install directly with `NVSHMEM_DIR=/path/to/nvshmem python setup.py install`.
- Format and lint with `bash format.sh` (changed files) or `bash format.sh --all` (entire repo). This script runs `yapf`, `ruff check`, and `clang-format` for C/CUDA sources; versions are pinned in `requirements-lint.txt`.
- There is no pytest suite. The tests are standalone distributed scripts that spawn workers with `torch.multiprocessing.spawn`, so “run a single test” means running one script:
  - Intranode: `python tests/test_intranode.py --num-processes 8`
  - Internode: `WORLD_SIZE=<num_nodes> RANK=<node_rank> MASTER_ADDR=<host> MASTER_PORT=<port> python tests/test_internode.py --num-processes 8`
  - Low-latency: `python tests/test_low_latency.py --num-processes 8`
- Before changing distributed test behavior, read `tests/utils.py`; `init_dist()` is intentionally the local customization point for cluster setup and launch semantics.

## High-level architecture

- `setup.py` builds a single CUDA extension named `deep_ep_cpp` from `csrc/deep_ep.cpp` plus CUDA kernels under `csrc/kernels/`. The Python package in `deep_ep/` is a thin wrapper over that extension, not a separate implementation.
- Public Python entrypoints are exported from `deep_ep/__init__.py`: `Buffer`, `EventOverlap`, `Config`, and `topk_idx_t`. `Buffer` in `deep_ep/buffer.py` owns the runtime, chooses intranode vs internode code paths, and exposes the three communication modes used by the project:
  - normal intranode dispatch/combine over NVLink
  - normal internode dispatch/combine over NVLink + RDMA/NVSHMEM
  - low-latency dispatch/combine over RDMA/IBGDA
- The normal workflow is split into two phases: `get_dispatch_layout(...)` computes routing metadata, then `dispatch(...)` sends tokens and returns a reusable `handle`, and `combine(...)` reduces results back with that same handle. The backward path in model code mirrors this: dispatch backward is combine, and combine backward is dispatch.
- The C++ `Buffer` runtime in `csrc/deep_ep.hpp` / `csrc/deep_ep.cpp` manages CUDA IPC/shared memory for local NVLink peers, NVSHMEM state for RDMA peers, a dedicated communication stream, and host-mapped counters/workspaces. Python `Buffer` methods are mostly argument validation, handle packing/unpacking, and mode selection around that runtime.
- `csrc/config.hpp` is where buffer-size hints and low-latency RDMA layout are defined. `Config` is performance-critical, not just metadata: tests print tuned values that are meant to replace or inform the defaults in `Buffer.get_dispatch_config()` / `get_combine_config()`.
- The test scripts are also tuning harnesses. They validate correctness, benchmark kernels, and emit the best SM/chunk-size settings for the current cluster; use them as the reference when changing kernel behavior or default configs.

## Key conventions

- `topk_idx` is not an arbitrary integer tensor: it must use `deep_ep.topk_idx_t`, and `-1` is the sentinel for “no expert selected”. Preserve that dtype/sentinel behavior across Python and C++ changes.
- `Buffer` must be constructed with either a `torch.distributed.ProcessGroup` or an `mpi4py` communicator. The wrapper synchronizes device IDs, CUDA IPC handles, and NVSHMEM unique IDs during initialization; edits that bypass that setup will break cross-rank communication.
- Normal-kernel defaults are driven by the class variable `Buffer.num_sms`. `Buffer.set_num_sms()` changes the recommended configs globally for later calls; low-latency kernels do not use this SM tuning path.
- `previous_event`, `async_finish`, and `allocate_on_comm_stream` are the repo’s standard overlap controls. Keep event chaining explicit rather than adding hidden synchronizations.
- If `explicitly_destroy=True`, callers are expected to call `buffer.destroy()`. This is intentional: the code avoids relying on Python destructor timing for GPU communication resource cleanup.
- Low-latency mode has stricter lifecycle rules than normal mode:
  - the registered RDMA buffers are double-buffered, so you cannot safely retain more than two outstanding low-latency result tensors at once
  - call `clean_low_latency_buffer(...)` before switching from normal kernels to low-latency kernels on a reused buffer
  - for best performance, `num_qps_per_rank` should equal the number of local experts
- Intranode assumptions are enforced, not advisory: the code checks NVLink visibility, and PCIe-only setups are only accepted for small pairwise cases. Internode tests additionally assume 8 local ranks per node.
- `setup.py` uses environment variables as feature gates. The important ones are `NVSHMEM_DIR`, `DISABLE_SM90_FEATURES`, `TORCH_CUDA_ARCH_LIST`, `DISABLE_AGGRESSIVE_PTX_INSTRS`, and `TOPK_IDX_BITS`. Respect existing compile-time gates instead of hardcoding architecture-specific behavior in Python.

## Intel XPU migration notes

- For CUDA-to-Intel-XPU work, also consult `.github/agents/xpu-migration.agent.md`. It is the repo-local source for SYCL, Level Zero IPC, PTX/vISA, and NVSHMEM-to-iSHMEM migration guidance.
- Migrate incrementally and keep the current CUDA path working unless the task is explicitly XPU-only. Preserve the public Python API and communication flow: `Buffer`, `EventOverlap`, `Config`, `topk_idx_t`, `get_dispatch_layout(...) -> dispatch(...) -> combine(...)`, plus the overlap controls.
- The main migration seams are:
  - `deep_ep/buffer.py` and `deep_ep/utils.py` for `torch.cuda`, CUDA stream/event handling, and NVLink/NVML assumptions
  - `tests/utils.py` for `nccl`, CUDA default-device setup, and distributed bootstrap
  - `csrc/deep_ep.cpp` / `csrc/deep_ep.hpp` for runtime ownership, shared-memory/IPC, stream handling, and explicit teardown
  - `csrc/kernels/*` for CUDA kernels, PTX assumptions, and NVSHMEM/IBGDA logic
- `setup.py` is CUDA-only today (`CUDAExtension`, `.cu` sources, `TORCH_CUDA_ARCH_LIST`, SM90/PTX flags). XPU support should use a separate or conditional build path rather than forcing XPU behavior through the current CUDA build flags.
- Treat `shared_memory::SharedMemoryAllocator` and `Buffer::sync()` as the swap points for CUDA IPC/fabric to Level Zero IPC. Preserve handle exchange, ownership, synchronization, and teardown ordering when changing them.
- Do not mechanically translate PTX or NVSHMEM internals. Prefer SYCL/compiler primitives where possible, keep ordering/completion semantics explicit, and map NVSHMEM/IBGDA behavior to iSHMEM/IBGDA equivalents without dropping barriers or `quiet`/`fence` semantics.
