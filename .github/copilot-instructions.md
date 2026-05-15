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
