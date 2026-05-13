# DeepEP Copilot Instructions

## First references

- Prefer linking to existing docs instead of duplicating them:
  - Main usage and platform notes: [README.md](../README.md)
  - NVSHMEM setup details: [third-party/README.md](../third-party/README.md)
  - Build switches and source list: [setup.py](../setup.py)

## Build, test, and lint commands

### Build and install

```bash
# Development build of the CUDA/PyTorch extension
NVSHMEM_DIR=/path/to/nvshmem python setup.py build

# Install from source
NVSHMEM_DIR=/path/to/nvshmem python setup.py install

# Build a wheel and install it
bash install.sh
```

- `setup.py` builds `deep_ep_cpp` from `csrc/deep_ep.cpp` plus files under `csrc/kernels/`.
- If `NVSHMEM_DIR` is unset and `nvidia.nvshmem` is unavailable, internode and low-latency features are compiled out.
- Important build env vars: `NVSHMEM_DIR`, `DISABLE_SM90_FEATURES`, `TORCH_CUDA_ARCH_LIST`, `DISABLE_AGGRESSIVE_PTX_INSTRS`, and optional `TOPK_IDX_BITS`.

### Tests

```bash
# Intranode
python tests/test_intranode.py --num-processes 2

# Internode (cluster env required)
WORLD_SIZE=2 RANK=0 MASTER_ADDR=host0 MASTER_PORT=8361 \
python tests/test_internode.py --num-processes 2

# Low latency
python tests/test_low_latency.py --num-processes 2
```

- Test granularity is per script.
- `tests/utils.py:init_dist()` reads `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `RANK`.

### Format

```bash
bash format.sh
bash format.sh --all
```

## Layered architecture contract

Agents should preserve strict boundaries to make CUDA to Intel XPU migration feasible.

1. Python API layer
  - Files: `deep_ep/__init__.py`, `deep_ep/buffer.py`
  - Responsibility: user-facing API, argument validation, process-group coordination, event/handle plumbing.
  - Rule: do not introduce backend-specific device code or low-level memory semantics here.

2. Runtime and binding layer
  - Files: `csrc/deep_ep.cpp`, `csrc/deep_ep.hpp`
  - Responsibility: pybind boundary, runtime ownership, buffer/IPC setup, stream wiring.
  - Rule: keep this layer backend-dispatch oriented; avoid embedding kernel math or routing logic.

3. Kernel API and orchestration layer
  - Files: `csrc/kernels/api.cuh`, `csrc/kernels/runtime.cu`, `csrc/kernels/layout.cu`, `csrc/kernels/launch.cuh`
  - Responsibility: algorithm staging, launch decisions, backend-neutral kernel entry signatures.
  - Rule: avoid direct inline PTX or backend-only intrinsics in orchestration paths.

4. Backend primitive and device-kernel layer
  - Files: `csrc/kernels/utils.cuh`, `csrc/kernels/intranode.cu`, `csrc/kernels/internode.cu`, `csrc/kernels/internode_ll.cu`
  - Responsibility: backend-specific primitives, memory-order details, transport-specific fast paths.
  - Rule: confine CUDA-only constructs (inline PTX, cooperative launch attributes, CUDA IPC specifics) to this layer.

## CUDA to XPU migration guidance (kernel-first)

When implementing migration work, prefer this sequence:

1. Isolate primitives before porting kernels
  - Wrap memory fences, atomics, volatile/noncoherent loads, and launch attributes behind reusable helper APIs.
  - New raw `asm volatile(...)` usage should be rejected unless it is inside a backend-primitive helper.

2. Keep layout and metadata logic backend-agnostic
  - `csrc/kernels/layout.cu` style logic should remain algorithm-centric and avoid transport/device primitive details.

3. Separate transport concerns from math/data movement
  - Keep NVSHMEM/RDMA specialization out of generic dispatch/combine shape logic where possible.

4. Minimize Python and pybind churn
  - Preserve public API contracts (`Buffer`, `EventOverlap`, handle reuse semantics) while changing backend internals.

5. Preserve compatibility switches
  - Any backend refactor should continue honoring `DISABLE_SM90_FEATURES`, `DISABLE_AGGRESSIVE_PTX_INSTRS`, and `TOPK_IDX_BITS` behavior.

## Project conventions that must stay true

- Treat `deep_ep.Buffer` as communication-state ownership boundary.
- `topk_idx` must use `deep_ep.topk_idx_t`.
- Reuse `handle` from `dispatch()` for cached dispatch paths and `combine()`.
- Prefer size/config helpers over hard-coded sizes:
  - `Buffer.get_dispatch_config(...)`
  - `Buffer.get_combine_config(...)`
  - `Config.get_nvl_buffer_size_hint(...)`
  - `Config.get_rdma_buffer_size_hint(...)`
  - `Buffer.get_low_latency_rdma_size_hint(...)`
- For low-latency mode, `num_qps_per_rank` is expected to equal local experts.
- Before switching a buffer from normal mode to low latency, call `Buffer.clean_low_latency_buffer(...)`.
- Low-latency APIs reuse only two internal buffers; do not keep more than two outstanding low-latency results.
- Prefer explicit `destroy()` with `explicitly_destroy=True` in tests and long-running workflows.
