# DeepEP

This `xpu/` subtree is an **experimental mirror** of DeepEP for Intel XPU/SYCL. It preserves the upstream Python/C++ API shape where practical, but the current XPU state is narrower than the CUDA README at the repository root.

Current status:

- `DEEP_EP_XPU_BUILD_MODE=intranode`: experimental build/import path plus the currently exercised intranode runtime
- `DEEP_EP_XPU_BUILD_MODE=full`: experimental build/import path for the mirrored full tree, including internode sources and explicit unsupported low-latency entrypoints
- low-latency/decode APIs: intentionally unsupported on XPU for now; they fail immediately with a clear runtime error instead of pretending to work
- tests under `xpu/tests/`: XPU-specific validation harnesses, with quick smoke modes for the paths that are expected to work today

Notice: the implementation in this library may have some slight differences from the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper.

## Upstream context vs. current XPU state

The upstream DeepEP project targets CUDA/NVSHMEM and documents mature throughput/latency results there. This mirrored `xpu/` tree does **not** claim those CUDA results as XPU results.

For XPU today:

- intranode correctness is the current experimental runtime focus on peer-access-capable local XPU setups
- full mode build/import is available experimentally
- internode runtime exists in the mirrored tree but should still be treated as early/cluster-dependent
- low-latency API coverage is intentionally negative testing only (`test_low_latency.py`)

## Quick start

### XPU build status

The mirrored `xpu/` tree is **experimental**. The build path is now explicit about that state:

- default behavior is a status/requirements probe
- actual extension compilation is gated behind `DEEP_EP_XPU_ALLOW_EXPERIMENTAL_BUILD=1`
- `DEEP_EP_XPU_BUILD_MODE=intranode` limits the source set to the current intranode-oriented mirror
- `DEEP_EP_XPU_BUILD_MODE=full` adds the mirrored internode sources plus an explicit unsupported low-latency surface and requires iSHMEM

### Requirements

- Intel XPU-capable PyTorch environment
- Python 3.8 and above
- oneAPI compiler environment loaded via `source /opt/intel/oneapi/setvars.sh --force`
- `CC=icx`
- `CXX=icpx`
- `TORCH_XPU_ARCH_LIST` set appropriately for the target device (defaults to `pvc`)
- `ISHMEM_DIR=/opt/intel/ishmem` for `DEEP_EP_XPU_BUILD_MODE=full`

### Build status probe

```bash
cd xpu
python setup.py xpu_build_info
# or
./install.sh --check
```

### Experimental build attempt

```bash
cd xpu
export CC=icx
export CXX=icpx
source /opt/intel/oneapi/setvars.sh --force

# Intranode-only mirrored source set
DEEP_EP_XPU_BUILD_MODE=intranode DEEP_EP_XPU_ALLOW_EXPERIMENTAL_BUILD=1 \
python setup.py build_ext --inplace

# Full mirrored source set (requires iSHMEM)
ISHMEM_DIR=/opt/intel/ishmem \
DEEP_EP_XPU_BUILD_MODE=full DEEP_EP_XPU_ALLOW_EXPERIMENTAL_BUILD=1 \
python setup.py build_ext --inplace

# Import smoke test
python -c "import deep_ep, deep_ep_xpu_cpp; print('DeepEP XPU import OK')"
```

Or through the wrapper script:

```bash
cd xpu
./install.sh --build --experimental --mode intranode
./install.sh --build --experimental --mode full
```

The wrapper script ends with an import smoke test so a "successful build" is also a successful `import deep_ep` / `import deep_ep_xpu_cpp`.

#### XPU build environment variables

- `DEEP_EP_XPU_BUILD_MODE`: `intranode` or `full`
- `DEEP_EP_XPU_ALLOW_EXPERIMENTAL_BUILD`: set to `1` to opt into a real compile attempt
- `TORCH_XPU_ARCH_LIST`: target XPU architecture list, default `pvc`
- `ISHMEM_DIR`: required for `full` mode, optional for `intranode`
- `TOPK_IDX_BITS`: optional, preserved from the CUDA build flow

## Validation

The following checks match the current XPU state on a peer-access-capable local XPU node:

```bash
cd xpu

# Build/import path
python setup.py xpu_build_info
python -c "import deep_ep, deep_ep_xpu_cpp; print('imports OK')"

# Expected-pass single-node smoke coverage
python tests/test_low_latency.py --num-processes 2
python tests/test_intranode.py \
  --num-processes 8 \
  --num-tokens 32 \
  --hidden 128 \
  --num-topk 2 \
  --num-experts 16 \
  --smoke-only
```

If oneCCL/topology discovery reports the local devices as PCIe-only or otherwise cannot establish the expected peer-access fabric, treat intranode as unsupported in that environment even though the mirrored intranode path exists.

Internode runtime validation still requires a real multi-node launch with 8 local XPU ranks per node. A quick correctness-only run looks like:

```bash
WORLD_SIZE=<num_nodes> RANK=<node_rank> MASTER_ADDR=<host> MASTER_PORT=<port> \
python tests/test_internode.py \
  --num-processes 8 \
  --num-tokens 32 \
  --hidden 128 \
  --num-topk 2 \
  --num-experts 16 \
  --smoke-only
```

## Interfaces and examples

### Example use in model training or inference prefilling

The normal kernels can be used in model training or the inference prefilling phase (without the backward part) as the below example code shows.

```python
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

from deep_ep import Buffer, EventOverlap

# Communication buffer (will allocate at runtime)
_buffer: Optional[Buffer] = None

# Set the number of SMs to use
# NOTES: this is a static variable
Buffer.set_num_sms(24)


# You may call this function at the framework initialization
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    # NOTES: an optional `previous_event` means an event captured on the current XPU stream that you want to make it as a dependency
    # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    global _buffer

    # Calculate layout before actual dispatch
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(topk_idx, num_experts,
                                    previous_event=previous_event, async_finish=True,
                                    allocate_on_comm_stream=previous_event is not None)
    # Do MoE dispatch
    # NOTES: the CPU will wait for the device signal to arrive, so this is not compatible with graph capture
    # Unless you specify `num_worst_tokens`, but this flag is for intranode only
    # For more advanced usages, please refer to the docs of the `dispatch` function
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=True,
                         allocate_on_comm_stream=True)
    # For event management, please refer to the docs of the `EventOverlap` class
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor, grad_recv_topk_weights: torch.Tensor, handle: Tuple) -> \
        Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    global _buffer

    # The backward process of MoE dispatch is actually a combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_grad_x, combined_grad_recv_topk_weights, event = \
        _buffer.combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights, async_finish=True)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_grad_x, combined_grad_recv_topk_weights, event


def combine_forward(x: torch.Tensor, handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # Do MoE combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(x, handle, async_finish=True, previous_event=previous_event,
                                           allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    global _buffer

    # The backward process of MoE combine is actually a dispatch
    # For more advanced usages, please refer to the docs of the `dispatch` function
    grad_x, _, _, _, _, event = _buffer.dispatch(grad_combined_x, handle=handle, async_finish=True,
                                                 previous_event=previous_event,
                                                 allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return grad_x, event
```

Moreover, inside the dispatch function, we may not know how many tokens to receive for the current rank. So an implicit CPU wait for GPU received count signal will be involved, as the following figure shows.

![normal](figures/normal.png)

### Example use in inference decoding

Unlike CUDA, the mirrored XPU tree does **not** currently provide this decoding path. The relevant entrypoints (`Buffer(..., low_latency_mode=True)`, `get_low_latency_rdma_size_hint`, `low_latency_dispatch`, `low_latency_combine`, and mask-buffer helpers) are kept only so callers fail immediately with a clear unsupported error instead of reaching half-translated CUDA/NVSHMEM assumptions.

The overlap figure below therefore remains reference material from the original CUDA design, not an enabled XPU feature today.

![low-latency](figures/low-latency.png)

## XPU roadmap

- [x] Experimental intranode build/import/runtime path
- [x] Experimental full build/import path with mirrored internode sources
- [x] Explicit unsupported low-latency API surface and negative tests
- [ ] Broader internode runtime validation on real clusters
- [ ] Portable low-latency SYCL/iSHMEM implementation

## Notices

#### Easier potential overall design

The current DeepEP implementation uses queues for communication buffers which save memory but introduce complexity and potential deadlocks. If you're implementing your own version based on DeepEP, consider using fixed-size buffers allocated to maximum capacity for simplicity and better performance. For a detailed discussion of this alternative approach, see https://github.com/deepseek-ai/DeepEP/issues/39.

#### Upstream CUDA/PTX notes

Many design notes in the upstream project discuss CUDA/PTX/NVSHMEM behavior. Keep those as upstream context only; the mirrored XPU tree does not currently claim feature parity there.

#### Auto-tuning on your cluster

For better performance on your cluster, we recommend to run all the tests and use the best auto-tuned configuration. The default configurations are optimized on the DeepSeek's internal cluster.

For the mirrored XPU tree specifically, the native intranode implementation now includes a restored cached no-topk fast path with reusable host-staged Level Zero IPC transport. It passes the smoke/correctness coverage and produces meaningful point benchmarks, but it is still not a CUDA-style steady-state transport/kernel path, so the upstream intranode auto-tuning loop remains intentionally disabled on XPU.

## License

This code repository is released under [the MIT License](LICENSE), except for codes that reference NVSHMEM (including `csrc/kernels/ibgda_device.cuh` and `third-party/nvshmem.patch`), which are subject to [NVSHMEM SLA](https://docs.nvidia.com/nvshmem/api/sla.html).

## Experimental Branches

- [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
  - Removing the copy between PyTorch tensors and communication buffers, which reduces the SM usages significantly for normal kernels
  - This PR is authored by **Tencent Network Platform Department**
- [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
  - Using a low-latency protocol removes the extra RTT latency introduced by RDMA atomic OPs
- [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
  - A new backend implementation using TMA instructions for minimal SM usage and larger NVLink domain support
  - Fine-grained communication-computation overlap for single-batch scenarios
  - PCIe kernel support for non-NVLink environments
  - NVFP4 data type support
- [AntGroup-Opt](https://github.com/deepseek-ai/DeepEP/tree/antgroup-opt)
  - This optimization series is authored by **AntGroup Network Platform Department**
  - [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347) Eliminating SM from RDMA path by decoupling comm-kernel execution from NIC token transfer, freeing SMs for compute
  - [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483) Overlapping Down GEMM computation with Combine Send communication via signaling mechanism to reduce end-to-end latency
  - [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500) Optimizing cross-node LL operator communication using rail-optimized forwarding and data merging to reduce latency
- [Mori-EP](https://github.com/deepseek-ai/DeepEP/tree/mori-ep)
  - ROCm/AMD GPU support powered by [MORI](https://github.com/ROCm/mori) backend (low-latency mode)

## Community Forks

- [uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep) - Enables running DeepEP on heterogeneous GPUs (e.g., Nvidia, AMD) and NICs (e.g., EFA, Broadcom, CX7)
- [Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP) - Adds multi-QP solution and dual-port NIC support in IBRC transport
- [antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace) - A diagnostic analyzer for efficient and precise localization of slow ranks
- [ROCm/mori](https://github.com/ROCm/mori) - AMD's next-generation communication library for performance-critical AI workloads (e.g., Wide EP, KVCache transfer, Collectives)

## Citation

If you use this codebase or otherwise find our work valuable, please cite:

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```
