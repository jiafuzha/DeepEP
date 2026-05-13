import os
from dataclasses import dataclass

import torch

try:
    import deep_ep_cpp as _cpp_backend
except ImportError:
    _cpp_backend = None


CPP_BACKEND_AVAILABLE = _cpp_backend is not None


@dataclass(frozen=True)
class Config:
    num_sms: int
    nvl_chunk_size: int
    nvl_buffer_size: int
    rdma_chunk_size: int = 0
    rdma_buffer_size: int = 0


class EventHandle:

    def current_stream_wait(self) -> None:
        return


if CPP_BACKEND_AVAILABLE:
    Config = _cpp_backend.Config
    EventHandle = _cpp_backend.EventHandle
    topk_idx_t = _cpp_backend.topk_idx_t

    def is_sm90_compiled() -> bool:
        return _cpp_backend.is_sm90_compiled()

    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int) -> int:
        return _cpp_backend.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)
else:
    topk_idx_t = torch.int32 if int(os.environ.get("TOPK_IDX_BITS", "64")) == 32 else torch.int64

    def is_sm90_compiled() -> bool:
        return False

    def get_low_latency_rdma_size_hint(*_args, **_kwargs) -> int:
        raise NotImplementedError("Low-latency kernels are not available in the Python XPU backend")


def get_cpp_backend():
    return _cpp_backend


def using_python_backend() -> bool:
    return not CPP_BACKEND_AVAILABLE
