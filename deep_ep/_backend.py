import os
from dataclasses import dataclass

import torch

try:
    import deep_ep_cpp as _cpp_backend
except ImportError:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        from . import _xpu_cpp_stub as _cpp_backend
    else:
        _cpp_backend = None


CPP_BACKEND_AVAILABLE = _cpp_backend is not None
CPP_RUNTIME_AVAILABLE = _cpp_backend is not None and hasattr(_cpp_backend, 'Buffer')


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


def supports_xpu_native_intranode() -> bool:
    if get_runtime_backend_name() != 'xpu' or not CPP_RUNTIME_AVAILABLE:
        return False
    if hasattr(_cpp_backend, 'supports_xpu_native_intranode'):
        return _cpp_backend.supports_xpu_native_intranode()
    required_methods = ('get_dispatch_layout', 'intranode_dispatch', 'intranode_combine')
    return all(hasattr(_cpp_backend.Buffer, name) for name in required_methods)


def _get_default_device_type() -> str:
    try:
        return torch.get_default_device().type
    except (AttributeError, RuntimeError):
        return 'cpu'


def get_runtime_backend_name() -> str:
    default_device_type = _get_default_device_type()
    if default_device_type == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if default_device_type == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available() and not torch.cuda.is_available():
        return 'xpu'
    if torch.cuda.is_available():
        return 'cuda'
    return default_device_type


def supports_native_runtime() -> bool:
    runtime_backend = get_runtime_backend_name()
    if runtime_backend == 'xpu':
        return supports_xpu_native_intranode()
    return CPP_RUNTIME_AVAILABLE


def supports_intranode() -> bool:
    runtime_backend = get_runtime_backend_name()
    return runtime_backend == 'xpu' or supports_native_runtime()


def supports_internode() -> bool:
    return supports_native_runtime()


def supports_low_latency() -> bool:
    runtime_backend = get_runtime_backend_name()
    return runtime_backend == 'xpu' or supports_native_runtime()


def using_python_backend() -> bool:
    runtime_backend = get_runtime_backend_name()
    return (runtime_backend == 'xpu' and not supports_native_runtime()) or not CPP_RUNTIME_AVAILABLE
