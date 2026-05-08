"""Temporary Python stub for the XPU extension during staged migration."""

from dataclasses import dataclass
from typing import Any, Optional

import torch


topk_idx_t = torch.int64


@dataclass
class Config:
    num_sms: int
    ctas_per_sm: int
    nvl_chunk_size: int
    rdma_channels: int
    bytes_per_element: int


class EventHandle:
    def current_stream_wait(self) -> None:
        return None


class Buffer:
    def __init__(self,
                 rank: int,
                 num_ranks: int,
                 num_nvl_bytes: int,
                 num_rdma_bytes: int,
                 low_latency_mode: bool,
                 explicitly_destroy: bool,
                 enable_shrink: bool,
                 use_fabric: bool) -> None:
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink
        self.use_fabric = use_fabric
        self.available = False
        self.destroyed = False

        self._nvl_rank = rank % 8
        self._num_nvl_ranks = min(num_ranks, 8)
        self._rdma_rank = rank // 8
        self._num_rdma_ranks = max(1, (num_ranks + 7) // 8)

        self._device = self._detect_device()
        self._nvl_buffer = self._alloc_byte_buffer(num_nvl_bytes)
        self._rdma_buffer = self._alloc_byte_buffer(num_rdma_bytes)
        self._low_latency_mask_status = torch.zeros((num_ranks,), dtype=torch.int)

    @staticmethod
    def _detect_device() -> torch.device:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            idx = torch.xpu.current_device() if hasattr(torch.xpu, 'current_device') else 0
            return torch.device('xpu', idx)
        if torch.cuda.is_available():
            return torch.device('cuda', torch.cuda.current_device())
        return torch.device('cpu')

    def _alloc_byte_buffer(self, num_bytes: int) -> Optional[torch.Tensor]:
        if num_bytes <= 0:
            return None
        return torch.empty((num_bytes,), dtype=torch.uint8, device=self._device)

    @staticmethod
    def _dtype_nbytes(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    def _unsupported(self, op_name: str) -> None:
        raise NotImplementedError(f'{op_name} is not implemented for XPU staged backend yet')

    def is_available(self) -> bool:
        return self.available and not self.destroyed

    def is_internode_available(self) -> bool:
        return self.is_available() and self.num_rdma_bytes > 0

    def get_num_rdma_ranks(self) -> int:
        return self._num_rdma_ranks

    def get_rdma_rank(self) -> int:
        return self._rdma_rank

    def get_root_rdma_rank(self, global_rank: bool) -> int:
        if global_rank:
            return self._rdma_rank * self._num_nvl_ranks
        return 0

    def get_local_device_id(self) -> int:
        if self._device.type in ('cuda', 'xpu'):
            return int(self._device.index or 0)
        return 0

    def get_local_ipc_handle(self) -> bytes:
        # Keep a fixed-size opaque payload to match native handle exchange flow.
        payload = bytearray(72)
        payload[0:8] = int(self.rank).to_bytes(8, byteorder='little', signed=True)
        payload[8:16] = int(self.num_ranks).to_bytes(8, byteorder='little', signed=True)
        payload[16:24] = int(self.num_nvl_bytes).to_bytes(8, byteorder='little', signed=True)
        payload[24:32] = int(self.num_rdma_bytes).to_bytes(8, byteorder='little', signed=True)
        return bytes(payload)

    def get_local_nvshmem_unique_id(self) -> bytes:
        return b'DEEPEP_XPU_STAGED_UID_000000000000'

    def get_local_buffer_tensor(self, dtype: torch.dtype, offset: int, use_rdma_buffer: bool) -> torch.Tensor:
        source = self._rdma_buffer if use_rdma_buffer else self._nvl_buffer
        if source is None:
            return torch.empty((0,), dtype=dtype, device=self._device)
        elem_size = self._dtype_nbytes(dtype)
        if offset < 0:
            raise ValueError('offset must be non-negative')
        byte_offset = offset
        if byte_offset > source.numel():
            raise ValueError('offset exceeds buffer size')
        usable_bytes = source.numel() - byte_offset
        usable_elems = usable_bytes // elem_size
        if usable_elems == 0:
            return torch.empty((0,), dtype=dtype, device=self._device)
        sliced = source.narrow(0, byte_offset, usable_elems * elem_size)
        return sliced.view(dtype)

    def get_comm_stream(self) -> Any:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.xpu.current_stream()
        if torch.cuda.is_available():
            return torch.cuda.current_stream()
        return None

    def sync(self, *_: Any, **__: Any) -> None:
        self.available = True

    def destroy(self) -> None:
        self.available = False
        self.destroyed = True
        self._nvl_buffer = None
        self._rdma_buffer = None

    def get_dispatch_layout(self, *_: Any, **__: Any) -> Any:
        self._unsupported('get_dispatch_layout')

    def intranode_dispatch(self, *_: Any, **__: Any) -> Any:
        self._unsupported('intranode_dispatch')

    def intranode_combine(self, *_: Any, **__: Any) -> Any:
        self._unsupported('intranode_combine')

    def internode_dispatch(self, *_: Any, **__: Any) -> Any:
        self._unsupported('internode_dispatch')

    def internode_combine(self, *_: Any, **__: Any) -> Any:
        self._unsupported('internode_combine')

    def low_latency_dispatch(self, *_: Any, **__: Any) -> Any:
        self._unsupported('low_latency_dispatch')

    def low_latency_combine(self, *_: Any, **__: Any) -> Any:
        self._unsupported('low_latency_combine')

    def clean_low_latency_buffer(self, *_: Any, **__: Any) -> None:
        # Staged backend does not keep native low-latency device-side metadata.
        return None

    def low_latency_update_mask_buffer(self, rank_to_mask: int, mask: bool = False) -> None:
        if rank_to_mask < 0 or rank_to_mask >= self.num_ranks:
            raise ValueError('rank_to_mask out of range')
        self._low_latency_mask_status[rank_to_mask] = 1 if mask else 0

    def low_latency_query_mask_buffer(self, mask_status: torch.Tensor) -> None:
        if mask_status.numel() < self.num_ranks:
            raise ValueError('mask_status tensor is too small')
        mask_status.zero_()
        mask_status[:self.num_ranks].copy_(self._low_latency_mask_status.to(mask_status.device, dtype=mask_status.dtype))

    def low_latency_clean_mask_buffer(self) -> None:
        self._low_latency_mask_status.zero_()

    def get_next_low_latency_combine_buffer(
        self,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
    ) -> torch.Tensor:
        if num_max_dispatch_tokens_per_rank < 0 or hidden < 0 or num_experts < 0:
            raise ValueError('shape arguments must be non-negative')
        return torch.empty(
            (num_experts, self.num_ranks * num_max_dispatch_tokens_per_rank, hidden),
            dtype=torch.bfloat16,
            device=self._device,
        )


def is_sm90_compiled() -> bool:
    return False


def get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_ranks: int,
    num_experts: int,
) -> int:
    # Keep API available for migration-time integration.
    return num_max_dispatch_tokens_per_rank * hidden * 2
