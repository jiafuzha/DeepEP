"""Temporary Python stub for the XPU extension during staged migration."""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


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
        self._low_latency_combine_buffers: list[Optional[torch.Tensor]] = [None, None]
        self._next_low_latency_combine_buffer_slot = 0
        self._last_low_latency_combine_buffer: Optional[torch.Tensor] = None

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
        return bytes(b'DEEPEP_XPU_STAGED_UID_000000000000')

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
        usable_bytes -= usable_bytes % elem_size
        if usable_bytes == 0:
            return torch.empty((0,), dtype=dtype, device=self._device)
        sliced = source.narrow(0, byte_offset, usable_bytes)
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
        self._low_latency_combine_buffers = [None, None]
        self._last_low_latency_combine_buffer = None

    @staticmethod
    def _apply_bias(x: torch.Tensor,
                    bias_0: Optional[torch.Tensor],
                    bias_1: Optional[torch.Tensor]) -> torch.Tensor:
        out = x
        if bias_0 is not None:
            out = out + bias_0.to(device=out.device, dtype=out.dtype)
        if bias_1 is not None:
            out = out + bias_1.to(device=out.device, dtype=out.dtype)
        return out

    def get_dispatch_layout(self,
                            topk_idx: torch.Tensor,
                            num_experts: int,
                            previous_event: Any = None,
                            async_finish: bool = False,
                            allocate_on_comm_stream: bool = False) -> Any:
        del previous_event, async_finish, allocate_on_comm_stream
        if topk_idx.dim() != 2:
            raise ValueError('topk_idx must be 2D')
        if num_experts <= 0:
            raise ValueError('num_experts must be positive')
        if self.num_ranks <= 0:
            raise ValueError('num_ranks must be positive')
        if num_experts % self.num_ranks != 0:
            raise ValueError('num_experts must be divisible by num_ranks')

        num_tokens, _ = topk_idx.shape
        experts_per_rank = num_experts // self.num_ranks
        topk_idx_i64 = topk_idx.to(torch.int64)
        valid = (topk_idx_i64 >= 0) & (topk_idx_i64 < num_experts)
        safe_topk_idx = torch.where(valid, topk_idx_i64, torch.zeros_like(topk_idx_i64))

        num_tokens_per_expert = torch.bincount(safe_topk_idx[valid], minlength=num_experts).to(dtype=torch.int)

        rank_idx = safe_topk_idx // experts_per_rank
        rank_one_hot = F.one_hot(rank_idx, num_classes=self.num_ranks).to(torch.bool)
        is_token_in_rank = (rank_one_hot & valid.unsqueeze(-1)).any(dim=1)
        num_tokens_per_rank = is_token_in_rank.sum(dim=0, dtype=torch.int)

        num_tokens_per_rdma_rank = None
        if self._num_rdma_ranks > 1:
            if self.num_ranks % self._num_rdma_ranks != 0:
                raise ValueError('num_ranks must be divisible by num_rdma_ranks')
            num_local_ranks = self.num_ranks // self._num_rdma_ranks
            token_in_rdma_rank = is_token_in_rank.view(num_tokens, self._num_rdma_ranks, num_local_ranks).any(dim=2)
            num_tokens_per_rdma_rank = token_in_rdma_rank.sum(dim=0, dtype=torch.int)

        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, None

    def intranode_dispatch(self,
                           x: torch.Tensor,
                           x_scales: Optional[torch.Tensor],
                           topk_idx: Optional[torch.Tensor],
                           topk_weights: Optional[torch.Tensor],
                           num_tokens_per_rank: Optional[torch.Tensor],
                           is_token_in_rank: Optional[torch.Tensor],
                           num_tokens_per_expert: Optional[torch.Tensor],
                           num_recv_tokens: int,
                           rank_prefix_matrix: Optional[torch.Tensor],
                           channel_prefix_matrix: Optional[torch.Tensor],
                           expert_alignment: int,
                           num_worst_tokens: int,
                           config: Config,
                           previous_event: Any = None,
                           async_finish: bool = False,
                           allocate_on_comm_stream: bool = False) -> Any:
        del expert_alignment, num_worst_tokens, config, previous_event, async_finish, allocate_on_comm_stream
        recv_tokens = int(num_recv_tokens) if num_recv_tokens > 0 else int(x.size(0))

        recv_x = x[:recv_tokens]
        recv_x_scales = x_scales[:recv_tokens] if x_scales is not None else None
        recv_topk_idx = topk_idx[:recv_tokens] if topk_idx is not None else None
        recv_topk_weights = topk_weights[:recv_tokens] if topk_weights is not None else None

        if num_tokens_per_expert is not None:
            num_recv_tokens_per_expert_list = num_tokens_per_expert.to('cpu', dtype=torch.int).tolist()
        else:
            num_recv_tokens_per_expert_list = []

        if rank_prefix_matrix is None:
            rank_prefix_matrix = torch.zeros((self.num_ranks, self.num_ranks), dtype=torch.int, device=x.device)
        if channel_prefix_matrix is None:
            channel_prefix_matrix = torch.zeros((self.num_ranks, self.num_ranks), dtype=torch.int, device=x.device)
        recv_channel_prefix_matrix = torch.zeros_like(channel_prefix_matrix)
        recv_src_idx = torch.arange(recv_tokens, dtype=torch.int, device=x.device)
        send_head = torch.zeros((max(1, self.num_ranks),), dtype=torch.int, device=x.device)

        return (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
                num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix,
                recv_channel_prefix_matrix, recv_src_idx, send_head, None)

    def intranode_combine(self,
                          x: torch.Tensor,
                          topk_weights: Optional[torch.Tensor],
                          bias_0: Optional[torch.Tensor],
                          bias_1: Optional[torch.Tensor],
                          src_idx: torch.Tensor,
                          rank_prefix_matrix: torch.Tensor,
                          channel_prefix_matrix: torch.Tensor,
                          send_head: torch.Tensor,
                          config: Config,
                          previous_event: Any = None,
                          async_finish: bool = False,
                          allocate_on_comm_stream: bool = False) -> Any:
        del src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head, config
        del previous_event, async_finish, allocate_on_comm_stream
        recv_x = self._apply_bias(x, bias_0, bias_1)
        return recv_x, topk_weights, None

    def internode_dispatch(self,
                           x: torch.Tensor,
                           x_scales: Optional[torch.Tensor],
                           topk_idx: Optional[torch.Tensor],
                           topk_weights: Optional[torch.Tensor],
                           num_tokens_per_rank: Optional[torch.Tensor],
                           num_tokens_per_rdma_rank: Optional[torch.Tensor],
                           is_token_in_rank: Optional[torch.Tensor],
                           num_tokens_per_expert: Optional[torch.Tensor],
                           num_recv_tokens: int,
                           num_rdma_recv_tokens: int,
                           rdma_channel_prefix_matrix: Optional[torch.Tensor],
                           recv_rdma_rank_prefix_sum: Optional[torch.Tensor],
                           gbl_channel_prefix_matrix: Optional[torch.Tensor],
                           recv_gbl_rank_prefix_sum: Optional[torch.Tensor],
                           expert_alignment: int,
                           num_worst_tokens: int,
                           config: Config,
                           previous_event: Any = None,
                           async_finish: bool = False,
                           allocate_on_comm_stream: bool = False) -> Any:
        del num_tokens_per_rank, num_tokens_per_rdma_rank, expert_alignment, num_worst_tokens, config
        del previous_event, async_finish, allocate_on_comm_stream
        recv_tokens = int(num_recv_tokens) if num_recv_tokens > 0 else int(x.size(0))

        recv_x = x[:recv_tokens]
        recv_x_scales = x_scales[:recv_tokens] if x_scales is not None else None
        recv_topk_idx = topk_idx[:recv_tokens] if topk_idx is not None else None
        recv_topk_weights = topk_weights[:recv_tokens] if topk_weights is not None else None

        if num_tokens_per_expert is not None:
            num_recv_tokens_per_expert_list = num_tokens_per_expert.to('cpu', dtype=torch.int).tolist()
        else:
            num_recv_tokens_per_expert_list = []

        if is_token_in_rank is None:
            is_token_in_rank = torch.zeros((recv_tokens, self.num_ranks), dtype=torch.bool, device=x.device)
            if self.num_ranks > 0:
                is_token_in_rank[:, 0] = True

        if rdma_channel_prefix_matrix is None:
            rdma_channel_prefix_matrix = torch.zeros((self._num_rdma_ranks, self._num_rdma_ranks),
                                                     dtype=torch.int,
                                                     device=x.device)
        if gbl_channel_prefix_matrix is None:
            gbl_channel_prefix_matrix = torch.zeros((self.num_ranks, self.num_ranks), dtype=torch.int, device=x.device)
        recv_rdma_channel_prefix_matrix = torch.zeros_like(rdma_channel_prefix_matrix)
        recv_gbl_channel_prefix_matrix = torch.zeros_like(gbl_channel_prefix_matrix)

        if recv_rdma_rank_prefix_sum is None:
            recv_rdma_rank_prefix_sum = torch.zeros((self._num_rdma_ranks,), dtype=torch.int, device=x.device)
        if recv_gbl_rank_prefix_sum is None:
            recv_gbl_rank_prefix_sum = torch.zeros((self.num_ranks,), dtype=torch.int, device=x.device)

        recv_src_meta = torch.arange(recv_tokens, dtype=torch.int, device=x.device)
        rdma_tokens = int(num_rdma_recv_tokens) if num_rdma_recv_tokens > 0 else recv_tokens
        send_rdma_head = torch.zeros((max(1, self._num_rdma_ranks),), dtype=torch.int, device=x.device)
        send_nvl_head = torch.zeros((max(1, rdma_tokens),), dtype=torch.int, device=x.device)

        return (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,
                num_recv_tokens_per_expert_list, rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                recv_src_meta, send_rdma_head, send_nvl_head, None)

    def internode_combine(self,
                          x: torch.Tensor,
                          topk_weights: Optional[torch.Tensor],
                          bias_0: Optional[torch.Tensor],
                          bias_1: Optional[torch.Tensor],
                          src_meta: torch.Tensor,
                          is_combined_token_in_rank: torch.Tensor,
                          rdma_channel_prefix_matrix: torch.Tensor,
                          rdma_rank_prefix_sum: torch.Tensor,
                          gbl_channel_prefix_matrix: torch.Tensor,
                          send_rdma_head: torch.Tensor,
                          send_nvl_head: torch.Tensor,
                          config: Config,
                          previous_event: Any = None,
                          async_finish: bool = False,
                          allocate_on_comm_stream: bool = False) -> Any:
        del src_meta, is_combined_token_in_rank, rdma_channel_prefix_matrix, rdma_rank_prefix_sum
        del gbl_channel_prefix_matrix, send_rdma_head, send_nvl_head, config
        del previous_event, async_finish, allocate_on_comm_stream
        recv_x = self._apply_bias(x, bias_0, bias_1)
        return recv_x, topk_weights, None

    def low_latency_dispatch(self, *_: Any, **__: Any) -> Any:
        x = _[0]
        topk_idx = _[1]
        cumulative_local_expert_recv_stats = _[2]
        dispatch_wait_recv_cost_stats = _[3]
        num_max_dispatch_tokens_per_rank = int(_[4])
        num_experts = int(_[5])
        use_fp8 = bool(_[6])
        round_scale = bool(_[7])
        use_ue8m0 = bool(_[8])
        async_finish = bool(_[9])
        return_recv_hook = bool(_[10])
        del round_scale, async_finish

        if x.dim() != 2 or topk_idx.dim() != 2:
            raise ValueError('x and topk_idx must be 2D')
        if topk_idx.size(0) != x.size(0):
            raise ValueError('topk_idx must have the same number of tokens as x')
        if num_experts <= 0:
            raise ValueError('num_experts must be positive')
        if num_max_dispatch_tokens_per_rank <= 0:
            raise ValueError('num_max_dispatch_tokens_per_rank must be positive')

        num_tokens = int(x.size(0))
        hidden = int(x.size(1))
        num_topk = int(topk_idx.size(1))
        capacity = self.num_ranks * num_max_dispatch_tokens_per_rank

        recv_count = torch.zeros((num_experts,), dtype=torch.int, device=x.device)
        packed_recv_x = torch.zeros((num_experts, capacity, hidden), dtype=x.dtype, device=x.device)

        src_info = torch.full((num_experts, capacity, 2), -1, dtype=torch.int, device=x.device)
        layout_range = torch.zeros((num_experts, 2), dtype=torch.int, device=x.device)

        topk_idx_i64 = topk_idx.to(torch.int64)
        for token_idx in range(num_tokens):
            for topk_pos in range(num_topk):
                expert_idx = int(topk_idx_i64[token_idx, topk_pos].item())
                if expert_idx < 0 or expert_idx >= num_experts:
                    continue
                slot = int(recv_count[expert_idx].item())
                if slot >= capacity:
                    continue
                packed_recv_x[expert_idx, slot].copy_(x[token_idx])
                src_info[expert_idx, slot, 0] = token_idx
                src_info[expert_idx, slot, 1] = topk_pos
                recv_count[expert_idx] = slot + 1

        if cumulative_local_expert_recv_stats is not None:
            n = min(int(cumulative_local_expert_recv_stats.numel()), num_experts)
            cumulative_local_expert_recv_stats[:n] += recv_count[:n].to(cumulative_local_expert_recv_stats.dtype)

        if dispatch_wait_recv_cost_stats is not None and dispatch_wait_recv_cost_stats.numel() > 0:
            dispatch_wait_recv_cost_stats.zero_()

        layout_range[:, 1] = recv_count

        packed_recv_x_scales = None
        if use_fp8:
            # Keep staged behavior robust: keep BF16 payload and provide unit scales.
            scale_hidden = max(1, hidden // (512 if use_ue8m0 else 128))
            scale_dtype = torch.int if use_ue8m0 else torch.float
            packed_recv_x_scales = torch.ones((num_experts, capacity, scale_hidden), dtype=scale_dtype, device=x.device)

        hook = (lambda: None) if return_recv_hook else None
        return packed_recv_x, packed_recv_x_scales, recv_count, src_info, layout_range, None, hook

    def low_latency_combine(self, *_: Any, **__: Any) -> Any:
        x = _[0]
        topk_idx = _[1]
        topk_weights = _[2]
        src_info = _[3]
        layout_range = _[4]
        combine_wait_recv_cost_stats = _[5]
        num_max_dispatch_tokens_per_rank = int(_[6])
        num_experts = int(_[7])
        use_logfmt = bool(_[8])
        zero_copy = bool(_[9])
        async_finish = bool(_[10])
        return_recv_hook = bool(_[11])
        out = _[12]
        del num_max_dispatch_tokens_per_rank, num_experts, use_logfmt, async_finish

        if x.dim() != 3:
            raise ValueError('x must be 3D [num_local_experts, capacity, hidden]')
        if topk_idx.dim() != 2 or topk_weights.dim() != 2:
            raise ValueError('topk_idx and topk_weights must be 2D')
        if topk_idx.shape != topk_weights.shape:
            raise ValueError('topk_idx and topk_weights must have the same shape')
        if src_info.dim() != 3 or src_info.size(-1) != 2:
            raise ValueError('src_info must have shape [num_local_experts, capacity, 2]')
        if layout_range.dim() != 2 or layout_range.size(-1) != 2:
            raise ValueError('layout_range must have shape [num_local_experts, 2]')

        combine_source = x
        if zero_copy:
            if self._last_low_latency_combine_buffer is None:
                raise ValueError('zero_copy=True requires get_next_low_latency_combine_buffer to be called first')
            combine_source = self._last_low_latency_combine_buffer
            if combine_source.dim() != 3:
                raise ValueError('staged zero-copy combine buffer must be 3D')

        num_combined_tokens = int(topk_idx.size(0))
        hidden = int(combine_source.size(2))
        combined_x = out if out is not None else torch.zeros((num_combined_tokens, hidden), dtype=combine_source.dtype, device=combine_source.device)
        if out is not None:
            combined_x.zero_()

        num_local_experts = int(combine_source.size(0))
        capacity = int(combine_source.size(1))
        for expert_idx in range(num_local_experts):
            end = int(layout_range[expert_idx, 1].item())
            end = max(0, min(end, capacity))
            for slot in range(end):
                src_token = int(src_info[expert_idx, slot, 0].item())
                src_topk_pos = int(src_info[expert_idx, slot, 1].item())
                if src_token < 0 or src_token >= num_combined_tokens:
                    continue
                if src_topk_pos < 0 or src_topk_pos >= int(topk_idx.size(1)):
                    continue
                combined_x[src_token] += combine_source[expert_idx, slot] * topk_weights[src_token, src_topk_pos]

        if combine_wait_recv_cost_stats is not None and combine_wait_recv_cost_stats.numel() > 0:
            combine_wait_recv_cost_stats.zero_()

        hook = (lambda: None) if return_recv_hook else None
        return combined_x, None, hook

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
        slot = self._next_low_latency_combine_buffer_slot
        expected_shape = (num_experts, self.num_ranks * num_max_dispatch_tokens_per_rank, hidden)
        buffer = self._low_latency_combine_buffers[slot]
        if buffer is None or tuple(buffer.shape) != expected_shape or buffer.device != self._device:
            buffer = torch.empty(expected_shape, dtype=torch.bfloat16, device=self._device)
            self._low_latency_combine_buffers[slot] = buffer

        self._last_low_latency_combine_buffer = buffer
        self._next_low_latency_combine_buffer_slot = (slot + 1) % len(self._low_latency_combine_buffers)
        return buffer


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
