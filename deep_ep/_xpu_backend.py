import torch
import torch.distributed as dist

from .utils import EventOverlap

SUPPORTS_INTRANODE = True
SUPPORTS_INTERNODE = False
SUPPORTS_LOW_LATENCY = True


def _to_cpu_object(tensor):
    return None if tensor is None else tensor.detach().cpu()


def _to_device_tensor(value, device):
    return None if value is None else value.to(device)


class XpuIntranodeBuffer:

    def __init__(self, parent, group, low_latency_mode: bool) -> None:
        self.parent = parent
        self.group = group
        self.rank = parent.rank
        self.group_size = parent.group_size
        self.device = torch.device(f"xpu:{torch.xpu.current_device()}")
        self.low_latency_mode = low_latency_mode
        self._low_latency_mask = torch.zeros((self.group_size,), dtype=torch.int32, device=self.device)
        self._next_combine_buffer_idx = 0
        self._combine_buffers = {}

    def destroy(self) -> None:
        self._combine_buffers.clear()
        return

    def get_comm_stream(self):
        return torch.xpu.current_stream()

    def get_local_buffer_tensor(self, *_args, **_kwargs):
        raise NotImplementedError("Raw buffer access is not available on the Python XPU backend")

    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int):
        num_tokens = topk_idx.size(0)
        experts_per_rank = num_experts // self.group_size
        valid = topk_idx.ge(0)

        num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device=topk_idx.device)
        if valid.any():
            flat = topk_idx[valid].to(torch.int64)
            num_tokens_per_expert.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int, device=topk_idx.device))

        is_token_in_rank = torch.zeros((num_tokens, self.group_size), dtype=torch.bool, device=topk_idx.device)
        if valid.any():
            rank_idx = torch.where(valid, topk_idx // experts_per_rank, torch.full_like(topk_idx, -1))
            for rank in range(self.group_size):
                is_token_in_rank[:, rank] = rank_idx.eq(rank).any(dim=1)

        num_tokens_per_rank = is_token_in_rank.sum(dim=0, dtype=torch.int)
        return num_tokens_per_rank, None, num_tokens_per_expert, is_token_in_rank, EventOverlap()

    def _all_gather_fixed(self, tensor: torch.Tensor):
        gathered = [None for _ in range(self.group_size)]
        dist.all_gather_object(gathered, tensor.detach().cpu(), group=self.group)
        return [item.to(tensor.device) for item in gathered]

    def _all_gather_object(self, obj):
        gathered = [None for _ in range(self.group_size)]
        dist.all_gather_object(gathered, obj, group=self.group)
        return gathered

    def _build_rank_prefix_matrix_from_masks(self, all_is_token_in_rank, device: torch.device):
        counts = torch.zeros((self.group_size, self.group_size), dtype=torch.int, device=device)
        for src_rank, src_mask in enumerate(all_is_token_in_rank):
            for dst_rank in range(self.group_size):
                counts[src_rank, dst_rank] = src_mask[:, dst_rank].sum()
        return counts.cumsum(dim=0), counts

    def _select_tokens_for_rank(self, src_rank: int, dst_rank: int, src_x: torch.Tensor, src_topk_idx: torch.Tensor,
                                src_topk_weights: torch.Tensor, num_experts: int):
        experts_per_rank = num_experts // self.group_size
        valid = src_topk_idx.ge(0)
        src_rank_mask = torch.where(valid, src_topk_idx // experts_per_rank, torch.full_like(src_topk_idx, -1)).eq(dst_rank)
        token_mask = src_rank_mask.any(dim=1)
        token_indices = token_mask.nonzero(as_tuple=False).squeeze(-1)
        recv_x = src_x.index_select(0, token_indices) if token_indices.numel() > 0 else src_x[:0]

        recv_topk_idx = None
        recv_topk_weights = None
        if src_topk_weights is not None:
            recv_topk_idx = torch.full((token_indices.numel(), src_topk_idx.size(1)), -1, dtype=src_topk_idx.dtype, device=src_topk_idx.device)
            recv_topk_weights = torch.zeros((token_indices.numel(), src_topk_weights.size(1)), dtype=src_topk_weights.dtype, device=src_topk_weights.device)
            if token_indices.numel() > 0:
                selected_mask = src_rank_mask.index_select(0, token_indices)
                selected_topk = src_topk_idx.index_select(0, token_indices)
                selected_weights = src_topk_weights.index_select(0, token_indices)
                recv_topk_idx[selected_mask] = (selected_topk[selected_mask] % experts_per_rank).to(src_topk_idx.dtype)
                recv_topk_weights[selected_mask] = selected_weights[selected_mask]

        return token_indices, recv_x, recv_topk_idx, recv_topk_weights

    def dispatch(self,
                 x: torch.Tensor,
                 handle=None,
                 num_tokens_per_rank=None,
                 is_token_in_rank=None,
                 num_tokens_per_expert=None,
                 topk_idx=None,
                 topk_weights=None,
                 num_worst_tokens: int = 0):
        if handle is not None:
            all_x = self._all_gather_fixed(x)
            rank_prefix_matrix, _, _, recv_src_idx, _, _, counts_by_src = handle
            recv_parts = []
            offset = 0
            for src_rank, src_count in enumerate(counts_by_src):
                src_count = int(src_count)
                src_indices = recv_src_idx[offset:offset + src_count]
                recv_parts.append(all_x[src_rank].index_select(0, src_indices))
                offset += src_count
            recv_x = torch.cat(recv_parts, dim=0) if recv_parts else x[:0]
            return recv_x, None, None, None, None, EventOverlap()

        assert num_tokens_per_expert is not None and is_token_in_rank is not None
        all_x = self._all_gather_fixed(x)
        all_is_token_in_rank = self._all_gather_fixed(is_token_in_rank)
        all_num_tokens_per_expert = self._all_gather_fixed(num_tokens_per_expert)
        all_topk_idx = self._all_gather_fixed(topk_idx) if topk_idx is not None else [None] * self.group_size
        all_topk_weights = self._all_gather_fixed(topk_weights) if topk_weights is not None else [None] * self.group_size
        rank_prefix_matrix, counts = self._build_rank_prefix_matrix_from_masks(all_is_token_in_rank, x.device)

        recv_x_parts, recv_idx_parts, recv_weight_parts = [], [], []
        recv_src_idx_parts = []
        for src_rank in range(self.group_size):
            if all_topk_idx[src_rank] is None:
                token_indices = all_is_token_in_rank[src_rank][:, self.rank].nonzero(as_tuple=False).squeeze(-1)
                recv_x_part = all_x[src_rank].index_select(0, token_indices) if token_indices.numel() > 0 else all_x[src_rank][:0]
                recv_idx_part = None
                recv_weight_part = None
            else:
                token_indices, recv_x_part, recv_idx_part, recv_weight_part = self._select_tokens_for_rank(
                    src_rank, self.rank, all_x[src_rank], all_topk_idx[src_rank], all_topk_weights[src_rank], num_tokens_per_expert.numel())
            recv_src_idx_parts.append(token_indices.to(torch.int64))
            recv_x_parts.append(recv_x_part)
            if recv_idx_part is not None:
                recv_idx_parts.append(recv_idx_part)
                recv_weight_parts.append(recv_weight_part)

        recv_x = torch.cat(recv_x_parts, dim=0) if recv_x_parts else x[:0]
        recv_src_idx = torch.cat(recv_src_idx_parts, dim=0) if recv_src_idx_parts else torch.empty((0,), dtype=torch.int64, device=x.device)
        recv_topk_idx = torch.cat(recv_idx_parts, dim=0) if recv_idx_parts else None
        recv_topk_weights = torch.cat(recv_weight_parts, dim=0) if recv_weight_parts else None

        local_experts = num_tokens_per_expert.numel() // self.group_size
        global_num_tokens_per_expert = torch.stack(all_num_tokens_per_expert, dim=0).sum(dim=0)
        recv_num_tokens_per_expert_list = global_num_tokens_per_expert.view(self.group_size, local_experts)[self.rank].tolist()

        counts_by_src = counts[:, self.rank].to(torch.int64)
        handle = (rank_prefix_matrix, None, None, recv_src_idx, is_token_in_rank, None, counts_by_src)

        if num_worst_tokens > 0:
            padding = max(0, num_worst_tokens - recv_x.size(0))
            if padding > 0:
                recv_x = torch.cat((recv_x, torch.zeros((padding, *recv_x.shape[1:]), dtype=recv_x.dtype, device=recv_x.device)), dim=0)
                if recv_topk_idx is not None:
                    recv_topk_idx = torch.cat((recv_topk_idx,
                                               torch.full((padding, recv_topk_idx.size(1)), -1, dtype=recv_topk_idx.dtype, device=recv_topk_idx.device)),
                                              dim=0)
                    recv_topk_weights = torch.cat((recv_topk_weights,
                                                   torch.zeros((padding, recv_topk_weights.size(1)),
                                                               dtype=recv_topk_weights.dtype, device=recv_topk_weights.device)),
                                                  dim=0)
            recv_num_tokens_per_expert_list = []

        return recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, EventOverlap()

    def combine(self, x: torch.Tensor, handle, topk_weights: torch.Tensor = None, bias=None):
        rank_prefix_matrix, _, _, recv_src_idx, _, _, counts_by_src = handle
        payload = {
            'x': _to_cpu_object(x),
            'topk_weights': _to_cpu_object(topk_weights),
            'recv_src_idx': _to_cpu_object(recv_src_idx),
            'counts_by_src': [int(v) for v in counts_by_src.tolist()],
        }
        gathered = self._all_gather_object(payload)

        num_tokens = int(recv_src_idx.max().item() + 1) if recv_src_idx.numel() > 0 else 0
        for item in gathered:
            if item['recv_src_idx'] is not None and item['recv_src_idx'].numel() > 0:
                num_tokens = max(num_tokens, int(item['recv_src_idx'].max().item() + 1))

        combined_x = torch.zeros((num_tokens, x.size(1)), dtype=x.dtype, device=x.device)
        combined_topk_weights = None if topk_weights is None else torch.zeros((num_tokens, topk_weights.size(1)),
                                                                              dtype=topk_weights.dtype, device=x.device)

        for item in gathered:
            src_counts = item['counts_by_src']
            recv_src_idx_item = _to_device_tensor(item['recv_src_idx'], x.device)
            recv_x_item = _to_device_tensor(item['x'], x.device)
            recv_topk_weights_item = _to_device_tensor(item['topk_weights'], x.device)

            begin = sum(src_counts[:self.rank])
            end = begin + src_counts[self.rank]
            if begin == end:
                continue
            src_indices = recv_src_idx_item[begin:end]
            combined_x.index_add_(0, src_indices, recv_x_item[begin:end])
            if combined_topk_weights is not None:
                combined_topk_weights.index_add_(0, src_indices, recv_topk_weights_item[begin:end])

        if bias is not None:
            bias_0, bias_1 = self.parent._unpack_bias(bias)
            if bias_0 is not None:
                combined_x = combined_x + bias_0
            if bias_1 is not None:
                combined_x = combined_x + bias_1

        return combined_x, combined_topk_weights, EventOverlap()

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        for idx in range(2):
            key = (idx, num_max_dispatch_tokens_per_rank, hidden, num_experts)
            if key in self._combine_buffers:
                self._combine_buffers[key].zero_()

    def low_latency_update_mask_buffer(self, rank_to_mask: int, mask: bool = False) -> None:
        self._low_latency_mask[rank_to_mask] = int(mask)

    def low_latency_query_mask_buffer(self, mask_status: torch.Tensor):
        mask_status.copy_(self._low_latency_mask.to(mask_status.dtype))

    def low_latency_clean_mask_buffer(self):
        self._low_latency_mask.zero_()

    def _per_token_cast_to_fp8(self, x: torch.Tensor, round_scale: bool):
        num_tokens, hidden = x.shape
        x_view = x.to(torch.float32).view(num_tokens, hidden // 128, 128)
        amax = x_view.abs().amax(dim=-1).clamp(1e-4)
        if round_scale:
            scale_inv = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
        else:
            scale_inv = amax / 448.0
        scale = torch.reciprocal(scale_inv)
        x_fp8 = (x_view * scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(num_tokens, hidden).contiguous()
        return x_fp8, scale_inv.contiguous()

    def _build_low_latency_dispatch(self,
                                    x: torch.Tensor,
                                    topk_idx: torch.Tensor,
                                    num_max_dispatch_tokens_per_rank: int,
                                    num_experts: int,
                                    cumulative_local_expert_recv_stats: torch.Tensor | None,
                                    use_fp8: bool,
                                    round_scale: bool):
        num_tokens, hidden = x.shape
        num_local_experts = num_experts // self.group_size
        capacity = self.group_size * num_max_dispatch_tokens_per_rank
        all_x = self._all_gather_fixed(x)
        all_topk_idx = self._all_gather_fixed(topk_idx)

        packed_recv_x_bf16 = torch.zeros((num_local_experts, capacity, hidden), dtype=torch.bfloat16, device=self.device)
        packed_recv_src_info = torch.full((num_local_experts, capacity), -1, dtype=torch.int32, device=self.device)
        packed_recv_layout_range = torch.zeros((num_local_experts, self.group_size), dtype=torch.int64, device=self.device)
        packed_recv_count = torch.zeros((num_local_experts,), dtype=torch.int32, device=self.device)

        for local_expert_idx in range(num_local_experts):
            expert_id = self.rank * num_local_experts + local_expert_idx
            write_offset = 0
            for src_rank in range(self.group_size):
                if self._low_latency_mask[src_rank].item() != 0:
                    continue
                selected = (all_topk_idx[src_rank] == expert_id).any(dim=1)
                token_indices = selected.nonzero(as_tuple=False).squeeze(-1)
                count = int(token_indices.numel())
                packed_recv_layout_range[local_expert_idx, src_rank] = (int(write_offset) << 32) | count
                if count == 0:
                    continue
                packed_recv_x_bf16[local_expert_idx, write_offset:write_offset + count] = all_x[src_rank].index_select(0, token_indices)
                packed_recv_src_info[local_expert_idx, write_offset:write_offset + count] = token_indices.to(torch.int32)
                write_offset += count
            packed_recv_count[local_expert_idx] = write_offset

        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.add_(packed_recv_count.to(cumulative_local_expert_recv_stats.dtype))

        if use_fp8:
            flat_fp8, flat_scales = self._per_token_cast_to_fp8(packed_recv_x_bf16.view(-1, hidden), round_scale)
            packed_recv_x = flat_fp8.view_as(packed_recv_x_bf16)
            packed_recv_x_scales = flat_scales.view(num_local_experts, capacity, hidden // 128)
        else:
            packed_recv_x = packed_recv_x_bf16
            packed_recv_x_scales = None

        handle = (packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts)
        return packed_recv_x, packed_recv_x_scales, packed_recv_count, handle

    def low_latency_dispatch(self,
                             x: torch.Tensor,
                             topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int,
                             num_experts: int,
                             cumulative_local_expert_recv_stats=None,
                             dispatch_wait_recv_cost_stats=None,
                             use_fp8: bool = True,
                             round_scale: bool = False,
                             use_ue8m0: bool = False,
                             async_finish: bool = False,
                             return_recv_hook: bool = False):
        del dispatch_wait_recv_cost_stats, use_ue8m0, async_finish

        packed_recv_x = None
        packed_recv_x_scales = None
        packed_recv_count = None
        handle = None

        def finalize():
            nonlocal packed_recv_x, packed_recv_x_scales, packed_recv_count, handle
            if packed_recv_x is None:
                packed_recv_x, packed_recv_x_scales, packed_recv_count, handle = self._build_low_latency_dispatch(
                    x,
                    topk_idx,
                    num_max_dispatch_tokens_per_rank,
                    num_experts,
                    cumulative_local_expert_recv_stats,
                    use_fp8,
                    round_scale)

        finalize()
        hook = finalize if return_recv_hook else (lambda: None)
        return packed_recv_x, packed_recv_x_scales, packed_recv_count, handle, EventOverlap(), hook

    def get_next_low_latency_combine_buffer(self, handle):
        _, _, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle[:5]
        num_local_experts = num_experts // self.group_size
        capacity = self.group_size * num_max_dispatch_tokens_per_rank
        key = (self._next_combine_buffer_idx, num_max_dispatch_tokens_per_rank, hidden, num_experts)
        buffer = self._combine_buffers.get(key)
        if buffer is None:
            buffer = torch.empty((num_local_experts, capacity, hidden), dtype=torch.bfloat16, device=self.device)
            self._combine_buffers[key] = buffer
        self._next_combine_buffer_idx = (self._next_combine_buffer_idx + 1) % 2
        return buffer

    def low_latency_combine(self,
                            x: torch.Tensor,
                            topk_idx: torch.Tensor,
                            topk_weights: torch.Tensor,
                            handle: tuple,
                            use_logfmt: bool = False,
                            zero_copy: bool = False,
                            async_finish: bool = False,
                            return_recv_hook: bool = False,
                            out: torch.Tensor | None = None,
                            combine_wait_recv_cost_stats: torch.Tensor | None = None):
        del use_logfmt, zero_copy, async_finish, combine_wait_recv_cost_stats
        packed_recv_src_info, packed_recv_layout_range, _, _, num_experts = handle[:5]
        num_local_experts = num_experts // self.group_size
        payload = {
            "x": _to_cpu_object(x),
            "src_info": _to_cpu_object(packed_recv_src_info),
            "layout_range": _to_cpu_object(packed_recv_layout_range),
        }
        gathered = self._all_gather_object(payload)

        combined_x = torch.zeros_like(topk_weights[:, :1], dtype=x.dtype).expand(topk_idx.size(0), x.size(-1)).clone() if out is None else out
        combined_x.zero_()

        for owner_rank, item in enumerate(gathered):
            owner_x = _to_device_tensor(item["x"], self.device)
            owner_src_info = _to_device_tensor(item["src_info"], self.device)
            owner_layout_range = _to_device_tensor(item["layout_range"], self.device)
            for local_expert_idx in range(num_local_experts):
                expert_id = owner_rank * num_local_experts + local_expert_idx
                layout_entry = int(owner_layout_range[local_expert_idx, self.rank].item())
                begin_idx = layout_entry >> 32
                count = layout_entry & ((1 << 32) - 1)
                if count == 0:
                    continue
                token_indices = owner_src_info[local_expert_idx, begin_idx:begin_idx + count].to(torch.long)
                weight_mask = topk_idx.index_select(0, token_indices).eq(expert_id)
                weights = topk_weights.index_select(0, token_indices).masked_fill(~weight_mask, 0).sum(dim=1).to(x.dtype)
                combined_x.index_add_(0, token_indices, owner_x[local_expert_idx, begin_idx:begin_idx + count] * weights.unsqueeze(1))

        hook = (lambda: None)
        return combined_x, EventOverlap(), hook
