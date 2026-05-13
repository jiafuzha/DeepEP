import torch
import torch.distributed as dist

from .utils import EventOverlap


def _to_cpu_object(tensor):
    return None if tensor is None else tensor.detach().cpu()


def _to_device_tensor(value, device):
    return None if value is None else value.to(device)


class XpuIntranodeBuffer:

    def __init__(self, parent, group, low_latency_mode: bool) -> None:
        if low_latency_mode:
            raise NotImplementedError("Low-latency mode is not yet implemented on the Python XPU backend")
        self.parent = parent
        self.group = group
        self.rank = parent.rank
        self.group_size = parent.group_size
        self.device = torch.device(f"xpu:{torch.xpu.current_device()}")

    def destroy(self) -> None:
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
