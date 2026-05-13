import os
import math
import torch
import torch.distributed as dist
from typing import Callable, List, Tuple, Optional, Union

from .utils import EventOverlap, check_nvlink_connections

_BACKEND_IMPORT_ERROR: Optional[Exception] = None

try:
    # noinspection PyUnresolvedReferences
    import deep_ep_xpu_cpp
    # noinspection PyUnresolvedReferences
    from deep_ep_xpu_cpp import Config, EventHandle
except Exception as exc:  # pragma: no cover - exercised when the native extension is unavailable
    deep_ep_xpu_cpp = None  # type: ignore[assignment]
    _BACKEND_IMPORT_ERROR = exc

    class Config:  # type: ignore[no-redef]

        def __init__(self, *args, **kwargs):
            _require_backend()

    EventHandle = None  # type: ignore[assignment]


def _require_backend() -> None:
    if deep_ep_xpu_cpp is None:
        raise ModuleNotFoundError(
            "deep_ep_xpu_cpp is not available. The Python XPU wrapper can be imported, "
            "but Buffer operations require the native extension to be built and importable.") from _BACKEND_IMPORT_ERROR


XPU_LOW_LATENCY_UNSUPPORTED_MESSAGE = (
    "DeepEP XPU low-latency kernels are intentionally unsupported: the mirrored API surface is kept for compatibility, "
    "but a portable SYCL/iSHMEM implementation is not available yet.")


def _raise_low_latency_unsupported(api: str) -> None:
    raise NotImplementedError(f'{api} is unavailable on XPU. {XPU_LOW_LATENCY_UNSUPPORTED_MESSAGE}')


class _ImmediateEvent:

    def current_stream_wait(self) -> None:
        return


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using the fast intra-node peer path)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA plus intra-node peer access)
        - a mirrored low-latency API surface that is explicit unsupported on XPU today

    Attributes:
        num_sms: legacy tuning knob name preserved for API compatibility on the XPU backend.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: legacy name for the intra-node peer-communication buffer size.
        num_rdma_bytes: the RDMA/fabric communication buffer size.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

    def __init__(self,
                 group: Optional[dist.ProcessGroup],
                 num_nvl_bytes: int = 0,
                 num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False,
                 num_qps_per_rank: int = 24,
                 allow_nvlink_for_low_latency_mode: bool = True,
                 allow_mnnvl: bool = False,
                 use_fabric: bool = False,
                 explicitly_destroy: bool = False,
                 enable_shrink: bool = False,
                 comm: Optional["mpi4py.MPI.Comm"] = None) -> None:  # noqa: F821
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: legacy name for the intra-node peer-communication buffer size.
            num_rdma_bytes: the RDMA/fabric communication buffer size.
            low_latency_mode: mirrored for API compatibility only; low-latency mode is not supported on XPU.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: legacy compatibility knob controlling whether the low-latency path may
                use the fast intra-node peer path in addition to RDMA. This can be incompatible with hook-based overlap.
            allow_mnnvl: legacy backend flag forwarded to the native runtime.
            use_fabric: whether to use fabric API for memory buffers.
            enable_shrink: whether to enable shrink mode. The enable mode allocates a mask buffer to support masking ranks dynamically.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            comm: the `mpi4py.MPI.Comm` communicator to use in case the group parameter is absent.
        """
        _require_backend()
        if low_latency_mode:
            _raise_low_latency_unsupported('Buffer(..., low_latency_mode=True)')
        check_nvlink_connections(group)

        # Initialize the CPP runtime
        if group is not None:
            self.rank = dist.get_rank(group)
            self.group = group
            self.group_size = dist.get_world_size(group)

            def all_gather_object(obj):
                object_list = [None] * self.group_size
                dist.all_gather_object(object_list, obj, group=group)
                return object_list
        elif comm is not None:
            self.rank = comm.Get_rank()
            self.group = comm
            self.group_size = comm.Get_size()

            def all_gather_object(obj):
                return comm.allgather(obj)
        else:
            raise ValueError("Either 'group' or 'comm' must be provided.")
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink
        self._all_gather_object = all_gather_object
        self.runtime = deep_ep_xpu_cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode,
                                              explicitly_destroy, enable_shrink, use_fabric)

        # Synchronize device IDs
        local_device_id = self.runtime.get_local_device_id()
        device_ids = all_gather_object(local_device_id)

        # Synchronize IPC handles
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        ipc_handles = all_gather_object(local_ipc_handle)

        # Synchronize the iSHMEM bootstrap identifier.
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            # Enable iSHMEM IBGDA.
            assert num_qps_per_rank > 0
            os.environ['ISHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'
            os.environ['ISHMEM_IB_ENABLE_IBGDA'] = '1'
            os.environ['ISHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'

            # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
            self.ishmem_qp_depth = int(os.environ.get('ISHMEM_QP_DEPTH', '1024'))
            os.environ['ISHMEM_QP_DEPTH'] = str(self.ishmem_qp_depth)

            # Reduce XPU memory usage.
            # 6 default teams + 1 extra team
            os.environ['ISHMEM_MAX_TEAMS'] = '7'
            os.environ['ISHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'

            if not allow_mnnvl:
                os.environ['ISHMEM_DISABLE_MNNVL'] = '1'

            # Synchronize using the root bootstrap ID. The runtime still exposes
            # the legacy method name for this value.
            if (low_latency_mode and self.rank == 0) or (not low_latency_mode and self.runtime.get_rdma_rank() == 0):
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            ishmem_unique_ids = all_gather_object(root_unique_id)
            root_unique_id = ishmem_unique_ids[0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)]

        # Make CPP runtime available
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert self.runtime.is_available()
        self._use_intranode_collective_fallback = (self.runtime.get_num_rdma_ranks() == 1 and self.group_size > 1
                                                   and os.environ.get('DEEP_EP_XPU_USE_COLLECTIVE_FALLBACK', '0') == '1')
        if self._use_intranode_collective_fallback and self.rank == 0:
            print('[DeepEP XPU] Using collective intranode fallback instead of the experimental peer-buffer runtime.', flush=True)

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """

        assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'
        if self.runtime is None:
            return

        if self._use_intranode_collective_fallback:
            self.runtime = None
            return

        self.runtime.destroy()
        self.runtime = None

    @staticmethod
    def is_sm90_compiled():
        return getattr(deep_ep_xpu_cpp, 'is_sm90_compiled', lambda: False)()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the legacy kernel-width tuning knob used by the high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, '`num_sms` must be even'
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture an XPU event on the current stream, i.e. `torch.xpu.current_stream()`.

        Returns:
            event: the captured event.
        """
        _require_backend()
        return EventOverlap(EventHandle())

    @staticmethod
    def supports_intranode_autotune() -> bool:
        """
        Whether the current XPU intranode path provides meaningful throughput auto-tuning data.

        The native XPU path now includes a cached no-topk fast path, but it still uses
        host-staged Level Zero IPC over the active PCIe transport rather than the original
        CUDA-style steady-state kernel path. Keep the CUDA auto-tuning loop disabled until
        the XPU transport/kernel configuration itself becomes the thing being tuned.
        """
        return False

    @staticmethod
    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int) -> int:
        """
        Unsupported on XPU; retained only so callers fail explicitly instead of assuming low-latency support exists.
        """
        _raise_low_latency_unsupported('Buffer.get_low_latency_rdma_size_hint')

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        stream: torch.Stream = self.runtime.get_comm_stream()
        if getattr(stream, 'device_type', None) == 'xpu':
            return stream
        return torch.xpu.Stream(stream_id=stream.stream_id, device_index=stream.device_index, device_type=stream.device_type)

    def get_local_buffer_tensor(self,
                                dtype: torch.dtype,
                                size: Optional[torch.Size] = None,
                                offset: int = 0,
                                use_rdma_buffer: bool = False) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        tensor = self.runtime.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)
        if size is None:
            return tensor

        assert tensor.numel() >= size.numel()
        return tensor[:size.numel()].view(size)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 36, 288, 20, 128),
            24: Config(Buffer.num_sms, 32, 288, 8, 128),
            32: Config(Buffer.num_sms, 32, 288, 8, 128),
            48: Config(Buffer.num_sms, 32, 288, 8, 128),
            64: Config(Buffer.num_sms, 32, 288, 8, 128),
            96: Config(Buffer.num_sms, 20, 480, 12, 128),
            128: Config(Buffer.num_sms, 20, 560, 12, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 4, 288, 12, 128),
            24: Config(Buffer.num_sms, 1, 288, 8, 128),
            32: Config(Buffer.num_sms, 1, 288, 8, 128),
            48: Config(Buffer.num_sms, 1, 288, 8, 128),
            64: Config(Buffer.num_sms, 1, 288, 8, 128),
            96: Config(Buffer.num_sms, 1, 480, 8, 128),
            128: Config(Buffer.num_sms, 1, 560, 8, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int,
                            previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                            allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same local device index across nodes), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        if previous_event is not None:
            previous_event.current_stream_wait()

        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank = \
            self._get_dispatch_layout_fallback(topk_idx, num_experts)
        event = _ImmediateEvent() if async_finish else None
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, EventOverlap(event)

    def _get_dispatch_layout_fallback(self, topk_idx: torch.Tensor, num_experts: int) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        assert topk_idx.dim() == 2 and topk_idx.is_contiguous()
        assert num_experts > 0 and num_experts % self.group_size == 0

        device = topk_idx.device
        num_tokens = topk_idx.size(0)
        num_experts_per_rank = num_experts // self.group_size
        valid_topk = topk_idx >= 0

        rank_idx = torch.full_like(topk_idx, -1, dtype=torch.int64)
        rank_idx[valid_topk] = topk_idx[valid_topk].to(torch.int64) // num_experts_per_rank

        is_token_in_rank = torch.zeros((num_tokens, self.group_size), dtype=torch.bool, device=device)
        for topk_slot in range(topk_idx.size(1)):
            selected = rank_idx[:, topk_slot]
            valid_selected = selected >= 0
            if valid_selected.any():
                is_token_in_rank[valid_selected, selected[valid_selected]] = True
        is_token_in_rank = is_token_in_rank.contiguous()

        num_tokens_per_rank = is_token_in_rank.to(torch.int32).sum(dim=0).to(torch.int32).contiguous()
        num_tokens_per_expert = torch.bincount(
            topk_idx[valid_topk].to(torch.int64),
            minlength=num_experts,
        ).to(torch.int32).contiguous()

        num_tokens_per_rdma_rank = None
        num_rdma_ranks = self.runtime.get_num_rdma_ranks()
        if num_rdma_ranks > 1:
            assert self.group_size % num_rdma_ranks == 0
            num_local_ranks = self.group_size // num_rdma_ranks
            rdma_rank_idx = torch.full_like(rank_idx, -1)
            valid_rank_idx = rank_idx >= 0
            rdma_rank_idx[valid_rank_idx] = rank_idx[valid_rank_idx] // num_local_ranks

            is_token_in_rdma_rank = torch.zeros((num_tokens, num_rdma_ranks), dtype=torch.bool, device=device)
            for topk_slot in range(topk_idx.size(1)):
                selected = rdma_rank_idx[:, topk_slot]
                valid_selected = selected >= 0
                if valid_selected.any():
                    is_token_in_rdma_rank[valid_selected, selected[valid_selected]] = True
            num_tokens_per_rdma_rank = is_token_in_rdma_rank.to(torch.int32).sum(dim=0).to(torch.int32).contiguous()

        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank

    # noinspection PyTypeChecker
    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 handle: Optional[Tuple] = None,
                 num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                 is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                 topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None,
                 expert_alignment: int = 1, num_worst_tokens: int = 0,
                 config: Optional[Config] = None,
                 previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                 allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
                  Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require the ranks to have fast peer connectivity within a node.
        Internode kernels additionally require RDMA connectivity across nodes for matching local device indices.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the host will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same local device index across nodes), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert indices
                selected by each token, `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no host sync, and it
                will be graph-capture friendly. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_dispatch(x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank,
                                           num_tokens_per_expert, topk_idx, topk_weights, expert_alignment, num_worst_tokens, config,
                                           previous_event, async_finish, allocate_on_comm_stream)
        if self._use_intranode_collective_fallback:
            return self._intranode_dispatch_fallback(x, handle, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, topk_idx,
                                                     topk_weights, num_worst_tokens, previous_event, async_finish)

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            if len(handle) == 6:
                rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
                send_pos = None
                row_src_rank = None
            else:
                rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head, send_pos, row_src_rank = handle
            num_recv_tokens = recv_src_idx.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, event = self.runtime.intranode_dispatch(
                x, x_scales, None, None, None, is_token_in_rank, None, num_recv_tokens, rank_prefix_matrix, channel_prefix_matrix, send_pos,
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            num_recv_tokens, rank_prefix_matrix, channel_prefix_matrix = self._get_intranode_dispatch_metadata(
                num_tokens_per_rank, is_token_in_rank,
                getattr(config, 'num_sms', Buffer.num_sms) // 2)
            send_pos = self._get_intranode_send_pos(is_token_in_rank)
            dist.barrier(group=self.group)
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event = \
                self.runtime.intranode_dispatch(x, x_scales, topk_idx, topk_weights,
                                                None, is_token_in_rank, num_tokens_per_expert, num_recv_tokens, rank_prefix_matrix,
                                                channel_prefix_matrix, send_pos,
                                                expert_alignment, num_worst_tokens, config,
                                                getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            recv_src_idx, _ = self._get_intranode_recv_src_idx(is_token_in_rank)
            row_src_rank = self._get_intranode_row_src_rank(rank_prefix_matrix)
            if topk_idx is not None and topk_weights is not None:
                assert recv_topk_idx is not None and recv_topk_weights is not None
                if num_worst_tokens > 0:
                    num_recv_tokens_per_expert_list = []
                elif not num_recv_tokens_per_expert_list and num_tokens_per_expert is not None:
                    num_local_experts = num_tokens_per_expert.numel() // self.group_size
                    recv_topk_idx_cpu = recv_topk_idx.cpu()
                    flat_recv_topk_idx = recv_topk_idx_cpu[recv_topk_idx_cpu >= 0].to(torch.int64)
                    num_recv_tokens_per_expert_list = torch.bincount(
                        flat_recv_topk_idx,
                        minlength=num_local_experts,
                    ).tolist()
            elif num_worst_tokens == 0 and not num_recv_tokens_per_expert_list and num_tokens_per_expert is not None:
                num_recv_tokens_per_expert_list = self._get_intranode_recv_tokens_per_expert_list(num_tokens_per_expert)
            handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head,
                      send_pos, row_src_rank)
            return (
                recv_x, recv_x_scales
            ) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(
                event)

    # noinspection PyTypeChecker
    def combine(self, x: torch.Tensor, handle: Tuple,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                config: Optional[Config] = None,
                previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require the ranks to have fast peer connectivity within a node.
        Internode kernels additionally require RDMA connectivity across nodes for matching local device indices.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            bias: 0, 1 or 2 `[num_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(x, handle, topk_weights, bias, config, previous_event, async_finish, allocate_on_comm_stream)
        if self._use_intranode_collective_fallback:
            return self._intranode_combine_fallback(x, handle, topk_weights, bias, previous_event, async_finish)

        # NOTES: the second `_` is for the sending side, so we should use the third one
        if len(handle) == 6:
            rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
            row_src_rank = None
        else:
            rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head, _, row_src_rank = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix,
                                                                          channel_prefix_matrix, send_head, row_src_rank, config,
                                                                          getattr(previous_event, 'event',
                                                                                  None), async_finish, allocate_on_comm_stream)
        return recv_x, recv_topk_weights, EventOverlap(event)

    # noinspection PyTypeChecker
    def internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           handle: Optional[Tuple] = None,
                           num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                           is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                           topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,
                           num_worst_tokens: int = 0, config: Optional[Config] = None,
                           previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                           allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            is_token_in_rank, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head = handle
            num_recv_tokens = recv_src_meta.size(0)
            num_rdma_recv_tokens = send_nvl_head.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, _, _, _, _, event = self.runtime.internode_dispatch(
                x, x_scales, topk_idx, topk_weights, None, None, is_token_in_rank, None, num_recv_tokens, num_rdma_recv_tokens,
                rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head, event = self.runtime.internode_dispatch(
                x, x_scales, topk_idx, topk_weights,
                num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert,
                0, 0, None, None, None, None,
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            handle = (is_token_in_rank, rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, recv_rdma_channel_prefix_matrix,
                      recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, recv_src_meta, send_rdma_head,
                      send_nvl_head)
            return (
                recv_x, recv_x_scales
            ) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(
                event)

    # noinspection PyTypeChecker
    def internode_combine(self, x: torch.Tensor, handle: Union[tuple, list],
                          topk_weights: Optional[torch.Tensor] = None,
                          bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                          config: Optional[Config] = None,
                          previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                          allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        assert config is not None

        # Unpack handle and bias
        is_combined_token_in_rank, \
            _, _, \
            rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, gbl_rank_prefix_sum, \
            src_meta, send_rdma_head, send_nvl_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        combined_x, combined_topk_weights, event = self.runtime.internode_combine(x, topk_weights, bias_0, bias_1, src_meta,
                                                                                  is_combined_token_in_rank, rdma_channel_prefix_matrix,
                                                                                  rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
                                                                                  send_rdma_head, send_nvl_head, config,
                                                                                  getattr(previous_event, 'event',
                                                                                          None), async_finish, allocate_on_comm_stream)
        return combined_x, combined_topk_weights, EventOverlap(event)

    def _intranode_dispatch_fallback(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], handle: Optional[Tuple],
                                     num_tokens_per_rank: Optional[torch.Tensor], is_token_in_rank: Optional[torch.Tensor],
                                     num_tokens_per_expert: Optional[torch.Tensor], topk_idx: Optional[torch.Tensor],
                                     topk_weights: Optional[torch.Tensor], num_worst_tokens: int,
                                     previous_event: Optional[EventOverlap], async_finish: bool) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], List[int], Tuple,
                  EventOverlap]:
        if previous_event is not None:
            previous_event.current_stream_wait()
        if isinstance(x, tuple):
            raise NotImplementedError('The XPU intranode collective fallback currently supports tensor inputs only.')

        if handle is not None:
            rank_prefix_matrix, _, _, recv_src_idx, is_token_in_rank, metadata = handle
            gathered_x = self._all_gather_object(x.cpu())
            recv_rows = [gathered_x[src_rank][src_idx] for src_rank, src_idx in zip(metadata['recv_src_rank'], recv_src_idx.cpu().tolist())]
            recv_x = self._stack_or_empty(recv_rows, x)
            return recv_x, None, None, [], handle, EventOverlap(_ImmediateEvent() if async_finish else None)

        assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
        assert topk_idx is not None or topk_weights is None

        payload = {
            'x': x.cpu(),
            'topk_idx': topk_idx.cpu() if topk_idx is not None else None,
            'topk_weights': topk_weights.cpu() if topk_weights is not None else None,
            'is_token_in_rank': is_token_in_rank.cpu(),
            'num_tokens_per_expert': num_tokens_per_expert.cpu(),
        }
        gathered = self._all_gather_object(payload)
        num_local_experts = num_tokens_per_expert.numel() // self.group_size
        local_expert_begin = self.rank * num_local_experts
        local_expert_end = local_expert_begin + num_local_experts

        recv_rows = []
        recv_src_rank = []
        recv_src_idx = []
        recv_topk_idx_rows = []
        recv_topk_weight_rows = []
        counts_by_source = [0] * self.group_size
        num_recv_tokens_per_expert_list = [0] * num_local_experts

        for src_rank, package in enumerate(gathered):
            src_x = package['x']
            src_topk_idx = package['topk_idx']
            src_topk_weights = package['topk_weights']
            src_is_token_in_rank = package['is_token_in_rank']
            src_num_tokens_per_expert = package['num_tokens_per_expert']

            if src_topk_idx is None:
                num_recv_tokens_per_expert_list = [
                    curr + incoming for curr, incoming in zip(
                        num_recv_tokens_per_expert_list,
                        src_num_tokens_per_expert[local_expert_begin:local_expert_end].tolist(),
                    )
                ]
                for token_idx in range(src_is_token_in_rank.size(0)):
                    if not bool(src_is_token_in_rank[token_idx, self.rank]):
                        continue
                    recv_rows.append(src_x[token_idx].clone())
                    recv_src_rank.append(src_rank)
                    recv_src_idx.append(token_idx)
                    counts_by_source[src_rank] += 1
                continue

            for token_idx in range(src_topk_idx.size(0)):
                token_experts = src_topk_idx[token_idx]
                local_mask = (token_experts >= local_expert_begin) & (token_experts < local_expert_end)
                if not bool(local_mask.any()):
                    continue

                recv_rows.append(src_x[token_idx].clone())
                recv_src_rank.append(src_rank)
                recv_src_idx.append(token_idx)
                counts_by_source[src_rank] += 1

                if topk_weights is not None:
                    local_topk_idx = torch.full_like(token_experts, -1)
                    local_topk_idx[local_mask] = token_experts[local_mask] - local_expert_begin
                    recv_topk_idx_rows.append(local_topk_idx)

                    local_topk_weight = torch.zeros_like(src_topk_weights[token_idx])
                    local_topk_weight[local_mask] = src_topk_weights[token_idx][local_mask]
                    recv_topk_weight_rows.append(local_topk_weight)

                    for expert_idx in local_topk_idx[local_topk_idx >= 0].tolist():
                        num_recv_tokens_per_expert_list[expert_idx] += 1

        recv_x = self._stack_or_empty(recv_rows, x)
        recv_src_idx_tensor = torch.tensor(recv_src_idx, dtype=torch.int32, device=x.device)
        counts_tensor = torch.tensor(counts_by_source, dtype=torch.int32, device=x.device)
        gathered_counts = [torch.zeros_like(counts_tensor) for _ in range(self.group_size)]
        dist.all_gather(gathered_counts, counts_tensor, group=self.group)
        rank_prefix_matrix = torch.stack(gathered_counts, dim=1).cumsum(dim=0).contiguous()

        metadata = {
            'fallback': 'intranode_collective',
            'recv_src_rank': recv_src_rank,
            'source_num_tokens': x.size(0),
        }
        handle = (rank_prefix_matrix, None, None, recv_src_idx_tensor, is_token_in_rank, metadata)

        recv_topk_idx_tensor = None
        recv_topk_weights_tensor = None
        if topk_weights is not None:
            recv_topk_idx_tensor = self._stack_or_empty(recv_topk_idx_rows, topk_idx)
            recv_topk_weights_tensor = self._stack_or_empty(recv_topk_weight_rows, topk_weights)

        if num_worst_tokens > 0:
            recv_x = self._pad_rows(recv_x, num_worst_tokens, 0)
            if recv_topk_idx_tensor is not None:
                recv_topk_idx_tensor = self._pad_rows(recv_topk_idx_tensor, num_worst_tokens, -1)
            if recv_topk_weights_tensor is not None:
                recv_topk_weights_tensor = self._pad_rows(recv_topk_weights_tensor, num_worst_tokens, 0)
            num_recv_tokens_per_expert_list = []

        return recv_x, recv_topk_idx_tensor, recv_topk_weights_tensor, num_recv_tokens_per_expert_list, handle, EventOverlap(
            _ImmediateEvent() if async_finish else None)

    def _intranode_combine_fallback(self, x: torch.Tensor, handle: Tuple, topk_weights: Optional[torch.Tensor],
                                    bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                                    previous_event: Optional[EventOverlap], async_finish: bool) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        if previous_event is not None:
            previous_event.current_stream_wait()

        _, _, _, src_idx, _, metadata = handle
        payload = {
            'x': x.cpu(),
            'topk_weights': topk_weights.cpu() if topk_weights is not None else None,
            'src_rank': metadata['recv_src_rank'],
            'src_idx': src_idx.cpu(),
        }
        gathered = self._all_gather_object(payload)

        combined_x = torch.zeros((metadata['source_num_tokens'], x.size(1)), dtype=torch.float32, device='cpu')
        combined_topk_weights = None if topk_weights is None else torch.zeros(
            (metadata['source_num_tokens'], topk_weights.size(1)), dtype=torch.float32, device='cpu')

        for package in gathered:
            src_rank = torch.tensor(package['src_rank'], dtype=torch.int64, device='cpu')
            token_idx = package['src_idx'].to(dtype=torch.int64, device='cpu')
            rank_mask = src_rank == self.rank
            if not bool(rank_mask.any()):
                continue
            selected_idx = token_idx[rank_mask]
            combined_x.index_add_(0, selected_idx, package['x'][rank_mask].to(torch.float32))
            if combined_topk_weights is not None:
                combined_topk_weights.index_add_(0, selected_idx, package['topk_weights'][rank_mask].to(torch.float32))

        combined_x = combined_x.to(device=x.device, dtype=x.dtype)
        if combined_topk_weights is not None:
            combined_topk_weights = combined_topk_weights.to(device=x.device, dtype=topk_weights.dtype)

        bias_0, bias_1 = Buffer._unpack_bias(bias)
        if bias_0 is not None:
            combined_x = combined_x + bias_0
        if bias_1 is not None:
            combined_x = combined_x + bias_1

        return combined_x, combined_topk_weights, EventOverlap(_ImmediateEvent() if async_finish else None)

    @staticmethod
    def _stack_or_empty(rows: List[torch.Tensor], template: torch.Tensor) -> torch.Tensor:
        if rows:
            return torch.stack([row.to(device=template.device, dtype=template.dtype) for row in rows]).contiguous()
        return template.new_empty((0, *template.shape[1:]))

    @staticmethod
    def _pad_rows(tensor: torch.Tensor, target_rows: int, fill_value: int) -> torch.Tensor:
        if tensor.size(0) >= target_rows:
            return tensor
        padded = tensor.new_full((target_rows, *tensor.shape[1:]), fill_value)
        padded[:tensor.size(0)] = tensor
        return padded

    def _get_intranode_dispatch_metadata(self, num_tokens_per_rank: torch.Tensor, is_token_in_rank: torch.Tensor,
                                         num_channels: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        gathered_rank_counts = [torch.empty_like(num_tokens_per_rank) for _ in range(self.group_size)]
        dist.all_gather(gathered_rank_counts, num_tokens_per_rank.contiguous(), group=self.group)
        rank_prefix_matrix = torch.stack(gathered_rank_counts, dim=0).cumsum(dim=0).to(dtype=torch.int32).contiguous()

        num_tokens = is_token_in_rank.size(0)
        num_tokens_per_channel = math.ceil(num_tokens / num_channels)
        per_channel_counts = []
        for channel_id in range(num_channels):
            token_start = min(num_tokens_per_channel * channel_id, num_tokens)
            token_end = min(token_start + num_tokens_per_channel, num_tokens)
            if token_start >= token_end:
                counts = torch.zeros((self.group_size, ), dtype=torch.int32, device=is_token_in_rank.device)
            else:
                counts = is_token_in_rank[token_start:token_end].to(dtype=torch.int32).sum(dim=0)
            per_channel_counts.append(counts.to(dtype=torch.int32))
        channel_prefix_matrix = torch.stack(per_channel_counts, dim=1).cumsum(dim=1).to(dtype=torch.int32).contiguous()

        return int(rank_prefix_matrix[-1, self.rank].item()), rank_prefix_matrix, channel_prefix_matrix

    def _get_intranode_recv_tokens_per_expert_list(self, num_tokens_per_expert: torch.Tensor) -> List[int]:
        global_num_tokens_per_expert = num_tokens_per_expert.contiguous().clone()
        dist.all_reduce(global_num_tokens_per_expert, group=self.group)
        num_local_experts = global_num_tokens_per_expert.numel() // self.group_size
        local_expert_begin = self.rank * num_local_experts
        local_expert_end = local_expert_begin + num_local_experts
        return global_num_tokens_per_expert[local_expert_begin:local_expert_end].cpu().tolist()

    def _get_intranode_send_pos(self, is_token_in_rank: torch.Tensor) -> torch.Tensor:
        send_pos_cpu = torch.full((self.group_size, is_token_in_rank.size(0)), -1, dtype=torch.int32, device='cpu')
        mask_cpu = is_token_in_rank.cpu()
        for dst_rank in range(self.group_size):
            dst_tokens = mask_cpu[:, dst_rank].to(torch.bool).nonzero(as_tuple=False).view(-1)
            if dst_tokens.numel() == 0:
                continue
            send_pos_cpu[dst_rank, :dst_tokens.numel()] = dst_tokens.to(dtype=torch.int32)
        return send_pos_cpu.to(device=is_token_in_rank.device)

    def _get_intranode_recv_src_idx(self, is_token_in_rank: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        gathered_masks = self._all_gather_object(is_token_in_rank.cpu())
        recv_src_rank = []
        recv_src_idx = []
        for src_rank, src_mask in enumerate(gathered_masks):
            selected = src_mask[:, self.rank].to(torch.bool)
            token_indices = selected.nonzero(as_tuple=False).view(-1).tolist()
            recv_src_rank.extend([src_rank] * len(token_indices))
            recv_src_idx.extend(token_indices)
        return torch.tensor(recv_src_idx, dtype=torch.int32, device=is_token_in_rank.device), recv_src_rank

    def _get_intranode_recv_topk_metadata(self, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                                          num_tokens_per_expert: torch.Tensor, num_worst_tokens: int) -> \
            Tuple[torch.Tensor, torch.Tensor, List[int]]:
        payload = {
            'topk_idx': topk_idx.cpu(),
            'topk_weights': topk_weights.cpu(),
        }
        gathered = self._all_gather_object(payload)
        num_local_experts = num_tokens_per_expert.numel() // self.group_size
        local_expert_begin = self.rank * num_local_experts
        local_expert_end = local_expert_begin + num_local_experts
        recv_topk_idx_rows = []
        recv_topk_weight_rows = []
        num_recv_tokens_per_expert_list = [0] * num_local_experts

        for package in gathered:
            src_topk_idx = package['topk_idx']
            src_topk_weights = package['topk_weights']
            for token_idx in range(src_topk_idx.size(0)):
                token_experts = src_topk_idx[token_idx]
                local_mask = (token_experts >= local_expert_begin) & (token_experts < local_expert_end)
                if not bool(local_mask.any()):
                    continue

                local_topk_idx = torch.full_like(token_experts, -1)
                local_topk_idx[local_mask] = token_experts[local_mask] - local_expert_begin
                recv_topk_idx_rows.append(local_topk_idx)

                local_topk_weight = torch.zeros_like(src_topk_weights[token_idx])
                local_topk_weight[local_mask] = src_topk_weights[token_idx][local_mask]
                recv_topk_weight_rows.append(local_topk_weight)

                for expert_idx in local_topk_idx[local_topk_idx >= 0].tolist():
                    num_recv_tokens_per_expert_list[expert_idx] += 1

        recv_topk_idx_tensor = self._stack_or_empty(recv_topk_idx_rows, topk_idx)
        recv_topk_weight_tensor = self._stack_or_empty(recv_topk_weight_rows, topk_weights)
        if num_worst_tokens > 0:
            recv_topk_idx_tensor = self._pad_rows(recv_topk_idx_tensor, num_worst_tokens, -1)
            recv_topk_weight_tensor = self._pad_rows(recv_topk_weight_tensor, num_worst_tokens, 0)
            num_recv_tokens_per_expert_list = []
        return recv_topk_idx_tensor, recv_topk_weight_tensor, num_recv_tokens_per_expert_list

    def _combine_topk_weights_metadata(self, topk_weights: torch.Tensor, handle: Tuple) -> torch.Tensor:
        rank_prefix_matrix, _, _, src_idx, _, _ = handle[:6]
        src_rank = []
        start = 0
        for src_rank_id in range(self.group_size):
            end = int(rank_prefix_matrix[src_rank_id, self.rank].item())
            src_rank.extend([src_rank_id] * (end - start))
            start = end
        payload = {
            'topk_weights': topk_weights.cpu(),
            'src_rank': src_rank,
            'src_idx': src_idx.cpu(),
        }
        gathered = self._all_gather_object(payload)
        combined_topk_weights = torch.zeros((handle[5].size(0), topk_weights.size(1)), dtype=torch.float32, device='cpu')
        for package in gathered:
            src_rank_tensor = torch.tensor(package['src_rank'], dtype=torch.int64, device='cpu')
            token_idx = package['src_idx'].to(dtype=torch.int64, device='cpu')
            rank_mask = src_rank_tensor == self.rank
            if not bool(rank_mask.any()):
                continue
            selected_idx = token_idx[rank_mask]
            combined_topk_weights.index_add_(0, selected_idx, package['topk_weights'][rank_mask].to(torch.float32))
        return combined_topk_weights.to(device=topk_weights.device, dtype=topk_weights.dtype)

    def _get_intranode_row_src_rank(self, rank_prefix_matrix: torch.Tensor) -> torch.Tensor:
        rank_prefix_cpu = rank_prefix_matrix.cpu()
        row_src_rank = []
        start = 0
        for src_rank_id in range(self.group_size):
            end = int(rank_prefix_cpu[src_rank_id, self.rank].item())
            row_src_rank.extend([src_rank_id] * (end - start))
            start = end
        return torch.tensor(row_src_rank, dtype=torch.int32, device=rank_prefix_matrix.device)

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        """
        Unsupported on XPU; retained only for API compatibility.
        """
        _raise_low_latency_unsupported('Buffer.clean_low_latency_buffer')

    # noinspection PyTypeChecker
    def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int, num_experts: int,
                             cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                             dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
                             use_fp8: bool = True, round_scale: bool = False, use_ue8m0: bool = False,
                             async_finish: bool = False, return_recv_hook: bool = False) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """
        Unsupported on XPU; retained only for API compatibility.
        """
        _raise_low_latency_unsupported('Buffer.low_latency_dispatch')

    # noinspection PyTypeChecker
    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: tuple, use_logfmt: bool = False, zero_copy: bool = False, async_finish: bool = False,
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None,
                            combine_wait_recv_cost_stats: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        Unsupported on XPU; retained only for API compatibility.
        """
        _raise_low_latency_unsupported('Buffer.low_latency_combine')

    def low_latency_update_mask_buffer(self, rank_to_mask: int, mask: bool = False):
        """
        Unsupported on XPU; retained only for API compatibility.
        """
        _raise_low_latency_unsupported('Buffer.low_latency_update_mask_buffer')

    def low_latency_query_mask_buffer(self, mask_status: torch.Tensor):
        """
        Unsupported on XPU; retained only for API compatibility.
        """
        _raise_low_latency_unsupported('Buffer.low_latency_query_mask_buffer')

    def low_latency_clean_mask_buffer(self):
        """
        Unsupported on XPU; retained only for API compatibility.
        """
        _raise_low_latency_unsupported('Buffer.low_latency_clean_mask_buffer')

    def get_next_low_latency_combine_buffer(self, handle: object):
        """
        Unsupported on XPU; retained only for API compatibility.
        """
        _raise_low_latency_unsupported('Buffer.get_next_low_latency_combine_buffer')
