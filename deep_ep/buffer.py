import os
import socket
import struct
import tempfile
import time
import numpy as np
import torch
import torch.distributed as dist
from contextlib import suppress
from typing import Callable, List, Tuple, Optional, Union

# noinspection PyUnresolvedReferences
import deep_ep_cpp
# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config, EventHandle
from .utils import EventOverlap, check_nvlink_connections


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        - low-latency all-to-all (dispatch and combine, using RDMA)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
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
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: whether allow NVLink traffic for low-latency mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            allow_mnnvl: whether to allow MNNVL
            use_fabric: whether to use fabric API for memory buffers.
            enable_shrink: whether to enable shrink mode. The enable mode allocates a mask buffer to support masking ranks dynamically.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            comm: the `mpi4py.MPI.Comm` communicator to use in case the group parameter is absent.
        """
        check_nvlink_connections(group)

        # Initialize the CPP runtime
        if group is not None:
            self.rank = group.rank()
            self.group = group
            self.group_size = group.size()

            def all_gather_object(obj):
                object_list = [None] * self.group_size
                dist.all_gather_object(object_list, obj, group)
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
        self.is_xpu_runtime = hasattr(deep_ep_cpp, '_xpu_get_ipc_handle_fd')
        self._xpu_internode_handle_cache = {}
        self.runtime = deep_ep_cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode, explicitly_destroy,
                                          enable_shrink, use_fabric)

        # Synchronize device IDs
        local_device_id = self.runtime.get_local_device_id()
        device_ids = all_gather_object(local_device_id)

        # Synchronize IPC handles
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        ipc_handles = all_gather_object(local_ipc_handle)
        if num_nvl_bytes > 0 and self.group_size > 1 and self.is_xpu_runtime:
            ipc_handles = self._exchange_xpu_ipc_fds(ipc_handles, all_gather_object)

        # Synchronize NVSHMEM unique IDs
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            if self.is_xpu_runtime:
                if low_latency_mode:
                    raise RuntimeError('XPU low-latency mode is not migrated yet')
                os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
                if os.environ.get('I_MPI_MPCP_SERVER_PORT', '') == os.environ.get('MASTER_PORT', ''):
                    os.environ['I_MPI_MPCP_SERVER_PORT'] = str(int(os.environ['MASTER_PORT']) + 1)
            else:
                # Enable IBGDA
                assert num_qps_per_rank > 0
                os.environ['NVSHMEM_DISABLE_P2P'] = '0' if allow_nvlink_for_low_latency_mode else '1'
                os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
                os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'

                # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
                self.nvshmem_qp_depth = int(os.environ.get('NVSHMEM_QP_DEPTH', '1024'))
                os.environ['NVSHMEM_QP_DEPTH'] = str(self.nvshmem_qp_depth)

                # Reduce gpu memory usage
                # 6 default teams + 1 extra team
                os.environ['NVSHMEM_MAX_TEAMS'] = '7'
                # Disable NVLink SHArP
                os.environ['NVSHMEM_DISABLE_NVLS'] = '1'
                # NOTES: NVSHMEM initialization requires at least 256 MiB
                os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'

                if not allow_mnnvl:
                    # Disable multi-node NVLink detection
                    os.environ['NVSHMEM_DISABLE_MNNVL'] = '1'

            # Synchronize using the root ID
            if (low_latency_mode and self.rank == 0) or (not low_latency_mode and self.runtime.get_rdma_rank() == 0):
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            nvshmem_unique_ids = all_gather_object(root_unique_id)
            root_unique_id = nvshmem_unique_ids[0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)]

        # Make CPP runtime available
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert self.runtime.is_available()

    def _exchange_xpu_ipc_fds(self, ipc_handles, all_gather_object):
        local_fd = deep_ep_cpp._xpu_get_ipc_handle_fd(ipc_handles[self.rank])
        if local_fd < 0:
            raise RuntimeError('XPU IPC FD exchange required but no local FD was exported')

        fd_size = struct.calcsize('i')
        socket_path = os.path.join(tempfile.gettempdir(), f'deep_ep_xpu_ipc_{os.getpid()}_{self.rank}_{id(self)}.sock')
        timeout_s = 30.0
        with suppress(FileNotFoundError):
            os.unlink(socket_path)

        num_nvl_ranks = min(self.group_size, 8)
        nvl_group_start = self.runtime.get_rdma_rank() * num_nvl_ranks
        nvl_group_end = nvl_group_start + num_nvl_ranks
        local_ipc_ranks = [rank for rank in range(nvl_group_start, nvl_group_end) if rank < self.group_size]
        if self.rank not in local_ipc_ranks:
            raise RuntimeError(f'XPU IPC rank {self.rank} is outside its local NVL group {local_ipc_ranks}')

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(socket_path)
            listener.listen(self.group_size)
            listener.settimeout(timeout_s)
            socket_paths = all_gather_object(socket_path)
            if len(socket_paths) != self.group_size:
                raise RuntimeError(f'XPU IPC socket path list size mismatch: {len(socket_paths)}')

            for dst_rank in local_ipc_ranks:
                if dst_rank == self.rank:
                    continue
                dst_socket_path = socket_paths[dst_rank]
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sender:
                    sender.settimeout(timeout_s)
                    deadline = time.monotonic() + timeout_s
                    while True:
                        try:
                            sender.connect(dst_socket_path)
                            break
                        except (FileNotFoundError, ConnectionRefusedError) as exc:
                            if time.monotonic() >= deadline:
                                raise TimeoutError(f'timed out connecting to XPU IPC socket for rank {dst_rank}') from exc
                            time.sleep(0.01)
                    sender.sendmsg([struct.pack('i', self.rank)], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack('i', local_fd))])

            for _ in range(len(local_ipc_ranks) - 1):
                try:
                    conn, _ = listener.accept()
                except socket.timeout as exc:
                    raise TimeoutError('timed out waiting for XPU IPC socket connection') from exc
                with conn:
                    conn.settimeout(timeout_s)
                    try:
                        msg, ancdata, _, _ = conn.recvmsg(fd_size, socket.CMSG_SPACE(fd_size))
                    except socket.timeout as exc:
                        raise TimeoutError('timed out receiving XPU IPC file descriptor') from exc
                if len(msg) != fd_size:
                    raise RuntimeError('failed to receive XPU IPC source rank')
                src_rank = struct.unpack('i', msg)[0]
                if src_rank not in local_ipc_ranks or src_rank == self.rank:
                    raise RuntimeError(f'invalid XPU IPC source rank received: {src_rank}')
                remote_fd = None
                for level, kind, data in ancdata:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        remote_fd = struct.unpack('i', data[:fd_size])[0]
                        break
                if remote_fd is None:
                    raise RuntimeError(f'failed to receive XPU IPC file descriptor from rank {src_rank}')
                try:
                    ipc_handles[src_rank] = deep_ep_cpp._xpu_set_ipc_handle_fd(ipc_handles[src_rank], remote_fd)
                except Exception:
                    os.close(remote_fd)
                    raise
        finally:
            listener.close()
            with suppress(FileNotFoundError):
                os.unlink(socket_path)

        return ipc_handles

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """

        assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'

        self.runtime.destroy()
        self.runtime = None

    @staticmethod
    def is_sm90_compiled():
        return deep_ep_cpp.is_sm90_compiled()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, 'The SM count must be even'
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int) -> int:
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        return deep_ep_cpp.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        ts: torch.Stream = self.runtime.get_comm_stream()
        if hasattr(torch, 'xpu') and torch.xpu.is_available() and ts.device_type == torch.xpu.current_stream().device_type:
            return torch.xpu.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)
        return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)

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
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = \
            self.runtime.get_dispatch_layout(topk_idx, num_experts, getattr(previous_event, 'event', None),
                                             async_finish, allocate_on_comm_stream)
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, EventOverlap(event)

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
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert indices
                selected by each token, `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
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

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
            num_recv_tokens = recv_src_idx.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, event = self.runtime.intranode_dispatch(
                x, x_scales, None, None, None, is_token_in_rank, None, num_recv_tokens, rank_prefix_matrix, channel_prefix_matrix,
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event = \
                self.runtime.intranode_dispatch(x, x_scales, topk_idx, topk_weights,
                                                num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, 0, None, None,
                                                expert_alignment, num_worst_tokens, config,
                                                getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
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
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

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

        # NOTES: the second `_` is for the sending side, so we should use the third one
        rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix,
                                                                          channel_prefix_matrix, send_head, config,
                                                                          getattr(previous_event, 'event',
                                                                                  None), async_finish, allocate_on_comm_stream)
        return recv_x, recv_topk_weights, EventOverlap(event)

    # noinspection PyTypeChecker
    def _xpu_all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        gathered = [torch.empty_like(tensor) for _ in range(self.group_size)]
        dist.all_gather(gathered, tensor, self.group)
        return gathered

    def _xpu_internode_event(self, async_finish: bool) -> EventOverlap:
        return EventOverlap(EventHandle() if async_finish else None)

    def _xpu_internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                                handle: Optional[Tuple],
                                num_tokens_per_rank: Optional[torch.Tensor],
                                num_tokens_per_rdma_rank: Optional[torch.Tensor],
                                is_token_in_rank: Optional[torch.Tensor],
                                num_tokens_per_expert: Optional[torch.Tensor],
                                topk_idx: Optional[torch.Tensor],
                                topk_weights: Optional[torch.Tensor],
                                expert_alignment: int,
                                num_worst_tokens: int,
                                config: Config,
                                async_finish: bool) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        assert config is not None
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            cache = self._xpu_internode_handle_cache.get(id(handle))
            if cache is None:
                raise RuntimeError('missing XPU internode cached dispatch metadata')
            is_token_in_rank = handle[0]
            topk_idx = None
            topk_weights = None
        else:
            assert num_tokens_per_rank is not None and num_tokens_per_rdma_rank is not None
            assert is_token_in_rank is not None and num_tokens_per_expert is not None

        if x_scales is not None:
            gathered_x = self._xpu_all_gather_tensor(x)
            gathered_scales = self._xpu_all_gather_tensor(x_scales)
        else:
            gathered_x = self._xpu_all_gather_tensor(x)
            gathered_scales = None
        gathered_masks = self._xpu_all_gather_tensor(is_token_in_rank)
        gathered_topk_idx = self._xpu_all_gather_tensor(topk_idx) if topk_idx is not None else None
        gathered_topk_weights = self._xpu_all_gather_tensor(topk_weights) if topk_weights is not None else None

        recv_chunks, recv_scale_chunks, recv_topk_idx_chunks, recv_topk_weight_chunks = [], [], [], []
        recv_src_rank, recv_src_token = [], []
        per_source_counts = []
        local_experts = None
        if num_tokens_per_expert is not None:
            expert_counts = num_tokens_per_expert.clone()
            dist.all_reduce(expert_counts, group=self.group)
            num_local_experts = expert_counts.numel() // self.group_size
            local_begin = self.rank * num_local_experts
            local_experts = expert_counts[local_begin:local_begin + num_local_experts].cpu().tolist()

        for src_rank in range(self.group_size):
            mask = gathered_masks[src_rank][:, self.rank]
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            per_source_counts.append(int(indices.numel()))
            if indices.numel() == 0:
                continue
            recv_chunks.append(gathered_x[src_rank].index_select(0, indices))
            if gathered_scales is not None:
                recv_scale_chunks.append(gathered_scales[src_rank].index_select(0, indices))
            recv_src_rank.append(torch.full((indices.numel(), ), src_rank, dtype=torch.int32, device=x.device))
            recv_src_token.append(indices.to(torch.int32))
            if gathered_topk_idx is not None:
                src_topk_idx = gathered_topk_idx[src_rank].index_select(0, indices)
                src_topk_weights = gathered_topk_weights[src_rank].index_select(0, indices)
                num_experts = num_tokens_per_expert.numel()
                num_experts_per_rank = num_experts // self.group_size
                expert_begin = self.rank * num_experts_per_rank
                expert_end = expert_begin + num_experts_per_rank
                local_idx = src_topk_idx - expert_begin
                valid = (src_topk_idx >= expert_begin) & (src_topk_idx < expert_end)
                recv_topk_idx_chunks.append(local_idx.masked_fill(~valid, -1).to(src_topk_idx.dtype))
                recv_topk_weight_chunks.append(src_topk_weights.masked_fill(~valid, 0.0))

        recv_x = torch.cat(recv_chunks, dim=0) if recv_chunks else x[:0].clone()
        real_recv_tokens = recv_x.size(0)
        if num_worst_tokens > 0 and num_worst_tokens > real_recv_tokens:
            pad = torch.empty((num_worst_tokens - real_recv_tokens, x.size(1)), dtype=x.dtype, device=x.device)
            recv_x = torch.cat((recv_x, pad), dim=0)
        recv_x_scales = None
        if gathered_scales is not None:
            recv_x_scales = torch.cat(recv_scale_chunks, dim=0) if recv_scale_chunks else x_scales[:0].clone()
            if num_worst_tokens > 0 and num_worst_tokens > recv_x_scales.size(0):
                pad_shape = (num_worst_tokens - recv_x_scales.size(0), ) + tuple(recv_x_scales.shape[1:])
                recv_x_scales = torch.cat((recv_x_scales, torch.empty(pad_shape, dtype=x_scales.dtype, device=x_scales.device)), dim=0)

        recv_topk_idx, recv_topk_weights = None, None
        if gathered_topk_idx is not None:
            recv_topk_idx = torch.cat(recv_topk_idx_chunks, dim=0) if recv_topk_idx_chunks else topk_idx[:0].clone()
            recv_topk_weights = torch.cat(recv_topk_weight_chunks, dim=0) if recv_topk_weight_chunks else topk_weights[:0].clone()
            if num_worst_tokens > 0 and num_worst_tokens > recv_topk_idx.size(0):
                pad_rows = num_worst_tokens - recv_topk_idx.size(0)
                recv_topk_idx = torch.cat(
                    (recv_topk_idx, torch.full((pad_rows, topk_idx.size(1)), -1, dtype=topk_idx.dtype, device=topk_idx.device)), dim=0)
                recv_topk_weights = torch.cat(
                    (recv_topk_weights, torch.zeros(
                        (pad_rows, topk_weights.size(1)), dtype=topk_weights.dtype, device=topk_weights.device)),
                    dim=0)

        num_recv_tokens = recv_x.size(0)
        num_channels = max(Buffer.num_sms // 2, 1)
        rdma_channel_prefix_matrix = torch.zeros((self.runtime.get_num_rdma_ranks(), num_channels), dtype=torch.int32, device=x.device)
        gbl_channel_prefix_matrix = torch.zeros((self.group_size, num_channels), dtype=torch.int32, device=x.device)
        recv_rdma_channel_prefix_matrix = torch.zeros_like(rdma_channel_prefix_matrix)
        recv_gbl_channel_prefix_matrix = torch.zeros_like(gbl_channel_prefix_matrix)
        recv_gbl_rank_prefix_sum = torch.tensor(np.cumsum(per_source_counts).tolist(), dtype=torch.int32, device=x.device)
        rdma_counts = [0] * self.runtime.get_num_rdma_ranks()
        for src_rank, count in enumerate(per_source_counts):
            rdma_counts[src_rank // 8] += count
        recv_rdma_rank_prefix_sum = torch.tensor(np.cumsum(rdma_counts).tolist(), dtype=torch.int32, device=x.device)
        recv_src_meta = torch.zeros(
            (num_recv_tokens, self.runtime.get_source_meta_bytes() if hasattr(self.runtime, 'get_source_meta_bytes') else 8),
            dtype=torch.uint8,
            device=x.device)
        send_rdma_head = torch.full((x.size(0), self.runtime.get_num_rdma_ranks()), -1, dtype=torch.int32, device=x.device)
        send_nvl_head = torch.full((num_recv_tokens, 8), -1, dtype=torch.int32, device=x.device)
        src_rank_tensor = torch.cat(recv_src_rank, dim=0) if recv_src_rank else torch.empty((0, ), dtype=torch.int32, device=x.device)
        src_token_tensor = torch.cat(recv_src_token, dim=0) if recv_src_token else torch.empty((0, ), dtype=torch.int32, device=x.device)

        handle = (is_token_in_rank, rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, recv_rdma_channel_prefix_matrix,
                  recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, recv_src_meta, send_rdma_head,
                  send_nvl_head)
        self._xpu_internode_handle_cache[id(handle)] = {
            'src_rank': src_rank_tensor,
            'src_token': src_token_tensor,
            'real_recv_tokens': real_recv_tokens,
        }
        return (
            recv_x, recv_x_scales
        ) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, local_experts or [], handle, self._xpu_internode_event(
            async_finish)

    def _xpu_internode_combine(self, x: torch.Tensor, handle: Union[tuple, list], topk_weights: Optional[torch.Tensor],
                               bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                               async_finish: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        cache = self._xpu_internode_handle_cache.get(id(handle))
        if cache is None:
            raise RuntimeError('missing XPU internode combine metadata')
        is_token_in_rank = handle[0]
        num_tokens, hidden = is_token_in_rank.size(0), x.size(1)
        bias_0, bias_1 = Buffer._unpack_bias(bias)
        real_recv_tokens = cache['real_recv_tokens']
        payload = (x[:real_recv_tokens].detach().cpu(), cache['src_rank'].cpu(), cache['src_token'].cpu(),
                   topk_weights[:real_recv_tokens].detach().cpu() if topk_weights is not None else None)
        gathered = [None] * self.group_size
        dist.all_gather_object(gathered, payload, self.group)

        combined_x = torch.zeros((num_tokens, hidden), dtype=x.dtype, device=x.device)
        combined_topk_weights = None
        if topk_weights is not None:
            combined_topk_weights = torch.zeros((num_tokens, topk_weights.size(1)), dtype=topk_weights.dtype, device=x.device)
        for values_cpu, src_ranks_cpu, src_tokens_cpu, weights_cpu in gathered:
            if values_cpu.numel() == 0:
                continue
            select = src_ranks_cpu == self.rank
            if not bool(select.any()):
                continue
            dst_idx = src_tokens_cpu[select].to(device=x.device, dtype=torch.long)
            combined_x.index_add_(0, dst_idx, values_cpu[select].to(x.device))
            if combined_topk_weights is not None and weights_cpu is not None:
                combined_topk_weights.index_add_(0, dst_idx, weights_cpu[select].to(x.device))
        if bias_0 is not None:
            combined_x += bias_0
        if bias_1 is not None:
            combined_x += bias_1
        return combined_x, combined_topk_weights, self._xpu_internode_event(async_finish)

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
        if self.is_xpu_runtime:
            return self._xpu_internode_dispatch(x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank,
                                                num_tokens_per_expert, topk_idx, topk_weights, expert_alignment, num_worst_tokens, config,
                                                async_finish)

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
        if self.is_xpu_runtime:
            return self._xpu_internode_combine(x, handle, topk_weights, bias, async_finish)

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

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        self.runtime.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)

    # noinspection PyTypeChecker
    def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int, num_experts: int,
                             cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                             dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
                             use_fp8: bool = True, round_scale: bool = False, use_ue8m0: bool = False,
                             async_finish: bool = False, return_recv_hook: bool = False) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """
        A low-latency implementation for dispatching with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `torch.Tensor` with `deep_ep.topk_idx_t` (typically `torch.int64`), shaped as `[num_tokens, num_topk]`,
                only several top-k shapes are supported. `-1` indices (not selecting any expert) are supported.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            cumulative_local_expert_recv_stats: a cumulative expert count tensor for statistics, which should have shape
                `[num_local_experts]` and be typed as `torch.int`. This is useful for online service EP load balance
                monitoring.
            dispatch_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                This is useful for detecting and precisely localizing slow anomalies.
            use_fp8: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            round_scale: whether round the scaling factors into power of 2.
            use_ue8m0: whether use UE8M0 as scaling factor format (available only with `round_scale=True`).
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: a tensor or tuple with received tokens for each expert.
                With `use_fp8=True`: the first element is a `torch.Tensor` shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.float8_e4m3fn`.
                The second tensor is the corresponding scales for the first element with shape
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`,
                if `use_ue8m0=False`. With `use_ue8m0=True`, the second one is packed and shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 512]` with type `torch.int`.
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                With `use_fp8=False`, the result would be a tensor shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if synced).
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receives. As mentioned before, not all tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        assert self.nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2
        packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, hook = \
            self.runtime.low_latency_dispatch(x, topk_idx,
                                              cumulative_local_expert_recv_stats,
                                              dispatch_wait_recv_cost_stats,
                                              num_max_dispatch_tokens_per_rank, num_experts,
                                              use_fp8, round_scale, use_ue8m0,
                                              async_finish, return_recv_hook)
        handle = (packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, x.size(1), num_experts)
        tensors_to_record = (x, topk_idx, packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info,
                             packed_recv_layout_range, cumulative_local_expert_recv_stats)
        return (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x, packed_recv_count, handle, \
            EventOverlap(event, tensors_to_record if async_finish else None), hook

    # noinspection PyTypeChecker
    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: tuple, use_logfmt: bool = False, zero_copy: bool = False, async_finish: bool = False,
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None,
                            combine_wait_recv_cost_stats: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            topk_idx: `[num_combined_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by the dispatched tokens. `-1` indices (not selecting any expert) are supported. Note that,
                `num_combined_tokens` equals to the number of dispatched tokens.
            topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            use_logfmt: whether to use an internal "LogFMT with dynamic per-64-channel cast" format (10 bits).
            zero_copy: whether the tensor is already copied into the RDMA buffer, should be cooperative
                with `get_next_low_latency_combine_buffer`.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it directly.
            combine_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                This is useful for detecting and pre-cisely localizing slow anomalies.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle
        assert self.nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2
        combined_x, event, hook = self.runtime.low_latency_combine(x, topk_idx, topk_weights, src_info, layout_range,
                                                                   combine_wait_recv_cost_stats, num_max_dispatch_tokens_per_rank,
                                                                   num_experts, use_logfmt, zero_copy, async_finish, return_recv_hook, out)
        tensors_to_record = (x, topk_idx, topk_weights, src_info, layout_range, combined_x)
        return combined_x, EventOverlap(event, tensors_to_record if async_finish else None), hook

    def low_latency_update_mask_buffer(self, rank_to_mask: int, mask: bool = False):
        """
        Mask (unmask) a rank during communication (dispatch, combine, and clean)

        Arguments:
            rank: the rank to mask (unmask).
            mask: if True, will mask the rank (do not recvfrom/sendto the rank), otherwise will unmask the rank.

        """
        self.runtime.low_latency_update_mask_buffer(rank_to_mask, mask)

    def low_latency_query_mask_buffer(self, mask_status: torch.Tensor):
        """
        Query the mask status of all ranks

        Arguments:
            mask_status: `[num_ranks]` with `torch.int`, the mask status of each rank. `1` means mask and `0` means unmasked.

        """
        self.runtime.low_latency_query_mask_buffer(mask_status)

    def low_latency_clean_mask_buffer(self):
        """
        Clean the mask buffer

        """
        self.runtime.low_latency_clean_mask_buffer()

    def get_next_low_latency_combine_buffer(self, handle: object):
        """
        Get the raw registered RDMA buffer tensor for next low-latency combine, so that the next combine kernel can skip the copying.

        Arguments:
            handle: the communication handle given by the `dispatch` function.

        Returns:
            buffer: the raw RDMA low-latency buffer as a BF16 PyTorch tensor with shape
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`, you should fill this buffer
                by yourself.
        """
        src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle
        return self.runtime.get_next_low_latency_combine_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)
