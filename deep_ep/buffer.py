import os
import socket
import struct
import tempfile
import time
import glob as _glob
import signal as _signal
import weakref as _weakref
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


_ORPHAN_REAP_DONE = False

# --- Abnormal-exit GPU/NIC drain on external termination signals ---------------
#
# A run can die from an external SIGTERM/SIGINT (an orchestrator killing a slow
# job, ``timeout(1)``, ``docker stop``, Ctrl-C) that does NOT flow through the
# caller's try/finally teardown. On the Intel BMG + mlx5/IBGDA stack, if the
# process dies while a long-running IBGDA poll/quiet exec queue is still merely
# *submitted* on the GT, the Xe GuC cannot preempt it at teardown -> GT reset ->
# the next run inherits a wedged GT (init hang / DEVICE_LOST). Draining the GPU
# streams (retiring that exec queue) before exit prevents this.
#
# CRITICAL DESIGN NOTE: the drain MUST run from a NORMAL Python context, never
# from a C signal handler. A C ``sigaction`` handler that calls into SYCL/L0 to
# drain DEADLOCKS, because the signal almost always interrupts the process while
# it is inside a driver call holding an internal lock, and the drain re-enters the
# same locked runtime. A Python ``signal`` handler runs between bytecodes -- i.e.
# only AFTER the interrupted native call has returned -- so re-entering the driver
# to drain is safe. We therefore reuse the proven ``Buffer.quiesce()`` here.
#
# SIGKILL/SIGSEGV cannot be handled this way (uncatchable / not deliverable to
# Python); those rely on the startup orphan reaper plus the optional driver reset.
# Opt out with ``DEEP_EP_XPU_SIGNAL_CLEANUP=0``.
_LIVE_BUFFERS = _weakref.WeakSet()
_SIGNAL_CLEANUP_INSTALLED = False
_PREV_SIGNAL_HANDLERS = {}


def _signal_cleanup_handler(signum, frame):
    # Runs in normal Python context (between bytecodes) -> safe to call into the
    # driver. Drain every live buffer's GPU/NIC before the process terminates.
    print(f'[DeepEP] signal {signum}: draining GPU/NIC of {len(_LIVE_BUFFERS)} '
          f'live buffer(s) before exit', flush=True)
    for buf in list(_LIVE_BUFFERS):
        with suppress(Exception):
            buf.quiesce()
    print(f'[DeepEP] signal {signum}: drain done, chaining to previous handler', flush=True)
    # Chain to the previous disposition so the exit code / KeyboardInterrupt
    # semantics are preserved.
    prev = _PREV_SIGNAL_HANDLERS.get(signum, _signal.SIG_DFL)
    if callable(prev):
        prev(signum, frame)
        return
    if prev == _signal.SIG_IGN:
        return
    # SIG_DFL: restore default and re-raise so the process terminates normally.
    with suppress(Exception):
        _signal.signal(signum, _signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_signal_cleanup():
    global _SIGNAL_CLEANUP_INSTALLED
    if _SIGNAL_CLEANUP_INSTALLED:
        return
    _SIGNAL_CLEANUP_INSTALLED = True
    if os.environ.get('DEEP_EP_XPU_SIGNAL_CLEANUP', '1') == '0':
        return
    armed = []
    for signum in (_signal.SIGTERM, _signal.SIGINT):
        try:
            _PREV_SIGNAL_HANDLERS[signum] = _signal.getsignal(signum)
            _signal.signal(signum, _signal_cleanup_handler)
            armed.append(int(signum))
        except (ValueError, OSError):
            # signal.signal() only works on the main thread; skip otherwise.
            _PREV_SIGNAL_HANDLERS.pop(signum, None)
    print(f'[DeepEP] signal cleanup armed for {armed} (pid {os.getpid()})', flush=True)


def _reap_orphan_xpu_ipc():
    """Best-effort removal of PID-tagged IPC sockets left behind by DEAD runs.

    A run killed by an uncatchable SIGKILL (``docker rm -f``, ``timeout -s KILL``)
    or a hard crash cannot run any in-process cleanup, so its PID-tagged UNIX
    domain sockets survive. Accumulated stale IPC state is a prime trigger for the
    next run's init hang / DEVICE_LOST cascade on the Intel BMG + mlx5/IBGDA stack.

    This reaps ONLY files whose embedded PID is provably dead, so it never touches
    a concurrently-live run's resources (multiple ranks on the same node share
    /tmp). Non-PID-tagged shared memory (PSM3/oneCCL/gloo sems in /dev/shm) is left
    to the external node reset ritual. Runs once per process; opt out with
    ``DEEP_EP_XPU_REAP_ORPHANS=0``.
    """
    global _ORPHAN_REAP_DONE
    if _ORPHAN_REAP_DONE:
        return
    _ORPHAN_REAP_DONE = True
    if os.environ.get('DEEP_EP_XPU_REAP_ORPHANS', '1') == '0':
        return

    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but owned by another user
        except OSError:
            return True  # be conservative: assume alive
        return True

    tmp = tempfile.gettempdir()
    # (glob pattern, function extracting the owner PID from the basename)
    reapers = [
        # deep_ep_xpu_ipc_<pid>_<rank>_<id>.sock  (created in Buffer._exchange_xpu_ipc_fds)
        (os.path.join(tmp, 'deep_ep_xpu_ipc_*.sock'),
         lambda base: base[len('deep_ep_xpu_ipc_'):].split('_')[0]),
        # ishmem-ipc-fd-sock-<pid>:<pe>  (created in ishmem_ibgda/src/ipc.cpp)
        (os.path.join(tmp, 'ishmem-ipc-fd-sock-*'),
         lambda base: base[len('ishmem-ipc-fd-sock-'):].split(':')[0]),
    ]
    my_pid = os.getpid()
    for pattern, pid_of in reapers:
        for path in _glob.glob(pattern):
            try:
                pid = int(pid_of(os.path.basename(path)))
            except (ValueError, IndexError):
                continue
            if pid == my_pid or _pid_alive(pid):
                continue
            with suppress(OSError):
                os.unlink(path)


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
        self._xpu_internode_handle_cache = {}
        self._xpu_low_latency_handle_cache = {}
        self._xpu_low_latency_combine_buffer_cache = {}
        # Persistent, shape-stable, REUSED device buffers for the LL comm path so
        # torch's XPU caching allocator never grows a NEW segment (a zeMemAllocDevice
        # VM_BIND) during dispatch/combine. A fresh VM_BIND concurrent with the bcs
        # copy engine is the necessary trigger for the residual transient spurious
        # OUT_OF_DEVICE_MEMORY (err-39) / bcs+VM-worker-EBUSY wedge. Keyed by
        # (role, shape, dtype); dispatch outputs use a 2-slot ring (ping-pong) to
        # honour the "cannot hold more than 2 low-latency results at once" contract.
        self._xpu_ll_persist = {}
        self._xpu_low_latency_mask_status = None
        if enable_shrink:
            self._xpu_low_latency_mask_status = torch.zeros((self.group_size, ), dtype=torch.int32, device='xpu')
        # Reap PID-dead orphaned IPC sockets from previously SIGKILLed/crashed runs
        # BEFORE the C++ runtime re-creates iSHMEM IPC sockets under the same /tmp
        # namespace, so accumulated stale state cannot wedge this run's init.
        _reap_orphan_xpu_ipc()
        # Auto-tune the IBGDA multi-QP count for the XPU low-latency path. Both the LL
        # dispatch and combine send kernels key the destination QP by the LOCAL expert
        # index `le` (qp_idx = le & (qps_per_pe - 1) inside iSHMEM), so exposing one QP
        # per local expert lets the NIC drive each expert's RDMA on an independent QP
        # instead of serializing every expert through QP 0 (the old QPS_PER_PE=1 default).
        # Measured ~27-30% lower LL dispatch+combine latency (2.07 -> ~2.8 GB/s) across
        # 512..4096 tokens with no correctness regression once the combine per-QP
        # producer race was fixed. iSHMEM rounds this to a power of 2 and clamps [1, 16];
        # a user-provided ISHMEM_IBGDA_QPS_PER_PE always wins (setdefault).
        if low_latency_mode:
            qpp = max(1, min(int(num_qps_per_rank), 16))
            qpp = 1 << (qpp - 1).bit_length() if qpp > 1 else 1
            os.environ.setdefault('ISHMEM_IBGDA_QPS_PER_PE', str(qpp))
        elif num_rdma_bytes > 0:
            # Normal (high-throughput) internode: SINGLE QP by default.
            #
            # The CUDA-faithful design stripes the RDMA payload put across one RC QP per
            # channel (qp_id == channel, mirroring internode.cu:818/835). On this BMG +
            # mlx5/IBGDA stack that striping is a NET LOSS: the path is NIC-latency-bound,
            # not bandwidth-bound, and every extra QP channel adds a full extra NIC
            # round-trip on the critical path (per-QP ishmemx_fence_qp + tail
            # ishmemx_long_atomic_add_qp on the sender, plus one more flag the receiver
            # must poll before it may read the payload). Cost scales with C; bandwidth
            # does not improve.
            #
            # Measured (H=7168, TOPK=2, EXPERTS=8, DB=8, 4 ranks / 2 nodes; round_trip
            # min, the least jitter-contaminated metric) — monotonic in C at every size:
            #
            #     NT      C=1        C=2        C=4        C=8       C=16
            #     32    3875.7*    3885.6     3983.8     4183.3     4667.1
            #     64    6845.7*    6846.3     6963.1     7165.2     7622.4
            #    128   12854.1*   12912.7    13021.5    13367.9    14129.3
            #    512   51137.7    51042.6*   51060.6    51649.2    51475.4
            #   1024  101163.6*  101516.4   101453.2   101553.1   101809.9
            #
            # C=1 vs the previous C=8 default: -7.3% round-trip at NT=32 (307 us, ~45x the
            # 6.8 us run-to-run stdev), -4.5% at NT=64, -3.8% at NT=128. At NT>=2048 the
            # two are equal on min round-trip but C=1 has far lower tail jitter (NT=2048
            # max 207 ms vs 310 ms). A 3x repeat at NT=32 separates C=1 (3855.3 +/- 6.8)
            # from C=2 (3900.5 +/- 4.7) by 6.7 sigma, so C=1 is the optimum, not merely
            # noise-equivalent.
            #
            # The kernel derives its QP-channel count C from this same env
            # (internode_num_qp_channels()), so C=1 also collapses the per-QP flag fan-out
            # to a single flag per destination. setdefault => an explicit user/harness
            # ISHMEM_IBGDA_QPS_PER_PE always wins, so multi-QP striping stays available for
            # stacks where the NIC is the bottleneck rather than the round-trip latency.
            #
            # 2026-08 CORRECTNESS OVERRIDE (this branch only - the low-latency branch above
            # is untouched, C=1 stays load-bearing there).  The C=1 study above is a PERF
            # study and it is still valid on perf; but C=1 is NOT SAFE for the fused
            # internode-normal kernels.  With one QP per PE every channel's RDMA sender
            # drives the SAME send queue, and once ~12 channels share it the run
            # intermittently hangs / silently drops whole tokens (measured 4/6 hangs at
            # num_sms=24, 2048 tok, hidden 7168; 6 channels/QP 0/6; 4 channels/QP 0/16).
            # One QP per channel removes it: num_sms=24 + QPS_PER_PE=16 is 16/16 @2048 and
            # 8/8 @4096 (hidden 7168) against a ~60% base failure rate, AND it is faster
            # (round-trip 30.6 ms vs the grid-clamped 53.7 ms at 2048 tok) because the
            # clamp that C=1 forces costs far more than the extra per-QP flags.
            # See .github/agents/cuda-to-xpu-internode-normal-migration.agent.md 24.2.1.
            # `setdefault` semantics preserved: an explicit user/harness value still wins.
            qpp = max(1, min(int(num_qps_per_rank), 16))
            qpp = 1 << (qpp - 1).bit_length() if qpp > 1 else 1
            os.environ.setdefault('ISHMEM_IBGDA_QPS_PER_PE', str(qpp))
        self.runtime = deep_ep_cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode, explicitly_destroy,
                                          enable_shrink, use_fabric)
        # Register for abnormal-exit GPU/NIC drain on external SIGTERM/SIGINT, so a
        # killed run retires its long-running IBGDA exec queue instead of leaving it
        # submitted (which would GT-reset and wedge the next run). See the module
        # docstring near _signal_cleanup_handler for why this must be a PYTHON
        # signal handler (normal context), not a C sigaction handler (deadlocks).
        _LIVE_BUFFERS.add(self)
        _install_signal_cleanup()

        # Synchronize device IDs
        local_device_id = self.runtime.get_local_device_id()
        device_ids = all_gather_object(local_device_id)

        # Synchronize IPC handles
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        ipc_handles = all_gather_object(local_ipc_handle)
        if num_nvl_bytes > 0 and self.group_size > 1:
            ipc_handles = self._exchange_xpu_ipc_fds(ipc_handles, all_gather_object)

        # Synchronize iSHMEM unique IDs
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            master_port = os.environ.get('MASTER_PORT', '')
            if master_port and os.environ.get('I_MPI_MPCP_SERVER_PORT', '') == master_port:
                os.environ['I_MPI_MPCP_SERVER_PORT'] = str(int(master_port) + 1)
            self.nvshmem_qp_depth = max(int(os.environ.get('ISHMEM_QP_DEPTH', '1024')), (num_qps_per_rank + 1) * 2)

            # Synchronize using the root ID — only one rank obtains the unique ID
            if self.rank == 0:
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            nvshmem_unique_ids = all_gather_object(root_unique_id)
            root_unique_id = nvshmem_unique_ids[0]

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

        num_nvl_ranks = self.runtime.get_num_nvl_ranks() if hasattr(self.runtime, 'get_num_nvl_ranks') else min(self.group_size, 8)
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

    def quiesce(self):
        """
        Lightweight per-process GPU + NIC quiesce, safe to call right before a hard
        process exit (e.g. os._exit on the XPU LL direct-doorbell path) that bypasses
        destroy(). Drains all in-flight GPU work -- including the long-running IBGDA
        poll/quiet exec queues -- and this PE's outbound RDMA, WITHOUT tearing down the
        symmetric heap / QPs / NIC BAR. This lets the GuC retire the long-running exec
        queue cleanly at exit instead of GT-resetting it (which would wedge the next
        process's first GPU submission -> init hang / DEVICE_LOST).
        """
        if self.runtime is not None:
            self.runtime.quiesce()

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
        Capture an event on the current stream, i.e. `torch.xpu.current_stream()`.

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
        return torch.xpu.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)

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
        if self.runtime.get_num_rdma_ranks() > 1 or (self.num_nvl_bytes > 0 and self.runtime.get_num_rdma_ranks() == 1
                                                     and self.group_size > 1):
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
        if self.runtime.get_num_rdma_ranks() > 1 or (self.num_nvl_bytes > 0 and self.runtime.get_num_rdma_ranks() == 1
                                                     and self.group_size > 1):
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

    @staticmethod
    def _xpu_per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2
        m, n = x.shape
        aligned_n = ((n + 127) // 128) * 128
        x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode='constant', value=0)
        x_view = x_padded.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        x_fp8 = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, aligned_n)[:, :n].contiguous()
        return x_fp8, (x_amax / 448.0).contiguous()

    @staticmethod
    def _xpu_noop_hook() -> None:
        return None

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
        if os.environ.get('DEEP_EP_XPU_INTERNODE_HOST_FALLBACK') == '1':
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
        if os.environ.get('DEEP_EP_XPU_INTERNODE_HOST_FALLBACK') == '1':
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
                                                                                  gbl_rank_prefix_sum,
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
        self._xpu_low_latency_handle_cache.clear()
        self._xpu_low_latency_combine_buffer_cache.clear()
        if self._xpu_low_latency_mask_status is not None:
            gathered = self._xpu_all_gather_tensor(self._xpu_low_latency_mask_status)
            self._xpu_low_latency_mask_status.copy_(torch.stack(gathered, dim=0).amax(dim=0))
        self.runtime.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)
        if self._xpu_low_latency_mask_status is not None:
            for rank_to_mask, value in enumerate(self._xpu_low_latency_mask_status.cpu().tolist()):
                if value:
                    self.runtime.low_latency_update_mask_buffer(rank_to_mask, True)

    def _xpu_alloc_retry(self, thunk: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Transient-retry safety net for the LL comm path.

        A transiently-wedged bcs (blitter/copy) engine can make a device allocation's
        VM_BIND fail with EBUSY, surfaced by the L0/UR stack as
        UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY (err-39) even though VRAM is free. The
        wedge is self-recovering, so synchronize + short exponential backoff + retry
        turns a spurious hard failure into a small stall. Non-transient OOM (genuine
        exhaustion) is re-raised immediately on the last attempt.
        """
        retries = int(os.environ.get('DEEP_EP_LL_ALLOC_RETRIES', '8'))
        delay = 0.001
        last = None
        for attempt in range(retries + 1):
            try:
                return thunk()
            except RuntimeError as e:
                s = str(e)
                sl = s.lower()
                transient = ('out_of_device_memory' in sl or 'out of memory' in sl
                             or 'error 39' in sl or 'error: 39' in sl
                             or 'device_lost' in sl)
                if not transient or attempt == retries:
                    raise
                last = e
                try:
                    torch.xpu.synchronize()
                except Exception:
                    pass
                time.sleep(delay)
                delay = min(delay * 2, 0.005)
        raise last  # pragma: no cover

    def _xpu_ll_buf(self, role: str, shape, dtype: torch.dtype, device,
                    fill=None, slots: int = 1) -> torch.Tensor:
        """Return a persistent, REUSED device buffer keyed by (role, shape, dtype).

        With DEEP_EP_LL_PERSIST_BUFFERS on (default), the buffer is allocated once
        (via the transient-retry net) and thereafter reused in-place, keeping peak
        memory flat so the caching allocator never grows a new segment during comm.
        `slots`=2 gives a ping-pong ring so two back-to-back dispatch results don't
        alias. `fill` (0 -> zero_, other -> fill_) reproduces the current init.
        """
        shape = tuple(int(s) for s in shape)
        if os.environ.get('DEEP_EP_LL_PERSIST_BUFFERS', '1') != '1':
            t = self._xpu_alloc_retry(lambda: torch.empty(shape, dtype=dtype, device=device))
            if fill == 0:
                t.zero_()
            elif fill is not None:
                t.fill_(fill)
            return t
        key = (role, shape, dtype, str(device))
        entry = self._xpu_ll_persist.get(key)
        if entry is None:
            entry = {'bufs': [None] * slots, 'idx': 0}
            self._xpu_ll_persist[key] = entry
        idx = entry['idx']
        buf = entry['bufs'][idx]
        if buf is None:
            buf = self._xpu_alloc_retry(lambda: torch.empty(shape, dtype=dtype, device=device))
            entry['bufs'][idx] = buf
        entry['idx'] = (idx + 1) % slots
        if fill == 0:
            buf.zero_()
        elif fill is not None:
            buf.fill_(fill)
        return buf

    def _xpu_low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                                  num_max_dispatch_tokens_per_rank: int, num_experts: int,
                                  cumulative_local_expert_recv_stats: Optional[torch.Tensor],
                                  use_fp8: bool, async_finish: bool, return_recv_hook: bool) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        assert not (async_finish and return_recv_hook)
        assert num_experts % self.group_size == 0
        num_tokens, hidden = x.shape
        num_local_experts = num_experts // self.group_size
        num_slots = self.group_size * num_max_dispatch_tokens_per_rank
        device = x.device
        mask_status = self._xpu_low_latency_mask_status
        active_mask = None if mask_status is None else mask_status == 0

        gathered_x = self._xpu_alloc_retry(lambda: self._xpu_all_gather_tensor(x))
        gathered_topk_idx = self._xpu_alloc_retry(lambda: self._xpu_all_gather_tensor(topk_idx))
        # Persistent 2-slot ring for the recurring dispatch outputs (ping-pong keeps
        # two back-to-back results non-aliasing) so no fresh VM_BIND on the hot path.
        packed_bf16 = self._xpu_ll_buf('packed_bf16', (num_local_experts, num_slots, hidden),
                                       torch.bfloat16, device, fill=0, slots=2)
        packed_recv_src_info = self._xpu_ll_buf('packed_recv_src_info', (num_local_experts, num_slots),
                                                torch.int32, device, fill=-1, slots=2)
        packed_recv_layout_range = self._xpu_ll_buf('packed_recv_layout_range', (num_local_experts, self.group_size),
                                                    torch.int64, device, fill=0, slots=2)
        packed_recv_count = self._xpu_ll_buf('packed_recv_count', (num_local_experts, ),
                                             torch.int32, device, fill=0, slots=2)

        # ---- Token dedup: if a token's two experts are on the same rank, only send once ----
        # The second expert gets the data via a local GPU copy after the main scatter.
        # Dedup entries: list of (dst_local_expert, src_local_expert, src_position_in_packed, token_idx)
        dedup_entries = []

        handle_entries = []
        local_expert_begin = self.rank * num_local_experts
        int_mask = (1 << 32) - 1
        for local_expert in range(num_local_experts):
            global_expert = local_expert_begin + local_expert
            write_offset = 0
            for src_rank in range(self.group_size):
                if active_mask is not None and not bool(active_mask[src_rank].item()):
                    continue
                src_topk = gathered_topk_idx[src_rank]
                selected = (src_topk == global_expert).any(dim=1)
                token_indices = torch.nonzero(selected, as_tuple=False).flatten()
                count = int(token_indices.numel())
                begin = write_offset
                if count > 0:
                    end = begin + count
                    if end > num_slots:
                        raise RuntimeError('XPU low-latency dispatch receive buffer is too small')
                    packed_bf16[local_expert, begin:end].copy_(gathered_x[src_rank].index_select(0, token_indices))
                    packed_recv_src_info[local_expert, begin:end].copy_(token_indices.to(torch.int32))
                    handle_entries.append((local_expert, global_expert, src_rank, begin, count, token_indices.to(torch.int64)))
                    write_offset = end
                packed_recv_layout_range[local_expert, src_rank] = (begin << 32) | (count & int_mask)
            packed_recv_count[local_expert] = write_offset

        # ---- Post-scatter dedup: copy tokens between local experts on the same rank ----
        # For each token where both top-k experts are on this rank's local experts,
        # the data was only written to the first expert. Copy it to the second.
        local_global_experts = [local_expert_begin + i for i in range(num_local_experts)]
        for src_rank in range(self.group_size):
            if active_mask is not None and not bool(active_mask[src_rank].item()):
                continue
            src_topk = gathered_topk_idx[src_rank]
            for token_i in range(num_tokens):
                experts_for_token = src_topk[token_i].tolist()
                # Find which of this token's experts are local
                local_experts_for_token = [e for e in experts_for_token
                                           if e >= 0 and e in local_global_experts]
                if len(local_experts_for_token) >= 2:
                    # Token maps to 2+ local experts. Data was written to the first one.
                    # Copy from first to the other(s).
                    first_global = local_experts_for_token[0]
                    first_local = first_global - local_expert_begin
                    # Find position of this token in first_local's packed data
                    first_pos = None
                    for le, ge, sr, b, c, tids in handle_entries:
                        if le == first_local and sr == src_rank:
                            # Find the position of token_i in this entry
                            mask = tids == token_i
                            if mask.any():
                                pos = int(b + mask.nonzero(as_tuple=False)[0].item())
                                first_pos = pos
                                break
                    if first_pos is not None:
                        for dup_global in local_experts_for_token[1:]:
                            dup_local = dup_global - local_expert_begin
                            # Allocate slot for dup_local
                            dup_pos = int(packed_recv_count[dup_local].item())
                            if dup_pos >= num_slots:
                                raise RuntimeError('XPU low-latency dispatch receive buffer too small for dedup')
                            # Copy data
                            packed_bf16[dup_local, dup_pos].copy_(
                                packed_bf16[first_local, first_pos])
                            packed_recv_src_info[dup_local, dup_pos] = token_i
                            # Update layout_range: extend the last src_rank entry
                            prev = packed_recv_layout_range[dup_local, src_rank].item()
                            prev_count = int(prev & int_mask)
                            prev_begin = int(prev >> 32)
                            packed_recv_layout_range[dup_local, src_rank] = ((prev_begin << 32) |
                                                                             ((prev_count + 1) & int_mask))
                            packed_recv_count[dup_local] = dup_pos + 1
                            # Register as handle entry for combine path
                            dedup_entries.append(
                                (dup_local, dup_global, src_rank, dup_pos, 1,
                                 torch.tensor([token_i], dtype=torch.int64, device=device)))

        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.add_(packed_recv_count)

        if use_fp8:
            flat_fp8, flat_scales = self._xpu_per_token_cast_to_fp8(packed_bf16.view(-1, hidden))
            # Copy into persistent buffers so the returned payload holds no freshly
            # VM_BIND'd segment; the cast temporaries are same-size each call and are
            # served from the caching allocator's freelist (no new segment growth).
            persist_fp8 = self._xpu_ll_buf('flat_fp8', flat_fp8.shape, flat_fp8.dtype, device, slots=2)
            persist_scales = self._xpu_ll_buf('flat_scales', flat_scales.shape, flat_scales.dtype, device, slots=2)
            persist_fp8.copy_(flat_fp8)
            persist_scales.copy_(flat_scales)
            packed_recv_x = persist_fp8.view(num_local_experts, num_slots, hidden)
            packed_recv_x_scales = persist_scales.view(num_local_experts, num_slots, -1)
            recv_payload = (packed_recv_x, packed_recv_x_scales)
        else:
            packed_recv_x = packed_bf16
            packed_recv_x_scales = None
            recv_payload = packed_recv_x

        handle = (packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts)
        self._xpu_low_latency_handle_cache[id(handle)] = {
            'entries': handle_entries,
            'dedup_entries': dedup_entries if dedup_entries else [],
            'num_tokens': num_tokens,
            'num_local_experts': num_local_experts,
        }
        return recv_payload, packed_recv_count, handle, self._xpu_internode_event(
            async_finish), self._xpu_noop_hook if return_recv_hook else None

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

    def _xpu_low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor, handle: tuple,
                                 async_finish: bool, return_recv_hook: bool,
                                 out: Optional[torch.Tensor]) -> Tuple[torch.Tensor, EventOverlap, Callable]:
        assert not (async_finish and return_recv_hook)
        cache = self._xpu_low_latency_handle_cache.get(id(handle))
        if cache is None:
            raise RuntimeError('missing XPU low-latency combine metadata')
        _, _, _, hidden, num_experts = handle
        num_combined_tokens = topk_idx.size(0)
        contributions = []
        for local_expert, global_expert, src_rank, begin, count, token_indices in cache['entries']:
            if count == 0:
                continue
            values = x[local_expert, begin:begin + count].detach().cpu()
            contributions.append((global_expert, src_rank, token_indices.cpu(), values))
        # Also include dedup entries (tokens replicated locally between experts)
        for local_expert, global_expert, src_rank, begin, count, token_indices in cache.get('dedup_entries', []):
            if count == 0:
                continue
            values = x[local_expert, begin:begin + count].detach().cpu()
            contributions.append((global_expert, src_rank, token_indices.cpu(), values))
        gathered = [None] * self.group_size
        dist.all_gather_object(gathered, contributions, self.group)

        combined_x = out if out is not None else self._xpu_ll_buf(
            'combined_x', (num_combined_tokens, hidden), x.dtype, x.device, slots=1)
        combined_x.zero_()
        for rank_contribs in gathered:
            for global_expert, src_rank, token_indices_cpu, values_cpu in rank_contribs:
                if src_rank != self.rank:
                    continue
                token_indices = token_indices_cpu.to(device=x.device, dtype=torch.long)
                values = values_cpu.to(device=x.device, dtype=x.dtype)
                weight_mask = topk_idx.index_select(0, token_indices) == global_expert
                weights = topk_weights.index_select(0, token_indices).masked_fill(~weight_mask, 0).sum(dim=1).to(x.dtype)
                combined_x.index_add_(0, token_indices, values * weights.view(-1, 1))
        return combined_x, self._xpu_internode_event(async_finish), self._xpu_noop_hook if return_recv_hook else None

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
        if id(handle) not in self._xpu_low_latency_handle_cache:
            src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle
            combined_x, event, hook = self.runtime.low_latency_combine(x, topk_idx, topk_weights, src_info, layout_range,
                                                                       combine_wait_recv_cost_stats, num_max_dispatch_tokens_per_rank,
                                                                       num_experts, use_logfmt, zero_copy, async_finish,
                                                                       return_recv_hook, out)
            tensors_to_record = (x, topk_idx, topk_weights, src_info, layout_range, combined_x)
            return combined_x, EventOverlap(event, tensors_to_record if async_finish else None), hook
        combined_x, event, hook = self._xpu_low_latency_combine(x, topk_idx, topk_weights, handle, async_finish, return_recv_hook, out)
        tensors_to_record = (x, topk_idx, topk_weights, combined_x)
        return combined_x, EventOverlap(event.event, tensors_to_record if async_finish else None), hook

    def low_latency_update_mask_buffer(self, rank_to_mask: int, mask: bool = False):
        """
        Mask (unmask) a rank during communication (dispatch, combine, and clean)

        Arguments:
            rank: the rank to mask (unmask).
            mask: if True, will mask the rank (do not recvfrom/sendto the rank), otherwise will unmask the rank.

        """
        if self._xpu_low_latency_mask_status is None:
            raise RuntimeError('XPU low-latency shrink mode is not enabled')
        self._xpu_low_latency_mask_status[rank_to_mask] = 1 if mask else 0

    def low_latency_query_mask_buffer(self, mask_status: torch.Tensor):
        """
        Query the mask status of all ranks

        Arguments:
            mask_status: `[num_ranks]` with `torch.int`, the mask status of each rank. `1` means mask and `0` means unmasked.

        """
        if self._xpu_low_latency_mask_status is None:
            raise RuntimeError('XPU low-latency shrink mode is not enabled')
        gathered = self._xpu_all_gather_tensor(self._xpu_low_latency_mask_status)
        self._xpu_low_latency_mask_status.copy_(torch.stack(gathered, dim=0).amax(dim=0))
        mask_status.copy_(self._xpu_low_latency_mask_status.to(mask_status.device))

    def low_latency_clean_mask_buffer(self):
        """
        Clean the mask buffer

        """
        if self._xpu_low_latency_mask_status is None:
            raise RuntimeError('XPU low-latency shrink mode is not enabled')
        self._xpu_low_latency_mask_status.zero_()

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
        shape = (num_experts // self.group_size, self.group_size * num_max_dispatch_tokens_per_rank, hidden)
        buffer = torch.empty(shape, dtype=torch.bfloat16, device=src_info.device)
        self._xpu_low_latency_combine_buffer_cache[id(handle)] = buffer
        return buffer
