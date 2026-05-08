import torch
import pytest

import xpu.deep_ep_cpp_xpu as backend


def test_stub_buffer_metadata_and_handles():
    buf = backend.Buffer(
        9,      # rank
        16,     # num_ranks
        128,    # num_nvl_bytes
        256,    # num_rdma_bytes
        False,  # low_latency_mode
        False,  # explicitly_destroy
        False,  # enable_shrink
        False,  # use_fabric
    )

    assert buf.get_num_rdma_ranks() == 2
    assert buf.get_rdma_rank() == 1
    assert buf.get_root_rdma_rank(global_rank=True) == 8
    assert buf.get_root_rdma_rank(global_rank=False) == 0

    handle = buf.get_local_ipc_handle()
    assert isinstance(handle, bytes)
    assert len(handle) == 72

    uid = buf.get_local_nvshmem_unique_id()
    assert isinstance(uid, bytes)
    assert len(uid) > 0


def test_stub_buffer_tensor_views_and_sync_destroy():
    buf = backend.Buffer(
        0,      # rank
        1,      # num_ranks
        64,     # num_nvl_bytes
        64,     # num_rdma_bytes
        False,  # low_latency_mode
        False,  # explicitly_destroy
        False,  # enable_shrink
        False,  # use_fabric
    )

    assert not buf.is_available()
    buf.sync([], [], None)
    assert buf.is_available()

    full = buf.get_local_buffer_tensor(torch.float32, offset=0, use_rdma_buffer=False)
    assert full.numel() == 16

    sliced = buf.get_local_buffer_tensor(torch.float32, offset=16, use_rdma_buffer=False)
    assert sliced.numel() == 12

    rdma = buf.get_local_buffer_tensor(torch.uint8, offset=8, use_rdma_buffer=True)
    assert rdma.numel() == 56

    buf.destroy()
    assert not buf.is_available()


def test_stub_stream_and_size_hint_api():
    buf = backend.Buffer(
        0,      # rank
        1,      # num_ranks
        0,      # num_nvl_bytes
        0,      # num_rdma_bytes
        False,  # low_latency_mode
        False,  # explicitly_destroy
        False,  # enable_shrink
        False,  # use_fabric
    )

    stream = buf.get_comm_stream()
    # Stream may be None on CPU-only environments; API should still be callable.
    assert stream is None or hasattr(stream, "device")

    hint = backend.get_low_latency_rdma_size_hint(32, 4096, 8, 16)
    assert hint == 32 * 4096 * 2


def test_stub_internode_requires_rdma_bytes():
    is_native_backend = getattr(backend, '__file__', '').endswith(('.so', '.pyd'))
    if is_native_backend:
        pytest.skip('pure-Python stub backend is not active')

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        _ = torch.empty((1,), device='xpu')
    buf = backend.Buffer(9, 16, 128, 0, False, False, False, False)

    buf.sync([], [], None)
    assert not buf.is_internode_available()
    buf.destroy()
