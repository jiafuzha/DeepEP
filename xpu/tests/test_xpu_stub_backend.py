import torch
import pytest


def test_stub_buffer_metadata_and_handles(xpu_stub_backend_module):
    backend = xpu_stub_backend_module
    buf = backend.Buffer(9, 16, 128, 256, False, False, False, False)

    assert buf.get_num_rdma_ranks() == 2
    assert buf.get_rdma_rank() == 1
    assert buf.get_root_rdma_rank(True) == 8
    assert buf.get_root_rdma_rank(False) == 0

    handle = buf.get_local_ipc_handle()
    assert isinstance(handle, (bytes, bytearray))
    assert len(handle) == 72

    uid = buf.get_local_nvshmem_unique_id()
    assert isinstance(uid, bytes)
    assert len(uid) > 0


def test_stub_buffer_tensor_views_and_sync_destroy(xpu_stub_backend_module):
    backend = xpu_stub_backend_module
    buf = backend.Buffer(0, 1, 128, 128, False, False, False, False)

    assert not buf.is_available()
    device_id = buf.get_local_device_id()
    local_handle = buf.get_local_ipc_handle()
    buf.sync([device_id], [local_handle], None)
    assert buf.is_available()

    full = buf.get_local_buffer_tensor(torch.float32, 0, False)
    assert full.numel() == 32

    sliced = buf.get_local_buffer_tensor(torch.float32, 16, False)
    assert sliced.numel() == 28

    rdma = buf.get_local_buffer_tensor(torch.uint8, 8, True)
    assert rdma.numel() == 120

    buf.destroy()
    assert not buf.is_available()


def test_stub_stream_and_size_hint_api(xpu_stub_backend_module):
    backend = xpu_stub_backend_module
    buf = backend.Buffer(0, 1, 0, 0, False, False, False, False)

    stream = buf.get_comm_stream()
    # Stream may be None on CPU-only environments; API should still be callable.
    assert stream is None or hasattr(stream, "device")

    hint_small = backend.get_low_latency_rdma_size_hint(16, 1024, 4, 8)
    hint_large = backend.get_low_latency_rdma_size_hint(32, 4096, 8, 16)
    assert hint_small > 0
    assert hint_large >= hint_small


def test_stub_internode_requires_rdma_bytes(xpu_stub_backend_module):
    backend = xpu_stub_backend_module
    is_native_backend = getattr(backend, '__file__', '').endswith(('.so', '.pyd'))
    if is_native_backend:
        pytest.skip('pure-Python stub backend is not active')

    buf = backend.Buffer(9, 16, 128, 0, False, False, False, False)

    buf.sync([], [], None)
    assert not buf.is_internode_available()
    buf.destroy()
