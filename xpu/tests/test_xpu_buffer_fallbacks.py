import torch

from xpu.deep_ep.buffer import Buffer, Config


class _RaisingInternodeRuntime:
    def internode_dispatch(self, *args, **kwargs):
        raise RuntimeError('internode_dispatch not implemented for XPU yet')

    def internode_combine(self, *args, **kwargs):
        raise RuntimeError('internode_combine not implemented for XPU yet')


class _RaisingLowLatencyRuntime:
    def low_latency_dispatch(self, *args, **kwargs):
        raise RuntimeError('low_latency_dispatch not implemented for XPU yet')

    def low_latency_combine(self, *args, **kwargs):
        raise RuntimeError('low_latency_combine not implemented for XPU yet')

    def clean_low_latency_buffer(self, *args, **kwargs):
        raise RuntimeError('low_latency clean not implemented for XPU yet')

    def low_latency_update_mask_buffer(self, *args, **kwargs):
        raise RuntimeError('low_latency mask update not implemented for XPU yet')

    def low_latency_query_mask_buffer(self, *args, **kwargs):
        raise RuntimeError('low_latency mask query not implemented for XPU yet')

    def low_latency_clean_mask_buffer(self, *args, **kwargs):
        raise RuntimeError('low_latency mask clean not implemented for XPU yet')

    def get_next_low_latency_combine_buffer(self, *args, **kwargs):
        raise RuntimeError('low_latency next buffer not implemented for XPU yet')


def _make_buffer_for_fallback_tests() -> Buffer:
    buf = object.__new__(Buffer)
    buf.group_size = 1
    buf.runtime = _RaisingInternodeRuntime()
    return buf


def _make_buffer_for_low_latency_fallback_tests() -> Buffer:
    buf = object.__new__(Buffer)
    buf.group_size = 1
    buf.runtime = _RaisingLowLatencyRuntime()
    buf._low_latency_mask_status = torch.zeros((1,), dtype=torch.int)
    return buf


def _config() -> Config:
    return Config(num_sms=20, 
                  num_max_nvl_chunked_send_tokens=6,
                  num_max_nvl_chunked_recv_tokens=256,
                  num_max_rdma_chunked_send_tokens=6,
                  num_max_rdma_chunked_recv_tokens=256)


def _config_fingerprint(config: Config) -> tuple[int, int, int, int]:
    return (
        config.get_nvl_buffer_size_hint(8, 128),
        config.get_nvl_buffer_size_hint(16, 256),
        config.get_rdma_buffer_size_hint(8, 128),
        config.get_rdma_buffer_size_hint(16, 256),
    )


def test_rank_config_lookup_falls_back_to_profiled_neighbors():
    assert _config_fingerprint(Buffer.get_dispatch_config(1)) == _config_fingerprint(Buffer.get_dispatch_config(2))
    assert _config_fingerprint(Buffer.get_dispatch_config(3)) == _config_fingerprint(Buffer.get_dispatch_config(4))
    assert _config_fingerprint(Buffer.get_combine_config(1)) == _config_fingerprint(Buffer.get_combine_config(2))
    assert _config_fingerprint(Buffer.get_combine_config(3)) == _config_fingerprint(Buffer.get_combine_config(4))


def test_internode_dispatch_fallback_single_rank(monkeypatch):
    from xpu.deep_ep import buffer as buffer_mod

    monkeypatch.setattr(buffer_mod, '_has_xpu_runtime', lambda: True)
    buf = _make_buffer_for_fallback_tests()

    x = torch.randn(4, 8, dtype=torch.bfloat16)
    topk_idx = torch.zeros((4, 1), dtype=torch.int64)
    topk_weights = torch.ones((4, 1), dtype=torch.float32)
    num_tokens_per_rank = torch.tensor([4], dtype=torch.int)
    num_tokens_per_rdma_rank = torch.tensor([4], dtype=torch.int)
    is_token_in_rank = torch.ones((4, 1), dtype=torch.bool)
    num_tokens_per_expert = torch.tensor([4], dtype=torch.int)

    recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, event = buf.internode_dispatch(
        x,
        handle=None,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        config=_config(),
    )

    assert torch.equal(recv_x, x)
    assert torch.equal(recv_topk_idx, topk_idx)
    assert torch.equal(recv_topk_weights, topk_weights)
    assert recv_counts == [4]
    assert isinstance(handle, dict)
    assert handle.get('fallback') == 'xpu_internode_dispatch'
    assert event is not None


def test_internode_combine_fallback_single_rank(monkeypatch):
    from xpu.deep_ep import buffer as buffer_mod

    monkeypatch.setattr(buffer_mod, '_has_xpu_runtime', lambda: True)
    buf = _make_buffer_for_fallback_tests()

    x = torch.randn(4, 8, dtype=torch.bfloat16)
    topk_weights = torch.ones((4, 1), dtype=torch.float32)
    bias = torch.full_like(x, 0.5)
    handle = {'fallback': 'xpu_internode_dispatch'}

    combined_x, combined_topk_weights, event = buf.internode_combine(
        x,
        handle=handle,
        topk_weights=topk_weights,
        bias=bias,
        config=_config(),
    )

    assert torch.allclose(combined_x, x + bias)
    assert torch.equal(combined_topk_weights, topk_weights)
    assert event is not None


def test_low_latency_dispatch_fallback_single_rank(monkeypatch):
    from xpu.deep_ep import buffer as buffer_mod

    monkeypatch.setattr(buffer_mod, '_has_xpu_runtime', lambda: True)
    buf = _make_buffer_for_low_latency_fallback_tests()

    x = torch.randn(4, 8, dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)

    recv_x, recv_count, handle, event, hook = buf.low_latency_dispatch(
        x,
        topk_idx,
        num_max_dispatch_tokens_per_rank=4,
        num_experts=2,
        use_fp8=False,
        return_recv_hook=True,
    )

    assert recv_x.shape == (2, 4, 8)
    assert torch.equal(recv_count, torch.tensor([2, 2], dtype=torch.int, device=x.device))
    assert isinstance(handle, dict)
    assert handle.get('fallback') == 'xpu_low_latency_dispatch'
    assert event is not None
    assert callable(hook)


def test_low_latency_combine_fallback_single_rank(monkeypatch):
    from xpu.deep_ep import buffer as buffer_mod

    monkeypatch.setattr(buffer_mod, '_has_xpu_runtime', lambda: True)
    buf = _make_buffer_for_low_latency_fallback_tests()

    x = torch.arange(64, dtype=torch.bfloat16).view(4, 16)
    topk_idx = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    topk_weights = torch.ones((4, 1), dtype=torch.float32)

    recv_x, _, handle, _, _ = buf.low_latency_dispatch(
        x,
        topk_idx,
        num_max_dispatch_tokens_per_rank=4,
        num_experts=2,
        use_fp8=False,
    )
    combined_x, event, hook = buf.low_latency_combine(
        recv_x,
        topk_idx,
        topk_weights,
        handle,
        return_recv_hook=True,
    )

    assert torch.equal(combined_x, x)
    assert event is not None
    assert callable(hook)


def test_low_latency_mask_buffer_fallback_single_rank(monkeypatch):
    from xpu.deep_ep import buffer as buffer_mod

    monkeypatch.setattr(buffer_mod, '_has_xpu_runtime', lambda: True)
    buf = _make_buffer_for_low_latency_fallback_tests()

    status = torch.zeros((1,), dtype=torch.int)
    buf.low_latency_update_mask_buffer(0, True)
    buf.low_latency_query_mask_buffer(status)
    assert int(status[0].item()) == 1

    buf.low_latency_clean_mask_buffer()
    buf.low_latency_query_mask_buffer(status)
    assert int(status[0].item()) == 0


def test_get_next_low_latency_combine_buffer_fallback_single_rank(monkeypatch):
    from xpu.deep_ep import buffer as buffer_mod

    monkeypatch.setattr(buffer_mod, '_has_xpu_runtime', lambda: False)
    buf = _make_buffer_for_low_latency_fallback_tests()

    handle = (
        torch.zeros((1,), dtype=torch.int),
        torch.zeros((1,), dtype=torch.int),
        4,
        8,
        2,
    )
    next_buf = buf.get_next_low_latency_combine_buffer(handle)
    assert next_buf.shape == (2, 4, 8)
    assert next_buf.dtype == torch.bfloat16
    assert next_buf.device.type == 'cpu'


def test_low_latency_mask_buffer_fallback_without_runtime_detection(monkeypatch):
    from xpu.deep_ep import buffer as buffer_mod

    # Regression: maintenance fallback should depend on error semantics, not runtime detection helper.
    monkeypatch.setattr(buffer_mod, '_has_xpu_runtime', lambda: False)
    buf = _make_buffer_for_low_latency_fallback_tests()

    status = torch.zeros((1,), dtype=torch.int)
    buf.low_latency_update_mask_buffer(0, True)
    buf.low_latency_query_mask_buffer(status)
    assert int(status[0].item()) == 1

    buf.low_latency_clean_mask_buffer()
    buf.low_latency_query_mask_buffer(status)
    assert int(status[0].item()) == 0


def test_local_single_rank_init_without_group(monkeypatch):
    from xpu.deep_ep import buffer as buffer_mod

    class _FakeRuntime:
        def __init__(self, rank, group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode, explicitly_destroy, enable_shrink, use_fabric):
            self.rank = rank
            self.group_size = group_size
            self._available = False

        def get_local_device_id(self):
            return 0

        def get_local_ipc_handle(self):
            return b"ipc"

        def get_num_rdma_ranks(self):
            return 1

        def sync(self, device_ids, ipc_handles, root_unique_id):
            assert device_ids == [0]
            assert ipc_handles == [b"ipc"]
            assert root_unique_id is None
            self._available = True

        def is_available(self):
            return self._available

    monkeypatch.setattr(buffer_mod, '_has_xpu_runtime', lambda: True)
    monkeypatch.setattr(buffer_mod, 'check_nvlink_connections', lambda group: None)
    monkeypatch.setattr(buffer_mod.deep_ep_cpp, 'Buffer', _FakeRuntime)

    buf = buffer_mod.Buffer(group=None, num_nvl_bytes=0, num_rdma_bytes=0)
    assert buf.rank == 0
    assert buf.group_size == 1
    assert buf.group is None
    assert buf.runtime.is_available()
