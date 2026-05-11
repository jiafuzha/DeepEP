import torch


def test_force_python_stub_low_latency_dispatch_combine_smoke(forced_stub_backend) -> None:
    _, ext = forced_stub_backend

    buf = ext.Buffer(0, 1, 0, 256, True, False, False, False)
    buf.sync([], [], None)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]], dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0], [1], [0]], dtype=ext.topk_idx_t)
    topk_w = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float32)
    recv_stats = torch.zeros((2,), dtype=torch.int)
    wait_stats = torch.ones((1, 1), dtype=torch.int64)

    packed_x, packed_scales, recv_count, src_info, layout_range, event, hook = buf.low_latency_dispatch(
        x, topk_idx,
        recv_stats, wait_stats,
        4, 2,
        False, False, False,
        False, True,
    )
    assert packed_scales is None
    assert event is None
    assert callable(hook)
    assert recv_count.tolist() == [2, 1]
    assert recv_stats.tolist() == [2, 1]
    assert wait_stats.tolist() == [[0]]

    combined, combine_event, combine_hook = buf.low_latency_combine(
        packed_x, topk_idx, topk_w,
        src_info, layout_range,
        wait_stats, 4, 2,
        False, False, False, True, None,
    )
    assert combine_event is None
    assert callable(combine_hook)
    expected = torch.tensor([[0.5, 1.0], [3.0, 4.0], [4.0, 2.0]], dtype=torch.bfloat16)
    assert torch.allclose(combined.float(), expected.float(), atol=1e-4)

    # FP8 mode in staged path keeps BF16 payload and returns unit scales.
    packed_x2, packed_scales2, recv_count2, _, _, _, _ = buf.low_latency_dispatch(
        x, topk_idx,
        None, None,
        4, 2,
        True, False, False,
        False, False,
    )
    assert packed_x2.dtype == torch.bfloat16
    assert packed_scales2 is not None
    assert packed_scales2.dtype == torch.float32
    assert recv_count2.tolist() == [2, 1]
