import torch


def test_force_python_stub_low_latency_zero_copy_smoke(forced_stub_backend) -> None:
    _, ext = forced_stub_backend

    buf = ext.Buffer(0, 1, 0, 256, True, False, False, False)
    buf.sync([], [], None)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]], dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0], [1], [0]], dtype=ext.topk_idx_t)
    topk_w = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float32)

    packed_x, _, _, src_info, layout_range, _, _ = buf.low_latency_dispatch(
        x, topk_idx,
        None, None,
        4, 2,
        False, False, False,
        False, False,
    )

    combine_buffer = buf.get_next_low_latency_combine_buffer(4, 2, 2)
    combine_buffer.zero_()
    combine_buffer.copy_(packed_x)

    # Pass an all-zero tensor as x; zero_copy=True should consume pre-registered buffer instead.
    dummy_x = torch.zeros_like(packed_x)
    combined, combine_event, combine_hook = buf.low_latency_combine(
        dummy_x, topk_idx, topk_w,
        src_info, layout_range,
        None, 4, 2,
        False, True, False, True, None,
    )
    assert combine_event is None
    assert callable(combine_hook)
    expected = torch.tensor([[0.5, 1.0], [3.0, 4.0], [4.0, 2.0]], dtype=torch.bfloat16)
    assert torch.allclose(combined.float().cpu(), expected.float(), atol=1e-4)

    # zero_copy=True without pre-registered buffer should fail.
    fresh_buf = ext.Buffer(0, 1, 0, 256, True, False, False, False)
    fresh_buf.sync([], [], None)
    try:
        fresh_buf.low_latency_combine(
            dummy_x, topk_idx, topk_w,
            src_info, layout_range,
            None, 4, 2,
            False, True, False, False, None,
        )
        raise AssertionError('expected ValueError for missing zero-copy buffer')
    except ValueError:
        pass
