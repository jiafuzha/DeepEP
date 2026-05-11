import torch


def test_force_python_stub_dispatch_layout_smoke(forced_stub_backend) -> None:
    _, ext = forced_stub_backend

    # Single RDMA rank case.
    buf = ext.Buffer(1, 4, 0, 0, False, False, False, False)
    topk_idx = torch.tensor([[0, 5], [2, -1], [7, 3]], dtype=ext.topk_idx_t)
    npr, nprd, npe, itr, event = buf.get_dispatch_layout(topk_idx, 8, None, False, False)
    assert event is None
    assert nprd is None
    assert npr.tolist() == [1, 2, 1, 1]
    assert npe.tolist() == [1, 0, 1, 1, 0, 1, 0, 1]
    assert itr.dtype == torch.bool
    assert itr.shape == (3, 4)

    # Multi RDMA rank case (16 ranks -> 2 RDMA ranks with 8 local ranks each).
    buf_rdma = ext.Buffer(0, 16, 0, 0, False, False, False, False)
    topk_idx_rdma = torch.tensor([[0, 8], [15, -1], [1, 9]], dtype=ext.topk_idx_t)
    npr2, nprd2, npe2, itr2, event2 = buf_rdma.get_dispatch_layout(topk_idx_rdma, 16, None, False, False)
    assert event2 is None
    assert npr2.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    assert nprd2 is not None
    assert nprd2.tolist() == [2, 3]
    assert len(npe2) == 16
    assert itr2.shape == (3, 16)
