import torch


def test_force_python_stub_dispatch_combine_smoke(forced_stub_backend) -> None:
    _, ext = forced_stub_backend

    cfg = ext.Config(1, 1, 1, 1, 2)

    # Intranode staged dispatch/combine path.
    buf = ext.Buffer(0, 1, 0, 0, False, False, False, False)
    buf.sync([], [], None)
    x = torch.ones((3, 4), dtype=torch.bfloat16)
    topk_idx = torch.zeros((3, 1), dtype=ext.topk_idx_t)
    topk_w = torch.ones((3, 1), dtype=torch.float32)
    npr = torch.tensor([3], dtype=torch.int)
    itr = torch.ones((3, 1), dtype=torch.bool)
    npe = torch.tensor([3], dtype=torch.int)

    out = buf.intranode_dispatch(
        x, None, topk_idx, topk_w,
        npr, itr, npe, 0, None, None,
        1, 0, cfg, None, False, False,
    )
    assert len(out) == 11
    recv_x, _, recv_topk_idx, recv_topk_w, _, rpm, cpm, _, recv_src_idx, send_head, event = out
    assert event is None
    assert recv_x.shape == x.shape
    assert recv_topk_idx.shape == topk_idx.shape
    assert recv_topk_w.shape == topk_w.shape

    combined_x, combined_w, combine_event = buf.intranode_combine(
        recv_x, recv_topk_w, None, None,
        recv_src_idx, rpm, cpm, send_head,
        cfg, None, False, False,
    )
    assert combine_event is None
    assert combined_x.shape == x.shape
    assert combined_w.shape == topk_w.shape

    # Internode staged dispatch/combine path.
    buf2 = ext.Buffer(0, 16, 0, 128, False, False, False, False)
    buf2.sync([], [], None)
    x2 = torch.ones((3, 4), dtype=torch.bfloat16)
    topk_idx2 = torch.zeros((3, 1), dtype=ext.topk_idx_t)
    topk_w2 = torch.ones((3, 1), dtype=torch.float32)
    npr2 = torch.tensor([3] + [0] * 15, dtype=torch.int)
    nprd2 = torch.tensor([3, 0], dtype=torch.int)
    itr2 = torch.zeros((3, 16), dtype=torch.bool)
    itr2[:, 0] = True
    npe2 = torch.tensor([3] + [0] * 15, dtype=torch.int)

    out2 = buf2.internode_dispatch(
        x2, None, topk_idx2, topk_w2,
        npr2, nprd2, itr2, npe2,
        0, 0, None, None, None, None,
        1, 0, cfg, None, False, False,
    )
    assert len(out2) == 15
    recv_x2, _, recv_topk_idx2, recv_topk_w2, _, _, _, recv_rdma_cpm2, recv_rdma_rps2, recv_gbl_cpm2, _, recv_src_meta2, send_rdma_head2, send_nvl_head2, event2 = out2
    assert event2 is None
    assert recv_x2.shape == x2.shape
    assert recv_topk_idx2.shape == topk_idx2.shape
    assert recv_topk_w2.shape == topk_w2.shape

    combined_x2, combined_w2, combine_event2 = buf2.internode_combine(
        recv_x2, recv_topk_w2, None, None,
        recv_src_meta2, itr2, recv_rdma_cpm2, recv_rdma_rps2,
        recv_gbl_cpm2, send_rdma_head2, send_nvl_head2,
        cfg, None, False, False,
    )
    assert combine_event2 is None
    assert combined_x2.shape == x2.shape
    assert combined_w2.shape == topk_w2.shape
