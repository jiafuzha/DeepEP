import os
import sys

import torch
import torch.distributed as dist

REPO = "/root/jiafuzha/code-repo/zjf2012/DeepEP"
sys.path.insert(0, REPO)

import deep_ep  # noqa: E402


def route(src_rank, token, world):
    return (src_rank + token) % world


def make_input(rank, num_tokens, hidden, device):
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device) * (rank + 1)
    x[:, -1] = torch.arange(num_tokens, dtype=torch.bfloat16, device=device)
    return x


def main():
    rank = int(os.environ.get("PMI_RANK", os.environ.get("PMIX_RANK", "0")))
    world = int(os.environ.get("PMI_SIZE", os.environ.get("PMIX_SIZE", "2")))
    assert world == 2, f"expected 2 ranks, got {world}"

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")

    dist.init_process_group(backend="xccl")
    group = dist.new_group(list(range(world)))

    num_tokens, hidden = 4, 128
    num_worst_tokens = num_tokens * world
    buffer = deep_ep.Buffer(group,
                            num_nvl_bytes=0,
                            num_rdma_bytes=16 * 1024 * 1024,
                            low_latency_mode=False,
                            num_qps_per_rank=1,
                            explicitly_destroy=True)
    try:
        device = torch.device(f"xpu:{buffer.runtime.get_local_device_id()}")
        x = make_input(rank, num_tokens, hidden, device)
        topk_idx = torch.stack((
            torch.arange(num_tokens, dtype=deep_ep.topk_idx_t, device=device),
            torch.arange(num_tokens, dtype=deep_ep.topk_idx_t, device=device) + 10 * (rank + 1),
        ),
                               dim=1).contiguous()
        topk_weights = torch.stack((
            torch.arange(num_tokens, dtype=torch.float32, device=device) + rank + 0.25,
            torch.arange(num_tokens, dtype=torch.float32, device=device) + rank + 0.75,
        ),
                                   dim=1).contiguous()
        is_token_in_rank = torch.zeros((num_tokens, world), dtype=torch.bool, device=device)
        for token in range(num_tokens):
            is_token_in_rank[token, route(rank, token, world)] = True
        num_tokens_per_rank = is_token_in_rank.sum(dim=0).to(torch.int32).contiguous()
        num_tokens_per_rdma_rank = num_tokens_per_rank.clone()
        num_tokens_per_expert = torch.zeros((world, ), dtype=torch.int32, device=device)
        config = deep_ep.Config(2, 1, 2, 1, 64)

        print(f"[rank {rank}] starting normal dispatch", flush=True)
        recv_x, recv_topk_idx, recv_topk_weights, _, handle, event = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_worst_tokens=num_worst_tokens,
            config=config,
            async_finish=False,
        )
        if event.event is not None:
            event.current_stream_wait()
        torch.xpu.synchronize()
        print(f"[rank {rank}] normal dispatch synchronized", flush=True)
        assert recv_topk_idx is not None and recv_topk_weights is not None

        recv_prefix = handle[6].cpu().tolist()
        assert recv_prefix == [(src_rank + 1) * num_tokens for src_rank in range(world)]
        recv_meta = handle[7].view(torch.int32).view(num_worst_tokens, 2)
        send_rdma_head = handle[8]
        send_nvl_head = handle[9]
        assert torch.all(send_nvl_head == -1).item()
        for dst_rank in range(world):
            for token in range(num_tokens):
                expected_head = rank * num_tokens + token if route(rank, token, world) == dst_rank else -1
                actual_head = send_rdma_head[token, dst_rank].item()
                assert actual_head == expected_head, (
                    f"rank {rank} token {token} dst {dst_rank}: send_rdma_head {actual_head} != {expected_head}")
        for src_rank in range(world):
            tokens = [token for token in range(num_tokens) if route(src_rank, token, world) == rank]
            for token in tokens:
                row = src_rank * num_tokens + token
                expected_x = torch.ones((hidden, ), dtype=torch.bfloat16, device=device) * (src_rank + 1)
                expected_x[-1] = token
                actual_x = recv_x[row]
                max_abs = (actual_x - expected_x).abs().max().item()
                actual_meta = recv_meta[row].cpu().tolist()
                print(f"[rank {rank}] dispatch src={src_rank} token={token} row={row} "
                      f"meta={actual_meta} max_abs={max_abs}", flush=True)
                assert actual_meta == [src_rank, token]
                assert max_abs == 0.0
                assert recv_topk_idx[row].cpu().tolist() == [
                    token,
                    token + 10 * (src_rank + 1),
                ]
                expected_weights = torch.tensor([token + src_rank + 0.25, token + src_rank + 0.75], dtype=torch.float32, device=device)
                assert torch.allclose(recv_topk_weights[row], expected_weights)

        print(f"[rank {rank}] starting normal combine", flush=True)
        combined_x, combined_topk_weights, event = buffer.combine(recv_x,
                                                                  handle,
                                                                  topk_weights=recv_topk_weights,
                                                                  config=config,
                                                                  async_finish=False)
        if event.event is not None:
            event.current_stream_wait()
        torch.xpu.synchronize()
        print(f"[rank {rank}] normal combine synchronized", flush=True)
        assert combined_topk_weights is not None
        max_abs = (combined_x - x).abs().max().item()
        weight_max_abs = (combined_topk_weights - topk_weights).abs().max().item()
        print(f"[rank {rank}] combine max_abs={max_abs}", flush=True)
        assert max_abs == 0.0
        assert weight_max_abs == 0.0

        print(f"[rank {rank}] starting normal FP8 dispatch", flush=True)
        x_fp8 = x.to(torch.float8_e4m3fn)
        x_scales = torch.ones((num_tokens, hidden // 128), dtype=torch.float32, device=device)
        recv_fp8_pair, recv_fp8_topk_idx, recv_fp8_topk_weights, _, fp8_handle, event = buffer.dispatch(
            (x_fp8, x_scales),
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_worst_tokens=num_worst_tokens,
            config=config,
            async_finish=False,
        )
        if event.event is not None:
            event.current_stream_wait()
        torch.xpu.synchronize()
        recv_fp8, recv_fp8_scales = recv_fp8_pair
        recv_fp8_meta = fp8_handle[7].view(torch.int32).view(num_worst_tokens, 2)
        for src_rank in range(world):
            tokens = [token for token in range(num_tokens) if route(src_rank, token, world) == rank]
            for token in tokens:
                row = src_rank * num_tokens + token
                expected_fp8 = (torch.ones((hidden, ), dtype=torch.bfloat16, device=device) * (src_rank + 1)).to(torch.float8_e4m3fn)
                expected_fp8[-1] = torch.tensor(token, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
                assert torch.equal(recv_fp8[row].cpu(), expected_fp8.cpu())
                assert torch.allclose(recv_fp8_scales[row], torch.ones_like(recv_fp8_scales[row]))
                assert recv_fp8_meta[row].cpu().tolist() == [src_rank, token]
                assert recv_fp8_topk_idx[row].cpu().tolist() == [
                    token,
                    token + 10 * (src_rank + 1),
                ]
                expected_weights = torch.tensor([token + src_rank + 0.25, token + src_rank + 0.75], dtype=torch.float32, device=device)
                assert torch.allclose(recv_fp8_topk_weights[row], expected_weights)
        print(f"[rank {rank}] PASS normal internode BF16 dispatch+combine and FP8 dispatch", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
