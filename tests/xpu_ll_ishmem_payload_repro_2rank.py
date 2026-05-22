import os
import sys

import torch
import torch.distributed as dist

REPO = "/root/jiafuzha/code-repo/zjf2012/DeepEP"
sys.path.insert(0, REPO)

import deep_ep  # noqa: E402


def expected_expert_count(expert, world, num_tokens, num_topk, num_experts):
    count = 0
    for src_rank in range(world):
        for token in range(num_tokens):
            for topk in range(num_topk):
                count += int((token + topk + src_rank) % num_experts == expert)
    return count


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

    num_tokens, hidden, num_experts, num_topk = 4, 128, 4, 2
    num_max_dispatch_tokens_per_rank = num_tokens * num_topk
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, world, num_experts)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True, num_qps_per_rank=1, explicitly_destroy=True)
    try:
        buffer.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)
        device = torch.device(f"xpu:{buffer.runtime.get_local_device_id()}")

        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device) * (rank + 1)
        x[:, -1] = torch.arange(num_tokens, dtype=torch.bfloat16, device=device)
        token_ids = torch.arange(num_tokens, device=device).view(-1, 1)
        topk_offsets = torch.arange(num_topk, device=device).view(1, -1)
        topk_idx = ((token_ids + topk_offsets + rank) % num_experts).to(deep_ep.topk_idx_t).contiguous()
        recv_stats = torch.zeros((num_experts // world, ), dtype=torch.int32, device=device)

        packed_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x,
            topk_idx,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            cumulative_local_expert_recv_stats=recv_stats,
            use_fp8=False,
            async_finish=False,
            return_recv_hook=False,
        )
        if hook is not None:
            hook()
        if event.event is not None:
            event.current_stream_wait()
        torch.xpu.synchronize()

        src_info, layout_range, _, _, _ = handle
        num_local_experts = num_experts // world
        for local_expert in range(num_local_experts):
            expert = rank * num_local_experts + local_expert
            expected = expected_expert_count(expert, world, num_tokens, num_topk, num_experts)
            actual = recv_count[local_expert].item()
            print(f"[rank {rank}] expert {expert}: recv_count={actual} expected={expected}", flush=True)
            assert actual == expected

            for src_rank in range(world):
                count = int((layout_range[local_expert, src_rank] & ((1 << 32) - 1)).item())
                begin = int((layout_range[local_expert, src_rank] >> 32).item())
                if count == 0:
                    continue
                tokens = src_info[local_expert, begin:begin + count].to(torch.long)
                expected_x = torch.ones((count, hidden), dtype=torch.bfloat16, device=device) * (src_rank + 1)
                expected_x[:, -1] = tokens.to(torch.bfloat16)
                actual_x = packed_x[local_expert, begin:begin + count]
                max_abs = (actual_x - expected_x).abs().max().item()
                print(
                    f"[rank {rank}] expert {expert} src {src_rank}: max_abs={max_abs} "
                    f"tokens={tokens.cpu().tolist()} actual_first={actual_x[:, 0].cpu().tolist()} "
                    f"actual_last={actual_x[:, -1].cpu().tolist()}",
                    flush=True)
                assert max_abs == 0.0, ("iSHMEM payload put path failed: count/src_info arrived, but payload data did not match")

        print(f"[rank {rank}] PASS iSHMEM payload put repro", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
