import os
import sys

import torch
import torch.distributed as dist

REPO = "/root/jiafuzha/code-repo/zjf2012/DeepEP"
sys.path.insert(0, REPO)

import deep_ep  # noqa: E402


def main():
    rank = int(os.environ.get("PMI_RANK", os.environ.get("PMIX_RANK", "0")))
    world = int(os.environ.get("PMI_SIZE", os.environ.get("PMIX_SIZE", "2")))
    assert world == 2, f"expected 2 ranks, got {world}"

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "28741")
    torch.xpu.set_device(0)

    # Use gloo for host metadata collectives. XCCL object collectives allocate
    # invalid huge XPU tensors when each MPI rank has a narrow ZE affinity mask.
    dist.init_process_group("gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)))

    num_tokens, hidden, num_experts, num_topk = 4, 128, 4, 2
    num_max_dispatch_tokens_per_rank = num_tokens * num_topk
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, world, num_experts)
    buffer = deep_ep.Buffer(group,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=True,
                            num_qps_per_rank=1,
                            explicitly_destroy=True)
    try:
        buffer.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)

        x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="xpu") * (rank + 1)
        x[:, -1] = torch.arange(num_tokens, dtype=torch.bfloat16, device="xpu")
        token_ids = torch.arange(num_tokens, device="xpu").view(-1, 1)
        topk_offsets = torch.arange(num_topk, device="xpu").view(1, -1)
        topk_idx = ((token_ids + topk_offsets + rank) % num_experts).to(deep_ep.topk_idx_t).contiguous()
        topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="xpu")
        recv_stats = torch.zeros((num_experts // world, ), dtype=torch.int32, device="xpu")

        print(f"[rank {rank}] starting dispatch", flush=True)
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
        print(f"[rank {rank}] dispatch synchronized", flush=True)

        gathered_topk_idx = [None] * world
        print(f"[rank {rank}] gathering metadata", flush=True)
        dist.all_gather_object(gathered_topk_idx, topk_idx.cpu(), group=group)
        print(f"[rank {rank}] validating dispatch counts", flush=True)
        all_topk_idx = torch.stack(gathered_topk_idx).to(device="xpu", dtype=topk_idx.dtype)
        for local_expert in range(num_experts // world):
            expert = rank * (num_experts // world) + local_expert
            expected = (all_topk_idx == expert).sum().item()
            actual_count = recv_count[local_expert].item()
            actual_stats = recv_stats[local_expert].item()
            print(
                f"[rank {rank}] expert {expert}: recv_count={actual_count} recv_stats={actual_stats} expected={expected}",
                flush=True)
            assert actual_count == expected, f"rank {rank} expert {expert}: recv_count {actual_count} != {expected}"
            assert actual_stats == expected, f"rank {rank} expert {expert}: recv_stats {actual_stats} != {expected}"
        print(f"[rank {rank}] dispatch counts valid", flush=True)

        print(f"[rank {rank}] starting combine", flush=True)
        combined_x, event, hook = buffer.low_latency_combine(
            packed_x,
            topk_idx,
            topk_weights,
            handle,
            async_finish=False,
            return_recv_hook=False,
        )
        if hook is not None:
            hook()
        if event.event is not None:
            event.current_stream_wait()
        torch.xpu.synchronize()
        print(f"[rank {rank}] combine synchronized", flush=True)

        expected_x = (x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)).to(torch.bfloat16)
        if not torch.allclose(combined_x, expected_x, rtol=0, atol=0):
            print(f"[rank {rank}] combined first={combined_x[:, 0].cpu().tolist()} last={combined_x[:, -1].cpu().tolist()}", flush=True)
            print(f"[rank {rank}] expected first={expected_x[:, 0].cpu().tolist()} last={expected_x[:, -1].cpu().tolist()}", flush=True)
        assert torch.allclose(combined_x, expected_x, rtol=0, atol=0), (
            f"rank {rank}: combine mismatch max_abs={(combined_x - expected_x).abs().max().item()}")
        print(f"[rank {rank}] PASS tuned low-latency dispatch+combine", flush=True)
    finally:
        buffer.destroy()
        dist.barrier(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
