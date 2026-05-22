import os
import sys
import argparse

import torch
import torch.distributed as dist

REPO = "/root/jiafuzha/code-repo/zjf2012/DeepEP"
sys.path.insert(0, REPO)

import deep_ep  # noqa: E402
from utils import per_token_cast_back  # noqa: E402


def make_inputs(rank, device, num_tokens, hidden, num_experts, num_topk):
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device) * (rank + 1)
    x[:, -1] = torch.arange(num_tokens, dtype=torch.bfloat16, device=device)
    token_ids = torch.arange(num_tokens, device=device).view(-1, 1)
    topk_offsets = torch.arange(num_topk, device=device).view(1, -1)
    topk_idx = ((token_ids + topk_offsets + rank) % num_experts).to(deep_ep.topk_idx_t).contiguous()
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=device)
    return x, topk_idx, topk_weights


def expected_expert_count(expert, world, num_tokens, num_topk, num_experts):
    count = 0
    for src_rank in range(world):
        for token in range(num_tokens):
            for topk in range(num_topk):
                count += int((token + topk + src_rank) % num_experts == expert)
    return count


def validate_dispatch_payload(rank, mode, device, x, packed_x, recv_count, handle, num_tokens, num_topk, num_experts, world):
    src_info, layout_range, _, hidden, _ = handle
    num_local_experts = num_experts // world
    if isinstance(packed_x, tuple):
        decoded_x = per_token_cast_back(packed_x[0].view(-1, hidden), packed_x[1].view(-1, hidden // 128)).view(packed_x[0].shape)
    else:
        decoded_x = packed_x

    for local_expert in range(num_local_experts):
        expert = rank * num_local_experts + local_expert
        expected = expected_expert_count(expert, world, num_tokens, num_topk, num_experts)
        actual_count = recv_count[local_expert].item()
        print(f"[rank {rank}] {mode} expert {expert}: recv_count={actual_count} expected={expected}", flush=True)
        assert actual_count == expected, f"rank {rank} {mode} expert {expert}: recv_count {actual_count} != {expected}"

        for src_rank in range(world):
            count = int((layout_range[local_expert, src_rank] & ((1 << 32) - 1)).item())
            begin = int((layout_range[local_expert, src_rank] >> 32).item())
            if count == 0:
                continue
            tokens = src_info[local_expert, begin:begin + count].to(torch.long)
            expected_x = torch.ones((count, hidden), dtype=torch.bfloat16, device=device) * (src_rank + 1)
            expected_x[:, -1] = tokens.to(torch.bfloat16)
            actual_x = decoded_x[local_expert, begin:begin + count]
            max_abs = (actual_x - expected_x).abs().max().item()
            tolerance = 0.5 if mode == "ue8m0" else 0.05 if mode == "fp8" else 0.0
            if max_abs > tolerance:
                print(
                    f"[rank {rank}] {mode} expert {expert} src {src_rank} tokens={tokens.cpu().tolist()} "
                    f"actual_first={actual_x[:, 0].detach().cpu().tolist()} expected_first={expected_x[:, 0].cpu().tolist()} "
                    f"actual_last={actual_x[:, -1].detach().cpu().tolist()} expected_last={expected_x[:, -1].cpu().tolist()}",
                    flush=True)
            assert max_abs <= tolerance, f"rank {rank} {mode} expert {expert} src {src_rank}: max_abs={max_abs}"


def run_case(rank, world, group, buffer, mode, skip_combine):
    use_fp8 = mode in ("fp8", "ue8m0")
    round_scale = mode == "ue8m0"
    use_ue8m0 = mode == "ue8m0"
    num_tokens, hidden, num_experts, num_topk = 4, 512 if use_ue8m0 else 128, 4, 2
    num_max_dispatch_tokens_per_rank = num_tokens * num_topk
    buffer.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)
    device = torch.device(f"xpu:{buffer.runtime.get_local_device_id()}")
    x, topk_idx, topk_weights = make_inputs(rank, device, num_tokens, hidden, num_experts, num_topk)
    recv_stats = torch.zeros((num_experts // world, ), dtype=torch.int32, device=device)

    print(f"[rank {rank}] starting {mode} dispatch", flush=True)
    packed_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        cumulative_local_expert_recv_stats=recv_stats,
        use_fp8=use_fp8,
        round_scale=round_scale,
        use_ue8m0=use_ue8m0,
        async_finish=False,
        return_recv_hook=False,
    )
    if hook is not None:
        hook()
    if event.event is not None:
        event.current_stream_wait()
    torch.xpu.synchronize()
    print(f"[rank {rank}] {mode} dispatch synchronized", flush=True)

    validate_dispatch_payload(rank, mode, device, x, packed_x, recv_count, handle, num_tokens, num_topk, num_experts, world)
    assert torch.equal(recv_stats, recv_count), f"rank {rank} {mode}: recv_stats {recv_stats} != recv_count {recv_count}"
    print(f"[rank {rank}] {mode} dispatch valid", flush=True)

    if skip_combine:
        return

    combine_x = per_token_cast_back(packed_x[0].view(-1, hidden), packed_x[1].view(-1, hidden // 128)).view(
        packed_x[0].shape) if isinstance(packed_x, tuple) else packed_x
    print(f"[rank {rank}] starting {mode} combine", flush=True)
    combined_x, event, hook = buffer.low_latency_combine(
        combine_x,
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
    print(f"[rank {rank}] {mode} combine synchronized", flush=True)

    expected_x = (x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)).to(torch.bfloat16)
    if not torch.allclose(combined_x, expected_x, rtol=0, atol=0):
        print(f"[rank {rank}] {mode} combined first={combined_x[:, 0].cpu().tolist()} last={combined_x[:, -1].cpu().tolist()}", flush=True)
        print(f"[rank {rank}] {mode} expected first={expected_x[:, 0].cpu().tolist()} last={expected_x[:, -1].cpu().tolist()}", flush=True)
    assert torch.allclose(combined_x, expected_x, rtol=0,
                          atol=0), (f"rank {rank} {mode}: combine mismatch max_abs={(combined_x - expected_x).abs().max().item()}")
    print(f"[rank {rank}] PASS {mode} low-latency dispatch+combine", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["bf16", "fp8", "ue8m0"], choices=["bf16", "fp8", "ue8m0"])
    parser.add_argument("--skip-combine", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("PMI_RANK", os.environ.get("PMIX_RANK", "0")))
    world = int(os.environ.get("PMI_SIZE", os.environ.get("PMIX_SIZE", "2")))
    assert world == 2, f"expected 2 ranks, got {world}"

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    # torch.xpu.set_device(0)

    dist.init_process_group(backend="xccl")
    group = dist.new_group(list(range(world)))

    hidden, num_experts = 512, 4
    num_tokens, num_topk = 4, 2
    num_max_dispatch_tokens_per_rank = num_tokens * num_topk
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, world, num_experts)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True, num_qps_per_rank=1, explicitly_destroy=True)
    try:
        for mode in args.cases:
            run_case(rank, world, group, buffer, mode, args.skip_combine)
        print(f"[rank {rank}] PASS tuned low-latency cases={args.cases} skip_combine={args.skip_combine}", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
