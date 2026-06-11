"""NVL-only internode normal-mode validation with 2 ranks on the same node.

Uses NVL IPC buffers (no iSHMEM/RDMA). Both ranks must be on the same node
with adjacent GPU devices (e.g. ZE_AFFINITY_MASK=4,5).

Run with:
    export ZE_AFFINITY_MASK=4,5
    timeout 120 mpirun -n 2 \
      -genv ZE_ENABLE_PCI_ID_DEVICE_ORDER 1 \
      -genv I_MPI_FABRICS shm \
      -genv MASTER_ADDR 127.0.0.1 \
      -genv MASTER_PORT 29520 \
      -genv ZE_AFFINITY_MASK 4,5 \
      python tests/xpu_normal_nvl_mpirun_2rank.py
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

REPO = "/data/jiafuzha/code-repo/zjf2012/DeepEP"
sys.path.insert(0, REPO)

import deep_ep  # noqa: E402


def route(src_rank, token, world):
    return (src_rank + token) % world


def count_to_rank(src_rank, dst_rank, world, num_tokens):
    return sum(1 for token in range(num_tokens) if route(src_rank, token, world) == dst_rank)


def compact_row(src_rank, token, dst_rank, world, num_tokens):
    prefix = sum(count_to_rank(prev_src, dst_rank, world, num_tokens) for prev_src in range(src_rank))
    ordinal = sum(1 for prior_token in range(token) if route(src_rank, prior_token, world) == dst_rank)
    return prefix + ordinal


def make_input(rank, num_tokens, hidden, device):
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device) * (rank + 1)
    x[:, -1] = torch.arange(num_tokens, dtype=torch.bfloat16, device=device)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=4)
    args = parser.parse_args()

    rank = int(os.environ.get("PMI_RANK", os.environ.get("PMIX_RANK", "0")))
    world = int(os.environ.get("PMI_SIZE", os.environ.get("PMIX_SIZE", "2")))
    assert world == 2, f"expected 2 ranks, got {world}"

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29520")

    dist.init_process_group(backend="xccl")
    group = dist.new_group(list(range(world)))

    num_tokens, hidden = args.num_tokens, 128
    num_worst_tokens = num_tokens * world

    # NVL-only: num_nvl_bytes > 0, num_rdma_bytes = 0
    buffer = deep_ep.Buffer(group,
                            num_nvl_bytes=4 * 1024 * 1024,
                            num_rdma_bytes=0,
                            low_latency_mode=False,
                            num_qps_per_rank=0,
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
        num_channels = 2
        config = deep_ep.Config(num_channels * 2, 1, 2)

        print(f"[rank {rank}] starting NVL-only dispatch", flush=True)
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
        print(f"[rank {rank}] NVL-only dispatch synchronized", flush=True)

        # Validate dispatch results
        recv_meta = handle[7].view(torch.int32).view(num_worst_tokens, 2)
        send_rdma_head = handle[8]
        send_nvl_head = handle[9]
        print(f"[rank {rank}] send_rdma_head shape={send_rdma_head.shape}", flush=True)
        assert send_rdma_head.shape[1] == world, f"NVL-only send_rdma_head must have {world} columns, got {send_rdma_head.shape[1]}"
        assert torch.all(send_nvl_head == -1).item(), "NVL-only send_nvl_head must be all -1"

        for src_rank in range(world):
            tokens = [token for token in range(num_tokens) if route(src_rank, token, world) == rank]
            for token in tokens:
                row = compact_row(src_rank, token, rank, world, num_tokens)
                expected_x = torch.ones((hidden, ), dtype=torch.bfloat16, device=device) * (src_rank + 1)
                expected_x[-1] = token
                actual_x = recv_x[row]
                max_abs = (actual_x - expected_x).abs().max().item()
                actual_meta = recv_meta[row].cpu().tolist()
                print(f"[rank {rank}] dispatch src={src_rank} token={token} row={row} "
                      f"meta={actual_meta} max_abs={max_abs}", flush=True)
                assert actual_meta == [src_rank, token], f"meta mismatch: {actual_meta} != [{src_rank}, {token}]"
                assert max_abs == 0.0, f"dispatch payload mismatch: max_abs={max_abs}"

        # Validate combine
        print(f"[rank {rank}] starting NVL-only combine", flush=True)
        combined_x, combined_topk_weights, event = buffer.combine(recv_x,
                                                                  handle,
                                                                  topk_weights=recv_topk_weights,
                                                                  config=config,
                                                                  async_finish=False)
        if event.event is not None:
            event.current_stream_wait()
        torch.xpu.synchronize()
        print(f"[rank {rank}] NVL-only combine synchronized", flush=True)
        max_abs = (combined_x - x).abs().max().item()
        print(f"[rank {rank}] combine max_abs={max_abs}", flush=True)
        assert max_abs == 0.0, f"combine mismatch: max_abs={max_abs}"
        if combined_topk_weights is not None:
            weight_max_abs = (combined_topk_weights - topk_weights).abs().max().item()
            assert weight_max_abs == 0.0, f"combine weights mismatch: max_abs={weight_max_abs}"

        print(f"[rank {rank}] PASS NVL-only internode BF16 dispatch+combine", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
