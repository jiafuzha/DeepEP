#!/usr/bin/env python3
"""Minimal multi-node collectives smoke test for torch.distributed on XPU."""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        fail("torch.xpu is not available inside the container.")

    required_env = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    missing = [name for name in required_env if name not in os.environ]
    if missing:
        fail(f"Missing torchrun environment variables: {', '.join(missing)}")

    backend = os.environ.get("TORCH_BACKEND", "ccl")
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    allgather_input = torch.full((4,), float(rank), device=device, dtype=torch.float32)
    gathered = [torch.empty_like(allgather_input) for _ in range(world_size)]
    dist.all_gather(gathered, allgather_input)

    reduce_scatter_input = torch.arange(world_size * 4, device=device, dtype=torch.float32) + rank * 1000.0
    reduce_scatter_output = torch.empty(4, device=device, dtype=torch.float32)
    dist.reduce_scatter_tensor(reduce_scatter_output, reduce_scatter_input, op=dist.ReduceOp.SUM)

    torch.xpu.synchronize()
    dist.barrier()

    gathered_host = [tensor.cpu().tolist() for tensor in gathered]
    reduce_scatter_host = reduce_scatter_output.cpu().tolist()
    print(
        f"rank={rank} local_rank={local_rank} device={device} "
        f"all_gather={gathered_host} reduce_scatter={reduce_scatter_host}",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
