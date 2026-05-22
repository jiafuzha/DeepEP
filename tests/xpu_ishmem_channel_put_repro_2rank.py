import argparse
import os
import sys

import torch
import torch.distributed as dist

REPO = "/root/jiafuzha/code-repo/zjf2012/DeepEP"
sys.path.insert(0, REPO)

import deep_ep  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-ints", type=int, default=64)
    parser.add_argument("--num-channels", type=int, default=2)
    parser.add_argument("--queue-stride", type=int, default=64)
    args = parser.parse_args()

    rank = int(os.environ.get("PMI_RANK", os.environ.get("PMIX_RANK", "0")))
    world = int(os.environ.get("PMI_SIZE", os.environ.get("PMIX_SIZE", "2")))
    assert world == 2, f"expected 2 ranks, got {world}"

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")

    dist.init_process_group(backend="xccl")
    group = dist.new_group(list(range(world)))
    buffer = deep_ep.Buffer(group,
                            num_nvl_bytes=0,
                            num_rdma_bytes=16 * 1024 * 1024,
                            low_latency_mode=False,
                            num_qps_per_rank=1,
                            explicitly_destroy=True)
    try:
        result = buffer.runtime.debug_ishmem_channel_put(args.row_ints, args.num_channels, args.queue_stride)
        torch.xpu.synchronize()
        host = result.cpu()
        print(f"[rank {rank}] channel put result: {host.tolist()}", flush=True)
        assert int(host[:, 7].sum().item()) == 0
        print(f"[rank {rank}] PASS iSHMEM channel put repro", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
