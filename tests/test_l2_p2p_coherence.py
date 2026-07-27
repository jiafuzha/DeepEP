#!/usr/bin/env python3
"""UT: BMG+igub L2/P2P cache coherence test using torch IPC.

Rank 0 writes random data to Rank 1 via IPC. Rank 1 reads back and checks.
Usage: mpirun -n 2 -ppn 2 python3 tests/test_l2_p2p_coherence.py --iters 50
"""
import os, sys, argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-rows', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    # Init distributed (expect MPI-launched with LOCAL_RANK set)
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('MPI_LOCALRANKID', 0)))
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29570')
    rank = local_rank
    dist.init_process_group(backend='cpu:gloo,xpu:xccl', rank=rank, world_size=2)

    writer_rank, reader_rank = 0, 1
    is_writer = (rank == writer_rank)
    is_reader = (rank == reader_rank)
    dtype = torch.bfloat16
    device = torch.device(f'xpu:{local_rank}')
    num_rows, hidden, niters = args.num_rows, args.hidden, args.iters

    # Staging buffer on each GPU
    stage = torch.zeros((num_rows, hidden), dtype=dtype, device=device)
    expected = torch.zeros((num_rows, hidden), dtype=dtype, device=device)

    results = {'cached_bad': 0, 'uncached_bad': 0}

    for it in range(niters):
        # Generate expected data on both ranks (deterministic LCG)
        for r in range(num_rows):
            seed = it * num_rows + r
            vals = torch.zeros(hidden, dtype=torch.float32)
            for h in range(hidden):
                seed = seed * 1103515245 + 12345
                vals[h] = float((seed >> 16) & 0xFFFF) / 65535.0 - 0.5
            expected[r] = vals.to(dtype)

        if is_writer:
            dist.send(expected.clone(), reader_rank)  # clone to avoid in-place
        if is_reader:
            dist.recv(stage, writer_rank)
        dist.barrier()

        if is_reader:
            torch.xpu.synchronize()
            # Cached read
            err = (stage.float() - expected.float()).abs()
            bad = int((err > 1e-3).sum().item())
            results['cached_bad'] += 1 if bad > 0 else 0

            # Uncached read: sync first
            torch.xpu.synchronize()
            err2 = (stage.float() - expected.float()).abs()
            bad2 = int((err2 > 1e-3).sum().item())
            results['uncached_bad'] += 1 if bad2 > 0 else 0

            if bad > 0 or bad2 > 0:
                print(f'[iter {it:3d}] cached_bad={bad:6d} uncached_bad={bad2:6d}', flush=True)
        dist.barrier()

    if is_reader:
        print(f'\n=== Results ({niters} iterations, {num_rows}x{hidden}) ===')
        print(f'Cached read bad iters:   {results["cached_bad"]}/{niters}')
        print(f'Uncached read bad iters: {results["uncached_bad"]}/{niters}')
        if results['cached_bad'] > 0 or results['uncached_bad'] > 0:
            print('L2 INCOHERENCE DETECTED')
        else:
            print('All clean — L2 coherence OK')

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
