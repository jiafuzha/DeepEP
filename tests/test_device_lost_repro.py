"""DEVICE_LOST reproducer for DeepEP internode (NVL+RDMA) on XPU/iSHMEM-IBGDA.

================================================================================
TWO MODES — pick one based on what you want to debug:
================================================================================

(1) DETERMINISTIC reproducer  ->  use `tests/test_internode.py` directly:
        cd tests/docker-2node && TIMEOUT_SEC=300 ./run.sh
    With `csrc/xpu/internode.cpp` at HEAD `06a2199` (per-WI distributed
    dispatch puts) this RELIABLY raises:
        RuntimeError: level_zero backend failed with error: 20
        (UR_RESULT_ERROR_DEVICE_LOST)
    on rank 2/3 (node1) inside `internode_dispatch` after the first 2 BF16
    sub-tests pass cleanly. This is the failure the reproducer is named after.

(2) MINIMAL reproducer (this script):
        cd tests/docker-2node && \\
          TEST_SCRIPT=tests/test_device_lost_repro.py TIMEOUT_SEC=300 ./run.sh
    Runs `--num-iters` rounds of dispatch+combine (default 50) cycling through
    the same 4 sub-test variants test_internode.py uses before failing:
        (rand, no-topk) (rand, with-topk) (struct, no-topk) (struct, with-topk).
    Each round prints a clear "ITERATION N START / END" boundary and reports
    per-rank dispatch / combine outcomes. On RuntimeError the script aborts
    immediately and tags the failing iteration so you can correlate with logs.

    NOTE: in this debugging environment the minimal mode COMPLETES 50/50
    iterations successfully — it does NOT reproduce DEVICE_LOST. The bug
    requires the additional GPU command-list pressure that test_internode.py
    produces via its `bench`/`bench_kineto` profiling, FP8 conversions, and
    interleaved all_reduce traffic. Use mode (2) as a starting point for
    targeted experiments (env-var tweaks, alternate put strategies, NIC/QP
    instrumentation), and fall back to mode (1) for an actual repro.

================================================================================
What to look for in the log
================================================================================

    [rank N] iter K (label) dispatch ok=recv:M expected:M match=True
        -> good iteration

    [rank N] iter K (label) TOKEN-COUNT MISMATCH (not DEVICE_LOST)
        -> token undercount (separate, less severe issue)

    [rank N] !!! iter K (label) dispatch FAILED: RuntimeError: ...
    UR_RESULT_ERROR_DEVICE_LOST
        -> DEVICE_LOST reproduced; failed_iter is reported at the end

================================================================================
Tunables (env vars, all optional)
================================================================================
    REPRO_NUM_ITERS    int     default 50     dispatch+combine cycles to run
    REPRO_NUM_TOKENS   int     default 32     per-rank token count
    REPRO_HIDDEN       int     default 1024
    REPRO_NUM_TOPK     int     default 2
    REPRO_NUM_EXPERTS  int     default 8
    REPRO_BENCH_LAYOUT 0/1     default 1      run 50 layout warmups before loop

================================================================================
Topology assumptions (same as tests/docker-2node/run.sh)
================================================================================
    2 nodes x 2 local ranks = 4 ranks total
    rank 0,1 on node0 (mlx5_4, mlx5_5)
    rank 2,3 on node1 (mlx5_6, mlx5_7)
    DEEP_EP_NVL_RANKS=2 simulates 2 nodes x 2 GPUs.
"""
import argparse
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, get_accelerator_device_type, inplace_unique, create_grouped_scores


def _xpu_all_reduce(tensor, group=None, op=dist.ReduceOp.SUM):
    """CPU-routed all_reduce to avoid xccl<->iSHMEM resource contention on XPU.
    Mirrors the helper in tests/test_internode.py."""
    if tensor.is_cpu:
        dist.all_reduce(tensor, op=op, group=group)
        return
    cpu_tensor = tensor.detach().to('cpu')
    dist.all_reduce(cpu_tensor, op=op, group=group)
    tensor.copy_(cpu_tensor.to(tensor.device))


def _print_rank0(local_rank, msg):
    if local_rank == 0:
        print(msg, flush=True)


def repro_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    device_type = get_accelerator_device_type()
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    num_nodes = int(os.getenv('WORLD_SIZE', '1'))
    nvl_ranks_override = int(os.getenv('DEEP_EP_NVL_RANKS', '0'))
    if 0 < nvl_ranks_override < num_local_ranks:
        num_nodes = num_ranks // nvl_ranks_override
        num_local_ranks = nvl_ranks_override

    num_tokens = int(os.environ.get('REPRO_NUM_TOKENS', args.num_tokens))
    hidden = int(os.environ.get('REPRO_HIDDEN', args.hidden))
    num_topk = int(os.environ.get('REPRO_NUM_TOPK', args.num_topk))
    num_experts = int(os.environ.get('REPRO_NUM_EXPERTS', args.num_experts))
    num_iters = int(os.environ.get('REPRO_NUM_ITERS', args.num_iters))
    num_topk_groups = min(num_nodes, args.num_topk_groups or num_nodes)

    assert num_local_ranks >= 2, 'need >=2 local ranks per node'
    assert num_ranks >= 4, 'need >=4 total ranks (>=2 nodes) to exercise RDMA'
    assert num_experts % num_ranks == 0
    _print_rank0(
        local_rank,
        f'[repro config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}, '
        f'num_experts={num_experts}, num_nodes={num_nodes}, num_local_ranks={num_local_ranks}, '
        f'num_ranks={num_ranks}, num_iters={num_iters}')

    # Deterministic seed per rank so all 4 ranks compute consistent topology.
    torch.manual_seed(rank + 0)

    # Two BF16 inputs: structured `x` (each token row == rank) and random.
    x_struct = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device_type) * rank
    x_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device_type)

    # Build a topology consistent with test_internode.py.
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device_type).abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    rank_idx = (topk_idx // (num_experts // num_ranks)).to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device=device_type)
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device=device_type)
    num_tokens_per_rdma_rank = torch.empty((num_nodes,), dtype=torch.int, device=device_type)
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device=device_type)
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device=device_type)
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    _xpu_all_reduce(gbl_num_tokens_per_rank, group=group)

    # Buffer (mirrors test_internode.py defaults).
    num_sms = 24
    nvl_bytes = int(os.environ.get('DEEP_EP_NVL_BYTES', int(2e9)))
    rdma_bytes = int(os.environ.get('DEEP_EP_RDMA_BYTES', int(1e9)))
    _print_rank0(
        local_rank,
        f'[repro] Creating Buffer (nvl_bytes={nvl_bytes}, rdma_bytes={rdma_bytes}) ...')
    buffer = deep_ep.Buffer(group, nvl_bytes, rdma_bytes,
                            low_latency_mode=False,
                            num_qps_per_rank=num_sms,
                            explicitly_destroy=True)
    if group is not None:
        group.barrier()
    time.sleep(1)
    rdma_buffer_size, nvl_buffer_size = 128, 512
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Stress GPU/ibgda the same way tests/test_internode.py does before the
    # dispatch loop: many warmup invocations of get_dispatch_layout. Empirically
    # the DEVICE_LOST does not surface in pure dispatch+combine repetition; it
    # needs the additional GPU command-list pressure produced by the bench()
    # warmups + sub-test variation.
    if int(os.environ.get('REPRO_BENCH_LAYOUT', '1')):
        if local_rank == 0:
            print('[repro] warming up get_dispatch_layout (50 iters) ...', flush=True)
        for _ in range(50):
            buffer.get_dispatch_layout(topk_idx, num_experts)
        if device_type == 'xpu':
            torch.xpu.synchronize()
        if group is not None:
            group.barrier()
        time.sleep(1)

    # Schedule mirrors the failing prefix of test_internode.py:
    #   for previous_mode in (False, True):
    #     for async_mode in (False, True):
    #       for current_x in (x_pure_rand, x, ...):
    #         for with_topk in (False, True):
    # The DEVICE_LOST surfaced at the 3rd sub-test which is
    # (previous=False, async=False, current_x=x, with_topk=False) — i.e. a
    # toggle from with_topk=True (sub-test 1) back to with_topk=False, AND a
    # switch from x_pure_rand to x. The reproducer covers BOTH knobs.
    schedule = []
    for it in range(num_iters):
        # Repeat the failing 4-element prefix until num_iters is reached.
        sub = it % 4
        if sub == 0:
            schedule.append(('rand_no_topk', x_rand, False))
        elif sub == 1:
            schedule.append(('rand_with_topk', x_rand, True))
        elif sub == 2:
            schedule.append(('struct_no_topk', x_struct, False))
        else:
            schedule.append(('struct_with_topk', x_struct, True))

    # topk_weights: random or structured depending on the data variant, mirrors
    # tests/test_internode.py.
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=device_type) * rank
    topk_weights_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device_type)

    failed_iter = None
    for it, (label, current_x, with_topk) in enumerate(schedule):
        if local_rank == 0:
            print(f'\n========== ITERATION {it} START (x={label}) ==========', flush=True)
        if group is not None:
            group.barrier()
        try:
            dispatch_args = dict(
                x=current_x,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                config=config,
                async_finish=False,
                num_worst_tokens=num_tokens * num_topk if device_type == 'xpu' else 0,
            )
            if with_topk:
                dispatch_args['topk_idx'] = topk_idx
                dispatch_args['topk_weights'] = topk_weights_rand if 'rand' in label else topk_weights

            recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = \
                buffer.dispatch(**dispatch_args)
            recv_gbl_rank_prefix_sum = handle[-4]
            if device_type == 'xpu' and recv_gbl_rank_prefix_sum is not None:
                actual_count = int(recv_gbl_rank_prefix_sum[-1].item())
                if isinstance(recv_x, tuple):
                    recv_x = (recv_x[0][:actual_count], recv_x[1][:actual_count])
                else:
                    recv_x = recv_x[:actual_count]
            n_recv = recv_x[0].size(0) if isinstance(recv_x, tuple) else recv_x.size(0)
            expected = gbl_num_tokens_per_rank[rank].item()
            ok = (n_recv == expected)
            print(f'[rank {rank}] iter {it} ({label}) dispatch ok=recv:{n_recv} '
                  f'expected:{expected} match={ok}', flush=True)
            if not ok:
                print(f'[rank {rank}] iter {it} TOKEN-COUNT MISMATCH (not DEVICE_LOST)', flush=True)

            # Mirror test_internode's combine call so QP/CQ pressure on the
            # ibgda stack matches the failing scenario. The original DEVICE_LOST
            # only surfaced after several full dispatch+combine cycles.
            combine_in = recv_x if not isinstance(recv_x, tuple) else recv_x[0].to(torch.bfloat16)
            try:
                combined_x, _, combine_event = buffer.combine(
                    x=combine_in,
                    handle=handle,
                    config=config,
                    async_finish=False,
                )
                if local_rank == 0:
                    print(f'[rank {rank}] iter {it} ({label}) combine ok=shape:'
                          f'{tuple(combined_x.shape)}', flush=True)
            except Exception as cexc:  # noqa: BLE001
                print(f'\n[rank {rank}] !!! iter {it} ({label}) COMBINE FAILED: '
                      f'{type(cexc).__name__}: {cexc}', flush=True)
                traceback.print_exc()
                failed_iter = it
                break
        except Exception as exc:  # noqa: BLE001
            print(f'\n[rank {rank}] !!! iter {it} ({label}) dispatch FAILED: '
                  f'{type(exc).__name__}: {exc}', flush=True)
            traceback.print_exc()
            failed_iter = it
            break

        if group is not None:
            group.barrier()
        if local_rank == 0:
            print(f'========== ITERATION {it} END ==========', flush=True)

    if failed_iter is None:
        if local_rank == 0:
            print(f'\n[repro] All {num_iters} iterations completed without DEVICE_LOST. '
                  f'Bug NOT reproduced this run.', flush=True)
        rc = 0
    else:
        if local_rank == 0:
            print(f'\n[repro] DEVICE_LOST (or other failure) reproduced at '
                  f'iter {failed_iter} on rank {rank}.', flush=True)
        rc = 1

    try:
        buffer.destroy()
    except Exception:  # noqa: BLE001
        pass
    if group is not None:
        try:
            group.barrier()
        except Exception:  # noqa: BLE001
            pass
    sys.exit(rc)


def main():
    parser = argparse.ArgumentParser(
        description='Minimal DEVICE_LOST reproducer for DeepEP internode dispatch.')
    parser.add_argument('--num-processes', type=int, default=int(os.getenv('LOCAL_WORLD_SIZE', '2')),
                        help='Local processes per node (default: $LOCAL_WORLD_SIZE or 2).')
    parser.add_argument('--num-tokens', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--num-topk-groups', type=int, default=None)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--num-iters', type=int,
                        default=int(os.environ.get('REPRO_NUM_ITERS', '50')),
                        help='Number of dispatch+combine iterations to run (default: 50, '
                             'or $REPRO_NUM_ITERS if set). DEVICE_LOST in the original '
                             'test_internode.py surfaced after ~3 sub-tests; on the minimal '
                             'reproducer we run more iterations to apply equivalent QP/CQ '
                             'pressure to the ibgda stack.')
    args = parser.parse_args()

    # Mirror test_internode.py launch: a single worker per process; rank/local_rank
    # come from MPI/torchrun env vars that init_dist() consumes.
    local_rank = int(os.environ.get('LOCAL_RANK',
                                    os.environ.get('PMI_LOCAL_RANK',
                                                   os.environ.get('MPI_LOCALRANKID', '0'))))
    repro_main(local_rank, args.num_processes, args)


if __name__ == '__main__':
    main()
