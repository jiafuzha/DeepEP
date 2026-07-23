import argparse
import os
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import (init_dist, bench, bench_kineto, calc_diff, create_grouped_scores, get_accelerator_device_type, hash_tensor,
                   inplace_unique, per_token_cast_back, per_token_cast_to_fp8)

# Test compatibility with low latency functions
import test_low_latency

def _xpu_all_reduce(tensor, group=None, op=dist.ReduceOp.SUM):
    """all_reduce that routes through CPU to avoid xccl<->iSHMEM GPU resource contention.

    Both xccl and iSHMEM/IBGDA allocate Level-Zero events/command-lists on the same
    XPU; mixing them at scale causes UR_RESULT_ERROR_OUT_OF_RESOURCES (40) or
    UR_RESULT_ERROR_DEVICE_LOST (20) during dispatch. The collective payloads here
    are tiny (counters/per-rank token counts), so a CPU detour is negligible while
    still using the xccl process group for rendezvous and barrier semantics.
    """
    if tensor.is_cpu:
        dist.all_reduce(tensor, op=op, group=group)
        return
    cpu_tensor = tensor.detach().to('cpu')
    dist.all_reduce(cpu_tensor, op=op, group=group)
    tensor.copy_(cpu_tensor.to(tensor.device))


def _xpu_all_gather(output_list, input_tensor, group=None):
    """all_gather that routes through CPU; same rationale as _xpu_all_reduce."""
    if input_tensor.is_cpu:
        dist.all_gather(output_list, input_tensor, group=group)
        return
    cpu_list = [torch.empty_like(t, device='cpu') for t in output_list]
    cpu_input = input_tensor.detach().to('cpu')
    dist.all_gather(cpu_list, cpu_input, group=group)
    for out, cpu_out in zip(output_list, cpu_list):
        out.copy_(cpu_out.to(out.device))


# The most-recently-created Buffer, so the __main__ failure handler can drain the
# GPU/NIC even when the test raised deep inside test_main (see teardown notes below).
_active_buffer = None


def _xpu_drain(buffer):
    """Drain every XPU stream + stop the iSHMEM host proxy thread. Runs on BOTH the
    success and failure paths. This is the fix for the two coupled XPU teardown bugs:

      1. rc=255 SIGABRT at shutdown. iSHMEM's host ``proxy_thread`` (a static
         ``std::thread`` in ishmem_ibgda/src/proxy.cpp) is joined only inside
         ``ishmemi_proxy_fini()``, which is reached only via ``ishmem_finalize`` --
         intentionally never called by DeepEP (it wedges this BMG/mlx5 stack). Left
         joinable, its destructor at normal interpreter shutdown calls
         ``std::terminate`` -> ``abort()`` -> SIGABRT, which mpirun reports as rc=255
         even though the test passed. It is a normal-exit path (not a signal), so a
         SIGABRT handler cannot intercept it and /dev/shm cleanup cannot prevent it.
      2. "GPU wedge needs a driver reset" cascade. DeepEP submits long-running
         IBGDA/comm exec queues; if the process tears down with one still submitted,
         the Xe GuC cannot preempt it at Level-Zero context destroy ("Schedule disable
         failed to respond" -> xe_guc_exec_queue_lr_cleanup -> GT reset), wedging the
         NEXT run's first GPU submission -> DEVICE_LOST accumulation.

    ``buffer.quiesce()`` fixes both: it drains all XPU streams (so the GuC retires the
    exec queue cleanly -> no GT reset) and calls ``stop_proxy()`` ->
    ``ishmemi_proxy_fini()`` which sets proxy_state=EXIT and JOINS ``proxy_thread``.
    It does NO NIC-BAR unmap / heap free (the ops that destabilize this stack), so it is
    safe even on a soft-failed device -- which is why it MUST also run on failure: a
    failing run that skips the drain leaves the exec queue submitted and turns an
    isolated soft failure into a wedge that cascades into every following run. See
    csrc/xpu/deep_ep_xpu.cpp quiesce()/stop_proxy()."""
    try:
        buffer.quiesce()
    except Exception as e:
        print(f'[teardown] quiesce raised {e} (continuing)', flush=True)


def _mpi_finalize():
    try:
        import ctypes
        libmpi = ctypes.CDLL('libmpi.so')
        initialized = ctypes.c_int()
        finalized = ctypes.c_int()
        libmpi.MPI_Initialized(ctypes.byref(initialized))
        libmpi.MPI_Finalized(ctypes.byref(finalized))
        if initialized.value and not finalized.value:
            libmpi.MPI_Finalize()
    except Exception as e:
        print(f'[teardown] MPI_Finalize raised {e} (continuing)', flush=True)


def teardown_success(buffer, group, local_rank, device_type):
    """Graceful post-test teardown (success path).

    XPU path mirrors the proven low-latency teardown (tests/test_low_latency.py,
    DEEP_EP_LL_ORDERLY_EXIT=2): the rc=255 SIGABRT here is NOT only iSHMEM's proxy
    thread -- with the XPU (oneCCL) process group, calling buffer.destroy()/dist.barrier()
    at teardown RE-ACTIVATES the iSHMEM host proxy + XCCL collectives, which then race the
    Level-Zero context destroy and leave oneCCL only partially finalized; a subsequent
    MPI_Finalize warns 'MPI_Finalize has been called before CCL finalization' and oneCCL's
    static destructors call std::terminate -> SIGABRT -> rc=255 even on a passing run.
    The proven order is therefore:
      1) buffer.quiesce() FIRST -- drain every XPU stream + ishmem_quiet + JOIN the iSHMEM
         proxy thread (see _xpu_drain / csrc/xpu/deep_ep_xpu.cpp), as the LAST GPU/NIC op.
      2) dist.destroy_process_group() -- fully finalize oneCCL. SKIP buffer.destroy() and
         dist.barrier() (they re-activate proxy+XCCL and race the L0 context destroy).
      3) settle so the (stopped) proxy is provably idle before libze's static dtors destroy
         the L0 context.
      4) MPI_Finalize, then let NORMAL interpreter shutdown run libze's destructors in order
         (so the next process's same-affinity exec queue does not collide -> no init hang).
    CUDA path keeps the original destroy()+barrier()+destroy_pg teardown."""
    if device_type == 'xpu':
        _xpu_drain(buffer)
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f'[teardown] destroy_process_group raised {e} (continuing)', flush=True)
        try:
            settle = float(os.getenv('DEEP_EP_XPU_EXIT_SETTLE_SEC', '2'))
        except Exception:
            settle = 2.0
        if settle > 0:
            time.sleep(settle)
        _mpi_finalize()
        if local_rank == 0:
            print('[teardown] all done, exiting cleanly', flush=True)
        return
    buffer.destroy()
    try:
        dist.barrier(group=group)
    except Exception:
        pass
    dist.destroy_process_group()
    if local_rank == 0:
        print('[teardown] all done, exiting cleanly', flush=True)


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace,
              num_sms: int,
              local_rank: int,
              num_local_ranks: int,
              num_ranks: int,
              num_nodes: int,
              rank: int,
              buffer: deep_ep.Buffer,
              group: dist.ProcessGroup,
              skip_benchmark: bool = False):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk_groups, num_topk, num_experts = args.num_topk_groups, args.num_topk, args.num_experts
    device_type = get_accelerator_device_type()

    assert num_experts % num_ranks == 0 and num_local_ranks >= 2
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device_type) * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device_type)
    x_e4m3 = per_token_cast_to_fp8(x)
    x_pure_rand_e4m3 = per_token_cast_to_fp8(x_pure_rand)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device_type).abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=device_type) * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device_type)
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)
    hash_value = 0

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device=device_type)
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    _xpu_all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device=device_type)
    num_tokens_per_rdma_rank = torch.empty((num_nodes, ), dtype=torch.int, device=device_type)
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

    ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    if group is not None:
        group.barrier()
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (24, 48, 96, 144, 160) else 512)
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        if not torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1)):
            bad = (check_x.amin(dim=1) != check_x.amax(dim=1)).nonzero(as_tuple=True)[0][:5]
            print(f'[check_data FAIL rank={rank}] {len(bad)} bad rows (showing 5): {bad.tolist()}', flush=True)
            for i in bad[:3]:
                ii = int(i)
                row = check_x[ii]
                print(f'  row {ii}: min={row.amin().item()}, max={row.amax().item()}, first8={row[:8].tolist()}, last8={row[-8:].tolist()}',
                      flush=True)
            assert False, 'check_x rows not uniform'  # noqa: B011
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            seg = check_x[check_start:check_end, :].int()
            err = (seg - i).sum().item()
            if err != 0:
                vals = seg[:, 0].tolist() if seg.numel() else []
                print(
                    f'[check_data FAIL rank={rank}] segment for src_rank={i} rows [{check_start},{check_end}) err_sum={err} first_col_vals={vals[:10]}',
                    flush=True)
                assert False, f'segment src_rank={i} values mismatch'  # noqa: B011
            check_start = check_end

    # DEEP_EP_MIN: minimal reproducer mode. Shrinks the 32-combination sweep to
    # a single deterministic case (BF16 `x`, with top-k, sync, no previous
    # event) so the internode-normal DEVICE_LOST can be reproduced with the
    # fewest dispatch/combine calls.
    _min = os.getenv('DEEP_EP_MIN')
    _prev_modes = (False, ) if _min else (False, True)
    _async_modes = (False, ) if _min else (False, True)
    _x_variants = (x, ) if _min else (x_pure_rand, x, x_pure_rand_e4m3, x_e4m3)
    _topk_variants = (True, ) if _min else (False, True)

    for previous_mode in _prev_modes:
        for async_mode in _async_modes:
            for current_x in _x_variants:
                for with_topk in _topk_variants:
                    is_rand = current_x is x_pure_rand or current_x is x_pure_rand_e4m3
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            flush=True,
                            end='')
                    dispatch_args = {
                        'x': current_x,
                        'num_tokens_per_rank': num_tokens_per_rank,
                        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                        'is_token_in_rank': is_token_in_rank,
                        'num_tokens_per_expert': num_tokens_per_expert,
                        'config': config,
                        'async_finish': async_mode,
                        'num_worst_tokens': num_tokens * num_topk if device_type == 'xpu' else 0,
                    }
                    if with_topk:
                        dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if is_rand else topk_weights})
                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(
                        **dispatch_args)
                    event.current_stream_wait() if async_mode else ()

                    # Debug: dump send_nvl_head after dispatch for with-topk on local_rank 0
                    if False:  # noqa: SIM223 - kept for ad-hoc diagnostics
                        if with_topk and device_type == 'xpu' and local_rank == 0:
                            snvl = handle[9]
                            srdma = handle[8]
                            if snvl is not None:
                                for dbg_t in range(min(5, snvl.size(0))):
                                    itr = is_token_in_rank[dbg_t]
                                    print(
                                        f'[dispatch-head rank={rank}] token {dbg_t}: is_in_rank={itr.tolist()}, '
                                        f'nvl_head={snvl[dbg_t].tolist()}, rdma_head={srdma[dbg_t].tolist() if srdma is not None else None}',
                                        flush=True)

                    # On XPU, num_worst_tokens > 0 returns padded tensors; trim to actual count
                    recv_gbl_rank_prefix_sum = handle[-4]
                    if os.getenv('DEEP_EP_MIN') and device_type == 'xpu':
                        _ps = recv_gbl_rank_prefix_sum
                        print(
                            f'[MIN rank={rank}] recv_gbl_rank_prefix_sum={_ps.tolist() if _ps is not None else None}, '
                            f'recv_x.shape={tuple(recv_x[0].shape) if isinstance(recv_x, tuple) else tuple(recv_x.shape)}, '
                            f'recv_topk_idx.shape={None if recv_topk_idx is None else tuple(recv_topk_idx.shape)}',
                            flush=True)
                        if recv_topk_idx is not None:
                            _ti = recv_topk_idx
                            print(
                                f'[MIN rank={rank}] recv_topk_idx min={int(_ti.min())} max={int(_ti.max())} '
                                f'first16={_ti.flatten()[:16].tolist()}',
                                flush=True)
                    if device_type == 'xpu' and recv_gbl_rank_prefix_sum is not None:
                        actual_count = int(recv_gbl_rank_prefix_sum[-1].item())
                        if isinstance(recv_x, tuple):
                            recv_x = (recv_x[0][:actual_count], recv_x[1][:actual_count])
                        else:
                            recv_x = recv_x[:actual_count]
                        if recv_topk_idx is not None:
                            recv_topk_idx = recv_topk_idx[:actual_count]
                        if recv_topk_weights is not None:
                            recv_topk_weights = recv_topk_weights[:actual_count]

                    if current_x is x_pure_rand or current_x is x:
                        hash_value += hash_tensor(recv_x)
                    else:
                        hash_value += hash_tensor(recv_x[0])
                        hash_value += hash_tensor(recv_x[1])

                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), \
                        f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    if device_type != 'xpu':
                        assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if not is_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    recv_topk_weights_clone = None
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) |
                                ((recv_topk_idx >= 0) &
                                 (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()
                        if not is_rand:
                            # NOTE: boolean-mask indexing (`t[mask]`) and
                            # `.nonzero()` hit a PyTorch-XPU allocation bug that
                            # attempts a 128 GiB alloc even for an all-False
                            # mask. Use `torch.where` (elementwise select, no
                            # nonzero) instead of masked assignment.
                            _fill = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)
                            recv_topk_weights = torch.where(recv_topk_idx.eq(-1), _fill, recv_topk_weights)
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test `num_worst_tokens != 0`
                    if with_topk and device_type != 'xpu':
                        num_worst_tokens = num_tokens * num_ranks
                        dispatch_args.update({'num_worst_tokens': num_worst_tokens})
                        recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_worst_x = per_token_cast_back(*recv_worst_x) if isinstance(recv_worst_x, tuple) else recv_worst_x
                        assert len(empty_list) == 0
                        assert num_worst_tokens == recv_worst_x.size(0)
                        assert num_worst_tokens == recv_worst_topk_idx.size(0)
                        assert num_worst_tokens == recv_worst_topk_weights.size(0)
                        assert torch.equal(recv_x, recv_worst_x[:recv_x.size(0)])
                        assert torch.equal(recv_topk_idx, recv_worst_topk_idx[:recv_x.size(0)])
                        assert torch.equal(recv_topk_weights_clone, recv_worst_topk_weights[:recv_x.size(0)])
                        assert torch.all(recv_worst_topk_idx[recv_x.size(0):] == -1).item()

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        if not is_rand:
                            check_data(recv_x, recv_gbl_rank_prefix_sum)

                    # Test combine
                    bias_0 = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device_type)
                    bias_1 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device_type)
                    combine_args = {'x': recv_x, 'bias': (bias_0, bias_1), 'handle': handle, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
                    check_x = (combined_x.float() - bias_0.float() - bias_1.float()) / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if is_rand else x
                    x_diff = calc_diff(check_x, ref_x)
                    if with_topk and device_type == 'xpu' and local_rank == 0:
                        # Per-token x check: identify tokens where x contributions are also wrong
                        per_token_x_err = (check_x - ref_x.float()).abs().max(dim=1).values
                        x_fail = (per_token_x_err > 1e-3).nonzero(as_tuple=True)[0][:5]
                        if len(x_fail) > 0:
                            print(
                                f'\n[x ALSO WRONG rank={rank}] {len(x_fail)} tokens with x_err>1e-3: '
                                f'{x_fail.tolist()}, errs={per_token_x_err[x_fail].tolist()}',
                                flush=True)
                        else:
                            print(f'\n[x OK rank={rank}] all per-token x_err < 1e-3, global diff={x_diff:.6e}', flush=True)
                    assert x_diff < 5e-4 if current_x is x_pure_rand_e4m3 else 5e-6
                    if with_topk:
                        dest_counts = is_token_in_rank.sum(dim=1).unsqueeze(1)
                        check_topk_weights = combined_topk_weights / dest_counts
                        ref_topk_weights = topk_weights_pure_rand if is_rand else topk_weights
                        tw_diff = calc_diff(check_topk_weights, ref_topk_weights)
                        if tw_diff >= 1e-9 and local_rank == 0:
                            abs_err = (check_topk_weights - ref_topk_weights).abs()
                            print(f'\n[topk_weights FAIL rank={rank}] diff={tw_diff:.6e} max_abs_err={abs_err.max().item():.6e}',
                                  flush=True)
                            # Find first few failing tokens
                            token_err = abs_err.max(dim=1).values
                            fail_tokens = (token_err > 1e-6).nonzero(as_tuple=True)[0][:5]
                            for ft in fail_tokens:
                                ft = ft.item()
                                itr = is_token_in_rank[ft]  # [num_ranks] bool
                                nvl_h = handle[9][ft] if handle[9] is not None else None  # send_nvl_head
                                rdma_h = handle[8][ft] if handle[8] is not None else None  # send_rdma_head
                                print(
                                    f'  token {ft}: is_in_rank={itr.tolist()}, '
                                    f'nvl_head={nvl_h.tolist() if nvl_h is not None else None}, '
                                    f'rdma_head={rdma_h.tolist() if rdma_h is not None else None}',
                                    flush=True)
                                print(f'    combined={combined_topk_weights[ft].tolist()}, ref={ref_topk_weights[ft].tolist()}', flush=True)
                        assert tw_diff < 1e-9, f'topk_weights diff={tw_diff:.6e} on rank={rank}'

                    hash_value += hash_tensor(recv_x)

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    # DEEP_EP_PERF: lightweight wall-clock perf capture (XPU-safe). The kineto
    # tuning sweep above is skipped on XPU (skip_benchmark), so this uses the
    # accelerator-Event based bench() on the cached dispatch/combine to report
    # latency and effective RDMA/NVL bandwidth without the fragile profiler path.
    if os.getenv('DEEP_EP_PERF'):
        perf_dispatch_args = {'x': x, 'handle': handle, 'config': config}
        perf_combine_args = {'x': recv_x, 'handle': handle, 'config': config}
        # Round-trip (dispatch -> combine) is the representative measurement: it
        # matches real usage (combine always follows dispatch) and, for the faithful
        # path, each op's deferred-doorbell AMO flag is flushed by the following op's
        # doorbell, so it does NOT suffer the isolated-repetition artifact (an
        # isolated dispatch/combine loop leaves the AMO egress lazy -> receiver poll
        # stalls). Isolated dispatch/combine are still reported below for reference.
        def _round_trip():
            buffer.dispatch(**perf_dispatch_args)
            buffer.combine(**perf_combine_args)
        rt_avg, rt_min, rt_max = bench(_round_trip, num_warmups=10, num_tests=20)
        d_avg, d_min, d_max = bench(lambda: buffer.dispatch(**perf_dispatch_args), num_warmups=10, num_tests=20)
        c_avg, c_min, c_max = bench(lambda: buffer.combine(**perf_combine_args), num_warmups=10, num_tests=20)
        rt_bytes = dispatch_bf16_rdma_send_bytes + combine_bf16_rdma_recv_bytes
        print(
            f'[PERF rank={rank}] num_tokens={num_tokens} hidden={hidden} '
            f'round_trip: {rt_avg * 1e6:.1f} us (min {rt_min * 1e6:.1f}, max {rt_max * 1e6:.1f}), '
            f'rdma_rt={rt_bytes / 1e6:.3f} MB @ {rt_bytes / 1e9 / rt_avg:.4f} GB/s | '
            f'dispatch(iso): {d_avg * 1e6:.1f} us (min {d_min * 1e6:.1f}, max {d_max * 1e6:.1f}), '
            f'rdma_send={dispatch_bf16_rdma_send_bytes / 1e6:.3f} MB @ {dispatch_bf16_rdma_send_bytes / 1e9 / d_avg:.4f} GB/s, '
            f'nvl_recv={dispatch_bf16_nvl_recv_bytes / 1e6:.3f} MB @ {dispatch_bf16_nvl_recv_bytes / 1e9 / d_avg:.4f} GB/s | '
            f'combine(iso): {c_avg * 1e6:.1f} us (min {c_min * 1e6:.1f}, max {c_max * 1e6:.1f}), '
            f'rdma_recv={combine_bf16_rdma_recv_bytes / 1e6:.3f} MB @ {combine_bf16_rdma_recv_bytes / 1e9 / c_avg:.4f} GB/s, '
            f'nvl_send={combine_bf16_nvl_send_bytes / 1e6:.3f} MB @ {combine_bf16_nvl_send_bytes / 1e9 / c_avg:.4f} GB/s',
            flush=True)

    if skip_benchmark:
        return hash_value

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in range(4, 45, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                tune_args = {'x': current_x, 'handle': handle, 'config': config}
                t, notify_t = bench_kineto(
                    lambda: buffer.dispatch(**tune_args),  # noqa: B023
                    ('dispatch', 'notify'),
                    suppress_kineto_output=True)
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)
                if local_rank == 0:
                    print(
                        f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: '
                        f'{notify_t * 1e6:.0f} + {t * 1e6:.0f} us, '
                        f'{rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ',
                        flush=True)
        if local_rank == 0:
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: '
                f'{best_results[3] * 1e6:.0f} + {best_time * 1e6:.0f} us, '
                f'{rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)',
                flush=True)
            print('', flush=True)

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor([best_results[0], best_results[1], best_results[2]], dtype=torch.int32, device=device_type)
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            _xpu_all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size, best_dispatch_results[2],
                                     rdma_buffer_size)

    dispatch_args = {
        'x': x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'config': dispatch_config if dispatch_config is not None else config
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 8, 1):
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
            tune_args = {'x': recv_x, 'handle': handle, 'config': config}
            t, notify_t = bench_kineto(
                lambda: buffer.combine(**tune_args),  # noqa: B023
                ('combine', 'notify'),
                suppress_kineto_output=True)
            if local_rank == 0:
                print(
                    f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: '
                    f'{notify_t * 1e6:.0f} + {t * 1e6:.0f} us, '
                    f'{combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), '
                    f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ',
                    flush=True)
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)

    if local_rank == 0:
        print(
            f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, '
            f'{best_results[3] * 1e6:.2f} + {best_time * 1e6:.2f} us, '
            f'{combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)',
            flush=True)
        print('', flush=True)
    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    device_type = get_accelerator_device_type()

    # Use PyTorch distributed (xccl on XPU, nccl on CUDA) via init_dist.
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # When DEEP_EP_NVL_RANKS is set, simulate multi-node topology:
    # num_local_ranks processes are split into groups of nvl_ranks.
    nvl_ranks = int(os.getenv('DEEP_EP_NVL_RANKS', '0'))
    if nvl_ranks > 0 and nvl_ranks < num_local_ranks:
        num_nodes = num_ranks // nvl_ranks
        num_local_ranks = nvl_ranks
    else:
        num_nodes = int(os.getenv('WORLD_SIZE', 1))
    if args.test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9

    num_sms = 24
    num_qps_per_rank = max(num_sms, ll_num_experts // num_ranks if args.test_ll_compatibility else 0)

    nvl_bytes = int(os.environ.get('DEEP_EP_NVL_BYTES', int(2e9)))
    rdma_bytes = int(os.environ.get('DEEP_EP_RDMA_BYTES', int(1e9)))

    print(
        f'[rank {rank}] Creating buffer: num_local_ranks={num_local_ranks}, num_nodes={num_nodes}, num_ranks={num_ranks}, nvl_bytes={nvl_bytes}, rdma_bytes={rdma_bytes}',
        flush=True)
    buffer = deep_ep.Buffer(group,
                            nvl_bytes,
                            rdma_bytes,
                            low_latency_mode=args.test_ll_compatibility,
                            num_qps_per_rank=num_qps_per_rank,
                            explicitly_destroy=True)
    print(f'[rank {rank}] Buffer created successfully', flush=True)
    global _active_buffer
    _active_buffer = buffer
    assert num_local_ranks >= 2 and num_ranks >= num_local_ranks

    # DEEP_EP_PERF_TOKENS: single-launch perf sweep. Reuse ONE iSHMEM init +
    # buffer across a list of token counts (comma-separated), calling the proven
    # test_main + DEEP_EP_PERF path for each. This avoids relaunching (and the
    # associated DEVICE_LOST-accumulation risk) once per token count. A single
    # deterministic correctness config runs first (via DEEP_EP_MIN) as a sanity
    # check, then the accelerator-Event bench() reports latency + throughput.
    _perf_tokens_env = os.getenv('DEEP_EP_PERF_TOKENS')
    if _perf_tokens_env:
        os.environ.setdefault('DEEP_EP_MIN', '1')
        os.environ['DEEP_EP_PERF'] = '1'
        token_list = [int(t) for t in _perf_tokens_env.replace(',', ' ').split()]
        for tk in token_list:
            args.num_tokens = tk
            torch.manual_seed(rank)
            if local_rank == 0:
                print(f'===== [perf-sweep] num_tokens={tk} =====', flush=True)
            test_main(args, num_sms, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group, True)
            if group is not None:
                try:
                    group.barrier()
                except Exception:
                    pass
        teardown_success(buffer, group, local_rank, device_type)
        return

    for seed in range(int(1e9)):
        if local_rank == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        torch.manual_seed(rank + seed)
        ref_hash = 0
        for i in (num_sms, ):
            ref_hash += test_main(args, i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group,
                                  args.pressure_test_mode == 1 or device_type == 'xpu')
            if local_rank == 0:
                print('', flush=True)
        if args.pressure_test_mode == 0:
            break

        if local_rank == 0:
            print(f'{ref_hash=}')
            print('', flush=True)

        for _ in range(20):
            torch.manual_seed(rank + seed)
            current_hash = 0
            for i in (num_sms, ):
                current_hash += test_main(args, i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group,
                                          args.pressure_test_mode == 1 or device_type == 'xpu')
                if local_rank == 0:
                    print('', flush=True)
            assert current_hash == ref_hash

    # Test compatibility with low latency functions
    if args.test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the buffer runtime and communication group. On XPU this drains the GPU/NIC
    # and stops the iSHMEM proxy thread before a clean interpreter shutdown; see
    # teardown_success()/_xpu_drain() for why this fixes both the rc=255 SIGABRT and the
    # DEVICE_LOST wedge cascade. On failure, the __main__ handler runs _xpu_drain() too.
    device_type = get_accelerator_device_type()
    teardown_success(buffer, group, local_rank, device_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test internode EP kernels')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk-groups', type=int, default=None, help='Number of top-k groups (default: `min(num_nodes, 4)`)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument(
        '--pressure-test-mode',
        type=int,
        default=0,
        help='Pressure test mode. 0: don\'t do pressure test, 1: do pressure test without benchmarks, 2: do pressure test with benchmarks')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts (default: 256')
    parser.add_argument('--test-ll-compatibility', action='store_true', help='whether to test compatibility with low-latency kernels')
    args = parser.parse_args()

    # Set default `num_topk_groups` if not provided
    if args.num_topk_groups is None:
        nvl_ranks = int(os.getenv('DEEP_EP_NVL_RANKS', '0'))
        if nvl_ranks > 0:
            num_nodes = args.num_processes // nvl_ranks
        else:
            num_nodes = int(os.getenv('WORLD_SIZE', 1))
        args.num_topk_groups = min(num_nodes, 4)

    num_processes = args.num_processes

    # Detect if running under mpirun (MPI_LOCALRANKID or PMI_RANK set)
    mpi_local_rank = os.environ.get('MPI_LOCALRANKID') or os.environ.get('PMI_RANK')
    if mpi_local_rank is not None:
        # Running under mpirun: each MPI process calls test_loop directly
        local_rank = int(mpi_local_rank)
        try:
            test_loop(local_rank, num_processes, args)
        except BaseException:
            # Failure path. On XPU, drain the GPU/NIC and stop the iSHMEM proxy BEFORE the
            # exception propagates: this idles the long-running exec queue (no GT reset ->
            # the NEXT run is NOT wedged, breaking the DEVICE_LOST cascade) and joins
            # ishmem's static proxy_thread (so shutdown does not std::terminate/SIGABRT,
            # giving a real non-zero rc for the failure instead of masking it as rc=255).
            # See _xpu_drain(). Best-effort; the original exception is always re-raised.
            if get_accelerator_device_type() == 'xpu' and _active_buffer is not None:
                _xpu_drain(_active_buffer)
            raise
    else:
        torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
