import argparse
import ctypes
import os
import random
import subprocess
import sys
import time
import torch
import torch.distributed as dist
from functools import partial
from typing import Literal, Set

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, get_accelerator_device_type, hash_tensor, per_token_cast_back


def debug_print(rank: int, message: str):
    if os.getenv('DEEP_EP_TEST_DEBUG', '0') == '1':
        print(f'[rank {rank}] {message}', flush=True)


def get_launcher_rank_env():
    for name in ('PMI_RANK', 'PMIX_RANK', 'OMPI_COMM_WORLD_RANK'):
        if name in os.environ:
            rank_name = name
            break
    else:
        return None

    for name in ('PMI_SIZE', 'PMIX_SIZE', 'OMPI_COMM_WORLD_SIZE'):
        if name in os.environ:
            return int(os.environ[rank_name]), int(os.environ[name])
    return None


def is_xpu_direct_doorbell_run() -> bool:
    return get_accelerator_device_type() == 'xpu' and os.getenv('ISHMEM_IBGDA_DIRECT_DOORBELL', '0') == '1'


def finalize_mpi_and_exit():
    libmpi = ctypes.CDLL("libmpi.so")
    initialized = ctypes.c_int()
    finalized = ctypes.c_int()
    libmpi.MPI_Initialized(ctypes.byref(initialized))
    libmpi.MPI_Finalized(ctypes.byref(finalized))
    if initialized.value and not finalized.value:
        libmpi.MPI_Finalize()
    os._exit(0)


def configure_xpu_rank_affinity(local_rank: int):
    if not is_xpu_direct_doorbell_run():
        return

    device_ids = os.getenv('DEEP_EP_XPU_DEVICE_IDS')
    if device_ids is None:
        device_ids = os.getenv('ZE_AFFINITY_MASK', '5,6')
        os.environ.setdefault('DEEP_EP_XPU_DEVICE_IDS', device_ids)
    os.environ.setdefault('ZE_AFFINITY_MASK', device_ids)

    physical_devices = [device.strip() for device in device_ids.split(',') if device.strip()]
    if local_rank < len(physical_devices):
        os.environ.setdefault('ISHMEM_IBGDA_NIC', f'mlx5_{physical_devices[local_rank]}')


def maybe_launch_xpu_direct_doorbell_with_mpirun(args: argparse.Namespace):
    if not is_xpu_direct_doorbell_run() or get_launcher_rank_env() is not None:
        return False
    if os.getenv('DEEP_EP_TEST_LOW_LATENCY_NO_MPIRUN', '0') == '1':
        return False

    device_ids = os.getenv('DEEP_EP_XPU_DEVICE_IDS', os.getenv('ZE_AFFINITY_MASK', '5,6'))
    os.environ.setdefault('DEEP_EP_XPU_DEVICE_IDS', device_ids)
    os.environ.setdefault('ZE_AFFINITY_MASK', device_ids)

    command = ['mpirun', '-n', str(args.num_processes)]
    for name in (
            'ISHMEM_IB_ENABLE_IBGDA',
            'ISHMEM_IBGDA_DIRECT_DOORBELL',
            'ISHMEM_ENABLE_GPU_IPC',
            'ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP',
            'ISHMEM_SYMMETRIC_SIZE',
            'ZE_ENABLE_PCI_ID_DEVICE_ORDER',
            'ZE_AFFINITY_MASK',
            'DEEP_EP_XPU_DEVICE_IDS',
            'ISHMEM_IBGDA_QPS_PER_PE',
            'ISHMEM_IBGDA_DB_BATCH_SIZE',
            'ISHMEM_IBGDA_BAR_BACKEND',
            'I_MPI_FABRICS',
            'ISHMEM_DEBUG',
            'PYTHONPATH',
            'MASTER_ADDR',
            'MASTER_PORT',
            'I_MPI_MPCP_SERVER_PORT',
            'PYTHONUNBUFFERED',
            'DEEP_EP_TEST_DEBUG',
    ):
        value = os.environ.get(name)
        if value is not None:
            command.extend(['-genv', name, value])
    command.extend([sys.executable, '-u', __file__, *sys.argv[1:]])
    print("command is: ", command, flush=True)
    return subprocess.run(command, check=False).returncode


def simulate_failure_and_skip(rank: int, api: Literal["dispatch", "combine", "clean"], expected_masked_ranks: Set[int]):
    # Simulates rank failure when the rank first calls the corresponding communication API
    failed_api_ranks = {
        # API -> rank to fail (rank fails when it first calls the corresponding communication API)
        'dispatch': 1,
        'combine': 3,
        'clean': 5
    }
    if rank in expected_masked_ranks:
        # Rank already failed
        return True
    if api in failed_api_ranks.keys():
        expected_masked_ranks.add(failed_api_ranks[api])
        if failed_api_ranks[api] == rank:
            print(f"Rank {rank} failed when first calling {api} communication API, exit...", flush=True)
            return True
    return False


def query_mask_buffer_and_check(api: Literal["dispatch", "combine", "clean"], buffer: deep_ep.Buffer, mask_status: torch.Tensor,
                                expected_masked_ranks: Set[int]):
    buffer.low_latency_query_mask_buffer(mask_status)
    assert set(mask_status.nonzero().squeeze(-1).tolist()) == expected_masked_ranks


def test_main(num_tokens: int,
              hidden: int,
              num_experts: int,
              num_topk: int,
              rank: int,
              num_ranks: int,
              group: dist.ProcessGroup,
              buffer: deep_ep.Buffer,
              use_logfmt: bool = False,
              shrink_test: bool = False,
              seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    device_type = get_accelerator_device_type()

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=device_type) * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device=device_type).to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for _ in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device_type) * 0.5 * random.random())
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device_type) * 0.1)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device_type).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device_type).abs()

    # Randomly mask some positions
    for _ in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    debug_print(rank, 'before topk all_gather')
    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device=device_type)
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)
    debug_print(rank, 'after topk all_gather')

    # For failure simulation and shrink testing
    mask_status = torch.zeros((num_ranks, ), dtype=torch.int, device=device_type)
    expected_masked_ranks = set()

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False, ):
                    for use_ue8m0 in (False, True) if round_scale else (False, ):
                        if shrink_test and simulate_failure_and_skip(rank, "dispatch", expected_masked_ranks):
                            break
                        num_times += 1
                        for _ in range((num_times % 2) + 1):
                            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device=device_type)
                            debug_print(
                                rank,
                                f'before low_latency_dispatch use_fp8={dispatch_use_fp8} round_scale={round_scale} use_ue8m0={use_ue8m0}')
                            packed_recv_x, packed_recv_count, handle, event, hook = \
                                buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                                            use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                                            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                            async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                            debug_print(rank, 'after low_latency_dispatch call')
                            hook() if return_recv_hook else event.current_stream_wait()
                            debug_print(rank, 'after low_latency_dispatch wait')
                        if shrink_test:
                            query_mask_buffer_and_check("dispatch", buffer, mask_status, expected_masked_ranks)
                        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                        simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden),
                                                               packed_recv_x[1].view(-1, packed_recv_x[1].size(-1))).view(packed_recv_x[0].shape) \
                            if dispatch_use_fp8 else packed_recv_x.clone()
                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
                            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

                            # Check expert indices
                            int_mask = (2**32) - 1
                            num_valid_tokens = recv_count.item()
                            assert cumulative_local_expert_recv_stats[i].item(
                            ) == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'
                            assert num_valid_tokens == (
                                recv_layout_range
                                & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
                            assert num_valid_tokens == (all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status == 0].sum().item(
                            ), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status==0].sum().item()}'

                            if num_valid_tokens == 0:
                                continue
                            # Check received data
                            if current_x is x:
                                recv_x = recv_x[:num_valid_tokens]
                                recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                recv_src_info = recv_src_info[:num_valid_tokens]
                                assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
                                if round_scale:
                                    assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
                                else:
                                    assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                                for j in range(num_ranks):
                                    if shrink_test and mask_status[j]:
                                        continue
                                    begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()
                                    if not round_scale:
                                        assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
                                        assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0
                            if dispatch_use_fp8:
                                hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                                hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
                            else:
                                hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])

                        # Check combine correctness
                        if shrink_test and simulate_failure_and_skip(rank, "combine", expected_masked_ranks):
                            break
                        for zero_copy in (False, ) if use_logfmt else (False, True):
                            if zero_copy:
                                buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
                            out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=device_type)
                            combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                                                 topk_idx,
                                                                                 topk_weights,
                                                                                 handle,
                                                                                 use_logfmt=use_logfmt,
                                                                                 async_finish=not return_recv_hook,
                                                                                 zero_copy=zero_copy,
                                                                                 return_recv_hook=return_recv_hook,
                                                                                 out=out)
                            hook() if return_recv_hook else event.current_stream_wait()
                            if shrink_test:
                                query_mask_buffer_and_check("combine", buffer, mask_status, expected_masked_ranks)
                            if do_check:
                                if shrink_test:
                                    owner_by_expert = (torch.arange(num_experts, device=device_type) // num_local_experts)
                                    fail_owner_mask = (mask_status == 1).index_select(0, owner_by_expert)
                                    valid_topk_idx = topk_idx >= 0
                                    failed_topk_idx = torch.zeros_like(topk_idx, device=device_type, dtype=torch.bool)
                                    failed_topk_idx[valid_topk_idx] = fail_owner_mask.index_select(0, topk_idx[valid_topk_idx])
                                    topk_idx[failed_topk_idx] = -1
                                diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
                                assert torch.isnan(combined_x).sum().item() == 0
                                if not round_scale:
                                    assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'
                                hash_value ^= hash_tensor(combined_x)

                        # Clean buffer API
                        if shrink_test:
                            if simulate_failure_and_skip(rank, "clean", expected_masked_ranks):
                                break

                            buffer.clean_low_latency_buffer(num_tokens, hidden, num_experts)
                            query_mask_buffer_and_check("clean", buffer, mask_status, expected_masked_ranks)

    if shrink_test:
        return

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                             topk_idx,
                                                             topk_weights,
                                                             handle,
                                                             use_logfmt=use_logfmt,
                                                             return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    print(
        f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
        f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us',
        flush=True)
    if get_accelerator_device_type() == 'xpu':
        return hash_value

    # Separate profiling
    for return_recv_hook in (False, True):
        group.barrier()
        dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
                                             kernel_names=('dispatch', 'combine'),
                                             barrier_comm_profiling=True,
                                             suppress_kineto_output=True,
                                             num_kernels_per_period=2 if return_recv_hook else 1)
        if not return_recv_hook:
            print(
                f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us',
                flush=True)
        else:
            print(
                f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
                f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us',
                flush=True)
    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    configure_xpu_rank_affinity(local_rank)
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
                            explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl,
                            enable_shrink=args.shrink_test)
    completed = False
    try:
        test_main(num_tokens,
                  hidden,
                  num_experts,
                  num_topk,
                  rank,
                  num_ranks,
                  group,
                  buffer,
                  use_logfmt=args.use_logfmt,
                  shrink_test=args.shrink_test,
                  seed=1)

        do_pressure_test = args.pressure_test
        for seed in range(int(1e9) if do_pressure_test else 0):
            if local_rank == 0:
                print(f'Testing with seed {seed} ...', flush=True)
            ref_hash = test_main(num_tokens,
                                 hidden,
                                 num_experts,
                                 num_topk,
                                 rank,
                                 num_ranks,
                                 group,
                                 buffer,
                                 use_logfmt=args.use_logfmt,
                                 seed=seed)
            for _ in range(20):
                assert test_main(num_tokens,
                                 hidden,
                                 num_experts,
                                 num_topk,
                                 rank,
                                 num_ranks,
                                 group,
                                 buffer,
                                 use_logfmt=args.use_logfmt,
                                 seed=seed) == ref_hash, f'Error: seed={seed}'
        completed = True
    finally:
        if is_xpu_direct_doorbell_run():
            # CRITICAL (XPU LL lifecycle): drain all in-flight GPU work (long-running
            # IBGDA poll/quiet exec queues) + outbound RDMA BEFORE the process exits,
            # otherwise the GuC cannot preempt the still-submitted long-running exec
            # queue at process teardown ("Schedule disable failed to respond" ->
            # xe_guc_exec_queue_lr_cleanup -> GT reset), wedging the NEXT process's
            # first GPU submission (Buffer ctor memset) -> init hang / DEVICE_LOST.
            #
            # This drain MUST run on FAILURE too, not just on success: a failed run
            # (soft data-mismatch, poll-cap "0 tokens", or a caught exception on a
            # still-healthy GPU) that skips the drain leaves the long-running exec
            # queue submitted -> GT reset -> the *next* run inherits a wedged GT and
            # hard-DEVICE_LOSTs. That is precisely the "one failure cascades into the
            # following runs" amplifier. Draining here converts an isolated soft
            # failure into an isolated soft failure (no cascade). It is best-effort:
            # if the GPU is already DEVICE_LOST the stream syncs simply throw and are
            # swallowed, and quiesce() is already watchdog-guarded internally.
            if os.getenv('DEEP_EP_LL_QUIESCE', '1') != '0':
                try:
                    try:
                        torch.xpu.synchronize()
                    except Exception as e:
                        print(f'[rank {rank}] quiesce: torch.xpu.synchronize raised {e}', flush=True)
                    print(f'[rank {rank}] quiesce: draining GPU/NIC before exit '
                          f'(completed={completed}) ...', flush=True)
                    buffer.quiesce()
                    print(f'[rank {rank}] quiesce: done', flush=True)
                except Exception as e:
                    print(f'[rank {rank}] quiesce: raised {e} (continuing teardown)', flush=True)
            # The orderly-EXIT handling below returns / exits 0 and would mask a
            # test failure, so it only runs when the test actually completed. On
            # FAILURE we have already drained the GPU/NIC above (breaking the
            # cascade); we then fall through so the original exception propagates
            # and the run is correctly reported as failed (non-zero exit). Normal
            # interpreter shutdown still runs libze's static destructors in order.
            if not completed:
                print(f'[rank {rank}] teardown: run FAILED; GPU/NIC drained, '
                      f'propagating failure without orderly-exit masking.', flush=True)
                # GPU/NIC already drained above (breaking the cascade). Do NOT call
                # buffer.destroy()/dist.barrier() -- on a wedged/DEVICE_LOST device
                # those can hang (rc=124) and mask the real failure. Re-raise the
                # original exception for a clean non-zero exit; normal interpreter
                # shutdown still runs libze's static destructors in order.
                raise
            else:
                # Default teardown mode = 2 (minimal orderly): quiesce (drain the
                # long-running IBGDA exec queue) + stop the iSHMEM host proxy, then
                # destroy_process_group + MPI_Finalize + NORMAL process exit so
                # libze runs zeCommandQueueDestroy/zeContextDestroy in order. This
                # ELIMINATES the LL init-hang (rc=124 in Buffer ctor memset) that the
                # old os._exit(0) fast-teardown (mode 0) caused by leaving the LR exec
                # queue's GuC registration pending -> next same-affinity submission
                # never signals. See csrc/xpu/deep_ep_xpu.cpp quiesce()/stop_proxy().
                _orderly = os.getenv('DEEP_EP_LL_ORDERLY_EXIT', '2')
                if _orderly == '0':
                    finalize_mpi_and_exit()
                elif _orderly == '2':
                    # MINIMAL orderly teardown: skip buffer.destroy()/dist.barrier()
                    # (which re-activate the iSHMEM host proxy + XCCL collectives and
                    # race the Level-Zero context destroy -> UNINITIALIZED copy errors
                    # + teardown GT reset). Just close the process group, MPI_Finalize,
                    # and let normal interpreter shutdown run libze's static destructors
                    # (zeCommandQueueDestroy/zeContextDestroy) so the next process's
                    # same-affinity exec queue does not collide -> no init hang.
                    print(f'[rank {rank}] orderly-exit(min): destroy_pg + MPI_Finalize + normal exit', flush=True)
                    try:
                        dist.destroy_process_group()
                    except Exception as e:
                        print(f'[rank {rank}] orderly-exit(min): destroy_pg raised {e}', flush=True)
                    # Settle: let the iSHMEM host proxy thread drain any in-flight device
                    # copy requests and go idle *before* the interpreter's libze static
                    # destructors tear down the Level-Zero context. Without this window
                    # the proxy can be mid-zeCommandListAppendMemoryCopy when the context
                    # is destroyed -> ishmemi_copy ZE_RESULT_ERROR_UNINITIALIZED storm ->
                    # teardown GT reset. quiesce() already idled the GPU/NIC; this just
                    # widens the gap so the proxy is provably idle at context destroy.
                    try:
                        _settle = float(os.getenv('DEEP_EP_LL_EXIT_SETTLE_SEC', '0'))
                    except Exception:
                        _settle = 2.0
                    if _settle > 0:
                        time.sleep(_settle)
                    try:
                        libmpi = ctypes.CDLL('libmpi.so')
                        initialized = ctypes.c_int(); finalized = ctypes.c_int()
                        libmpi.MPI_Initialized(ctypes.byref(initialized))
                        libmpi.MPI_Finalized(ctypes.byref(finalized))
                        if initialized.value and not finalized.value:
                            libmpi.MPI_Finalize()
                    except Exception as e:
                        print(f'[rank {rank}] orderly-exit(min): MPI_Finalize raised {e}', flush=True)
                    return
                else:
                    # Full orderly teardown: buffer.destroy() + dist teardown then exit.
                    print(f'[rank {rank}] orderly-exit: destroy() + dist teardown ...', flush=True)
        # Destroy the buffer runtime and communication group
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128, help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288, help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action="store_true", help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true', help='Whether to disable NVLink for testing')
    parser.add_argument('--use-logfmt', action='store_true', help='Whether to test LogFMT combine')
    parser.add_argument("--pressure-test", action='store_true', help='Whether to do pressure test')
    parser.add_argument("--shrink-test", action='store_true', help='Whether to simulate failure and test shrink mode')
    args = parser.parse_args()
    # print(args.pressure_test)

    num_processes = args.num_processes
    mpirun_returncode = maybe_launch_xpu_direct_doorbell_with_mpirun(args)
    if mpirun_returncode is not False:
        sys.exit(mpirun_returncode)

    launcher_env = get_launcher_rank_env()
    if launcher_env is not None:
        # Multi-node simulation (e.g. tests/docker-2node-ll): when more than one
        # node participates (WORLD_SIZE > 1), each MPI rank must map to a *per-node*
        # local device. Use MPI_LOCALRANKID (0..ppn-1) as the local rank and the
        # per-node process count as num_local_ranks; init_dist() derives the global
        # rank/world_size from RANK (node rank) and WORLD_SIZE (number of nodes).
        # Single-node runs keep the original global-launcher behavior.
        mpi_local_rank = os.environ.get('MPI_LOCALRANKID')
        num_nodes = int(os.environ.get('WORLD_SIZE', '1'))
        if mpi_local_rank is not None and num_nodes > 1:
            test_loop(int(mpi_local_rank), num_processes, args)
        else:
            rank, world_size = launcher_env
            test_loop(rank, world_size, args)
    else:
        torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
    # test_loop(num_processes, args)
