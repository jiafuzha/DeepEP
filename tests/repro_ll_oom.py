#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
# repro_ll_oom.py — MINIMAL, FAST reproducer for the residual DeepEP-LL mid-test
# UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY (level_zero err 39) that fires inside
# buffer.low_latency_dispatch while the GPU has ~22.7 GiB free.
#
# It combines the three trigger ingredients an iSHMEM-only substrate probe could
# NOT reproduce (see ishmem_ibgda/test/unit/oom_copy_engine_probe.cpp — 0/42):
#   1) real iSHMEM IBGDA init + symmetric heap via deep_ep.Buffer(low_latency_mode=True)
#   2) torch XPU caching-allocator segment growth via the REAL torch::empty output
#      allocs inside low_latency_dispatch/low_latency_combine
#   3) the heavy LL mega-kernels (fp8 quant, top-k, warp-specialized poll/quiet)
#
# It loops dispatch->combine tightly in-process for ITERS iterations (warm) so the
# caching-allocator segments grow/churn and any transient bcs-engine wedge has a
# chance to hit the allocation path.  On the OOM it prints iteration, failing
# tensor shape/bytes, free/total XPU VRAM, IBGDA stats (post_put_nbi/doorbell —
# to prove PRE-doorbell timing), and the dmesg tail.
#
# ISOLATION KNOBS (env):
#   REPRO_MODE : full | dispatch_only | alloc_churn_only | no_ishmem   (default full)
#     full             = dispatch + combine (mega-kernels + allocator + ishmem)
#     dispatch_only    = dispatch only (mega-kernel + allocator + ishmem, no combine)
#     alloc_churn_only = torch.empty churn of the SAME output shapes on comm/compute
#                        streams, WITH ishmem initialized but NO mega-kernels
#     no_ishmem        = torch allocator churn + fp8/topk compute, NO deep_ep.Buffer
#   ITERS       : iterations of the tight warm loop                    (default 200)
#   HIDDEN      : hidden dim (real failure was 7168)                   (default 7168)
#   NUM_TOKENS  : tokens per rank                                      (default 32)
#   NUM_EXPERTS : experts                                              (default 8)
#   NUM_TOPK    : top-k                                                (default 2)
#   DUAL_STREAM : 1=overlap a compute kernel on a 2nd xpu stream       (default 1)
#   FP8_ALTERNATE: 1=alternate use_fp8 each iter to hit quant kernel   (default 1)
#   STOP_ON_OOM : 1=stop at first OOM                                  (default 1)
#   HEARTBEAT   : print a free-mem line every N iters                  (default 20)
#
# Launched exactly like tests/test_low_latency.py by the LL docker runner (mpirun
# -n 4 -ppn 2, per-rank GPU/NIC pinning); accepts the same CLI flags and ignores
# the extra ones.  No ishmem_finalize (matches DeepEP).
#
# ---------------------------------------------------------------------------
# FINDINGS (2026-07, this reproducer):
#   * iSHMEM IBGDA init + real dispatch+combine with FIXED shapes does NOT
#     reproduce (8 warm launches x 200 iters = 1600 dispatch+combine, 0 wedge):
#     torch's XPU caching allocator REUSES cached segments, so no fresh
#     zeMemAllocDevice / VM_BIND happens after warmup -> no race.
#   * Allocator GROWTH churn (CHURN_MB) alone also does not reproduce: the
#     allocator grows ONCE then reuses the grown segments (free 22.7->15.5 GiB,
#     stable across loops) -> still no continuing VM_BIND.
#   * The trigger is a RACE between a FRESH zeMemAllocDevice VM_BIND (page-table
#     update) and concurrent bcs/blitter copy-engine + IBGDA activity. Force it
#     with EMPTY_CACHE_EVERY=1 (releases segments so the next alloc must
#     re-reserve). Then ~1/6 warm launches wedge: dmesg shows exactly
#     `Engine reset: engine_class=bcs` + `VM worker error: -16` (EBUSY) +
#     `Suspend fence failed to respond` -- the residual-OOM signature.
#   * The EBUSY surfaces as UR err-39 OUT_OF_DEVICE_MEMORY when it lands on a
#     torch zeMemAllocDevice (the dispatch output alloc, as in the real test),
#     and as a per-rank HANG when it lands on a collective/copy op. Both are the
#     SAME transient wedge; memory is genuinely free (GUARD_ALLOC proves the
#     exact output-shape alloc succeeds when the engine is healthy).
#   MINIMAL INGREDIENT SET = iSHMEM IBGDA init + real dispatch copy-engine
#     activity + FRESH zeMemAllocDevice/VM_BIND churn (EMPTY_CACHE_EVERY=1).
#     Fixed-shape reuse (no fresh VM_BIND) => never reproduces.
#   FIX DIRECTION: eliminate fresh zeMemAllocDevice on the LL comm path by
#     PRE-ALLOCATING & REUSING the dispatch/combine output+scratch tensors
#     (persistent buffers), and/or RETRY-on-transient (err-39 / VM-worker EBUSY)
#     in buffer.py low_latency_dispatch/combine. The wedge is a driver-level
#     VM_BIND-vs-copy-engine race, not a lost doorbell (0 puts/0 doorbells at
#     failure) -- so no iSHMEM doorbell/fence change addresses it.
#
#   REPRO RECIPE (best):
#     LOOPS=6 REPRO_MODE=full ITERS=300 EMPTY_CACHE_EVERY=1 GUARD_ALLOC=1 \
#       DUAL_STREAM=1 STOP_ON_OOM=1 ./run_repro_oom.sh
# ---------------------------------------------------------------------------

import argparse
import ctypes
import os
import subprocess
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_ep
from utils import init_dist, get_accelerator_device_type, per_token_cast_back


def envi(name, dflt):
    v = os.getenv(name)
    try:
        return int(v) if v not in (None, '') else dflt
    except ValueError:
        return dflt


def rank_print(rank, msg):
    print(f'[rank {rank}] {msg}', flush=True)


def free_total_gib():
    try:
        free, total = torch.xpu.mem_get_info()
        return free / 2**30, total / 2**30
    except Exception as e:
        return -1.0, -1.0


def read_ibgda_stats(rank):
    d = os.getenv('ISHMEM_IBGDA_STATS_DIR', '')
    if not d:
        return 'ISHMEM_IBGDA_STATS_DIR not set (no stats)'
    path = os.path.join(d, f'ibgda_stats_pe{rank}.txt')
    try:
        with open(path) as f:
            txt = f.read()
    except Exception as e:
        return f'(could not read {path}: {e})'
    keys = ('stats_post_put_nbi_calls', 'stats_doorbell_writes', 'flags',
            'db_mode', 'stats_cq_completed', 'direct_inflight')
    lines = [ln.strip() for ln in txt.splitlines()
             if any(k in ln for k in keys)]
    return ' | '.join(lines[:8]) if lines else '(stats file present but no key fields)'


def dmesg_tail():
    try:
        out = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5).stdout
        rel = [ln for ln in out.splitlines()
               if any(k in ln for k in ('bcs', 'ccs', 'reset', 'VM worker', 'lr_cleanup', 'guc'))]
        return '\n    '.join(rel[-8:]) if rel else '(no relevant dmesg lines)'
    except Exception as e:
        return f'(dmesg unavailable: {e})'


def is_oom(err):
    s = str(err)
    return ('OUT_OF_DEVICE_MEMORY' in s) or ('error: 39' in s) or ('error 39' in s)


def report_failure(rank, it, where, shape, dtype, err):
    free, total = free_total_gib()
    try:
        nbytes = torch.empty(0, dtype=dtype).element_size() * (
            int(torch.tensor(shape).prod().item()) if shape else 0)
    except Exception:
        nbytes = -1
    print(f'\n[rank {rank}] *** {("OOM(39)" if is_oom(err) else "DEVICE ERR")} '
          f'at iter={it} in {where} ***', flush=True)
    print(f'    failing tensor: shape={shape} dtype={dtype} bytes={nbytes}', flush=True)
    print(f'    XPU VRAM: FREE={free:.3f} GiB / TOTAL={total:.3f} GiB '
          f'-> memory was {"NOT" if free > 1.0 else "possibly"} exhausted', flush=True)
    print(f'    IBGDA stats: {read_ibgda_stats(rank)}', flush=True)
    print(f'    dmesg tail:\n    {dmesg_tail()}', flush=True)
    print(f'    exception: {err}', flush=True)


# --------------------------------------------------------------------------- #
def build_inputs(num_tokens, hidden, num_experts, num_topk, rank, dev, seed):
    torch.manual_seed(seed + rank)
    rank_offset = 128
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device=dev) * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device=dev).to(torch.bfloat16).view(-1, 1)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=dev).abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=dev).abs()
    return x, topk_idx, topk_weights


def do_dispatch(buffer, x, topk_idx, num_tokens, num_experts, use_fp8):
    packed_recv_x, packed_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                    use_fp8=use_fp8, round_scale=False, use_ue8m0=False,
                                    async_finish=True, return_recv_hook=False)
    event.current_stream_wait()
    return packed_recv_x, packed_recv_count, handle


def alloc_like_dispatch(num_local_experts, num_recv_slots, hidden, num_ranks,
                        num_tokens, use_fp8, dev, stream=None):
    """torch.empty the SAME output shapes low_latency_dispatch allocates, WITHOUT
    running any mega-kernel — to test whether caching-allocator growth alone is
    the trigger."""
    ctx = torch.xpu.stream(stream) if stream is not None else _null_ctx()
    outs = []
    with ctx:
        if use_fp8:
            outs.append(torch.empty((num_local_experts, num_recv_slots, hidden),
                                    dtype=torch.float8_e4m3fn, device=dev))
            outs.append(torch.empty((num_local_experts, num_recv_slots, hidden // 128),
                                    dtype=torch.float32, device=dev))
        else:
            outs.append(torch.empty((num_local_experts, num_recv_slots, hidden),
                                    dtype=torch.bfloat16, device=dev))
        outs.append(torch.empty((num_local_experts,), dtype=torch.int32, device=dev))
        outs.append(torch.empty((num_local_experts, num_recv_slots), dtype=torch.int32, device=dev))
        outs.append(torch.empty((num_local_experts, num_ranks), dtype=torch.int64, device=dev))
        # combine output
        outs.append(torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=dev))
        # a scalar .item() D2H like recv_count.item()
        _ = outs[-3].flatten()[0].item() if outs[-3].numel() else 0
    return outs


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-processes', type=int, default=2)
    ap.add_argument('--num-tokens', type=int, default=envi('NUM_TOKENS', 32))
    ap.add_argument('--hidden', type=int, default=envi('HIDDEN', 7168))
    ap.add_argument('--num-topk', type=int, default=envi('NUM_TOPK', 2))
    ap.add_argument('--num-experts', type=int, default=envi('NUM_EXPERTS', 8))
    ap.add_argument('--disable-nvlink', action='store_true')
    ap.add_argument('--allow-mnnvl', action='store_true')
    args, _unknown = ap.parse_known_args()

    mode = os.getenv('REPRO_MODE', 'full')
    iters = envi('ITERS', 200)
    dual_stream = envi('DUAL_STREAM', 1)
    fp8_alt = envi('FP8_ALTERNATE', 1)
    stop_on_oom = envi('STOP_ON_OOM', 1)
    heartbeat = envi('HEARTBEAT', 20)
    churn_mb = envi('CHURN_MB', 0)          # per-iter fragmenting allocator growth (MiB)
    churn_vary = envi('CHURN_VARY', 1)      # 1=vary sizes each iter to force segment growth
    empty_cache_every = envi('EMPTY_CACHE_EVERY', 0)  # torch.xpu.empty_cache() every N iters
                                                       # (forces zeMemFree+fresh zeMemAllocDevice
                                                       #  VM_BIND churn concurrent w/ copy engine)
    guard_alloc = envi('GUARD_ALLOC', 0)    # 1=guarded fresh output-shape alloc post empty_cache
                                            #   (catch the err-39 at the real OOM site)

    num_tokens, hidden = args.num_tokens, args.hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    dev = get_accelerator_device_type()

    # ---- distributed / device init (matches test_low_latency) ----
    mpi_local_rank = int(os.environ.get('MPI_LOCALRANKID', '0'))
    rank, num_ranks, group = init_dist(mpi_local_rank, args.num_processes)
    num_local_experts = num_experts // num_ranks
    num_recv_slots = num_ranks * num_tokens

    rank_print(rank, f'START repro_ll_oom MODE={mode} iters={iters} hidden={hidden} '
                     f'num_tokens={num_tokens} num_experts={num_experts} num_ranks={num_ranks} '
                     f'dual_stream={dual_stream} fp8_alt={fp8_alt}')
    f0, t0 = free_total_gib()
    rank_print(rank, f'VRAM at start: FREE={f0:.3f} / TOTAL={t0:.3f} GiB')

    buffer = None
    if mode != 'no_ishmem':
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
        buffer = deep_ep.Buffer(group,
                                num_rdma_bytes=num_rdma_bytes,
                                low_latency_mode=True,
                                num_qps_per_rank=num_experts // num_ranks,
                                allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
                                explicitly_destroy=True,
                                allow_mnnvl=args.allow_mnnvl)
        rank_print(rank, f'Buffer ready (rdma_bytes={num_rdma_bytes/1e6:.1f} MB)')

    second_stream = torch.xpu.Stream() if dual_stream else None

    x, topk_idx, topk_weights = build_inputs(num_tokens, hidden, num_experts, num_topk, rank, dev, seed=1)

    exit_code = 0
    reproduced = False
    _churn_keep = []
    for it in range(iters):
        use_fp8 = bool((it % 2) and fp8_alt)
        # Force fresh zeMemAllocDevice / VM_BIND churn: release cached segments so the
        # NEXT allocation must re-reserve device memory (a VM_BIND page-table update)
        # concurrent with the copy/blitter engine — the exact race that yields the
        # transient VM-worker-EBUSY spurious OOM in the real LL test.
        if empty_cache_every and (it % empty_cache_every) == 0:
            _churn_keep.clear()
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass
            # GUARDED fresh output-shape allocation at the REAL OOM site: right after
            # releasing segments, re-reserve the exact dispatch output tensors via a
            # fresh zeMemAllocDevice/VM_BIND while the prior iter's IBGDA/copy engine
            # drains. If the VM-worker EBUSY lands here it raises a catchable err-39
            # (instead of hanging a downstream collective).
            if guard_alloc:
                try:
                    _g = alloc_like_dispatch(num_local_experts, num_recv_slots, hidden,
                                             num_ranks, num_tokens, use_fp8, dev,
                                             stream=second_stream)
                    del _g
                except RuntimeError as e:
                    report_failure(rank, it, 'guarded output-shape alloc (post empty_cache)',
                                   [num_local_experts, num_recv_slots, hidden], torch.bfloat16, e)
                    exit_code = 39 if is_oom(e) else 2
                    reproduced = True
                    if stop_on_oom:
                        break
        # Fragmenting allocator-growth churn: force torch's XPU caching allocator to
        # keep reserving FRESH zeMemAllocDevice segments (the segment-expansion path
        # the fixed-shape reuse loop never hits). Vary sizes so cached blocks don't
        # satisfy the request; keep a rotating set alive so reserved memory grows.
        if churn_mb > 0:
            try:
                base = churn_mb * (2 ** 20)
                # vary the size each iter (±37%) so the allocator can't reuse a cached block
                if churn_vary:
                    jitter = 1.0 + 0.37 * ((it % 7) - 3) / 3.0
                    nbytes = max(1 << 16, int(base * jitter))
                else:
                    nbytes = base
                nel = nbytes // 2  # bfloat16
                t = torch.empty((nel,), dtype=torch.bfloat16, device=dev)
                t.normal_()  # touch it (forces real backing + copy/compute engine)
                _churn_keep.append(t)
                # keep a bounded rotating working set alive to drive reserved growth
                if len(_churn_keep) > 6:
                    _churn_keep.pop(0)
            except RuntimeError as e:
                report_failure(rank, it, 'churn torch.empty', [nbytes], torch.bfloat16, e)
                exit_code = 39 if is_oom(e) else 2
                reproduced = True
                if stop_on_oom:
                    break
        # optional dual-stream overlapping compute (like comm/compute overlap)
        if second_stream is not None:
            with torch.xpu.stream(second_stream):
                _spin = torch.randn((num_tokens, hidden), device=dev, dtype=torch.bfloat16)
                _spin = (_spin @ _spin[: hidden if hidden <= num_tokens else num_tokens].T if False else _spin * 1.0001)
        dbg = it < 3
        try:
            where = shape = None
            if mode in ('full', 'dispatch_only'):
                where = 'low_latency_dispatch'
                shape = [num_local_experts, num_recv_slots, hidden]
                if dbg: rank_print(rank, f'iter={it} -> dispatch use_fp8={use_fp8}')
                packed_recv_x, packed_recv_count, handle = \
                    do_dispatch(buffer, x, topk_idx, num_tokens, num_experts, use_fp8)
                if dbg: rank_print(rank, f'iter={it} <- dispatch done')
                # a D2H .item() like the real test (first op that sees a wedge)
                _ = packed_recv_count[0].item()
                if mode == 'full':
                    where = 'low_latency_combine'
                    if dbg: rank_print(rank, f'iter={it} -> combine')
                    if use_fp8:
                        x0, sc = packed_recv_x
                        simulated = per_token_cast_back(x0.view(-1, hidden),
                                                        sc.view(-1, sc.size(-1))).view(x0.shape)
                    else:
                        simulated = packed_recv_x.clone()
                    out = None if os.getenv('REPRO_COMBINE_PERSIST', '1') == '1' else \
                        torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=dev)
                    combined_x, event, hook = buffer.low_latency_combine(
                        simulated, topk_idx, topk_weights, handle,
                        async_finish=True, zero_copy=False, return_recv_hook=False, out=out)
                    event.current_stream_wait()
                    _ = combined_x[0, 0].item()
                    if dbg: rank_print(rank, f'iter={it} <- combine done')
            elif mode == 'alloc_churn_only':
                where = 'alloc_churn (torch.empty output shapes, no mega-kernel)'
                shape = [num_local_experts, num_recv_slots, hidden]
                outs = alloc_like_dispatch(num_local_experts, num_recv_slots, hidden,
                                           num_ranks, num_tokens, use_fp8, dev,
                                           stream=second_stream)
                torch.xpu.synchronize()
                del outs
            elif mode == 'no_ishmem':
                where = 'no_ishmem (torch alloc + fp8/topk compute, no ishmem)'
                shape = [num_local_experts, num_recv_slots, hidden]
                big = torch.empty((num_local_experts, num_recv_slots, hidden),
                                  dtype=torch.bfloat16, device=dev)
                big.normal_()
                # fp8 quant + topk mimic
                amax = big.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
                q = (big / amax * 448.0).to(torch.float8_e4m3fn)
                sc = (amax / 448.0).float()
                back = per_token_cast_back(q.view(-1, hidden), sc.view(-1, 1)).view(big.shape)
                sm = torch.randn((num_tokens, num_experts), device=dev)
                _ = torch.topk(sm, num_topk, dim=-1)[1]
                torch.xpu.synchronize()
                _ = back[0, 0, 0].item()
                del big, q, sc, back
        except RuntimeError as e:
            report_failure(rank, it, where, shape, torch.bfloat16, e)
            exit_code = 39 if is_oom(e) else 2
            reproduced = True
            if stop_on_oom:
                break
        except Exception as e:
            rank_print(rank, f'iter={it} NON-RUNTIME EXC in {where}: {e}')
            exit_code = 2
            reproduced = True
            if stop_on_oom:
                break

        if heartbeat and (it % heartbeat) == 0:
            f, t = free_total_gib()
            rank_print(rank, f'iter={it} OK  free={f:.3f} GiB  ibgda[{read_ibgda_stats(rank)}]')

    if reproduced:
        rank_print(rank, f'RESULT: REPRODUCED (exit={exit_code}) MODE={mode}')
    else:
        f, t = free_total_gib()
        rank_print(rank, f'RESULT: completed {iters} iters, NO OOM (did not reproduce) '
                         f'MODE={mode} free={f:.3f} GiB')

    # ---- teardown: quiesce + minimal orderly exit (matches test_low_latency mode 2) ----
    try:
        if buffer is not None and os.getenv('DEEP_EP_LL_QUIESCE', '1') != '0':
            torch.xpu.synchronize()
            buffer.quiesce()
    except Exception as e:
        rank_print(rank, f'quiesce raised {e}')
    try:
        dist.destroy_process_group()
    except Exception:
        pass
    try:
        settle = float(os.getenv('DEEP_EP_LL_EXIT_SETTLE_SEC', '2'))
        if settle > 0:
            time.sleep(settle)
        libmpi = ctypes.CDLL('libmpi.so')
        initd = ctypes.c_int(); find = ctypes.c_int()
        libmpi.MPI_Initialized(ctypes.byref(initd)); libmpi.MPI_Finalized(ctypes.byref(find))
        if initd.value and not find.value:
            libmpi.MPI_Finalize()
    except Exception:
        pass
    rank_print(rank, 'DONE (no ishmem_finalize)')
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
