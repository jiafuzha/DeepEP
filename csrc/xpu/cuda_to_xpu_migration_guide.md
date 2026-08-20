# CUDA → XPU migration guide: named barriers and intranode P2P

**Purpose.** Reference for **re-migrating `csrc/xpu/internode.cpp`** (normal/high-throughput
internode) from `csrc/cuda_kernels/internode.cu`.

The current `csrc/xpu/internode.cpp` is a *functionally correct but structurally unfaithful*
port. This document derives the correct migration idioms from the one part of the codebase
that **was** migrated properly — intranode — and states exactly what to change.

Companion docs: `named_barrier_usage.md` (how the builtin works + how it was validated),
`internode_ll_design.md`. Perf evidence: `INTERNODE_PERF_ANALYSIS.md` in session state.

---

## 0. The reference pair (the gold standard)

| | CUDA | SYCL/XPU |
|--|--|--|
| **Good** | `csrc/cuda_kernels/intranode.cu` | `/root/jiafuzha/code-repo/DeepSymm/csrc/sycl/intranode.cpp` (+ `utils.hpp`) |
| **Bad** | `csrc/cuda_kernels/internode.cu` | `csrc/xpu/internode.cpp` |

### The gap, quantified

| File | CUDA `__global__` | SYCL kernels | ratio | named barriers |
|--|--|--|--|--|
| intranode (DeepEP xpu) | 5 | 6 | **1.2×** | 0 |
| intranode (**DeepSymm sycl**) | 5 | 6 | **1.2×** | **5** |
| **internode (DeepEP xpu)** | **4** | **38** | **9.5×** | **0** |
| internode (DeepSymm sycl) | 4 | 39 | 9.8× | 0 |

DeepSymm's internode has the *same* defect — it is **not** a reference for internode. Only its
`intranode.cpp` + `utils.hpp` are.

**Root cause of the 9.5× kernel blowup:** CUDA's internode kernels are single fused kernels using
**warp specialization** — different warps take different roles and rendezvous with
`barrier.sync <id>, <count>` on *subsets* of the block. Plain SYCL has only
`group_barrier(work_group)` (all work-items) and `group_barrier(sub_group)` (one sub-group) —
neither can sync an arbitrary subset. Lacking a subset barrier, the port **split every
inter-role rendezvous into a kernel boundary**, turning 1 kernel into ~10.

`work_group_named_barrier` *is* the missing primitive, it *is* available on BMG, and it is
already declared and validated in `csrc/xpu/xpu_kernels.hpp` — **but used nowhere.**

---

## 1. Rule 1 — kernel structure: 1 CUDA `__global__` → 1 SYCL kernel

CUDA `internode.cu` dispatch (line 482) and combine (line 1741) declare explicit roles:

```cpp
// dispatch
enum class WarpRole { kRDMASender, kRDMASenderCoordinator,
                      kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };
// combine
enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };
```

A warp computes its role from `warp_id`, then the kernel is a `switch` over roles. Roles run
**concurrently** and rendezvous only where the protocol requires it. Preserve this shape.

Consequences of the current split-kernel form that fusion fixes:
- Producer and consumer roles cannot overlap — a kernel boundary is a **full device barrier**.
- All per-role state must round-trip through global memory instead of living in registers/SLM.
- Loses the pipelining that makes the CUDA design fast (RDMA send overlapping NVL forward).

> Note: kernel-launch overhead itself is **only 3.2 µs** on this stack and is *not* the reason to
> fuse (measured; see `INTERNODE_PERF_ANALYSIS.md` Appendix B — a Tier-3 fusion removing one
> launch produced no measurable gain). **Fuse for the overlap and the register-resident state,
> not for the launch count.**

---

## 2. Rule 2 — barrier mapping

### 2.1 Mapping table

| CUDA | XPU / SYCL |
|--|--|
| `__syncthreads()` | `item.barrier(fence_space::local_space)` or `group_barrier(group)` |
| `__syncwarp()` | `group_barrier(item.get_sub_group())` (implicit on Xe within a sub-group) |
| `bar.sync 0, kNumThreads` (whole block) | `item.barrier(...)` |
| **`barrier.sync <id>, <count>`** (subset) | **`work_group_named_barrier(handle, flags)`** |
| `named_barrier_init(n)` | `n` = number of **participating SUB-GROUPS** (CUDA count / 32) |

CUDA sites to translate in `internode.cu`:

```cpp
// dispatch
sync_rdma_sender_smem()  : barrier.sync 0, (kNumDispatchRDMASenderWarps + 1) * 32
sync_forwarder_smem()    : barrier.sync 1, (NUM_MAX_NVL_PEERS + 1) * 32
// combine
sync_forwarder_smem()    : barrier.sync 0, (kNumForwarders + 1) * 32
sync_rdma_receiver_smem(): barrier.sync 1, (kNumRDMAReceivers + 1) * 32
per-dst-rank forwarder   : bar.sync (dst_rdma_rank + 2), kNumWarpsPerForwarder * 32
```

Note the last one uses a **data-dependent barrier id** (`dst_rdma_rank + 2`). IGC accepts up to
**32 named barriers per work-group** on Xe2/Xe3, so one handle per rdma-rank is affordable — but
see the SSA constraint below: you must `named_barrier_init` each one into its **own SSA local**
and select between them with a **literal if/else chain**, never an array index.

### 2.2 Fence flags

```cpp
constexpr unsigned kNamedBarrierLocalFence  = 0x1;  // CLK_LOCAL_MEM_FENCE
constexpr unsigned kNamedBarrierGlobalFence = 0x2;  // CLK_GLOBAL_MEM_FENCE
// 0x3 = both
```

DeepSymm's convention, worth copying verbatim:
- `0x1` — rendezvous that only publishes **SLM** values (e.g. leader polled a queue head into
  shared memory and the other warps must read it).
- `0x3` — after `memory_fence_system()`, before a leader publishes a **tail/head** pointer that
  makes cross-GPU payload writes visible.

### 2.3 ⚠ CRITICAL: the IGC SSA constraint

From DeepSymm `utils.hpp`:

> IGC's `NamedBarriersResolution` pass requires the `work_group_named_barrier` argument to be
> either (a) a direct `LoadInst` from a unique `alloca`, or (b) a direct `CallInst` (the
> `named_barrier_init` SSA result). Any **GEP / PHI / Select / SLM round-trip breaks the match
> and crashes the pass**. Therefore the init results MUST be plain SSA locals in the kernel body
> and the barrier dispatch MUST be a literal if/else over those locals.

**Therefore `csrc/xpu/xpu_kernels.hpp`'s `NamedBarrier` class is UNSAFE as written** — it stores
the handle in a **class member** (`handle_`), which is exactly the GEP round-trip that crashes
IGC. Do **not** build the re-migration on it. Use the raw pattern:

```cpp
// CORRECT — direct SSA local in the same function as every wait
auto* fwd_barrier = named_barrier_init(kNumForwarderSubgroups);
...
work_group_named_barrier(fwd_barrier, 0x1);
```

Also required: keep `init` and **every** `sync` in the **same function**. Passing a handle through
a function argument, a lambda capture, a class member, or an array element will break it.

Link-time "undefined function" warnings for the two symbols are **expected** — they are SPIR-V
builtins IGC resolves at JIT, not at LLVM link time.

### 2.4 The internode build guard — TESTED, NOT A BLOCKER

DeepSymm `utils.hpp:21`:

```cpp
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__) && !defined(DEEPSYMM_USE_INTERNODE)
```

Named barriers are **compiled out for internode builds**, which suggested a conflict between the
named-barrier path and the iSHMEM device link (RDC machinery perturbing the IGC
`NamedBarriersResolution` pass).

**This was tested directly and the hypothesis is REFUTED.** A reproducer
(`/tmp/nb_ishmem_probe.cpp`, kernel containing two `named_barrier_init` handles, a **subset**
barrier over 2 of 4 sub-groups, a full-WG barrier, and an `ishmem_int_p` device call, linked
against `libishmem.a`) compiles cleanly:

* `-fsycl-targets=spir64_gen -Xs "-device pvc,bmg,arl-h,mtl-h,lnl-m,ptl-h,ptl-u"` →
  **`Build succeeded`** for all 7 devices.
* IGC dump (`IGC_ShaderDumpEnable=1`) shows the barriers were **really lowered**, not stubbed:
  `4 nbarrier.signal`, `4 nbarrier.wait`, `9 NBarrier` in the emitted vISA.

⚠ **Do not be fooled by the warnings.** The build emits
`warning: Undefined function _Z18named_barrier_initi ... may result in runtime errors` (and the
same for `work_group_named_barrier`). These are **benign** — `sycl-post-link` emits them for any
externally-resolved SPIR-V builtin, and `ishmem_int_p` produces an identical warning. IGC
resolves them afterwards. `xpu_kernels.hpp:11-15` documents these exact mangled names.

**Conclusion: named barriers and iSHMEM device calls coexist fine, including subset barriers and
AOT for `bmg`. The fusion work below is unblocked.** The reason for the DeepSymm guard is
something else (possibly stale, or a runtime issue never traced to compilation); it should not
be copied into DeepEP without its own evidence.

Reproducer command, for re-verification:

```
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -I$ISHMEM_DIR/include \
  nb_ishmem_probe.cpp -L$ISHMEM_DIR/lib -l:libishmem.a -lze_loader -lmpi -lmpicxx \
  -lmpifort -lhwloc -o nb_probe_bmg
# add IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/tmp/igcdump, then:
grep -rhoiE "nbarrier" /tmp/igcdump | sort | uniq -c
```

---

## 3. Rule 3 — intranode P2P access: use LSC ld/st assembly

This is the second half of the user's ask, and it is where the current internode is weakest.

### 3.1 PTX → vISA mapping

| Purpose | CUDA PTX (`utils.cuh`) | XPU vISA (DeepSymm `utils.hpp`) |
|--|--|--|
| Bulk load, bypass L1, keep L2/L3 | `ld.global.nc.L1::no_allocate.L2::256B` | `lsc_load.ugm.uc.ca (M1,32) %0:d32 flat[%1]:a64` |
| Bulk store, bypass L1 | `st.global.L1::no_allocate` | `lsc_store.ugm.wb.wb (M1,32) flat[%0]:a64 %1:d32` |
| **16-byte vector** load | `ld...nc....v4.s32 {..}` (`int4`) | `lsc_load.ugm.uc.ca (M1,32) %0:d32x4 flat[%1]:a64` |
| **16-byte vector** store | `st...v4.s32` | `lsc_store.ugm.wb.wb (M1,32) flat[%0]:a64 %1:d32x4` |
| Volatile flag load | `ld.volatile.global` | `lsc_load.ugm.uc.ca ... :d32` (`ld_volatile_global`) |
| Volatile flag store | `st.volatile.global` | `lsc_store.ugm.uc.wb ... :d32` (`st_volatile_global`) |
| System release fence | `fence.release.sys` | `atomic_fence(acq_rel, system)` + `lsc_fence.ugm.evict.sysrel` |

Cache-control suffix decoder: `.uc` = uncached at that level, `.ca` = cached, `.wb` = write-back.
So `uc.ca` (load) = **bypass L1, cache in L3**; `wb.wb` (store) = write-back both;
`uc.wb` (flag store) = **bypass L1** so a remote poller sees it promptly.

Helper names are identical to CUDA's on purpose — `ld_nc_global`, `st_na_global`,
`ld_nc_global_v`, `st_na_global_v`, `ld_volatile_global`, `st_volatile_global` — so CUDA code
transliterates almost line-for-line.

### 3.2 The unrolled cooperative copy

```cpp
UNROLLED_WARP_COPY(UNROLL, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)      // 32 lanes
UNROLLED_TWOWARP_COPY(UNROLL, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)   // 64 lanes (2 sub-groups)
```

These stage **`UNROLL` values into registers first, then store them all** — creating `UNROLL`
outstanding loads per lane. That memory-level parallelism is what hides PCIe latency. DeepSymm's
dispatch sender uses:

```cpp
UNROLLED_TWOWARP_COPY(5, thread_in_rank, hidden_int4_,
                      shifted_channel_x_buffers, shifted_x,
                      ld_nc_global_v, st_na_global_v);
```

i.e. **5-deep unroll, 16-byte vector ops, 64 cooperating lanes.**

### 3.3 What `internode.cpp` does instead — and why it is suboptimal

Current bulk P2P copy (`faithful_coop_copy`, `csrc/xpu/internode.cpp:267`):

```cpp
for (size_t j = lane; j < n16; j += lanes)
    d16[j] = s16[j];        // plain C++ assignment
```

Problems, in order of severity:

1. **No LSC cache control.** Plain assignment lets the compiler pick default cached
   loads/stores. The whole point of `.uc.ca` / `.wb.wb` is to avoid polluting/relying on L1 for
   data crossing PCIe.

   ⚠ **`csrc/xpu/` has no LSC helper library at all.** There is no `ld_nc_global`,
   `st_na_global`, `ld/st_volatile_global`, `UNROLLED_WARP_COPY` or `UNROLLED_TWOWARP_COPY`
   anywhere in the DeepEP XPU tree — the only inline vISA is a single
   `lsc_load.ugm.uc.uc ... :d32` in `xpu_kernels.hpp:142` backing `uc_load`, used for **flags**.
   In `internode.cpp` the counts are `uc_load` 14, `uc_store` 2, and `UNROLLED_WARP_COPY` **0**
   (the one grep hit is a *comment* at line 2358 claiming the copy "matches CUDA's int4
   UNROLLED_WARP_COPY" — it does not; the actual call is `faithful_coop_copy`).

   So the payload path currently has **no cache control and no vector LSC ops whatsoever**.
2. **No register staging / unroll.** One load→store dependency at a time per lane. With
   `row_bytes = 14336` over 512 lanes that is only **~1.75 int4 per lane per token** — the loop
   ends before any pipeline fills, so latency is exposed on **every token**.
3. **Metadata is single-work-item.** In `FaithfulDispatchFwdWriteKernel` the entire
   `topk_idx` / `topk_weights` / `x_scales` copy runs under `if (local_id == 0)` as a scalar
   loop of `uc_load`s. DeepSymm spreads the identical work across a sub-group's 32 lanes
   (`lane_in_warp < num_topk`, `for (i = lane_in_warp; i < num_scales_; i += 32)`).
   **Uncached scalar loads in a serial loop is close to the worst possible pattern.**
4. **Serial cross-token dependency.** `peer_offsets[peer]++` forces a strict token-by-token
   walk, preventing any batching. CUDA/DeepSymm get the same effect from a per-channel
   `cached_channel_tail_idx` that is advanced **uniformly by all lanes**, so the copy stays
   fully parallel.

**Measured consequence** (`INTERNODE_PERF_ANALYSIS.md` Appendix F): the size-proportional part of
dispatch runs at **~0.6 GB/s**, while the same GPUs sustain **22.6 GB/s** of kernel-driven P2P
and the NIC sustains 19 GB/s. That is a **~37× shortfall**, and it is *not* a bandwidth-contention
problem — total demand is ~1% of capacity.

### 3.4 The cross-GPU barrier

CUDA `barrier_block` → DeepSymm `barrier_block_bypass` (`utils.hpp`). Pattern: each rank
`st_volatile_global`s an incrementing epoch into its own slot, `memory_fence_system()`, then
spins with `ld_volatile_global` on the peer slot until `>= next_epoch`. Note it uses the
**volatile (uncached) LSC variants** for both the store and the poll — a cached load here can
spin forever on a stale line.

---

## 4. Re-migration plan for `csrc/xpu/internode.cpp`

**Step 0 — ✅ DONE, PASSED.** The named-barrier / iSHMEM coexistence question (§2.4) was tested
and **passed**: AOT `spir64_gen` for all 7 devices including `bmg`, with IGC emitting real
`nbarrier.signal`/`nbarrier.wait`. **The plan is unblocked; proceed directly to Step 1.**

**Step 1 — ✅ DONE, but MEASURED NEUTRAL. Landed as a prerequisite, not as a win.**
`csrc/xpu/` had none of these helpers, so this was a real porting task. Added to
`xpu_kernels.hpp`: `int4_t`, `ld_nc_global_v` (`lsc_load.ugm.uc.ca ... :d32x4`),
`st_na_global_v` (`lsc_store.ugm.wb.wb ... :d32x4`), and `UNROLLED_GROUP_COPY` (a
work-group-wide analogue of DeepSymm's 32-lane `UNROLLED_WARP_COPY`).
`faithful_coop_copy` now does a 4-deep register-staged LSC copy instead of
`d16[j] = s16[j]`; `faithful_coop_zero` uses `st_na_global_v`.

⚠ **Result: no measurable gain.** Both tests PASS, no regression, but:

| | baseline `min` | with LSC | Δ |
|--|--|--|--|
| NT=32 round_trip | 3875.7 µs | 3882.9 µs | +7.2 µs (noise floor ≈35 µs) |
| NT=1024 round_trip | 101177.5 µs | 101458.7 µs | +0.28% |
| NT=1024 dispatch | 72133.3 µs | 71986.1 µs | −0.2% |

**This is the expected outcome and it confirms the perf model:** the path is
**NIC-latency-bound**, not copy-bandwidth-bound. Payload movement at line rate is
only ~2% of a dispatch (~56 µs of 2876 µs at NT=32), so making the copy faster
cannot move the total. **Do not expect Steps 1-4 to pay off through bandwidth.**
The remaining ~1 ms of unexplained per-call fixed cost is a *rendezvous/latency*
problem (flag polling, deferred doorbells, `barrier_all` at ~39-48 µs each), and
that is where the next investigation should go — see `INTERNODE_PERF_ANALYSIS.md`.

Step 1 is still worth keeping: it is correct, faithful to CUDA's `ld.global.nc` /
`st.global.na` semantics, regression-free, and it provides the helper library that
Steps 3-4 need.

**Step 2.** Parallelize the metadata: move the `local_id == 0` topk/scales loops to lane-strided
sub-group work.

**Step 3.** Only then fuse. Reintroduce the CUDA `WarpRole` split in dispatch and combine,
replacing each kernel-boundary rendezvous with a named barrier. Do it **incrementally**, one
role-pair at a time, re-running the sweep after each.

**Step 4.** Remove the two `ishmemx_barrier_all_work_group` calls if the fused structure lets the
flag-zeroing rendezvous be expressed locally (measured cost: **39–48 µs each**; see Appendix D
for why they could not be removed in the split-kernel form).

### Validation

- Harness `tests/docker-2node-v2`, 4 ranks, H=7168, TOPK=2, EXPERTS=8, DB=8.
- `DEEP_EP_PERF_TOKENS="32,64,128,512,1024,2048,4096"`; compare **`round_trip` min**.
- Scale `ISHMEM_SYMMETRIC_SIZE` with `DEEP_EP_NVL_BYTES`/`DEEP_EP_RDMA_BYTES` or you get
  `ishmem_align failed`.
- Baseline to beat: `normal_perf_qp1_default.log` (NT=32 round_trip min **3843.4 µs**).
- **NT=128 fails pre-existing** (`x_diff=2.974172e-02`) — reproduces at HEAD, not a regression.
- Noise floor: NT=32 dispatch stdev 15.2 µs, range 35.6 µs. Ignore deltas below ~35 µs.

### Expected payoff — REVISED AFTER MEASUREMENT

The original estimate here was that Appendix F's decomposition (dispatch = **~0.8–1.25 ms fixed
cost** + a stream running at **0.6 GB/s vs 22.6 GB/s achievable**) made Steps 1-2 a ~10×
opportunity on the streaming term.

**Step 1 was implemented and measured, and that estimate did not hold: the gain was zero**
(NT=32 +7.2 µs, NT=1024 +0.28% — all inside noise). The reason is that the "0.6 GB/s stream" is
not a *bandwidth* limit at all. Payload movement at line rate accounts for only ~2% of a
dispatch (~56 µs of 2876 µs at NT=32); the apparent low bandwidth is latency and rendezvous
serialization being *amortized over* the byte count, not a slow copy. Making the copy faster
therefore cannot move the total.

**Revised guidance, in priority order:**

1. **The ~1 ms per-call fixed rendezvous cost is the only target that matters.** Only ~130 µs of
   it is accounted for (2× `barrier_all` @ 39-48 µs, ~24 launches @ 3.2 µs, ~56 µs on the wire).
   **~1 ms is still unexplained** — most likely deferred-doorbell / flag-poll stalls. *Profile
   this before writing any more optimization code.*
2. **Steps 3-4 (fusion) attack the fixed term** and remain plausible, but bound the expectation
   honestly: at 3.2 µs/launch, collapsing ~34 kernels is worth **~110 µs (~2.8%)**, not 10×. A
   38→4 refactor is high-risk for that return — do step 1 of the profiling above first.
3. **Steps 1-2 (streaming) are done/low-value.** Keep Step 1 for correctness and as the helper
   library Steps 3-4 need; do not expect throughput from Step 2.

The broader lesson, consistent with every other experiment in `INTERNODE_PERF_ANALYSIS.md`
(launch overhead, single-work-group, PCIe contention — all disproven by measurement): **on this
stack NIC round-trips dominate everything else. Measure before optimizing.**

---

## 5. Quick reference — files

| What | Where |
|--|--|
| Good SYCL intranode (gold standard) | `DeepSymm/csrc/sycl/intranode.cpp` |
| LSC ld/st + copy macros + `barrier_block_bypass` | `DeepSymm/csrc/sycl/utils.hpp` |
| CUDA intranode original | `csrc/cuda_kernels/intranode.cu` |
| CUDA internode original (re-migration source) | `csrc/cuda_kernels/internode.cu` |
| CUDA PTX helpers | `csrc/cuda_kernels/utils.cuh` (`LD_NC_FUNC`, `ST_NA_FUNC`) |
| Target to rewrite | `csrc/xpu/internode.cpp` |
| Named-barrier decl (⚠ class wrapper is unsafe) | `csrc/xpu/xpu_kernels.hpp` |
| Named-barrier validation notes | `csrc/xpu/named_barrier_usage.md` |
| Perf evidence + measurement methodology | `INTERNODE_PERF_ANALYSIS.md` (session state) |
