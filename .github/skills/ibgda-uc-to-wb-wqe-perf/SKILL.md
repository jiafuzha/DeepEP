# iSHMEM / IBGDA XPU perf tuning patterns (WQE UC→WB, quiet drain gate, commit-gate spin)

**Scope.** Perf-tuning tricks for `libishmem.a` device code (`src/ibgda_device_impl.h`,
`src/memory_ordering.cpp`, `src/ibgda.cpp`, `src/ibgda_types.h`, `src/ishmem/env_defs.h`)
that were proven on BMG + CX8/mlx5 with the docker-2node harnesses used by DeepEP's
`csrc/xpu/internode.cpp` and `csrc/xpu/internode_ll.cpp`. Distilled from four commits on
the `deepep_used_api_perf_tuning` branch:

| # | SHA prefix | Idea |
|---|---|---|
| 05 | `ea44077b` | UC WQE stores → WB stores; add `ISHMEM_IBGDA_QUIET_SKIP_DRAIN` env gate. |
| 06 | `db5ccd98` | Convert the last remaining UC WQE path (`rdma_atomic64`) to WB stores. |
| 07 | `c4aff87b` | `uc_load64` for the doorbell echo → plain `volatile` load (fence covers it). |
| 08 | `df53340e` | Commit-gate spin: `acquire` load → `relaxed` load + periodic `acquire` fence. |

Result on docker-2node-v3 (BMG + CX8 200GbE, from patch 05 commit msg):

    putmem_nbi_warp: 41.6 → 34.5 µs (-17%)
    quiet_qp:        38.3 → 32.8 µs (-14%)
    NBI batch 8×:   112.6 → 106.6 µs  (-5%)
    long_atomic_add: 76.1 → 70.7 µs  (-7%)

Patch 07 additionally shaves NBI batch −1.8%, AMO −4.5%. Patch 08 removes EU
serialization on shared-QP commit gates.

---

## Pattern 1 — UC WQE stores → WB stores (biggest win)

**Symptom in code.** Each 64-B WQE was published with 16 × `ishmemi_ibgda_uc_store32`.
Each UC store is a PCIe-MMIO round-trip (~0.5 µs), so a single WQE cost ~8 µs of
UC-store latency before the NIC even sees it.

**Fix.** Replace UC stores with plain (WB) stores. The doorbell function
(`ishmemi_ibgda_device_uc_uar_write`) already does a **system-scope `seq_cst`
fence before** the UAR LSC store, which flushes L3→VRAM so the NIC DMA reads a
fully-published WQE. Net: 16 UC MMIO round-trips (~8 µs) → 16 WB stores (~0 µs)
+ one shared L3 flush already on the critical path.

    // BEFORE
    ishmemi_ibgda_uc_store32(&wqe[0], ishmemi_ibgda_htobe32(...));
    ishmemi_ibgda_uc_store32(&wqe[1], ishmemi_ibgda_htobe32(...));
    ... (14 more)

    // AFTER
    wqe[0] = ishmemi_ibgda_htobe32(...);
    wqe[1] = ishmemi_ibgda_htobe32(...);
    ... (14 more)

**Where to look.** Every WQE-body / ctrl+rdma+data segment builder:
- `ishmemi_ibgda_device_emit_direct_wqe_skeleton` (RDMA_WRITE / RDMA_WRITE_INL WQEs)
- `ishmemi_ibgda_device_write_wqe_body`
- `ishmemi_ibgda_device_rdma_atomic64` (atomic WQE — CAS / FADD)
- Any inline-data / DUMP / batch WQE builder (identify by consecutive `wqe[N]` writes).

**Rules for correctness.**
1. **Do NOT convert** `snd_dbr[1]` (SND_DBR update) or the UAR write — those must
   remain UC MMIO. iSHMEM UC-count after conversion: ~55 → ~11 (just SND_DBR / UAR).
2. The fence sequence you rely on:
   `release fence (system) → uc_store32(snd_dbr) → release fence (system) → uc_uar_write`.
   As long as that pattern is intact, WB WQE stores are safe. If a WQE builder is
   *not* followed by a doorbell (rare — used only by staged/batch paths that later
   ring one doorbell for many WQEs), also safe: the batch-final doorbell flushes.
3. NVSHMEM does the same thing (`st.relaxed.gpu.global.L1::no_allocate` for WQEs).

---

## Pattern 2 — `ISHMEM_IBGDA_QUIET_SKIP_DRAIN` env gate

**Symptom.** `ishmem_quiet()` / `device_quiet*()` end with an L3 invalidate +
UC-load spin over each peer's CQ buffer to drain PCIe posted writes. This
protects **GPU reads of NIC-received data**. In DeepEP's PUT-only paths
(dispatch / combine only push), the GPU never reads inbound RDMA payload after
`quiet`, so the drain is dead weight.

**Fix.** Add an env-gated skip:

    // ibgda_types.h — new context field
    uint32_t quiet_skip_drain = 0;

    // env_defs.h
    ISHMEMI_ENV_DEF(IBGDA_QUIET_SKIP_DRAIN, size_t, 0, "...");

    // ibgda.cpp — device-ctx init
    ctx->quiet_skip_drain = (ishmemi_params.IBGDA_QUIET_SKIP_DRAIN != 0) ? 1u : 0u;

    // In device_quiet / device_quiet_pe / device_quiet_qp AND memory_ordering.cpp:
    if (!ctx->quiet_skip_drain) {
        __builtin_IB_lsc_fence_global_untyped(LSC_FS_SYSTEM_ACQUIRE, LSC_FT_INVALIDATE);
        // spin-load peers[*].nic_cq_buf ... (existing drain code)
    }

**Enable via env** `ISHMEM_IBGDA_QUIET_SKIP_DRAIN=1` on PUT-only workloads
(DeepEP normal + LL kernels qualify — combine/dispatch never `GET`s remotely
delivered data before another PUT-then-quiet cycle).

**Safety.** Do not set it if the GPU reads data delivered by a peer's PUT into
local memory before the next PUT+quiet cycle re-establishes ordering. In DeepEP,
inter-node barriers surround the NIC-completed region, and the L3 invalidate
happens elsewhere on the read side, so it is safe.

---

## Pattern 3 — `uc_load64` for doorbell echo → plain `volatile` load

Once WQE stores are WB (Pattern 1), the release fence before the doorbell
already flushes L3→VRAM. The subsequent `uc_load64(last_wqe_ptr)` used only to
build the doorbell value is redundant UC MMIO. Replace:

    uint64_t db_val = ishmemi_ibgda_uc_load64(last_wqe_ptr);
    // →
    uint64_t db_val = *reinterpret_cast<const volatile uint64_t *>(last_wqe_ptr);

`volatile` (not `relaxed atomic`) is enough because the surrounding
`sycl::atomic_fence(release, system)` prevents compiler reordering.

**Where.** Every `quiet*` path that reads the last WQE's ctrl segment to build
a MLX5 non-BF doorbell (search: `ishmemi_ibgda_uc_load64(...last_wqe`, or the
retry path after ringing SND_DBR).

---

## Pattern 4 — commit-gate spin: `acquire` → `relaxed` + periodic `acquire` fence

**Symptom.** The commit-gate spin in `device_quiet_qp` and `put_nbi_warp`:

    while (commit.load(sycl::memory_order::acquire) != base) { /* spin */ }

Every acquire load issues a cache-coherence probe that invalidates the line on
every other EU. On BMG this serializes concurrent sub-groups on the same QP.

**Fix.** Relaxed load + periodic acquire fence:

    uint32_t spin_iters = 0;
    while (commit.load(sycl::memory_order::relaxed) != base) {
        if (++spin_iters > 64u) {
            spin_iters = 0;
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
        }
    }

For `device_quiet_qp` there are two loads (initial + inside loop). Do the same
partial pattern: initial `relaxed`, in-loop `relaxed` most of the time, every
64th iter do `acquire fence + acquire load`.

**Why it wins.** EU scheduler can switch to productive wavefronts during
relaxed-load spins; occasional fences guarantee forward progress without
constant coherence traffic.

> **Limits of this pattern.** Relaxing the spin makes the gate *cheaper*, not
> *safe*. The gate is still an unbounded wait on a predecessor sub-group and
> will hang outright when producers on one QP are not co-resident. See
> [`QP_SCALING_AND_COMMIT_GATE.md`](QP_SCALING_AND_COMMIT_GATE.md) for the
> NVSHMEM comparison, the QP-count/latency/stability trade-off table, and the
> structural fixes (atomic-scope selection and spin bounding).

---

## Application checklist (grep-driven)

    grep -n 'ishmemi_ibgda_uc_store32' src/ibgda_device_impl.h
    grep -n 'ishmemi_ibgda_uc_load64'  src/ibgda_device_impl.h
    grep -n 'commit_ref.load(sycl::memory_order::acquire)' src/ibgda_device_impl.h
    grep -n 'commit.load(sycl::memory_order::acquire)' src/ibgda_device_impl.h

For each grep hit:
- `uc_store32(&wqe[N], …)` where `wqe[]` is a 16-word MLX5 WQE builder block ⇒
  convert to `wqe[N] = …` (Pattern 1). **Skip** `&snd_dbr[…]` writes.
- `uc_load64(last_wqe_ptr)` or `uc_load64(db_wqe_ptr)` immediately before a
  `ishmemi_ibgda_device_uc_uar_write(...)` ⇒ replace with volatile deref (Pattern 3).
  **Skip** loads used to inspect ctrl16 for opcode extraction (`ctrl64 = ...`).
- Spin loops on `commit(_ref).load(acquire)` ⇒ apply Pattern 4.

For Pattern 2, only edit `ibgda_types.h`, `env_defs.h`, `ibgda.cpp` (context
init), and every `device_quiet*` / `ishmem_quiet` drain block.

---

## Validation harness (from session `fe593cee-dadd-40b1-83ad-872546a3c171`)

1. Rebuild iSHMEM **inside** the docker container (host oneAPI 2026.0 breaks
   torch build; container has 2025.3):

       cd /root/jiafuzha/code-repo/zjf2012/DeepEP
       docker rm -f deepep-v2-node0 deepep-v2-node1 deepep-ll-v2-node0 deepep-ll-v2-node1
       docker run --rm -v /root/jiafuzha:/root/jiafuzha:rw -w /root/jiafuzha \
           deepep_jiafuzha bash -lc 'bash /root/jiafuzha/ishmem_ibgda/_build_ishmem.sh'

2. Rebuild DeepEP (its build script `rm -rf build`s so iSHMEM archive is re-extracted):

       docker run --rm -v /root/jiafuzha:/root/jiafuzha:rw \
         -w /root/jiafuzha/code-repo/zjf2012/DeepEP \
         deepep_jiafuzha bash -lc 'bash /root/jiafuzha/code-repo/zjf2012/DeepEP/_build_deepep.sh'

3. Internode-normal sweep (`csrc/xpu/internode.cpp`):

       cd /root/jiafuzha/code-repo/zjf2012/DeepEP/tests/docker-2node-v2
       rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/*ishmem* \
             /tmp/deep_ep_xpu_ipc_*.sock
       sleep 2
       TOKENS="32 64 128 256 1024 2048 4096" OUTDIR=_norm_v2 TIMEOUT_SEC=1500 \
           bash token_sweep.sh
       grep -c "===== PASS" _norm_v2/*.log
       grep -H "PERF rank=0" _norm_v2/*.log

   Per-token driver used (`token_sweep.sh` internals) — HIDDEN=7168, TOPK=2,
   EXPERTS=8, `DEEP_EP_NVL_BYTES=1073741824`, `DEEP_EP_RDMA_BYTES=536870912`,
   `ISHMEM_SYMMETRIC_SIZE=4294967296`, `DB_BATCH_SIZE` *unset* (defaults to 0).

4. Internode low-latency sweep (`csrc/xpu/internode_ll.cpp`):

       cd /root/jiafuzha/code-repo/zjf2012/DeepEP/tests/docker-2node-ll-v2
       docker rm -f deepep-ll-v2-node0 deepep-ll-v2-node1
       rm -f /dev/shm/sem.psm3* /dev/shm/psm3_* /dev/shm/sem.ishmem* /dev/shm/*ishmem* \
             /tmp/deep_ep_xpu_ipc_*.sock
       sleep 3
       TIMEOUT_SEC=1800 TOKENS="32 64 128 256 512 1024 2048 4096" OUTDIR=_ll_newish \
           bash ll_sweep.sh "auto:"
       grep -H "PERF rank=0\|avg_t" _ll_newish/*.log

5. Comparison: capture `avg_t` (LL) and RDMA/NVL bandwidth (normal) before and
   after; a −5 % to −17 % improvement on `putmem_nbi_warp` / `quiet_qp` is
   expected, translating to a few-µs `avg_t` drop on LL at small token counts.
