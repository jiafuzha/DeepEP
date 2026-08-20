#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "xpu_kernels.hpp"

#include <sycl/ext/oneapi/experimental/builtins.hpp>

#ifdef DEEP_EP_ENABLE_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

namespace deep_ep {
namespace internode {
namespace {

// Work-group size for iSHMEM collective kernels (putmem_nbi_work_group, quiet_work_group, etc.).
// Matches NVSHMEM's `_block` API pattern: a whole WG cooperates on payload assembly and
// posts one work request, amortizing doorbell overhead.
constexpr int kIshmemWGSize = 32;
// Work-group size for the pure-compute combine/dispatch stages (Pack, RdmaSend,
// Reduce). These stages issue NO ishmem collective/AMO/put, so they are NOT
// constrained to a single 32-wide work-group; they are launched as a grid of
// one work-group per (combined) token to use the whole GPU, mirroring the CUDA
// multi-block design. Wider WG also shortens the strided hidden-dimension loop.
constexpr int kComputeWGSize = 512;

class DispatchInitKernel;
class DispatchCountExchangeKernel;
class DispatchPackKernel;
class DispatchOffsetComputeKernel;
class DispatchPayloadKernel;

class DispatchQueueResetKernel;
class DispatchQueueCopyKernel;
class DispatchCopyKernel;

class CombinedDispatchInitKernel;
class CombinedDispatchPackKernel;
class CombinedDispatchPackBarrierKernel;
class CombinedDispatchStageKernel;
class CombinedDispatchStageBarrierKernel;
class CombinedDispatchRdmaSendKernel;
class CombinedDispatchRdmaPutKernel;
class CombinedDispatchFwdWriteKernel;
class CombinedDispatchFwdBarrierKernel;
class CombinedDispatchAssembleKernel;
class CombinedDispatchHeadKernel;
class DispatchChannelCountsKernel;

// ---- Faithful (CUDA-parity) dispatch transport (the only transport) ----
// Collapses the serial single_task Pack/Stage/RdmaSend/RdmaPut into fewer,
// work-group/sub-group-parallel kernels and swaps the scalar blocking put +
// ishmemx_barrier_all_work_group data-path sync for ishmemx_putmem_nbi_subgroup +
// ishmemx_fence_qp + 64-bit ishmemx_long_atomic_add_qp flags (LL pattern).
class FaithfulDispatchPackStageKernel;
class FaithfulDispatchPackBarrierKernel;
class FaithfulDispatchRdmaBarrierKernel;
class FaithfulDispatchRdmaSendKernel;
class FaithfulDispatchRdmaFlagKernel;
class FaithfulDispatchFwdWriteKernel;
class FaithfulDispatchFwdBarrierKernel;
// ---- Producer-PUSH intra-node NVL transport (gap #2: eliminate cross-rank IPC READS) ----
// Each producer WRITES its intra-node tokens into the DESTINATION peer's recv-staging
// region; the consumer reads only its OWN region locally. See dispatch_nvl_rdma below.
class FaithfulDispatchNvlPushKernel;
class FaithfulDispatchNvlPushBarrierKernel;
class FaithfulDispatchCountsBarrierKernel;

// ---- Faithful-path flag/poll helpers (mirror internode_ll.cpp) ----------------
// Bounded spin cap for the 64-bit cross-PE arrival flag on the RDMA-forward
// (F6-FwdWrite) receive side. On a cold/first dispatch the NIC-delivered AMO
// flag reliably lands at ~3-4 MILLION spins, so the previous hard-coded 4M cap
// sat mid-distribution and silently dropped in-flight flags -> RDMA-token
// under-delivery (the "56 != 25" bug). 50M gives ~11x cold-start headroom while
// still bounded so a genuinely-never-landing flag fails fast. Mirrors
// internode_ll.cpp ll_poll_cap() exactly. Override with DEEP_EP_INTERNODE_POLL_CAP.
inline uint64_t internode_poll_cap() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_POLL_CAP");
    if (env != nullptr && env[0] != '\0') {
        long long v = std::atoll(env);
        if (v > 0)
            return static_cast<uint64_t>(v);
    }
    return static_cast<uint64_t>(50) * 1000 * 1000;
}

// Flag-read primitive selector (mirror internode_ll.cpp ll_flag_lsc_mode).
// >=2 => issue an explicit system-scope acquire/invalidate fence before the
// uc_load so the GPU L2 becomes coherent with the NIC AMO write. Default 2 here
// (dispatch correctness on 2-node depends on the acquire before decode).
inline int internode_flag_lsc_mode() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_FLAG_LSC");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v >= 0)
            return v;
    }
    return 2;
}

// A/B knobs for the faithful RDMA send/flag path (see F4a2/F4b). Read once on the
// host and captured into the kernels by value.
//   DEEP_EP_INTERNODE_FORCE_DB (default 1/ON): ring the F4a2 payload put's doorbell
//     immediately (force_db=true) so the put self-egresses & completes every
//     dispatch, instead of deferring the doorbell to F4b's quiet_qp. Decouples the
//     put's NIC completion from the AMO path (candidate fix for the leader hang in
//     F4a2-Put on a later cumulative dispatch). The AMO stays in its own kernel
//     (F4b) so this does NOT reintroduce the fused-kernel AMO-egress bug.
//   DEEP_EP_INTERNODE_POST_AMO_QUIET (default 0/OFF): add ONE targeted
//     ishmemx_fence_qp(dst_pe,0) AFTER the AMO in F4b to reap the AMO's QP
//     completion (targeted quiet is LL-safe; NOT the full ishmem_quiet that wedged).
inline bool internode_force_db() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_FORCE_DB");
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    return true;  // default ON (most-likely fix)
}

inline bool internode_post_amo_quiet() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_POST_AMO_QUIET");
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    return false;  // default OFF (LL does NOT quiet after the AMO; the AMO's own
                   // QP completion + the following op's put doorbell suffice)
}

// DEEP_EP_INTERNODE_ENTRY_QUIET (default 0/OFF): at F4a2-Put entry, issue a
// targeted ishmemx_fence_qp(dst_pe,0) per dst_rdma to drain any residue left on
// qp0 by the interleaved serial combine (blocking ishmem_putmem) or a prior
// faithful AMO before posting new qp0 ops. Targeted quiet_qp only (NOT full
// ishmem_quiet, which wedged). NOTE: quiet_qp doorbells + polls the CQ but does
// NOT advance nic_wq_commit, so it does NOT by itself clear a put_nbi_warp
// commit-gate gap (see internode_blocking_put below).
inline bool internode_entry_quiet() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_ENTRY_QUIET");
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    return false;  // default OFF
}

// DEEP_EP_INTERNODE_BLOCKING_PUT (default 0/OFF): use the SAME blocking scalar
// ishmem_putmem (single WI) for the F4a2 payload put that the fallback and the
// serial combine use, instead of ishmemx_putmem_nbi_subgroup. RATIONALE: the warp
// put's ordered-commit gate spins UNBOUNDED on `nic_wq_commit == base`, while
// the blocking skeleton put (emit_direct_wqe_skeleton) only advances nic_wq_cnt
// and self-completes via its internal ishmem_quiet -- it has NO unbounded commit
// gate. Since the interleaved serial combine posts skeleton WQEs on qp0 that
// advance nic_wq_cnt WITHOUT advancing nic_wq_commit, a later warp put's gate can
// wedge (base=cnt while commit lags). The fallback never hangs precisely because
// it never uses the warp-put commit gate. The F4b AMO's own gate is BOUNDED
// (MAX_AMO_DB_ITERS) and its Step-6 CAS-max reconciles commit->cnt, so keeping
// the AMO in F4b is safe. This is the most robust candidate fix.
inline bool internode_blocking_put() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_BLOCKING_PUT");
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    return false;  // default OFF: use the CUDA-faithful warp-collective put
                   // (ishmemx_putmem_nbi_subgroup). This is SAFE now that COMBINE is also
                   // faithful: every qp0 payload put is followed by a reconciling AMO
                   // (dispatch F4b / combine FC5c ishmemx_long_atomic_add_qp, whose
                   // Step-6 CAS-max raises nic_wq_commit->nic_wq_cnt), so the warp-put's
                   // ordered-commit gate never wedges. HW-validated on both nodes.
                   // Set =1 to force the gate-free single-WI put as an A/B fallback
                   // (dispatch F4a2 / combine FC5b). The put API is selected by
                   // DEEP_EP_INTERNODE_NBI_MODE below.
}

// DEEP_EP_INTERNODE_NBI_MODE: when DEEP_EP_INTERNODE_BLOCKING_PUT=1, selects the
// single-WI put API used in dispatch F4a2 and combine FC5b.
//   0:           ishmem_putmem       (blocking, default)
//   1:           ishmem_putmem_nbi   (non-blocking, faster dispatch)
inline bool internode_nbi_mode() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_NBI_MODE");
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    return false;  // default: ishmem_putmem (blocking)
}

// DEEP_EP_INTERNODE_PAR_GATHER (default ON): parallelize the combine leader gather
// (FC5b) across a GRID of work-groups instead of a single 512-wide leader WG looping
// all rows serially. Each input token (peer,t) maps to exactly one dst_rdma region
// (= its src_rdma_rank); a device-scope atomic on that region's rdma_count grabs the
// output slot, then the WG cooperatively copies the row. Order among rows is
// irrelevant because each row carries its own recv_pos/src_nvl metadata and the final
// reduce is a commutative SUM. This turned the ~16 ms serial gather (measured
// FC5b-Gather @1024 tok) into a full-GPU parallel copy. Set =0 for the serial
// single-WG gather (A/B fallback, byte-identical ordering).
inline bool internode_par_gather() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_PAR_GATHER");
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    return true;
}

// CUDA-faithful per-GPU RDMA transport (the ONLY inter-node model). CUDA
// internode.cu has NO leader: every GPU runs kRDMASender and issues its OWN RDMA to
// the SAME-nvl-slot peer on the remote node (dst_pe = dst_rdma*num_nvl_ranks +
// nvl_rank), and the NVL all-to-all happens on the RECEIVE side
// (kRDMAAndNVLForwarder pushes received tokens to the correct local NVL peer, keyed
// by the source nvl-plane so concurrent forwarders don't collide). Each nvl_rank:
// (a) zeroes its own receive flags, (b) compacts ONLY its own packed tokens per
// dst_rdma and RDMA-puts on its own NIC, (c) posts its own count flag, and (d) polls
// its own received stream and forwards to local peers into a per-source-plane fwd
// slice. This uses all num_nvl_ranks NICs per node. The intra-node NVL gather (Pack +
// Assemble peer reads) is unchanged.

// Number of RC QP-channels the RDMA payload put is striped across (CUDA-faithful
// multi-QP; qp_id == channel, mirroring internode.cu:818/835 where the put + tail AMO
// ride qp_id == channel_id). CUDA runs num_channels (= num_sms/2) SM-channels, each
// sending its own token range on its own QP. The XPU RDMA region is field-major
// (all-x | meta | idx | wt | scales) rather than token-major like CUDA's per-channel
// SymBuffer, so the equivalent NIC parallelism is realized by splitting the contiguous
// payload put [0,rdma_count_offset) into C 16B-aligned byte chunks, chunk c issued by
// sub-group c on qp=c (concurrent NIC send queues), then per-c ishmemx_fence_qp(dst,c)
// + ishmemx_long_atomic_add_qp(flag[c],...,c). RC in-order keeps flag[c] after chunk c
// on the SAME qp; the receiver waits for all C flags.
//
// C is bound to the QPs actually provisioned by iSHMEM: it reads ISHMEM_IBGDA_QPS_PER_PE
// (which deep_ep/buffer.py auto-sets to clamp_pow2(num_channels) for normal internode),
// so qp_id ∈ [0,C) never exceeds the provisioned QP pool. iSHMEM rounds QPS_PER_PE to a
// power of 2 and clamps [1,16]; C is additionally capped at kComputeWGSize/32 (one
// sub-group per QP-channel). QPS_PER_PE unset (or 1) => single-QP, byte-for-byte the
// original layout.
inline int internode_num_qp_channels() {
    const char* env = std::getenv("ISHMEM_IBGDA_QPS_PER_PE");
    int c = 1;
    if (env != nullptr && env[0] != '\0') c = std::atoi(env);
    if (c < 1) c = 1;
    int p = 1;
    while (p * 2 <= c) p *= 2;  // clamp down to power of 2 (matches iSHMEM provisioning)
    c = p;
    const int max_ch = kComputeWGSize / 32;  // one sub-group drives one QP-channel
    if (c > max_ch) c = max_ch;
    return c;
}

// Contiguous byte-chunk boundary for QP-channel `c` of `C` over a payload of `L`
// bytes. start(0)=0, start(C)=L, interior boundaries rounded UP to 16B so each
// chunk is 16B-aligned and the chunks exactly tile [0,L) with no gap/overlap.
inline size_t internode_qp_chunk_start(size_t L, int c, int C) {
    if (c <= 0) return 0;
    if (c >= C) return L;
    size_t v = (L * static_cast<size_t>(c)) / static_cast<size_t>(C);
    v = (v + 15) & ~static_cast<size_t>(15);
    return v > L ? L : v;
}

// 64-bit flag read (mirror internode_ll.cpp ll_read_flag64): optional sysacq
// invalidate fence then a hint-based uncached uc_load<long>. 0 == not arrived;
// -count-1 once the NIC-delivered AMO lands after the payload.
inline long internode_read_flag64(const long* p, int lsc_mode) {
#ifdef __SYCL_DEVICE_ONLY__
    if (lsc_mode >= 2) {
        deep_ep::lsc_fence_sysacq();
    }
#endif
    return deep_ep::uc_load<long>(p);
}

// Cooperative 16-byte-vectorized copy of `n` bytes across `lanes` cooperating
// work-items (lane in [0, lanes)). The prior byte-at-a-time stage copy issued
// ~n individual 1-byte P2P/IPC stores when staging a non-leader rank's send
// buffer into the leader's slot -> multi-second F2 stall. 16-byte vector stores
// cut the transaction count 16x and coalesce the cross-device write. Falls back
// to a byte tail for any non-16B-aligned remainder. Mirrors internode_ll.cpp
// coop_copy_bytes.
inline void faithful_coop_copy(uint8_t* dst, const uint8_t* src, size_t n, int lane, int lanes) {
    size_t done = 0;
    if ((reinterpret_cast<uintptr_t>(dst) & 0xF) == 0 && (reinterpret_cast<uintptr_t>(src) & 0xF) == 0) {
        const size_t n16 = n >> 4;
        auto* d16 = reinterpret_cast<int4_t*>(dst);
        auto* s16 = reinterpret_cast<const int4_t*>(src);
        // Register-staged LSC copy: 4 loads issued before any store, so PCIe/L3
        // latency overlaps instead of serializing one dependent load-store pair
        // per iteration (the old `d16[j] = s16[j]` form). `.uc.ca`/`.wb.wb` keep
        // this payload out of L1 while staying L3-cacheable.
        UNROLLED_GROUP_COPY(4, lane, lanes, n16, d16, s16, ld_nc_global_v, st_na_global_v);
        done = n16 << 4;
    }
    for (size_t b = done + static_cast<size_t>(lane); b < n; b += static_cast<size_t>(lanes))
        dst[b] = src[b];
}

inline void faithful_coop_zero(uint8_t* dst, size_t n, int lane, int lanes) {
    size_t done = 0;
    if ((reinterpret_cast<uintptr_t>(dst) & 0xF) == 0) {
        const size_t n16 = n >> 4;
        auto* d16 = reinterpret_cast<int4_t*>(dst);
        const int4_t z{0u, 0u, 0u, 0u};
        for (size_t j = static_cast<size_t>(lane); j < n16; j += static_cast<size_t>(lanes))
            st_na_global_v(d16 + j, z);
        done = n16 << 4;
    }
    for (size_t b = done + static_cast<size_t>(lane); b < n; b += static_cast<size_t>(lanes))
        dst[b] = 0;
}

// ---- Reusable combine_token reduction (mirrors CUDA combine_token without TMA) ----
// Reduces tokens across up to kNumSrcRanks source buffers into one output row +
// topk_weights. The CUDA original uses warp-scope shfl to gather heads; here each
// WG owns a single combined output row, so the heads/slot_indices are already
// per-WG (broadcast or precomputed).  kDtypePerInt4 == sizeof(int4)/sizeof(dtype_t)
// (8 for bfloat16). Bias accumulation is handled separately by the caller.
template <int kNumSrcRanks, typename dtype_t>
inline void combine_token_sycl(
    const int* __restrict__ head_indices,   // [kNumSrcRanks]: head slot per source
    const int* __restrict__ slot_indices,   // [kNumSrcRanks]: slot index per source
    const bool* __restrict__ is_source_active, // [kNumSrcRanks]: whether source contributes
    const int hidden,                         // element count (not int4 count)
    const int num_topk,
    dtype_t* __restrict__ combined_row,
    float* __restrict__ combined_topk_weights,
    const uint8_t* const* __restrict__ source_bases, // [kNumSrcRanks] base pointers
    const size_t row_bytes,                  // bytes per row in source buffer
    const float* const* __restrict__ source_topk_bases, // [kNumSrcRanks] topk_weight bases
    int local_id,
    int wg_size) {
    constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
    // --- hidden-dimension reduce (per-lane strided) ---
    for (int h = local_id; h < hidden; h += wg_size) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < kNumSrcRanks; ++s) {
            if (!is_source_active[s]) continue;
            const auto* src_row = reinterpret_cast<const dtype_t*>(
                source_bases[s] + static_cast<size_t>(slot_indices[s]) * row_bytes);
            acc += static_cast<float>(src_row[h]);
        }
        combined_row[h] = static_cast<dtype_t>(acc);
    }
    // --- topk_weights reduce ---
    if (combined_topk_weights != nullptr) {
        for (int k = local_id; k < num_topk; k += wg_size) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < kNumSrcRanks; ++s) {
                if (!is_source_active[s]) continue;
                acc += source_topk_bases[s][slot_indices[s] * num_topk + k];
            }
            combined_topk_weights[k] = acc;
        }
    }
}

// ---- Reusable combine_token for variable-rank (dynamic kNumSrcRanks) ----
// Same as above but kNumSrcRanks is a runtime parameter, so the inner loop
// cannot be unrolled. Use this when the source rank count varies per token.
template <typename dtype_t>
inline void combine_token_sycl_variable(
    const int* __restrict__ head_indices,
    const int* __restrict__ slot_indices,
    const bool* __restrict__ is_source_active,
    int num_src_ranks,
    const int hidden,
    const int num_topk,
    dtype_t* __restrict__ combined_row,
    float* __restrict__ combined_topk_weights,
    const uint8_t* const* __restrict__ source_bases,
    const size_t row_bytes,
    const float* const* __restrict__ source_topk_bases,
    int local_id,
    int wg_size) {
    for (int h = local_id; h < hidden; h += wg_size) {
        float acc = 0.0f;
        for (int s = 0; s < num_src_ranks; ++s) {
            if (!is_source_active[s]) continue;
            const auto* src_row = reinterpret_cast<const dtype_t*>(
                source_bases[s] + static_cast<size_t>(slot_indices[s]) * row_bytes);
            acc += static_cast<float>(src_row[h]);
        }
        combined_row[h] = static_cast<dtype_t>(acc);
    }
    if (combined_topk_weights != nullptr) {
        for (int k = local_id; k < num_topk; k += wg_size) {
            float acc = 0.0f;
            for (int s = 0; s < num_src_ranks; ++s) {
                if (!is_source_active[s]) continue;
                acc += source_topk_bases[s][slot_indices[s] * num_topk + k];
            }
            combined_topk_weights[k] = acc;
        }
    }
}

template <typename dtype_t>
class CombinedCombinePackKernel;
template <typename dtype_t>
class CombinedCombinePackBarrierKernel;
template <typename dtype_t>
class CombinedCombineInitKernel;
template <typename dtype_t>
class CombinedCombineRdmaSendKernel;
template <typename dtype_t>
class CombinedCombineRdmaPushKernel;
template <typename dtype_t>
class CombinedCombineFwdWriteKernel;
template <typename dtype_t>
class CombinedCombineFwdBarrierKernel;
template <typename dtype_t>
class CombinedCombineReduceKernel;

// Faithful combine: the serial RdmaPush kernel is
// split into an init barrier + gate-free blocking payload put + kernel-boundary
// 64-bit AMO count flag, and FwdWrite polls that flag (mirrors dispatch F4a0/
// F4a2/F4b/F6). Init/Pack/PackBarrier/RdmaSend/FwdBarrier/Reduce are reused
// verbatim so the reduce math (and thus the output) stays byte-identical.
template <typename dtype_t>
class CombineNvlPlaneInitKernel;
template <typename dtype_t>
class FaithfulCombineRdmaBarrierKernel;
template <typename dtype_t>
class FaithfulCombineRdmaPutKernel;
template <typename dtype_t>
class FaithfulCombineRdmaPut2Kernel;
template <typename dtype_t>
class FaithfulCombineGatherZeroKernel;
template <typename dtype_t>
class FaithfulCombineParGatherKernel;
template <typename dtype_t>
class FaithfulCombineRdmaFlagKernel;
template <typename dtype_t>
class FaithfulCombineFwdWriteKernel;

// Producer-PUSH intra-node NVL combine transport (gap #2, mirrors dispatch nvlrecv):
// each rank PUSHES its Pack output into every peer's combine-staging plane; the gather
// kernels read their OWN staging plane (local read) instead of peer send buffers.
template <typename dtype_t>
class CombineNvlPushKernel;
template <typename dtype_t>
class CombineNvlPushBarrierKernel;

template <typename dtype_t>
class CombineInitKernel;

template <typename dtype_t>
class CombinePackKernel;

template <typename dtype_t>
class CombinePayloadKernel;

template <typename dtype_t>
class CombinePayloadQuietKernel;

template <typename dtype_t>
class CombineQueueResetKernel;

template <typename dtype_t>
class CombineQueueCopyKernel;

template <typename dtype_t>
class CombineBiasKernel;

}  // namespace

size_t align_offset(size_t offset, size_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

// ===========================================================================
// Shared NVL helpers for the combined NVL + RDMA internode path
// (nvl_barrier device barrier and NvlBufferLayout buffer offsets).
// ===========================================================================

// Runtime-dispatched NVL barrier: routes to barrier_block<N> for the actual peer count.
SYCL_EXTERNAL inline void nvl_barrier(int** barrier_signal_ptrs, int rank, int signal, int num_peers, sycl::nd_item<1> item) {
    switch (num_peers) {
        case 1:
            barrier_block<1>(barrier_signal_ptrs, rank, signal, item);
            break;
        case 2:
            barrier_block<2>(barrier_signal_ptrs, rank, signal, item);
            break;
        case 3:
            barrier_block<3>(barrier_signal_ptrs, rank, signal, item);
            break;
        case 4:
            barrier_block<4>(barrier_signal_ptrs, rank, signal, item);
            break;
        case 5:
            barrier_block<5>(barrier_signal_ptrs, rank, signal, item);
            break;
        case 6:
            barrier_block<6>(barrier_signal_ptrs, rank, signal, item);
            break;
        case 7:
            barrier_block<7>(barrier_signal_ptrs, rank, signal, item);
            break;
        case 8:
            barrier_block<8>(barrier_signal_ptrs, rank, signal, item);
            break;
        default:
            break;
    }
}

// ===========================================================================
// FUSED, warp-specialized CUDA-faithful dispatch/combine (see the .inc files for
// the full CUDA<->SYCL mapping).  This is the only implementation; the historical
// phase-split port is preserved untouched in csrc/xpu/internode_old.cpp.
// ===========================================================================
#include "internode_dispatch_fused.inc"

#include "internode_notify_fused.inc"
#include "internode_combine_fused.inc"

struct NvlBufferLayout {
    // Offsets within each rank's NVL buffer (buffer_ptrs[nvl_rank])
    size_t count_offset = 0;          // int[num_ranks]: per-destination token counts
    size_t channel_count_offset = 0;  // int[num_ranks * num_channels]
    size_t send_x_offset = 0;         // uint8[num_tokens * row_bytes]: token payload
    size_t send_meta_offset = 0;      // SourceMeta[num_tokens]
    size_t send_topk_idx_offset = 0;
    size_t send_topk_weights_offset = 0;
    size_t send_x_scales_offset = 0;
    size_t send_dst_token_offset = 0;     // int[num_tokens]: compact destination row
    size_t send_routing_bits_offset = 0;  // int[num_tokens]: per-token NVL destination bitmask
    size_t send_rdma_dest_bits_offset = 0;
    size_t send_is_token_in_rank_offset = 0;  // bool[num_tokens * num_ranks]
    size_t total_bytes = 0;

    NvlBufferLayout(int num_tokens, int num_ranks, int num_channels, size_t row_bytes, int num_topk, int num_scales) {
        size_t off = 0;
        auto add_aligned = [&](size_t bytes, size_t align = 128) {
            off = align_offset(off, align);
            size_t result = off;
            off += bytes;
            return result;
        };
        count_offset = add_aligned(static_cast<size_t>(num_ranks) * sizeof(int), alignof(int));
        channel_count_offset = add_aligned(static_cast<size_t>(num_ranks) * num_channels * sizeof(int), alignof(int));
        send_x_offset = add_aligned(static_cast<size_t>(num_tokens) * row_bytes, 128);
        send_meta_offset = add_aligned(static_cast<size_t>(num_tokens) * sizeof(SourceMeta), alignof(SourceMeta));
        send_topk_idx_offset = add_aligned(static_cast<size_t>(num_tokens) * num_topk * sizeof(topk_idx_t), alignof(topk_idx_t));
        send_topk_weights_offset = add_aligned(static_cast<size_t>(num_tokens) * num_topk * sizeof(float), alignof(float));
        send_x_scales_offset = add_aligned(static_cast<size_t>(num_tokens) * num_scales * sizeof(float), alignof(float));
        send_dst_token_offset = add_aligned(static_cast<size_t>(num_tokens) * sizeof(int), alignof(int));
        send_routing_bits_offset = add_aligned(static_cast<size_t>(num_tokens) * sizeof(int), alignof(int));
        send_rdma_dest_bits_offset = add_aligned(static_cast<size_t>(num_tokens) * sizeof(int), alignof(int));
        send_is_token_in_rank_offset = add_aligned(static_cast<size_t>(num_tokens) * num_ranks * sizeof(bool), alignof(bool));
        total_bytes = align_offset(off, 128);
    }
};

// ======================================================================
// Combined NVL + RDMA internode dispatch
// ======================================================================
//
// Topology: num_ranks = num_rdma_ranks * num_nvl_ranks
//   rdma_rank = rank / NUM_MAX_NVL_PEERS
//   nvl_rank  = rank % NUM_MAX_NVL_PEERS
//
// Data flow for a token going from (src_rdma_rank, src_nvl_rank) to
// (dst_rdma_rank, dst_nvl_rank) where dst_rdma_rank != src_rdma_rank:
//   1. Source packs token into its NVL buffer with routing info
//   2. Source's RDMA peer (nvl_rank=src_nvl_rank on src node) packs into
//      iSHMEM buffer and sends to its peer on the destination node
//   3. Destination RDMA peer receives, forwards to local NVL peers
//   4. Destination NVL peer reads from its forwarding buffer
//
// For same-node tokens (dst_rdma_rank == src_rdma_rank):
//   Delivered via NVL IPC directly.

// NVL forwarding buffer layout: used for RDMA→NVL forwarding on the receive side.
// Each rank's NVL buffer (buffer_ptrs[nvl_rank]) is split into two regions:
//   Region A: NVL-local tokens (NvlBufferLayout for same-node NVL-local tokens)
//   Region B: RDMA-forwarded tokens
struct NvlForwardLayout {
    size_t fwd_x_offset = 0;     // uint8[num_planes * max_fwd_tokens * row_bytes]
    size_t fwd_meta_offset = 0;  // SourceMeta[num_planes * max_fwd_tokens]
    size_t fwd_topk_idx_offset = 0;
    size_t fwd_topk_weights_offset = 0;
    size_t fwd_x_scales_offset = 0;
    size_t fwd_count_offset = 0;  // int[num_planes * num_rdma_ranks]: per-(plane,src-rdma) count
    size_t total_bytes = 0;
    int plane_tokens = 0;  // per-plane token capacity (== max_fwd_tokens)

    // num_planes: number of disjoint source-nvl-plane slices (per-GPU RDMA forward
    // needs one slice per forwarding nvl_rank so concurrent forwarders don't collide;
    // legacy single-leader forward uses num_planes==1 and is byte-for-byte unchanged).
    NvlForwardLayout(int max_fwd_tokens, size_t row_bytes, int num_topk, int num_scales, int num_rdma_ranks,
                     int num_planes = 1) {
        plane_tokens = max_fwd_tokens;
        const size_t slots = static_cast<size_t>(max_fwd_tokens) * static_cast<size_t>(num_planes);
        size_t off = 0;
        auto add_aligned = [&](size_t bytes, size_t align = 128) {
            off = align_offset(off, align);
            size_t result = off;
            off += bytes;
            return result;
        };
        fwd_x_offset = add_aligned(slots * row_bytes, 128);
        fwd_meta_offset = add_aligned(slots * sizeof(SourceMeta), alignof(SourceMeta));
        fwd_topk_idx_offset = add_aligned(slots * num_topk * sizeof(topk_idx_t), alignof(topk_idx_t));
        fwd_topk_weights_offset = add_aligned(slots * num_topk * sizeof(float), alignof(float));
        fwd_x_scales_offset = add_aligned(slots * num_scales * sizeof(float), alignof(float));
        fwd_count_offset =
            add_aligned(static_cast<size_t>(num_rdma_ranks) * static_cast<size_t>(num_planes) * sizeof(int), alignof(int));
        total_bytes = align_offset(off, 128);
    }
};

// ============================================================================
// Combined NVL + RDMA: the ONLY internode normal implementation.
//
// Requires num_nvl_bytes > 0 AND num_rdma_ranks > 1; both are asserted in
// Buffer::internode_dispatch / internode_combine. Tokens are packed, sent over
// iSHMEM RDMA to the peer node's leader, then forwarded to local NVL peers.
//
// The former pure-iSHMEM `dispatch`/`combine` pair (global_rdma_mode,
// num_nvl_bytes == 0) has been removed.
//
// This path is fully per-QP: payload goes out
// via ishmemx_putmem_nbi_subgroup(..., qp_id, ...) and is released by
// ishmemx_fence_qp + ishmemx_long_atomic_add_qp on the SAME qp, so RC in-order
// delivery orders flag[c] behind chunk c. It deliberately never calls the
// global ishmem_quiet() (see the note at F-K3b).
// ============================================================================
// ============================================================================
// Exported CUDA-faithful `notify_dispatch` (see internode_notify_fused.inc).
// ============================================================================
void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_worst_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int rank,
                     int num_ranks,
                     int num_nvl_ranks,
                     sycl::queue& queue) {
    const int num_rdma_ranks = num_ranks / num_nvl_ranks;
#define DEEP_EP_NOTIFY_CASE(R)                                                                                        \
    case R:                                                                                                           \
        launch_fused_notify_dispatch<R>(num_tokens_per_rank, moe_recv_counter_mapped, num_tokens_per_rdma_rank,        \
                                        moe_recv_rdma_counter_mapped, num_tokens_per_expert,                          \
                                        moe_recv_expert_counter_mapped, num_experts, is_token_in_rank, num_tokens,     \
                                        num_worst_tokens, num_channels, expert_alignment, hidden_int4, num_scales,     \
                                        num_topk, num_max_rdma_chunked_recv_tokens, num_max_nvl_chunked_recv_tokens,   \
                                        rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,                         \
                                        gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, rdma_buffer_ptr,          \
                                        buffer_ptrs, rank, num_ranks, num_nvl_ranks, queue);                          \
        break
    switch (num_rdma_ranks) {
        DEEP_EP_NOTIFY_CASE(2);
        DEEP_EP_NOTIFY_CASE(4);
        DEEP_EP_NOTIFY_CASE(8);
        default:
            TORCH_CHECK(false, "internode notify_dispatch supports 2/4/8 RDMA ranks, got ", num_rdma_ranks);
    }
#undef DEEP_EP_NOTIFY_CASE
}

void dispatch_nvl_rdma(void* recv_x,
                       float* recv_x_scales,
                       topk_idx_t* recv_topk_idx,
                       float* recv_topk_weights,
                       void* recv_src_meta,
                       void* rdma_buffer_ptr,
                       const void* x,
                       const float* x_scales,
                       const topk_idx_t* topk_idx,
                       const float* topk_weights,
                       int* send_rdma_head,
                       int* send_nvl_head,
                       int* recv_rdma_channel_prefix_matrix,
                       int* recv_gbl_channel_prefix_matrix,
                       int* rdma_channel_prefix_matrix,
                       int* recv_rdma_rank_prefix_sum,
                       int* gbl_channel_prefix_matrix,
                       int* recv_gbl_rank_prefix_sum,
                       const int* num_tokens_per_rank,
                       const bool* is_token_in_rank,
                       int num_tokens,
                       int num_recv_tokens,
                       int hidden,
                       int element_size,
                       int num_topk,
                       int num_scales,
                       int num_channels,
                       int num_max_rdma_chunked_send_tokens,
                       int num_max_rdma_chunked_recv_tokens,
                       void** buffer_ptrs_gpu,
                       int** barrier_signal_ptrs_gpu,
                       int nvl_rank,
                       int num_nvl_ranks,
                       int barrier_signal_base,
                       int rank,
                       int num_ranks,
                       int num_experts,
                       int num_max_nvl_chunked_send_tokens,
                       int num_max_nvl_chunked_recv_tokens,
                       sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(recv_x != nullptr && x != nullptr, "dispatch_nvl_rdma requires input and output tensors");
    TORCH_CHECK(rdma_buffer_ptr != nullptr, "dispatch_nvl_rdma requires a symmetric iSHMEM RDMA buffer");
    TORCH_CHECK(is_token_in_rank != nullptr, "dispatch_nvl_rdma requires is_token_in_rank");

    // The pure RDMA-only (no NVL peers) implementation has been removed, so a
    // degenerate single-NVL-rank configuration is no longer supported here.
    TORCH_CHECK(num_nvl_ranks > 1,
                "dispatch_nvl_rdma requires num_nvl_ranks > 1; the RDMA-only internode path "
                "(e.g. DEEP_EP_NVL_RANKS=1) is no longer supported");

    TORCH_CHECK(buffer_ptrs_gpu != nullptr, "dispatch_nvl_rdma requires NVL buffer pointers");
    TORCH_CHECK(barrier_signal_ptrs_gpu != nullptr, "dispatch_nvl_rdma requires barrier signal pointers");
    TORCH_CHECK(num_nvl_ranks > 1 && num_nvl_ranks <= NUM_MAX_NVL_PEERS, "dispatch_nvl_rdma NVL peer count out of range");
    TORCH_CHECK(num_ranks % num_nvl_ranks == 0, "dispatch_nvl_rdma requires num_ranks divisible by num_nvl_ranks");

    const size_t row_bytes = static_cast<size_t>(hidden) * element_size;
    const int my_rdma_rank = rank / num_nvl_ranks;
    const int my_global_rank = my_rdma_rank * num_nvl_ranks + nvl_rank;
    const int num_rdma_ranks = num_ranks / num_nvl_ranks;
    TORCH_CHECK(num_rdma_ranks <= NUM_MAX_NVL_PEERS, "dispatch_nvl_rdma currently supports up to ", NUM_MAX_NVL_PEERS, " RDMA ranks");

    {
        const int hidden_int4 = static_cast<int>((static_cast<size_t>(hidden) * element_size) / sizeof(int4_t));
        TORCH_CHECK(static_cast<size_t>(hidden) * element_size % sizeof(int4_t) == 0,
                    "fused internode dispatch requires the token row to be 16B-aligned");
        const bool cached_mode = (send_rdma_head == nullptr);
        (void)fused_num_bytes_per_token(hidden_int4, num_scales, num_topk);
        // In non-cached mode `notify_dispatch` already cleaned both control planes
        // between two cross-PE barriers (CUDA internode.cu:153-166). In cached mode
        // CUDA does the same work in `cached_notify` (internode.cu:1327-1345), so it
        // is performed here, likewise fenced by a cross-PE barrier on both sides.
#define DEEP_EP_FUSED_DISPATCH_CASE(R)                                                                                \
    case R:                                                                                                           \
        if (cached_mode) {                                                                                            \
            const auto rc = fused_get_rdma_clean_meta(hidden_int4, num_scales, num_topk, R, num_nvl_ranks,             \
                                                      num_max_rdma_chunked_recv_tokens, num_channels);                \
            const auto nc = fused_get_nvl_clean_meta(hidden_int4, num_scales, num_topk, R, num_nvl_ranks,              \
                                                     num_max_nvl_chunked_recv_tokens, num_channels);                  \
            queue.wait();                                                                                             \
            ishmem_barrier_all();                                                                                     \
            launch_fused_clean_planes<R>(rdma_buffer_ptr, buffer_ptrs_gpu, nvl_rank, rc.first, rc.second, nc.first,    \
                                         nc.second, queue);                                                           \
            queue.wait();                                                                                             \
            ishmem_barrier_all();                                                                                     \
        }                                                                                                             \
        launch_fused_dispatch<R>(recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights,                              \
                                 static_cast<SourceMeta*>(recv_src_meta), x, x_scales, topk_idx, topk_weights,         \
                                 send_rdma_head, send_nvl_head, recv_rdma_channel_prefix_matrix,                       \
                                 recv_gbl_channel_prefix_matrix, rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,\
                                 gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, is_token_in_rank, num_tokens,    \
                                 hidden_int4, num_scales, num_topk, num_experts, rdma_buffer_ptr,                      \
                                 num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,  \
                                 num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, rank, num_ranks,    \
                                 num_nvl_ranks, num_channels, cached_mode, queue);                                    \
        break
        switch (num_rdma_ranks) {
            DEEP_EP_FUSED_DISPATCH_CASE(2);
            DEEP_EP_FUSED_DISPATCH_CASE(4);
            DEEP_EP_FUSED_DISPATCH_CASE(8);
            default:
                TORCH_CHECK(false, "fused internode dispatch supports 2/4/8 RDMA ranks, got ", num_rdma_ranks);
        }
#undef DEEP_EP_FUSED_DISPATCH_CASE
        queue.wait();
    }
#else
    TORCH_CHECK(false, "dispatch_nvl_rdma requires DEEP_EP_ENABLE_ISHMEM");
#endif
}

// Combined NVL+RDMA combine
void combine_nvl_rdma(DataType type,
                      void* combined_x,
                      float* combined_topk_weights,
                      const bool* is_combined_token_in_rank,
                      void* rdma_buffer_ptr,
                      const void* x,
                      const float* topk_weights,
                      const void* bias_0,
                      const void* bias_1,
                      int* combined_rdma_head,
                      int* combined_nvl_head,
                      const void* src_meta,
                      const int* rdma_channel_prefix_matrix,
                      const int* rdma_rank_prefix_sum,
                      const int* gbl_channel_prefix_matrix,
                      int num_tokens,
                      int num_combined_tokens,
                      int hidden,
                      int num_topk,
                      int num_max_rdma_chunked_send_tokens,
                      int num_max_rdma_chunked_recv_tokens,
                      void** buffer_ptrs_gpu,
                      int** barrier_signal_ptrs_gpu,
                      int nvl_rank,
                      int num_nvl_ranks,
                      int barrier_signal_base,
                      int rank,
                      int num_ranks,
                      int num_channels_arg,
                      int num_max_nvl_chunked_send_tokens,
                      int num_max_nvl_chunked_recv_tokens,
                      sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    // Mirrors dispatch_nvl_rdma: the RDMA-only fallback no longer exists.
    TORCH_CHECK(num_nvl_ranks > 1,
                "combine_nvl_rdma requires num_nvl_ranks > 1; the RDMA-only internode path "
                "(e.g. DEEP_EP_NVL_RANKS=1) is no longer supported");

    TORCH_CHECK(buffer_ptrs_gpu != nullptr, "combine_nvl_rdma requires NVL buffer pointers");
    TORCH_CHECK(barrier_signal_ptrs_gpu != nullptr, "combine_nvl_rdma requires barrier signal pointers");
    TORCH_CHECK(rdma_buffer_ptr != nullptr, "combine_nvl_rdma requires a symmetric iSHMEM RDMA buffer");
    TORCH_CHECK(num_nvl_ranks > 1 && num_nvl_ranks <= NUM_MAX_NVL_PEERS, "combine_nvl_rdma NVL peer count out of range");
    TORCH_CHECK(num_ranks % num_nvl_ranks == 0, "combine_nvl_rdma requires num_ranks divisible by num_nvl_ranks");
    TORCH_CHECK(type == DataType::kBFloat16, "Combined NVL+RDMA combine only supports BF16");
    using dtype_t = sycl::ext::oneapi::bfloat16;

    const int my_rdma_rank = rank / num_nvl_ranks;
    const int num_rdma_ranks = num_ranks / num_nvl_ranks;
    TORCH_CHECK(num_rdma_ranks <= NUM_MAX_NVL_PEERS, "combine_nvl_rdma currently supports up to ", NUM_MAX_NVL_PEERS, " RDMA ranks");

    {
        const int hidden_int4 = hidden / static_cast<int>(sizeof(int4_t) / sizeof(dtype_t));
        TORCH_CHECK(hidden % static_cast<int>(sizeof(int4_t) / sizeof(dtype_t)) == 0,
                    "fused internode combine requires the token row to be 16B-aligned");
        const int num_channels = num_channels_arg;
        const int nbpt = fused_combine_num_bytes_per_token(hidden_int4, num_topk);
#define DEEP_EP_FUSED_COMBINE_CASE(R)                                                                                 \
    case R: {                                                                                                         \
        const int rdma_clean_off = static_cast<int>(static_cast<int64_t>(nbpt) * num_max_rdma_chunked_recv_tokens * R  \
                                                    * 2 * num_channels / sizeof(int));                                \
        const int rdma_clean_n = (num_nvl_ranks * 2 + 4) * R * 2 * num_channels;                                       \
        const int nvl_clean_off = static_cast<int>(static_cast<int64_t>(num_max_nvl_chunked_recv_tokens) * nbpt *      \
                                                   num_nvl_ranks * num_channels / sizeof(int));                       \
        const int nvl_clean_n = num_nvl_ranks * (2 * R + 2) * num_channels;                                            \
        queue.wait();                                                                                                 \
        ishmem_barrier_all();                                                                                         \
        launch_fused_clean_planes<R>(rdma_buffer_ptr, buffer_ptrs_gpu, nvl_rank, rdma_clean_off, rdma_clean_n,         \
                                     nvl_clean_off, nvl_clean_n, queue);                                              \
        launch_fused_cached_notify_heads<R>(combined_rdma_head, combined_nvl_head, num_combined_tokens, num_channels,  \
                                            rdma_channel_prefix_matrix, rdma_rank_prefix_sum, num_nvl_ranks, queue);   \
        queue.wait();                                                                                                 \
        ishmem_barrier_all();                                                                                         \
        launch_fused_combine<R>(combined_x, combined_topk_weights, x, topk_weights, bias_0, bias_1,               \
                                combined_rdma_head, combined_nvl_head, static_cast<const SourceMeta*>(src_meta),       \
                                rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix,           \
                                num_tokens, num_combined_tokens, hidden, num_topk, rdma_buffer_ptr,                    \
                                num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,   \
                                num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, rank, num_ranks,     \
                                num_nvl_ranks, num_channels, queue);                                                  \
        break;                                                                                                        \
    }
        // The fused combine kernel declares 2 + kNumRDMARanks SPIR-V named barriers.  On BMG
        // (oneAPI 2025.3) this kernel JITs with 8 named barriers but raises
        // "IGC: internal compiler error" with 9 - measured by bisection - so kNumRDMARanks is
        // capped at 4.  See the comment at the named_barrier_init() block in
        // internode_combine_fused.inc.
        TORCH_CHECK(num_rdma_ranks == 2 || num_rdma_ranks == 4,
                    "fused internode combine supports 2 or 4 RDMA ranks (BMG named-barrier limit), got ",
                    num_rdma_ranks);
        switch (num_rdma_ranks) {
            DEEP_EP_FUSED_COMBINE_CASE(2)
            DEEP_EP_FUSED_COMBINE_CASE(4)
            default:
                TORCH_CHECK(false, "fused internode combine supports 2/4 RDMA ranks, got ", num_rdma_ranks);
        }
#undef DEEP_EP_FUSED_COMBINE_CASE
        queue.wait();
    }
#else
    TORCH_CHECK(false, "combine_nvl_rdma requires DEEP_EP_ENABLE_ISHMEM");
#endif
}

}  // namespace internode
}  // namespace deep_ep
