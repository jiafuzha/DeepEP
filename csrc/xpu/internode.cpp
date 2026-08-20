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
// FUSED, warp-specialized CUDA-faithful dispatch (see the .inc for the full
// CUDA<->SYCL mapping). Selected by DEEP_EP_INTERNODE_FUSED=1; the phase-split
// implementation below remains the default until the matching fused COMBINE
// lands, because the two are coupled through the `send_rdma_head` /
// `send_nvl_head` handle semantics.
// ===========================================================================
#include "internode_dispatch_fused.inc"

inline bool internode_fused_enabled() {
    const char* e = std::getenv("DEEP_EP_INTERNODE_FUSED");
    return e != nullptr && e[0] == '1';
}

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

    if (internode_fused_enabled()) {
        const int hidden_int4 = static_cast<int>((static_cast<size_t>(hidden) * element_size) / sizeof(int4_t));
        TORCH_CHECK(static_cast<size_t>(hidden) * element_size % sizeof(int4_t) == 0,
                    "fused internode dispatch requires the token row to be 16B-aligned");
        const bool cached_mode = (send_rdma_head == nullptr);
        const int num_bytes_per_token = fused_num_bytes_per_token(hidden_int4, num_scales, num_topk);
#define DEEP_EP_FUSED_DISPATCH_CASE(R)                                                                                \
    case R:                                                                                                           \
        launch_fused_dispatch_clean<R>(rdma_buffer_ptr, buffer_ptrs_gpu[nvl_rank], num_channels, num_nvl_ranks,        \
                                       num_bytes_per_token, num_max_rdma_chunked_recv_tokens,                         \
                                       num_max_nvl_chunked_recv_tokens, queue);                                       \
        queue.wait();                                                                                                 \
        ishmem_barrier_all();                                                                                         \
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
        return;
    }

    // CUDA-faithful per-GPU RDMA: every nvl_rank sends its own RDMA on its own NIC
    // (dst_pe = dst_rdma*num_nvl_ranks + nvl_rank). The receive-side forward buffer
    // needs one disjoint slice per source nvl-plane so concurrent forwarders don't
    // collide (num_fwd_planes == num_nvl_ranks).
    const int num_fwd_planes = num_nvl_ranks;

    const auto* src = static_cast<const uint8_t*>(x);
    auto* dst = static_cast<uint8_t*>(recv_x);
    auto* meta = static_cast<SourceMeta*>(recv_src_meta);
    auto* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);

    NvlBufferLayout layout(num_tokens, num_ranks, 1, row_bytes, num_topk, num_scales);
    NvlForwardLayout fwd_layout(num_recv_tokens, row_bytes, num_topk, num_scales, num_rdma_ranks, num_fwd_planes);
    const size_t fwd_base_offset = align_offset(layout.total_bytes, 128);

    // Push-staging region (fixes the racy cross-rank IPC READ in RdmaSend).
    // On this BMG stack a GPU's IPC/P2P READ of an NVL peer's *freshly-Pack-
    // written* send buffer is not P2P-coherent in time, but a cross-device
    // WRITE (owner -> peer) is reliable. So instead of the leader READING peer
    // send buffers, each rank COPIES its own (coherent) send buffer into a
    // per-src_nvl slot in the leader's NVL buffer; the leader then reads its
    // OWN staging (same-GPU coherent). The staging region mirrors the full NVL
    // buffer layout so RdmaSend can reuse layout.send_*_offset unchanged.
    const size_t stage_base_offset = align_offset(fwd_base_offset + fwd_layout.total_bytes, 128);

    // ---- Producer-PUSH intra-node NVL region (gap #2) ----------------------------------
    // On this BMG+igub stack a cross-rank IPC READ of a peer's send buffer is unstable
    // (DEVICE_LOST-class), while a cross-rank IPC WRITE is stable. So instead of the
    // consumer (Assemble/Head) READING each peer's send buffer, each producer WRITES its
    // intra-node tokens into the DESTINATION peer's recv-staging region here, and the
    // consumer reads only its OWN region (local read). Structured exactly like the RDMA
    // fwd region: num_nvl_ranks disjoint source planes (plane == source nvl_rank), each
    // holding up to num_tokens tokens (worst case: one source routes all its tokens to
    // one dest), plus a per-plane count. Passing num_rdma_ranks==1 makes the count array
    // int[num_nvl_ranks] (one settled count per source plane). Reuses the now-dead
    // push-staging region location to avoid growing the NVL buffer footprint.
    NvlForwardLayout nvlrecv_layout(num_tokens, row_bytes, num_topk, num_scales, /*num_rdma_ranks=*/1, num_nvl_ranks);
    const size_t nvlrecv_base_offset = stage_base_offset;
    // Per-source published receive counts: each rank's Assemble pushes its per_src_count
    // (num_ranks ints, grouped by GLOBAL src rank) into every peer's slot [producer_nvl],
    // so the Head kernel derives send_nvl_head base offsets with LOCAL reads only.
    const size_t peer_counts_offset = align_offset(nvlrecv_base_offset + nvlrecv_layout.total_bytes, 128);

    const size_t rdma_x_size = static_cast<size_t>(num_recv_tokens) * row_bytes;
    const size_t rdma_meta_size = static_cast<size_t>(num_recv_tokens) * sizeof(SourceMeta);
    const size_t rdma_topk_idx_size = static_cast<size_t>(num_recv_tokens) * num_topk * sizeof(topk_idx_t);
    const size_t rdma_topk_wt_size = static_cast<size_t>(num_recv_tokens) * num_topk * sizeof(float);
    const size_t rdma_scales_size = static_cast<size_t>(num_recv_tokens) * num_scales * sizeof(float);
    const size_t rdma_meta_offset = align_offset(rdma_x_size, alignof(SourceMeta));
    const size_t rdma_topk_idx_offset = align_offset(rdma_meta_offset + rdma_meta_size, alignof(topk_idx_t));
    const size_t rdma_topk_wt_offset = align_offset(rdma_topk_idx_offset + rdma_topk_idx_size, alignof(float));
    const size_t rdma_scales_offset = align_offset(rdma_topk_wt_offset + rdma_topk_wt_size, alignof(float));
    const size_t rdma_count_offset = align_offset(rdma_scales_offset + rdma_scales_size, alignof(int));
    // Faithful path uses a 64-bit AMO count flag (ishmemx_long_atomic_add_qp) at
    // rdma_flag_offset. Kept separate from rdma_count_offset (the existing int
    // sentinel field) so the non-faithful fallback layout is byte-unchanged. With
    // multi-QP (num_qp_ch>1) the flag is an ARRAY of num_qp_ch longs (one per
    // channel/QP); default num_qp_ch==1 enlarges the region by a single long as
    // before, so the single-QP layout is byte-for-byte unchanged.
    // GUARD: the blocking-put A/B path (DEEP_EP_INTERNODE_BLOCKING_PUT) issues its
    // payload through the plain ishmem_putmem*/ishmem_putmem_nbi APIs, because this
    // iSHMEM build provides no single-work-item per-QP put (ishmemx_putmem_nbi_qp is
    // declared in ishmemx.h but NOT defined in libishmem.a -- only fence_qp, quiet_qp
    // and int/long_atomic_add_qp are implemented). Those plain APIs select a QP by
    // atomic round-robin once num_qps_per_pe > 1, which would silently break F-K3b's
    // "flag[c] lands after chunk c on the SAME qp" RC-in-order invariant and let the
    // receiver read a partially-written region. Pin to a single channel/QP in that
    // mode. combine_nvl_rdma applies the identical guard so both layouts agree.
    const int num_qp_ch = internode_blocking_put() ? 1 : internode_num_qp_channels();
    const size_t rdma_flag_offset = align_offset(rdma_count_offset + sizeof(int), alignof(long));
    const size_t rdma_region_bytes =
        align_offset(rdma_flag_offset + static_cast<size_t>(num_qp_ch) * sizeof(long), 128);
    const size_t rdma_send_base = static_cast<size_t>(num_rdma_ranks) * rdma_region_bytes;

    const size_t total_recv_bytes = static_cast<size_t>(num_recv_tokens) * row_bytes;
    const size_t total_recv_topk = static_cast<size_t>(num_recv_tokens) * num_topk;
    const size_t total_recv_scales = static_cast<size_t>(num_recv_tokens) * num_scales;
    const size_t total_send_rdma = static_cast<size_t>(num_tokens) * num_rdma_ranks;
    const size_t total_send_nvl = static_cast<size_t>(num_tokens) * num_ranks;
    const size_t total_recv_regions = static_cast<size_t>(num_rdma_ranks) * rdma_region_bytes;
    const size_t init_range = std::max({total_recv_bytes,
                                        total_recv_topk,
                                        total_recv_scales,
                                        static_cast<size_t>(num_recv_tokens),
                                        total_send_rdma,
                                        total_send_nvl,
                                        static_cast<size_t>(num_ranks),
                                        static_cast<size_t>(num_rdma_ranks),
                                        static_cast<size_t>(num_ranks) * num_channels,
                                        static_cast<size_t>(num_rdma_ranks) * num_channels,
                                        total_recv_regions,
                                        static_cast<size_t>(1)});

    // FUSED: Init zeroing moved into FaithfulDispatchPackStageKernel below.
    // The separate CombinedDispatchInitKernel launch is eliminated.

    // ===================================================================
    // ===================================================================
    // ARCHITECTURAL DEVIATIONS FROM CUDA internode.cu
    // ===================================================================
    // The XPU/SYCL faithful path differs from the CUDA original in 5 deliberate ways:
    //
    // 1. PRODUCER-PUSH INTRA-NODE (gap #6):
    //    CUDA consumers READ peer NVL buffers via IPC. On BMG/igub, cross-rank
    //    IPC READ is unstable (DEVICE_LOST-class). We instead use Producer-PUSH:
    //    each rank WRITES its tokens into every destination peer's staging.
    //    Consumers read only their OWN local buffer. Required for BMG stability.
    //
    // 2. MICRO-KERNEL DECOMPOSITION (gap #4):
    //    CUDA uses warp specialization within a SINGLE kernel (5 WarpRoles in
    //    dispatch, 4 in combine) coordinated by named barrier.sync. SYCL lacks
    //    named barrier.sync for subset-of-WG synchronization, so we split each
    //    role into its own kernel launch. In-order queue-ordering (no
    //    intermediate host waits) preserves cross-kernel dependency.
    //
    // 3. AMO-FLAG CROSS-NODE SYNCHRONIZATION (gaps #5,#22):
    //    CUDA uses nvshmem_sync_all() (global PE barrier) and a sliding-window
    //    credit model with spinlocks for RDMA backpressure. We use the LL pattern:
    //    ishmemx_long_atomic_add_qp(-count-1) flag + acquire-poll, with no global
    //    barrier. Cross-node ordering is guaranteed by RC ordering on the same QP
    //    (payload put → AMO flag → receiver poll). Buffer lifecycle relies on
    //    queue-ordering + the next dispatch's Init-zero to recycle buffers.
    //
    // 4. TMA → COOPERATIVE COPY (gap #7):
    //    CUDA uses SM90 TMA (tma_load_1d, tma_store_1d, mbarrier) for bulk copies.
    //    Intel GPU has no TMA equivalent. We use faithful_coop_copy() — 16B-
    //    vectorized lane-strided cooperative byte copies. Correctness is identical;
    //    performance may differ.
    //
    // 5. TOKEN-RANGE PARALLELISM (gap #18):
    //    CUDA stripes tokens across num_sms/2 SM-channels with per-channel QP.
    //    We use one WG per QP-channel (num_qp_ch from ISHMEM_IBGDA_QPS_PER_PE)
    //    with per-token WG parallelism (one WG per combined token) instead of
    //    per-channel token ranges. For large token counts, consider CUDA-style
    //    channel token-range partitioning via get_channel_task_range().
    //
    // Handle semantics (send_nvl_head/send_rdma_head) are verified byte-identical
    // between this path and the serial fallback (gap #21). The combine reduce
    // reads combined_nvl_head[ct * num_ranks + dst_rank] which gives the correct
    // recv_x position; the combine staging planes mirror recv_x positions exactly.
    // ===================================================================
    //
    // FAITHFUL (CUDA-parity) DISPATCH TRANSPORT (stage-1 collapse)
    // ===================================================================
    // We run a collapsed transport that
    // (a) fuses Pack+Stage into one WG/sub-group-parallel kernel (drops the
    //     serial single_task and the separate Stage + StageBarrier launches),
    // (b) fuses RdmaSend compaction + RDMA put into one kernel that issues
    //     ishmemx_putmem_nbi_subgroup (warp-collective, deferred doorbell) +
    //     ishmemx_fence_qp + a 64-bit ishmemx_long_atomic_add_qp count flag
    //     instead of the scalar blocking ishmem_putmem + double
    //     ishmemx_barrier_all_work_group, and
    // (c) gates the receiver on an acquire-fence + uc_load bounded-spin poll of
    //     that flag (device-scope everywhere except the single cross-PE
    //     post-zero rendezvous).
    // It writes the SAME peer send buffers + leader staging + fwd_* buffers the
    // shared Assemble/Head kernels (run after this branch) consume, so the
    // output handle semantics (recv_x ordering, prefix sums, absolute-position
    // send_nvl_head/send_rdma_head) are byte-for-byte identical to the fallback.
    {  // faithful (CUDA-parity) dispatch transport (only path)
        const uint64_t rdma_poll_cap = internode_poll_cap();
        const int rdma_flag_lsc_mode = internode_flag_lsc_mode();
        const bool faithful_force_db = internode_force_db();          // F4a2 put doorbell
        const bool faithful_post_amo_quiet = internode_post_amo_quiet();  // F4b post-AMO quiet
        const bool faithful_entry_quiet = internode_entry_quiet();     // F4a2 entry drain qp0
        const bool faithful_blocking_put = internode_blocking_put();   // F4a2 blocking payload put
        const bool faithful_nbi_mode = internode_nbi_mode();
        // ---- F-K1: fused Init + Pack + Stage (WG-parallel zero + payload; WI0 routing meta) ----
        // FUSION: the former CombinedDispatchInitKernel zeroing is done here as a
        // cooperative per-WG pre-pass, eliminating one queue.submit + queue.wait.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchPackStageKernel>(
                sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    // ---- PHASE 0: cooperative zero-init (was separate CombinedDispatchInitKernel) ----
                    // Stride over the init_range, zeroing recv buffers, prefix sums, and RDMA regions.
                    // Only nvl_rank==0 zeroes the shared RDMA recv regions to avoid redundant writes.
                    {
                        for (size_t linear = static_cast<size_t>(local_id); linear < init_range; linear += kComputeWGSize) {
                            if (linear < total_recv_bytes) {
                                dst[linear] = 0;
                            }
                            if (linear < static_cast<size_t>(num_recv_tokens)) {
                                if (meta != nullptr) {
                                    meta[linear].src_rdma_rank = -1;
                                    meta[linear].is_token_in_nvl_rank_bits = 0;
                                    meta[linear].src_nvl_rank = -1;
                                }
                            }
                            if (linear < total_recv_topk && recv_topk_idx != nullptr) {
                                recv_topk_idx[linear] = -1;
                                recv_topk_weights[linear] = 0.0f;
                            }
                            if (linear < total_recv_scales && recv_x_scales != nullptr) {
                                recv_x_scales[linear] = 0.0f;
                            }
                            if (linear < total_send_rdma && send_rdma_head != nullptr) {
                                send_rdma_head[linear] = -1;
                            }
                            if (linear < total_send_nvl && send_nvl_head != nullptr) {
                                send_nvl_head[linear] = -1;
                            }
                            if (linear < static_cast<size_t>(num_ranks) && recv_gbl_rank_prefix_sum != nullptr) {
                                recv_gbl_rank_prefix_sum[linear] = 0;
                            }
                            if (linear < static_cast<size_t>(num_rdma_ranks) && recv_rdma_rank_prefix_sum != nullptr) {
                                recv_rdma_rank_prefix_sum[linear] = 0;
                            }
                            if (linear < static_cast<size_t>(num_ranks) * num_channels) {
                                if (gbl_channel_prefix_matrix != nullptr) {
                                    gbl_channel_prefix_matrix[linear] = 0;
                                }
                                if (recv_gbl_channel_prefix_matrix != nullptr) {
                                    recv_gbl_channel_prefix_matrix[linear] = 0;
                                }
                            }
                            if (linear < static_cast<size_t>(num_rdma_ranks) * num_channels) {
                                if (rdma_channel_prefix_matrix != nullptr) {
                                    rdma_channel_prefix_matrix[linear] = 0;
                                }
                                if (recv_rdma_channel_prefix_matrix != nullptr) {
                                    recv_rdma_channel_prefix_matrix[linear] = 0;
                                }
                            }
                            if (nvl_rank == 0 && linear < total_recv_regions) {
                                rdma_base[linear] = 0;
                            }
                        }
                    }
                    sycl::group_barrier(group);

                    // ---- PHASE 1: token packing (original PackStage body) ----
                    auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                    auto* my_counts = reinterpret_cast<int*>(my_buf + layout.count_offset);
                    auto* my_channel_counts = reinterpret_cast<int*>(my_buf + layout.channel_count_offset);
                    auto* my_send_x = my_buf + layout.send_x_offset;
                    auto* my_send_meta = reinterpret_cast<SourceMeta*>(my_buf + layout.send_meta_offset);
                    auto* my_send_topk_idx = reinterpret_cast<topk_idx_t*>(my_buf + layout.send_topk_idx_offset);
                    auto* my_send_topk_weights = reinterpret_cast<float*>(my_buf + layout.send_topk_weights_offset);
                    auto* my_send_x_scales = reinterpret_cast<float*>(my_buf + layout.send_x_scales_offset);
                    auto* my_send_dst_token = reinterpret_cast<int*>(my_buf + layout.send_dst_token_offset);
                    auto* my_send_routing_bits = reinterpret_cast<int*>(my_buf + layout.send_routing_bits_offset);
                    auto* my_send_rdma_bits = reinterpret_cast<int*>(my_buf + layout.send_rdma_dest_bits_offset);
                    auto* my_send_is_in_rank = reinterpret_cast<bool*>(my_buf + layout.send_is_token_in_rank_offset);
                    constexpr int kPackNumChannels = 1;  // matches NvlBufferLayout(num_channels=1)

                    if (local_id == 0) {
                        for (int d = 0; d < num_ranks; ++d) {
                            my_counts[d] = 0;
                            for (int c = 0; c < kPackNumChannels; ++c)
                                my_channel_counts[d * kPackNumChannels + c] = 0;
                        }
                    }
                    sycl::group_barrier(group);

                    for (int token = 0; token < num_tokens; ++token) {
                        // Routing metadata + counts by WI0 (cheap, keeps counts race-free).
                        if (local_id == 0) {
                            int nvl_bits = 0, rdma_bits = 0;
                            for (int d = 0; d < num_ranks; ++d) {
                                const bool in_rank = is_token_in_rank[token * num_ranks + d];
                                my_send_is_in_rank[token * num_ranks + d] = in_rank;
                                if (!in_rank) continue;
                                my_counts[d] += 1;
                                nvl_bits |= 1 << (d % num_nvl_ranks);
                                rdma_bits |= 1 << (d / num_nvl_ranks);
                            }
                            my_send_meta[token] = SourceMeta{my_rdma_rank, nvl_bits, nvl_rank};
                            my_send_dst_token[token] = -1;
                            my_send_routing_bits[token] = nvl_bits;
                            my_send_rdma_bits[token] = rdma_bits;
                            if (topk_idx != nullptr) {
                                for (int k = 0; k < num_topk; ++k) {
                                    my_send_topk_idx[token * num_topk + k] = topk_idx[token * num_topk + k];
                                    my_send_topk_weights[token * num_topk + k] = topk_weights[token * num_topk + k];
                                }
                            }
                            if (x_scales != nullptr) {
                                for (int s = 0; s < num_scales; ++s)
                                    my_send_x_scales[token * num_scales + s] = x_scales[token * num_scales + s];
                            }
                        }
                        // Payload byte-copy parallelized across the whole work-group (cache-hot).
                        const auto* src_row = src + static_cast<size_t>(token) * row_bytes;
                        auto* dst_row = my_send_x + static_cast<size_t>(token) * row_bytes;
                        faithful_coop_copy(dst_row, src_row, row_bytes, local_id, kComputeWGSize);
                    }
                    sycl::group_barrier(group);

                    if (local_id == 0) {
                        for (int d = 0; d < num_ranks; ++d) {
                            int cumulative = 0;
                            for (int c = 0; c < kPackNumChannels; ++c) {
                                const int start = (static_cast<int64_t>(num_tokens) * c) / kPackNumChannels;
                                const int end = (static_cast<int64_t>(num_tokens) * (c + 1)) / kPackNumChannels;
                                int cnt = 0;
                                for (int token = start; token < end; ++token)
                                    cnt += my_send_is_in_rank[token * num_ranks + d] ? 1 : 0;
                                cumulative += cnt;
                                my_channel_counts[d * kPackNumChannels + c] = cumulative;
                            }
                        }
                    }

                    // Flush own send buffer so this GPU's own per-GPU RdmaSend (which
                    // reads buffer_ptrs_gpu[nvl_rank] directly) observes settled data.
                    // No leader staging: per-GPU RDMA never funnels through nvl_rank==0.
                    sycl::group_barrier(group);
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    lsc_fence_sysrel();
                });
        });

        // ---- F-K2 + F-K3a1 FUSED: NVL barrier + flag-zero + cross-PE barrier ----
        // Was two separate kernel launches (PackBarrier then RdmaBarrier); merged since
        // both use a single kIshmemWGSize WG with independent operations, saving one
        // queue.submit + launch-overhead round-trip.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchPackBarrierKernel>(
                sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));

                    // Phase 1 (was PackBarrier): intra-node NVL device-scope barrier so
                    // all NVL peers finished their PackStage before we advance.
                    nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base, num_nvl_ranks, item);

                    // Phase 2 (was RdmaBarrier): zero MY receive-region flags so remote
                    // AMOs land onto 0. Under per-GPU RDMA every nvl_rank receives its own
                    // same-plane stream, so every PE zeroes its own flags.
                    if (local_id == 0) {
                        for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                            if (src_rdma == my_rdma_rank) continue;
                            auto* fl = reinterpret_cast<long*>(rdma_base + static_cast<size_t>(src_rdma) * rdma_region_bytes +
                                                               rdma_flag_offset);
                            for (int c = 0; c < num_qp_ch; ++c) uc_store<long>(fl + c, 0L);
                        }
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        lsc_fence_sysrel();
                    }
                    sycl::group_barrier(group);
                    // The ONE necessary cross-PE rendezvous: every PE zeroed its flags
                    // before any AMO posts. Called by ALL PEs' work-groups (not leader-
                    // gated) so the collective does not hang.
                    ishmemx_barrier_all_work_group(group);
                    sycl::group_barrier(group);
                });
        });

        // ---- F-K3a2: Put — leader compaction + payload warp-put (force_db=false) ----
        // Split from the F4a0 barrier and the F4b flag AMO. NO full ishmem_quiet() here
        // or in F4b: LL (internode_ll.cpp L940-942 / L1245-1247) uses ONLY a targeted
        // quiet_qp BEFORE the AMO and NEVER a full quiet, nor any quiet after the AMO.
        // A full ishmem_quiet() blocks forever if an AMO completion is not reaped by the
        // IBGDA layer (leader-only, data-independent, surfaces on a later dispatch).
        queue.submit([&](sycl::handler& cgh) {
            // GAP#8 + GAP#18: CUDA-CHANNEL TOKEN-RANGE PARALLELISM. Launch
            // num_use_channels * num_qp_ch WGs: each (channel_id, qp) pair owns the
            // intersection of the channel token range and a QP sub-range within it.
            // Each WG derives its own contiguous row offset by scanning matching
            // tokens in prior channels + prior QP sub-ranges. Rows tile [0, total)
            // in token order. The receiver reads [0, total) after waiting for all
            // num_qp_ch flags (per-QP AMO guarantees all channel payloads on that QP
            // have landed, RC in-order).
            //
            // Cap channels to max(num_qp_ch, min(num_channels, num_tokens/4)) so each
            // WG gets at least 4 tokens of work.
            const int kMinTokensPerChannel = 4;
            const int max_ch_by_tokens = std::max(1, num_tokens / kMinTokensPerChannel);
            const int num_use_channels = std::max(num_qp_ch, std::min(num_channels, max_ch_by_tokens));
            const int num_send_ch = num_use_channels * num_qp_ch;
            cgh.parallel_for<FaithfulDispatchRdmaSendKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_send_ch) * kComputeWGSize),
                                  sycl::range<1>(kComputeWGSize)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    auto group = item.get_group();
                    auto sg = item.get_sub_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    const int linear = static_cast<int>(item.get_group_linear_id());
                    // Each WG = (channel_id, qp) pair:
                    //   channel_id in [0, num_use_channels) — CUDA-style token-range partition
                    //   qp in [0, num_qp_ch) — QP stripe within that channel
                    const int channel_id = linear / num_qp_ch;
                    const int ch = linear % num_qp_ch;  // QP for warp-collective put
                    // Channel token range [ch_start, ch_end), same as CUDA
                    // get_channel_task_range(num_tokens, num_use_channels, channel_id)
                    const int ch_tokens_per = num_tokens / num_use_channels;
                    const int ch_rem = num_tokens % num_use_channels;
                    const int ch_start = channel_id <= ch_rem
                        ? (ch_tokens_per + 1) * channel_id
                        : ch_tokens_per * channel_id + ch_rem;
                    const int ch_end = channel_id < ch_rem
                        ? ch_start + ch_tokens_per + 1
                        : ch_start + ch_tokens_per;
                    // QP sub-range within the channel
                    const int ch_n = ch_end - ch_start;
                    const int ch_t0 = ch_start + static_cast<int>((static_cast<int64_t>(ch_n) * ch) / num_qp_ch);
                    const int ch_t1 = ch_start + static_cast<int>((static_cast<int64_t>(ch_n) * (ch + 1)) / num_qp_ch);
                    {  // per-GPU RDMA: every nvl_rank issues its own RDMA on its own NIC
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();
                        // Optional entry drain (A/B via DEEP_EP_INTERNODE_ENTRY_QUIET, default
                        // OFF): each channel drains ITS OWN qp before posting.
                        if (faithful_entry_quiet && local_id == 0) {
                            for (int dq = 0; dq < num_rdma_ranks; ++dq) {
                                if (dq == my_rdma_rank) continue;
                                ishmemx_fence_qp(dq * num_nvl_ranks + nvl_rank, static_cast<unsigned>(ch));
                            }
                        }
                        sycl::group_barrier(group);
                        // Per-GPU RDMA: this GPU compacts ONLY its own packed tokens
                        // (single plane == nvl_rank, read from its own NVL buffer).
                        auto* leader_self_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                        auto* peer_x = leader_self_buf + layout.send_x_offset;
                        auto* peer_m = reinterpret_cast<SourceMeta*>(leader_self_buf + layout.send_meta_offset);
                        auto* peer_idx = reinterpret_cast<topk_idx_t*>(leader_self_buf + layout.send_topk_idx_offset);
                        auto* peer_wt = reinterpret_cast<float*>(leader_self_buf + layout.send_topk_weights_offset);
                        auto* peer_scales = reinterpret_cast<float*>(leader_self_buf + layout.send_x_scales_offset);
                        auto* peer_rdma_bits = reinterpret_cast<int*>(leader_self_buf + layout.send_rdma_dest_bits_offset);
                        auto* peer_is_in_rank = reinterpret_cast<bool*>(leader_self_buf + layout.send_is_token_in_rank_offset);
                        for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                            if (dst_rdma == my_rdma_rank) continue;  // local node: no RDMA
                            auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                            auto* rdma_x = region;
                            auto* rdma_m = reinterpret_cast<SourceMeta*>(region + rdma_meta_offset);
                            auto* rdma_idx = reinterpret_cast<topk_idx_t*>(region + rdma_topk_idx_offset);
                            auto* rdma_wt = reinterpret_cast<float*>(region + rdma_topk_wt_offset);
                            auto* rdma_scales = reinterpret_cast<float*>(region + rdma_scales_offset);

                            // WI0 derives this WG's row prefix `off` (matches in [0, ch_t0)
                            // spanning: all prior channels [0, ch_start) + current channel's
                            // prior QP sub-ranges [ch_start, ch_t0)) and its own token count
                            // `cnt` (matches in [ch_t0, ch_t1)). The first WG (channel 0, qp 0)
                            // also computes the grand `total` (matches in [0, num_tokens))
                            // stashed for the flag kernel.
                            int off = 0, cnt = 0, total = 0;
                            if (local_id == 0) {
                                // prior channels [0, ch_start)
                                for (int t = 0; t < ch_start; ++t) {
                                    if (((peer_rdma_bits[t] >> dst_rdma) & 1) == 0) continue;
                                    ++off;
                                }
                                // current channel: [ch_start, ch_t0) → prefix; [ch_t0, ch_t1) → cnt
                                for (int t = ch_start; t < ch_t1; ++t) {
                                    if (((peer_rdma_bits[t] >> dst_rdma) & 1) == 0) continue;
                                    if (t < ch_t0) ++off; else ++cnt;
                                }
                                // first WG computes grand total
                                if (channel_id == 0 && ch == 0) {
                                    total = off + cnt;
                                    for (int t = ch_t1; t < ch_end; ++t)
                                        if ((peer_rdma_bits[t] >> dst_rdma) & 1) ++total;
                                    for (int t = ch_end; t < num_tokens; ++t)
                                        if ((peer_rdma_bits[t] >> dst_rdma) & 1) ++total;
                                }
                            }
                            off = sycl::group_broadcast(group, off, 0);
                            cnt = sycl::group_broadcast(group, cnt, 0);

                            // Compact this channel's matching tokens into rows [off, off+cnt).
                            // All lanes run identical control flow so `w` stays coherent; the
                            // payload copy is split across lanes, scalar meta by WI0.
                            int w = off;
                            for (int t = ch_t0; t < ch_t1; ++t) {
                                if (((peer_rdma_bits[t] >> dst_rdma) & 1) == 0) continue;
                                if (w >= num_recv_tokens) continue;  // capacity guard (never hit for legit count)
                                auto* s_row = peer_x + static_cast<size_t>(t) * row_bytes;
                                auto* d_row = rdma_x + static_cast<size_t>(w) * row_bytes;
                                faithful_coop_copy(d_row, s_row, row_bytes, local_id, kComputeWGSize);
                                if (local_id == 0) {
                                    int dst_nvl_bits = 0;
                                    for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                                        const int dst_rank = dst_rdma * num_nvl_ranks + dst_nvl;
                                        if (peer_is_in_rank[t * num_ranks + dst_rank]) dst_nvl_bits |= 1 << dst_nvl;
                                    }
                                    SourceMeta sm = peer_m[t];
                                    sm.is_token_in_nvl_rank_bits = dst_nvl_bits;
                                    rdma_m[w] = sm;
                                    if (topk_idx != nullptr) {
                                        for (int k = 0; k < num_topk; ++k) {
                                            rdma_idx[w * num_topk + k] = peer_idx[t * num_topk + k];
                                            rdma_wt[w * num_topk + k] = peer_wt[t * num_topk + k];
                                        }
                                    }
                                    if (x_scales != nullptr) {
                                        for (int s = 0; s < num_scales; ++s)
                                            rdma_scales[w * num_scales + s] = peer_scales[t * num_scales + s];
                                    }
                                }
                                ++w;
                            }

                            // Payload warp-put ONLY, deferred doorbell (force_db selected by
                            // DEEP_EP_INTERNODE_FORCE_DB). quiet + AMO are deferred to F-K3b
                            // AFTER the kernel boundary.
                            sycl::group_barrier(group);
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                            lsc_fence_sysrel();
                            const int dst_pe = dst_rdma * num_nvl_ranks + nvl_rank;
                            auto* dst_region = rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes;

                            // This channel's rows are the disjoint slice [off, off+cnt) of every
                            // field array; put each field-slice on THIS channel's qp=ch. Rows are
                            // contiguous per channel and tile [0,total), so the peer reassembles
                            // the exact contiguous region. One warp (sub-group 0) drives qp=ch
                            // (LL one-warp-per-qp discipline).
                            const size_t x_off = static_cast<size_t>(off) * row_bytes;
                            const size_t x_len = static_cast<size_t>(cnt) * row_bytes;
                            const size_t m_off = rdma_meta_offset + static_cast<size_t>(off) * sizeof(SourceMeta);
                            const size_t m_len = static_cast<size_t>(cnt) * sizeof(SourceMeta);
                            const size_t i_off = rdma_topk_idx_offset + static_cast<size_t>(off) * num_topk * sizeof(topk_idx_t);
                            const size_t i_len = static_cast<size_t>(cnt) * num_topk * sizeof(topk_idx_t);
                            const size_t w_off = rdma_topk_wt_offset + static_cast<size_t>(off) * num_topk * sizeof(float);
                            const size_t w_len = static_cast<size_t>(cnt) * num_topk * sizeof(float);
                            const size_t sc_off = rdma_scales_offset + static_cast<size_t>(off) * num_scales * sizeof(float);
                            const size_t sc_len = static_cast<size_t>(cnt) * num_scales * sizeof(float);
                            if (faithful_blocking_put) {
                                // Robust path (A/B via DEEP_EP_INTERNODE_BLOCKING_PUT): single-WI
                                // put per field-slice. DEEP_EP_INTERNODE_NBI_MODE selects the API:
                                //   0 (default): ishmem_putmem      (blocking)
                                //   1:           ishmem_putmem_nbi  (non-blocking, with release fence)
                                if (cnt > 0 && local_id == 0) {
                                    // NOTE: these are the NON-per-QP APIs on purpose — this
                                    // iSHMEM build declares ishmemx_putmem_nbi_qp/_qp in
                                    // ishmemx.h but does NOT define them in libishmem.a (only
                                    // fence_qp, quiet_qp, int/long_atomic_add_qp are
                                    // implemented), and there is no single-work-item per-QP put.
                                    // The plain APIs pick a QP by atomic round-robin
                                    // (ishmemi_ibgda_device_peer_context_rr) once
                                    // num_qps_per_pe > 1, which would break F-K3b's
                                    // "flag[ch] lands after chunk ch on the SAME qp" invariant.
                                    // The launcher therefore pins num_qp_ch = 1 whenever this
                                    // blocking-put A/B path is enabled; see dispatch_nvl_rdma.
                                    if (faithful_nbi_mode) {
                                        ishmem_putmem_nbi(dst_region + x_off, region + x_off, x_len, dst_pe);
                                        ishmem_putmem_nbi(dst_region + m_off, region + m_off, m_len, dst_pe);
                                        if (i_len > 0) ishmem_putmem_nbi(dst_region + i_off, region + i_off, i_len, dst_pe);
                                        if (w_len > 0) ishmem_putmem_nbi(dst_region + w_off, region + w_off, w_len, dst_pe);
                                        if (sc_len > 0) ishmem_putmem_nbi(dst_region + sc_off, region + sc_off, sc_len, dst_pe);
                                    } else {
                                        ishmem_putmem(dst_region + x_off, region + x_off, x_len, dst_pe);
                                        ishmem_putmem(dst_region + m_off, region + m_off, m_len, dst_pe);
                                        if (i_len > 0) ishmem_putmem(dst_region + i_off, region + i_off, i_len, dst_pe);
                                        if (w_len > 0) ishmem_putmem(dst_region + w_off, region + w_off, w_len, dst_pe);
                                        if (sc_len > 0) ishmem_putmem(dst_region + sc_off, region + sc_off, sc_len, dst_pe);
                                    }
                                }
                                sycl::group_barrier(group);
                                if (faithful_nbi_mode) {
                                    // NBI path: release fence to make data NIC-visible before the
                                    // deferred quiet+AMO in F-K3b (separate kernel boundary).
                                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                                    lsc_fence_sysrel();
                                }
                            } else {
                                // Default path: warp-collective NBI put (cache-hot, fast) on qp=ch.
                                if (cnt > 0 && sg.get_group_id()[0] == 0) {
                                    const unsigned qp = static_cast<unsigned>(ch);
                                    ishmemx_putmem_nbi_subgroup(dst_region + x_off, region + x_off, x_len, dst_pe, qp, true, sg,
                                                            /*force_db=*/faithful_force_db);
                                    ishmemx_putmem_nbi_subgroup(dst_region + m_off, region + m_off, m_len, dst_pe, qp, true, sg,
                                                            /*force_db=*/faithful_force_db);
                                    if (i_len > 0)
                                        ishmemx_putmem_nbi_subgroup(dst_region + i_off, region + i_off, i_len, dst_pe, qp, true, sg,
                                                                /*force_db=*/faithful_force_db);
                                    if (w_len > 0)
                                        ishmemx_putmem_nbi_subgroup(dst_region + w_off, region + w_off, w_len, dst_pe, qp, true, sg,
                                                                /*force_db=*/faithful_force_db);
                                    if (sc_len > 0)
                                        ishmemx_putmem_nbi_subgroup(dst_region + sc_off, region + sc_off, sc_len, dst_pe, qp, true, sg,
                                                                /*force_db=*/faithful_force_db);
                                    sycl::group_barrier(sg);
                                }
                                sycl::group_barrier(group);
                            }
                            // Channel 0 stashes the grand token count for F-K3b's -count-1 flag.
                            // This slot is NOT transmitted (put lengths stop before it); it lives
                            // in the leader's own iSHMEM buffer and survives the kernel boundary.
                            if (channel_id == 0 && ch == 0 && local_id == 0) {
                                reinterpret_cast<int*>(region + rdma_count_offset)[0] = total;
                            }
                            sycl::group_barrier(group);
                        }
                    }
                });
        });

        // ---- F-K3b: RdmaFlag — quiet + 64-bit AMO ONLY, after the kernel boundary.
        // Mirrors internode_ll.cpp LLDispatchRecvKernel phase A (L938-942): quiet_qp
        // flushes the deferred doorbells posted in F-K3a, then the RC-ordered AMO on
        // the SAME qp lands after the payload.
        queue.submit([&](sycl::handler& cgh) {
            // GAP#8 P1: one WG per (dst_rdma, channel c) so the C per-channel quiet/AMO run
            // CONCURRENTLY (was a single WG serially looping dst_rdma x c). Mirrors
            // internode_ll.cpp LLDispatchRecvKernel phase A one-channel-per-WG independence.
            const int flag_wgs = num_rdma_ranks * num_qp_ch;
            cgh.parallel_for<FaithfulDispatchRdmaFlagKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(flag_wgs) * kIshmemWGSize),
                                  sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if (local_id != 0) return;
                    const int wg_id = static_cast<int>(item.get_group_linear_id());
                    const int dst_rdma = wg_id / num_qp_ch;
                    const int c = wg_id % num_qp_ch;
                    if (dst_rdma == my_rdma_rank) return;  // local node: no RDMA
                    auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                    const int count = reinterpret_cast<int*>(region + rdma_count_offset)[0];  // stashed by F-K3a
                    const int dst_pe = dst_rdma * num_nvl_ranks + nvl_rank;
                    auto* dst_flag = reinterpret_cast<long*>(
                        rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes + rdma_flag_offset);
                    // Channel c on its OWN work-group/qp: quiet qp c (flush F-K3a's deferred
                    // chunk-c doorbell) then post the tail AMO on qp c. Every flag carries
                    // -count-1 (total token count); RC in-order makes flag[c] land after
                    // chunk c on the same qp. The receiver waits for ALL C flags (=> all
                    // chunks placed) then reads [0,count).
                    ishmemx_fence_qp(dst_pe, static_cast<unsigned>(c));
                    lsc_fence_sysrel();
                    ishmemx_long_atomic_add_qp(dst_flag + c, static_cast<long>(-count - 1), dst_pe,
                                               static_cast<unsigned>(c));
                    if (faithful_post_amo_quiet) ishmemx_fence_qp(dst_pe, static_cast<unsigned>(c));
                });
        });

        // ---- F-K4: leader forwards RDMA-received tokens to local NVL peers (flag poll) ----
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchFwdWriteKernel>(
                sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)), [=](sycl::nd_item<1> item) {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    // Per-GPU RDMA: this GPU forwards its OWN received same-plane stream. It
                    // writes into each local peer's fwd buffer at the disjoint slice reserved
                    // for its source nvl-plane (== nvl_rank) so concurrent forwarders (one per
                    // plane) never collide.
                    const int fwd_plane = nvl_rank;
                    const int fwd_plane_token_base = fwd_plane * fwd_layout.plane_tokens;
                    const int fwd_plane_count_base = fwd_plane * num_rdma_ranks;
                    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                    lsc_fence_sysacq();
                    int peer_offsets[NUM_MAX_NVL_PEERS] = {0};
                    if (local_id == 0) {
                        for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                            auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                            auto* fwd_base = peer_buf + fwd_base_offset;
                            auto* fwd_counts = reinterpret_cast<int*>(fwd_base + fwd_layout.fwd_count_offset);
                            for (int r = 0; r < num_rdma_ranks; ++r) fwd_counts[fwd_plane_count_base + r] = 0;
                        }
                    }
                    sycl::group_barrier(group);

                    for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                        if (src_rdma == my_rdma_rank) continue;  // local node has no RDMA stream
                        int before[NUM_MAX_NVL_PEERS] = {0};
                        for (int peer = 0; peer < num_nvl_ranks; ++peer) before[peer] = peer_offsets[peer];

                        auto* region = rdma_base + static_cast<size_t>(src_rdma) * rdma_region_bytes;
                        auto* rdma_x = region;
                        auto* rdma_m = reinterpret_cast<SourceMeta*>(region + rdma_meta_offset);
                        auto* rdma_idx = reinterpret_cast<topk_idx_t*>(region + rdma_topk_idx_offset);
                        auto* rdma_wt = reinterpret_cast<float*>(region + rdma_topk_wt_offset);
                        auto* rdma_scales = reinterpret_cast<float*>(region + rdma_scales_offset);
                        auto* rdma_flag = reinterpret_cast<long*>(region + rdma_flag_offset);

                        // Bounded acquire-fence + uc_load spin on the 64-bit flag (0 == not
                        // arrived; -count-1 once the NIC-delivered AMO lands after the data).
                        // Only WI0 spins; the settled count is broadcast so every WI's token
                        // loop bound (and peer_offsets running counter) stays identical.
                        // Mirrors internode_ll.cpp's proven poll loop (ll_read_flag64 +
                        // env-tunable poll cap). The former hard-coded 4M cap sat mid cold-
                        // start distribution and silently dropped in-flight flags.
                        int count = 0;
                        if (local_id == 0) {
                            long raw0 = 0;
                            // CUDA-faithful multi-QP: each channel c posts its own tail flag on
                            // qp c after chunk c's payload. Wait for ALL num_qp_ch flags (=> all
                            // byte chunks placed) before reading [0,count). Every flag carries
                            // -count-1; use flag[0] for the count. A never-landing flag falls
                            // through the bounded poll cap => undercount, never a hang.
                            for (int c = 0; c < num_qp_ch; ++c) {
                                uint64_t spins = 0;
                                long raw = 0;
                                while (true) {
                                    raw = internode_read_flag64(rdma_flag + c, rdma_flag_lsc_mode);
                                    if (raw != 0) break;
                                    if (++spins >= rdma_poll_cap) break;
                                    visa_spin_hint();
                                }
                                if (c == 0) raw0 = raw;
                            }
                            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                            count = (raw0 == 0) ? 0 : static_cast<int>(-raw0 - 1);
                            // Clamp to send-ring capacity: a legit count is <= num_recv_tokens,
                            // so this is a no-op normally; it bounds the forward loop and every
                            // dst_idx against a stale/garbage flag so a runaway loop or OOB write
                            // (which would corrupt barrier/QP state and hang a later dispatch)
                            // cannot happen.
                            if (count < 0) count = 0;
                            if (count > num_recv_tokens) count = num_recv_tokens;
                        }
                        count = sycl::group_broadcast(group, count, 0);
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();
                        for (int i = 0; i < count; ++i) {
                            SourceMeta sm;
                            // Cached reads (coherent after lsc_fence_sysacq, same as payload):
                            // the former uncached uc_loads ran redundantly on every work-item.
                            sm.src_rdma_rank = rdma_m[i].src_rdma_rank;
                            sm.is_token_in_nvl_rank_bits = rdma_m[i].is_token_in_nvl_rank_bits;
                            sm.src_nvl_rank = rdma_m[i].src_nvl_rank;
                            for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                                if (((sm.is_token_in_nvl_rank_bits >> peer) & 1) == 0) continue;
                                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                                auto* fwd_base = peer_buf + fwd_base_offset;
                                auto* fwd_x = fwd_base + fwd_layout.fwd_x_offset;
                                auto* fwd_m = reinterpret_cast<SourceMeta*>(fwd_base + fwd_layout.fwd_meta_offset);
                                auto* fwd_idx = reinterpret_cast<topk_idx_t*>(fwd_base + fwd_layout.fwd_topk_idx_offset);
                                auto* fwd_wt = reinterpret_cast<float*>(fwd_base + fwd_layout.fwd_topk_weights_offset);
                                auto* fwd_scales = reinterpret_cast<float*>(fwd_base + fwd_layout.fwd_x_scales_offset);
                                const int local_idx = peer_offsets[peer]++;
                                if (local_idx >= fwd_layout.plane_tokens) continue;  // per-plane fwd capacity guard
                                                                                     // (uniform: all lanes track
                                                                                     // peer_offsets identically)
                                const int dst_idx = fwd_plane_token_base + local_idx;
                                auto* src_row = rdma_x + static_cast<size_t>(i) * row_bytes;
                                auto* dst_row = fwd_x + static_cast<size_t>(dst_idx) * row_bytes;
                                // int4-vectorized copy: the lsc_fence_sysacq above already
                                // invalidated the GPU cache to the system domain, so plain
                                // (re-cached) loads observe the NIC-delivered RDMA bytes. This
                                // replaces a per-BYTE uncached uc_load loop (row_bytes uncached
                                // transactions/token) with 16-byte transfers, matching CUDA's
                                // int4 UNROLLED_WARP_COPY.
                                faithful_coop_copy(dst_row, src_row, row_bytes, local_id, kComputeWGSize);
                                if (local_id == 0) {
                                    SourceMeta fwd_sm = sm;
                                    fwd_sm.is_token_in_nvl_rank_bits = i;
                                    fwd_m[dst_idx] = fwd_sm;
                                    if (topk_idx != nullptr) {
                                        for (int k = 0; k < num_topk; ++k) {
                                            fwd_idx[dst_idx * num_topk + k] = uc_load(&rdma_idx[i * num_topk + k]);
                                            fwd_wt[dst_idx * num_topk + k] = uc_load(&rdma_wt[i * num_topk + k]);
                                        }
                                    }
                                    if (x_scales != nullptr) {
                                        for (int s = 0; s < num_scales; ++s)
                                            fwd_scales[dst_idx * num_scales + s] = uc_load(&rdma_scales[i * num_scales + s]);
                                    }
                                }
                            }
                        }

                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        lsc_fence_sysrel();
                        sycl::group_barrier(group);
                        if (local_id == 0) {
                            for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                                auto* fwd_base = peer_buf + fwd_base_offset;
                                auto* fwd_counts = reinterpret_cast<int*>(fwd_base + fwd_layout.fwd_count_offset);
                                fwd_counts[fwd_plane_count_base + src_rdma] = peer_offsets[peer] - before[peer];
                            }
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                            lsc_fence_sysrel();
                        }
                    }
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    lsc_fence_sysrel();
                    sycl::group_barrier(group);
                });
        });

        // ---- F-K5: NVL barrier before shared Assemble/Head consume fwd_* ----
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchFwdBarrierKernel>(
                sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
                [=](sycl::nd_item<1> item) {
                    nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 1, num_nvl_ranks, item);
                });
        });
    }

    // ===== Producer-PUSH intra-node NVL exchange (gap #2) =====
    // CUDA-faithful PRODUCER-PUSH (internode.cu kRDMAAndNVLForwarder / kNVLReceivers):
    // the producer writes tokens into the DESTINATION peer's recv ring; the consumer
    // reads only its OWN buffer. Deterministic placement (one source plane per producer,
    // packed in token order) is used instead of a dynamic ring so there is NO unbounded
    // back-pressure spin (BMG GuC watchdog safe). This replaces the former consumer-PULL
    // in CombinedDispatchAssembleKernel/CombinedDispatchHeadKernel that read peer send
    // buffers over IPC (unstable on BMG+igub).
    //
    // F-K8 NvlPush: grid = 1 WG. Each rank reads its OWN send buffer (packed in F-K1) and
    // WRITES its intra-node tokens (destined for a local peer dst_nvl) into that peer's
    // nvlrecv plane [nvl_rank], packed in token order. WI0 publishes the per-plane count
    // after a device/system release fence (remote WRITE + local READ only, zero remote
    // reads).
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<FaithfulDispatchNvlPushKernel>(
            sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)),
            [=](sycl::nd_item<1> item) {
                auto group = item.get_group();
                const int local_id = static_cast<int>(item.get_local_id(0));
                auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                auto* my_send_x = my_buf + layout.send_x_offset;
                auto* my_send_m = reinterpret_cast<SourceMeta*>(my_buf + layout.send_meta_offset);
                auto* my_send_idx = reinterpret_cast<topk_idx_t*>(my_buf + layout.send_topk_idx_offset);
                auto* my_send_wt = reinterpret_cast<float*>(my_buf + layout.send_topk_weights_offset);
                auto* my_send_scales = reinterpret_cast<float*>(my_buf + layout.send_x_scales_offset);
                auto* my_send_is_in_rank = reinterpret_cast<bool*>(my_buf + layout.send_is_token_in_rank_offset);
                // The whole GPU's own send buffer is coherent (same device, kernel
                // boundary after F-K1). Order our own read before the remote writes.
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);

                for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                    const int dst_rank = my_rdma_rank * num_nvl_ranks + dst_nvl;
                    auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[dst_nvl]);
                    auto* rc_base = peer_buf + nvlrecv_base_offset;
                    auto* rc_x = rc_base + nvlrecv_layout.fwd_x_offset;
                    auto* rc_m = reinterpret_cast<SourceMeta*>(rc_base + nvlrecv_layout.fwd_meta_offset);
                    auto* rc_idx = reinterpret_cast<topk_idx_t*>(rc_base + nvlrecv_layout.fwd_topk_idx_offset);
                    auto* rc_wt = reinterpret_cast<float*>(rc_base + nvlrecv_layout.fwd_topk_weights_offset);
                    auto* rc_scales = reinterpret_cast<float*>(rc_base + nvlrecv_layout.fwd_x_scales_offset);
                    auto* rc_counts = reinterpret_cast<int*>(rc_base + nvlrecv_layout.fwd_count_offset);
                    const int plane_base = nvl_rank * nvlrecv_layout.plane_tokens;
                    int slot = 0;
                    for (int t = 0; t < num_tokens; ++t) {
                        if (!my_send_is_in_rank[t * num_ranks + dst_rank]) continue;
                        if (slot >= nvlrecv_layout.plane_tokens) break;  // per-plane capacity guard
                        const int dst_idx = plane_base + slot;
                        auto* src_row = my_send_x + static_cast<size_t>(t) * row_bytes;
                        auto* dst_row = rc_x + static_cast<size_t>(dst_idx) * row_bytes;
                        faithful_coop_copy(dst_row, src_row, row_bytes, local_id, kComputeWGSize);
                        if (local_id == 0) {
                            rc_m[dst_idx] = my_send_m[t];
                            if (topk_idx != nullptr) {
                                for (int k = 0; k < num_topk; ++k) {
                                    rc_idx[dst_idx * num_topk + k] = my_send_idx[t * num_topk + k];
                                    rc_wt[dst_idx * num_topk + k] = my_send_wt[t * num_topk + k];
                                }
                            }
                            if (x_scales != nullptr) {
                                for (int s = 0; s < num_scales; ++s)
                                    rc_scales[dst_idx * num_scales + s] = my_send_scales[t * num_scales + s];
                            }
                        }
                        ++slot;
                    }
                    // Publish this producer plane's settled count into the dest peer AFTER a
                    // release fence so the consumer's acquire observes payload-then-count.
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    lsc_fence_sysrel();
                    sycl::group_barrier(group);
                    if (local_id == 0) {
                        rc_counts[nvl_rank] = slot;
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        lsc_fence_sysrel();
                    }
                    sycl::group_barrier(group);
                }
            });
    });

    // F-K9 NvlPushBarrier: all producers' intra pushes landed before the consumer reads
    // its OWN nvlrecv region (ordering via kernel boundary + device-scope NVL barrier).
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<FaithfulDispatchNvlPushBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) {
                nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 2, num_nvl_ranks, item);
            });
    });

    // ===== SHARED: Assemble + Head (identical output for both transports) =====
    // CUDA-faithful local-expert remap of recv_topk_idx (internode.cu:1060-61,1176-79):
    // rewrite global expert ids to this receiver's local range, dropping non-local
    // selections (idx -> -1, weight -> 0). Disabled when num_experts == 0 (cached mode).
    const int num_local_experts_a = (num_experts > 0 && num_ranks > 0) ? num_experts / num_ranks : 0;
    const bool do_expert_remap_a = (recv_topk_idx != nullptr && num_local_experts_a > 0);
    const int local_expert_begin_a = do_expert_remap_a ? rank * num_local_experts_a : 0;
    const int local_expert_end_a = local_expert_begin_a + num_local_experts_a;
    // FUSION: ChannelCounts → Assemble → Head. The former DispatchChannelCountsKernel
    // (channel prefix sums) is run as a pre-pass inside Assemble. The former
    // CombinedDispatchHeadKernel (send_nvl_head/send_rdma_head handles) runs as a
    // trailing pass after peer_counts publish. Eliminates 2 queue.submit + 2
    // queue.wait. Only the CountsBarrier must stay (cross-rank nvl_barrier between
    // publish and read).
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchAssembleKernel>(
            sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)), [=](sycl::nd_item<1> item) {
            auto group = item.get_group();
            const int local_id = static_cast<int>(item.get_local_id(0));
            // Acquire fence: invalidate any stale local cache and order all
            // subsequent reads of the NVL peers' forwarded/send buffers
            // (IPC-mapped remote GPU memory) mirroring CUDA's pattern.
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);

            // ---- PHASE 0: channel-count prefix sums (was DispatchChannelCountsKernel) ----
            // Compute gbl_channel_prefix_matrix and rdma_channel_prefix_matrix
            // from is_token_in_rank. Strided across work-items; reduce per WG.
            {
                const int total_gbl_rows = num_ranks * num_channels;
                const int total_rdma_rows = num_rdma_ranks * num_channels;
                const int total_rows = std::max(total_gbl_rows, total_rdma_rows);
                for (int row_id = local_id; row_id < total_rows; row_id += kComputeWGSize) {
                    if (row_id < total_gbl_rows && gbl_channel_prefix_matrix != nullptr) {
                        const int dst_rank = row_id / num_channels;
                        const int c = row_id % num_channels;
                        const int ch_start = static_cast<int>((static_cast<int64_t>(num_tokens) * c) / num_channels);
                        const int ch_end = static_cast<int>((static_cast<int64_t>(num_tokens) * (c + 1)) / num_channels);
                        int count = 0;
                        for (int token = ch_start; token < ch_end; ++token)
                            count += is_token_in_rank[token * num_ranks + dst_rank] ? 1 : 0;
                        // Single-WG kernel: no cross-WG reduce needed
                        int cumulative = count;
                        for (int pc = 0; pc < c; ++pc) {
                            const int ps = static_cast<int>((static_cast<int64_t>(num_tokens) * pc) / num_channels);
                            const int pe = static_cast<int>((static_cast<int64_t>(num_tokens) * (pc + 1)) / num_channels);
                            for (int t = ps; t < pe; ++t)
                                cumulative += is_token_in_rank[t * num_ranks + dst_rank] ? 1 : 0;
                        }
                        gbl_channel_prefix_matrix[row_id] = cumulative;
                    }
                    if (row_id < total_rdma_rows && rdma_channel_prefix_matrix != nullptr) {
                        const int dst_rdma = row_id / num_channels;
                        const int c = row_id % num_channels;
                        const int ch_start = static_cast<int>((static_cast<int64_t>(num_tokens) * c) / num_channels);
                        const int ch_end = static_cast<int>((static_cast<int64_t>(num_tokens) * (c + 1)) / num_channels);
                        int count = 0;
                        for (int token = ch_start; token < ch_end; ++token) {
                            bool hit = false;
                            for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                                if (is_token_in_rank[token * num_ranks + dst_rdma * num_nvl_ranks + dst_nvl]) {
                                    hit = true; break;
                                }
                            }
                            count += hit ? 1 : 0;
                        }
                        int cumulative = count;
                        for (int pc = 0; pc < c; ++pc) {
                            const int ps = static_cast<int>((static_cast<int64_t>(num_tokens) * pc) / num_channels);
                            const int pe = static_cast<int>((static_cast<int64_t>(num_tokens) * (pc + 1)) / num_channels);
                            for (int t = ps; t < pe; ++t) {
                                bool hit = false;
                                for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                                    if (is_token_in_rank[t * num_ranks + dst_rdma * num_nvl_ranks + dst_nvl]) {
                                        hit = true; break;
                                    }
                                }
                                cumulative += hit ? 1 : 0;
                            }
                        }
                        rdma_channel_prefix_matrix[row_id] = cumulative;
                    }
                }
            }
            // Barrier: channel counts must be fully written before the per_src_count
            // scan that writes gbl/rdma prefix sums (those are separate arrays, but
            // the scan reads recv_gbl_rank_prefix_sum and we want a consistent view).
            sycl::group_barrier(group);

            // ---- PHASE 1: Assemble recv_x (original body) ----
            constexpr int kMaxRanks = 64;
            // per_src_count/cursors are recomputed identically by every
            // work-item from deterministic reads, so token placement (pos) is
            // consistent across the work-group without a broadcast. The bulk
            // row copy is split by local_id; per-token scalar metadata is
            // written by WI 0 only.
            int per_src_count[kMaxRanks] = {0};
            int cursors[kMaxRanks] = {0};

            auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
            auto* my_fwd_base = my_buf + fwd_base_offset;
            auto* my_fwd_x = my_fwd_base + fwd_layout.fwd_x_offset;
            auto* my_fwd_m = reinterpret_cast<SourceMeta*>(my_fwd_base + fwd_layout.fwd_meta_offset);
            auto* my_fwd_idx = reinterpret_cast<topk_idx_t*>(my_fwd_base + fwd_layout.fwd_topk_idx_offset);
            auto* my_fwd_wt = reinterpret_cast<float*>(my_fwd_base + fwd_layout.fwd_topk_weights_offset);
            auto* my_fwd_scales = reinterpret_cast<float*>(my_fwd_base + fwd_layout.fwd_x_scales_offset);
            auto* my_fwd_counts = reinterpret_cast<int*>(my_fwd_base + fwd_layout.fwd_count_offset);

            // Producer-PUSH intra region: this rank's OWN recv-staging (peers pushed here).
            auto* my_rc_base = my_buf + nvlrecv_base_offset;
            auto* my_rc_x = my_rc_base + nvlrecv_layout.fwd_x_offset;
            auto* my_rc_m = reinterpret_cast<SourceMeta*>(my_rc_base + nvlrecv_layout.fwd_meta_offset);
            auto* my_rc_idx = reinterpret_cast<topk_idx_t*>(my_rc_base + nvlrecv_layout.fwd_topk_idx_offset);
            auto* my_rc_wt = reinterpret_cast<float*>(my_rc_base + nvlrecv_layout.fwd_topk_weights_offset);
            auto* my_rc_scales = reinterpret_cast<float*>(my_rc_base + nvlrecv_layout.fwd_x_scales_offset);
            auto* my_rc_counts = reinterpret_cast<int*>(my_rc_base + nvlrecv_layout.fwd_count_offset);

            // PASS 1: count tokens per src_rank (intra peers pushed into OUR ring).
            for (int src_nvl = 0; src_nvl < num_nvl_ranks; ++src_nvl) {
                const int src_rank = my_rdma_rank * num_nvl_ranks + src_nvl;
                // Local read of the count producer src_nvl PUSHED into our plane. Clamp to
                // plane capacity so a stale/garbage count can never drive an OOB scan.
                int c = my_rc_counts[src_nvl];
                if (c < 0) c = 0;
                if (c > nvlrecv_layout.plane_tokens) c = nvlrecv_layout.plane_tokens;
                per_src_count[src_rank] += c;
            }
            // Per-plane scan: plane p's forwarded tokens live in the disjoint slice
            // [p*plane_tokens, ...), packed in src_rdma order (peer_offsets monotonic
            // within a forwarder). num_fwd_planes==1 (leader) reduces to the flat scan.
            for (int p = 0; p < num_fwd_planes; ++p) {
                int fwd_off_scan = p * fwd_layout.plane_tokens;
                for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                    const int count = my_fwd_counts[p * num_rdma_ranks + src_rdma];
                    for (int i = 0; i < count; ++i) {
                        const SourceMeta sm = my_fwd_m[fwd_off_scan + i];
                        const int src_rank = sm.src_rdma_rank * num_nvl_ranks + sm.src_nvl_rank;
                        per_src_count[src_rank] += 1;
                    }
                    fwd_off_scan += count;
                }
            }

            // Build exclusive prefix into cursors[] and inclusive prefix into output prefix-sum arrays.
            int total = 0;
            for (int s = 0; s < num_ranks; ++s) {
                cursors[s] = total;
                total += per_src_count[s];
            }
            if (recv_gbl_rank_prefix_sum != nullptr && local_id == 0) {
                int prefix = 0;
                for (int s = 0; s < num_ranks; ++s) {
                    prefix += per_src_count[s];
                    recv_gbl_rank_prefix_sum[s] = prefix;
                    if (recv_gbl_channel_prefix_matrix != nullptr) {
                        const int base = prefix - per_src_count[s];
                        for (int c = 0; c < num_channels; ++c) {
                            recv_gbl_channel_prefix_matrix[s * num_channels + c] =
                                base + ((per_src_count[s] * (c + 1)) / num_channels);
                        }
                    }
                }
            }
            if (recv_rdma_rank_prefix_sum != nullptr && local_id == 0) {
                int prefix = 0;
                for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                    int count = 0;
                    for (int snvl = 0; snvl < num_nvl_ranks; ++snvl) {
                        count += per_src_count[src_rdma * num_nvl_ranks + snvl];
                    }
                    prefix += count;
                    recv_rdma_rank_prefix_sum[src_rdma] = prefix;
                    if (recv_rdma_channel_prefix_matrix != nullptr) {
                        const int base = prefix - count;
                        for (int c = 0; c < num_channels; ++c) {
                            recv_rdma_channel_prefix_matrix[src_rdma * num_channels + c] =
                                base + ((count * (c + 1)) / num_channels);
                        }
                    }
                }
            }

            // PASS 2: place each token at cursors[src_rank]++ so recv_x is grouped by src_rank in canonical order.
            for (int src_nvl = 0; src_nvl < num_nvl_ranks; ++src_nvl) {
                const int src_rank = my_rdma_rank * num_nvl_ranks + src_nvl;
                // Read our OWN recv-staging plane that producer src_nvl PUSHED (local read).
                // Tokens were pushed in the producer's token order, so scanning slots 0..count
                // reproduces the exact canonical ordering the consumer-PULL path produced.
                const int plane_base = src_nvl * nvlrecv_layout.plane_tokens;
                int c = my_rc_counts[src_nvl];
                if (c < 0) c = 0;
                if (c > nvlrecv_layout.plane_tokens) c = nvlrecv_layout.plane_tokens;
                for (int i = 0; i < c; ++i) {
                    const int sidx = plane_base + i;
                    const int pos = cursors[src_rank]++;
                    auto* src_row = my_rc_x + static_cast<size_t>(sidx) * row_bytes;
                    auto* dst_row = dst + static_cast<size_t>(pos) * row_bytes;
                    faithful_coop_copy(dst_row, src_row, row_bytes, local_id, kComputeWGSize);
                    if (local_id == 0) {
                        if (meta != nullptr) {
                            meta[pos] = my_rc_m[sidx];
                        }
                        if (recv_topk_idx != nullptr) {
                            for (int k = 0; k < num_topk; ++k) {
                                const auto gv = my_rc_idx[sidx * num_topk + k];
                                if (do_expert_remap_a) {
                                    const bool loc = (gv >= local_expert_begin_a && gv < local_expert_end_a);
                                    recv_topk_idx[pos * num_topk + k] = loc ? static_cast<topk_idx_t>(gv - local_expert_begin_a)
                                                                            : static_cast<topk_idx_t>(-1);
                                    recv_topk_weights[pos * num_topk + k] = loc ? my_rc_wt[sidx * num_topk + k] : 0.0f;
                                } else {
                                    recv_topk_idx[pos * num_topk + k] = gv;
                                    recv_topk_weights[pos * num_topk + k] = my_rc_wt[sidx * num_topk + k];
                                }
                            }
                        }
                        if (recv_x_scales != nullptr) {
                            for (int s = 0; s < num_scales; ++s) {
                                recv_x_scales[pos * num_scales + s] = my_rc_scales[sidx * num_scales + s];
                            }
                        }
                    }
                }
            }
            for (int p = 0; p < num_fwd_planes; ++p) {
                int fwd_offset = p * fwd_layout.plane_tokens;
                for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                    const int count = my_fwd_counts[p * num_rdma_ranks + src_rdma];
                    for (int i = 0; i < count; ++i) {
                        const int idx = fwd_offset + i;
                        const SourceMeta sm = my_fwd_m[idx];
                        const int src_rank = sm.src_rdma_rank * num_nvl_ranks + sm.src_nvl_rank;
                        const int pos = cursors[src_rank]++;
                        auto* src_row = my_fwd_x + static_cast<size_t>(idx) * row_bytes;
                        auto* dst_row = dst + static_cast<size_t>(pos) * row_bytes;
                        faithful_coop_copy(dst_row, src_row, row_bytes, local_id, kComputeWGSize);
                        if (local_id == 0) {
                            if (meta != nullptr) {
                                meta[pos] = sm;
                            }
                            if (recv_topk_idx != nullptr) {
                                for (int k = 0; k < num_topk; ++k) {
                                    const auto gv = my_fwd_idx[idx * num_topk + k];
                                    if (do_expert_remap_a) {
                                        const bool loc = (gv >= local_expert_begin_a && gv < local_expert_end_a);
                                        recv_topk_idx[pos * num_topk + k] = loc ? static_cast<topk_idx_t>(gv - local_expert_begin_a)
                                                                                : static_cast<topk_idx_t>(-1);
                                        recv_topk_weights[pos * num_topk + k] = loc ? my_fwd_wt[idx * num_topk + k] : 0.0f;
                                    } else {
                                        recv_topk_idx[pos * num_topk + k] = gv;
                                        recv_topk_weights[pos * num_topk + k] = my_fwd_wt[idx * num_topk + k];
                                    }
                                }
                            }
                            if (recv_x_scales != nullptr) {
                                for (int s = 0; s < num_scales; ++s) {
                                    recv_x_scales[pos * num_scales + s] = my_fwd_scales[idx * num_scales + s];
                                }
                            }
                        }
                    }
                    fwd_offset += count;
                }
            }

            // Zero-fill leftover rows beyond the actual receive count (capacity may exceed actual).
            for (int idx = total; idx < num_recv_tokens; ++idx) {
                auto* dst_row = dst + static_cast<size_t>(idx) * row_bytes;
                faithful_coop_zero(dst_row, row_bytes, local_id, kComputeWGSize);
                if (local_id == 0) {
                    if (meta != nullptr) {
                        meta[idx].src_rdma_rank = -1;
                        meta[idx].is_token_in_nvl_rank_bits = 0;
                        meta[idx].src_nvl_rank = -1;
                    }
                    if (recv_topk_idx != nullptr) {
                        for (int k = 0; k < num_topk; ++k) {
                            recv_topk_idx[idx * num_topk + k] = -1;
                            recv_topk_weights[idx * num_topk + k] = 0.0f;
                        }
                    }
                    if (recv_x_scales != nullptr) {
                        for (int s = 0; s < num_scales; ++s) {
                            recv_x_scales[idx * num_scales + s] = 0.0f;
                        }
                    }
                }
            }

            // Producer-PUSH of our per_src_count[] (grouped by GLOBAL src rank, incl. fwd)
            // into EVERY local peer's peer_counts slot [nvl_rank]. The Head kernel then
            // derives send_nvl_head base offsets by reading its OWN peer_counts region
            // (local read) instead of PULLing peer send buffers / peer fwd_meta over IPC.
            if (local_id == 0) {
                for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                    auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                    auto* peer_counts = reinterpret_cast<int*>(peer_buf + peer_counts_offset);
                    for (int s = 0; s < num_ranks; ++s)
                        peer_counts[nvl_rank * num_ranks + s] = per_src_count[s];
                }
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                lsc_fence_sysrel();
            }
        });
    });

    // F-K10 CountsBarrier: every rank's per_src_count is visible in all peers before Head.
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<FaithfulDispatchCountsBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) {
                nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 3, num_nvl_ranks, item);
            });
    });

    // FUSED: DispatchChannelCountsKernel is now fused into CombinedDispatchAssembleKernel
    // as a pre-pass (PHASE 0). The separate kernel launch is eliminated.

    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<CombinedDispatchHeadKernel>([=]() {
            // send_nvl_head/send_rdma_head (gap #2): compute the base offset
            // of THIS rank's token group in each local dst peer's recv_x using ONLY local
            // reads. Each dst peer's Assemble pushed its per_src_count[] (grouped by GLOBAL
            // src rank, incl. RDMA-forwarded contributions) into our OWN peer_counts region
            // at slot [dst_nvl]. recv_x on dst=(my_rdma,dst_nvl) is grouped by global src
            // rank ascending, so the base for our group (src rank == my_global_rank) is the
            // exclusive prefix sum of that peer's per_src_count over s < my_global_rank.
            auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
            auto* my_peer_counts = reinterpret_cast<int*>(my_buf + peer_counts_offset);
            // Order the read of the peers' pushed counts after the CountsBarrier release.
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            int intra_base_my[NUM_MAX_NVL_PEERS] = {0};  // base for MY tokens in dst=(my_rdma,dst_nvl)
            for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                int prefix = 0;
                for (int s = 0; s < my_global_rank; ++s) {
                    prefix += my_peer_counts[dst_nvl * num_ranks + s];
                }
                intra_base_my[dst_nvl] = prefix;
            }

            int rdma_head[NUM_MAX_NVL_PEERS] = {0};
            int per_peer_dst_ordinal[NUM_MAX_NVL_PEERS] = {0};  // [dst_nvl] ordinal of my token to dst
            // Only THIS rank's own tokens produce handle entries; read our OWN routing
            // (is_token_in_rank is this rank's input, a local device pointer). No remote reads.
            for (int token = 0; token < num_tokens; ++token) {
                for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                    const int dst_rank = my_rdma_rank * num_nvl_ranks + dst_nvl;
                    if (!is_token_in_rank[token * num_ranks + dst_rank]) {
                        continue;
                    }
                    if (send_nvl_head != nullptr) {
                        send_nvl_head[token * num_ranks + dst_rank] =
                            intra_base_my[dst_nvl] + per_peer_dst_ordinal[dst_nvl];
                    }
                    per_peer_dst_ordinal[dst_nvl] += 1;
                }
                for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                    if (dst_rdma == my_rdma_rank) continue;
                    bool hit = false;
                    for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                        const int dst_rank = dst_rdma * num_nvl_ranks + dst_nvl;
                        if (is_token_in_rank[token * num_ranks + dst_rank]) {
                            hit = true;
                            break;
                        }
                    }
                    if (!hit) continue;
                    // Per-GPU RDMA has one independent RDMA stream per nvl_rank. The recv_pos
                    // stamped by F6 is the ordinal within this rank's same-plane stream, so
                    // the handle counts only this nvl plane.
                    if (send_rdma_head != nullptr) {
                        send_rdma_head[token * num_rdma_ranks + dst_rdma] = rdma_head[dst_rdma];
                    }
                    rdma_head[dst_rdma] += 1;
                }
            }
        });
    });
    queue.wait();
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
                      const int* combined_rdma_head,
                      const int* combined_nvl_head,
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

    const size_t row_bytes = static_cast<size_t>(hidden) * sizeof(dtype_t);
    auto* dst = static_cast<dtype_t*>(combined_x);
    const auto* src = static_cast<const dtype_t*>(x);
    const auto* b0 = static_cast<const dtype_t*>(bias_0);
    const auto* b1 = static_cast<const dtype_t*>(bias_1);
    const auto* sm = static_cast<const SourceMeta*>(src_meta);
    auto* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);

    NvlBufferLayout layout(num_tokens, num_ranks, 1, row_bytes, num_topk, 0);
    // Forward buffer must hold every token forwarded to a single NVL peer,
    // summed across ALL source RDMA ranks (remote ranks plus the self RDMA
    // rank now handled locally). Bound it by num_rdma_ranks * num_combined_tokens
    // so a peer receiving contributions from multiple source RDMA ranks never
    // overruns its forward region. num_nvl_bytes is provisioned far above this.
    const int max_fwd_tokens = num_rdma_ranks * num_combined_tokens;
    // Per-GPU RDMA combine: each nvl_rank RDMA-sends the tokens it holds whose original
    // source nvl-plane == its own nvl_rank, back to (src_rdma, nvl_rank). The receiver
    // (src_rdma, nvl_rank) therefore always forwards to ITSELF (rdma_src_nvl == its own
    // nvl_rank), so unlike dispatch there is no concurrent forwarder and the fwd buffer
    // stays single-plane (num_planes==1).
    NvlForwardLayout fwd_layout(max_fwd_tokens, row_bytes, num_topk, 0, num_rdma_ranks);
    // fwd_base_offset must be IDENTICAL across all ranks, otherwise a rank
    // writing into a peer's NVL buffer via IPC at its own offset will land at
    // a different address than where the peer reads from. Use the global upper
    // bound (max possible combine input rows = num_nvl_ranks *
    // num_combined_tokens) so the offset is rank-invariant. Local layout above
    // still uses num_tokens for correct local data sizing.
    const int max_combine_tokens = num_nvl_ranks * num_combined_tokens;
    NvlBufferLayout fwd_anchor_layout(max_combine_tokens, num_ranks, 1, row_bytes, num_topk, 0);
    const size_t fwd_base_offset = align_offset(fwd_anchor_layout.total_bytes, 128);

    // ---- Producer-PUSH intra-node NVL combine staging (gap #2) -------------------------
    // On BMG+igub a cross-rank IPC READ of a peer's Pack output is unstable, while a
    // cross-rank IPC WRITE is stable. So instead of the gather kernels READING each peer's
    // Pack output, each rank WRITES its Pack output into every peer's combine-staging plane
    // (plane == producer nvl_rank), and the gathers read only their OWN plane locally.
    // num_nvl_ranks disjoint source planes, plane_tokens == max_combine_tokens (rank-
    // invariant) so the meta/topk
    // sub-array offsets are FIXED (rank-invariant, no count-dependent arithmetic on the
    // read side). num_tokens (the combine recv/packed count that Pack stores at
    // The producer writes its plane at plane_base = nvl_rank * plane_tokens into every
    // peer's buffer; the consumer reads plane_base = dst_nvl * plane_tokens from its own
    // buffer. For those offsets to agree across ranks, plane_tokens MUST be rank-invariant.
    // num_tokens is the per-rank dispatch-receive count and DIFFERS across ranks (the OLD
    // PULL path avoided planes: it read each peer's send_x at a fixed base indexed by
    // peer_recv_pos, so per-rank num_tokens never entered the offset). Use the invariant
    // upper bound max_combine_tokens (== num_nvl_ranks*num_combined_tokens, >= num_tokens);
    // the pushed count (== num_tokens) is clamped to it. num_scales==0; NvlForwardLayout's
    // fwd_topk_weights sub-array holds the combine topk_weights (float). Placed after the
    // fwd region (rank-invariant offset, reuses otherwise-unused NVL buffer space).
    NvlForwardLayout combine_stage_layout(max_combine_tokens, row_bytes, num_topk, 0, /*num_rdma_ranks=*/1, num_nvl_ranks);
    const size_t combine_stage_base = align_offset(fwd_base_offset + fwd_layout.total_bytes, 128);

    // Each RDMA region must hold up to num_nvl_ranks * num_combined_tokens
    // tokens (the max number of tokens any source rdma_rank could have
    // originally sent to this rdma_rank's NVL peers). Use num_combined_tokens
    // (the original sender count, identical on all ranks) instead of num_tokens
    // (per-rank receive count from dispatch, which DIFFERS across ranks and
    // would yield mismatched rdma_region_bytes between sender/receiver).
    const int max_rdma_tokens = num_nvl_ranks * num_combined_tokens;
    const size_t rdma_x_size = static_cast<size_t>(max_rdma_tokens) * row_bytes;
    const size_t rdma_topk_wt_size = static_cast<size_t>(max_rdma_tokens) * num_topk * sizeof(float);
    const size_t rdma_recv_pos_size = static_cast<size_t>(max_rdma_tokens) * sizeof(int);
    const size_t rdma_src_nvl_size = static_cast<size_t>(max_rdma_tokens) * sizeof(int);
    const size_t rdma_topk_wt_offset = align_offset(rdma_x_size, alignof(float));
    const size_t rdma_recv_pos_offset = align_offset(rdma_topk_wt_offset + rdma_topk_wt_size, alignof(int));
    const size_t rdma_src_nvl_offset = align_offset(rdma_recv_pos_offset + rdma_recv_pos_size, alignof(int));
    const size_t rdma_count_offset = align_offset(rdma_src_nvl_offset + rdma_src_nvl_size, alignof(int));
    // Faithful path uses a 64-bit AMO count flag at rdma_flag_offset
    // (ishmemx_long_atomic_add_qp, -count-1), mirroring the dispatch F4b flag.
    // Kept separate from rdma_count_offset and INSIDE the region so every rank
    // computes the identical rdma_region_bytes (sender/receiver agree). With
    // multi-QP (num_qp_ch>1) the flag is an ARRAY of num_qp_ch longs (one per
    // channel/QP); default num_qp_ch==1 is byte-for-byte the single-QP layout.
    // The serial fallback simply never touches these slots.
    // GUARD: identical to dispatch_nvl_rdma -- the blocking-put A/B path uses the
    // plain (round-robin) ishmem_putmem* APIs because no single-work-item per-QP put
    // exists in this iSHMEM build, so pin to one channel/QP in that mode. Both
    // launchers must apply this identically or the RDMA region layouts disagree.
    const int num_qp_ch = internode_blocking_put() ? 1 : internode_num_qp_channels();
    const size_t rdma_flag_offset = align_offset(rdma_count_offset + sizeof(int), alignof(long));
    const size_t rdma_region_bytes =
        align_offset(rdma_flag_offset + static_cast<size_t>(num_qp_ch) * sizeof(long), 128);
    const size_t rdma_send_base = static_cast<size_t>(num_rdma_ranks) * rdma_region_bytes;

    const size_t total_combined = static_cast<size_t>(num_combined_tokens) * hidden;
    const size_t total_topk = static_cast<size_t>(num_combined_tokens) * num_topk;
    const size_t total_recv_regions = static_cast<size_t>(num_rdma_ranks) * rdma_region_bytes;
    const size_t init_range = std::max({total_combined, total_topk, total_recv_regions, static_cast<size_t>(1)});

    // Faithful combine gate (shared with dispatch). Empty/"0" => OFF (serial
    // fallback). All faithful transport primitives (blocking put + AMO flag)
    // were validated by the dispatch port; here they replace the serial
    // RdmaPush's two barrier_all collectives + the count-sentinel handshake.
    const uint64_t rdma_poll_cap = internode_poll_cap();
    const int rdma_flag_lsc_mode = internode_flag_lsc_mode();
    const bool faithful_post_amo_quiet = internode_post_amo_quiet();
    const bool faithful_force_db = internode_force_db();          // FC5b put doorbell
    const bool faithful_blocking_put = internode_blocking_put();  // FC5b blocking payload put
    const bool faithful_nbi_mode = internode_nbi_mode();          // FC5b NBI vs blocking API
    const bool faithful_par_gather = internode_par_gather();      // FC5b grid-parallel gather

    // FUSION: init zeroing merged into Pack kernel below (WG 0's cooperative pre-pass).
    // Eliminates the separate CombinedCombineInitKernel launch.

    queue.submit([&](sycl::handler& cgh) {
        const size_t pack_groups = static_cast<size_t>(std::max(num_tokens, 1));
        cgh.parallel_for<CombinedCombinePackKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(pack_groups * kComputeWGSize), sycl::range<1>(kComputeWGSize)), [=](sycl::nd_item<1> item) {
            auto group = item.get_group();
            const int local_id = static_cast<int>(item.get_local_id(0));
            const int t = static_cast<int>(item.get_group(0));
            // ---- PHASE 0: cooperative init zero (was CombinedCombineInitKernel + CombineNvlPlaneInitKernel) ----
            // Only WG 0 performs the zero initiation; the init_range is small relative
            // to hidden-sized token copies, so it's fine to serialize here.
            if (t == 0) {
                for (size_t linear = static_cast<size_t>(local_id); linear < init_range; linear += kComputeWGSize) {
                    if (linear < total_combined) {
                        dst[linear] = dtype_t{};
                    }
                    if (linear < total_topk && combined_topk_weights != nullptr) {
                        combined_topk_weights[linear] = 0.0f;
                    }
                    if (nvl_rank == 0 && linear < total_recv_regions) {
                        rdma_base[linear] = 0;
                    }
                }
                // Also seed cs_meta src_nvl_rank=-1 sentinels (was CombineNvlPlaneInitKernel).
                // This must happen BEFORE the PackBarrier (which is a cross-rank nvl_barrier),
                // and it does because PackBarrier is a separate kernel after this one.
                const size_t total_cs_slots = static_cast<size_t>(num_nvl_ranks) *
                                              static_cast<size_t>(combine_stage_layout.plane_tokens);
                auto* my_buf_cs = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                auto* cs_meta = reinterpret_cast<SourceMeta*>(
                    my_buf_cs + combine_stage_base + combine_stage_layout.fwd_meta_offset);
                for (size_t i = static_cast<size_t>(local_id); i < total_cs_slots; i += kComputeWGSize) {
                    cs_meta[i].src_nvl_rank = -1;
                }
            }
            sycl::group_barrier(group);
            // ---- PHASE 1: token packing (original Pack body) ----
            auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
            auto* combine_count = reinterpret_cast<int*>(my_buf + layout.count_offset);
            auto* combine_x = reinterpret_cast<dtype_t*>(my_buf + layout.send_x_offset);
            auto* combine_meta = reinterpret_cast<SourceMeta*>(my_buf + layout.send_meta_offset);
            auto* combine_topk = reinterpret_cast<float*>(my_buf + layout.send_topk_weights_offset);
            if (t == 0 && local_id == 0) {
                *combine_count = num_tokens;
            }
            // One work-group per token: the token loop is now parallel across
            // work-groups; each work-group strides its token's hidden row.
            if (t < num_tokens) {
                faithful_coop_copy(reinterpret_cast<uint8_t*>(combine_x + static_cast<size_t>(t) * hidden),
                                   reinterpret_cast<const uint8_t*>(src + static_cast<size_t>(t) * hidden),
                                   static_cast<size_t>(hidden) * sizeof(dtype_t), local_id, kComputeWGSize);
                if (num_topk > 0) {
                    for (int k = local_id; k < num_topk; k += kComputeWGSize) {
                        combine_topk[t * num_topk + k] = topk_weights ? topk_weights[t * num_topk + k] : 0.0f;
                    }
                }
                if (local_id == 0) {
                    combine_meta[t] = sm[t];
                }
            }
            // Release fence: flush this rank's packed combine buffer across PCIe
            // so peers' RdmaSend reduction reads up-to-date payload. Every
            // work-item flushes its own writes (WI 0's fence alone would not
            // flush the other work-items' send_x writes to system scope).
            sycl::group_barrier(group);
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        });
    });

    // FUSED: CombineNvlPlaneInitKernel (cs_meta src_nvl_rank=-1 sentinel seed) is
    // merged into the Pack kernel above (WG 0's zero pass also seeds the combine-
    // staging meta sentinels). The separate PlaneInit launch is eliminated.

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombinePackBarrierKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base, num_nvl_ranks, item); });
    });

    // ===== Producer-PUSH intra-node NVL combine exchange (gap #2 + selective routing) =====
    // CN-K1 CombineNvlPush: grid = 1 WG. Each rank reads its OWN Pack output and routes each
    // token t to ONLY the consumer that owns it (C = my_meta[t].src_nvl_rank), writing at the
    // VERBATIM index t in that consumer's plane[nvl_rank] (remote WRITE). Verbatim index is
    // required because the reduce indexes plane[dst_nvl][peer_recv_pos] with the dispatch
    // position; token t is owned by exactly one consumer, so writing it once (not to every
    // peer) is ~num_nvl_ranks x less copy while keeping both the reduce and the gather correct.
    // Unwritten slots were seeded to src_nvl_rank=-1 (CombineNvlPlaneInit) so the gather's
    // filter skips them. WI0 publishes each producer's count into every consumer's plane after
    // a release fence. Gather/reduce then read their OWN plane locally (zero remote reads).
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombineNvlPushKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)),
            [=](sycl::nd_item<1> item) {
                auto group = item.get_group();
                const int local_id = static_cast<int>(item.get_local_id(0));
                auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                auto* my_count_ptr = reinterpret_cast<int*>(my_buf + layout.count_offset);
                auto* my_x = reinterpret_cast<dtype_t*>(my_buf + layout.send_x_offset);
                auto* my_meta = reinterpret_cast<SourceMeta*>(my_buf + layout.send_meta_offset);
                auto* my_topk = reinterpret_cast<float*>(my_buf + layout.send_topk_weights_offset);
                // Own Pack output is coherent (same device, kernel boundary after Pack).
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                int count = *my_count_ptr;  // == num_tokens
                if (count < 0) count = 0;
                if (count > combine_stage_layout.plane_tokens) count = combine_stage_layout.plane_tokens;
                const int plane_base = nvl_rank * combine_stage_layout.plane_tokens;
                for (int t = 0; t < count; ++t) {
                    const int dst_c = my_meta[t].src_nvl_rank;  // owning consumer
                    if (dst_c < 0 || dst_c >= num_nvl_ranks) continue;
                    auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[dst_c]);
                    auto* cs_base = peer_buf + combine_stage_base;
                    auto* cs_x = reinterpret_cast<dtype_t*>(cs_base + combine_stage_layout.fwd_x_offset);
                    auto* cs_meta = reinterpret_cast<SourceMeta*>(cs_base + combine_stage_layout.fwd_meta_offset);
                    auto* cs_topk = reinterpret_cast<float*>(cs_base + combine_stage_layout.fwd_topk_weights_offset);
                    const int dst_idx = plane_base + t;  // VERBATIM index (== peer_recv_pos read side)
                    faithful_coop_copy(reinterpret_cast<uint8_t*>(&cs_x[static_cast<size_t>(dst_idx) * hidden]),
                                       reinterpret_cast<const uint8_t*>(&my_x[static_cast<size_t>(t) * hidden]),
                                       static_cast<size_t>(hidden) * sizeof(dtype_t), local_id, kComputeWGSize);
                    if (local_id == 0) {
                        cs_meta[dst_idx] = my_meta[t];  // valid meta (src_nvl_rank==dst_c) overrides -1 sentinel
                        if (num_topk > 0) {
                            for (int k = 0; k < num_topk; ++k)
                                cs_topk[dst_idx * num_topk + k] = my_topk[t * num_topk + k];
                        }
                    }
                }
                sycl::group_barrier(group);
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                lsc_fence_sysrel();
                // Publish our token count into EVERY consumer's plane (each consumer's gather
                // bounds its plane[nvl_rank] iteration by cs_counts[nvl_rank]).
                if (local_id == 0) {
                    for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                        auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                        auto* cs_counts = reinterpret_cast<int*>(
                            peer_buf + combine_stage_base + combine_stage_layout.fwd_count_offset);
                        cs_counts[nvl_rank] = count;
                    }
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    lsc_fence_sysrel();
                }
                sycl::group_barrier(group);
            });
    });

    // CN-K2 CombineNvlPushBarrier: all producers' pushes landed before any gather reads
    // its OWN staging plane (ordering via kernel boundary + device-scope NVL barrier).
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombineNvlPushBarrierKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) {
                nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 1, num_nvl_ranks, item);
            });
    });

    queue.submit([&](sycl::handler& cgh) {
        const size_t rs_groups = static_cast<size_t>(std::max(num_combined_tokens, 1));
        cgh.parallel_for<CombinedCombineRdmaSendKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(rs_groups * kComputeWGSize), sycl::range<1>(kComputeWGSize)), [=](sycl::nd_item<1> item) {
            const int local_id = static_cast<int>(item.get_local_id(0));
            const int ct = static_cast<int>(item.get_group(0));
            if (ct >= num_combined_tokens) return;
            // Producer-PUSH (gap #2): peers PUSHED their Pack output into OUR combine-
            // staging planes. Read only our OWN buffer (local read); acquire orders the
            // read after the CombineNvlPushBarrier release. FIXED plane sub-array offsets
            // (plane_tokens == max_combine_tokens, rank-invariant) replace count-dep arithmetic.
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            lsc_fence_sysacq();
            auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
            auto* cs_base = my_buf + combine_stage_base;
            auto* cs_x = reinterpret_cast<dtype_t*>(cs_base + combine_stage_layout.fwd_x_offset);
            auto* cs_topk = reinterpret_cast<float*>(cs_base + combine_stage_layout.fwd_topk_weights_offset);
            auto* cs_counts = reinterpret_cast<int*>(cs_base + combine_stage_layout.fwd_count_offset);
            const int plane_tokens = combine_stage_layout.plane_tokens;

            // One work-group per combined token: each work-group owns dst[ct],
            // so the accumulation across NVL peers is race-free (no two work-
            // groups write the same output row).
            for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                if (dst_rank / num_nvl_ranks != my_rdma_rank) {
                    continue;
                }
                if (is_combined_token_in_rank != nullptr && !is_combined_token_in_rank[ct * num_ranks + dst_rank]) {
                    continue;
                }
                const int peer_recv_pos = combined_nvl_head[ct * num_ranks + dst_rank];
                const int dst_nvl = dst_rank % num_nvl_ranks;
                // Plane dst_nvl == the data producer dst_nvl PUSHED into our staging.
                int peer_count = cs_counts[dst_nvl];
                if (peer_count < 0) peer_count = 0;
                if (peer_count > plane_tokens) peer_count = plane_tokens;  // count clamp
                if (peer_recv_pos < 0 || peer_recv_pos >= peer_count) {
                    continue;
                }
                const size_t plane_base = static_cast<size_t>(dst_nvl) * plane_tokens;
                auto* peer_x = cs_x + plane_base * hidden;
                for (int h = local_id; h < hidden; h += kComputeWGSize) {
                    float value = static_cast<float>(dst[ct * hidden + h]);
                    value += static_cast<float>(peer_x[peer_recv_pos * hidden + h]);
                    dst[ct * hidden + h] = static_cast<dtype_t>(value);
                }
                if (combined_topk_weights != nullptr) {
                    auto* peer_topk = cs_topk + plane_base * num_topk;
                    for (int k = local_id; k < num_topk; k += kComputeWGSize) {
                        combined_topk_weights[ct * num_topk + k] += peer_topk[peer_recv_pos * num_topk + k];
                    }
                }
            }
        });
    });

    // ==================== FAITHFUL COMBINE RDMA PUSH + FORWARD ====================
    // Replaces the serial RdmaPush's TWO ishmemx_barrier_all_work_group data-path
    // collectives + count-sentinel handshake with the dispatch-proven transport:
    //   FC5a  one pre-put init barrier (zero my recv flags, then the single
    //         cross-PE barrier_all so every PE's flags read 0 before any AMO),
    //   FC5b  leader gather/compact + gate-free BLOCKING ishmem_putmem payload
    //         (commit-gap-safe: never uses the unbounded warp-put commit gate),
    //   FC5c  kernel-boundary 64-bit AMO count flag (quiet_qp + long_atomic_add_qp
    //         of -count-1, posted for EVERY dst_rdma regardless of count so the
    //         receiver poll always terminates),
    //   FC6   FwdWrite that polls that 64-bit flag (internode_read_flag64) instead
    //         of the count sentinel.
    // Init/Pack/PackBarrier/RdmaSend/FwdBarrier/Reduce are shared verbatim, so the
    // reduce math and the output stay byte-identical to the serial fallback.
    {  // faithful (CUDA-parity) combine transport (only path)
        // ---- FC5a: init barrier (mirror dispatch F4a0). Zero MY recv-region
        // flags so remote AMOs land onto 0, then the ONE necessary cross-PE
        // rendezvous. Reached by ALL PEs (not leader-gated) so it cannot hang.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulCombineRdmaBarrierKernel<dtype_t>>(
                sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if (local_id == 0) {
                        for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                            if (src_rdma == my_rdma_rank) continue;
                            auto* fl = reinterpret_cast<long*>(
                                rdma_base + static_cast<size_t>(src_rdma) * rdma_region_bytes + rdma_flag_offset);
                            for (int c = 0; c < num_qp_ch; ++c) uc_store<long>(fl + c, 0L);
                        }
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        lsc_fence_sysrel();
                    }
                    sycl::group_barrier(group);
                    ishmemx_barrier_all_work_group(group);
                    sycl::group_barrier(group);
                });
        });

        // ---- FC5b: leader gather/compact + gate-free BLOCKING payload put. No
        // cross-PE op here, so non-leaders early-return. Self RDMA rank gathers
        // into its LOCAL recv region (no put); remote dst_rdma puts data+count.
        // The compacted count is stashed in the send region's rdma_count field
        // for FC5c to post the flag (== dispatch F4a2 stash).
        // The grid-parallel gather is unsafe for the tiny topk case on BMG:
        // concurrent IPC reads of peer BF16 rows can return stale cache lines,
        // which FP8-dequantized inputs expose as a systematic combine bias. This
        // affects BOTH the leader and per-GPU paths (both read peer rows over
        // IPC). Use the single-WG gather for that small case; at <=64 tokens the
        // combine is barrier-bound, so the parallel gather gives no perf benefit
        // anyway. Larger runs keep the fast path.
        const bool use_par_gather =
            faithful_par_gather && !(combined_topk_weights != nullptr && num_tokens <= 64);
        if (!use_par_gather) {
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulCombineRdmaPutKernel<dtype_t>>(
                sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    // Producer-PUSH (gap #2): read our OWN combine-staging planes (peers
                    // pushed here) with FIXED sub-array offsets. No peer reads.
                    auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                    auto* cs_base = my_buf + combine_stage_base;
                    auto* cs_x = reinterpret_cast<dtype_t*>(cs_base + combine_stage_layout.fwd_x_offset);
                    auto* cs_meta = reinterpret_cast<SourceMeta*>(cs_base + combine_stage_layout.fwd_meta_offset);
                    auto* cs_topk = reinterpret_cast<float*>(cs_base + combine_stage_layout.fwd_topk_weights_offset);
                    auto* cs_counts = reinterpret_cast<int*>(cs_base + combine_stage_layout.fwd_count_offset);
                    const int plane_tokens = combine_stage_layout.plane_tokens;
                    for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                        const bool is_self_rdma = (dst_rdma == my_rdma_rank);
                        auto* region = is_self_rdma
                            ? (rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes)
                            : (rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes);
                        auto* rdma_x = reinterpret_cast<dtype_t*>(region);
                        auto* rdma_wt = reinterpret_cast<float*>(region + rdma_topk_wt_offset);
                        auto* rdma_recv_pos = reinterpret_cast<int*>(region + rdma_recv_pos_offset);
                        auto* rdma_src_nvl = reinterpret_cast<int*>(region + rdma_src_nvl_offset);
                        auto* rdma_count = reinterpret_cast<int*>(region + rdma_count_offset);
                        if (local_id == 0) {
                            *rdma_count = 0;
                        }
                        sycl::group_barrier(group);
                        {
                            // Cooperative gather across the work-group; identical to the
                            // serial RdmaPush gather (uniform count sequence, bulk copy split
                            // by local_id, scalar metadata by WI 0).
                            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                            lsc_fence_sysacq();  // invalidate -> cached int4 loads see pushed data
                            int count = 0;
                            for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                                const size_t plane_base = static_cast<size_t>(peer) * plane_tokens;
                                auto* peer_x = cs_x + plane_base * hidden;
                                auto* peer_meta = cs_meta + plane_base;
                                auto* peer_topk = cs_topk + plane_base * num_topk;
                                int peer_count = cs_counts[peer];
                                if (peer_count < 0) peer_count = 0;
                                if (peer_count > plane_tokens) peer_count = plane_tokens;  // count clamp
                                for (int t = 0; t < peer_count; ++t) {
                                    if (peer_meta[t].src_rdma_rank != dst_rdma) {
                                        continue;
                                    }
                                    // Per-GPU RDMA: this nvl_rank only handles tokens whose
                                    // original source nvl-plane == its own nvl_rank (so it sends
                                    // them back on RDMA plane nvl_rank). Planes partition tokens
                                    // disjointly, replacing the leader funnel.
                                    if (peer_meta[t].src_nvl_rank != nvl_rank) {
                                        continue;
                                    }
                                    faithful_coop_copy(reinterpret_cast<uint8_t*>(&rdma_x[count * hidden]),
                                                       reinterpret_cast<const uint8_t*>(&peer_x[t * hidden]),
                                                       static_cast<size_t>(hidden) * sizeof(dtype_t),
                                                       local_id, kComputeWGSize);
                                    if (combined_topk_weights != nullptr) {
                                        for (int k = local_id; k < num_topk; k += kComputeWGSize) {
                                            rdma_wt[count * num_topk + k] = peer_topk[t * num_topk + k];
                                        }
                                    }
                                    if (local_id == 0) {
                                        rdma_recv_pos[count] = peer_meta[t].is_token_in_nvl_rank_bits;
                                        rdma_src_nvl[count] = peer_meta[t].src_nvl_rank;
                                    }
                                    ++count;
                                }
                            }
                            sycl::group_barrier(group);
                            if (local_id == 0) {
                                *rdma_count = count;  // stashed for FC5c's -count-1 flag
                                if (is_self_rdma) {
                                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                                }
                            }
                        }
                        sycl::group_barrier(group);
                    }
                });
        });
        } else {
            // ---- FC5b (parallel, ORDER-PRESERVING): the reduce (8-Reduce) indexes
            // peer_x[peer_recv_pos] where peer_recv_pos = combined_nvl_head[...] is the
            // dispatch-established position, so the compaction order is SEMANTIC and must
            // match the serial gather byte-for-byte. A cheap serial metadata pass assigns
            // each input token (peer,t) its exact slot; a GRID then copies the rows in
            // parallel. Slots live in a persistent device scratch (grown on demand).
            const int par_per_peer = std::max(num_tokens, 1);
            const size_t slot_n = static_cast<size_t>(num_nvl_ranks) * static_cast<size_t>(par_per_peer);
            static int* s_gather_slot = nullptr;
            static size_t s_gather_slot_n = 0;
            if (slot_n > s_gather_slot_n) {
                if (s_gather_slot != nullptr) sycl::free(s_gather_slot, queue);
                s_gather_slot = sycl::malloc_device<int>(slot_n, queue);
                s_gather_slot_n = slot_n;
            }
            int* gather_slot = s_gather_slot;
            // Pass 1: serial single-WI metadata scan (no row copy -> cheap). Walks peers/
            // tokens in the SAME order as the serial gather; matching tokens get sequential
            // slots. Records gather_slot[peer*par_per_peer + t] and stashes *rdma_count.
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<FaithfulCombineGatherZeroKernel<dtype_t>>(
                    sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)),
                    [=](sycl::nd_item<1> item) {
                        const int local_id = static_cast<int>(item.get_local_id(0));
                        if (local_id != 0) return;
                        // Producer-PUSH (gap #2): read OUR OWN combine-staging planes with
                        // FIXED sub-array offsets (no peer reads).
                        auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                        auto* cs_base = my_buf + combine_stage_base;
                        auto* cs_meta = reinterpret_cast<SourceMeta*>(cs_base + combine_stage_layout.fwd_meta_offset);
                        auto* cs_topk = reinterpret_cast<float*>(cs_base + combine_stage_layout.fwd_topk_weights_offset);
                        auto* cs_counts = reinterpret_cast<int*>(cs_base + combine_stage_layout.fwd_count_offset);
                        const int plane_tokens = combine_stage_layout.plane_tokens;
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();
                        // Slot assignment + scalar metadata, in the exact serial order (must
                        // match the serial gather byte-for-byte). Only the bulk x row copy is
                        // parallelized in Pass 2.
                        for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                            const bool is_self_rdma = (dst_rdma == my_rdma_rank);
                            auto* region = is_self_rdma
                                ? (rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes)
                                : (rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes);
                            auto* rdma_wt = reinterpret_cast<float*>(region + rdma_topk_wt_offset);
                            auto* rdma_recv_pos = reinterpret_cast<int*>(region + rdma_recv_pos_offset);
                            auto* rdma_src_nvl = reinterpret_cast<int*>(region + rdma_src_nvl_offset);
                            int count = 0;
                            for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                                const size_t plane_base = static_cast<size_t>(peer) * plane_tokens;
                                auto* peer_meta = cs_meta + plane_base;
                                auto* peer_topk = cs_topk + plane_base * num_topk;
                                int peer_count = cs_counts[peer];
                                if (peer_count < 0) peer_count = 0;
                                if (peer_count > plane_tokens) peer_count = plane_tokens;  // count clamp
                                for (int t = 0; t < peer_count; ++t) {
                                    if (peer_meta[t].src_rdma_rank != dst_rdma) continue;
                                    if (peer_meta[t].src_nvl_rank != nvl_rank) continue;
                                    gather_slot[static_cast<size_t>(peer) * par_per_peer + t] = count;
                                    if (combined_topk_weights != nullptr) {
                                        for (int k = 0; k < num_topk; ++k)
                                            rdma_wt[count * num_topk + k] = peer_topk[t * num_topk + k];
                                    }
                                    rdma_recv_pos[count] = peer_meta[t].is_token_in_nvl_rank_bits;
                                    rdma_src_nvl[count] = peer_meta[t].src_nvl_rank;
                                    ++count;
                                }
                            }
                            *reinterpret_cast<int*>(region + rdma_count_offset) = count;
                        }
                    });
            });
            // Pass 2: one work-group per input token (peer,t); copy the row + topk +
            // recv_pos/src_nvl into the precomputed slot. Full-GPU parallel row copy.
            const size_t par_groups = slot_n;
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<FaithfulCombineParGatherKernel<dtype_t>>(
                    sycl::nd_range<1>(sycl::range<1>(par_groups * kComputeWGSize),
                                      sycl::range<1>(kComputeWGSize)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                        const int local_id = static_cast<int>(item.get_local_id(0));
                        const size_t g = item.get_group(0);
                        const int peer = static_cast<int>(g / static_cast<size_t>(par_per_peer));
                        const int t = static_cast<int>(g % static_cast<size_t>(par_per_peer));
                        if (peer >= num_nvl_ranks) return;
                        // Producer-PUSH (gap #2): read OUR OWN combine-staging plane (local).
                        auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                        auto* cs_base = my_buf + combine_stage_base;
                        auto* cs_x = reinterpret_cast<dtype_t*>(cs_base + combine_stage_layout.fwd_x_offset);
                        auto* cs_meta = reinterpret_cast<SourceMeta*>(cs_base + combine_stage_layout.fwd_meta_offset);
                        auto* cs_counts = reinterpret_cast<int*>(cs_base + combine_stage_layout.fwd_count_offset);
                        const int plane_tokens = combine_stage_layout.plane_tokens;
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();  // invalidate -> cached int4 loads see pushed data
                        const size_t plane_base = static_cast<size_t>(peer) * plane_tokens;
                        int peer_count = cs_counts[peer];
                        if (peer_count < 0) peer_count = 0;
                        if (peer_count > plane_tokens) peer_count = plane_tokens;  // count clamp
                        if (t >= peer_count) return;
                        auto* peer_x = cs_x + plane_base * hidden;
                        auto* peer_meta = cs_meta + plane_base;
                        const int dst_rdma = peer_meta[t].src_rdma_rank;
                        if (dst_rdma < 0 || dst_rdma >= num_rdma_ranks) return;
                        if (peer_meta[t].src_nvl_rank != nvl_rank) return;
                        const int slot = gather_slot[static_cast<size_t>(peer) * par_per_peer + t];
                        if (slot < 0 || slot >= max_rdma_tokens) return;
                        const bool is_self_rdma = (dst_rdma == my_rdma_rank);
                        auto* region = is_self_rdma
                            ? (rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes)
                            : (rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes);
                        auto* rdma_x = reinterpret_cast<dtype_t*>(region);
                        // Only the bulk x row copy runs in the grid (topk/recv_pos/src_nvl were
                        // written serially in Pass 1). Full-GPU parallel over rows.
                        faithful_coop_copy(reinterpret_cast<uint8_t*>(&rdma_x[slot * hidden]),
                                           reinterpret_cast<const uint8_t*>(&peer_x[t * hidden]),
                                           static_cast<size_t>(hidden) * sizeof(dtype_t),
                                           local_id, kComputeWGSize);
                    });
            });
        }

        // ---- FC5b2: leader payload put (split from FC5b gather so the NIC put and
        // the local compaction are timed independently). Reads the region compacted
        // by FC5b above (visible across the kernel boundary) and puts [0,rdma_count_offset).
        queue.submit([&](sycl::handler& cgh) {
            // GAP#8: TRUE CONCURRENT-WG combine send. One WG per (dst_rdma, channel c) so
            // the C per-channel byte-chunk puts run on C Xe-cores CONCURRENTLY (was a single
            // WG whose C sub-groups drove all qps from one Xe-core). Mirrors the stable
            // dispatch concurrent send (F-K3a/F4b) and internode_ll.cpp LLCombineSendKernel:
            // each qp is EXCLUSIVELY owned + quiesced by exactly one WG, so no cross-WG
            // doorbell contention on a shared qp (the discipline that keeps LL's concurrent
            // grid stable and avoids the doorbell-loss/quiet-spin hang).
            const int send_wgs = num_rdma_ranks * num_qp_ch;
            cgh.parallel_for<FaithfulCombineRdmaPut2Kernel<dtype_t>>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(send_wgs) * kIshmemWGSize),
                                  sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    auto sg = item.get_sub_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    const int wg_id = static_cast<int>(item.get_group_linear_id());
                    const int dst_rdma = wg_id / num_qp_ch;  // WG owns this destination
                    const int c = wg_id % num_qp_ch;         // ... and EXCLUSIVELY this qp c
                    if (dst_rdma == my_rdma_rank) return;    // self: no RDMA
                    auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                    const int dst_pe = dst_rdma * num_nvl_ranks + nvl_rank;
                    auto* dst_region = rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes;
                    // Clamp against a stale/garbage count read (the pristine version sized the
                    // put with the compile-time-constant rdma_count_offset and so could never
                    // go OOB; a count-sized put MUST guard the length or a bad read under fused
                    // timing sends L=count*row_bytes past the region -> ccs wedge).
                    const int count_raw = reinterpret_cast<int*>(region + rdma_count_offset)[0];
                    const int count = (count_raw < 0) ? 0 : (count_raw > max_rdma_tokens ? max_rdma_tokens : count_raw);
                    // Payload put of the data region, matching dispatch F4a2. The count is
                    // carried by FC5c's -count-1 AMO flag, so the count field itself is NOT
                    // transmitted. Default path is the warp-collective NBI put with force_db:
                    // a BLOCKING ishmem_putmem leaves the qp's nic_wq_commit lagging nic_wq_cnt
                    // (no gate reconciles it in an ISOLATED, repeated combine), so FC5c's AMO
                    // never egresses and the receiver poll spins to the cap. The forced-doorbell
                    // warp-put advances the commit watermark in-call, so each combine is
                    // self-contained. DEEP_EP_INTERNODE_BLOCKING_PUT keeps the old blocking put
                    // as an A/B fallback (only the c==0 WG ships the whole region).
                    if (faithful_blocking_put) {
                        if (c == 0 && local_id == 0) {
                            // Non-per-QP on purpose: this iSHMEM build has no single-work-item
                            // per-QP put (ishmemx_putmem_nbi_qp is declared in ishmemx.h but
                            // undefined in libishmem.a). Safe only because the launcher pins
                            // num_qp_ch = 1 in blocking-put mode; see combine_nvl_rdma.
                            if (faithful_nbi_mode) {
                                ishmem_putmem_nbi(dst_region, region, rdma_count_offset, dst_pe);
                            } else {
                                ishmem_putmem(dst_region, region, rdma_count_offset, dst_pe);
                            }
                        }
                        if (faithful_nbi_mode) {
                            // NBI path: release fence to make data NIC-visible before the kernel
                            // boundary separates us from FC5c's quiet_qp + AMO.
                            auto group = item.get_group();
                            sycl::group_barrier(group);
                            if (local_id == 0) {
                                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                                lsc_fence_sysrel();
                            }
                            sycl::group_barrier(group);
                        }
                        return;
                    }
                    // LL-faithful COUNT-SIZED multi-QP put. Ship only the count-sized VALID
                    // slices (x + 3 small metadata) to ~halve combine bytes, while keeping
                    // EVERY qp busy with a forced doorbell + a device-scope release fence
                    // (mirror LLCombineSendKernel), so FC5c's per-qp quiet+AMO never spins on
                    // an un-committed qp (the P4 empty-channel hang). This WG drives chunk c on
                    // its own qp=c.
                    // Device-scope release: make the gathered bytes NIC-visible (HBM/L2 via
                    // PCIe P2P) before the doorbells (== internode_ll.cpp F2).
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                    // Stripe the count-sized x payload [0,count*row_bytes) across qps. A min
                    // length (num_qp_ch*16) keeps every qp's commit watermark advanced even
                    // when count==0 (receiver ignores the extra bytes: it reads only `count`
                    // rows).
                    const size_t x_valid = static_cast<size_t>(count) * row_bytes;
                    const size_t L = x_valid > 0 ? x_valid : static_cast<size_t>(num_qp_ch) * 16;
                    const size_t s = internode_qp_chunk_start(L, c, num_qp_ch);
                    const size_t e = internode_qp_chunk_start(L, c + 1, num_qp_ch);
                    if (e > s) {
                        ishmemx_putmem_nbi_subgroup(dst_region + s, region + s, e - s, dst_pe,
                                                static_cast<unsigned>(c), true, sg,
                                                /*force_db=*/faithful_force_db);
                    }
                    // Channel 0's WG also ships the 3 small metadata slices (wt/recv_pos/
                    // src_nvl) on qp 0; flag[0] (posted by FC5c after quiet(qp0)) therefore
                    // guards both chunk-0 and the metadata. Receiver waits for ALL flags =>
                    // x + metadata all landed before it reads.
                    if (c == 0 && count > 0) {
                        const size_t w_len = static_cast<size_t>(count) * num_topk * sizeof(float);
                        const size_t rp_len = static_cast<size_t>(count) * sizeof(int);
                        const size_t sn_len = static_cast<size_t>(count) * sizeof(int);
                        ishmemx_putmem_nbi_subgroup(dst_region + rdma_topk_wt_offset,
                                                region + rdma_topk_wt_offset, w_len, dst_pe,
                                                0u, true, sg, /*force_db=*/faithful_force_db);
                        ishmemx_putmem_nbi_subgroup(dst_region + rdma_recv_pos_offset,
                                                region + rdma_recv_pos_offset, rp_len, dst_pe,
                                                0u, true, sg, /*force_db=*/faithful_force_db);
                        ishmemx_putmem_nbi_subgroup(dst_region + rdma_src_nvl_offset,
                                                region + rdma_src_nvl_offset, sn_len, dst_pe,
                                                0u, true, sg, /*force_db=*/faithful_force_db);
                    }
                    sycl::group_barrier(sg);
                });
        });

        // ---- FC5c: kernel-boundary 64-bit AMO count flag (mirror dispatch F4b).
        // The kernel boundary after FC5b guarantees the blocking puts landed; the
        // quiet_qp flushes the QP, then the RC-ordered AMO posts -count-1 to the
        // receiver's recv-region flag. Posted for EVERY dst_rdma != self.
        queue.submit([&](sycl::handler& cgh) {
            // GAP#8: TRUE CONCURRENT-WG per-channel quiet/AMO. One WG per (dst_rdma, c) so
            // each qp is quiesced + flagged by EXACTLY its owning WG (mirrors dispatch F4b
            // and internode_ll.cpp) -> no shared-qp doorbell contention.
            const int flag_wgs = num_rdma_ranks * num_qp_ch;
            cgh.parallel_for<FaithfulCombineRdmaFlagKernel<dtype_t>>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(flag_wgs) * kIshmemWGSize),
                                  sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if (local_id != 0) return;
                    const int wg_id = static_cast<int>(item.get_group_linear_id());
                    const int dst_rdma = wg_id / num_qp_ch;
                    const int c = wg_id % num_qp_ch;
                    if (dst_rdma == my_rdma_rank) return;  // local node: no RDMA
                    auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                    const int count = reinterpret_cast<int*>(region + rdma_count_offset)[0];  // stashed by FC5b
                    const int dst_pe = dst_rdma * num_nvl_ranks + nvl_rank;
                    auto* dst_flag = reinterpret_cast<long*>(
                        rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes + rdma_flag_offset);
                    // Channel c on its OWN work-group/qp: quiet qp c (flush FC5b2's chunk-c
                    // doorbell) then post the tail AMO on qp c (every flag carries -count-1).
                    // Receiver waits for all num_qp_ch flags.
                    ishmemx_fence_qp(dst_pe, static_cast<unsigned>(c));
                    lsc_fence_sysrel();
                    ishmemx_long_atomic_add_qp(dst_flag + c, static_cast<long>(-count - 1), dst_pe,
                                               static_cast<unsigned>(c));
                    if (faithful_post_amo_quiet) ishmemx_fence_qp(dst_pe, static_cast<unsigned>(c));
                });
        });

        // ---- FC6: leader forwards RDMA-received tokens to local NVL peers,
        // polling the 64-bit AMO flag instead of the count sentinel. The forward
        // body is IDENTICAL to the serial FwdWrite (byte-identical output). Self
        // RDMA rank has no flag (count written locally by FC5b) -> read directly.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulCombineFwdWriteKernel<dtype_t>>(
                sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)),
                [=](sycl::nd_item<1> item) {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    // Per-GPU RDMA: this GPU only received tokens whose src_nvl == nvl_rank
                    // (the sender filtered by plane), so every forwarded token targets the
                    // local peer nvl_rank == self. Each GPU is therefore the SOLE writer of
                    // its own fwd buffer -> touch only own plane (peer==nvl_rank) to avoid
                    // cross-GPU fwd_count collisions.
                    const int fwd_peer_lo = nvl_rank;
                    const int fwd_peer_hi = nvl_rank + 1;
                    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                    lsc_fence_sysacq();
                    int peer_offsets[NUM_MAX_NVL_PEERS] = {0};
                    if (local_id == 0) {
                        for (int peer = fwd_peer_lo; peer < fwd_peer_hi; ++peer) {
                            auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                            auto* fwd_base = peer_buf + fwd_base_offset;
                            auto* fwd_counts = reinterpret_cast<int*>(fwd_base + fwd_layout.fwd_count_offset);
                            for (int r = 0; r < num_rdma_ranks; ++r) {
                                fwd_counts[r] = 0;
                            }
                        }
                    }
                    sycl::group_barrier(group);

                    for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                        int before[NUM_MAX_NVL_PEERS] = {0};
                        for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                            before[peer] = peer_offsets[peer];
                        }
                        auto* region = rdma_base + static_cast<size_t>(src_rdma) * rdma_region_bytes;
                        auto* rdma_x = reinterpret_cast<dtype_t*>(region);
                        auto* rdma_wt = reinterpret_cast<float*>(region + rdma_topk_wt_offset);
                        auto* rdma_recv_pos = reinterpret_cast<int*>(region + rdma_recv_pos_offset);
                        auto* rdma_src_nvl = reinterpret_cast<int*>(region + rdma_src_nvl_offset);
                        auto* rdma_count = reinterpret_cast<int*>(region + rdma_count_offset);
                        auto* rdma_flag = reinterpret_cast<long*>(region + rdma_flag_offset);

                        // Determine the arrived token count. Self region: written locally by
                        // FC5b (no flag). Remote: poll the 64-bit AMO flag (0 == not arrived;
                        // -count-1 once the NIC-delivered AMO lands after the data). All WIs
                        // read the same flag (uniform), so no broadcast is needed.
                        int count;
                        if (src_rdma == my_rdma_rank) {
                            count = uc_load(rdma_count);
                        } else {
                            // Multi-QP: wait for ALL num_qp_ch channel flags (=> every byte chunk
                            // placed) before reading. Every flag carries -count-1; use flag[0] for
                            // the count. Uniform across WIs (matches the existing no-broadcast read).
                            long raw0 = 0;
                            for (int c = 0; c < num_qp_ch; ++c) {
                                long raw = internode_read_flag64(rdma_flag + c, rdma_flag_lsc_mode);
                                for (uint64_t spins = 0; raw == 0 && spins < rdma_poll_cap; ++spins) {
                                    if ((spins & 0x3FFF) == 0) {
                                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                                    }
                                    visa_spin_hint();
                                    raw = internode_read_flag64(rdma_flag + c, rdma_flag_lsc_mode);
                                }
                                if (c == 0) raw0 = raw;
                            }
                            count = (raw0 == 0) ? 0 : static_cast<int>(-raw0 - 1);
                        }
                        // Clamp against a stale/garbage flag (no-op for valid counts) so a
                        // runaway loop / OOB write cannot wedge the GPU.
                        if (count < 0) count = 0;
                        if (count > max_rdma_tokens) count = max_rdma_tokens;
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();  // invalidate GPU cache -> subsequent cached int4
                                             // loads observe the NIC-delivered RDMA payload

                        for (int i = 0; i < count; ++i) {
                            // Cached read (coherent after lsc_fence_sysacq, same as the payload
                            // copy below): the former per-token uncached uc_load ran redundantly
                            // on every work-item.
                            const int target_nvl = rdma_src_nvl[i];
                            if (target_nvl < 0 || target_nvl >= num_nvl_ranks) {
                                continue;
                            }
                            auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[target_nvl]);
                            auto* fwd_base = peer_buf + fwd_base_offset;
                            auto* fwd_x = reinterpret_cast<dtype_t*>(fwd_base + fwd_layout.fwd_x_offset);
                            auto* fwd_meta = reinterpret_cast<SourceMeta*>(fwd_base + fwd_layout.fwd_meta_offset);
                            auto* fwd_topk = reinterpret_cast<float*>(fwd_base + fwd_layout.fwd_topk_weights_offset);
                            if (peer_offsets[target_nvl] >= max_fwd_tokens) {
                                continue;
                            }
                            const int dst_idx = peer_offsets[target_nvl]++;
                            // int4-vectorized copy of the row (post-invalidate cached loads),
                            // replacing a per-element uncached uc_load loop.
                            faithful_coop_copy(reinterpret_cast<uint8_t*>(&fwd_x[dst_idx * hidden]),
                                               reinterpret_cast<const uint8_t*>(&rdma_x[i * hidden]),
                                               static_cast<size_t>(hidden) * sizeof(dtype_t),
                                               local_id, kComputeWGSize);
                            if (combined_topk_weights != nullptr) {
                                for (int k = local_id; k < num_topk; k += kComputeWGSize) {
                                    fwd_topk[dst_idx * num_topk + k] = uc_load(&rdma_wt[i * num_topk + k]);
                                }
                            }
                            if (local_id == 0) {
                                fwd_meta[dst_idx] = SourceMeta{src_rdma, uc_load(&rdma_recv_pos[i]), target_nvl};
                            }
                        }

                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        sycl::group_barrier(group);
                        if (local_id == 0) {
                            for (int peer = fwd_peer_lo; peer < fwd_peer_hi; ++peer) {
                                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                                auto* fwd_base = peer_buf + fwd_base_offset;
                                auto* fwd_counts = reinterpret_cast<int*>(fwd_base + fwd_layout.fwd_count_offset);
                                fwd_counts[src_rdma] = peer_offsets[peer] - before[peer];
                            }
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        }
                    }
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    sycl::group_barrier(group);
                });
        });
    }

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombineFwdBarrierKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 2, num_nvl_ranks, item); });
    });

    queue.submit([&](sycl::handler& cgh) {
        const size_t rd_groups = static_cast<size_t>(std::max(num_combined_tokens, 1));
        cgh.parallel_for<CombinedCombineReduceKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(rd_groups * kComputeWGSize), sycl::range<1>(kComputeWGSize)), [=](sycl::nd_item<1> item) {
            const int local_id = static_cast<int>(item.get_local_id(0));
            const int ct = static_cast<int>(item.get_group(0));
            if (ct >= num_combined_tokens) return;
            // Acquire fence: order all subsequent reads of this rank's
            // forwarded NVL buffer (written by the leader across PCIe) after
            // the leader's release fence (CUDA ld_acquire_sys_global equivalent).
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
            auto* fwd_base = my_buf + fwd_base_offset;
            auto* fwd_x = reinterpret_cast<dtype_t*>(fwd_base + fwd_layout.fwd_x_offset);
            auto* fwd_meta = reinterpret_cast<SourceMeta*>(fwd_base + fwd_layout.fwd_meta_offset);
            auto* fwd_topk = reinterpret_cast<float*>(fwd_base + fwd_layout.fwd_topk_weights_offset);
            auto* fwd_counts = reinterpret_cast<int*>(fwd_base + fwd_layout.fwd_count_offset);
            const int fwd_capacity = num_rdma_ranks * num_combined_tokens;
            // One work-group per output (combined) token: this work-group owns
            // dst[ct], so the accumulation is race-free. This is the per-ct
            // inverse of the original per-idx scan and is byte-identical to it:
            // the original maps each forwarded token to the FIRST combined token
            // whose rdma_head matches its recv position. Here we reconstruct
            // that same rule -- ct only reduces a source's contribution when it
            // is the first combined token matching that source's recv position
            // (the `is_first` guard), so ties resolve exactly as the original.
            int fwd_offset = 0;
            for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                int count = fwd_counts[src_rdma];
                if (count < 0) {
                    count = 0;
                }
                if (fwd_offset + count > fwd_capacity) {
                    count = fwd_capacity - fwd_offset;
                    if (count < 0) {
                        count = 0;
                    }
                }
                const int recv_pos_target = combined_rdma_head[ct * num_rdma_ranks + src_rdma];
                bool is_first = true;
                for (int ctp = 0; ctp < ct; ++ctp) {
                    if (combined_rdma_head[ctp * num_rdma_ranks + src_rdma] == recv_pos_target) {
                        is_first = false;
                        break;
                    }
                }
                if (is_first) {
                    for (int i = 0; i < count; ++i) {
                        const int idx = fwd_offset + i;
                        if (fwd_meta[idx].is_token_in_nvl_rank_bits != recv_pos_target) {
                            continue;
                        }
                        for (int h = local_id; h < hidden; h += kComputeWGSize) {
                            float value = static_cast<float>(dst[ct * hidden + h]);
                            value += static_cast<float>(fwd_x[idx * hidden + h]);
                            dst[ct * hidden + h] = static_cast<dtype_t>(value);
                        }
                        if (combined_topk_weights != nullptr) {
                            for (int k = local_id; k < num_topk; k += kComputeWGSize) {
                                combined_topk_weights[ct * num_topk + k] += fwd_topk[idx * num_topk + k];
                            }
                        }
                    }
                }
                fwd_offset += count;
            }

            for (int h = local_id; h < hidden; h += kComputeWGSize) {
                float value = static_cast<float>(dst[ct * hidden + h]);
                if (b0 != nullptr) {
                    value += static_cast<float>(b0[ct * hidden + h]);
                }
                if (b1 != nullptr) {
                    value += static_cast<float>(b1[ct * hidden + h]);
                }
                dst[ct * hidden + h] = static_cast<dtype_t>(value);
            }
        });
    });
    queue.wait();
#else
    TORCH_CHECK(false, "combine_nvl_rdma requires DEEP_EP_ENABLE_ISHMEM");
#endif
}

}  // namespace internode
}  // namespace deep_ep
