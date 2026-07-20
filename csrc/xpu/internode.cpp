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

// ---- Faithful (CUDA-parity) dispatch transport, gated by DEEP_EP_INTERNODE_FAITHFUL ----
// Collapses the serial single_task Pack/Stage/RdmaSend/RdmaPut into fewer,
// work-group/sub-group-parallel kernels and swaps the scalar blocking put +
// ishmemx_barrier_all_work_group data-path sync for ishmemx_putmem_nbi_warp +
// ishmemx_quiet_qp + 64-bit ishmemx_long_atomic_add_qp flags (LL pattern).
class FaithfulDispatchPackStageKernel;
class FaithfulDispatchPackBarrierKernel;
class FaithfulDispatchRdmaBarrierKernel;
class FaithfulDispatchRdmaSendKernel;
class FaithfulDispatchRdmaFlagKernel;
class FaithfulDispatchFwdWriteKernel;
class FaithfulDispatchFwdBarrierKernel;

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
//     ishmemx_quiet_qp(dst_pe,0) AFTER the AMO in F4b to reap the AMO's QP
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
// targeted ishmemx_quiet_qp(dst_pe,0) per dst_rdma to drain any residue left on
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
// serial combine use, instead of ishmemx_putmem_nbi_warp. RATIONALE: the warp
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
                   // (ishmemx_putmem_nbi_warp). This is SAFE now that COMBINE is also
                   // faithful: every qp0 payload put is followed by a reconciling AMO
                   // (dispatch F4b / combine FC5c ishmemx_long_atomic_add_qp, whose
                   // Step-6 CAS-max raises nic_wq_commit->nic_wq_cnt), so the warp-put's
                   // ordered-commit gate never wedges. HW-validated on both nodes.
                   // Set =1 to force the gate-free blocking ishmem_putmem (robust
                   // fallback; needed only if an interleaved path leaves a commit gap
                   // unreconciled, e.g. the serial combine when FAITHFUL is OFF).
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

// DEEP_EP_INTERNODE_PER_GPU_RDMA (default ON): CUDA-faithful per-GPU RDMA transport.
// The legacy path funnels ALL local GPUs' inter-node tokens through the nvl_rank==0
// "leader" GPU, which alone issues RDMA (dst_pe = dst_rdma*num_nvl_ranks, the remote
// node's leader) — using only 1 NIC per node. CUDA internode.cu has NO leader: every
// GPU runs kRDMASender and issues its OWN RDMA to the SAME-nvl-slot peer on the remote
// node (dst_pe = dst_rdma*num_nvl_ranks + nvl_rank), and the NVL all-to-all happens on
// the RECEIVE side (kRDMAAndNVLForwarder pushes received tokens to the correct local
// NVL peer, keyed by the source nvl-plane so concurrent forwarders don't collide).
// When ON, each nvl_rank: (a) zeroes its own receive flags, (b) compacts ONLY its own
// packed tokens per dst_rdma and RDMA-puts on its own NIC, (c) posts its own count
// flag, and (d) polls its own received stream and forwards to local peers into a
// per-source-plane fwd slice. This uses all num_nvl_ranks NICs per node. The
// intra-node NVL gather (Pack + Assemble peer reads) is unchanged. Set =0 for the
// legacy single-leader path (A/B fallback, byte-identical output).
inline bool internode_per_gpu_rdma() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_PER_GPU_RDMA");
    if (env != nullptr && env[0] != '\0') return std::atoi(env) != 0;
    return true;
}

// DEEP_EP_INTERNODE_QP_CHANNELS (default 1): number of RC QPs per PE to stripe the
// RDMA payload put across (CUDA-faithful multi-QP; qp_id = channel, mirrors
// internode.cu:818/835 where the put + tail AMO ride qp_id == channel_id). The
// contiguous payload put [0,rdma_count_offset) is split into this many byte chunks,
// chunk c issued by sub-group c on qp=c (concurrent NIC send queues), then per-c
// ishmemx_quiet_qp(dst,c) + ishmemx_long_atomic_add_qp(flag[c],...,c). RC in-order
// keeps flag[c] after chunk c on the SAME qp. Must be a power of 2 and <=
// kComputeWGSize/32 (one sub-group per channel). The launcher MUST set
// ISHMEM_IBGDA_QPS_PER_PE to the SAME value so iSHMEM provisions that many QPs.
// Default 1 preserves the original single-QP behavior byte-for-byte.
inline int internode_num_qp_channels() {
    const char* env = std::getenv("DEEP_EP_INTERNODE_QP_CHANNELS");
    int c = 1;
    if (env != nullptr && env[0] != '\0') c = std::atoi(env);
    if (c < 1) c = 1;
    int p = 1;
    while (p * 2 <= c) p *= 2;  // clamp down to power of 2
    c = p;
    const int max_ch = kComputeWGSize / 32;  // one sub-group drives one channel/QP
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
        auto* d16 = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst);
        auto* s16 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src);
        for (size_t j = static_cast<size_t>(lane); j < n16; j += static_cast<size_t>(lanes))
            d16[j] = s16[j];
        done = n16 << 4;
    }
    for (size_t b = done + static_cast<size_t>(lane); b < n; b += static_cast<size_t>(lanes))
        dst[b] = src[b];
}

inline void faithful_coop_zero(uint8_t* dst, size_t n, int lane, int lanes) {
    size_t done = 0;
    if ((reinterpret_cast<uintptr_t>(dst) & 0xF) == 0) {
        const size_t n16 = n >> 4;
        auto* d16 = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst);
        const sycl::vec<uint32_t, 4> z{0u, 0u, 0u, 0u};
        for (size_t j = static_cast<size_t>(lane); j < n16; j += static_cast<size_t>(lanes))
            d16[j] = z;
        done = n16 << 4;
    }
    for (size_t b = done + static_cast<size_t>(lane); b < n; b += static_cast<size_t>(lanes))
        dst[b] = 0;
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

// Faithful combine (DEEP_EP_INTERNODE_FAITHFUL): the serial RdmaPush kernel is
// split into an init barrier + gate-free blocking payload put + kernel-boundary
// 64-bit AMO count flag, and FwdWrite polls that flag (mirrors dispatch F4a0/
// F4a2/F4b/F6). Init/Pack/PackBarrier/RdmaSend/FwdBarrier/Reduce are reused
// verbatim so the reduce math (and thus the output) stays byte-identical.
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

void dispatch(void* recv_x,
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
              int rank,
              int num_ranks,
              sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(recv_x != nullptr && x != nullptr, "XPU internode dispatch requires input and output tensors");
    TORCH_CHECK(rdma_buffer_ptr != nullptr, "XPU internode dispatch requires a symmetric iSHMEM RDMA buffer");
    TORCH_CHECK(is_token_in_rank != nullptr, "XPU internode dispatch requires is_token_in_rank");
    TORCH_CHECK(num_channels > 0, "XPU internode dispatch requires a positive channel count");
    TORCH_CHECK(num_max_rdma_chunked_send_tokens > 0, "XPU internode dispatch requires a positive RDMA send window");
    TORCH_CHECK(num_max_rdma_chunked_recv_tokens > 0, "XPU internode dispatch requires a positive RDMA queue window");
    TORCH_CHECK(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens,
                "XPU internode dispatch send window must fit in the RDMA receive queue window");
    TORCH_CHECK(element_size > 0, "XPU internode dispatch element size must be positive");
    TORCH_CHECK((x_scales == nullptr) == (recv_x_scales == nullptr), "XPU internode dispatch scale tensors must be paired");
    TORCH_CHECK((num_scales == 0) == (x_scales == nullptr), "XPU internode dispatch scale shape mismatch");
    TORCH_CHECK((topk_idx == nullptr) == (topk_weights == nullptr), "XPU internode dispatch top-k inputs must be paired");
    TORCH_CHECK((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr), "XPU internode dispatch top-k outputs must be paired");
    TORCH_CHECK((num_topk == 0) == (topk_idx == nullptr), "XPU internode dispatch top-k shape mismatch");
    TORCH_CHECK(static_cast<size_t>(num_recv_tokens) >= static_cast<size_t>(num_tokens) * num_ranks,
                "XPU internode dispatch correctness path requires num_worst_tokens >= num_tokens * num_ranks");

    const size_t row_bytes = static_cast<size_t>(hidden) * element_size;
    auto* src = static_cast<const uint8_t*>(x);
    auto* dst = static_cast<uint8_t*>(recv_x);
    auto* meta = static_cast<SourceMeta*>(recv_src_meta);
    auto* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);
    size_t offset = 0;
    auto* rdma_count_matrix = reinterpret_cast<int*>(rdma_base + offset);
    offset += static_cast<size_t>(num_ranks) * num_ranks * sizeof(int);
    offset = align_offset(offset, alignof(int));
    auto* rdma_channel_count_matrix = reinterpret_cast<int*>(rdma_base + offset);
    offset += static_cast<size_t>(num_ranks) * num_ranks * num_channels * sizeof(int);
    offset = align_offset(offset, 128);
    auto* rdma_x = rdma_base + offset;
    offset += static_cast<size_t>(num_recv_tokens) * row_bytes;
    offset = align_offset(offset, alignof(SourceMeta));
    auto* rdma_meta = reinterpret_cast<SourceMeta*>(rdma_base + offset);
    offset += static_cast<size_t>(num_recv_tokens) * sizeof(SourceMeta);
    offset = align_offset(offset, alignof(topk_idx_t));
    auto* rdma_topk_idx = reinterpret_cast<topk_idx_t*>(rdma_base + offset);
    offset += static_cast<size_t>(num_recv_tokens) * num_topk * sizeof(topk_idx_t);
    offset = align_offset(offset, alignof(float));
    auto* rdma_topk_weights = reinterpret_cast<float*>(rdma_base + offset);
    offset += static_cast<size_t>(num_recv_tokens) * num_topk * sizeof(float);
    offset = align_offset(offset, alignof(float));
    auto* rdma_x_scales = reinterpret_cast<float*>(rdma_base + offset);
    offset += static_cast<size_t>(num_recv_tokens) * num_scales * sizeof(float);
    offset = align_offset(offset, 128);
    auto* rdma_send_x = rdma_base + offset;
    const size_t num_send_slots = static_cast<size_t>(num_recv_tokens) * num_ranks;
    offset += num_send_slots * row_bytes;
    offset = align_offset(offset, alignof(SourceMeta));
    auto* rdma_send_meta = reinterpret_cast<SourceMeta*>(rdma_base + offset);
    offset += num_send_slots * sizeof(SourceMeta);
    offset = align_offset(offset, alignof(topk_idx_t));
    auto* rdma_send_topk_idx = reinterpret_cast<topk_idx_t*>(rdma_base + offset);
    offset += num_send_slots * num_topk * sizeof(topk_idx_t);
    offset = align_offset(offset, alignof(float));
    auto* rdma_send_topk_weights = reinterpret_cast<float*>(rdma_base + offset);
    offset += num_send_slots * num_topk * sizeof(float);
    offset = align_offset(offset, alignof(float));
    auto* rdma_send_x_scales = reinterpret_cast<float*>(rdma_base + offset);
    offset += num_send_slots * num_scales * sizeof(float);
    offset = align_offset(offset, alignof(int));
    auto* rdma_send_dst_token = reinterpret_cast<int*>(rdma_base + offset);
    offset += num_send_slots * sizeof(int);
    const int queue_window = num_max_rdma_chunked_recv_tokens;
    const int queue_stride = std::max(queue_window, 64);
    const size_t num_queue_slots = static_cast<size_t>(num_ranks) * num_channels * queue_stride;
    offset = align_offset(offset, 128);
    auto* rdma_queue_x = rdma_base + offset;
    offset += num_queue_slots * row_bytes;
    offset = align_offset(offset, alignof(SourceMeta));
    auto* rdma_queue_meta = reinterpret_cast<SourceMeta*>(rdma_base + offset);
    offset += num_queue_slots * sizeof(SourceMeta);
    offset = align_offset(offset, alignof(topk_idx_t));
    auto* rdma_queue_topk_idx = reinterpret_cast<topk_idx_t*>(rdma_base + offset);
    offset += num_queue_slots * num_topk * sizeof(topk_idx_t);
    offset = align_offset(offset, alignof(float));
    auto* rdma_queue_topk_weights = reinterpret_cast<float*>(rdma_base + offset);
    offset += num_queue_slots * num_topk * sizeof(float);
    offset = align_offset(offset, alignof(float));
    auto* rdma_queue_x_scales = reinterpret_cast<float*>(rdma_base + offset);
    offset += num_queue_slots * num_scales * sizeof(float);
    offset = align_offset(offset, alignof(int));
    auto* rdma_queue_dst_token = reinterpret_cast<int*>(rdma_base + offset);
    offset += num_queue_slots * sizeof(int);
    offset = align_offset(offset, alignof(int));
    auto* rdma_queue_head = reinterpret_cast<int*>(rdma_base + offset);
    const size_t num_queue_pairs = static_cast<size_t>(num_ranks) * num_channels;
    offset += num_queue_pairs * sizeof(int);
    auto* rdma_queue_tail = reinterpret_cast<int*>(rdma_base + offset);
    const size_t total_recv_elements = static_cast<size_t>(num_recv_tokens) * row_bytes;
    const size_t total_send_elements = num_send_slots * row_bytes;
    const size_t total_queue_elements = num_queue_slots * row_bytes;
    const size_t total_topk_elements = static_cast<size_t>(num_recv_tokens) * num_topk;
    const size_t total_send_topk_elements = num_send_slots * num_topk;
    const size_t total_queue_topk_elements = num_queue_slots * num_topk;
    const size_t total_scale_elements = static_cast<size_t>(num_recv_tokens) * num_scales;
    const size_t total_send_scale_elements = num_send_slots * num_scales;
    const size_t total_queue_scale_elements = num_queue_slots * num_scales;
    const size_t total_count_elements = static_cast<size_t>(num_ranks) * num_ranks;
    const size_t total_channel_count_elements = total_count_elements * num_channels;
    const size_t init_range = std::max({total_recv_elements,
                                        total_send_elements,
                                        total_queue_elements,
                                        static_cast<size_t>(num_recv_tokens),
                                        num_send_slots,
                                        num_queue_slots,
                                        total_topk_elements,
                                        total_send_topk_elements,
                                        total_queue_topk_elements,
                                        total_scale_elements,
                                        total_send_scale_elements,
                                        total_queue_scale_elements,
                                        total_channel_count_elements,
                                        static_cast<size_t>(num_tokens) * num_ranks,
                                        num_queue_pairs,
                                        static_cast<size_t>(num_recv_tokens) * NUM_MAX_NVL_PEERS});

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<DispatchInitKernel>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
            const size_t linear = static_cast<size_t>(id[0]);
            if (linear < total_recv_elements) {
                dst[linear] = 0;
                rdma_x[linear] = 0;
            }
            if (linear < total_send_elements) {
                rdma_send_x[linear] = 0;
            }
            if (linear < total_queue_elements) {
                rdma_queue_x[linear] = 0;
            }
            if (linear < static_cast<size_t>(num_recv_tokens)) {
                if (meta != nullptr) {
                    meta[linear].src_rdma_rank = -1;
                    meta[linear].is_token_in_nvl_rank_bits = -1;
                    meta[linear].src_nvl_rank = -1;
                }
                rdma_meta[linear].src_rdma_rank = -1;
                rdma_meta[linear].is_token_in_nvl_rank_bits = -1;
                rdma_meta[linear].src_nvl_rank = -1;
            }
            if (linear < num_send_slots) {
                rdma_send_meta[linear].src_rdma_rank = -1;
                rdma_send_meta[linear].is_token_in_nvl_rank_bits = -1;
                rdma_send_meta[linear].src_nvl_rank = -1;
                rdma_send_dst_token[linear] = -1;
            }
            if (linear < num_queue_slots) {
                rdma_queue_meta[linear].src_rdma_rank = -1;
                rdma_queue_meta[linear].is_token_in_nvl_rank_bits = -1;
                rdma_queue_meta[linear].src_nvl_rank = -1;
                rdma_queue_dst_token[linear] = -1;
            }
            if (linear < num_queue_pairs) {
                rdma_queue_head[linear] = 0;
                rdma_queue_tail[linear] = 0;
            }
            if (linear < total_topk_elements) {
                if (recv_topk_idx != nullptr) {
                    recv_topk_idx[linear] = -1;
                    recv_topk_weights[linear] = 0.0f;
                }
                rdma_topk_idx[linear] = -1;
                rdma_topk_weights[linear] = 0.0f;
            }
            if (linear < total_send_topk_elements) {
                rdma_send_topk_idx[linear] = -1;
                rdma_send_topk_weights[linear] = 0.0f;
            }
            if (linear < total_queue_topk_elements) {
                rdma_queue_topk_idx[linear] = -1;
                rdma_queue_topk_weights[linear] = 0.0f;
            }
            if (linear < total_scale_elements) {
                if (recv_x_scales != nullptr) {
                    recv_x_scales[linear] = 0.0f;
                }
                rdma_x_scales[linear] = 0.0f;
            }
            if (linear < total_send_scale_elements) {
                rdma_send_x_scales[linear] = 0.0f;
            }
            if (linear < total_queue_scale_elements) {
                rdma_queue_x_scales[linear] = 0.0f;
            }
            if (linear < total_count_elements) {
                rdma_count_matrix[linear] = 0;
            }
            if (linear < total_channel_count_elements) {
                rdma_channel_count_matrix[linear] = 0;
            }
            if (send_rdma_head != nullptr && linear < static_cast<size_t>(num_tokens) * num_ranks) {
                send_rdma_head[linear] = -1;
            }
            if (send_nvl_head != nullptr && linear < static_cast<size_t>(num_recv_tokens) * NUM_MAX_NVL_PEERS) {
                send_nvl_head[linear] = -1;
            }
        });
    });
    queue.wait();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<DispatchCountExchangeKernel>(
            sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
                auto group = item.get_group();
                const int local_id = static_cast<int>(item.get_local_id(0));
                if (local_id == 0) {
                    auto* local_count_row = rdma_count_matrix + rank * num_ranks;
                    for (int i = 0; i < num_ranks; ++i) {
                        if (num_tokens_per_rank != nullptr) {
                            local_count_row[i] = num_tokens_per_rank[i];
                        } else {
                            int count = 0;
                            for (int token = 0; token < num_tokens; ++token) {
                                count += is_token_in_rank[token * num_ranks + i] ? 1 : 0;
                            }
                            local_count_row[i] = count;
                        }
                    }
                }
                sycl::group_barrier(group);
                auto* local_count_row = rdma_count_matrix + rank * num_ranks;
                for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                    auto* remote_count_row = rdma_count_matrix + rank * num_ranks;
                    if (dst_rank != rank && local_id == 0) {
                        // CUDA-style scalar API: leader posts a single IBGDA WQE.
                        // For inter-PE puts this matches the original NVSHMEM
                        // _warp pattern (one thread doorbells, rest do useful
                        // work).  Functionally equivalent to the work_group
                        // variant on the IBGDA path.
                        ishmem_putmem_nbi(
                            remote_count_row, local_count_row, static_cast<size_t>(num_ranks) * sizeof(int), dst_rank);
                    }
                    sycl::group_barrier(group);
                }

                if (local_id == 0) {
                    auto* local_channel_row = rdma_channel_count_matrix + static_cast<size_t>(rank) * num_ranks * num_channels;
                    for (int dst = 0; dst < num_ranks; ++dst) {
                        int cumulative = 0;
                        for (int channel = 0; channel < num_channels; ++channel) {
                            const int start = (static_cast<int64_t>(num_tokens) * channel) / num_channels;
                            const int end = (static_cast<int64_t>(num_tokens) * (channel + 1)) / num_channels;
                            int count = 0;
                            for (int token = start; token < end; ++token) {
                                count += is_token_in_rank[token * num_ranks + dst] ? 1 : 0;
                            }
                            cumulative += count;
                            local_channel_row[dst * num_channels + channel] = cumulative;
                        }
                    }
                }
                sycl::group_barrier(group);
                auto* local_channel_row = rdma_channel_count_matrix + static_cast<size_t>(rank) * num_ranks * num_channels;
                for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                    auto* remote_channel_row = rdma_channel_count_matrix + static_cast<size_t>(rank) * num_ranks * num_channels;
                    if (dst_rank != rank && local_id == 0) {
                        ishmem_putmem_nbi(remote_channel_row,
                                          local_channel_row,
                                          static_cast<size_t>(num_ranks) * num_channels * sizeof(int),
                                          dst_rank);
                    }
                    sycl::group_barrier(group);
                }
                if (local_id == 0) {
                    ishmem_quiet();
                }
                sycl::group_barrier(group);
            });
    });
    queue.wait();

    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<DispatchOffsetComputeKernel>([=]() {
            int recv_sum = 0;
            for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                const int src_count = rdma_count_matrix[src_rank * num_ranks + rank];
                for (int channel = 0; channel < num_channels; ++channel) {
                    const int channel_end =
                        rdma_channel_count_matrix[(static_cast<size_t>(src_rank) * num_ranks + rank) * num_channels + channel];
                    const int channel_start = channel == 0
                        ? 0
                        : rdma_channel_count_matrix[(static_cast<size_t>(src_rank) * num_ranks + rank) * num_channels + channel - 1];
                    if (recv_gbl_channel_prefix_matrix != nullptr) {
                        recv_gbl_channel_prefix_matrix[src_rank * num_channels + channel] = recv_sum + channel_start;
                    }
                    if (recv_rdma_channel_prefix_matrix != nullptr) {
                        recv_rdma_channel_prefix_matrix[src_rank * num_channels + channel] = channel_end;
                    }
                }
                recv_sum += src_count;
                recv_gbl_rank_prefix_sum[src_rank] = recv_sum;
            }
            if (recv_rdma_rank_prefix_sum != nullptr) {
                int rdma_sum = 0;
                for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                    rdma_sum += rdma_count_matrix[src_rank * num_ranks + rank];
                    recv_rdma_rank_prefix_sum[src_rank] = rdma_sum;
                }
            }
            if (rdma_channel_prefix_matrix != nullptr) {
                for (int dst = 0; dst < num_ranks; ++dst) {
                    for (int channel = 0; channel < num_channels; ++channel) {
                        const int channel_end =
                            rdma_channel_count_matrix[(static_cast<size_t>(rank) * num_ranks + dst) * num_channels + channel];
                        rdma_channel_prefix_matrix[dst * num_channels + channel] = channel_end;
                        gbl_channel_prefix_matrix[dst * num_channels + channel] = channel_end;
                    }
                }
            }
        });
    });
    queue.wait();
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<DispatchPackKernel>(sycl::range<1>(static_cast<size_t>(num_ranks) * num_tokens), [=](sycl::id<1> id) {
            const int linear = static_cast<int>(id[0]);
            const int dst_rank = linear / num_tokens;
            const int token = linear - dst_rank * num_tokens;
            if (!is_token_in_rank[token * num_ranks + dst_rank]) {
                return;
            }

            int dst_token = 0;
            for (int src_rank = 0; src_rank < rank; ++src_rank) {
                dst_token += rdma_count_matrix[src_rank * num_ranks + dst_rank];
            }
            for (int prior_token = 0; prior_token < token; ++prior_token) {
                dst_token += is_token_in_rank[prior_token * num_ranks + dst_rank] ? 1 : 0;
            }
            if (dst_token >= num_recv_tokens) {
                return;
            }
            if (send_rdma_head != nullptr) {
                send_rdma_head[token * num_ranks + dst_rank] = dst_token;
            }

            const size_t send_slot = static_cast<size_t>(dst_rank) * num_recv_tokens + dst_token;
            auto* send_x = rdma_send_x + send_slot * row_bytes;
            auto* src_x = src + static_cast<size_t>(token) * row_bytes;
            for (size_t h = 0; h < row_bytes; ++h) {
                send_x[h] = src_x[h];
            }
            rdma_send_meta[send_slot] = SourceMeta{rank, token, -1};
            rdma_send_dst_token[send_slot] = dst_token;
            if (x_scales != nullptr) {
                auto* send_scales = rdma_send_x_scales + send_slot * num_scales;
                auto* src_scales = x_scales + static_cast<size_t>(token) * num_scales;
                for (int k = 0; k < num_scales; ++k) {
                    send_scales[k] = src_scales[k];
                }
            }
            if (topk_idx != nullptr) {
                auto* send_topk_idx = rdma_send_topk_idx + send_slot * num_topk;
                auto* send_topk_weights = rdma_send_topk_weights + send_slot * num_topk;
                auto* src_topk_idx = topk_idx + static_cast<size_t>(token) * num_topk;
                auto* src_topk_weights = topk_weights + static_cast<size_t>(token) * num_topk;
                for (int k = 0; k < num_topk; ++k) {
                    send_topk_idx[k] = src_topk_idx[k];
                    send_topk_weights[k] = src_topk_weights[k];
                }
            }
        });
    });
    queue.wait();
    for (int channel = 0; channel < num_channels; ++channel) {
        const int channel_start = (static_cast<int64_t>(num_tokens) * channel) / num_channels;
        const int channel_end = (static_cast<int64_t>(num_tokens) * (channel + 1)) / num_channels;
        const int channel_tokens = channel_end - channel_start;
        for (int window_offset = 0; window_offset < channel_tokens; window_offset += queue_window) {
            if (window_offset > 0) {
                queue.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<DispatchQueueResetKernel>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
                        const size_t linear = static_cast<size_t>(id[0]);
                        if (linear < total_queue_elements) {
                            rdma_queue_x[linear] = 0;
                        }
                        if (linear < num_queue_slots) {
                            rdma_queue_meta[linear].src_rdma_rank = -1;
                            rdma_queue_meta[linear].is_token_in_nvl_rank_bits = -1;
                            rdma_queue_meta[linear].src_nvl_rank = -1;
                            rdma_queue_dst_token[linear] = -1;
                        }
                        if (linear < total_queue_topk_elements) {
                            rdma_queue_topk_idx[linear] = -1;
                            rdma_queue_topk_weights[linear] = 0.0f;
                        }
                        if (linear < total_queue_scale_elements) {
                            rdma_queue_x_scales[linear] = 0.0f;
                        }
                        if (linear < num_queue_pairs) {
                            rdma_queue_head[linear] = 0;
                            rdma_queue_tail[linear] = 0;
                        }
                    });
                });
                queue.wait();
            }

            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<DispatchPayloadKernel>(
                    sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
                        auto group = item.get_group();
                        const int local_id = static_cast<int>(item.get_local_id(0));
                        const int local_size = static_cast<int>(item.get_local_range(0));
                        for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                            int window_count = 0;
                            for (int token = channel_start; token < channel_end; ++token) {
                                if (!is_token_in_rank[token * num_ranks + dst_rank]) {
                                    continue;
                                }
                                int queue_ordinal = 0;
                                for (int prior_token = channel_start; prior_token < token; ++prior_token) {
                                    queue_ordinal += is_token_in_rank[prior_token * num_ranks + dst_rank] ? 1 : 0;
                                }
                                const int queue_offset = queue_ordinal - window_offset;
                                if (queue_offset < 0 || queue_offset >= queue_window) {
                                    continue;
                                }

                                int dst_token = 0;
                                for (int src_rank = 0; src_rank < rank; ++src_rank) {
                                    dst_token += rdma_count_matrix[src_rank * num_ranks + dst_rank];
                                }
                                for (int prior_token = 0; prior_token < token; ++prior_token) {
                                    dst_token += is_token_in_rank[prior_token * num_ranks + dst_rank] ? 1 : 0;
                                }
                                if (dst_token >= num_recv_tokens) {
                                    continue;
                                }

                                const size_t queue_pair = static_cast<size_t>(rank) * num_channels + channel;
                                const size_t queue_slot = queue_pair * queue_stride + queue_offset;
                                auto* dst_x = rdma_queue_x + queue_slot * row_bytes;
                                const size_t send_slot = static_cast<size_t>(dst_rank) * num_recv_tokens + dst_token;
                                auto* src_x = rdma_send_x + send_slot * row_bytes;
                                auto* dst_meta = rdma_queue_meta + queue_slot;
                                auto* src_meta = rdma_send_meta + send_slot;
                                auto* dst_topk_idx = rdma_queue_topk_idx + queue_slot * num_topk;
                                auto* dst_topk_weights = rdma_queue_topk_weights + queue_slot * num_topk;
                                auto* src_topk_idx = rdma_send_topk_idx + send_slot * num_topk;
                                auto* src_topk_weights = rdma_send_topk_weights + send_slot * num_topk;
                                auto* dst_scales = rdma_queue_x_scales + queue_slot * num_scales;
                                auto* src_scales = rdma_send_x_scales + send_slot * num_scales;
                                auto* src_dst_token = rdma_send_dst_token + send_slot;
                                if (dst_rank == rank) {
                                    for (size_t h = local_id; h < row_bytes; h += local_size) {
                                        dst_x[h] = src_x[h];
                                    }
                                    for (int k = local_id; k < num_topk; k += local_size) {
                                        dst_topk_idx[k] = src_topk_idx[k];
                                        dst_topk_weights[k] = src_topk_weights[k];
                                    }
                                    for (int k = local_id; k < num_scales; k += local_size) {
                                        dst_scales[k] = src_scales[k];
                                    }
                                    if (local_id == 0) {
                                        rdma_queue_dst_token[queue_slot] = dst_token;
                                        *dst_meta = *src_meta;
                                    }
                                } else {
                                    // CUDA-style scalar API: leader-only post
                                    // matches the original NVSHMEM
                                    // nvshmemi_ibgda_put_nbi_warp pattern.
                                    // For cross-node peers this is identical
                                    // to the work_group variant (which falls
                                    // through to leader-only IBGDA post on the
                                    // RDMA path).  For intra-node SHM peers
                                    // below 32KB cutover, this serializes the
                                    // copy on a single WI; on this stack the
                                    // 2-rank validation is cross-node only.
                                    if (local_id == 0) {
                                        ishmem_putmem_nbi(dst_x, src_x, row_bytes, dst_rank);
                                        if (num_topk > 0) {
                                            ishmem_putmem_nbi(dst_topk_idx,
                                                              src_topk_idx,
                                                              static_cast<size_t>(num_topk) * sizeof(topk_idx_t),
                                                              dst_rank);
                                            ishmem_putmem_nbi(dst_topk_weights,
                                                              src_topk_weights,
                                                              static_cast<size_t>(num_topk) * sizeof(float),
                                                              dst_rank);
                                        }
                                        if (num_scales > 0) {
                                            ishmem_putmem_nbi(
                                                dst_scales, src_scales, static_cast<size_t>(num_scales) * sizeof(float), dst_rank);
                                        }
                                        ishmem_putmem_nbi(
                                            rdma_queue_dst_token + queue_slot, src_dst_token, sizeof(int), dst_rank);
                                        ishmem_putmem_nbi(dst_meta, src_meta, sizeof(SourceMeta), dst_rank);
                                    }
                                }
                                sycl::group_barrier(group);
                                if (local_id == 0) {
                                    ++window_count;
                                }
                            }
                            // Broadcast window_count from work-item 0 to all items via shared local var
                            sycl::group_barrier(group);
                            int wg_window_count = sycl::group_broadcast(group, window_count, 0);
                            if (wg_window_count > 0) {
                                const size_t queue_pair = static_cast<size_t>(rank) * num_channels + channel;
                                if (dst_rank == rank) {
                                    if (local_id == 0) {
                                        rdma_queue_tail[queue_pair] += wg_window_count;
                                    }
                                } else {
                                    // Stage tail in rdma_queue_head per (rank, channel) — unique per WG.
                                    if (local_id == 0) {
                                        rdma_queue_head[queue_pair] = wg_window_count;
                                    }
                                    sycl::group_barrier(group);
                                    if (local_id == 0) {
                                        ishmem_putmem_nbi(
                                            rdma_queue_tail + queue_pair, rdma_queue_head + queue_pair, sizeof(int), dst_rank);
                                    }
                                }
                            }
                            sycl::group_barrier(group);
                            if (local_id == 0) {
                                ishmem_quiet();
                            }
                            sycl::group_barrier(group);
                        }  // end for dst_rank
                    });
            });
            queue.wait();

            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<DispatchQueueCopyKernel>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
                    const size_t linear = static_cast<size_t>(id[0]);
                    if (linear < total_queue_elements) {
                        const size_t queue_slot = linear / row_bytes;
                        const size_t byte_idx = linear - queue_slot * row_bytes;
                        const int dst_token = rdma_queue_dst_token[queue_slot];
                        const size_t queue_pair = queue_slot / queue_stride;
                        const int queue_offset = static_cast<int>(queue_slot - queue_pair * queue_stride);
                        if (queue_offset < rdma_queue_tail[queue_pair] && rdma_queue_meta[queue_slot].src_rdma_rank >= 0 &&
                            dst_token >= 0 && dst_token < num_recv_tokens) {
                            rdma_x[static_cast<size_t>(dst_token) * row_bytes + byte_idx] = rdma_queue_x[linear];
                        }
                    }
                    if (linear < num_queue_slots) {
                        const int dst_token = rdma_queue_dst_token[linear];
                        const size_t queue_pair = linear / queue_stride;
                        const int queue_offset = static_cast<int>(linear - queue_pair * queue_stride);
                        if (queue_offset < rdma_queue_tail[queue_pair] && rdma_queue_meta[linear].src_rdma_rank >= 0 && dst_token >= 0 &&
                            dst_token < num_recv_tokens) {
                            rdma_meta[dst_token] = rdma_queue_meta[linear];
                        }
                    }
                    if (linear < total_queue_topk_elements) {
                        const size_t queue_slot = linear / num_topk;
                        const size_t topk_idx = linear - queue_slot * num_topk;
                        const int dst_token = rdma_queue_dst_token[queue_slot];
                        const size_t queue_pair = queue_slot / queue_stride;
                        const int queue_offset = static_cast<int>(queue_slot - queue_pair * queue_stride);
                        if (queue_offset < rdma_queue_tail[queue_pair] && rdma_queue_meta[queue_slot].src_rdma_rank >= 0 &&
                            dst_token >= 0 && dst_token < num_recv_tokens) {
                            rdma_topk_idx[static_cast<size_t>(dst_token) * num_topk + topk_idx] = rdma_queue_topk_idx[linear];
                            rdma_topk_weights[static_cast<size_t>(dst_token) * num_topk + topk_idx] = rdma_queue_topk_weights[linear];
                        }
                    }
                    if (linear < total_queue_scale_elements) {
                        const size_t queue_slot = linear / num_scales;
                        const size_t scale_idx = linear - queue_slot * num_scales;
                        const int dst_token = rdma_queue_dst_token[queue_slot];
                        const size_t queue_pair = queue_slot / queue_stride;
                        const int queue_offset = static_cast<int>(queue_slot - queue_pair * queue_stride);
                        if (queue_offset < rdma_queue_tail[queue_pair] && rdma_queue_meta[queue_slot].src_rdma_rank >= 0 &&
                            dst_token >= 0 && dst_token < num_recv_tokens) {
                            rdma_x_scales[static_cast<size_t>(dst_token) * num_scales + scale_idx] = rdma_queue_x_scales[linear];
                        }
                    }
                    if (linear < num_queue_pairs) {
                        rdma_queue_head[linear] = rdma_queue_tail[linear];
                    }
                });
            });
            queue.wait();
        }
    }
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<DispatchCopyKernel>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
            const size_t linear = static_cast<size_t>(id[0]);
            if (linear < total_recv_elements) {
                dst[linear] = rdma_x[linear];
            }
            if (linear < static_cast<size_t>(num_recv_tokens)) {
                if (meta != nullptr) {
                    meta[linear] = rdma_meta[linear];
                }
            }
            if (linear < total_topk_elements && recv_topk_idx != nullptr) {
                recv_topk_idx[linear] = rdma_topk_idx[linear];
                recv_topk_weights[linear] = rdma_topk_weights[linear];
            }
            if (linear < total_scale_elements && recv_x_scales != nullptr) {
                recv_x_scales[linear] = rdma_x_scales[linear];
            }
        });
    });
#else
    TORCH_CHECK(false, "XPU internode dispatch requires iSHMEM support");
#endif

    (void)send_rdma_head;
    (void)send_nvl_head;
    (void)recv_rdma_channel_prefix_matrix;
    (void)recv_gbl_channel_prefix_matrix;
    (void)rdma_channel_prefix_matrix;
    (void)recv_rdma_rank_prefix_sum;
    (void)gbl_channel_prefix_matrix;
}

template <typename dtype_t>
void launch_combine_copy(void* combined_x,
                         float* combined_topk_weights,
                         void* rdma_buffer_ptr,
                         const void* x,
                         const float* topk_weights,
                         const void* bias_0,
                         const void* bias_1,
                         const void* src_meta,
                         int num_tokens,
                         int num_combined_tokens,
                         int hidden,
                         int num_topk,
                         int num_max_rdma_chunked_send_tokens,
                         int num_max_rdma_chunked_recv_tokens,
                         int rank,
                         int num_ranks,
                         sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(rdma_buffer_ptr != nullptr, "XPU internode combine requires a symmetric iSHMEM RDMA buffer");
    TORCH_CHECK(num_max_rdma_chunked_send_tokens > 0, "XPU internode combine requires a positive RDMA send window");
    TORCH_CHECK(num_max_rdma_chunked_recv_tokens > 0, "XPU internode combine requires a positive RDMA queue window");
    TORCH_CHECK(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens,
                "XPU internode combine send window must fit in the RDMA receive queue window");
    const int queue_window = num_max_rdma_chunked_recv_tokens;
    const int queue_stride = std::max(queue_window, 64);
    auto dst = static_cast<dtype_t*>(combined_x);
    auto src = static_cast<const dtype_t*>(x);
    auto b0 = static_cast<const dtype_t*>(bias_0);
    auto b1 = static_cast<const dtype_t*>(bias_1);
    auto meta = static_cast<const SourceMeta*>(src_meta);
    auto* rdma_recv_x = static_cast<dtype_t*>(rdma_buffer_ptr);
    auto* rdma_recv_topk_weights = reinterpret_cast<float*>(rdma_recv_x + static_cast<int64_t>(num_combined_tokens) * hidden);
    auto* rdma_send_x = reinterpret_cast<dtype_t*>(rdma_recv_topk_weights + static_cast<int64_t>(num_combined_tokens) * num_topk);
    auto* rdma_send_topk_weights = reinterpret_cast<float*>(rdma_send_x + static_cast<int64_t>(num_tokens) * hidden);
    auto* rdma_queue_x = reinterpret_cast<dtype_t*>(rdma_send_topk_weights + static_cast<int64_t>(num_tokens) * num_topk);
    auto* rdma_queue_topk_weights = reinterpret_cast<float*>(rdma_queue_x + static_cast<int64_t>(num_ranks) * queue_stride * hidden);
    auto* rdma_queue_head = reinterpret_cast<int*>(rdma_queue_topk_weights + static_cast<int64_t>(num_ranks) * queue_stride * num_topk);
    auto* rdma_queue_tail = rdma_queue_head + num_ranks;
    const int64_t total_combined = static_cast<int64_t>(num_combined_tokens) * hidden;
    const int64_t total_recv = static_cast<int64_t>(num_tokens) * hidden;
    const int64_t total_combined_topk = static_cast<int64_t>(num_combined_tokens) * num_topk;
    const int64_t total_recv_topk = static_cast<int64_t>(num_tokens) * num_topk;
    const int64_t total_queue = static_cast<int64_t>(num_ranks) * queue_stride * hidden;
    const int64_t total_queue_topk = static_cast<int64_t>(num_ranks) * queue_stride * num_topk;
    const int64_t init_range =
        std::max({total_combined, total_combined_topk, total_queue, total_queue_topk, static_cast<int64_t>(num_ranks)});

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombineInitKernel<dtype_t>>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
            const int64_t i = static_cast<int64_t>(id[0]);
            if (i < total_combined) {
                dst[i] = dtype_t{};
                rdma_recv_x[i] = dtype_t{};
            }
            if (i < total_queue) {
                rdma_queue_x[i] = dtype_t{};
            }
            if (i < total_combined_topk) {
                combined_topk_weights[i] = 0.0f;
                rdma_recv_topk_weights[i] = 0.0f;
            }
            if (i < total_queue_topk) {
                rdma_queue_topk_weights[i] = 0.0f;
            }
            if (i < num_ranks) {
                rdma_queue_head[i] = 0;
                rdma_queue_tail[i] = 0;
            }
        });
    });
    queue.wait();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinePackKernel<dtype_t>>(sycl::range<1>(total_recv), [=](sycl::id<1> id) {
            const int64_t i = static_cast<int64_t>(id[0]);
            rdma_send_x[i] = src[i];
        });
    });
    if (total_recv_topk > 0) {
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<CombinePayloadQuietKernel<dtype_t>>(sycl::range<1>(total_recv_topk), [=](sycl::id<1> id) {
                const int64_t i = static_cast<int64_t>(id[0]);
                rdma_send_topk_weights[i] = topk_weights[i];
            });
        });
    }
    queue.wait();

    for (int window_offset = 0; window_offset < num_combined_tokens; window_offset += queue_window) {
        const int window_tokens = std::min(queue_window, num_combined_tokens - window_offset);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<CombineQueueResetKernel<dtype_t>>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
                const int64_t i = static_cast<int64_t>(id[0]);
                if (i < total_queue) {
                    rdma_queue_x[i] = dtype_t{};
                }
                if (i < total_queue_topk) {
                    rdma_queue_topk_weights[i] = 0.0f;
                }
                if (i < num_ranks) {
                    rdma_queue_head[i] = 0;
                    rdma_queue_tail[i] = 0;
                }
            });
        });
        queue.wait();

        queue.submit([&](sycl::handler& cgh) {
            constexpr int kQueueGroupSize = 32;
            cgh.parallel_for<CombinePayloadKernel<dtype_t>>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_ranks) * kQueueGroupSize), sycl::range<1>(kQueueGroupSize)),
                [=](sycl::nd_item<1> item) {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    const int local_size = static_cast<int>(item.get_local_range(0));
                    const int src_rank = static_cast<int>(item.get_group(0));
                    int window_count = 0;
                    for (int recv_token = 0; recv_token < num_tokens; ++recv_token) {
                        const SourceMeta token_meta = meta[recv_token];
                        const int src_token = token_meta.is_token_in_nvl_rank_bits;
                        if (token_meta.src_rdma_rank != src_rank || src_token < window_offset ||
                            src_token >= window_offset + window_tokens) {
                            continue;
                        }

                        const int queue_token = src_token - window_offset;
                        auto* remote_dst = rdma_queue_x + (static_cast<int64_t>(rank) * queue_stride + queue_token) * hidden;
                        auto* local_src = rdma_send_x + static_cast<int64_t>(recv_token) * hidden;
                        auto* remote_topk_weights =
                            rdma_queue_topk_weights + (static_cast<int64_t>(rank) * queue_stride + queue_token) * num_topk;
                        auto* local_topk_weights = rdma_send_topk_weights + static_cast<int64_t>(recv_token) * num_topk;
                        if (src_rank == rank) {
                            for (int h = local_id; h < hidden; h += local_size) {
                                remote_dst[h] = local_src[h];
                            }
                            for (int k = local_id; k < num_topk; k += local_size) {
                                remote_topk_weights[k] = local_topk_weights[k];
                            }
                        } else {
                            if (local_id == 0) {
                                ishmem_putmem_nbi(
                                    remote_dst, local_src, static_cast<size_t>(hidden) * sizeof(dtype_t), src_rank);
                                if (num_topk > 0) {
                                    ishmem_putmem_nbi(remote_topk_weights,
                                                      local_topk_weights,
                                                      static_cast<size_t>(num_topk) * sizeof(float),
                                                      src_rank);
                                }
                            }
                        }
                        sycl::group_barrier(group);
                        if (local_id == 0) {
                            ++window_count;
                        }
                    }
                    sycl::group_barrier(group);
                    int wg_window_count = sycl::group_broadcast(group, window_count, 0);
                    if (wg_window_count > 0) {
                        if (src_rank == rank) {
                            if (local_id == 0) {
                                rdma_queue_tail[rank] += wg_window_count;
                            }
                        } else {
                            // Stage in rdma_queue_head; unique per work-group since
                            // only one (src_rank, channel) WG runs at a time.
                            if (local_id == 0) {
                                rdma_queue_head[rank] = wg_window_count;
                            }
                            sycl::group_barrier(group);
                            if (local_id == 0) {
                                ishmem_putmem_nbi(rdma_queue_tail + rank, rdma_queue_head + rank, sizeof(int), src_rank);
                                ishmem_quiet();
                            }
                            sycl::group_barrier(group);
                        }
                    }
                    sycl::group_barrier(group);
                });
        });
        queue.wait();

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<CombineQueueCopyKernel<dtype_t>>(
                sycl::range<1>(std::max(static_cast<int64_t>(window_tokens) * hidden, static_cast<int64_t>(window_tokens) * num_topk)),
                [=](sycl::id<1> id) {
                    const int64_t i = static_cast<int64_t>(id[0]);
                    if (i < static_cast<int64_t>(window_tokens) * hidden) {
                        const int token_offset = static_cast<int>(i / hidden);
                        const int h = static_cast<int>(i - static_cast<int64_t>(token_offset) * hidden);
                        dtype_t value{};
                        for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                            value += rdma_queue_x[(static_cast<int64_t>(src_rank) * queue_stride + token_offset) * hidden + h];
                        }
                        rdma_recv_x[static_cast<int64_t>(window_offset + token_offset) * hidden + h] = value;
                    }
                    if (i < static_cast<int64_t>(window_tokens) * num_topk) {
                        const int token_offset = static_cast<int>(i / num_topk);
                        const int k = static_cast<int>(i - static_cast<int64_t>(token_offset) * num_topk);
                        float value = 0.0f;
                        for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                            value += rdma_queue_topk_weights[(static_cast<int64_t>(src_rank) * queue_stride + token_offset) * num_topk + k];
                        }
                        rdma_recv_topk_weights[static_cast<int64_t>(window_offset + token_offset) * num_topk + k] = value;
                    }
                    if (i < num_ranks) {
                        rdma_queue_head[i] = rdma_queue_tail[i];
                    }
                });
        });
        queue.wait();
    }

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombineBiasKernel<dtype_t>>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
            const int64_t i = static_cast<int64_t>(id[0]);
            if (i < total_combined) {
                dtype_t value = rdma_recv_x[i];
                if (b0 != nullptr) {
                    value += b0[i];
                }
                if (b1 != nullptr) {
                    value += b1[i];
                }
                dst[i] = value;
            }
            if (i < total_combined_topk) {
                combined_topk_weights[i] = rdma_recv_topk_weights[i];
            }
        });
    });
#else
    TORCH_CHECK(false, "XPU internode combine requires iSHMEM support");
#endif
}

void combine(DataType type,
             void* combined_x,
             float* combined_topk_weights,
             const bool*,
             void* rdma_buffer_ptr,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int*,
             const int*,
             const void* src_meta,
             const int*,
             const int*,
             const int*,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             int rank,
             int num_ranks,
             sycl::queue& queue) {
    if (type == DataType::kBFloat16) {
        launch_combine_copy<sycl::ext::oneapi::bfloat16>(combined_x,
                                                         combined_topk_weights,
                                                         rdma_buffer_ptr,
                                                         x,
                                                         topk_weights,
                                                         bias_0,
                                                         bias_1,
                                                         src_meta,
                                                         num_tokens,
                                                         num_combined_tokens,
                                                         hidden,
                                                         num_topk,
                                                         num_max_rdma_chunked_send_tokens,
                                                         num_max_rdma_chunked_recv_tokens,
                                                         rank,
                                                         num_ranks,
                                                         queue);
    } else {
        launch_combine_copy<int32_t>(combined_x,
                                     combined_topk_weights,
                                     rdma_buffer_ptr,
                                     x,
                                     topk_weights,
                                     bias_0,
                                     bias_1,
                                     src_meta,
                                     num_tokens,
                                     num_combined_tokens,
                                     hidden,
                                     num_topk,
                                     num_max_rdma_chunked_send_tokens,
                                     num_max_rdma_chunked_recv_tokens,
                                     rank,
                                     num_ranks,
                                     queue);
    }
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
                       sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(recv_x != nullptr && x != nullptr, "dispatch_nvl_rdma requires input and output tensors");
    TORCH_CHECK(rdma_buffer_ptr != nullptr, "dispatch_nvl_rdma requires a symmetric iSHMEM RDMA buffer");
    TORCH_CHECK(is_token_in_rank != nullptr, "dispatch_nvl_rdma requires is_token_in_rank");

    // When num_nvl_ranks == 1, each rank is its own RDMA rank with no NVL peers.
    // This degenerates to the pure RDMA-only dispatch path.
    if (num_nvl_ranks <= 1) {
        internode::dispatch(recv_x,
                            recv_x_scales,
                            recv_topk_idx,
                            recv_topk_weights,
                            recv_src_meta,
                            rdma_buffer_ptr,
                            x,
                            x_scales,
                            topk_idx,
                            topk_weights,
                            send_rdma_head,
                            send_nvl_head,
                            recv_rdma_channel_prefix_matrix,
                            recv_gbl_channel_prefix_matrix,
                            rdma_channel_prefix_matrix,
                            recv_rdma_rank_prefix_sum,
                            gbl_channel_prefix_matrix,
                            recv_gbl_rank_prefix_sum,
                            num_tokens_per_rank,
                            is_token_in_rank,
                            num_tokens,
                            num_recv_tokens,
                            hidden,
                            element_size,
                            num_topk,
                            num_scales,
                            num_channels,
                            num_max_rdma_chunked_send_tokens,
                            num_max_rdma_chunked_recv_tokens,
                            rank,
                            num_ranks,
                            queue);
        return;
    }

    TORCH_CHECK(buffer_ptrs_gpu != nullptr, "dispatch_nvl_rdma requires NVL buffer pointers");
    TORCH_CHECK(barrier_signal_ptrs_gpu != nullptr, "dispatch_nvl_rdma requires barrier signal pointers");
    TORCH_CHECK(num_nvl_ranks > 1 && num_nvl_ranks <= NUM_MAX_NVL_PEERS, "dispatch_nvl_rdma NVL peer count out of range");
    TORCH_CHECK(num_ranks % num_nvl_ranks == 0, "dispatch_nvl_rdma requires num_ranks divisible by num_nvl_ranks");

    const size_t row_bytes = static_cast<size_t>(hidden) * element_size;
    const int my_rdma_rank = rank / num_nvl_ranks;
    const int my_global_rank = my_rdma_rank * num_nvl_ranks + nvl_rank;
    const int num_rdma_ranks = num_ranks / num_nvl_ranks;
    TORCH_CHECK(num_rdma_ranks <= NUM_MAX_NVL_PEERS, "dispatch_nvl_rdma currently supports up to ", NUM_MAX_NVL_PEERS, " RDMA ranks");

    // CUDA-faithful per-GPU RDMA: every nvl_rank sends its own RDMA on its own NIC
    // (dst_pe = dst_rdma*num_nvl_ranks + nvl_rank) instead of funneling through the
    // nvl_rank==0 leader. The receive-side forward buffer then needs one disjoint slice
    // per source nvl-plane so concurrent forwarders don't collide (num_fwd_planes).
    const bool per_gpu_rdma = internode_per_gpu_rdma();
    const int num_fwd_planes = per_gpu_rdma ? num_nvl_ranks : 1;

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
    // OWN staging (same-GPU coherent). stage_stride mirrors the full NVL
    // buffer layout so RdmaSend can reuse layout.send_*_offset unchanged.
    const size_t stage_stride = align_offset(layout.total_bytes, 128);
    const size_t stage_base_offset = align_offset(fwd_base_offset + fwd_layout.total_bytes, 128);

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
    const int num_qp_ch = internode_num_qp_channels();
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

    static const bool kDbgDispatch = std::getenv("DEEP_EP_DBG_DISPATCH") != nullptr;
    auto ddbg_last = std::chrono::high_resolution_clock::now();
    auto ddbg_stage = [&](const char* name) {
        if (kDbgDispatch) {
            auto now = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - ddbg_last).count();
            ddbg_last = now;
            std::fprintf(stderr, "[dispatch rank=%d nvl=%d rdma=%d] stage done: %s (+%.2f ms)\n",
                         rank, nvl_rank, my_rdma_rank, name, ms);
            std::fflush(stderr);
        }
    };
    // In-order XPU stream => device executes kernels back-to-back; the per-stage
    // host queue.wait() is pure round-trip overhead (~2 ms each, ~8 stages =>
    // ~16 ms). Skip intermediate waits in production (DEEP_EP_INTERNODE_FUSE_WAITS,
    // default ON); force them when timing stages (kDbgDispatch) or when the fuse is
    // disabled for A/B. The FINAL wait before returning to Python stays unconditional.
    static const bool kFuseWaits = [] {
        const char* e = std::getenv("DEEP_EP_INTERNODE_FUSE_WAITS");
        return !(e != nullptr && e[0] != '\0' && std::atoi(e) == 0);
    }();
    const bool wait_each = kDbgDispatch || !kFuseWaits;
    ddbg_stage("0-entry");
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchInitKernel>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
            const size_t linear = static_cast<size_t>(id[0]);
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
        });
    });
    if (wait_each) queue.wait();
    ddbg_stage("1-Init");

    // ===================================================================
    // FAITHFUL (CUDA-parity) DISPATCH TRANSPORT (guarded, stage-1 collapse)
    // ===================================================================
    // When DEEP_EP_INTERNODE_FAITHFUL is set we run a collapsed transport that
    // (a) fuses Pack+Stage into one WG/sub-group-parallel kernel (drops the
    //     serial single_task and the separate Stage + StageBarrier launches),
    // (b) fuses RdmaSend compaction + RDMA put into one kernel that issues
    //     ishmemx_putmem_nbi_warp (warp-collective, deferred doorbell) +
    //     ishmemx_quiet_qp + a 64-bit ishmemx_long_atomic_add_qp count flag
    //     instead of the scalar blocking ishmem_putmem + double
    //     ishmemx_barrier_all_work_group, and
    // (c) gates the receiver on an acquire-fence + uc_load bounded-spin poll of
    //     that flag (device-scope everywhere except the single cross-PE
    //     post-zero rendezvous).
    // It writes the SAME peer send buffers + leader staging + fwd_* buffers the
    // shared Assemble/Head kernels (run after this branch) consume, so the
    // output handle semantics (recv_x ordering, prefix sums, absolute-position
    // send_nvl_head/send_rdma_head) are byte-for-byte identical to the fallback.
    static const bool kFaithful = [] {
        const char* env = std::getenv("DEEP_EP_INTERNODE_FAITHFUL");
        return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
    }();
    if (kFaithful) {
        const size_t stage_bytes = layout.total_bytes;
        const uint64_t rdma_poll_cap = internode_poll_cap();
        const int rdma_flag_lsc_mode = internode_flag_lsc_mode();
        const bool faithful_force_db = internode_force_db();          // F4a2 put doorbell
        const bool faithful_post_amo_quiet = internode_post_amo_quiet();  // F4b post-AMO quiet
        const bool faithful_entry_quiet = internode_entry_quiet();     // F4a2 entry drain qp0
        const bool faithful_blocking_put = internode_blocking_put();   // F4a2 blocking payload put
        // EXP3: host-visible per-rdma-rank debug slots (send: count/dst_pe/flag,
        // recv: raw/count/spins). Gated by DEEP_EP_DBG_DISPATCH; nullptr otherwise
        // so the captured-pointer writes compile out to a cheap null check.
        long* dbg_buf = nullptr;
        if (kDbgDispatch) {
            dbg_buf = sycl::malloc_shared<long>(static_cast<size_t>(num_rdma_ranks) * 6, queue);
            for (int i = 0; i < num_rdma_ranks * 6; ++i) dbg_buf[i] = -424242;  // sentinel = "kernel never wrote"
        }

        // ---- F-K1: fused Pack + Stage (WG-parallel payload; WI0 routing meta) ----
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchPackStageKernel>(
                sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)),
                [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
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

                    // Flush own send buffer, then STAGE into the leader's slot with a
                    // cross-device WRITE (producer-writes-into-peer; BMG P2P READ is
                    // unstable, WRITE is stable) so RdmaSendPut reads coherent same-GPU data.
                    sycl::group_barrier(group);
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    lsc_fence_sysrel();
                    auto* leader_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[0]);
                    auto* stage_dst = leader_buf + stage_base_offset + static_cast<size_t>(nvl_rank) * stage_stride;
                    // 16-byte-vectorized cross-device WRITE of ONLY the num_tokens-sized
                    // send buffer (layout is built with real num_tokens, never the
                    // worst-case num_recv_tokens). The prior byte-at-a-time IPC store of
                    // this whole region was the multi-second F2 stall.
                    faithful_coop_copy(stage_dst, my_buf, stage_bytes, local_id, kComputeWGSize);
                    sycl::group_barrier(group);
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    lsc_fence_sysrel();
                });
        });
        if (wait_each) queue.wait();
        ddbg_stage("F2-PackStage");

        // ---- F-K2: NVL barrier (device scope) so all peers staged before leader reads ----
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchPackBarrierKernel>(
                sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
                [=](sycl::nd_item<1> item) {
                    nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base, num_nvl_ranks, item);
                });
        });
        if (wait_each) queue.wait();
        ddbg_stage("F3-PackBarrier");

        // ---- F-K3a: RdmaSend — compaction + payload warp-put ONLY (deferred doorbell).
        // Mirrors internode_ll.cpp LLDispatchSendKernel: ishmemx_putmem_nbi_warp with
        // force_db=false and NO quiet/AMO. The kernel boundary after this (queue.wait)
        // is what guarantees the deferred doorbells are posted before F-K3b's quiet.
        // A fused put+quiet+AMO in ONE kernel does NOT egress the AMO on this BMG/IBGDA
        // stack (HW-confirmed: sent=1 but the remote flag never lands).
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchRdmaBarrierKernel>(
                sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));

                    // Zero MY receive-region flags so remote AMOs land onto 0. Under
                    // per-GPU RDMA every nvl_rank receives its own same-plane stream, so
                    // every PE zeroes its own flags (legacy: leader only).
                    if ((per_gpu_rdma || nvl_rank == 0) && local_id == 0) {
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
                    // Device-side fine-grained tracing (flushes DURING kernel execution,
                    // unlike the host-visible dbg_buf which is only printed after the
                    // queue.wait that this kernel may hang inside). Gated on dbg_buf!=null
                    // (i.e. DEEP_EP_DBG_DISPATCH). Leader/all-rank WI0 only.
                    if (dbg_buf && local_id == 0) {
                        sycl::ext::oneapi::experimental::printf(
                            "[F4a rank=%d nvl=%d] pre-barrier\n", rank, nvl_rank);
                    }
                    // The ONE necessary cross-PE rendezvous: every PE zeroed its flags
                    // before any AMO posts. Called by ALL PEs' work-groups (not leader-
                    // gated) so the collective does not hang. This replaces the two
                    // data-path barriers of the fallback with a single init barrier.
                    ishmemx_barrier_all_work_group(group);
                    sycl::group_barrier(group);
                    if (dbg_buf && local_id == 0) {
                        sycl::ext::oneapi::experimental::printf(
                            "[F4a rank=%d nvl=%d] post-barrier\n", rank, nvl_rank);
                    }
                });
        });
        if (wait_each) queue.wait();
        ddbg_stage("F4a0-Barrier");

        // ---- F-K3a2: Put — leader compaction + payload warp-put (force_db=false) ----
        // Split from the F4a0 barrier and the F4b flag AMO so any hang localizes to a
        // distinct HOST ddbg stage (device sycl printf is buffered and lost on SIGKILL;
        // only ddbg_stage host prints are reliably flushed). NO full ishmem_quiet() here
        // or in F4b: LL (internode_ll.cpp L940-942 / L1245-1247) uses ONLY a targeted
        // quiet_qp BEFORE the AMO and NEVER a full quiet, nor any quiet after the AMO.
        // A full ishmem_quiet() blocks forever if an AMO completion is not reaped by the
        // IBGDA layer (leader-only, data-independent, surfaces on a later dispatch).
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchRdmaSendKernel>(
                sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)),
                [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                    auto group = item.get_group();
                    auto sg = item.get_sub_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if (per_gpu_rdma || nvl_rank == 0) {
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();
                        // Optional entry drain of qp0 (A/B via DEEP_EP_INTERNODE_ENTRY_QUIET,
                        // default OFF): targeted quiet_qp per dst_pe to flush residue from the
                        // interleaved serial combine / prior faithful AMO before posting.
                        if (faithful_entry_quiet && local_id == 0) {
                            for (int dq = 0; dq < num_rdma_ranks; ++dq) {
                                if (dq == my_rdma_rank) continue;
                                ishmemx_quiet_qp(dq * num_nvl_ranks + (per_gpu_rdma ? nvl_rank : 0), 0u);
                            }
                        }
                        sycl::group_barrier(group);
                        auto* leader_self_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                        // Per-GPU RDMA: this GPU compacts ONLY its own packed tokens (single
                        // plane == nvl_rank, read from its own NVL buffer, no leader staging).
                        // Legacy leader: loop over all staged src_nvl planes.
                        const int src_nvl_lo = per_gpu_rdma ? nvl_rank : 0;
                        const int src_nvl_hi = per_gpu_rdma ? (nvl_rank + 1) : num_nvl_ranks;
                        for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                            if (dst_rdma == my_rdma_rank) continue;  // local node: no RDMA
                            if (dbg_buf && local_id == 0) {
                                sycl::ext::oneapi::experimental::printf(
                                    "[F4a rank=%d] dst=%d pre-compact\n", rank, dst_rdma);
                            }
                            auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                            auto* rdma_x = region;
                            auto* rdma_m = reinterpret_cast<SourceMeta*>(region + rdma_meta_offset);
                            auto* rdma_idx = reinterpret_cast<topk_idx_t*>(region + rdma_topk_idx_offset);
                            auto* rdma_wt = reinterpret_cast<float*>(region + rdma_topk_wt_offset);
                            auto* rdma_scales = reinterpret_cast<float*>(region + rdma_scales_offset);

                            // Compact all peers' tokens destined to dst_rdma into the send ring.
                            // All lanes run identical control flow so `count` is coherent; the
                            // payload copy is split across lanes, scalar meta by WI0.
                            int count = 0;
                            for (int src_nvl = src_nvl_lo; src_nvl < src_nvl_hi; ++src_nvl) {
                                auto* peer_buf = per_gpu_rdma
                                                     ? leader_self_buf  // own packed buffer (src_nvl == nvl_rank)
                                                     : (leader_self_buf + stage_base_offset +
                                                        static_cast<size_t>(src_nvl) * stage_stride);
                                auto* peer_x = peer_buf + layout.send_x_offset;
                                auto* peer_m = reinterpret_cast<SourceMeta*>(peer_buf + layout.send_meta_offset);
                                auto* peer_idx = reinterpret_cast<topk_idx_t*>(peer_buf + layout.send_topk_idx_offset);
                                auto* peer_wt = reinterpret_cast<float*>(peer_buf + layout.send_topk_weights_offset);
                                auto* peer_scales = reinterpret_cast<float*>(peer_buf + layout.send_x_scales_offset);
                                auto* peer_rdma_bits = reinterpret_cast<int*>(peer_buf + layout.send_rdma_dest_bits_offset);
                                auto* peer_is_in_rank = reinterpret_cast<bool*>(peer_buf + layout.send_is_token_in_rank_offset);
                                for (int t = 0; t < num_tokens; ++t) {
                                    if (((peer_rdma_bits[t] >> dst_rdma) & 1) == 0) continue;
                                    if (count >= num_recv_tokens) continue;  // capacity guard: send-ring holds
                                                                             // num_recv_tokens rows; never hit for a
                                                                             // legit count (<= num_nvl_ranks*num_tokens
                                                                             // == num_recv_tokens), prevents OOB otherwise
                                    auto* s_row = peer_x + static_cast<size_t>(t) * row_bytes;
                                    auto* d_row = rdma_x + static_cast<size_t>(count) * row_bytes;
                                    faithful_coop_copy(d_row, s_row, row_bytes, local_id, kComputeWGSize);
                                    if (local_id == 0) {
                                        int dst_nvl_bits = 0;
                                        for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                                            const int dst_rank = dst_rdma * num_nvl_ranks + dst_nvl;
                                            if (peer_is_in_rank[t * num_ranks + dst_rank]) dst_nvl_bits |= 1 << dst_nvl;
                                        }
                                        SourceMeta sm = peer_m[t];
                                        sm.is_token_in_nvl_rank_bits = dst_nvl_bits;
                                        rdma_m[count] = sm;
                                        if (topk_idx != nullptr) {
                                            for (int k = 0; k < num_topk; ++k) {
                                                rdma_idx[count * num_topk + k] = peer_idx[t * num_topk + k];
                                                rdma_wt[count * num_topk + k] = peer_wt[t * num_topk + k];
                                            }
                                        }
                                        if (x_scales != nullptr) {
                                            for (int s = 0; s < num_scales; ++s)
                                                rdma_scales[count * num_scales + s] = peer_scales[t * num_scales + s];
                                        }
                                    }
                                    ++count;
                                }
                            }

                            // Payload warp-put ONLY, deferred doorbell (force_db=false).
                            // quiet + AMO are deferred to F-K3b AFTER the kernel boundary.
                            sycl::group_barrier(group);
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                            lsc_fence_sysrel();
                            const int dst_pe = dst_rdma * num_nvl_ranks + (per_gpu_rdma ? nvl_rank : 0);
                            auto* dst_region = rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes;
                            if (dbg_buf && local_id == 0) {
                                sycl::ext::oneapi::experimental::printf(
                                    "[F4a rank=%d] dst=%d pre-put count=%d dst_pe=%d\n",
                                    rank, dst_rdma, count, dst_pe);
                            }
                            if (faithful_blocking_put) {
                                // Robust path (A/B via DEEP_EP_INTERNODE_BLOCKING_PUT): single-WI
                                // blocking ishmem_putmem, identical to the fallback/serial combine.
                                // No unbounded put_nbi_warp commit gate -> cannot wedge on a
                                // combine-induced nic_wq_commit gap (the observed F4a2 leader hang).
                                if (local_id == 0) {
                                    ishmem_putmem(dst_region, region, rdma_count_offset, dst_pe);
                                }
                                sycl::group_barrier(group);
                            } else {
                                // Default path: warp-collective NBI put (cache-hot, fast) with the
                                // deferred/forced doorbell selected by DEEP_EP_INTERNODE_FORCE_DB.
                                // CUDA-faithful multi-QP: split the contiguous payload
                                // [0,rdma_count_offset) into num_qp_ch 16B-aligned byte chunks,
                                // sub-group c drives chunk c on qp=c (concurrent NIC send queues,
                                // mirroring internode.cu qp_id==channel_id). Chunks tile [0,L)
                                // with no gap/overlap; each chunk's tail AMO (F-K3b) rides the
                                // SAME qp c so RC in-order keeps the flag after its payload.
                                const int sgid = sg.get_group_id()[0];
                                if (sgid < num_qp_ch) {
                                    const size_t L = rdma_count_offset;
                                    const size_t s = internode_qp_chunk_start(L, sgid, num_qp_ch);
                                    const size_t e = internode_qp_chunk_start(L, sgid + 1, num_qp_ch);
                                    if (e > s) {
                                        ishmemx_putmem_nbi_warp(dst_region + s, region + s, e - s, dst_pe,
                                                                static_cast<unsigned>(sgid), true, sg,
                                                                /*force_db=*/faithful_force_db);
                                    }
                                    sycl::group_barrier(sg);
                                }
                                sycl::group_barrier(group);
                            }
                            if (dbg_buf && local_id == 0) {
                                sycl::ext::oneapi::experimental::printf(
                                    "[F4a rank=%d] dst=%d post-put\n", rank, dst_rdma);
                            }
                            // Stash this dst_rdma's token count in the LOCAL send-ring count
                            // field so F-K3b can post the correct -count-1 flag. This slot is
                            // NOT transmitted (the put length is rdma_count_offset, i.e. up to
                            // but excluding this field); it lives in the leader's own iSHMEM
                            // buffer (same GPU) and survives the kernel boundary.
                            if (local_id == 0) {
                                reinterpret_cast<int*>(region + rdma_count_offset)[0] = count;
                            }
                            sycl::group_barrier(group);
                        }
                    }
                });
        });
        if (wait_each) queue.wait();
        ddbg_stage("F4a2-Put");

        // ---- F-K3b: RdmaFlag — quiet + 64-bit AMO ONLY, after the kernel boundary.
        // Mirrors internode_ll.cpp LLDispatchRecvKernel phase A (L938-942): quiet_qp
        // flushes the deferred doorbells posted in F-K3a, then the RC-ordered AMO on
        // the SAME qp lands after the payload.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchRdmaFlagKernel>(
                sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if ((!per_gpu_rdma && nvl_rank != 0) || local_id != 0) return;
                    for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                        if (dst_rdma == my_rdma_rank) continue;  // local node: no RDMA
                        auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                        const int count = reinterpret_cast<int*>(region + rdma_count_offset)[0];  // stashed by F-K3a
                        const int dst_pe = dst_rdma * num_nvl_ranks + (per_gpu_rdma ? nvl_rank : 0);
                        auto* dst_flag = reinterpret_cast<long*>(
                            rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes + rdma_flag_offset);
                        // CUDA-faithful multi-QP: per channel c, quiet qp c (flush F-K3a's
                        // deferred chunk-c doorbell) then post the tail AMO on qp c. Every
                        // flag carries -count-1 (total token count); RC in-order makes flag[c]
                        // land after chunk c on the same qp. The receiver waits for ALL C
                        // flags (=> all chunks placed) then reads [0,count).
                        for (int c = 0; c < num_qp_ch; ++c) {
                            ishmemx_quiet_qp(dst_pe, static_cast<unsigned>(c));
                            lsc_fence_sysrel();
                            ishmemx_long_atomic_add_qp(dst_flag + c, static_cast<long>(-count - 1), dst_pe,
                                                       static_cast<unsigned>(c));
                            if (faithful_post_amo_quiet) ishmemx_quiet_qp(dst_pe, static_cast<unsigned>(c));
                        }
                        if (dbg_buf) {  // EXP3 sender instrumentation
                            dbg_buf[dst_rdma * 3 + 0] = static_cast<long>(count);
                            dbg_buf[dst_rdma * 3 + 1] = static_cast<long>(dst_pe);
                            dbg_buf[dst_rdma * 3 + 2] = 1;
                        }
                    }
                });
        });
        if (wait_each) queue.wait();
        ddbg_stage("F4b-RdmaFlag");

        // ---- F-K4: leader forwards RDMA-received tokens to local NVL peers (flag poll) ----
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchFwdWriteKernel>(
                sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)), [=](sycl::nd_item<1> item) {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if (!per_gpu_rdma && nvl_rank != 0) return;
                    // Per-GPU RDMA: this GPU forwards its OWN received same-plane stream. It
                    // writes into each local peer's fwd buffer at the disjoint slice reserved
                    // for its source nvl-plane (== nvl_rank) so concurrent forwarders (one per
                    // plane) never collide. Legacy leader uses plane 0.
                    const int fwd_plane = per_gpu_rdma ? nvl_rank : 0;
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
                            uint64_t spins_total = 0;
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
                                spins_total += spins;
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
                            if (dbg_buf) {  // EXP3 receiver instrumentation
                                dbg_buf[num_rdma_ranks * 3 + src_rdma * 3 + 0] = raw0;
                                dbg_buf[num_rdma_ranks * 3 + src_rdma * 3 + 1] = static_cast<long>(count);
                                dbg_buf[num_rdma_ranks * 3 + src_rdma * 3 + 2] = static_cast<long>(spins_total);
                            }
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
        if (wait_each) queue.wait();
        ddbg_stage("F6-FwdWrite");
        if (kDbgDispatch && dbg_buf) {  // EXP3: dump sender/receiver flag signal
            for (int r = 0; r < num_rdma_ranks; ++r) {
                fprintf(stderr, "[F4 SEND] rank=%d my_rdma=%d -> dst_rdma=%d dst_pe=%ld count=%ld sent=%ld\n",
                        rank, my_rdma_rank, r, dbg_buf[r * 3 + 1], dbg_buf[r * 3 + 0], dbg_buf[r * 3 + 2]);
            }
            for (int r = 0; r < num_rdma_ranks; ++r) {
                fprintf(stderr, "[F6 RECV] rank=%d src_rdma=%d raw=%ld count=%ld spins=%ld\n",
                        rank, r, dbg_buf[num_rdma_ranks * 3 + r * 3 + 0],
                        dbg_buf[num_rdma_ranks * 3 + r * 3 + 1], dbg_buf[num_rdma_ranks * 3 + r * 3 + 2]);
            }
            fflush(stderr);
            sycl::free(dbg_buf, queue);
        }

        // ---- F-K5: NVL barrier before shared Assemble/Head consume fwd_* ----
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulDispatchFwdBarrierKernel>(
                sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
                [=](sycl::nd_item<1> item) {
                    nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 1, num_nvl_ranks, item);
                });
        });
        if (wait_each) queue.wait();
        ddbg_stage("F7-FwdBarrier");
    } else {  // ===== non-faithful fallback: existing 8-stage serial transport =====

    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<CombinedDispatchPackKernel>([=]() {
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

            // NvlBufferLayout above is constructed with num_channels=1 (combined-dispatch
            // single_task uses a single conceptual channel for routing metadata).
            // The kernel param num_channels may be larger (config-driven SM count); using
            // it in the channel_count loops below would write 4*num_channels ints into a
            // region only sized for 4 ints and overflow into send_x. Always use 1 here to
            // match the layout.
            constexpr int kPackNumChannels = 1;

            for (int d = 0; d < num_ranks; ++d) {
                my_counts[d] = 0;
                for (int c = 0; c < kPackNumChannels; ++c) {
                    my_channel_counts[d * kPackNumChannels + c] = 0;
                }
            }

            for (int token = 0; token < num_tokens; ++token) {
                int nvl_bits = 0;
                int rdma_bits = 0;
                for (int d = 0; d < num_ranks; ++d) {
                    const bool in_rank = is_token_in_rank[token * num_ranks + d];
                    my_send_is_in_rank[token * num_ranks + d] = in_rank;
                    if (!in_rank) {
                        continue;
                    }
                    my_counts[d] += 1;
                    nvl_bits |= 1 << (d % num_nvl_ranks);
                    rdma_bits |= 1 << (d / num_nvl_ranks);
                }

                const auto* src_row = src + static_cast<size_t>(token) * row_bytes;
                auto* dst_row = my_send_x + static_cast<size_t>(token) * row_bytes;
                for (size_t b = 0; b < row_bytes; ++b) {
                    dst_row[b] = src_row[b];
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
                    for (int s = 0; s < num_scales; ++s) {
                        my_send_x_scales[token * num_scales + s] = x_scales[token * num_scales + s];
                    }
                }
            }

            for (int d = 0; d < num_ranks; ++d) {
                int cumulative = 0;
                for (int c = 0; c < kPackNumChannels; ++c) {
                    const int start = (static_cast<int64_t>(num_tokens) * c) / kPackNumChannels;
                    const int end = (static_cast<int64_t>(num_tokens) * (c + 1)) / kPackNumChannels;
                    int count = 0;
                    for (int token = start; token < end; ++token) {
                        count += my_send_is_in_rank[token * num_ranks + d] ? 1 : 0;
                    }
                    cumulative += count;
                    my_channel_counts[d * kPackNumChannels + c] = cumulative;
                }
            }
            // Release fence: flush this rank's packed send buffer (payload, meta,
            // is_token_in_rank, counts) across PCIe so the other NVL peers'
            // Assemble kernel reads the up-to-date data, not stale cache.
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
        });
    });
    if (wait_each) queue.wait();
    ddbg_stage("2-Pack");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchPackBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base, num_nvl_ranks, item); });
    });
    if (wait_each) queue.wait();
    ddbg_stage("3-PackBarrier");

    // StageToLeader: every rank copies its OWN (coherent) send buffer into a
    // per-src_nvl slot in the LEADER's NVL buffer via a reliable cross-device
    // WRITE. This replaces the leader's racy IPC READ of peer send buffers.
    {
        const size_t stage_bytes = layout.total_bytes;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<CombinedDispatchStageKernel>(
                sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) {
                    auto group = item.get_group();
                    auto* leader_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[0]);
                    auto* self_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
                    auto* stage_dst = leader_buf + stage_base_offset + static_cast<size_t>(nvl_rank) * stage_stride;
                    const size_t lid = group.get_local_linear_id();
                    const size_t stride = group.get_local_linear_range();
                    for (size_t b = lid; b < stage_bytes; b += stride) {
                        stage_dst[b] = self_buf[b];
                    }
                    sycl::group_barrier(group);
                    // Release the P2P writes into the leader's VRAM so its
                    // RdmaSend (after the barrier) observes settled data.
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                });
        });
        if (wait_each) queue.wait();
        ddbg_stage("3b-Stage");
    }

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchStageBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 1, num_nvl_ranks, item); });
    });
    if (wait_each) queue.wait();
    ddbg_stage("3c-StageBarrier");

    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<CombinedDispatchRdmaSendKernel>([=]() {
            if (nvl_rank != 0) {
                return;
            }
            // Acquire fence: order reads of the staged send buffers (now in
            // THIS leader's OWN NVL VRAM, written by every rank's StageToLeader)
            // after the StageBarrier release.
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            auto* leader_self_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
            for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                auto* rdma_x = region;
                auto* rdma_m = reinterpret_cast<SourceMeta*>(region + rdma_meta_offset);
                auto* rdma_idx = reinterpret_cast<topk_idx_t*>(region + rdma_topk_idx_offset);
                auto* rdma_wt = reinterpret_cast<float*>(region + rdma_topk_wt_offset);
                auto* rdma_scales = reinterpret_cast<float*>(region + rdma_scales_offset);
                auto* rdma_count = reinterpret_cast<int*>(region + rdma_count_offset);
                *rdma_count = 0;
                if (dst_rdma == my_rdma_rank) {
                    continue;
                }

                int count = 0;
                for (int src_nvl = 0; src_nvl < num_nvl_ranks; ++src_nvl) {
                    // Read the src_nvl peer's data from the leader's OWN staging
                    // slot (coherent same-GPU read) instead of the peer's buffer
                    // over IPC (which races on freshly-written data).
                    auto* peer_buf = leader_self_buf + stage_base_offset + static_cast<size_t>(src_nvl) * stage_stride;
                    auto* peer_x = peer_buf + layout.send_x_offset;
                    auto* peer_m = reinterpret_cast<SourceMeta*>(peer_buf + layout.send_meta_offset);
                    auto* peer_idx = reinterpret_cast<topk_idx_t*>(peer_buf + layout.send_topk_idx_offset);
                    auto* peer_wt = reinterpret_cast<float*>(peer_buf + layout.send_topk_weights_offset);
                    auto* peer_scales = reinterpret_cast<float*>(peer_buf + layout.send_x_scales_offset);
                    auto* peer_rdma_bits = reinterpret_cast<int*>(peer_buf + layout.send_rdma_dest_bits_offset);
                    auto* peer_is_in_rank = reinterpret_cast<bool*>(peer_buf + layout.send_is_token_in_rank_offset);
                    for (int t = 0; t < num_tokens; ++t) {
                        if (((peer_rdma_bits[t] >> dst_rdma) & 1) == 0) {
                            continue;
                        }
                        auto* src_row = peer_x + static_cast<size_t>(t) * row_bytes;
                        auto* dst_row = rdma_x + static_cast<size_t>(count) * row_bytes;
                        for (size_t b = 0; b < row_bytes; ++b) {
                            dst_row[b] = src_row[b];
                        }
                        int dst_nvl_bits = 0;
                        for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                            const int dst_rank = dst_rdma * num_nvl_ranks + dst_nvl;
                            if (peer_is_in_rank[t * num_ranks + dst_rank]) {
                                dst_nvl_bits |= 1 << dst_nvl;
                            }
                        }
                        SourceMeta sm = peer_m[t];
                        sm.is_token_in_nvl_rank_bits = dst_nvl_bits;
                        rdma_m[count] = sm;
                        if (topk_idx != nullptr) {
                            for (int k = 0; k < num_topk; ++k) {
                                rdma_idx[count * num_topk + k] = peer_idx[t * num_topk + k];
                                rdma_wt[count * num_topk + k] = peer_wt[t * num_topk + k];
                            }
                        }
                        if (x_scales != nullptr) {
                            for (int s = 0; s < num_scales; ++s) {
                                rdma_scales[count * num_scales + s] = peer_scales[t * num_scales + s];
                            }
                        }
                        ++count;
                    }
                }
                *rdma_count = count;
            }
        });
    });
    if (wait_each) queue.wait();
    ddbg_stage("4-RdmaSend");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchRdmaPutKernel>(
            sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
                auto group = item.get_group();
                // Mirror the original CUDA implementation: warp-collective
                // nvshmemi_ibgda_put_nbi_warp.  iSHMEM has no warp variant, so
                // use the SCALAR BLOCKING ishmem_putmem from a single WI (group
                // leader).  The blocking put polls the IBGDA CQ until the NIC
                // confirms the remote write is ACK'd, guaranteeing the data has
                // landed in the destination PE's symmetric heap before return.
                //
                // The non-blocking ishmem_putmem_nbi + barrier_all path relied
                // on the barrier's device_quiet to drain the SQ, but that quiet
                // can return early (it reads SND_DBR which may lag the just-
                // issued doorbell), so FwdWrite on the remote PE could read
                // rdma_count while the RDMA write was still in flight -> a
                // non-deterministic RDMA-half/off-by-N token undercount. The
                // blocking put removes that race at the source.
                //
                // Defence-in-depth for the RDMA-Write-to-GPU-VRAM visibility
                // gap (a posted PCIe-P2P write may not have landed in the
                // receiver's VRAM when the sender's RC ACK / barrier release
                // fires, observed as an iter-0/QP-warmup race): stamp each of
                // MY receive regions' count field with a sentinel BEFORE the
                // exchange, so FwdWrite can spin (UC load) until the real
                // count actually arrives instead of trusting the barrier.
                constexpr int kRdmaCountSentinel = -424242;
                if (nvl_rank == 0 && group.get_local_linear_id() == 0) {
                    for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                        if (src_rdma == my_rdma_rank) continue;
                        auto* rcv = rdma_base + static_cast<size_t>(src_rdma) * rdma_region_bytes;
                        uc_store(reinterpret_cast<int*>(rcv + rdma_count_offset), kRdmaCountSentinel);
                    }
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                }
                sycl::group_barrier(group);
                ishmemx_barrier_all_work_group(group);  // work-group barrier: single-WI ishmem_barrier_all() spins on host-proxy progress while host is parked in queue.wait() (cold-QP hang); see notify kernel
                sycl::group_barrier(group);

                if (nvl_rank == 0 && group.get_local_linear_id() == 0) {
                    for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                        if (dst_rdma == my_rdma_rank) continue;
                        auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                        const int dst_pe = dst_rdma * num_nvl_ranks;
                        auto* dst_region = rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes;
                        // Split-put for landing-race elimination: write the
                        // DATA portion (everything before the count field)
                        // first, then write the COUNT word alone. Both are
                        // blocking, on the same QP. RC + same-QP ordering
                        // guarantees the second write's bytes only commit
                        // to the destination memory AFTER the first write's
                        // bytes are committed. Receiver's count!=sentinel
                        // check then becomes a true "all data has landed"
                        // flag instead of an "ACK seen" flag, eliminating
                        // the RDMA-Write-to-VRAM byte-level landing race
                        // that caused the residual ~1/6 dispatch undercount.
                        // NOTE: kept blocking on this site because switching
                        // to NBI causes intermittent NIC DEVICE_LOST mid-run
                        // (likely CQ pressure interaction with the heavy NBI
                        // traffic from the dispatch payload puts above).
                        ishmem_putmem(dst_region, region, rdma_count_offset, dst_pe);
                        ishmem_putmem(dst_region + rdma_count_offset,
                                      region + rdma_count_offset,
                                      sizeof(int), dst_pe);
                    }
                }
                sycl::group_barrier(group);
                ishmemx_barrier_all_work_group(group);  // work-group barrier: single-WI ishmem_barrier_all() spins on host-proxy progress while host is parked in queue.wait() (cold-QP hang); see notify kernel
                sycl::group_barrier(group);
            });
    });
    if (wait_each) queue.wait();
    ddbg_stage("5-RdmaPut");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchFwdWriteKernel>(
            sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
            auto group = item.get_group();
            const int local_id = static_cast<int>(item.get_local_id(0));
            if (nvl_rank != 0) {
                return;
            }
            // Acquire fence: the RDMA receive regions (rdma_count / rdma_m /
            // rdma_x at rdma_base + src_rdma*rdma_region_bytes) were written by
            // the remote NIC into the local symmetric heap. Order all reads of
            // that data after the iSHMEM barrier so this kernel observes the
            // NIC-delivered bytes, not stale GPU L2 cache (the RDMA-half
            // visibility race).
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            int peer_offsets[NUM_MAX_NVL_PEERS] = {0};
            if (local_id == 0) {
                for (int peer = 0; peer < num_nvl_ranks; ++peer) {
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
                if (src_rdma == my_rdma_rank) {
                    continue;
                }
                int before[NUM_MAX_NVL_PEERS] = {0};
                for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                    before[peer] = peer_offsets[peer];
                }

                auto* region = rdma_base + static_cast<size_t>(src_rdma) * rdma_region_bytes;
                auto* rdma_x = region;
                auto* rdma_m = reinterpret_cast<SourceMeta*>(region + rdma_meta_offset);
                auto* rdma_idx = reinterpret_cast<topk_idx_t*>(region + rdma_topk_idx_offset);
                auto* rdma_wt = reinterpret_cast<float*>(region + rdma_topk_wt_offset);
                auto* rdma_scales = reinterpret_cast<float*>(region + rdma_scales_offset);
                auto* rdma_count = reinterpret_cast<int*>(region + rdma_count_offset);
                // Bounded spin (UC load) to let the NIC-delivered count replace
                // the pre-exchange sentinel, closing the RDMA-Write-to-VRAM
                // landing race that the blocking put + barrier alone don't fully
                // cover on the iter-0/QP-warmup path. Only WI 0 spins, then the
                // settled count is BROADCAST to every work-item so the token
                // loop bound (and hence every WI's peer_offsets[] running
                // counter and dst_idx) is IDENTICAL across the work-group. An
                // independent per-WI spin could observe the sentinel replacement
                // at different moments -> divergent loop bounds -> divergent
                // dst_idx -> corrupted/duplicated forward rows (the dispatch
                // parallelization regression). The broadcast removes that.
                constexpr int kRdmaCountSentinel = -424242;
                constexpr unsigned long kRdmaSpinLimit = 2000000ul;
                int count = 0;
                if (local_id == 0) {
                    count = uc_load(rdma_count);
                    for (unsigned long spins = 0; count == kRdmaCountSentinel && spins < kRdmaSpinLimit; ++spins) {
                        if ((spins & 0x3FFF) == 0) {
                            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        }
                        count = uc_load(rdma_count);
                    }
                    if (count == kRdmaCountSentinel) {
                        count = 0;  // dropped write: degrade to undercount, not a hang
                    }
                }
                count = sycl::group_broadcast(group, count, 0);
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                for (int i = 0; i < count; ++i) {
                    SourceMeta sm;
                    sm.src_rdma_rank = uc_load(&rdma_m[i].src_rdma_rank);
                    sm.is_token_in_nvl_rank_bits = uc_load(&rdma_m[i].is_token_in_nvl_rank_bits);
                    sm.src_nvl_rank = uc_load(&rdma_m[i].src_nvl_rank);
                    for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                        if (((sm.is_token_in_nvl_rank_bits >> peer) & 1) == 0) {
                            continue;
                        }
                        auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                        auto* fwd_base = peer_buf + fwd_base_offset;
                        auto* fwd_x = fwd_base + fwd_layout.fwd_x_offset;
                        auto* fwd_m = reinterpret_cast<SourceMeta*>(fwd_base + fwd_layout.fwd_meta_offset);
                        auto* fwd_idx = reinterpret_cast<topk_idx_t*>(fwd_base + fwd_layout.fwd_topk_idx_offset);
                        auto* fwd_wt = reinterpret_cast<float*>(fwd_base + fwd_layout.fwd_topk_weights_offset);
                        auto* fwd_scales = reinterpret_cast<float*>(fwd_base + fwd_layout.fwd_x_scales_offset);
                        const int dst_idx = peer_offsets[peer]++;
                        auto* src_row = rdma_x + static_cast<size_t>(i) * row_bytes;
                        auto* dst_row = fwd_x + static_cast<size_t>(dst_idx) * row_bytes;
                        for (size_t b = local_id; b < row_bytes; b += kIshmemWGSize) {
                            dst_row[b] = uc_load(&src_row[b]);
                        }
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
                                for (int s = 0; s < num_scales; ++s) {
                                    fwd_scales[dst_idx * num_scales + s] = uc_load(&rdma_scales[i * num_scales + s]);
                                }
                            }
                        }
                    }
                }

                // Flush all forwarded payload writes to the remote NVL peers
                // (IPC-mapped GPU memory across PCIe) BEFORE publishing the
                // per-source fwd_counts signal, mirroring CUDA's
                // st_release_sys_global on the NVL channel tail. Every work-item
                // flushes its own strided payload writes to system scope; WI 0's
                // fence alone would not order the other WIs' fwd_x stores.
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                sycl::group_barrier(group);
                if (local_id == 0) {
                    for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                        auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                        auto* fwd_base = peer_buf + fwd_base_offset;
                        auto* fwd_counts = reinterpret_cast<int*>(fwd_base + fwd_layout.fwd_count_offset);
                        fwd_counts[src_rdma] = peer_offsets[peer] - before[peer];
                    }
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                }
            }
            // Final release fence: ensure every payload + count write issued by
            // the NVL leader is flushed across PCIe to the remote peers before
            // this kernel retires and the forward barrier runs.
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            sycl::group_barrier(group);
        });
    });
    if (wait_each) queue.wait();
    ddbg_stage("6-FwdWrite");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchFwdBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 2, num_nvl_ranks, item); });
    });
    if (wait_each) queue.wait();
    ddbg_stage("7-FwdBarrier");
    }  // end else (non-faithful fallback transport)

    // ===== SHARED: Assemble + Head (identical output for both transports) =====
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchAssembleKernel>(
            sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)), [=](sycl::nd_item<1> item) {
            const int local_id = static_cast<int>(item.get_local_id(0));
            // Acquire fence: invalidate any stale local cache and order all
            // subsequent reads of the NVL peers' forwarded/send buffers
            // (IPC-mapped remote GPU memory) after the leader's release fence,
            // mirroring CUDA's ld_acquire_sys_global on the NVL channel tail.
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
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

            // PASS 1: count tokens per src_rank (intra peers + fwd entries).
            int intra_count = 0;
            int fwd_total = 0;
            for (int src_nvl = 0; src_nvl < num_nvl_ranks; ++src_nvl) {
                const int src_rank = my_rdma_rank * num_nvl_ranks + src_nvl;
                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[src_nvl]);
                auto* peer_is_in_rank = reinterpret_cast<bool*>(peer_buf + layout.send_is_token_in_rank_offset);
                int c = 0;
                for (int t = 0; t < num_tokens; ++t) {
                    if (peer_is_in_rank[t * num_ranks + my_global_rank]) {
                        c += 1;
                    }
                }
                per_src_count[src_rank] += c;
                intra_count += c;
            }
            for (int p = 0; p < num_fwd_planes; ++p) {
                for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                    fwd_total += my_fwd_counts[p * num_rdma_ranks + src_rdma];
                }
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
                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[src_nvl]);
                auto* peer_x = peer_buf + layout.send_x_offset;
                auto* peer_m = reinterpret_cast<SourceMeta*>(peer_buf + layout.send_meta_offset);
                auto* peer_idx = reinterpret_cast<topk_idx_t*>(peer_buf + layout.send_topk_idx_offset);
                auto* peer_wt = reinterpret_cast<float*>(peer_buf + layout.send_topk_weights_offset);
                auto* peer_scales = reinterpret_cast<float*>(peer_buf + layout.send_x_scales_offset);
                auto* peer_is_in_rank = reinterpret_cast<bool*>(peer_buf + layout.send_is_token_in_rank_offset);
                for (int t = 0; t < num_tokens; ++t) {
                    if (!peer_is_in_rank[t * num_ranks + my_global_rank]) {
                        continue;
                    }
                    const int pos = cursors[src_rank]++;
                    auto* src_row = peer_x + static_cast<size_t>(t) * row_bytes;
                    auto* dst_row = dst + static_cast<size_t>(pos) * row_bytes;
                    faithful_coop_copy(dst_row, src_row, row_bytes, local_id, kComputeWGSize);
                    if (local_id == 0) {
                        if (meta != nullptr) {
                            meta[pos] = peer_m[t];
                        }
                        if (recv_topk_idx != nullptr) {
                            for (int k = 0; k < num_topk; ++k) {
                                recv_topk_idx[pos * num_topk + k] = peer_idx[t * num_topk + k];
                                recv_topk_weights[pos * num_topk + k] = peer_wt[t * num_topk + k];
                            }
                        }
                        if (recv_x_scales != nullptr) {
                            for (int s = 0; s < num_scales; ++s) {
                                recv_x_scales[pos * num_scales + s] = peer_scales[t * num_scales + s];
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
                                    recv_topk_idx[pos * num_topk + k] = my_fwd_idx[idx * num_topk + k];
                                    recv_topk_weights[pos * num_topk + k] = my_fwd_wt[idx * num_topk + k];
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
        });
    });
    if (wait_each) queue.wait();
    ddbg_stage("8-Assemble");

    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<CombinedDispatchHeadKernel>([=]() {
            if (gbl_channel_prefix_matrix != nullptr) {
                for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                    int cumulative = 0;
                    for (int c = 0; c < num_channels; ++c) {
                        const int start = (static_cast<int64_t>(num_tokens) * c) / num_channels;
                        const int end = (static_cast<int64_t>(num_tokens) * (c + 1)) / num_channels;
                        int count = 0;
                        for (int token = start; token < end; ++token) {
                            count += is_token_in_rank[token * num_ranks + dst_rank] ? 1 : 0;
                        }
                        cumulative += count;
                        gbl_channel_prefix_matrix[dst_rank * num_channels + c] = cumulative;
                    }
                }
            }
            if (rdma_channel_prefix_matrix != nullptr) {
                for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                    int cumulative = 0;
                    for (int c = 0; c < num_channels; ++c) {
                        const int start = (static_cast<int64_t>(num_tokens) * c) / num_channels;
                        const int end = (static_cast<int64_t>(num_tokens) * (c + 1)) / num_channels;
                        int count = 0;
                        for (int token = start; token < end; ++token) {
                            bool hit = false;
                            for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                                const int dst_rank = dst_rdma * num_nvl_ranks + dst_nvl;
                                if (is_token_in_rank[token * num_ranks + dst_rank]) {
                                    hit = true;
                                    break;
                                }
                            }
                            count += hit ? 1 : 0;
                        }
                        cumulative += count;
                        rdma_channel_prefix_matrix[dst_rdma * num_channels + c] = cumulative;
                    }
                }
            }

            // Compute per-(src_rank, dst_rank) base offsets for INTRA dst_ranks (those in my rdma node).
            // For dst_rank R in my rdma, the position of src_rank S's token group in R's recv_x = sum over S' < S of count(S' -> R).
            // S' in my rdma: count via NVL peek of peer's is_token_in_rank.
            // S' in other rdma: count via NVL peek of R's fwd_meta (entries with matching src_rdma+src_nvl).
            constexpr int kMaxRanks = 64;
            int intra_base[NUM_MAX_NVL_PEERS][NUM_MAX_NVL_PEERS] = {{0}};  // intra_base[src_nvl][dst_nvl] = base on R=(my_rdma,dst_nvl) for src=(my_rdma,src_nvl)
            int per_src_per_dst_count[kMaxRanks][NUM_MAX_NVL_PEERS] = {{0}};  // [src_rank][dst_nvl] = count from src to R
            // Intra-rdma source contributions to R: per peer s.nvl, count tokens that go to dst_rank=(my_rdma, dst_nvl).
            for (int src_nvl = 0; src_nvl < num_nvl_ranks; ++src_nvl) {
                const int src_rank = my_rdma_rank * num_nvl_ranks + src_nvl;
                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[src_nvl]);
                auto* peer_is_in_rank = reinterpret_cast<bool*>(peer_buf + layout.send_is_token_in_rank_offset);
                for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                    const int dst_rank = my_rdma_rank * num_nvl_ranks + dst_nvl;
                    int c = 0;
                    for (int t = 0; t < num_tokens; ++t) {
                        if (peer_is_in_rank[t * num_ranks + dst_rank]) c += 1;
                    }
                    per_src_per_dst_count[src_rank][dst_nvl] = c;
                }
            }
            // Cross-rdma source contributions to R: count entries in R's fwd_meta with each (src_rdma, src_nvl).
            for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                auto* dst_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[dst_nvl]);
                auto* dst_fwd_base = dst_buf + fwd_base_offset;
                auto* dst_fwd_m = reinterpret_cast<SourceMeta*>(dst_fwd_base + fwd_layout.fwd_meta_offset);
                auto* dst_fwd_counts = reinterpret_cast<int*>(dst_fwd_base + fwd_layout.fwd_count_offset);
                for (int p = 0; p < num_fwd_planes; ++p) {
                    int scan = p * fwd_layout.plane_tokens;
                    for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                        const int n = dst_fwd_counts[p * num_rdma_ranks + src_rdma];
                        for (int i = 0; i < n; ++i) {
                            const SourceMeta sm = dst_fwd_m[scan + i];
                            const int src_rank = sm.src_rdma_rank * num_nvl_ranks + sm.src_nvl_rank;
                            per_src_per_dst_count[src_rank][dst_nvl] += 1;
                        }
                        scan += n;
                    }
                }
            }
            // Now build intra_base[src_nvl][dst_nvl] = sum over s < (my_rdma*num_nvl + src_nvl) of count[s][dst_nvl].
            for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                int prefix = 0;
                for (int s = 0; s < num_ranks; ++s) {
                    if (s >= my_rdma_rank * num_nvl_ranks && s < (my_rdma_rank + 1) * num_nvl_ranks) {
                        const int src_nvl = s - my_rdma_rank * num_nvl_ranks;
                        intra_base[src_nvl][dst_nvl] = prefix;
                    }
                    prefix += per_src_per_dst_count[s][dst_nvl];
                }
            }

            int same_node_head[NUM_MAX_NVL_PEERS] = {0};
            int rdma_head[NUM_MAX_NVL_PEERS] = {0};
            // For send_nvl_head: write absolute position = intra_base[my_nvl][dst_nvl] + per-peer ordinal within my src_rank's segment.
            // The original same_node_head[dst_nvl] counter is repurposed PER src_nvl peer below.
            int per_peer_dst_ordinal[NUM_MAX_NVL_PEERS][NUM_MAX_NVL_PEERS] = {{0}};  // [src_nvl][dst_nvl]
            for (int src_nvl = 0; src_nvl < num_nvl_ranks; ++src_nvl) {
                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[src_nvl]);
                auto* peer_is_in_rank = reinterpret_cast<bool*>(peer_buf + layout.send_is_token_in_rank_offset);
                auto* peer_rdma_bits = reinterpret_cast<int*>(peer_buf + layout.send_rdma_dest_bits_offset);
                for (int token = 0; token < num_tokens; ++token) {
                    const bool is_my_token = src_nvl == nvl_rank;
                    for (int dst_nvl = 0; dst_nvl < num_nvl_ranks; ++dst_nvl) {
                        const int dst_rank = my_rdma_rank * num_nvl_ranks + dst_nvl;
                        if (!peer_is_in_rank[token * num_ranks + dst_rank]) {
                            continue;
                        }
                        if (is_my_token && send_nvl_head != nullptr) {
                            send_nvl_head[token * num_ranks + dst_rank] =
                                intra_base[src_nvl][dst_nvl] + per_peer_dst_ordinal[src_nvl][dst_nvl];
                        }
                        per_peer_dst_ordinal[src_nvl][dst_nvl] += 1;
                        same_node_head[dst_nvl] += 1;  // kept for any future diagnostic use
                    }
                    for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                        if (dst_rdma == my_rdma_rank || ((peer_rdma_bits[token] >> dst_rdma) & 1) == 0) {
                            continue;
                        }
                        if (per_gpu_rdma) {
                            // Per-GPU RDMA has one independent RDMA stream per nvl_rank.
                            // The recv_pos stamped by F6 is the ordinal within this rank's
                            // same-plane stream, so the handle must count only this nvl plane.
                            if (is_my_token) {
                                if (send_rdma_head != nullptr) {
                                    send_rdma_head[token * num_rdma_ranks + dst_rdma] = rdma_head[dst_rdma];
                                }
                                rdma_head[dst_rdma] += 1;
                            }
                        } else {
                            if (is_my_token && send_rdma_head != nullptr) {
                                send_rdma_head[token * num_rdma_ranks + dst_rdma] = rdma_head[dst_rdma];
                            }
                            rdma_head[dst_rdma] += 1;
                        }
                    }
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
    // When num_nvl_ranks == 1, each rank is its own RDMA rank with no NVL peers.
    // This degenerates to the pure RDMA-only combine path.
    if (num_nvl_ranks <= 1) {
        internode::combine(type,
                           combined_x,
                           combined_topk_weights,
                           is_combined_token_in_rank,
                           rdma_buffer_ptr,
                           x,
                           topk_weights,
                           bias_0,
                           bias_1,
                           combined_rdma_head,
                           combined_nvl_head,
                           src_meta,
                           rdma_channel_prefix_matrix,
                           rdma_rank_prefix_sum,
                           gbl_channel_prefix_matrix,
                           num_tokens,
                           num_combined_tokens,
                           hidden,
                           num_topk,
                           num_max_rdma_chunked_send_tokens,
                           num_max_rdma_chunked_recv_tokens,
                           rank,
                           num_ranks,
                           queue);
        return;
    }

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
    const bool per_gpu_rdma = internode_per_gpu_rdma();
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
    // Faithful path (kFaithful) uses a 64-bit AMO count flag at rdma_flag_offset
    // (ishmemx_long_atomic_add_qp, -count-1), mirroring the dispatch F4b flag.
    // Kept separate from rdma_count_offset and INSIDE the region so every rank
    // computes the identical rdma_region_bytes (sender/receiver agree). With
    // multi-QP (num_qp_ch>1) the flag is an ARRAY of num_qp_ch longs (one per
    // channel/QP); default num_qp_ch==1 is byte-for-byte the single-QP layout.
    // The serial fallback simply never touches these slots.
    const int num_qp_ch = internode_num_qp_channels();
    const size_t rdma_flag_offset = align_offset(rdma_count_offset + sizeof(int), alignof(long));
    const size_t rdma_region_bytes =
        align_offset(rdma_flag_offset + static_cast<size_t>(num_qp_ch) * sizeof(long), 128);
    const size_t rdma_send_base = static_cast<size_t>(num_rdma_ranks) * rdma_region_bytes;

    const size_t total_combined = static_cast<size_t>(num_combined_tokens) * hidden;
    const size_t total_topk = static_cast<size_t>(num_combined_tokens) * num_topk;
    const size_t total_recv_regions = static_cast<size_t>(num_rdma_ranks) * rdma_region_bytes;
    const size_t init_range = std::max({total_combined, total_topk, total_recv_regions, static_cast<size_t>(1)});

    static const bool kDbgCombine = std::getenv("DEEP_EP_DBG_COMBINE") != nullptr;
    auto dbg_last = std::chrono::high_resolution_clock::now();
    auto dbg_stage = [&](const char* name) {
        if (kDbgCombine) {
            auto now = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(now - dbg_last).count();
            dbg_last = now;
            std::fprintf(stderr, "[combine rank=%d nvl=%d rdma=%d] stage done: %s (+%.2f ms)\n",
                         rank, nvl_rank, my_rdma_rank, name, ms);
            std::fflush(stderr);
        }
    };
    static const bool kFuseWaits = [] {
        const char* e = std::getenv("DEEP_EP_INTERNODE_FUSE_WAITS");
        return !(e != nullptr && e[0] != '\0' && std::atoi(e) == 0);
    }();
    const bool wait_each = kDbgCombine || !kFuseWaits;
    dbg_stage("0-entry");

    // Faithful combine gate (shared with dispatch). Empty/"0" => OFF (serial
    // fallback). All faithful transport primitives (blocking put + AMO flag)
    // were validated by the dispatch port; here they replace the serial
    // RdmaPush's two barrier_all collectives + the count-sentinel handshake.
    static const bool kFaithful = [] {
        const char* env = std::getenv("DEEP_EP_INTERNODE_FAITHFUL");
        return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
    }();
    const uint64_t rdma_poll_cap = internode_poll_cap();
    const int rdma_flag_lsc_mode = internode_flag_lsc_mode();
    const bool faithful_post_amo_quiet = internode_post_amo_quiet();
    const bool faithful_force_db = internode_force_db();          // FC5b put doorbell
    const bool faithful_blocking_put = internode_blocking_put();  // FC5b blocking payload put
    const bool faithful_par_gather = internode_par_gather();      // FC5b grid-parallel gather
    if (kFaithful) dbg_stage("F-combine ON");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombineInitKernel<dtype_t>>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
            const size_t linear = static_cast<size_t>(id[0]);
            if (linear < total_combined) {
                dst[linear] = dtype_t{};
            }
            if (linear < total_topk && combined_topk_weights != nullptr) {
                combined_topk_weights[linear] = 0.0f;
            }
            if (nvl_rank == 0 && linear < total_recv_regions) {
                rdma_base[linear] = 0;
            }
        });
    });
    if (wait_each) queue.wait();
    dbg_stage("1-Init");

    queue.submit([&](sycl::handler& cgh) {
        const size_t pack_groups = static_cast<size_t>(std::max(num_tokens, 1));
        cgh.parallel_for<CombinedCombinePackKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(pack_groups * kComputeWGSize), sycl::range<1>(kComputeWGSize)), [=](sycl::nd_item<1> item) {
            auto group = item.get_group();
            const int local_id = static_cast<int>(item.get_local_id(0));
            const int t = static_cast<int>(item.get_group(0));
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
    if (wait_each) queue.wait();
    dbg_stage("2-Pack");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombinePackBarrierKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base, num_nvl_ranks, item); });
    });
    if (wait_each) queue.wait();
    dbg_stage("3-PackBarrier");

    queue.submit([&](sycl::handler& cgh) {
        const size_t rs_groups = static_cast<size_t>(std::max(num_combined_tokens, 1));
        cgh.parallel_for<CombinedCombineRdmaSendKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(rs_groups * kComputeWGSize), sycl::range<1>(kComputeWGSize)), [=](sycl::nd_item<1> item) {
            const int local_id = static_cast<int>(item.get_local_id(0));
            const int ct = static_cast<int>(item.get_group(0));
            if (ct >= num_combined_tokens) return;
            // Acquire fence: order reads of the NVL peers' packed combine
            // buffers (IPC-mapped remote GPU memory) after their release fence.
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            // Compute per-peer topk_weights offset based on each peer's actual
            // num_tokens. The NvlBufferLayout offsets after send_x_offset depend
            // on num_tokens, so using this rank's layout to read from a peer with
            // a different received count would access wrong memory.
            auto peer_topk_offset = [=](int peer_n) -> size_t {
                size_t off = layout.send_x_offset + static_cast<size_t>(peer_n) * row_bytes;
                off = (off + alignof(SourceMeta) - 1) / alignof(SourceMeta) * alignof(SourceMeta);
                off += static_cast<size_t>(peer_n) * sizeof(SourceMeta);
                off = (off + alignof(topk_idx_t) - 1) / alignof(topk_idx_t) * alignof(topk_idx_t);
                off += static_cast<size_t>(peer_n) * num_topk * sizeof(topk_idx_t);
                off = (off + alignof(float) - 1) / alignof(float) * alignof(float);
                return off;
            };

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
                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[dst_nvl]);
                auto* peer_count_ptr = reinterpret_cast<int*>(peer_buf + layout.count_offset);
                const int peer_count = *peer_count_ptr;
                if (peer_recv_pos < 0 || peer_recv_pos >= peer_count) {
                    continue;
                }
                auto* peer_x = reinterpret_cast<dtype_t*>(peer_buf + layout.send_x_offset);
                for (int h = local_id; h < hidden; h += kComputeWGSize) {
                    float value = static_cast<float>(dst[ct * hidden + h]);
                    value += static_cast<float>(peer_x[peer_recv_pos * hidden + h]);
                    dst[ct * hidden + h] = static_cast<dtype_t>(value);
                }
                if (combined_topk_weights != nullptr) {
                    auto* peer_topk = reinterpret_cast<float*>(peer_buf + peer_topk_offset(peer_count));
                    for (int k = local_id; k < num_topk; k += kComputeWGSize) {
                        combined_topk_weights[ct * num_topk + k] += peer_topk[peer_recv_pos * num_topk + k];
                    }
                }
            }
        });
    });
    if (wait_each) queue.wait();
    dbg_stage("4-RdmaSend");

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
    if (kFaithful) {
        // ---- FC5a: init barrier (mirror dispatch F4a0). Zero MY recv-region
        // flags so remote AMOs land onto 0, then the ONE necessary cross-PE
        // rendezvous. Reached by ALL PEs (not leader-gated) so it cannot hang.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulCombineRdmaBarrierKernel<dtype_t>>(
                sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) {
                    auto group = item.get_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if ((per_gpu_rdma || nvl_rank == 0) && local_id == 0) {
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
        if (wait_each) queue.wait();
        dbg_stage("FC5a-RdmaBarrier");

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
                [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                    auto group = item.get_group();
                    auto sg = item.get_sub_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if (!per_gpu_rdma && nvl_rank != 0) return;  // leader-only unless per-GPU RDMA
                    auto peer_meta_off = [=](int peer_n) -> size_t {
                        size_t off = layout.send_x_offset + static_cast<size_t>(peer_n) * row_bytes;
                        return (off + alignof(SourceMeta) - 1) / alignof(SourceMeta) * alignof(SourceMeta);
                    };
                    auto peer_topk_off = [=](int peer_n) -> size_t {
                        size_t off = peer_meta_off(peer_n) + static_cast<size_t>(peer_n) * sizeof(SourceMeta);
                        off = (off + alignof(topk_idx_t) - 1) / alignof(topk_idx_t) * alignof(topk_idx_t);
                        off += static_cast<size_t>(peer_n) * num_topk * sizeof(topk_idx_t);
                        return (off + alignof(float) - 1) / alignof(float) * alignof(float);
                    };
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
                            // serial RdmaPush gather (uniform uc_load count sequence, bulk
                            // copy split by local_id, scalar metadata by WI 0).
                            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                            lsc_fence_sysacq();  // invalidate -> cached int4 loads see peers' data
                            int count = 0;
                            for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                                auto* peer_count_ptr = reinterpret_cast<int*>(peer_buf + layout.count_offset);
                                auto* peer_x = reinterpret_cast<dtype_t*>(peer_buf + layout.send_x_offset);
                                const int peer_count = uc_load(peer_count_ptr);
                                auto* peer_meta = reinterpret_cast<SourceMeta*>(peer_buf + peer_meta_off(peer_count));
                                auto* peer_topk = reinterpret_cast<float*>(peer_buf + peer_topk_off(peer_count));
                                for (int t = 0; t < peer_count; ++t) {
                                    // Cached read (coherent after the lsc_fence_sysacq above,
                                    // same as the payload copy): the former per-token uncached
                                    // uc_load was issued redundantly by every work-item.
                                    if (peer_meta[t].src_rdma_rank != dst_rdma) {
                                        continue;
                                    }
                                    // Per-GPU RDMA: this nvl_rank only handles tokens whose
                                    // original source nvl-plane == its own nvl_rank (so it sends
                                    // them back on RDMA plane nvl_rank). Planes partition tokens
                                    // disjointly, replacing the leader funnel.
                                    if (per_gpu_rdma && peer_meta[t].src_nvl_rank != nvl_rank) {
                                        continue;
                                    }
                                    faithful_coop_copy(reinterpret_cast<uint8_t*>(&rdma_x[count * hidden]),
                                                       reinterpret_cast<const uint8_t*>(&peer_x[t * hidden]),
                                                       static_cast<size_t>(hidden) * sizeof(dtype_t),
                                                       local_id, kComputeWGSize);
                                    if (combined_topk_weights != nullptr) {
                                        for (int k = local_id; k < num_topk; k += kComputeWGSize) {
                                            rdma_wt[count * num_topk + k] = uc_load(&peer_topk[t * num_topk + k]);
                                        }
                                    }
                                    if (local_id == 0) {
                                        rdma_recv_pos[count] = uc_load(&peer_meta[t].is_token_in_nvl_rank_bits);
                                        rdma_src_nvl[count] = uc_load(&peer_meta[t].src_nvl_rank);
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
                        if ((!per_gpu_rdma && nvl_rank != 0) || local_id != 0) return;
                        auto peer_meta_off = [=](int peer_n) -> size_t {
                            size_t off = layout.send_x_offset + static_cast<size_t>(peer_n) * row_bytes;
                            return (off + alignof(SourceMeta) - 1) / alignof(SourceMeta) * alignof(SourceMeta);
                        };
                        auto peer_topk_off = [=](int peer_n) -> size_t {
                            size_t off = peer_meta_off(peer_n) + static_cast<size_t>(peer_n) * sizeof(SourceMeta);
                            off = (off + alignof(topk_idx_t) - 1) / alignof(topk_idx_t) * alignof(topk_idx_t);
                            off += static_cast<size_t>(peer_n) * num_topk * sizeof(topk_idx_t);
                            return (off + alignof(float) - 1) / alignof(float) * alignof(float);
                        };
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();
                        // Slot assignment + scalar metadata, both in the exact serial order.
                        // The metadata (topk/recv_pos/src_nvl) MUST be produced serially: a
                        // parallel grid reading peer_topk/peer_meta over IPC concurrently is
                        // unstable on this HW (deterministically corrupts topk_weights), so only
                        // the bulk x row copy is parallelized in Pass 2. Cached loads (coherent
                        // after the fence above) replace the former per-token uncached loads.
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
                                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                                const int peer_count = uc_load(reinterpret_cast<int*>(peer_buf + layout.count_offset));
                                auto* peer_meta = reinterpret_cast<SourceMeta*>(peer_buf + peer_meta_off(peer_count));
                                auto* peer_topk = reinterpret_cast<float*>(peer_buf + peer_topk_off(peer_count));
                                for (int t = 0; t < peer_count; ++t) {
                                    if (peer_meta[t].src_rdma_rank != dst_rdma) continue;
                                    if (per_gpu_rdma && peer_meta[t].src_nvl_rank != nvl_rank) continue;
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
            if (wait_each) queue.wait();
            // Pass 2: one work-group per input token (peer,t); copy the row + topk +
            // recv_pos/src_nvl into the precomputed slot. Full-GPU parallel row copy.
            const size_t par_groups = slot_n;
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<FaithfulCombineParGatherKernel<dtype_t>>(
                    sycl::nd_range<1>(sycl::range<1>(par_groups * kComputeWGSize),
                                      sycl::range<1>(kComputeWGSize)),
                    [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                        const int local_id = static_cast<int>(item.get_local_id(0));
                        if (!per_gpu_rdma && nvl_rank != 0) return;
                        const size_t g = item.get_group(0);
                        const int peer = static_cast<int>(g / static_cast<size_t>(par_per_peer));
                        const int t = static_cast<int>(g % static_cast<size_t>(par_per_peer));
                        if (peer >= num_nvl_ranks) return;
                        auto peer_meta_off = [=](int peer_n) -> size_t {
                            size_t off = layout.send_x_offset + static_cast<size_t>(peer_n) * row_bytes;
                            return (off + alignof(SourceMeta) - 1) / alignof(SourceMeta) * alignof(SourceMeta);
                        };
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();  // invalidate -> cached int4 loads see peers' data
                        auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                        const int peer_count = uc_load(reinterpret_cast<int*>(peer_buf + layout.count_offset));
                        if (t >= peer_count) return;
                        auto* peer_x = reinterpret_cast<dtype_t*>(peer_buf + layout.send_x_offset);
                        auto* peer_meta = reinterpret_cast<SourceMeta*>(peer_buf + peer_meta_off(peer_count));
                        const int dst_rdma = peer_meta[t].src_rdma_rank;
                        if (dst_rdma < 0 || dst_rdma >= num_rdma_ranks) return;
                        if (per_gpu_rdma && peer_meta[t].src_nvl_rank != nvl_rank) return;
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
        if (wait_each) queue.wait();
        dbg_stage("FC5b-Gather");

        // ---- FC5b2: leader payload put (split from FC5b gather so the NIC put and
        // the local compaction are timed independently). Reads the region compacted
        // by FC5b above (visible across the kernel boundary) and puts [0,rdma_count_offset).
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulCombineRdmaPut2Kernel<dtype_t>>(
                sycl::nd_range<1>(sycl::range<1>(kComputeWGSize), sycl::range<1>(kComputeWGSize)),
                [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                    auto group = item.get_group();
                    auto sg = item.get_sub_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if (!per_gpu_rdma && nvl_rank != 0) return;  // leader-only unless per-GPU RDMA
                    for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                        if (dst_rdma == my_rdma_rank) continue;  // self: no RDMA
                        auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                        const int dst_pe = dst_rdma * num_nvl_ranks + (per_gpu_rdma ? nvl_rank : 0);
                        auto* dst_region = rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes;
                        // Payload put of the data region (length rdma_count_offset), matching
                        // dispatch F4a2. The count is carried by FC5c's -count-1 AMO flag, so
                        // (like dispatch) the count field itself is NOT transmitted. Default
                        // path is the warp-collective NBI put with force_db: a BLOCKING
                        // ishmem_putmem leaves qp0's nic_wq_commit lagging nic_wq_cnt (no gate
                        // reconciles it in an ISOLATED, repeated combine), so FC5c's AMO never
                        // egresses and the receiver poll spins to the cap. The forced-doorbell
                        // warp-put advances the commit watermark in-call, so each combine is
                        // self-contained. DEEP_EP_INTERNODE_BLOCKING_PUT keeps the old blocking
                        // put as an A/B fallback.
                        if (faithful_blocking_put) {
                            if (local_id == 0) {
                                ishmem_putmem(dst_region, region, rdma_count_offset, dst_pe);
                            }
                            sycl::group_barrier(group);
                        } else {
                            // CUDA-faithful multi-QP: split the payload [0,rdma_count_offset)
                            // into num_qp_ch 16B-aligned byte chunks, sub-group c drives chunk
                            // c on qp=c (concurrent NIC send queues, mirroring internode.cu
                            // qp_id==channel_id). FC5c posts a per-channel tail AMO on the same
                            // qp so RC in-order keeps flag[c] after chunk c.
                            const int sgid = sg.get_group_id()[0];
                            if (sgid < num_qp_ch) {
                                const size_t L = rdma_count_offset;
                                const size_t s = internode_qp_chunk_start(L, sgid, num_qp_ch);
                                const size_t e = internode_qp_chunk_start(L, sgid + 1, num_qp_ch);
                                if (e > s) {
                                    ishmemx_putmem_nbi_warp(dst_region + s, region + s, e - s, dst_pe,
                                                            static_cast<unsigned>(sgid), true, sg,
                                                            /*force_db=*/faithful_force_db);
                                }
                                sycl::group_barrier(sg);
                            }
                            sycl::group_barrier(group);
                        }
                    }
                });
        });
        if (wait_each) queue.wait();
        dbg_stage("FC5b2-Put");

        // ---- FC5c: kernel-boundary 64-bit AMO count flag (mirror dispatch F4b).
        // The kernel boundary after FC5b guarantees the blocking puts landed; the
        // quiet_qp flushes the QP, then the RC-ordered AMO posts -count-1 to the
        // receiver's recv-region flag. Posted for EVERY dst_rdma != self.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FaithfulCombineRdmaFlagKernel<dtype_t>>(
                sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) {
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    if ((!per_gpu_rdma && nvl_rank != 0) || local_id != 0) return;
                    for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                        if (dst_rdma == my_rdma_rank) continue;  // local node: no RDMA
                        auto* region = rdma_base + rdma_send_base + static_cast<size_t>(dst_rdma) * rdma_region_bytes;
                        const int count = reinterpret_cast<int*>(region + rdma_count_offset)[0];  // stashed by FC5b
                        const int dst_pe = dst_rdma * num_nvl_ranks + (per_gpu_rdma ? nvl_rank : 0);
                        auto* dst_flag = reinterpret_cast<long*>(
                            rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes + rdma_flag_offset);
                        // Multi-QP: per channel c, quiet qp c then post the tail AMO on qp c
                        // (every flag carries -count-1). Receiver waits for all num_qp_ch flags.
                        for (int c = 0; c < num_qp_ch; ++c) {
                            ishmemx_quiet_qp(dst_pe, static_cast<unsigned>(c));
                            lsc_fence_sysrel();
                            ishmemx_long_atomic_add_qp(dst_flag + c, static_cast<long>(-count - 1), dst_pe,
                                                       static_cast<unsigned>(c));
                            if (faithful_post_amo_quiet) ishmemx_quiet_qp(dst_pe, static_cast<unsigned>(c));
                        }
                    }
                });
        });
        if (wait_each) queue.wait();
        dbg_stage("FC5c-RdmaFlag");

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
                    if (!per_gpu_rdma && nvl_rank != 0) return;
                    // Per-GPU RDMA: this GPU only received tokens whose src_nvl == nvl_rank
                    // (the sender filtered by plane), so every forwarded token targets the
                    // local peer nvl_rank == self. Each GPU is therefore the SOLE writer of
                    // its own fwd buffer -> touch only own plane (peer==nvl_rank) to avoid
                    // cross-GPU fwd_count collisions. Leader model touches all peers.
                    const int fwd_peer_lo = per_gpu_rdma ? nvl_rank : 0;
                    const int fwd_peer_hi = per_gpu_rdma ? (nvl_rank + 1) : num_nvl_ranks;
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
        if (wait_each) queue.wait();
        dbg_stage("FC6-FwdWrite");
    } else {

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombineRdmaPushKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
                auto group = item.get_group();
                const int local_id = static_cast<int>(item.get_local_id(0));
                // NOTE: do NOT early-return for nvl_rank != 0; every PE must reach
                // ishmemx_barrier_all_work_group below (it is a cross-PE collective
                // and would hang waiting for non-puter PEs otherwise).
                const bool is_puter = (nvl_rank == 0);
                // Per-peer offset helpers (same logic as NVL accumulate)
                auto peer_meta_off = [=](int peer_n) -> size_t {
                    size_t off = layout.send_x_offset + static_cast<size_t>(peer_n) * row_bytes;
                    return (off + alignof(SourceMeta) - 1) / alignof(SourceMeta) * alignof(SourceMeta);
                };
                auto peer_topk_off = [=](int peer_n) -> size_t {
                    size_t off = peer_meta_off(peer_n) + static_cast<size_t>(peer_n) * sizeof(SourceMeta);
                    off = (off + alignof(topk_idx_t) - 1) / alignof(topk_idx_t) * alignof(topk_idx_t);
                    off += static_cast<size_t>(peer_n) * num_topk * sizeof(topk_idx_t);
                    return (off + alignof(float) - 1) / alignof(float) * alignof(float);
                };

                // Stamp my receive regions' count with a sentinel BEFORE the
                // exchange so CombineFwdWrite can spin until the real count
                // lands (RDMA-Write-to-VRAM visibility defence-in-depth, same
                // as the dispatch RdmaPut path).
                constexpr int kRdmaCountSentinel = -424242;
                if (is_puter && local_id == 0) {
                    for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                        if (src_rdma == my_rdma_rank) continue;
                        auto* rcv = rdma_base + static_cast<size_t>(src_rdma) * rdma_region_bytes;
                        uc_store(reinterpret_cast<int*>(rcv + rdma_count_offset), kRdmaCountSentinel);
                    }
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                }
                sycl::group_barrier(group);
                ishmemx_barrier_all_work_group(group);  // work-group barrier: single-WI ishmem_barrier_all() spins on host-proxy progress while host is parked in queue.wait() (cold-QP hang); see notify kernel
                sycl::group_barrier(group);

                if (is_puter) {
                    for (int dst_rdma = 0; dst_rdma < num_rdma_ranks; ++dst_rdma) {
                    // For the self RDMA rank, gather directly into this rank's
                    // LOCAL recv region (rdma_base + my_rdma_rank*region) with no
                    // RDMA put, so CombineFwdWrite forwards these same-node tokens
                    // to the correct NVL peer. This mirrors the CUDA combine,
                    // which uses the recv_buffer (not the send_buffer) when
                    // dst_rdma_rank == rdma_rank. Without it, a token whose
                    // combine head lands on the self RDMA rank but a *different*
                    // NVL peer (the off-diagonal ranks where nvl_rank != rdma_rank)
                    // is serviced by neither the NVL-local path nor the RDMA
                    // forward path and reduces to zero.
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
                        // Cooperative gather across the whole work-group. Every
                        // work-item walks the same loop and computes the SAME
                        // `count` sequence (all inputs are identical uncached
                        // reads of the NVL peers' packed buffers), so no
                        // broadcast is needed. Only the bulk hidden/topk copy is
                        // split across work-items by local_id; scalar metadata
                        // and the count are written by WI 0. This replaces the
                        // former single-work-item, per-element uncached copy
                        // (the combine hot path).
                        //
                        // Acquire fence + uncached reads: the NVL peers' packed
                        // combine buffers are IPC-mapped remote GPU VRAM written
                        // in the Pack stage. Cached/reordered cross-device reads
                        // here intermittently miss a peer's freshly-packed token
                        // (leader reads peer != nvl_rank), dropping it from the
                        // forward and yielding combined==0 for that token.
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        int count = 0;
                        for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                            auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                            auto* peer_count_ptr = reinterpret_cast<int*>(peer_buf + layout.count_offset);
                            auto* peer_x = reinterpret_cast<dtype_t*>(peer_buf + layout.send_x_offset);
                            const int peer_count = uc_load(peer_count_ptr);
                            auto* peer_meta = reinterpret_cast<SourceMeta*>(peer_buf + peer_meta_off(peer_count));
                            auto* peer_topk = reinterpret_cast<float*>(peer_buf + peer_topk_off(peer_count));
                            for (int t = 0; t < peer_count; ++t) {
                                if (uc_load(&peer_meta[t].src_rdma_rank) != dst_rdma) {
                                    continue;
                                }
                                for (int h = local_id; h < hidden; h += kIshmemWGSize) {
                                    rdma_x[count * hidden + h] = uc_load(&peer_x[t * hidden + h]);
                                }
                                if (combined_topk_weights != nullptr) {
                                    for (int k = local_id; k < num_topk; k += kIshmemWGSize) {
                                        rdma_wt[count * num_topk + k] = uc_load(&peer_topk[t * num_topk + k]);
                                    }
                                }
                                if (local_id == 0) {
                                    rdma_recv_pos[count] = uc_load(&peer_meta[t].is_token_in_nvl_rank_bits);
                                    rdma_src_nvl[count] = uc_load(&peer_meta[t].src_nvl_rank);
                                }
                                ++count;
                            }
                        }
                        // Ensure every work-item's bulk writes to the region are
                        // complete and visible within the group before WI 0
                        // publishes the count / issues the self-region release.
                        sycl::group_barrier(group);
                        if (local_id == 0) {
                            *rdma_count = count;
                            if (is_self_rdma) {
                                // Publish the locally-gathered self region before
                                // the forward kernel reads it (paired with
                                // FwdWrite's acquire fence). No RDMA put for self.
                                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                            }
                        }
                    }
                    sycl::group_barrier(group);

                    if (!is_self_rdma) {
                        const int dst_pe = dst_rdma * num_nvl_ranks;
                        auto* dst_region = rdma_base + static_cast<size_t>(my_rdma_rank) * rdma_region_bytes;
                        // Split-put for landing-race elimination (see dispatch
                        // RdmaPut for full rationale). Two sequential blocking
                        // puts on the same QP: data first, then count alone.
                        // Kept blocking (matching dispatch site) — NBI on this
                        // hot path causes intermittent NIC DEVICE_LOST.
                        if (local_id == 0) {
                            ishmem_putmem(dst_region, region, rdma_count_offset, dst_pe);
                            ishmem_putmem(dst_region + rdma_count_offset,
                                          region + rdma_count_offset,
                                          sizeof(int), dst_pe);
                        }
                        sycl::group_barrier(group);
                    }
                }
                }  // end if (is_puter)
                // Cross-PE sync to drain pending RDMA puts and make them visible at
                // every PE before FWD WRITE reads from the receive regions.
                // Every PE (including nvl_rank != 0) must reach this collective.
                sycl::group_barrier(group);
                ishmemx_barrier_all_work_group(group);  // work-group barrier: single-WI ishmem_barrier_all() spins on host-proxy progress while host is parked in queue.wait() (cold-QP hang); see notify kernel
                sycl::group_barrier(group);
            });
    });
    if (wait_each) queue.wait();
    dbg_stage("5-RdmaPush");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombineFwdWriteKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
            auto group = item.get_group();
            const int local_id = static_cast<int>(item.get_local_id(0));
            if (nvl_rank != 0) {
                return;
            }
            // Acquire fence: order reads of the NIC-delivered RDMA receive
            // regions after the iSHMEM barrier (RDMA-half visibility race).
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            int peer_offsets[NUM_MAX_NVL_PEERS] = {0};
            if (local_id == 0) {
                for (int peer = 0; peer < num_nvl_ranks; ++peer) {
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
                // NOTE: src_rdma == my_rdma_rank is NOT skipped. Its recv region
                // was populated locally by RdmaPush (no RDMA put), so the same
                // forward-to-NVL-peer logic delivers same-node tokens whose
                // combine head lands on a different NVL peer. Skipping it drops
                // those tokens on the off-diagonal ranks.
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
                // Bounded UC spin until the NIC-delivered count replaces the
                // sentinel (RDMA-Write-to-VRAM landing race; same as dispatch).
                // Every work-item reads the same deterministic UC value so the
                // token loop bound stays identical across the work-group.
                constexpr int kRdmaCountSentinel = -424242;
                constexpr unsigned long kRdmaSpinLimit = 2000000ul;
                int count = uc_load(rdma_count);
                for (unsigned long spins = 0; count == kRdmaCountSentinel && spins < kRdmaSpinLimit; ++spins) {
                    if ((spins & 0x3FFF) == 0) {
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                    }
                    count = uc_load(rdma_count);
                }
                if (count == kRdmaCountSentinel) {
                    count = 0;  // dropped write: degrade gracefully, no hang
                }
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                for (int i = 0; i < count; ++i) {
                    const int target_nvl = uc_load(&rdma_src_nvl[i]);
                    if (target_nvl < 0 || target_nvl >= num_nvl_ranks) {
                        continue;
                    }
                    auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[target_nvl]);
                    auto* fwd_base = peer_buf + fwd_base_offset;
                    auto* fwd_x = reinterpret_cast<dtype_t*>(fwd_base + fwd_layout.fwd_x_offset);
                    auto* fwd_meta = reinterpret_cast<SourceMeta*>(fwd_base + fwd_layout.fwd_meta_offset);
                    auto* fwd_topk = reinterpret_cast<float*>(fwd_base + fwd_layout.fwd_topk_weights_offset);
                    // Guard against overrunning the target peer's forward buffer
                    // (capacity == max_fwd_tokens). Overrunning would write OOB
                    // into an adjacent peer's IPC-mapped VRAM.
                    if (peer_offsets[target_nvl] >= max_fwd_tokens) {
                        continue;
                    }
                    const int dst_idx = peer_offsets[target_nvl]++;
                    // Bulk hidden/topk copy parallelized across the work-group;
                    // scalar metadata written by work-item 0.
                    for (int h = local_id; h < hidden; h += kIshmemWGSize) {
                        fwd_x[dst_idx * hidden + h] = uc_load(&rdma_x[i * hidden + h]);
                    }
                    if (combined_topk_weights != nullptr) {
                        for (int k = local_id; k < num_topk; k += kIshmemWGSize) {
                            fwd_topk[dst_idx * num_topk + k] = uc_load(&rdma_wt[i * num_topk + k]);
                        }
                    }
                    if (local_id == 0) {
                        fwd_meta[dst_idx] = SourceMeta{src_rdma, uc_load(&rdma_recv_pos[i]), target_nvl};
                    }
                }

                // Flush forwarded payload writes to the remote NVL peers
                // (IPC-mapped GPU memory across PCIe) BEFORE publishing the
                // fwd_counts signal (CUDA st_release_sys_global equivalent).
                // Every work-item flushes its own payload writes to system
                // scope; WI 0's fence alone would not flush the others' writes.
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                sycl::group_barrier(group);
                if (local_id == 0) {
                    for (int peer = 0; peer < num_nvl_ranks; ++peer) {
                        auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer]);
                        auto* fwd_base = peer_buf + fwd_base_offset;
                        auto* fwd_counts = reinterpret_cast<int*>(fwd_base + fwd_layout.fwd_count_offset);
                        fwd_counts[src_rdma] = peer_offsets[peer] - before[peer];
                    }
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                }
            }
            // Final release fence so all payload + count writes are flushed
            // across PCIe to the remote peers before the forward barrier runs.
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            sycl::group_barrier(group);
        });
    });
    if (wait_each) queue.wait();
    dbg_stage("6-FwdWrite");
    }  // end else (serial RdmaPush + FwdWrite; faithful path handled above)

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombineFwdBarrierKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 1, num_nvl_ranks, item); });
    });
    if (wait_each) queue.wait();
    dbg_stage("7-FwdBarrier");

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
    dbg_stage("8-Reduce");
#else
    TORCH_CHECK(false, "combine_nvl_rdma requires DEEP_EP_ENABLE_ISHMEM");
#endif
}

}  // namespace internode
}  // namespace deep_ep
