#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>

#include "xpu_kernels.hpp"

#include <sycl/ext/oneapi/experimental/builtins.hpp>

#ifdef DEEP_EP_ENABLE_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

namespace deep_ep {
namespace internode {
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

// ===========================================================================
// Co-residency clamp for the fused internode grid.
//
// Both fused kernels split every channel across TWO work-groups
// (`channel_id = sm_id / 2`, `is_forwarder = sm_id % 2`) which spin on each
// other's queue counters (`rdma_channel_tail`, `nvl_channel_tail/head`).  That
// makes the grid a producer/consumer network with hard forward-progress
// dependencies ACROSS work-groups, which is only correct if every work-group is
// simultaneously resident.  Intel GPUs give no such guarantee and do not preempt
// a spinning work-group, so an over-sized grid starves a producer and surfaces
// as a hang, or as a `kFusedSpinCap` break that silently drops whole tokens
// (measured: 0/8 failures at num_sms=8, 2/8 at 16, 6/10 at 24, 4/6 at 32).
//
// `csrc/xpu/internode_ll.cpp::ll_put_wgs()` applies the same rule to the LL put
// grid ("requires the producing sub-groups to be CO-RESIDENT"); this is the
// internode-normal counterpart.
//
// The limit is derived, not guessed: `kernel_queue_specific::max_num_work_groups`
// is the driver's own per-kernel occupancy answer (it accounts for the work-group
// size, SLM footprint and register/GRF pressure of THAT kernel).  Dispatch and
// combine have different occupancy, but `num_channels` is baked into the shared
// RDMA/NVL buffer layout and into the tensors dispatch hands to combine, so a
// single grid size must satisfy both: we take the min.
// ===========================================================================
namespace {

template <typename KernelName>
int query_max_coresident_wgs(sycl::queue& queue, int wg_size) {
    namespace syclex = sycl::ext::oneapi::experimental;
    try {
        auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(queue.get_context(),
                                                                             {queue.get_device()},
                                                                             {sycl::get_kernel_id<KernelName>()});
        auto kernel = bundle.get_kernel(sycl::get_kernel_id<KernelName>());
        const size_t n = kernel.template ext_oneapi_get_info<syclex::info::kernel_queue_specific::max_num_work_groups>(
            queue, sycl::range<3>(1, 1, static_cast<size_t>(wg_size)), 0);
        return static_cast<int>(n);
    } catch (const std::exception& e) {
        // Fallback: hardware-thread arithmetic.  A work-group of `wg_size` at
        // sub-group 32 needs `wg_size / 32` hardware threads, all on one Xe-core.
        auto dev = queue.get_device();
        int eu_count = 0, eu_per_ss = 0, threads_per_eu = 0;
        try {
            eu_count = static_cast<int>(dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>());
            eu_per_ss = static_cast<int>(dev.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>());
            threads_per_eu = static_cast<int>(dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>());
        } catch (...) {
        }
        if (eu_count <= 0 || eu_per_ss <= 0 || threads_per_eu <= 0)
            return static_cast<int>(dev.get_info<sycl::info::device::max_compute_units>());
        const int xe_cores = std::max(eu_count / eu_per_ss, 1);
        const int threads_per_core = eu_per_ss * threads_per_eu;
        const int threads_per_wg = std::max(wg_size / 32, 1);
        return std::max(xe_cores * (threads_per_core / threads_per_wg), 1);
    }
}

}  // namespace

namespace {
// ISHMEM_IBGDA_QPS_PER_PE as iSHMEM interprets it: rounded up to a power of two and
// clamped to [1, 16].  deep_ep/buffer.py sets this env BEFORE the Buffer is created.
inline int fused_qps_per_pe_env() {
    const char* e = std::getenv("ISHMEM_IBGDA_QPS_PER_PE");
    int v = (e != nullptr && e[0] != '\0') ? std::atoi(e) : 1;
    if (v < 1) v = 1;
    if (v > 16) v = 16;
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}
}  // namespace

int fused_max_coresident_sms(int num_rdma_ranks, sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    static std::mutex mtx;
    static std::map<int, int> cache;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(num_rdma_ranks);
        if (it != cache.end()) return it->second;
    }

    const int dispatch_wg = DEEP_EP_FUSED_NUM_THREADS;
    int dispatch_cap = 0, combine_cap = 0, combine_wg = 0;
    switch (num_rdma_ranks) {
        case 2:
            combine_wg = (FusedCombineShape<2>::kForwarders + 1) * 32;
            dispatch_cap = query_max_coresident_wgs<FusedDispatchKernel<2>>(queue, dispatch_wg);
            combine_cap = query_max_coresident_wgs<FusedCombineKernel<2>>(queue, combine_wg);
            break;
        case 4:
            combine_wg = (FusedCombineShape<4>::kForwarders + 1) * 32;
            dispatch_cap = query_max_coresident_wgs<FusedDispatchKernel<4>>(queue, dispatch_wg);
            combine_cap = query_max_coresident_wgs<FusedCombineKernel<4>>(queue, combine_wg);
            break;
        case 8:
            // Combine is capped at 4 RDMA ranks by the BMG named-barrier limit, so
            // only the dispatch kernel exists at R=8.
            dispatch_cap = query_max_coresident_wgs<FusedDispatchKernel<8>>(queue, dispatch_wg);
            combine_cap = dispatch_cap;
            break;
        default:
            return 2;
    }

    int sms = std::min(dispatch_cap, combine_cap);

    // EMPIRICAL SAFETY CAP.  On Arc Pro B60 (160 EU / 20 Xe-cores) the driver's own
    // per-kernel answer is 20 work-groups for both fused kernels (1 per Xe-core), but
    // measurement says 20 is NOT safe: at num_sms=20-24 the run still hangs, at 16 it
    // hangs 2/8, and only at 8 is it clean (0/8, reproduced on two independent builds).
    // So the driver bound is NECESSARY but not SUFFICIENT - something beyond raw
    // co-residency also scales with the grid.  Until that is explained we ship the
    // measured-safe value; `DEEP_EP_FUSED_MAX_SMS` overrides it for experiments.
    //
    // 2026-08 UPDATE (see playbook 24.2.1): the mechanism is NOT co-residency, it is
    // per-QP contention inside iSHMEM.  With ISHMEM_IBGDA_QPS_PER_PE=1 EVERY channel's
    // RDMA sender drives the SAME QP, and the failure appears once ~12 channels share
    // one QP (4/6 hangs) while 6 channels/QP (6/6) and 4 channels/QP (16/16) are clean.
    // Give each channel its own QP and the previously-failing grid is clean:
    // num_sms=24 + QPS_PER_PE=16 measured 16/16 @2048 tok and 8/8 @4096 tok, hidden 7168.
    // So the grid cap is only needed when channels must SHARE queue pairs.
    constexpr int kEmpiricalSafeSms = 8;         // safe when many channels share one QP
    constexpr int kSafeChannelsPerQp = 2;        // validated: 1 ch/QP clean, 6 ch/QP clean
    const int qps_per_pe = fused_qps_per_pe_env();
    const int qp_safe_sms = 2 * kSafeChannelsPerQp * qps_per_pe;  // grid = 2 WGs per channel
    const int safe_sms = std::max(kEmpiricalSafeSms, qp_safe_sms);
    if (sms > safe_sms) sms = safe_sms;

    if (sms < 2) sms = 2;
    sms -= (sms % 2);  // the grid is 2 work-groups per channel

    const char* env = std::getenv("DEEP_EP_FUSED_MAX_SMS");
    if (env != nullptr && env[0] != '\0') {
        const int v = std::atoi(env);
        if (v >= 2) {
            std::fprintf(stderr,
                         "[DeepEP] internode fused co-residency limit overridden by DEEP_EP_FUSED_MAX_SMS: "
                         "%d (device-derived value was %d; dispatch=%d combine=%d)\n",
                         v - (v % 2),
                         sms,
                         dispatch_cap,
                         combine_cap);
            sms = v - (v % 2);
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        cache[num_rdma_ranks] = sms;
    }
    return sms;
#else
    (void)num_rdma_ranks;
    (void)queue;
    return 2;
#endif
}

}  // namespace internode
}  // namespace deep_ep
