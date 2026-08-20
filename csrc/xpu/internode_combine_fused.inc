// ============================================================================
// FUSED, WARP-SPECIALIZED INTERNODE **COMBINE** (CUDA-faithful port)
// ----------------------------------------------------------------------------
// 1:1 port of the single fused CUDA kernel
//     csrc/cuda_kernels/internode.cu:1716  `combine<...>`
// plus the head-transform half of internode.cu:1311 `cached_notify<...>`
// (which is a separate kernel in CUDA too).
//
// Structure mirrored from CUDA:
//   * grid  = num_channels * 2 work-groups; ODD sm_id = forwarder SM,
//             EVEN sm_id = NVL-sender / RDMA-receiver SM              (:1745)
//   * block = (Shape::kForwarders + 1) * 32                               (:1716)
//   * roles = kNVLSender / kNVLAndRDMAForwarder / kRDMAReceiver /
//             kCoordinator                                            (:1759-1776)
//
// Warp-subset synchronization (`count` in SUB-GROUPS):
//   CUDA :1952 sync_forwarder_smem     barrier.sync 0, (Shape::kForwarders+1)*32
//   SYCL       named_barrier_init(Shape::kForwarders + 1)      == 25 sub-groups
//              participants: forwarder-SM warps 0..23 + coordinator warp 24
//
//   CUDA :1953 sync_rdma_receiver_smem barrier.sync 1, (Shape::kRDMAReceivers+1)*32
//   SYCL       named_barrier_init(Shape::kRDMAReceivers + 1)   == 17 sub-groups
//              participants: receiver-SM warps 8..23 + coordinator warp 24
//
//   CUDA :1966 sync_large_warp         bar.sync (dst_rdma_rank+2),
//                                                Shape::kWarpsPerForwarder*32
//   SYCL       one named_barrier_init(Shape::kWarpsPerForwarder) PER destination
//              RDMA rank, selected by a literal if/else chain on
//              dst_rdma_rank (IGC forbids indexing an array of handles).
//              When Shape::kWarpsPerForwarder == 1 CUDA degenerates to
//              __syncwarp(), which maps to sycl::group_barrier(sub_group).
//
// Deviations from CUDA:
//   C1. TMA -> lane-strided 16B (int4) LSC copies, i.e. `combine_token`'s
//       non-TMA branch (internode.cu:1646-1691) is used by BOTH the forwarder
//       and the receiver. Semantically identical.
//   C2. NUM_MAX_NVL_PEERS (8) is kept as the NVL-SENDER WARP COUNT so the
//       role split and hence the NamedBarrier counts stay compile-time
//       constants; every stride that is a PEER COUNT uses the runtime
//       num_nvl_ranks. Sender warps whose dst_nvl_rank >= num_nvl_ranks return
//       immediately (they participate in no barrier).
//   C3. All cross-rank NVL traffic is a PUSH, exactly as in CUDA: the sender
//       writes x/tail into the DESTINATION's buffer and reads head from its
//       OWN; the coordinator writes head into the PEER's buffer. There is no
//       IPC pull-read anywhere in combine.
//   C4. nvshmemi_ibgda_put_nbi_warp -> ishmemx_putmem_nbi_subgroup;
//       nvshmemi_ibgda_amo_nonfetch_add -> ishmemx_fence_qp +
//       ishmemx_long_atomic_add_qp on the SAME qp (RC in-order therefore
//       orders the tail AMO behind the payload), or a local atomic when the
//       destination is this RDMA rank. Never ishmem_quiet().
//   C5. Cache/coherence: every queue counter uses uc_load/uc_store plus a
//       system-scope fence; payload reads go through ld_nc_global_v.
//   C6. Spin loops are bounded (kFusedSpinCap) instead of CUDA's clock64()
//       timeout + trap().
// ============================================================================

#define DEEP_EP_COMBINE_FWD_WARPS 24

// CUDA internode.cu:1713-1715 -- kept as static members (never as kernel-lambda
// locals: captured constexpr breaks the host/device lambda layout check).
template <int R>
struct FusedCombineShape {
    static constexpr int kWarpsPerForwarder = (DEEP_EP_COMBINE_FWD_WARPS / R > 0) ? DEEP_EP_COMBINE_FWD_WARPS / R : 1;
    static constexpr int kForwarders = R * kWarpsPerForwarder;
    static constexpr int kRDMAReceivers = kForwarders - NUM_MAX_NVL_PEERS;
};

template <int kNumRDMARanks>
class FusedCombineKernel;
template <int kNumRDMARanks>
class CachedNotifyRdmaHeadKernel;
template <int kNumRDMARanks>
class CachedNotifyNvlHeadKernel;

// CUDA internode.cu:42 with num_scales == 0 and num_topk_idx == 0.
inline int fused_combine_num_bytes_per_token(int hidden_int4, int num_topk) {
    const size_t raw = static_cast<size_t>(hidden_int4) * sizeof(int4_t) + sizeof(SourceMeta) +
                       static_cast<size_t>(num_topk) * sizeof(float);
    return static_cast<int>(align_offset(raw, sizeof(int4_t)));
}

// ---------------------------------------------------------------------------
// `cached_notify` head transform (CUDA internode.cu:1375-1465). Rewrites every
// negative entry of combined_rdma_head / combined_nvl_head, scanning each
// channel's token range in REVERSE, to -(next_valid_head)-1. Purely local.
// ---------------------------------------------------------------------------
template <int kNumRDMARanks>
void launch_fused_cached_notify_heads(int* combined_rdma_head,
                                      int* combined_nvl_head,
                                      int num_combined_tokens,
                                      int num_channels,
                                      const int* rdma_channel_prefix_matrix,
                                      const int* rdma_rank_prefix_sum,
                                      int num_nvl_ranks,
                                      sycl::queue& queue) {
    (void)num_nvl_ranks;
    // combined_rdma_head: one lane per RDMA rank, one sub-group per channel.
    {
        const size_t wg = 32;
        const size_t global = static_cast<size_t>(num_channels) * wg;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<CachedNotifyRdmaHeadKernel<kNumRDMARanks>>(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    const int channel = static_cast<int>(item.get_group(0));
                    const int lane = static_cast<int>(item.get_local_id(0));
                    if (lane >= kNumRDMARanks) return;
                    int token_start_idx = 0, token_end_idx = 0;
                    get_channel_task_range(num_combined_tokens, num_channels, channel, token_start_idx, token_end_idx);
                    int last_head = 1 << 25;  // CUDA's heuristic large number
                    for (int t = token_end_idx - 1; t >= token_start_idx; --t) {
                        const int cur = combined_rdma_head[static_cast<int64_t>(t) * kNumRDMARanks + lane];
                        if (cur < 0)
                            combined_rdma_head[static_cast<int64_t>(t) * kNumRDMARanks + lane] = -last_head - 1;
                        else
                            last_head = cur;
                    }
                });
        });
    }
    // combined_nvl_head: one lane per NVL peer, one sub-group per (rdma rank, channel).
    {
        const size_t wg = 32;
        const size_t global = static_cast<size_t>(kNumRDMARanks) * num_channels * wg;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<CachedNotifyNvlHeadKernel<kNumRDMARanks>>(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    const int g = static_cast<int>(item.get_group(0));
                    const int dst_rdma_rank = g / num_channels;
                    const int channel = g % num_channels;
                    const int lane = static_cast<int>(item.get_local_id(0));
                    if (lane >= NUM_MAX_NVL_PEERS) return;  // CUDA internode.cu:1444
                    int token_start_idx =
                        channel == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel - 1];
                    int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel];
                    const int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                    token_start_idx += shift;
                    token_end_idx += shift;
                    int last_head = 1 << 25;
                    for (int t = token_end_idx - 1; t >= token_start_idx; --t) {
                        const int cur = combined_nvl_head[static_cast<int64_t>(t) * NUM_MAX_NVL_PEERS + lane];
                        if (cur < 0)
                            combined_nvl_head[static_cast<int64_t>(t) * NUM_MAX_NVL_PEERS + lane] = -last_head - 1;
                        else
                            last_head = cur;
                    }
                });
        });
    }
}

// Stack-call-free bf16 <-> float conversion (see deviation C7 below).
inline __attribute__((always_inline)) float fused_bf16_to_float(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    return sycl::bit_cast<float>(u);
}

inline __attribute__((always_inline)) uint16_t fused_float_to_bf16(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    // NaN stays NaN.
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u)
        return static_cast<uint16_t>((u >> 16) | 0x0040u);
    uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(rounded >> 16);
}

// Relaxed SLM publish/observe helpers: the coordinator polls this state with no
// intervening barrier, mirroring CUDA's `volatile __shared__`.
template <typename Acc>
inline __attribute__((always_inline)) void slm_store(const Acc& acc, int idx, int value) {
    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>
        r(acc[idx]);
    r.store(value);
}

// ---------------------------------------------------------------------------
// combine_token (CUDA internode.cu:1550), non-TMA branch only (deviation C1).
// ---------------------------------------------------------------------------
template <int kNumRanks, bool kMaybeWithBias, int kMaxNumRanks, typename GetAddrFn, typename RecvTWFn>
inline __attribute__((always_inline)) void fused_combine_token(bool is_token_in_rank,
                                int head_idx,
                                int lane_id,
                                int hidden_int4,
                                int num_topk,
                                int4_t* combined_row,
                                float* combined_topk_weights,
                                const int4_t* bias_0_int4,
                                const int4_t* bias_1_int4,
                                int num_max_recv_tokens,
                                const GetAddrFn& get_addr_fn,
                                const RecvTWFn& recv_tw_fn,
                                const sycl::sub_group& sg) {
    int num_topk_ranks = 0;
    int topk_ranks[kMaxNumRanks];
    int slot_indices[kMaxNumRanks];
    for (int i = 0; i < kNumRanks; ++i) {
        if (sycl::select_from_group(sg, is_token_in_rank ? 1 : 0, i)) {
            slot_indices[num_topk_ranks] = sycl::select_from_group(sg, head_idx, i) % num_max_recv_tokens;
            topk_ranks[num_topk_ranks++] = i;
        }
    }

    // C7: bf16 <-> float conversions are done with explicit bit manipulation instead of
    // sycl::ext::oneapi::bfloat16's conversion operators.  Those operators lower to the
    // *external* devicelib symbols __devicelib_ConvertBF16ToFINTEL /
    // __devicelib_ConvertFToBF16INTEL, which IGC materialises as vISA STACK CALLS.  A stack
    // call inside a kernel that also uses SPIR-V NamedBarrier makes IGC stamp NBarrierCnt on
    // both the kernel and the outlined function, and the compiler then dies with an
    // "internal compiler error" during codegen.  Doing the conversion inline keeps the
    // kernel stack-call free.  Rounding is round-to-nearest-even, matching the HW converter.
    constexpr int kDtypePerInt4 = static_cast<int>(sizeof(int4_t) / sizeof(uint16_t));

    for (int i = lane_id; i < hidden_int4; i += 32) {
        float values[kDtypePerInt4] = {0.f};
        if constexpr (kMaybeWithBias) {
            if (bias_0_int4 != nullptr) {
                int4_t b = ld_nc_global_v(bias_0_int4 + i);
                auto* bv = reinterpret_cast<const uint16_t*>(&b);
                for (int j = 0; j < kDtypePerInt4; ++j) values[j] += fused_bf16_to_float(bv[j]);
            }
            if (bias_1_int4 != nullptr) {
                int4_t b = ld_nc_global_v(bias_1_int4 + i);
                auto* bv = reinterpret_cast<const uint16_t*>(&b);
                for (int j = 0; j < kDtypePerInt4; ++j) values[j] += fused_bf16_to_float(bv[j]);
            }
        }
        for (int j = 0; j < num_topk_ranks; ++j) {
            int4_t v = ld_uc_global_v(get_addr_fn(topk_ranks[j], slot_indices[j], i));
            auto* rv = reinterpret_cast<const uint16_t*>(&v);
            for (int k = 0; k < kDtypePerInt4; ++k) values[k] += fused_bf16_to_float(rv[k]);
        }
        int4_t out;
        auto* od = reinterpret_cast<uint16_t*>(&out);
        for (int j = 0; j < kDtypePerInt4; ++j) od[j] = fused_float_to_bf16(values[j]);
        st_uc_global_v(combined_row + i, out);
    }

    if (lane_id < num_topk && combined_topk_weights != nullptr) {
        float value = 0.f;
        for (int i = 0; i < num_topk_ranks; ++i) value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
        uc_store(combined_topk_weights + lane_id, value);
    }
}

// ---------------------------------------------------------------------------
// The fused combine kernel.
// ---------------------------------------------------------------------------
template <int kNumRDMARanks>
void launch_fused_combine(void* combined_x,
                          float* combined_topk_weights,
                          const void* x,
                          const float* topk_weights,
                          const void* bias_0,
                          const void* bias_1,
                          const int* combined_rdma_head,
                          const int* combined_nvl_head,
                          const SourceMeta* src_meta,
                          const int* rdma_channel_prefix_matrix,
                          const int* rdma_rank_prefix_sum,
                          const int* gbl_channel_prefix_matrix,
                          int num_tokens,
                          int num_combined_tokens,
                          int hidden,
                          int num_topk,
                          void* rdma_buffer_ptr,
                          int num_max_rdma_chunked_send_tokens,
                          int num_max_rdma_chunked_recv_tokens,
                          void** buffer_ptrs,
                          int num_max_nvl_chunked_send_tokens,
                          int num_max_nvl_chunked_recv_tokens,
                          int rank,
                          int num_ranks,
                          int num_nvl_ranks,
                          int num_channels,
                          sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    using Shape = FusedCombineShape<kNumRDMARanks>;
    static_assert(Shape::kRDMAReceivers > 0, "not enough forwarder warps");

    const int hidden_int4 = hidden / static_cast<int>(sizeof(int4_t) / sizeof(sycl::ext::oneapi::bfloat16));
    const int hidden_bytes = hidden_int4 * static_cast<int>(sizeof(int4_t));
    const int num_bytes_per_token = fused_combine_num_bytes_per_token(hidden_int4, num_topk);
    const int num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;
    const unsigned qps_per_pe = fused_qps_per_pe();
    const int num_sms = num_channels * 2;

    // ---- NVL buffer offsets (see the C3 note; all cross-rank ops are pushes) ----
    const size_t nvl_x_region = static_cast<size_t>(num_max_nvl_chunked_recv_tokens) * num_bytes_per_token *
                                num_nvl_ranks * num_channels;
    const size_t nvl_hd_region = static_cast<size_t>(kNumRDMARanks) * sizeof(int) * num_nvl_ranks * num_channels;
    const size_t nvl_head_off = nvl_x_region;
    const size_t nvl_tail_off = nvl_x_region + nvl_hd_region;

    queue.submit([&](sycl::handler& cgh) {
        // CUDA internode.cu:1946-1949
        sycl::local_accessor<int, 1> smem_fwd_nvl_head(
            sycl::range<1>{static_cast<size_t>(Shape::kForwarders) * NUM_MAX_NVL_PEERS}, cgh);
        sycl::local_accessor<int, 1> smem_fwd_retired(sycl::range<1>{static_cast<size_t>(Shape::kForwarders)}, cgh);
        sycl::local_accessor<int, 1> smem_rcv_rdma_head(
            sycl::range<1>{static_cast<size_t>(Shape::kRDMAReceivers) * kNumRDMARanks}, cgh);
        sycl::local_accessor<int, 1> smem_rcv_retired(sycl::range<1>{static_cast<size_t>(Shape::kRDMAReceivers)}, cgh);

        cgh.parallel_for<FusedCombineKernel<kNumRDMARanks>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_sms) * ((Shape::kForwarders + 1) * 32)),
                              sycl::range<1>((Shape::kForwarders + 1) * 32)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                auto sg = item.get_sub_group();
                const int sm_id = static_cast<int>(item.get_group(0));
                const int phys_warp_id = static_cast<int>(sg.get_group_linear_id());
                const int lane_id = static_cast<int>(sg.get_local_linear_id());
                const int channel_id = sm_id / 2;
                const bool is_forwarder_sm = (sm_id % 2 == 1);
                const int rdma_rank = rank / num_nvl_ranks;
                const int nvl_rank = rank % num_nvl_ranks;

                // IGC: every NamedBarrier handle is a plain SSA local of this
                // function, materialized before any control flow.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                auto* nb_fwd = ::named_barrier_init(Shape::kForwarders + 1);        // CUDA barrier id 0
                auto* nb_rcv = ::named_barrier_init(Shape::kRDMAReceivers + 1);     // CUDA barrier id 1
                auto* nb_d0 = ::named_barrier_init(Shape::kWarpsPerForwarder);      // CUDA barrier id 2 + 0
                auto* nb_d1 = ::named_barrier_init(Shape::kWarpsPerForwarder);      // ... + 1
                // IGC LIMIT (measured): a kernel that declares ~10 SPIR-V named barriers makes the
                // BMG back-end die with "internal compiler error" during codegen.  Four barriers
                // compile fine.  Until the real limit is characterised we alias the per-destination
                // handles above dst_rdma_rank 1 and reject kNumRDMARanks > 2 on the host.
                auto* nb_d2 = nb_d0;
                auto* nb_d3 = nb_d0;
                auto* nb_d4 = nb_d0;
                auto* nb_d5 = nb_d0;
                auto* nb_d6 = nb_d0;
                auto* nb_d7 = nb_d0;
#endif

                // ---- Warp roles: CUDA internode.cu:1759-1776 ----
                enum Role { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };
                Role warp_role;
                int warp_id;
                if (!is_forwarder_sm) {
                    if (phys_warp_id < NUM_MAX_NVL_PEERS) {
                        warp_role = kNVLSender;
                        warp_id = (phys_warp_id + channel_id) % NUM_MAX_NVL_PEERS;
                    } else if (phys_warp_id < Shape::kForwarders) {
                        warp_role = kRDMAReceiver;
                        warp_id = phys_warp_id - NUM_MAX_NVL_PEERS;
                    } else {
                        warp_role = kCoordinator;
                        warp_id = 0;
                    }
                } else {
                    if (phys_warp_id < Shape::kForwarders) {
                        warp_role = kNVLAndRDMAForwarder;
                        warp_id = (phys_warp_id + channel_id) % Shape::kForwarders;
                    } else {
                        warp_role = kCoordinator;
                        warp_id = 0;
                    }
                }

                // ---- RDMA symmetric layout (CUDA :1918-1921) ----
                void* rdma_cursor = rdma_buffer_ptr;
                FusedSymBuffer<uint8_t> rdma_channel_data(
                    rdma_cursor, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id,
                    num_channels);
                FusedSymBuffer<uint64_t, false> rdma_channel_head(rdma_cursor, 1, kNumRDMARanks, channel_id, num_channels);
                FusedSymBuffer<uint64_t, false> rdma_channel_tail(rdma_cursor, 1, kNumRDMARanks, channel_id, num_channels);

                auto nvl_x = [=](void* base, int peer_slot) {
                    return static_cast<uint8_t*>(base) +
                           static_cast<size_t>(num_max_nvl_chunked_recv_tokens) * num_bytes_per_token *
                               (static_cast<size_t>(num_nvl_ranks) * channel_id + peer_slot);
                };
                auto nvl_head = [=](void* base, int peer_slot) {
                    return reinterpret_cast<int*>(static_cast<uint8_t*>(base) + nvl_head_off) +
                           static_cast<size_t>(kNumRDMARanks) * (static_cast<size_t>(num_nvl_ranks) * channel_id + peer_slot);
                };
                auto nvl_tail = [=](void* base, int peer_slot) {
                    return reinterpret_cast<int*>(static_cast<uint8_t*>(base) + nvl_tail_off) +
                           static_cast<size_t>(kNumRDMARanks) * (static_cast<size_t>(num_nvl_ranks) * channel_id + peer_slot);
                };

                // =====================================================================
                if (warp_role == kNVLSender) {
                    const int dst_nvl_rank = warp_id;
                    if (dst_nvl_rank >= num_nvl_ranks) return;  // deviation C2

                    void* dst_buffer_ptr = buffer_ptrs[dst_nvl_rank];
                    void* local_buffer_ptr = buffer_ptrs[nvl_rank];
                    uint8_t* ch_x = nvl_x(dst_buffer_ptr, nvl_rank);     // push into peer
                    int* ch_head = nvl_head(local_buffer_ptr, dst_nvl_rank);  // read locally
                    int* ch_tail = nvl_tail(dst_buffer_ptr, nvl_rank);   // push into peer

                    // CUDA :1849-1855
                    int token_start_idx = 0, token_end_idx = 0;
                    if (lane_id < kNumRDMARanks) {
                        const int prefix_idx = (lane_id * num_nvl_ranks + dst_nvl_rank) * num_channels + channel_id;
                        token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
                        token_end_idx = (prefix_idx == num_channels * num_ranks - 1)
                                            ? num_tokens
                                            : gbl_channel_prefix_matrix[prefix_idx + 1];
                    }
                    sycl::group_barrier(sg);

                    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
                    int current_rdma_idx = channel_id % kNumRDMARanks;
                    while (true) {
                        if (sycl::all_of_group(sg, token_start_idx >= token_end_idx)) break;

                        // CUDA :1868-1901: wait for a ready RDMA lane
                        bool is_lane_ready = false;
                        uint64_t spins = 0;
                        while (true) {
                            const int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                            is_lane_ready = lane_id < kNumRDMARanks && token_start_idx < token_end_idx &&
                                            num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >=
                                                num_max_nvl_chunked_send_tokens;
                            if (sycl::any_of_group(sg, is_lane_ready)) break;
                            if (lane_id < kNumRDMARanks && token_start_idx < token_end_idx)
                                cached_channel_head_idx = poll_load(ch_head + lane_id);
                            if (++spins > kFusedSpinCap) break;
                        }

                        // CUDA :1904-1946
                        for (int i = 0; i < kNumRDMARanks; ++i) {
                            current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks;
                            const int skip = sycl::select_from_group(
                                sg, ((token_start_idx >= token_end_idx) || (!is_lane_ready)) ? 1 : 0, current_rdma_idx);
                            if (skip) continue;

                            int64_t token_idx = sycl::select_from_group(sg, token_start_idx, current_rdma_idx);
                            const int num_tokens_in_chunk = sycl::select_from_group(
                                sg, sycl::min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx),
                                current_rdma_idx);

                            for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                                int dst_slot_idx = 0;
                                if (lane_id == current_rdma_idx) {
                                    dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                                    dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
                                }
                                dst_slot_idx = sycl::select_from_group(sg, dst_slot_idx, current_rdma_idx);

                                uint8_t* dst = ch_x + static_cast<size_t>(dst_slot_idx) * num_bytes_per_token;
                                const auto* src = static_cast<const int4_t*>(x) + token_idx * hidden_int4;
                                for (int j = lane_id; j < hidden_int4; j += 32)
                                    st_uc_global_v(reinterpret_cast<int4_t*>(dst) + j, ld_nc_global_v(src + j));
                                if (lane_id == num_topk)
                                    *reinterpret_cast<SourceMeta*>(dst + hidden_bytes) = src_meta[token_idx];
                                if (lane_id < num_topk)
                                    uc_store(reinterpret_cast<float*>(dst + hidden_bytes + sizeof(SourceMeta)) + lane_id,
                                             topk_weights[token_idx * num_topk + lane_id]);
                                sycl::group_barrier(sg);
                            }
                            if (lane_id == current_rdma_idx) token_start_idx = static_cast<int>(token_idx);
                        }

                        // CUDA :1949-1952: move queue tail (release)
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        lsc_fence_sysrel();
                        if (lane_id < kNumRDMARanks && is_lane_ready) uc_store(ch_tail + lane_id, cached_channel_tail_idx);
                        sycl::group_barrier(sg);
                    }
                    return;
                }

                // =====================================================================
                if (warp_role == kNVLAndRDMAForwarder) {
                    const int dst_rdma_rank = warp_id / Shape::kWarpsPerForwarder;
                    const int sub_warp_id = warp_id % Shape::kWarpsPerForwarder;
                    uint8_t* send_buffer = (dst_rdma_rank == rdma_rank) ? rdma_channel_data.recv_buffer(dst_rdma_rank)
                                                                        : rdma_channel_data.send_buffer(dst_rdma_rank);

                    void* local_buffer_ptr = buffer_ptrs[nvl_rank];
                    // nvl_channel_x/tail: LOCAL reads; advance by dst_rdma_rank (CUDA :1990-1992)
                    const size_t rdma_x_shift =
                        static_cast<size_t>(dst_rdma_rank) * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token;

                    // Clean SLM and sync (CUDA :1996-1999)
                    if (lane_id < NUM_MAX_NVL_PEERS) smem_fwd_nvl_head[warp_id * NUM_MAX_NVL_PEERS + lane_id] = 0;
                    if (lane_id == 0) smem_fwd_retired[warp_id] = 0;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    ::work_group_named_barrier(nb_fwd, 0x3);
#else
                    item.barrier(sycl::access::fence_space::global_and_local);
#endif

                    // CUDA :2002-2008
                    int cached_nvl_channel_tail_idx = 0;
                    int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
                    int num_tokens_prefix =
                        channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
                    num_tokens_to_combine -= num_tokens_prefix;
                    num_tokens_prefix += (dst_rdma_rank == 0) ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                    const int* my_combined_nvl_head =
                        combined_nvl_head + static_cast<int64_t>(num_tokens_prefix) * NUM_MAX_NVL_PEERS;

                    for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine;
                         token_start_idx += num_max_rdma_chunked_send_tokens) {
                        const int token_end_idx =
                            sycl::min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                        const int num_chunked_tokens = token_end_idx - token_start_idx;

                        // CUDA :2013-2040: wait for RDMA queue space
                        if (sub_warp_id == 0 && lane_id == 0) {
                            uint64_t spins = 0;
                            while (true) {
                                const int num_used_slots =
                                    token_start_idx - static_cast<int>(poll_load(rdma_channel_head.buffer(dst_rdma_rank)));
                                if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens) break;
                                if (++spins > kFusedSpinCap) break;
                            }
                        }
                        // CUDA :2041 sync_large_warp()
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                        if (Shape::kWarpsPerForwarder == 1) {
                            sycl::group_barrier(sg);
                        } else if (dst_rdma_rank == 0) {
                            ::work_group_named_barrier(nb_d0, 0x3);
                        } else if (dst_rdma_rank == 1) {
                            ::work_group_named_barrier(nb_d1, 0x3);
                        } else if (dst_rdma_rank == 2) {
                            ::work_group_named_barrier(nb_d2, 0x3);
                        } else if (dst_rdma_rank == 3) {
                            ::work_group_named_barrier(nb_d3, 0x3);
                        } else if (dst_rdma_rank == 4) {
                            ::work_group_named_barrier(nb_d4, 0x3);
                        } else if (dst_rdma_rank == 5) {
                            ::work_group_named_barrier(nb_d5, 0x3);
                        } else if (dst_rdma_rank == 6) {
                            ::work_group_named_barrier(nb_d6, 0x3);
                        } else {
                            ::work_group_named_barrier(nb_d7, 0x3);
                        }
#else
                        sycl::group_barrier(sg);
#endif

                        // CUDA :2044-2100: combine and write into the RDMA send buffer
                        for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx;
                             token_idx += Shape::kWarpsPerForwarder) {
                            int expected_head = -1;
                            if (lane_id < num_nvl_ranks) {
                                expected_head =
                                    my_combined_nvl_head[static_cast<int64_t>(token_idx) * NUM_MAX_NVL_PEERS + lane_id];
                                slm_store(smem_fwd_nvl_head, warp_id * NUM_MAX_NVL_PEERS + lane_id,
                                          expected_head < 0 ? (-expected_head - 1) : expected_head);
                            }

                            uint64_t spins = 0;
                            while (cached_nvl_channel_tail_idx <= expected_head) {
                                cached_nvl_channel_tail_idx = poll_load(nvl_tail(local_buffer_ptr, lane_id) + dst_rdma_rank);
                                if (++spins > kFusedSpinCap) break;
                            }
                            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                            lsc_fence_sysacq();

                            const int rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                            uint8_t* shifted = send_buffer + static_cast<size_t>(rdma_slot_idx) * num_bytes_per_token;
                            auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> const int4_t* {
                                return reinterpret_cast<const int4_t*>(
                                           nvl_x(local_buffer_ptr, src_nvl_rank) + rdma_x_shift +
                                           static_cast<size_t>(slot_idx) * num_bytes_per_token) +
                                       hidden_int4_idx;
                            };
                            auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int k) -> float {
                                return uc_load(reinterpret_cast<const float*>(
                                                   nvl_x(local_buffer_ptr, src_nvl_rank) + rdma_x_shift +
                                                   static_cast<size_t>(slot_idx) * num_bytes_per_token + hidden_bytes +
                                                   sizeof(SourceMeta)) +
                                               k);
                            };
                            fused_combine_token<NUM_MAX_NVL_PEERS, false, NUM_MAX_NVL_PEERS>(
                                expected_head >= 0, expected_head, lane_id, hidden_int4, num_topk,
                                reinterpret_cast<int4_t*>(shifted),
                                reinterpret_cast<float*>(shifted + hidden_bytes + sizeof(SourceMeta)), nullptr, nullptr,
                                num_max_nvl_chunked_recv_tokens_per_rdma, get_addr_fn, recv_tw_fn, sg);


                            if (lane_id < num_nvl_ranks)
                                slm_store(smem_fwd_nvl_head, warp_id * NUM_MAX_NVL_PEERS + lane_id,
                                          expected_head < 0 ? (-expected_head - 1) : (expected_head + 1));
                        }
                        // CUDA :2104 sync_large_warp()
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                        if (Shape::kWarpsPerForwarder == 1) {
                            sycl::group_barrier(sg);
                        } else if (dst_rdma_rank == 0) {
                            ::work_group_named_barrier(nb_d0, 0x3);
                        } else if (dst_rdma_rank == 1) {
                            ::work_group_named_barrier(nb_d1, 0x3);
                        } else if (dst_rdma_rank == 2) {
                            ::work_group_named_barrier(nb_d2, 0x3);
                        } else if (dst_rdma_rank == 3) {
                            ::work_group_named_barrier(nb_d3, 0x3);
                        } else if (dst_rdma_rank == 4) {
                            ::work_group_named_barrier(nb_d4, 0x3);
                        } else if (dst_rdma_rank == 5) {
                            ::work_group_named_barrier(nb_d5, 0x3);
                        } else if (dst_rdma_rank == 6) {
                            ::work_group_named_barrier(nb_d6, 0x3);
                        } else {
                            ::work_group_named_barrier(nb_d7, 0x3);
                        }
#else
                        sycl::group_barrier(sg);
#endif

                        // CUDA :2107-2140: issue the RDMA send + tail AMO
                        if (sub_warp_id == Shape::kWarpsPerForwarder - 1) {
                            const unsigned qp = static_cast<unsigned>(channel_id) % qps_per_pe;
                            const int dst_pe = dst_rdma_rank * num_nvl_ranks + nvl_rank;
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                            lsc_fence_sysrel();
                            if (dst_rdma_rank != rdma_rank) {
                                const int rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                                const size_t num_bytes_per_msg =
                                    static_cast<size_t>(num_chunked_tokens) * num_bytes_per_token;
                                uint8_t* dst_ptr = rdma_channel_data.recv_buffer(rdma_rank) +
                                                   static_cast<size_t>(rdma_slot_idx) * num_bytes_per_token;
                                uint8_t* src_ptr = rdma_channel_data.send_buffer(dst_rdma_rank) +
                                                   static_cast<size_t>(rdma_slot_idx) * num_bytes_per_token;
                                ishmemx_putmem_nbi_subgroup(dst_ptr, src_ptr, num_bytes_per_msg, dst_pe, qp, true, sg, true);
                                sycl::group_barrier(sg);
                                if (lane_id == 0) {
                                    ishmemx_fence_qp(dst_pe, qp);
                                    ishmemx_long_atomic_add_qp(
                                        reinterpret_cast<long*>(rdma_channel_tail.buffer(rdma_rank)),
                                        static_cast<long>(num_chunked_tokens), dst_pe, qp);
                                }
                            } else {
                                sycl::group_barrier(sg);
                                if (lane_id == 0) {
                                    sycl::atomic_ref<uint64_t, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                                     sycl::access::address_space::global_space>
                                        t(*rdma_channel_tail.buffer(rdma_rank));
                                    t.fetch_add(static_cast<uint64_t>(num_chunked_tokens));
                                }
                            }
                            sycl::group_barrier(sg);
                        }
                    }

                    sycl::group_barrier(sg);
                    if (lane_id == 0) {
                        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            r(smem_fwd_retired[warp_id]);
                        r.store(1);
                    }
                    return;
                }

                // =====================================================================
                if (warp_role == kRDMAReceiver) {
                    if (lane_id < kNumRDMARanks) smem_rcv_rdma_head[warp_id * kNumRDMARanks + lane_id] = 0;
                    if (lane_id == 0) smem_rcv_retired[warp_id] = 0;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    ::work_group_named_barrier(nb_rcv, 0x3);
#else
                    item.barrier(sycl::access::fence_space::global_and_local);
#endif

                    int token_start_idx = 0, token_end_idx = 0;
                    get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

                    int cached_channel_tail_idx = 0;
                    for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx;
                         token_idx += Shape::kRDMAReceivers) {
                        int expected_head = -1;
                        if (lane_id < kNumRDMARanks) {
                            expected_head = combined_rdma_head[token_idx * kNumRDMARanks + lane_id];
                            slm_store(smem_rcv_rdma_head, warp_id * kNumRDMARanks + lane_id,
                                      expected_head < 0 ? (-expected_head - 1) : expected_head);
                        }

                        uint64_t spins = 0;
                        while (cached_channel_tail_idx <= expected_head) {
                            cached_channel_tail_idx = static_cast<int>(poll_load(rdma_channel_tail.buffer(lane_id)));
                            if (++spins > kFusedSpinCap) break;
                        }
                        sycl::group_barrier(sg);
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();

                        auto get_addr_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> const int4_t* {
                            return reinterpret_cast<const int4_t*>(rdma_channel_data.recv_buffer(src_rdma_rank) +
                                                                   static_cast<size_t>(slot_idx) * num_bytes_per_token) +
                                   hidden_int4_idx;
                        };
                        auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int k) -> float {
                            return uc_load(reinterpret_cast<const float*>(
                                               rdma_channel_data.recv_buffer(src_rdma_rank) +
                                               static_cast<size_t>(slot_idx) * num_bytes_per_token + hidden_bytes +
                                               sizeof(SourceMeta)) +
                                           k);
                        };
                        fused_combine_token<kNumRDMARanks, true, kNumRDMARanks>(
                            expected_head >= 0, expected_head, lane_id, hidden_int4, num_topk,
                            static_cast<int4_t*>(combined_x) + token_idx * hidden_int4,
                            combined_topk_weights == nullptr ? nullptr : combined_topk_weights + token_idx * num_topk,
                            bias_0 == nullptr ? nullptr : static_cast<const int4_t*>(bias_0) + token_idx * hidden_int4,
                            bias_1 == nullptr ? nullptr : static_cast<const int4_t*>(bias_1) + token_idx * hidden_int4,
                            num_max_rdma_chunked_recv_tokens, get_addr_fn, recv_tw_fn, sg);
                    }

                    sycl::group_barrier(sg);
                    if (lane_id == 0) {
                        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            r(smem_rcv_retired[warp_id]);
                        r.store(1);
                    }
                    return;
                }

                // =====================================================================
                // Coordinator (CUDA :2216-2280)
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                if (is_forwarder_sm) {
                    ::work_group_named_barrier(nb_fwd, 0x3);
                } else {
                    ::work_group_named_barrier(nb_rcv, 0x3);
                }
#else
                item.barrier(sycl::access::fence_space::global_and_local);
#endif
                {
                    const int num_warps_per_rdma_rank = Shape::kForwarders / kNumRDMARanks;
                    int last_rdma_head = 0;
                    int last_nvl_head[kNumRDMARanks] = {0};
                    const int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
                    const int dst_nvl_rank = lane_id < num_nvl_ranks ? lane_id : 0;
                    uint64_t spins = 0;
                    while (true) {
                        if (!is_forwarder_sm) {
                            bool all_done = true;
                            for (int i = 0; i < Shape::kRDMAReceivers; ++i) {
                                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group,
                                                 sycl::access::address_space::local_space>
                                    r(smem_rcv_retired[i]);
                                if (r.load() == 0) all_done = false;
                            }
                            if (all_done) break;
                        } else {
                            bool all_done = true;
                            for (int i = 0; i < Shape::kForwarders; ++i) {
                                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group,
                                                 sycl::access::address_space::local_space>
                                    r(smem_fwd_retired[i]);
                                if (r.load() == 0) all_done = false;
                            }
                            if (all_done) break;
                        }

                        if (!is_forwarder_sm) {
                            int min_head = kFusedIntMax;
                            for (int i = 0; i < Shape::kRDMAReceivers; ++i) {
                                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group,
                                                 sycl::access::address_space::local_space>
                                    r(smem_rcv_retired[i]);
                                if (r.load() == 0) {
                                    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group,
                                                     sycl::access::address_space::local_space>
                                        h(smem_rcv_rdma_head[i * kNumRDMARanks + dst_rdma_rank]);
                                    min_head = sycl::min(min_head, h.load());
                                }
                            }
                            if (min_head != kFusedIntMax &&
                                min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens && lane_id < kNumRDMARanks) {
                                const int dst_pe = dst_rdma_rank * num_nvl_ranks + nvl_rank;
                                const unsigned qp =
                                    static_cast<unsigned>(channel_id + num_channels) % qps_per_pe;
                                if (dst_rdma_rank != rdma_rank) {
                                    ishmemx_long_atomic_add_qp(reinterpret_cast<long*>(rdma_channel_head.buffer(rdma_rank)),
                                                              static_cast<long>(min_head - last_rdma_head), dst_pe, qp);
                                } else {
                                    sycl::atomic_ref<uint64_t, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                                     sycl::access::address_space::global_space>
                                        h(*rdma_channel_head.buffer(rdma_rank));
                                    h.fetch_add(static_cast<uint64_t>(min_head - last_rdma_head));
                                }
                                last_rdma_head = min_head;
                            }
                        } else {
                            for (int i = 0; i < kNumRDMARanks; ++i) {
                                int min_head = kFusedIntMax;
                                for (int j = 0; j < num_warps_per_rdma_rank; ++j) {
                                    const int w = i * num_warps_per_rdma_rank + j;
                                    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group,
                                                     sycl::access::address_space::local_space>
                                        r(smem_fwd_retired[w]);
                                    if (r.load() == 0) {
                                        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group,
                                                         sycl::access::address_space::local_space>
                                            h(smem_fwd_nvl_head[w * NUM_MAX_NVL_PEERS + dst_nvl_rank]);
                                        min_head = sycl::min(min_head, h.load());
                                    }
                                }
                                if (min_head != kFusedIntMax && min_head > last_nvl_head[i] && lane_id < num_nvl_ranks) {
                                    last_nvl_head[i] = min_head;
                                    uc_store(nvl_head(buffer_ptrs[dst_nvl_rank], nvl_rank) + i, min_head);
                                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                                    lsc_fence_sysrel();
                                }
                            }
                        }
                        if (++spins > kFusedSpinCap) break;
                    }
                }
            });
    });
#endif
}
