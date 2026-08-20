// ============================================================================
// INTERNODE `notify_dispatch` (CUDA-faithful port) + `cached_notify` cleaning.
// ----------------------------------------------------------------------------
// Port of csrc/cuda_kernels/internode.cu:93 `notify_dispatch<...>` and the
// buffer-cleaning half of internode.cu:1311 `cached_notify<...>`.
//
// This is a SEPARATE kernel in CUDA too, and it legitimately uses device-wide /
// cross-PE barriers (`nvshmem_sync_with_same_gpu_idx`, `barrier_block`). The
// XPU port therefore keeps it outside the fused dispatch kernel and expresses
// every one of those barriers as a host-side `ishmem_barrier_all()` between
// queue submissions. `ishmem_barrier_all()` is a barrier over ALL PEs, so it
// subsumes both CUDA barriers (the RDMA-team sync and the NVL barrier_block)
// -- strictly stronger, never weaker.
//
// Outputs (consumed READ-ONLY by the fused dispatch, exactly as in CUDA
// internode.cu:461-466):
//   rdma_channel_prefix_matrix[num_rdma_ranks][num_channels]  (prefix-summed)
//   gbl_channel_prefix_matrix [num_ranks][num_channels]       (prefix-summed)
//   recv_rdma_rank_prefix_sum [num_rdma_ranks]
//   recv_gbl_rank_prefix_sum  [num_ranks]
//   *moe_recv_counter / *moe_recv_rdma_counter / moe_recv_expert_counter[]
//
// Deviations from CUDA:
//   N1. CUDA hard-codes NUM_MAX_NVL_PEERS (8) in every buffer stride and uses
//       the `uint64_t` reinterpretation of 8 `bool`s in is_token_in_rank. XPU
//       runs with num_nvl_ranks == 2, so all strides use the RUNTIME
//       num_nvl_ranks (matching the rest of csrc/xpu/internode*.cpp and
//       get_nvl_clean_meta) and the bool row is read element-wise.
//   N2. `nvshmemi_ibgda_quiet(pe, qp)` -> `ishmemx_fence_qp(pe, qp)`.
//   N3. The three barrier sites collapse to host-side ishmem_barrier_all(),
//       which also removes the need for a device-side barrier_block here.
//   N4. Both the RDMA and the NVL control-plane cleaning happen HERE (as in
//       CUDA), i.e. between two cross-PE barriers. That ordering matters: a
//       peer must not be able to push dispatch data into a region we are about
//       to zero.
// ============================================================================

#define DEEP_EP_NOTIFY_THREADS 512

template <int kNumRDMARanks>
class NotifyQuietKernel;
template <int kNumRDMARanks>
class NotifyRdmaExchangeKernel;
template <int kNumRDMARanks>
class NotifyNvlPushKernel;
template <int kNumRDMARanks>
class NotifyReduceKernel;
template <int kNumRDMARanks>
class NotifyMetaKernel;
template <int kNumRDMARanks>
class CachedNotifyCleanKernel;

// CUDA internode.cu:48
inline std::pair<int, int> fused_get_rdma_clean_meta(int hidden_int4,
                                                     int num_scales,
                                                     int num_topk,
                                                     int num_rdma_ranks,
                                                     int num_nvl_ranks,
                                                     int num_rdma_recv_buffer_tokens,
                                                     int num_channels) {
    return {static_cast<int>(static_cast<int64_t>(fused_num_bytes_per_token(hidden_int4, num_scales, num_topk)) *
                             num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_channels / sizeof(int)),
            (num_nvl_ranks * 2 + 4) * num_rdma_ranks * 2 * num_channels};
}

// CUDA internode.cu:62
inline std::pair<int, int> fused_get_nvl_clean_meta(int hidden_int4,
                                                    int num_scales,
                                                    int num_topk,
                                                    int num_rdma_ranks,
                                                    int num_nvl_ranks,
                                                    int num_nvl_recv_buffer_tokens,
                                                    int num_channels) {
    return {static_cast<int>(static_cast<int64_t>(num_nvl_recv_buffer_tokens) *
                             fused_num_bytes_per_token(hidden_int4, num_scales, num_topk) * num_nvl_ranks * num_channels /
                             sizeof(int)),
            num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels};
}

// ---------------------------------------------------------------------------
// Cleaning of both control planes, ordered between two cross-PE barriers.
// Shared by notify_dispatch and cached_notify (CUDA does the same work in both,
// internode.cu:150-166 / :1327-1345).
// ---------------------------------------------------------------------------
template <int kNumRDMARanks>
void launch_fused_clean_planes(void* rdma_buffer_ptr,
                               void** buffer_ptrs_gpu,
                               int nvl_rank,
                               int rdma_clean_offset,
                               int rdma_num_int_clean,
                               int nvl_clean_offset,
                               int nvl_num_int_clean,
                               sycl::queue& queue) {
    const int n = rdma_num_int_clean > nvl_num_int_clean ? rdma_num_int_clean : nvl_num_int_clean;
    const size_t kWG = 256;
    const size_t global = ((static_cast<size_t>(n) + kWG - 1) / kWG) * kWG;
    if (global == 0) return;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CachedNotifyCleanKernel<kNumRDMARanks>>(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWG)), [=](sycl::nd_item<1> item) {
                const int i = static_cast<int>(item.get_global_linear_id());
                if (i < rdma_num_int_clean) uc_store(static_cast<int*>(rdma_buffer_ptr) + rdma_clean_offset + i, 0);
                if (i < nvl_num_int_clean)
                    uc_store(static_cast<int*>(buffer_ptrs_gpu[nvl_rank]) + nvl_clean_offset + i, 0);
            });
    });
}

// ---------------------------------------------------------------------------
// The metadata half of CUDA notify_dispatch (`sm_id > 0` branch,
// internode.cu:283-345). Purely local: no communication at all.
// ---------------------------------------------------------------------------
template <int kNumRDMARanks>
void launch_fused_notify_meta(const bool* is_token_in_rank,
                              int* rdma_channel_prefix_matrix,
                              int* gbl_channel_prefix_matrix,
                              int num_tokens,
                              int num_ranks,
                              int num_nvl_ranks,
                              int num_channels,
                              sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NotifyMetaKernel<kNumRDMARanks>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(kNumRDMARanks) * DEEP_EP_NOTIFY_THREADS), sycl::range<1>(DEEP_EP_NOTIFY_THREADS)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                auto sg = item.get_sub_group();
                const int dst_rdma_rank = static_cast<int>(item.get_group(0));
                const int thread_id = static_cast<int>(item.get_local_id(0));
                const int warp_id = static_cast<int>(sg.get_group_linear_id());
                const int lane_id = static_cast<int>(sg.get_local_linear_id());
                const int num_warps = DEEP_EP_NOTIFY_THREADS / 32;

                for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
                    int token_start_idx = 0, token_end_idx = 0;
                    get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

                    int total_count = 0;
                    int per_nvl_rank_count[NUM_MAX_NVL_PEERS] = {0};
                    for (int i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
                        // Deviation N1: num_nvl_ranks may be < 8, so no uint64 trick.
                        bool any = false;
                        for (int j = 0; j < num_nvl_ranks; ++j) {
                            const bool v = is_token_in_rank[static_cast<int64_t>(i) * num_ranks +
                                                           dst_rdma_rank * num_nvl_ranks + j];
                            per_nvl_rank_count[j] += v ? 1 : 0;
                            any |= v;
                        }
                        total_count += any ? 1 : 0;
                    }

                    total_count = sycl::reduce_over_group(sg, total_count, sycl::plus<int>());
                    for (int i = 0; i < num_nvl_ranks; ++i)
                        per_nvl_rank_count[i] = sycl::reduce_over_group(sg, per_nvl_rank_count[i], sycl::plus<int>());

                    if (lane_id == 0) {
                        for (int i = 0; i < num_nvl_ranks; ++i)
                            gbl_channel_prefix_matrix[(dst_rdma_rank * num_nvl_ranks + i) * num_channels + channel_id] =
                                per_nvl_rank_count[i];
                        rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = total_count;
                    }
                }

                // Prefix sums (CUDA internode.cu:328-344)
                item.barrier(sycl::access::fence_space::global_and_local);
                if (thread_id == 0) {
                    int* row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;
                    for (int i = 1; i < num_channels; ++i) row[i] += row[i - 1];
                }
                if (thread_id < num_nvl_ranks) {
                    int* row = gbl_channel_prefix_matrix + (dst_rdma_rank * num_nvl_ranks + thread_id) * num_channels;
                    for (int i = 1; i < num_channels; ++i) row[i] += row[i - 1];
                }
            });
    });
}

// ---------------------------------------------------------------------------
// The communication half of CUDA notify_dispatch (`sm_id == 0` branch,
// internode.cu:130-281), split at every cross-PE barrier.
// ---------------------------------------------------------------------------
template <int kNumRDMARanks>
void launch_fused_notify_dispatch(const int* num_tokens_per_rank,
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
                                  int expert_alignment,
                                  int hidden_int4,
                                  int num_scales,
                                  int num_topk,
                                  int num_max_rdma_chunked_recv_tokens,
                                  int num_max_nvl_chunked_recv_tokens,
                                  int* rdma_channel_prefix_matrix,
                                  int* recv_rdma_rank_prefix_sum,
                                  int* gbl_channel_prefix_matrix,
                                  int* recv_gbl_rank_prefix_sum,
                                  void* rdma_buffer_ptr,
                                  void** buffer_ptrs_gpu,
                                  int rank,
                                  int num_ranks,
                                  int num_nvl_ranks,
                                  sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    const int rdma_rank = rank / num_nvl_ranks;
    const int nvl_rank = rank % num_nvl_ranks;
    const int num_rdma_experts = num_experts / kNumRDMARanks;
    const int num_nvl_experts = num_rdma_experts / num_nvl_ranks;
    const int mixed = num_nvl_ranks + num_rdma_experts + 1;
    const unsigned qps_per_pe = fused_qps_per_pe();

    const auto rdma_clean =
        fused_get_rdma_clean_meta(hidden_int4, num_scales, num_topk, kNumRDMARanks, num_nvl_ranks,
                                  num_max_rdma_chunked_recv_tokens, num_channels);
    const auto nvl_clean =
        fused_get_nvl_clean_meta(hidden_int4, num_scales, num_topk, kNumRDMARanks, num_nvl_ranks,
                                 num_max_nvl_chunked_recv_tokens, num_channels);

    // The purely local metadata computation has no ordering constraint at all
    // (CUDA runs it in parallel SMs of the same kernel), so submit it first.
    launch_fused_notify_meta<kNumRDMARanks>(is_token_in_rank, rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
                                            num_tokens, num_ranks, num_nvl_ranks, num_channels, queue);

    // --- CUDA :138-146: drain every in-flight WR before we rewrite the buffer ---
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NotifyQuietKernel<kNumRDMARanks>>(
            sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const int lane = static_cast<int>(item.get_local_id(0));
                for (int i = lane; i < static_cast<int>(qps_per_pe) * (kNumRDMARanks - 1); i += 32) {
                    const int dst_rdma_rank = (i / static_cast<int>(qps_per_pe) + rdma_rank + 1) % kNumRDMARanks;
                    ishmemx_fence_qp(dst_rdma_rank * num_nvl_ranks + nvl_rank, static_cast<unsigned>(i) % qps_per_pe);
                }
            });
    });
    queue.wait();
    ishmem_barrier_all();  // CUDA :148-151

    // --- CUDA :153-201: clean, fill send buffers, exchange counts over RDMA ---
    launch_fused_clean_planes<kNumRDMARanks>(rdma_buffer_ptr, buffer_ptrs_gpu, nvl_rank, rdma_clean.first,
                                             rdma_clean.second, nvl_clean.first, nvl_clean.second, queue);
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NotifyRdmaExchangeKernel<kNumRDMARanks>>(
            sycl::nd_range<1>(sycl::range<1>(DEEP_EP_NOTIFY_THREADS), sycl::range<1>(DEEP_EP_NOTIFY_THREADS)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                auto sg = item.get_sub_group();
                const int thread_id = static_cast<int>(item.get_local_id(0));
                const int warp_id = static_cast<int>(sg.get_group_linear_id());
                const int num_warps = DEEP_EP_NOTIFY_THREADS / 32;
                int* base = static_cast<int*>(rdma_buffer_ptr);
                // SymBuffer<int>(rdma_buffer_ptr, mixed, kNumRDMARanks)
                auto send_buffer = [=](int i) { return base + static_cast<int64_t>(i) * mixed; };
                auto recv_buffer = [=](int i) { return base + static_cast<int64_t>(kNumRDMARanks + i) * mixed; };

                // CUDA :168-176
                for (int i = thread_id; i < num_ranks; i += DEEP_EP_NOTIFY_THREADS)
                    send_buffer(i / num_nvl_ranks)[i % num_nvl_ranks] = num_tokens_per_rank[i];
                for (int i = thread_id; i < num_experts; i += DEEP_EP_NOTIFY_THREADS)
                    send_buffer(i / num_rdma_experts)[num_nvl_ranks + i % num_rdma_experts] = num_tokens_per_expert[i];
                if (thread_id < kNumRDMARanks)
                    send_buffer(thread_id)[num_nvl_ranks + num_rdma_experts] = num_tokens_per_rdma_rank[thread_id];
                item.barrier(sycl::access::fence_space::global_and_local);

                // CUDA :181-201
                for (int i = warp_id; i < kNumRDMARanks; i += num_warps) {
                    if (i != rdma_rank) {
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        lsc_fence_sysrel();
                        ishmemx_putmem_nbi_subgroup(recv_buffer(rdma_rank), send_buffer(i), sizeof(int) * mixed,
                                                    i * num_nvl_ranks + nvl_rank, 0u, true, sg, true);
                        sycl::group_barrier(sg);
                    } else {
                        for (int j = static_cast<int>(sg.get_local_linear_id()); j < mixed; j += 32)
                            uc_store(recv_buffer(rdma_rank) + j, uc_load(send_buffer(i) + j));
                    }
                }
                item.barrier(sycl::access::fence_space::global_and_local);

                // CUDA :204-206
                if (thread_id < kNumRDMARanks && thread_id != rdma_rank)
                    ishmemx_fence_qp(thread_id * num_nvl_ranks + nvl_rank, 0u);
            });
    });
    queue.wait();
    ishmem_barrier_all();  // CUDA :210-213

    // --- CUDA :216-256: reduce RDMA results, push per-rank/expert counts over NVL ---
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NotifyNvlPushKernel<kNumRDMARanks>>(
            sycl::nd_range<1>(sycl::range<1>(DEEP_EP_NOTIFY_THREADS), sycl::range<1>(DEEP_EP_NOTIFY_THREADS)),
            [=](sycl::nd_item<1> item) {
                const int thread_id = static_cast<int>(item.get_local_id(0));
                int* rdma_base = static_cast<int*>(rdma_buffer_ptr);
                auto recv_buffer = [=](int i) { return rdma_base + static_cast<int64_t>(kNumRDMARanks + i) * mixed; };

                // NVL scratch layout (CUDA :219-224), strides in num_nvl_ranks (dev N1):
                //   [0                            , num_rdma_experts)          reduced-per-expert
                //   [num_rdma_experts             , + kNumRDMARanks*num_nvl_ranks)   per-rank
                //   [..                           , + num_nvl_experts*num_nvl_ranks) per-expert
                const int off_rank = num_rdma_experts;
                const int off_expert = off_rank + kNumRDMARanks * num_nvl_ranks;
                int* my_nvl = static_cast<int*>(buffer_ptrs_gpu[nvl_rank]);

                // CUDA :240-247
                if (thread_id < num_rdma_experts) {
                    int sum = 0;
                    for (int i = 0; i < kNumRDMARanks; ++i) sum += uc_load(recv_buffer(i) + num_nvl_ranks + thread_id);
                    my_nvl[thread_id] = sum;
                }
                // CUDA :250-262
                if (thread_id == 0) {
                    int sum = 0;
                    for (int i = 0; i < kNumRDMARanks; ++i) {
                        sum += uc_load(recv_buffer(i) + num_nvl_ranks + num_rdma_experts);
                        recv_rdma_rank_prefix_sum[i] = sum;
                    }
                    if (num_worst_tokens == 0 && moe_recv_rdma_counter_mapped != nullptr)
                        uc_store(moe_recv_rdma_counter_mapped, sum);
                }
                item.barrier(sycl::access::fence_space::global_and_local);

                // CUDA :265-272 -- PUSH into every peer's NVL buffer.
                if (thread_id < num_nvl_ranks) {
                    int* peer = static_cast<int*>(buffer_ptrs_gpu[thread_id]);
                    for (int i = 0; i < kNumRDMARanks; ++i)
                        uc_store(peer + off_rank + nvl_rank * kNumRDMARanks + i,
                                 uc_load(recv_buffer(i) + thread_id));
                    for (int i = 0; i < num_nvl_experts; ++i)
                        uc_store(peer + off_expert + nvl_rank * num_nvl_experts + i,
                                 my_nvl[thread_id * num_nvl_experts + i]);
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    lsc_fence_sysrel();
                }
            });
    });
    queue.wait();
    ishmem_barrier_all();  // CUDA :273 barrier_block<NUM_MAX_NVL_PEERS>

    // --- CUDA :276-296: reduce the NVL results ---
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NotifyReduceKernel<kNumRDMARanks>>(
            sycl::nd_range<1>(sycl::range<1>(DEEP_EP_NOTIFY_THREADS), sycl::range<1>(DEEP_EP_NOTIFY_THREADS)),
            [=](sycl::nd_item<1> item) {
                const int thread_id = static_cast<int>(item.get_local_id(0));
                const int off_rank = num_rdma_experts;
                const int off_expert = off_rank + kNumRDMARanks * num_nvl_ranks;
                int* my_nvl = static_cast<int*>(buffer_ptrs_gpu[nvl_rank]);
                sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                lsc_fence_sysacq();

                if (thread_id == 0) {
                    int sum = 0;
                    for (int i = 0; i < num_ranks; ++i) {
                        const int src_rdma_rank = i / num_nvl_ranks, src_nvl_rank = i % num_nvl_ranks;
                        sum += uc_load(my_nvl + off_rank + src_nvl_rank * kNumRDMARanks + src_rdma_rank);
                        recv_gbl_rank_prefix_sum[i] = sum;
                    }
                    if (num_worst_tokens == 0 && moe_recv_counter_mapped != nullptr)
                        uc_store(moe_recv_counter_mapped, sum);
                }
                if (thread_id < num_nvl_experts) {
                    int sum = 0;
                    for (int i = 0; i < num_nvl_ranks; ++i)
                        sum += uc_load(my_nvl + off_expert + i * num_nvl_experts + thread_id);
                    sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
                    if (num_worst_tokens == 0 && moe_recv_expert_counter_mapped != nullptr)
                        uc_store(moe_recv_expert_counter_mapped + thread_id, sum);
                }
            });
    });
    queue.wait();
    ishmem_barrier_all();  // CUDA :299-301 final barrier
#endif
}
