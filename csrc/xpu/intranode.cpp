#include <limits>

#include "xpu_kernels.hpp"

namespace deep_ep {
namespace intranode {
namespace {

template <int kNumRanks>
class NotifyDispatchKernel;
template <int kNumRanks>
class CachedNotifyDispatchKernel;
template <int kNumRanks>
class DispatchKernel;
template <int kNumRanks>
class CachedNotifyCombineKernel;
template <typename dtype_t, int kNumRanks>
class CombineKernel;
template <int kNumRanks>
class BarrierKernel;

template <int kNumRanks>
void launch_notify_dispatch(const int* num_tokens_per_rank,
                            int* moe_recv_counter_mapped,
                            const int* num_tokens_per_expert,
                            int* moe_recv_expert_counter_mapped,
                            int num_experts,
                            int num_tokens,
                            int num_channels,
                            const bool* is_token_in_rank,
                            int* channel_prefix_matrix,
                            int* rank_prefix_matrix_copy,
                            int num_memset_int,
                            int expert_alignment,
                            void** buffer_ptrs,
                            int** barrier_signal_ptrs,
                            int rank,
                            int barrier_signal_base,
                            sycl::queue& queue) {
    constexpr int kThreads = 128;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NotifyDispatchKernel<kNumRanks>>(
            sycl::nd_range<1>(sycl::range<1>((1 + kNumRanks) * kThreads), sycl::range<1>(kThreads)), [=](sycl::nd_item<1> item) {
                const int sm_id = static_cast<int>(item.get_group(0));
                const int tid = static_cast<int>(item.get_local_id(0));
                const int num_threads = static_cast<int>(item.get_local_range(0));
                const int lid = tid % 32;
                const int warp_id = tid / 32;
                const int num_warps = num_threads / 32;

                if (sm_id == 0) {
                    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank, barrier_signal_base, item);

                    int num_experts_per_rank = num_experts / kNumRanks;
                    if (tid < kNumRanks) {
                        auto per_rank_buffer = static_cast<int*>(buffer_ptrs[tid]);
                        auto per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
                        per_rank_buffer[rank * kNumRanks + tid] = num_tokens_per_rank[tid];
                        for (int i = 0; i < num_experts_per_rank; ++i) {
                            per_expert_buffer[rank * num_experts_per_rank + i] = num_tokens_per_expert[tid * num_experts_per_rank + i];
                        }
                    }

                    barrier_block<kNumRanks>(barrier_signal_ptrs, rank, barrier_signal_base + 1, item);

                    auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);
                    if (tid < kNumRanks) {
                        for (int i = 1; i < kNumRanks; ++i) {
                            local_per_rank_buffer[i * kNumRanks + tid] += local_per_rank_buffer[(i - 1) * kNumRanks + tid];
                        }
                        if (tid == rank) {
                            *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
                        }
                    }

                    auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
                    if (tid < num_experts_per_rank) {
                        int sum = 0;
                        for (int i = 0; i < kNumRanks; ++i) {
                            sum += local_per_expert_buffer[i * num_experts_per_rank + tid];
                        }
                        sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
                        moe_recv_expert_counter_mapped[tid] = sum;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int i = tid; i < kNumRanks * kNumRanks; i += num_threads) {
                        rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];
                    }
                    for (int i = tid; i < num_memset_int; i += num_threads) {
                        local_per_expert_buffer[i] = 0;
                    }

                    barrier_block<kNumRanks>(barrier_signal_ptrs, rank, barrier_signal_base + 2, item);
                } else {
                    int dst_rank = sm_id - 1;
                    for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
                        int token_start_idx = 0;
                        int token_end_idx = 0;
                        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);
                        int count = 0;
                        for (int64_t i = token_start_idx + lid; i < token_end_idx; i += 32) {
                            count += is_token_in_rank[i * kNumRanks + dst_rank] ? 1 : 0;
                        }
                        count = subgroup_reduce_sum(count, item);
                        if (elect_one(item)) {
                            channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    if (tid == 0) {
                        for (int i = 1; i < num_channels; ++i) {
                            channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];
                        }
                    }
                }
            });
    });
}

template <int kNumRanks>
void launch_cached_notify_dispatch(const int* rank_prefix_matrix,
                                   int num_memset_int,
                                   void** buffer_ptrs,
                                   int** barrier_signal_ptrs,
                                   int rank,
                                   int barrier_signal_base,
                                   sycl::queue& queue) {
    constexpr int kThreads = 128;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CachedNotifyDispatchKernel<kNumRanks>>(
            sycl::nd_range<1>(sycl::range<1>(kThreads), sycl::range<1>(kThreads)), [=](sycl::nd_item<1> item) {
                int tid = static_cast<int>(item.get_local_id(0));
                int num_threads = static_cast<int>(item.get_local_range(0));
                barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank, barrier_signal_base, item);
                auto ptr = static_cast<int*>(buffer_ptrs[rank]);
                for (int i = tid; i < kNumRanks * kNumRanks; i += num_threads) {
                    ptr[i] = rank_prefix_matrix[i];
                }
                for (int i = tid; i < num_memset_int; i += num_threads) {
                    ptr[kNumRanks * kNumRanks + i] = 0;
                }
                barrier_block<kNumRanks>(barrier_signal_ptrs, rank, barrier_signal_base + 1, item);
            });
    });
}

template <int kNumRanks>
void launch_dispatch(void* recv_x,
                     float* recv_x_scales,
                     int* recv_src_idx,
                     topk_idx_t* recv_topk_idx,
                     float* recv_topk_weights,
                     int* recv_channel_offset,
                     int* send_head,
                     const void* x,
                     const float* x_scales,
                     const topk_idx_t* topk_idx,
                     const float* topk_weights,
                     const bool* is_token_in_rank,
                     const int* channel_prefix_matrix,
                     int num_tokens,
                     int hidden_int4,
                     int num_topk,
                     int num_experts,
                     int num_scales,
                     int scale_token_stride,
                     int scale_hidden_stride,
                     void** buffer_ptrs,
                     int rank,
                     int num_sms,
                     int num_max_send_tokens,
                     int num_recv_buffer_tokens,
                     sycl::queue& queue) {
    constexpr int kThreads = kNumRanks * 32;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<DispatchKernel<kNumRanks>>(
            sycl::nd_range<1>(sycl::range<1>(num_sms * kThreads), sycl::range<1>(kThreads)), [=](sycl::nd_item<1> item) {
                int num_channels = static_cast<int>(item.get_group_range(0)) / 2;
                int sm_id = static_cast<int>(item.get_group(0));
                int tid = static_cast<int>(item.get_local_id(0));
                int lid = lane_id(item);
                bool is_sender = sm_id % 2 == 0;
                int responsible_rank = tid / 32;
                int responsible_channel = sm_id / 2;
                int num_experts_per_rank = num_experts == 0 ? 0 : num_experts / kNumRanks;
                int num_channels_total = num_channels * kNumRanks;
                int target_rank = is_sender ? rank : responsible_rank;
                int channel_rank_offset = responsible_channel * kNumRanks + target_rank;
                void* ptr = static_cast<uint8_t*>(buffer_ptrs[is_sender ? responsible_rank : rank]) + kNumRanks * kNumRanks * sizeof(int);

                DeviceBuffer<int> channel_start_offset(ptr, num_channels_total, channel_rank_offset);
                DeviceBuffer<int> channel_end_offset(ptr, num_channels_total, channel_rank_offset);
                DeviceBuffer<int> channel_head_idx(ptr, num_channels_total, channel_rank_offset);
                DeviceBuffer<int> channel_tail_idx(ptr, num_channels_total, channel_rank_offset);
                DeviceBuffer<int4> channel_x_buffers(ptr,
                                                     static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens * hidden_int4,
                                                     static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens * hidden_int4);
                DeviceBuffer<int> channel_src_idx_buffers(ptr,
                                                          static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens,
                                                          static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens);
                DeviceBuffer<topk_idx_t> channel_topk_idx_buffers(
                    ptr,
                    static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens * num_topk,
                    static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens * num_topk);
                DeviceBuffer<float> channel_topk_weights_buffers(
                    ptr,
                    static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens * num_topk,
                    static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens * num_topk);
                DeviceBuffer<float> channel_x_scales_buffers(
                    ptr,
                    static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens * num_scales,
                    static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens * num_scales);

                if (is_sender) {
                    if (elect_one(item)) {
                        int value =
                            responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;
                        st_global(channel_start_offset.buffer(), -value - 1);
                        value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
                        st_global(channel_end_offset.buffer(), -value - 1);
                    }
                    sycl::group_barrier(item.get_sub_group());

                    int token_start_idx = 0;
                    int token_end_idx = 0;
                    get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);
                    int cached_tail = 0;
                    for (int token_idx = token_start_idx; token_idx < token_end_idx;) {
                        if (elect_one(item)) {
                            while (num_recv_buffer_tokens - (cached_tail - ld_global(channel_head_idx.buffer())) < num_max_send_tokens) {
                                visa_spin_hint();
                            }
                        }
                        sycl::group_barrier(item.get_sub_group());
                        int chunk = 0;
                        while (chunk < num_max_send_tokens && token_idx < token_end_idx) {
                            if (elect_one(item)) {
                                send_head[token_idx * kNumRanks + responsible_rank] =
                                    is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_tail : -1;
                            }
                            if (!is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                                ++token_idx;
                                continue;
                            }
                            int slot = (cached_tail++) % num_recv_buffer_tokens;
                            auto dst_x = channel_x_buffers.buffer() + slot * hidden_int4;
                            auto src_x = reinterpret_cast<const int4*>(x) + static_cast<int64_t>(token_idx) * hidden_int4;
                            for (int i = lid; i < hidden_int4; i += 32) {
                                st_global(dst_x + i, ld_global(src_x + i));
                            }
                            if (elect_one(item)) {
                                st_global(channel_src_idx_buffers.buffer() + slot, token_idx);
                            }
                            if (topk_idx != nullptr && lid < num_topk) {
                                int begin = responsible_rank * num_experts_per_rank;
                                int end = begin + num_experts_per_rank;
                                auto idx = topk_idx[token_idx * num_topk + lid];
                                idx = (idx >= begin && idx < end) ? idx - begin : -1;
                                st_global(channel_topk_idx_buffers.buffer() + slot * num_topk + lid, idx);
                                float weight = idx >= 0 ? topk_weights[token_idx * num_topk + lid] : 0.0f;
                                st_global(channel_topk_weights_buffers.buffer() + slot * num_topk + lid, weight);
                            }
                            if (x_scales != nullptr) {
                                for (int i = lid; i < num_scales; i += 32) {
                                    st_global(channel_x_scales_buffers.buffer() + slot * num_scales + i,
                                              x_scales[token_idx * scale_token_stride + i * scale_hidden_stride]);
                                }
                            }
                            ++chunk;
                            ++token_idx;
                        }
                        sycl::group_barrier(item.get_sub_group());
                        if (elect_one(item)) {
                            st_global(channel_tail_idx.buffer(), cached_tail);
                        }
                    }
                } else {
                    auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
                    int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;
                    int total_offset = 0;
                    int num_to_recv = 0;
                    if (elect_one(item)) {
                        while ((total_offset = ld_global(channel_start_offset.buffer())) == 0) {
                            visa_spin_hint();
                        }
                        while ((num_to_recv = ld_global(channel_end_offset.buffer())) == 0) {
                            visa_spin_hint();
                        }
                        total_offset = -total_offset - 1;
                        num_to_recv = -num_to_recv - 1;
                        recv_channel_offset[responsible_rank * num_channels + responsible_channel] = total_offset;
                        num_to_recv -= total_offset;
                    }
                    total_offset = subgroup_broadcast(total_offset, 0, item) + rank_offset;
                    num_to_recv = subgroup_broadcast(num_to_recv, 0, item);
                    int head = 0;
                    int tail = 0;
                    while (num_to_recv > 0) {
                        if (elect_one(item)) {
                            while ((tail = ld_global(channel_tail_idx.buffer())) == head) {
                                visa_spin_hint();
                            }
                        }
                        tail = subgroup_broadcast(tail, 0, item);
                        int count = tail - head;
                        for (int chunk_idx = 0; chunk_idx < count; ++chunk_idx) {
                            int slot = (head + chunk_idx) % num_recv_buffer_tokens;
                            auto src_x = channel_x_buffers.buffer() + slot * hidden_int4;
                            auto dst_x = reinterpret_cast<int4*>(recv_x) + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
                            for (int i = lid; i < hidden_int4; i += 32) {
                                st_global(dst_x + i, ld_global(src_x + i));
                            }
                        }
                        for (int chunk_idx = head + lid; chunk_idx < tail; chunk_idx += 32) {
                            recv_src_idx[total_offset + chunk_idx - head] =
                                ld_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);
                        }
                        if (recv_topk_idx != nullptr) {
                            for (int i = lid; i < count * num_topk; i += 32) {
                                int chunk_idx = i / num_topk;
                                int topk = i % num_topk;
                                int slot = (head + chunk_idx) % num_recv_buffer_tokens;
                                int64_t recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + topk;
                                recv_topk_idx[recv_idx] = ld_global(channel_topk_idx_buffers.buffer() + slot * num_topk + topk);
                                recv_topk_weights[recv_idx] = ld_global(channel_topk_weights_buffers.buffer() + slot * num_topk + topk);
                            }
                        }
                        if (recv_x_scales != nullptr) {
                            for (int i = lid; i < count * num_scales; i += 32) {
                                int chunk_idx = i / num_scales;
                                int scale_idx = i % num_scales;
                                int slot = (head + chunk_idx) % num_recv_buffer_tokens;
                                recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scale_idx] =
                                    ld_global(channel_x_scales_buffers.buffer() + slot * num_scales + scale_idx);
                            }
                        }
                        head += count;
                        total_offset += count;
                        sycl::group_barrier(item.get_sub_group());
                        if (elect_one(item)) {
                            st_global(channel_head_idx.buffer(), head);
                        }
                        num_to_recv -= count;
                    }
                }
            });
    });
}

template <int kNumRanks>
void launch_cached_notify_combine(void** buffer_ptrs,
                                  int* send_head,
                                  int num_channels,
                                  int num_recv_tokens,
                                  int num_memset_int,
                                  int** barrier_signal_ptrs,
                                  int rank,
                                  int barrier_signal_base,
                                  sycl::queue& queue) {
    int num_threads = sycl::max(128, 32 * kNumRanks);
    int num_blocks = 1 + num_channels;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CachedNotifyCombineKernel<kNumRanks>>(
            sycl::nd_range<1>(sycl::range<1>(num_blocks * num_threads), sycl::range<1>(num_threads)), [=](sycl::nd_item<1> item) {
                int sm_id = static_cast<int>(item.get_group(0));
                int tid = static_cast<int>(item.get_local_id(0));
                int num_threads_local = static_cast<int>(item.get_local_range(0));
                if (sm_id == 0) {
                    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank, barrier_signal_base, item);
                    auto ptr = static_cast<int*>(buffer_ptrs[rank]);
                    for (int i = tid; i < num_memset_int; i += num_threads_local) {
                        ptr[i] = 0;
                    }
                    barrier_block<kNumRanks>(barrier_signal_ptrs, rank, barrier_signal_base + 1, item);
                } else {
                    int channel_id = sm_id - 1;
                    int rank_id = tid / 32;
                    int lid = tid % 32;
                    if (rank_id >= kNumRanks) {
                        return;
                    }
                    int token_start_idx = 0;
                    int token_end_idx = 0;
                    get_channel_task_range(num_recv_tokens, num_channels, channel_id, token_start_idx, token_end_idx);
                    int last_head = 1 << 25;
                    for (int tail = token_end_idx - 1; tail >= token_start_idx; tail -= 32) {
                        int token_idx = tail - lid;
                        int current = token_idx >= token_start_idx ? ld_global(send_head + token_idx * kNumRanks + rank_id) : -1;
                        int expected = 0;
                        int num_iters = sycl::min(32, tail - token_start_idx + 1);
                        for (int i = 0; i < num_iters; ++i) {
                            int head = subgroup_broadcast(current, i, item);
                            if (head < 0) {
                                if (lid == i) {
                                    expected = -last_head - 1;
                                }
                            } else {
                                last_head = head;
                            }
                        }
                        if (current < 0 && token_idx >= token_start_idx) {
                            send_head[token_idx * kNumRanks + rank_id] = expected;
                        }
                    }
                }
            });
    });
}

template <typename dtype_t, int kNumRanks>
void launch_combine(DataType,
                    void* recv_x,
                    float* recv_topk_weights,
                    const void* x,
                    const float* topk_weights,
                    const void* bias_0,
                    const void* bias_1,
                    const int* src_idx,
                    const int* rank_prefix_matrix,
                    const int* channel_prefix_matrix,
                    int* send_head,
                    int num_tokens,
                    int num_recv_tokens,
                    int hidden,
                    int num_topk,
                    void** buffer_ptrs,
                    int rank,
                    int num_sms,
                    int num_max_send_tokens,
                    int num_recv_buffer_tokens,
                    sycl::queue& queue) {
    constexpr int kThreads = kNumRanks * 32;
    const int hidden_int4 = hidden * static_cast<int>(sizeof(dtype_t)) / static_cast<int>(sizeof(int4));
    const int num_channels = num_sms / 2;
    auto recv = static_cast<dtype_t*>(recv_x);
    auto src = static_cast<const dtype_t*>(x);
    auto b0 = static_cast<const dtype_t*>(bias_0);
    auto b1 = static_cast<const dtype_t*>(bias_1);
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombineKernel<dtype_t, kNumRanks>>(
            sycl::nd_range<1>(sycl::range<1>(num_sms * kThreads), sycl::range<1>(kThreads)), [=](sycl::nd_item<1> item) {
                int sm_id = static_cast<int>(item.get_group(0));
                int tid = static_cast<int>(item.get_local_id(0));
                int lid = lane_id(item);
                bool is_sender = sm_id % 2 == 0;
                int responsible_channel = sm_id / 2;
                auto x_int4 = reinterpret_cast<const int4*>(src);
                constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

                if (is_sender) {
                    int send_rank_id = tid / 32;
                    if (send_rank_id >= kNumRanks) {
                        return;
                    }
                    int rank_offset = send_rank_id > 0 ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank] : 0;
                    int num_rank_tokens = rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
                    int channel_offset = channel_prefix_matrix[send_rank_id * num_channels + responsible_channel];
                    int next_channel_offset = responsible_channel == num_channels - 1
                        ? num_rank_tokens
                        : channel_prefix_matrix[send_rank_id * num_channels + responsible_channel + 1];
                    int token_start_idx = rank_offset + channel_offset;
                    int token_end_idx = rank_offset + next_channel_offset;

                    void* ptr = static_cast<uint8_t*>(buffer_ptrs[send_rank_id]);
                    int num_channels_total = num_channels * kNumRanks;
                    int channel_rank_offset = responsible_channel * kNumRanks + rank;
                    DeviceBuffer<int> channel_head_idx(ptr, num_channels_total, channel_rank_offset);
                    DeviceBuffer<int> channel_tail_idx(ptr, num_channels_total, channel_rank_offset);
                    DeviceBuffer<int4> channel_x_buffers(ptr,
                                                         static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens * hidden_int4,
                                                         static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens * hidden_int4);
                    DeviceBuffer<int> channel_src_idx_buffers(ptr,
                                                              static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens,
                                                              static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens);
                    DeviceBuffer<float> channel_topk_weights_buffers(
                        ptr,
                        static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens * num_topk,
                        static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens * num_topk);

                    int current_channel_tail_idx = 0;
                    for (int token_idx = token_start_idx; token_idx < token_end_idx;) {
                        int num_round_tokens = sycl::min(num_max_send_tokens, token_end_idx - token_idx);
                        if (elect_one(item)) {
                            while (num_recv_buffer_tokens - (current_channel_tail_idx - ld_global(channel_head_idx.buffer())) <
                                   num_round_tokens) {
                                visa_spin_hint();
                            }
                        }
                        sycl::group_barrier(item.get_sub_group());

                        for (int i = 0; i < num_round_tokens; ++i) {
                            int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;
                            auto dst_x = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                            auto src_x = x_int4 + static_cast<int64_t>(token_idx + i) * hidden_int4;
                            for (int h = lid; h < hidden_int4; h += 32) {
                                st_global(dst_x + h, ld_global(src_x + h));
                            }
                            if (elect_one(item)) {
                                st_global(channel_src_idx_buffers.buffer() + dst_slot_idx, src_idx[token_idx + i]);
                            }
                            if (topk_weights != nullptr && lid < num_topk) {
                                st_global(channel_topk_weights_buffers.buffer() + dst_slot_idx * num_topk + lid,
                                          topk_weights[(token_idx + i) * num_topk + lid]);
                            }
                        }
                        token_idx += num_round_tokens;
                        current_channel_tail_idx += num_round_tokens;
                        sycl::group_barrier(item.get_sub_group());
                        if (elect_one(item)) {
                            st_global(channel_tail_idx.buffer(), current_channel_tail_idx);
                        }
                    }
                } else if (tid < 32) {
                    int token_start_idx = 0;
                    int token_end_idx = 0;
                    get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

                    int* channel_head_idx_ptr = static_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks;
                    int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;
                    int num_channels_total = num_channels * kNumRanks;
                    DeviceBuffer<int4> channel_x_buffers[kNumRanks];
                    DeviceBuffer<float> channel_topk_weights_buffers[kNumRanks];
                    for (int src_rank = 0; src_rank < kNumRanks; ++src_rank) {
                        int channel_rank_offset = responsible_channel * kNumRanks + src_rank;
                        void* ptr = static_cast<uint8_t*>(buffer_ptrs[rank]) + 2 * num_channels * kNumRanks * static_cast<int>(sizeof(int));
                        channel_x_buffers[src_rank] =
                            DeviceBuffer<int4>(ptr,
                                               static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens * hidden_int4,
                                               static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens * hidden_int4);
                        ptr = static_cast<uint8_t*>(ptr) + static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens * sizeof(int);
                        channel_topk_weights_buffers[src_rank] =
                            DeviceBuffer<float>(ptr,
                                                static_cast<int64_t>(num_channels_total) * num_recv_buffer_tokens * num_topk,
                                                static_cast<int64_t>(channel_rank_offset) * num_recv_buffer_tokens * num_topk);
                    }

                    for (int token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
                        int expected_heads[kNumRanks];
                        for (int src_rank = 0; src_rank < kNumRanks; ++src_rank) {
                            expected_heads[src_rank] = send_head[token_idx * kNumRanks + src_rank];
                        }

                        if (lid == 0) {
                            bool ready = false;
                            while (!ready) {
                                ready = true;
                                for (int src_rank = 0; src_rank < kNumRanks; ++src_rank) {
                                    if (expected_heads[src_rank] >= 0) {
                                        ready &= ld_global(channel_tail_idx_ptr + src_rank) > expected_heads[src_rank];
                                    }
                                }
                                if (!ready) {
                                    visa_spin_hint();
                                }
                            }
                        }
                        sycl::group_barrier(item.get_sub_group());

                        for (int h = lid; h < hidden_int4; h += 32) {
                            float values[kDtypePerInt4];
                            for (int j = 0; j < kDtypePerInt4; ++j) {
                                int hidden_idx = h * kDtypePerInt4 + j;
                                values[j] = 0.0f;
                                if (b0 != nullptr) {
                                    values[j] += static_cast<float>(b0[static_cast<int64_t>(token_idx) * hidden + hidden_idx]);
                                }
                                if (b1 != nullptr) {
                                    values[j] += static_cast<float>(b1[static_cast<int64_t>(token_idx) * hidden + hidden_idx]);
                                }
                            }
                            for (int src_rank = 0; src_rank < kNumRanks; ++src_rank) {
                                if (expected_heads[src_rank] >= 0) {
                                    int slot = expected_heads[src_rank] % num_recv_buffer_tokens;
                                    auto recv_values = reinterpret_cast<const dtype_t*>(channel_x_buffers[src_rank].buffer() +
                                                                                        static_cast<int64_t>(slot) * hidden_int4 + h);
                                    for (int j = 0; j < kDtypePerInt4; ++j) {
                                        values[j] += static_cast<float>(ld_global(recv_values + j));
                                    }
                                }
                            }
                            for (int j = 0; j < kDtypePerInt4; ++j) {
                                int hidden_idx = h * kDtypePerInt4 + j;
                                st_global(recv + static_cast<int64_t>(token_idx) * hidden + hidden_idx, static_cast<dtype_t>(values[j]));
                            }
                        }

                        if (recv_topk_weights != nullptr && lid < num_topk) {
                            float value = 0.0f;
                            for (int src_rank = 0; src_rank < kNumRanks; ++src_rank) {
                                if (expected_heads[src_rank] >= 0) {
                                    int slot = expected_heads[src_rank] % num_recv_buffer_tokens;
                                    value += ld_global(channel_topk_weights_buffers[src_rank].buffer() + slot * num_topk + lid);
                                }
                            }
                            st_global(recv_topk_weights + token_idx * num_topk + lid, value);
                        }

                        sycl::group_barrier(item.get_sub_group());
                        if (lid < kNumRanks) {
                            int expected = expected_heads[lid];
                            st_global(channel_head_idx_ptr + lid, expected < 0 ? -expected - 1 : expected + 1);
                        }
                    }
                }
            });
    });
}

template <int kNumRanks>
void launch_barrier(int** barrier_signal_ptrs, int rank, int barrier_signal_base, sycl::queue& queue) {
    constexpr int kThreads = 128;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<BarrierKernel<kNumRanks>>(
            sycl::nd_range<1>(sycl::range<1>(kThreads), sycl::range<1>(kThreads)),
            [=](sycl::nd_item<1> item) { barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank, barrier_signal_base, item); });
    });
}

#define SWITCH_RANKS(NUM_RANKS, BODY) \
    switch (NUM_RANKS) {              \
        case 1: {                     \
            BODY(1);                  \
            break;                    \
        }                             \
        case 2: {                     \
            BODY(2);                  \
            break;                    \
        }                             \
        case 4: {                     \
            BODY(4);                  \
            break;                    \
        }                             \
        case 8: {                     \
            BODY(8);                  \
            break;                    \
        }                             \
        default:                      \
            EP_HOST_ASSERT(false);    \
    }

}  // namespace

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     int num_tokens,
                     const bool* is_token_in_rank,
                     int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy,
                     int num_memset_int,
                     int expert_alignment,
                     void** buffer_ptrs,
                     int** barrier_signal_ptrs,
                     int rank,
                     int barrier_signal_base,
                     sycl::queue& queue,
                     int num_channels) {
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
#define BODY(ranks)                                               \
    launch_notify_dispatch<ranks>(num_tokens_per_rank,            \
                                  moe_recv_counter_mapped,        \
                                  num_tokens_per_expert,          \
                                  moe_recv_expert_counter_mapped, \
                                  num_experts,                    \
                                  num_tokens,                     \
                                  num_channels,                   \
                                  is_token_in_rank,               \
                                  channel_prefix_matrix,          \
                                  rank_prefix_matrix_copy,        \
                                  num_memset_int,                 \
                                  expert_alignment,               \
                                  buffer_ptrs,                    \
                                  barrier_signal_ptrs,            \
                                  rank,                           \
                                  barrier_signal_base,            \
                                  queue)
    SWITCH_RANKS(num_ranks, BODY);
#undef BODY
}

void cached_notify_dispatch(const int* rank_prefix_matrix,
                            int num_memset_int,
                            void** buffer_ptrs,
                            int** barrier_signal_ptrs,
                            int rank,
                            int num_ranks,
                            int barrier_signal_base,
                            sycl::queue& queue) {
#define BODY(ranks)                       \
    launch_cached_notify_dispatch<ranks>( \
        rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank, barrier_signal_base, queue)
    SWITCH_RANKS(num_ranks, BODY);
#undef BODY
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              int* recv_src_idx,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              int* recv_channel_offset,
              int* send_head,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              const bool* is_token_in_rank,
              const int* channel_prefix_matrix,
              int num_tokens,
              int,
              int hidden_int4,
              int num_topk,
              int num_experts,
              int num_scales,
              int scale_token_stride,
              int scale_hidden_stride,
              void** buffer_ptrs,
              int rank,
              int num_ranks,
              sycl::queue& queue,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens) {
    EP_HOST_ASSERT(num_sms % 2 == 0);
#define BODY(ranks)                                \
    launch_dispatch<ranks>(recv_x,                 \
                           recv_x_scales,          \
                           recv_src_idx,           \
                           recv_topk_idx,          \
                           recv_topk_weights,      \
                           recv_channel_offset,    \
                           send_head,              \
                           x,                      \
                           x_scales,               \
                           topk_idx,               \
                           topk_weights,           \
                           is_token_in_rank,       \
                           channel_prefix_matrix,  \
                           num_tokens,             \
                           hidden_int4,            \
                           num_topk,               \
                           num_experts,            \
                           num_scales,             \
                           scale_token_stride,     \
                           scale_hidden_stride,    \
                           buffer_ptrs,            \
                           rank,                   \
                           num_sms,                \
                           num_max_send_tokens,    \
                           num_recv_buffer_tokens, \
                           queue)
    SWITCH_RANKS(num_ranks, BODY);
#undef BODY
}

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** barrier_signal_ptrs,
                           int rank,
                           int num_ranks,
                           int barrier_signal_base,
                           sycl::queue& queue) {
#define BODY(ranks)                      \
    launch_cached_notify_combine<ranks>( \
        buffer_ptrs, send_head, num_channels, num_recv_tokens, num_memset_int, barrier_signal_ptrs, rank, barrier_signal_base, queue)
    SWITCH_RANKS(num_ranks, BODY);
#undef BODY
}

void combine(DataType type,
             void* recv_x,
             float* recv_topk_weights,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* src_idx,
             const int* rank_prefix_matrix,
             const int* channel_prefix_matrix,
             int* send_head,
             int num_tokens,
             int num_recv_tokens,
             int hidden,
             int num_topk,
             void** buffer_ptrs,
             int rank,
             int num_ranks,
             sycl::queue& queue,
             int num_sms,
             int num_max_send_tokens,
             int num_recv_buffer_tokens) {
#define BODY(ranks)                                                            \
    launch_combine<sycl::ext::oneapi::bfloat16, ranks>(type,                   \
                                                       recv_x,                 \
                                                       recv_topk_weights,      \
                                                       x,                      \
                                                       topk_weights,           \
                                                       bias_0,                 \
                                                       bias_1,                 \
                                                       src_idx,                \
                                                       rank_prefix_matrix,     \
                                                       channel_prefix_matrix,  \
                                                       send_head,              \
                                                       num_tokens,             \
                                                       num_recv_tokens,        \
                                                       hidden,                 \
                                                       num_topk,               \
                                                       buffer_ptrs,            \
                                                       rank,                   \
                                                       num_sms,                \
                                                       num_max_send_tokens,    \
                                                       num_recv_buffer_tokens, \
                                                       queue)
    if (type == DataType::kBFloat16) {
        SWITCH_RANKS(num_ranks, BODY);
    } else {
#undef BODY
#define BODY(ranks)                                    \
    launch_combine<int, ranks>(type,                   \
                               recv_x,                 \
                               recv_topk_weights,      \
                               x,                      \
                               topk_weights,           \
                               bias_0,                 \
                               bias_1,                 \
                               src_idx,                \
                               rank_prefix_matrix,     \
                               channel_prefix_matrix,  \
                               send_head,              \
                               num_tokens,             \
                               num_recv_tokens,        \
                               hidden,                 \
                               num_topk,               \
                               buffer_ptrs,            \
                               rank,                   \
                               num_sms,                \
                               num_max_send_tokens,    \
                               num_recv_buffer_tokens, \
                               queue)
        SWITCH_RANKS(num_ranks, BODY);
    }
#undef BODY
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, int barrier_signal_base, sycl::queue& queue) {
#define BODY(ranks) launch_barrier<ranks>(barrier_signal_ptrs, rank, barrier_signal_base, queue)
    SWITCH_RANKS(num_ranks, BODY);
#undef BODY
}

#undef SWITCH_RANKS

}  // namespace intranode
}  // namespace deep_ep
