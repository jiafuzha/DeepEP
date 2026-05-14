#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "api.cuh"
#include "buffer.cuh"
#include "exception.cuh"
#include "utils.cuh"

namespace deep_ep::intranode {

namespace {

sycl::queue& queue_from_stream(cudaStream_t stream) {
    return c10::xpu::XPUStream(stream).queue();
}

EP_HOST_DEVICE size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

EP_HOST_DEVICE int positive_or_one(int value) {
    return value > 0 ? value : 1;
}

struct StagingLayout {
    int* src_idx;
    topk_idx_t* topk_idx;
    float* topk_weights;
    float* x_scales;
    int4* x;
};

EP_HOST_DEVICE StagingLayout make_staging_layout(void* base,
                                                 int num_ranks,
                                                 int num_slots,
                                                 int hidden_int4,
                                                 int num_topk,
                                                 int num_scales) {
    auto cursor = reinterpret_cast<uintptr_t>(base) + num_ranks * num_ranks * sizeof(int);
    cursor = align_up_size(cursor, alignof(int4));

    auto* src_idx = reinterpret_cast<int*>(cursor);
    cursor += static_cast<size_t>(num_slots) * sizeof(int);
    cursor = align_up_size(cursor, alignof(topk_idx_t));

    auto* topk_idx = reinterpret_cast<topk_idx_t*>(cursor);
    cursor += static_cast<size_t>(num_slots) * positive_or_one(num_topk) * sizeof(topk_idx_t);
    cursor = align_up_size(cursor, alignof(float));

    auto* topk_weights = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(num_slots) * positive_or_one(num_topk) * sizeof(float);
    cursor = align_up_size(cursor, alignof(float));

    auto* x_scales = reinterpret_cast<float*>(cursor);
    cursor += static_cast<size_t>(num_slots) * positive_or_one(num_scales) * sizeof(float);
    cursor = align_up_size(cursor, alignof(int4));

    return {src_idx, topk_idx, topk_weights, x_scales, reinterpret_cast<int4*>(cursor)};
}

void enqueue_rank_barrier(sycl::queue& queue, int** barrier_signal_ptrs, int rank, int num_ranks) {
    EP_HOST_ASSERT(num_ranks > 0 && num_ranks <= NUM_MAX_NVL_PEERS);
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)), [=](sycl::nd_item<1>) {
            xpu_named_barrier::init<1>();
            switch (num_ranks) {
                case 2:
                    barrier_block<2>(barrier_signal_ptrs, rank);
                    break;
                case 4:
                    barrier_block<4>(barrier_signal_ptrs, rank);
                    break;
                case 8:
                    barrier_block<8>(barrier_signal_ptrs, rank);
                    break;
                default:
                    trap();
            }
        });
    });
}

}  // namespace

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
    auto& queue = queue_from_stream(stream);
    if (barrier_signal_ptrs == nullptr) {
        queue.wait_and_throw();
        return;
    }
    enqueue_rank_barrier(queue, barrier_signal_ptrs, rank, num_ranks);
}

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
                     cudaStream_t stream,
                     int num_channels) {
    (void)num_tokens_per_rank;
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    auto& queue = queue_from_stream(stream);
    const int num_local_experts = num_experts / num_ranks;
    const int num_channel_items = num_ranks * num_channels;
    const int work_items = std::max({num_ranks * (num_ranks + num_local_experts), num_channel_items, 1});

    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(work_items), [=](sycl::id<1> id) {
            int i = static_cast<int>(id[0]);
            for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                auto* per_rank_buffer = static_cast<int*>(buffer_ptrs[dst_rank]);
                auto* per_expert_buffer = per_rank_buffer + num_ranks * num_ranks;
                if (i < num_ranks) {
                    int count = 0;
                    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
                        count += is_token_in_rank[token_idx * num_ranks + i] ? 1 : 0;
                    }
                    per_rank_buffer[rank * num_ranks + i] = count;
                }
                if (i < num_local_experts) {
                    per_expert_buffer[rank * num_local_experts + i] =
                        num_tokens_per_expert[dst_rank * num_local_experts + i];
                }
            }
        });
    });

    barrier(barrier_signal_ptrs, rank, num_ranks, stream);

    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(std::max({num_ranks * num_ranks, num_local_experts, num_memset_int, num_channel_items, 1})),
                       [=](sycl::id<1> id) {
                           int i = static_cast<int>(id[0]);
                           auto* local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);
                           auto* local_per_expert_buffer = local_per_rank_buffer + num_ranks * num_ranks;

                           if (i < num_ranks) {
                               int prefix = 0;
                               for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                                   prefix += local_per_rank_buffer[src_rank * num_ranks + i];
                                   local_per_rank_buffer[src_rank * num_ranks + i] = prefix;
                               }
                               if (i == rank) {
                                   *moe_recv_counter_mapped = prefix;
                               }
                           }

                           if (i < num_local_experts) {
                               int sum = 0;
                               for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                                   sum += local_per_expert_buffer[src_rank * num_local_experts + i];
                               }
                               moe_recv_expert_counter_mapped[i] =
                                   (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
                           }

                           if (i < num_ranks * num_ranks) {
                               rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];
                           }
                           if (i < num_memset_int) {
                               local_per_expert_buffer[i] = 0;
                           }

                           if (i < num_channel_items) {
                               int dst_rank = i / num_channels;
                               int channel = i % num_channels;
                               int token_start_idx = 0;
                               int token_end_idx = 0;
                               get_channel_task_range(num_tokens, num_channels, channel, token_start_idx, token_end_idx);
                               int count = 0;
                               for (int token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
                                   count += is_token_in_rank[token_idx * num_ranks + dst_rank] ? 1 : 0;
                               }
                               int prefix = 0;
                               for (int c = 0; c <= channel; ++c) {
                                   int start = 0;
                                   int end = 0;
                                   get_channel_task_range(num_tokens, num_channels, c, start, end);
                                   for (int token_idx = start; token_idx < end; ++token_idx) {
                                       prefix += is_token_in_rank[token_idx * num_ranks + dst_rank] ? 1 : 0;
                                   }
                               }
                               (void)count;
                               channel_prefix_matrix[dst_rank * num_channels + channel] = prefix;
                           }
                       });
    });

    barrier(barrier_signal_ptrs, rank, num_ranks, stream);
}

void cached_notify_dispatch(const int* rank_prefix_matrix,
                            int num_memset_int,
                            void** buffer_ptrs,
                            int** barrier_signal_ptrs,
                            int rank,
                            int num_ranks,
                            cudaStream_t stream) {
    auto& queue = queue_from_stream(stream);
    barrier(barrier_signal_ptrs, rank, num_ranks, stream);
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(std::max(num_ranks * num_ranks, num_memset_int)), [=](sycl::id<1> id) {
            int i = static_cast<int>(id[0]);
            auto* ptr = static_cast<int*>(buffer_ptrs[rank]);
            if (i < num_ranks * num_ranks) {
                ptr[i] = rank_prefix_matrix[i];
            }
            if (i < num_memset_int) {
                ptr[num_ranks * num_ranks + i] = 0;
            }
        });
    });
    barrier(barrier_signal_ptrs, rank, num_ranks, stream);
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
              int num_worst_tokens,
              int hidden_int4,
              int num_topk,
              int num_experts,
              int num_scales,
               int scale_token_stride,
               int scale_hidden_stride,
               void** buffer_ptrs,
               int** barrier_signal_ptrs,
               int rank,
              int num_ranks,
              cudaStream_t stream,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens) {
    (void)num_worst_tokens;
    (void)num_experts;
    (void)num_sms;
    (void)num_max_send_tokens;
    (void)num_recv_buffer_tokens;
    EP_HOST_ASSERT(hidden_int4 > 0);
    auto& queue = queue_from_stream(stream);
    const int num_slots = std::max(1, num_tokens * num_ranks);
    const auto* x_int4 = static_cast<const int4*>(x);
    auto* recv_x_int4 = static_cast<int4*>(recv_x);

    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(num_tokens, num_ranks), [=](sycl::id<2> id) {
            int token_idx = static_cast<int>(id[0]);
            int dst_rank = static_cast<int>(id[1]);
            bool selected = is_token_in_rank[token_idx * num_ranks + dst_rank];
            send_head[token_idx * num_ranks + dst_rank] = selected ? 0 : -1;
            if (!selected) {
                return;
            }

            int local_idx = 0;
            for (int prev = 0; prev < token_idx; ++prev) {
                local_idx += is_token_in_rank[prev * num_ranks + dst_rank] ? 1 : 0;
            }
            int rank_offset = rank > 0 ? static_cast<int*>(buffer_ptrs[rank])[(rank - 1) * num_ranks + dst_rank] : 0;
            int slot = rank_offset + local_idx;
            send_head[token_idx * num_ranks + dst_rank] = slot;

            auto dst = make_staging_layout(buffer_ptrs[dst_rank], num_ranks, num_slots, hidden_int4, num_topk, num_scales);
            dst.src_idx[slot] = token_idx;
            for (int i = 0; i < hidden_int4; ++i) {
                dst.x[static_cast<int64_t>(slot) * hidden_int4 + i] = x_int4[static_cast<int64_t>(token_idx) * hidden_int4 + i];
            }
            if (topk_idx != nullptr && recv_topk_idx != nullptr) {
                int num_experts_per_rank = num_experts / num_ranks;
                int expert_begin = dst_rank * num_experts_per_rank;
                int expert_end = expert_begin + num_experts_per_rank;
                for (int i = 0; i < num_topk; ++i) {
                    auto expert = topk_idx[token_idx * num_topk + i];
                    bool in_dst = expert >= expert_begin && expert < expert_end;
                    dst.topk_idx[slot * num_topk + i] = in_dst ? expert - expert_begin : -1;
                    dst.topk_weights[slot * num_topk + i] = in_dst ? topk_weights[token_idx * num_topk + i] : 0.0f;
                }
            }
            if (x_scales != nullptr && recv_x_scales != nullptr) {
                for (int i = 0; i < num_scales; ++i) {
                    dst.x_scales[slot * num_scales + i] = x_scales[token_idx * scale_token_stride + i * scale_hidden_stride];
                }
            }
        });
    });

    barrier(barrier_signal_ptrs, rank, num_ranks, stream);

    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(num_slots), [=](sycl::id<1> id) {
            int slot = static_cast<int>(id[0]);
            auto src = make_staging_layout(buffer_ptrs[rank], num_ranks, num_slots, hidden_int4, num_topk, num_scales);
            int total_recv = static_cast<int*>(buffer_ptrs[rank])[(num_ranks - 1) * num_ranks + rank];
            if (slot >= total_recv) {
                return;
            }
            recv_src_idx[slot] = src.src_idx[slot];
            for (int i = 0; i < hidden_int4; ++i) {
                recv_x_int4[static_cast<int64_t>(slot) * hidden_int4 + i] = src.x[static_cast<int64_t>(slot) * hidden_int4 + i];
            }
            if (recv_topk_idx != nullptr) {
                for (int i = 0; i < num_topk; ++i) {
                    recv_topk_idx[slot * num_topk + i] = src.topk_idx[slot * num_topk + i];
                    recv_topk_weights[slot * num_topk + i] = src.topk_weights[slot * num_topk + i];
                }
            }
            if (recv_x_scales != nullptr) {
                for (int i = 0; i < num_scales; ++i) {
                    recv_x_scales[slot * num_scales + i] = src.x_scales[slot * num_scales + i];
                }
            }
        });
    });

    if (recv_channel_offset != nullptr) {
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(std::max(1, num_ranks * std::max(1, num_sms / 2))), [=](sycl::id<1> id) {
                int i = static_cast<int>(id[0]);
                int total = num_ranks * std::max(1, num_sms / 2);
                if (i < total) {
                    recv_channel_offset[i] = channel_prefix_matrix[i];
                }
            });
        });
    }
}

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** barrier_signal_ptrs,
                           int rank,
                           int num_ranks,
                           cudaStream_t stream) {
    (void)send_head;
    (void)num_channels;
    (void)num_recv_tokens;
    auto& queue = queue_from_stream(stream);
    barrier(barrier_signal_ptrs, rank, num_ranks, stream);
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(std::max(1, num_memset_int)), [=](sycl::id<1> id) {
            static_cast<int*>(buffer_ptrs[rank])[id[0]] = 0;
        });
    });
    enqueue_rank_barrier(queue, barrier_signal_ptrs, rank, num_ranks);
}

void combine(cudaDataType_t type,
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
             int** barrier_signal_ptrs,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_sms,
             int num_max_send_tokens,
             int num_recv_buffer_tokens) {
    (void)rank_prefix_matrix;
    (void)channel_prefix_matrix;
    (void)num_sms;
    (void)num_max_send_tokens;
    (void)num_recv_buffer_tokens;
    if (type != CUDA_R_16BF) {
        throw std::runtime_error("DeepEP XPU native intranode combine currently supports BF16 tensors only");
    }

    auto& queue = queue_from_stream(stream);
    const int hidden_int4 = hidden * static_cast<int>(sizeof(nv_bfloat16)) / static_cast<int>(sizeof(int4));
    const int num_slots = std::max(1, num_recv_tokens * num_ranks);
    const auto* x_bf16 = static_cast<const nv_bfloat16*>(x);
    auto* recv_bf16 = static_cast<nv_bfloat16*>(recv_x);
    const auto* bias0_bf16 = static_cast<const nv_bfloat16*>(bias_0);
    const auto* bias1_bf16 = static_cast<const nv_bfloat16*>(bias_1);

    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(std::max(1, num_tokens), hidden), [=](sycl::id<2> id) {
            int token_idx = static_cast<int>(id[0]);
            int hidx = static_cast<int>(id[1]);
            int original_token = src_idx[token_idx];
            if (original_token < 0 || original_token >= num_recv_tokens) {
                return;
            }
            int owner_rank = 0;
            while (owner_rank + 1 < num_ranks && token_idx >= rank_prefix_matrix[owner_rank * num_ranks + rank]) {
                ++owner_rank;
            }
            auto dst = make_staging_layout(buffer_ptrs[owner_rank], num_ranks, num_slots, hidden_int4, num_topk, 0);
            dst.src_idx[original_token * num_ranks + rank] = 1;
            reinterpret_cast<nv_bfloat16*>(dst.x)[static_cast<int64_t>(original_token * num_ranks + rank) * hidden + hidx] =
                x_bf16[static_cast<int64_t>(token_idx) * hidden + hidx];
        });
    });

    if (topk_weights != nullptr && recv_topk_weights != nullptr) {
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(std::max(1, num_tokens), num_topk), [=](sycl::id<2> id) {
                int token_idx = static_cast<int>(id[0]);
                int topk = static_cast<int>(id[1]);
                int original_token = src_idx[token_idx];
                if (original_token >= 0 && original_token < num_recv_tokens) {
                    int owner_rank = 0;
                    while (owner_rank + 1 < num_ranks && token_idx >= rank_prefix_matrix[owner_rank * num_ranks + rank]) {
                        ++owner_rank;
                    }
                    auto dst = make_staging_layout(buffer_ptrs[owner_rank], num_ranks, num_slots, hidden_int4, num_topk, 0);
                    dst.topk_weights[(original_token * num_ranks + rank) * num_topk + topk] = topk_weights[token_idx * num_topk + topk];
                }
            });
        });
    }

    barrier(barrier_signal_ptrs, rank, num_ranks, stream);

    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(std::max(1, num_recv_tokens), hidden), [=](sycl::id<2> id) {
            int token_idx = static_cast<int>(id[0]);
            int hidx = static_cast<int>(id[1]);
            float value = 0.0f;
            if (bias0_bf16 != nullptr) {
                value += static_cast<float>(bias0_bf16[static_cast<int64_t>(token_idx) * hidden + hidx]);
            }
            if (bias1_bf16 != nullptr) {
                value += static_cast<float>(bias1_bf16[static_cast<int64_t>(token_idx) * hidden + hidx]);
            }
            auto src = make_staging_layout(buffer_ptrs[rank], num_ranks, num_slots, hidden_int4, num_topk, 0);
            for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                if (send_head[token_idx * num_ranks + src_rank] >= 0) {
                    value += static_cast<float>(
                        reinterpret_cast<nv_bfloat16*>(src.x)[static_cast<int64_t>(token_idx * num_ranks + src_rank) * hidden + hidx]);
                }
            }
            recv_bf16[static_cast<int64_t>(token_idx) * hidden + hidx] = static_cast<nv_bfloat16>(value);
        });
    });

    if (recv_topk_weights != nullptr) {
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(std::max(1, num_recv_tokens), num_topk), [=](sycl::id<2> id) {
                int token_idx = static_cast<int>(id[0]);
                int topk = static_cast<int>(id[1]);
                float value = 0.0f;
                auto src = make_staging_layout(buffer_ptrs[rank], num_ranks, num_slots, hidden_int4, num_topk, 0);
                for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                    if (send_head[token_idx * num_ranks + src_rank] >= 0) {
                        value += src.topk_weights[(token_idx * num_ranks + src_rank) * num_topk + topk];
                    }
                }
                recv_topk_weights[token_idx * num_topk + topk] = value;
            });
        });
    }
}

}  // namespace deep_ep::intranode
