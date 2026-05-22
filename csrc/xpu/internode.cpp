#include <algorithm>
#include <iostream>

#include "xpu_kernels.hpp"

#ifdef DEEP_EP_ENABLE_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

namespace deep_ep {
namespace internode {
namespace {

class DispatchInitKernel;
class DispatchCountExchangeKernel;
class DispatchPackKernel;
class DispatchOffsetComputeKernel;
class DispatchPayloadKernel;
class DispatchPayloadQuietKernel;
class DispatchQueueCopyKernel;
class DispatchCopyKernel;

template <typename dtype_t>
class CombineInitKernel;

template <typename dtype_t>
class CombinePackKernel;

template <typename dtype_t>
class CombinePayloadKernel;

template <typename dtype_t>
class CombinePayloadQuietKernel;

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
              int num_max_rdma_chunked_recv_tokens,
              int rank,
              int num_ranks,
              sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(recv_x != nullptr && x != nullptr, "XPU internode dispatch requires input and output tensors");
    TORCH_CHECK(rdma_buffer_ptr != nullptr, "XPU internode dispatch requires a symmetric iSHMEM RDMA buffer");
    TORCH_CHECK(is_token_in_rank != nullptr, "XPU internode dispatch requires is_token_in_rank");
    TORCH_CHECK(num_channels > 0, "XPU internode dispatch requires a positive channel count");
    TORCH_CHECK(num_max_rdma_chunked_recv_tokens > 0, "XPU internode dispatch requires a positive RDMA queue window");
    TORCH_CHECK(num_tokens <= num_max_rdma_chunked_recv_tokens,
                "XPU internode dispatch validation path requires the per-channel RDMA queue window to cover the local token count");
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
    offset = align_offset(offset, std::max({alignof(SourceMeta), alignof(topk_idx_t), alignof(float)}));
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
    const size_t num_queue_slots = static_cast<size_t>(num_ranks) * num_channels * queue_window;
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
                }
                rdma_meta[linear].src_rdma_rank = -1;
                rdma_meta[linear].is_token_in_nvl_rank_bits = -1;
            }
            if (linear < num_send_slots) {
                rdma_send_meta[linear].src_rdma_rank = -1;
                rdma_send_meta[linear].is_token_in_nvl_rank_bits = -1;
                rdma_send_dst_token[linear] = -1;
            }
            if (linear < num_queue_slots) {
                rdma_queue_meta[linear].src_rdma_rank = -1;
                rdma_queue_meta[linear].is_token_in_nvl_rank_bits = -1;
                rdma_queue_dst_token[linear] = -1;
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
    internode::barrier();

    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<DispatchCountExchangeKernel>([=]() {
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
            for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                auto* remote_count_row = rdma_count_matrix + rank * num_ranks;
                if (dst_rank != rank) {
                    ishmem_putmem(remote_count_row, local_count_row, static_cast<size_t>(num_ranks) * sizeof(int), dst_rank);
                }
            }

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
            for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                auto* remote_channel_row = rdma_channel_count_matrix + static_cast<size_t>(rank) * num_ranks * num_channels;
                if (dst_rank != rank) {
                    ishmem_putmem(
                        remote_channel_row, local_channel_row, static_cast<size_t>(num_ranks) * num_channels * sizeof(int), dst_rank);
                }
            }
        });
    });
    queue.wait();
    internode::barrier();

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
            rdma_send_meta[send_slot] = SourceMeta{rank, token};
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
    internode::barrier();
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<DispatchPayloadKernel>(sycl::range<1>(static_cast<size_t>(num_ranks) * num_tokens), [=](sycl::id<1> id) {
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

            SourceMeta token_meta{rank, token};
            int channel = static_cast<int>((static_cast<int64_t>(token) * num_channels) / num_tokens);
            if (channel >= num_channels) {
                channel = num_channels - 1;
            }
            const int channel_start = (static_cast<int64_t>(num_tokens) * channel) / num_channels;
            int queue_ordinal = 0;
            for (int prior_token = channel_start; prior_token < token; ++prior_token) {
                queue_ordinal += is_token_in_rank[prior_token * num_ranks + dst_rank] ? 1 : 0;
            }
            if (queue_ordinal >= queue_window) {
                return;
            }
            const size_t queue_slot = (static_cast<size_t>(rank) * num_channels + channel) * queue_window + queue_ordinal;
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
                for (size_t h = 0; h < row_bytes; ++h) {
                    dst_x[h] = src_x[h];
                }
                *dst_meta = *src_meta;
                for (int k = 0; k < num_topk; ++k) {
                    dst_topk_idx[k] = src_topk_idx[k];
                    dst_topk_weights[k] = src_topk_weights[k];
                }
                for (int k = 0; k < num_scales; ++k) {
                    dst_scales[k] = src_scales[k];
                }
                rdma_queue_dst_token[queue_slot] = dst_token;
            } else {
                ishmem_putmem(dst_x, src_x, row_bytes, dst_rank);
                ishmem_putmem(dst_meta, src_meta, sizeof(SourceMeta), dst_rank);
                if (num_topk > 0) {
                    ishmem_putmem(dst_topk_idx, src_topk_idx, static_cast<size_t>(num_topk) * sizeof(topk_idx_t), dst_rank);
                    ishmem_putmem(dst_topk_weights, src_topk_weights, static_cast<size_t>(num_topk) * sizeof(float), dst_rank);
                }
                if (num_scales > 0) {
                    ishmem_putmem(dst_scales, src_scales, static_cast<size_t>(num_scales) * sizeof(float), dst_rank);
                }
                ishmem_putmem(rdma_queue_dst_token + queue_slot, src_dst_token, sizeof(int), dst_rank);
            }
        });
    });
    queue.wait();
    internode::barrier();
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<DispatchQueueCopyKernel>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
            const size_t linear = static_cast<size_t>(id[0]);
            if (linear < total_queue_elements) {
                const size_t queue_slot = linear / row_bytes;
                const size_t byte_idx = linear - queue_slot * row_bytes;
                const int dst_token = rdma_queue_dst_token[queue_slot];
                if (rdma_queue_meta[queue_slot].src_rdma_rank >= 0 && dst_token >= 0 && dst_token < num_recv_tokens) {
                    rdma_x[static_cast<size_t>(dst_token) * row_bytes + byte_idx] = rdma_queue_x[linear];
                }
            }
            if (linear < num_queue_slots) {
                const int dst_token = rdma_queue_dst_token[linear];
                if (rdma_queue_meta[linear].src_rdma_rank >= 0 && dst_token >= 0 && dst_token < num_recv_tokens) {
                    rdma_meta[dst_token] = rdma_queue_meta[linear];
                }
            }
            if (linear < total_queue_topk_elements) {
                const size_t queue_slot = linear / num_topk;
                const size_t topk_idx = linear - queue_slot * num_topk;
                const int dst_token = rdma_queue_dst_token[queue_slot];
                if (rdma_queue_meta[queue_slot].src_rdma_rank >= 0 && dst_token >= 0 && dst_token < num_recv_tokens) {
                    rdma_topk_idx[static_cast<size_t>(dst_token) * num_topk + topk_idx] = rdma_queue_topk_idx[linear];
                    rdma_topk_weights[static_cast<size_t>(dst_token) * num_topk + topk_idx] = rdma_queue_topk_weights[linear];
                }
            }
            if (linear < total_queue_scale_elements) {
                const size_t queue_slot = linear / num_scales;
                const size_t scale_idx = linear - queue_slot * num_scales;
                const int dst_token = rdma_queue_dst_token[queue_slot];
                if (rdma_queue_meta[queue_slot].src_rdma_rank >= 0 && dst_token >= 0 && dst_token < num_recv_tokens) {
                    rdma_x_scales[static_cast<size_t>(dst_token) * num_scales + scale_idx] = rdma_queue_x_scales[linear];
                }
            }
        });
    });
    queue.wait();
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
                         int num_max_rdma_chunked_recv_tokens,
                         int rank,
                         int num_ranks,
                         sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(rdma_buffer_ptr != nullptr, "XPU internode combine requires a symmetric iSHMEM RDMA buffer");
    TORCH_CHECK(num_max_rdma_chunked_recv_tokens > 0, "XPU internode combine requires a positive RDMA queue window");
    TORCH_CHECK(num_combined_tokens <= num_max_rdma_chunked_recv_tokens,
                "XPU internode combine validation path requires the RDMA queue window to cover the combined token count");
    const int queue_window = num_max_rdma_chunked_recv_tokens;
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
    auto* rdma_queue_topk_weights = reinterpret_cast<float*>(rdma_queue_x + static_cast<int64_t>(num_ranks) * queue_window * hidden);
    const int64_t total_combined = static_cast<int64_t>(num_combined_tokens) * hidden;
    const int64_t total_recv = static_cast<int64_t>(num_tokens) * hidden;
    const int64_t total_combined_topk = static_cast<int64_t>(num_combined_tokens) * num_topk;
    const int64_t total_recv_topk = static_cast<int64_t>(num_tokens) * num_topk;
    const int64_t total_queue = static_cast<int64_t>(num_ranks) * queue_window * hidden;
    const int64_t total_queue_topk = static_cast<int64_t>(num_ranks) * queue_window * num_topk;
    const int64_t init_range = std::max({total_combined, total_combined_topk, total_queue, total_queue_topk});

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
        });
    });
    queue.wait();
    internode::barrier();

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
    internode::barrier();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinePayloadKernel<dtype_t>>(sycl::range<1>(num_tokens), [=](sycl::id<1> id) {
            const int recv_token = static_cast<int>(id[0]);
            const SourceMeta token_meta = meta[recv_token];
            const int src_rank = token_meta.src_rdma_rank;
            const int src_token = token_meta.is_token_in_nvl_rank_bits;
            if (src_rank < 0 || src_rank >= num_ranks || src_token < 0 || src_token >= num_combined_tokens) {
                return;
            }

            auto* remote_dst = rdma_queue_x + (static_cast<int64_t>(rank) * queue_window + src_token) * hidden;
            auto* local_src = rdma_send_x + static_cast<int64_t>(recv_token) * hidden;
            auto* remote_topk_weights = rdma_queue_topk_weights + (static_cast<int64_t>(rank) * queue_window + src_token) * num_topk;
            auto* local_topk_weights = rdma_send_topk_weights + static_cast<int64_t>(recv_token) * num_topk;
            if (src_rank == rank) {
                for (int h = 0; h < hidden; ++h) {
                    remote_dst[h] = local_src[h];
                }
                for (int k = 0; k < num_topk; ++k) {
                    remote_topk_weights[k] = local_topk_weights[k];
                }
            } else {
                ishmem_putmem(remote_dst, local_src, static_cast<size_t>(hidden) * sizeof(dtype_t), src_rank);
                if (num_topk > 0) {
                    ishmem_putmem(remote_topk_weights, local_topk_weights, static_cast<size_t>(num_topk) * sizeof(float), src_rank);
                }
            }
        });
    });
    queue.wait();
    internode::barrier();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombineQueueCopyKernel<dtype_t>>(
            sycl::range<1>(std::max(total_combined, total_combined_topk)), [=](sycl::id<1> id) {
                const int64_t i = static_cast<int64_t>(id[0]);
                if (i < total_combined) {
                    dtype_t value{};
                    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                        value += rdma_queue_x[(static_cast<int64_t>(src_rank) * queue_window * hidden) + i];
                    }
                    rdma_recv_x[i] = value;
                }
                if (i < total_combined_topk) {
                    float value = 0.0f;
                    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                        value += rdma_queue_topk_weights[(static_cast<int64_t>(src_rank) * queue_window * num_topk) + i];
                    }
                    rdma_recv_topk_weights[i] = value;
                }
            });
    });
    queue.wait();

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
                                     num_max_rdma_chunked_recv_tokens,
                                     rank,
                                     num_ranks,
                                     queue);
    }
}

}  // namespace internode
}  // namespace deep_ep
