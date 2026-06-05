#include <algorithm>
#include <cstdlib>
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
class DispatchPayloadTailKernel;
class DispatchQueueResetKernel;
class DispatchQueueCopyKernel;
class DispatchCopyKernel;
class DebugChannelPutInitKernel;
class DebugChannelPutResetKernel;
class DebugChannelPutKernel;
class DebugChannelPutValidateKernel;

template <typename dtype_t>
class CombineInitKernel;

template <typename dtype_t>
class CombinePackKernel;

template <typename dtype_t>
class CombinePayloadKernel;

template <typename dtype_t>
class CombinePayloadTailKernel;

template <typename dtype_t>
class CombinePayloadQuietKernel;

template <typename dtype_t>
class CombineQueueResetKernel;

template <typename dtype_t>
class CombineQueueCopyKernel;

template <typename dtype_t>
class CombineBiasKernel;

#ifdef DEEP_EP_ENABLE_ISHMEM
struct DebugChannelPutInit {
    int* output;
    int* recv_payload;
    int* recv_meta;
    int* recv_dst_token;
    int* send_payload;
    int* send_meta;
    int* send_dst_token;
    int num_queue_slots;
    int row_ints;
    int num_channels;
    int output_cols;
    int rank;

    void operator()(sycl::id<1> id) const {
        const int i = static_cast<int>(id[0]);
        if (i < num_queue_slots * row_ints) {
            recv_payload[i] = -1;
        }
        if (i < num_queue_slots) {
            recv_meta[i] = -1;
            recv_dst_token[i] = -1;
        }
        if (i < num_channels * row_ints) {
            const int channel = i / row_ints;
            const int lane = i - channel * row_ints;
            send_payload[i] = (rank + 1) * 100000 + channel * 1000 + lane;
        }
        if (i < num_channels) {
            send_meta[i] = rank * 100 + i;
            send_dst_token[i] = i;
        }
        if (i < num_channels * output_cols) {
            output[i] = -999;
        }
    }
};

struct DebugChannelPutReset {
    int* recv_payload;
    int* recv_meta;
    int* recv_dst_token;
    int num_queue_slots;
    int row_ints;

    void operator()(sycl::id<1> id) const {
        const int i = static_cast<int>(id[0]);
        if (i < num_queue_slots * row_ints) {
            recv_payload[i] = -1;
        }
        if (i < num_queue_slots) {
            recv_meta[i] = -1;
            recv_dst_token[i] = -1;
        }
    }
};

struct DebugChannelPutPost {
    int* recv_payload;
    int* recv_meta;
    int* recv_dst_token;
    int* send_payload;
    int* send_meta;
    int* send_dst_token;
    int row_ints;
    int num_channels;
    int queue_stride;
    int rank;
    int channel;

    void operator()() const {
        const int peer = 1 - rank;
        const int queue_slot = (rank * num_channels + channel) * queue_stride;
        auto* dst_payload = recv_payload + queue_slot * row_ints;
        auto* dst_meta = recv_meta + queue_slot;
        auto* dst_token = recv_dst_token + queue_slot;
        auto* src_payload = send_payload + channel * row_ints;
        auto* src_meta = send_meta + channel;
        auto* src_token = send_dst_token + channel;
        ishmem_putmem(dst_payload, src_payload, static_cast<size_t>(row_ints) * sizeof(int), peer);
        ishmem_putmem(dst_token, src_token, sizeof(int), peer);
        ishmem_putmem(dst_meta, src_meta, sizeof(int), peer);
    }
};

struct DebugChannelPutValidate {
    int* output;
    int* recv_payload;
    int* recv_meta;
    int* recv_dst_token;
    int row_ints;
    int num_channels;
    int queue_stride;
    int output_cols;
    int rank;
    int channel;

    void operator()() const {
        const int peer = 1 - rank;
        const int queue_slot = (peer * num_channels + channel) * queue_stride;
        const int expected_meta = peer * 100 + channel;
        const int expected_first = (peer + 1) * 100000 + channel * 1000;
        const int actual_meta = recv_meta[queue_slot];
        const int actual_token = recv_dst_token[queue_slot];
        const int actual_first = recv_payload[queue_slot * row_ints];
        const int actual_last = recv_payload[queue_slot * row_ints + row_ints - 1];
        auto* row = output + channel * output_cols;
        row[0] = actual_meta;
        row[1] = actual_token;
        row[2] = actual_first;
        row[3] = actual_last;
        row[4] = expected_meta;
        row[5] = channel;
        row[6] = expected_first;
        row[7] = (actual_meta == expected_meta && actual_token == channel && actual_first == expected_first &&
                  actual_last == expected_first + row_ints - 1)
            ? 0
            : 1;
    }
};
#endif

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
    internode::mpi_barrier();

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
                    ishmem_putmem_nbi(remote_count_row, local_count_row, static_cast<size_t>(num_ranks) * sizeof(int), dst_rank);
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
                    ishmem_putmem_nbi(
                        remote_channel_row, local_channel_row, static_cast<size_t>(num_ranks) * num_channels * sizeof(int), dst_rank);
                }
            }
            // Drain all NBI puts
            ishmem_quiet();
        });
    });
    queue.wait();
    internode::mpi_barrier();

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
    internode::mpi_barrier();
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
                internode::mpi_barrier();
            }

            queue.submit([&](sycl::handler& cgh) {
                constexpr int kQueueGroupSize = 32;
                cgh.parallel_for<DispatchPayloadKernel>(
                    sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_ranks) * kQueueGroupSize), sycl::range<1>(kQueueGroupSize)),
                    [=](sycl::nd_item<1> item) {
                        auto group = item.get_group();
                        const int local_id = static_cast<int>(item.get_local_id(0));
                        const int local_size = static_cast<int>(item.get_local_range(0));
                        const int dst_rank = static_cast<int>(item.get_group(0));
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
                                ishmemx_putmem_nbi_work_group(dst_x, src_x, row_bytes, dst_rank, group);
                                if (num_topk > 0) {
                                    ishmemx_putmem_nbi_work_group(
                                        dst_topk_idx, src_topk_idx, static_cast<size_t>(num_topk) * sizeof(topk_idx_t), dst_rank, group);
                                    ishmemx_putmem_nbi_work_group(
                                        dst_topk_weights, src_topk_weights, static_cast<size_t>(num_topk) * sizeof(float), dst_rank, group);
                                }
                                if (num_scales > 0) {
                                    ishmemx_putmem_nbi_work_group(
                                        dst_scales, src_scales, static_cast<size_t>(num_scales) * sizeof(float), dst_rank, group);
                                }
                                ishmemx_putmem_nbi_work_group(
                                    rdma_queue_dst_token + queue_slot, src_dst_token, sizeof(int), dst_rank, group);
                                ishmemx_putmem_nbi_work_group(dst_meta, src_meta, sizeof(SourceMeta), dst_rank, group);
                            }
                            sycl::group_barrier(group);
                            if (local_id == 0) {
                                ++window_count;
                            }
                        }
                        if (local_id == 0 && window_count > 0) {
                            const size_t queue_pair = static_cast<size_t>(rank) * num_channels + channel;
                            if (dst_rank == rank) {
                                rdma_queue_tail[queue_pair] += window_count;
                            } else {
                                // NBI put the tail to remote — stage in rdma_queue_head to avoid
                                // clobbering the self-send tail already in rdma_queue_tail.
                                rdma_queue_head[queue_pair] = window_count;
                                ishmem_putmem_nbi(rdma_queue_tail + queue_pair, rdma_queue_head + queue_pair,
                                                  sizeof(int), dst_rank);
                            }
                        }
                    });
            });
            queue.wait();
            // Quiet drains all outstanding NBI puts (payload data + tail values)
            queue.submit([&](sycl::handler& cgh) {
                cgh.single_task<DispatchPayloadTailKernel>([=]() {
                    ishmem_quiet();
                });
            });
            queue.wait();
            internode::mpi_barrier();

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
    internode::mpi_barrier();

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
    internode::mpi_barrier();

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
        internode::mpi_barrier();

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
                            ishmemx_putmem_nbi_work_group(remote_dst, local_src, static_cast<size_t>(hidden) * sizeof(dtype_t), src_rank, group);
                            if (num_topk > 0) {
                                ishmemx_putmem_nbi_work_group(
                                    remote_topk_weights, local_topk_weights, static_cast<size_t>(num_topk) * sizeof(float), src_rank, group);
                            }
                        }
                        sycl::group_barrier(group);
                        if (local_id == 0) {
                            ++window_count;
                        }
                    }
                    if (local_id == 0 && window_count > 0) {
                        if (src_rank == rank) {
                            rdma_queue_tail[rank] += window_count;
                        }
                    }
                });
        });
        queue.wait();
        queue.submit([&](sycl::handler& cgh) {
            cgh.single_task<CombinePayloadTailKernel<dtype_t>>([=]() {
                // NBI put tail values instead of RDMA atomics.
                // Each sender writes to queue_pair = rank (single-writer).
                for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                    if (src_rank == rank) {
                        continue;
                    }
                    int window_count = 0;
                    for (int recv_token = 0; recv_token < num_tokens; ++recv_token) {
                        const SourceMeta token_meta = meta[recv_token];
                        const int src_token = token_meta.is_token_in_nvl_rank_bits;
                        if (token_meta.src_rdma_rank == src_rank && src_token >= window_offset &&
                            src_token < window_offset + window_tokens) {
                            ++window_count;
                        }
                    }
                    if (window_count > 0) {
                        // Stage in rdma_queue_head to avoid clobbering self-send tail
                        rdma_queue_head[rank] = window_count;
                        ishmem_putmem_nbi(rdma_queue_tail + rank, rdma_queue_head + rank,
                                          sizeof(int), src_rank);
                    }
                }
                ishmem_quiet();
            });
        });
        queue.wait();
        internode::mpi_barrier();

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

void debug_channel_put(
    int* output, void* rdma_buffer_ptr, int row_ints, int num_channels, int queue_stride, int rank, int num_ranks, sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(output != nullptr, "debug_channel_put requires an output tensor");
    TORCH_CHECK(rdma_buffer_ptr != nullptr, "debug_channel_put requires a symmetric iSHMEM RDMA buffer");
    TORCH_CHECK(num_ranks == 2, "debug_channel_put currently expects exactly 2 ranks");
    TORCH_CHECK(row_ints > 0 && num_channels > 0 && queue_stride >= num_channels,
                "debug_channel_put received invalid row/channel/stride parameters");

    auto* base = static_cast<int*>(rdma_buffer_ptr);
    const int num_queue_slots = num_ranks * num_channels * queue_stride;
    auto* recv_payload = base;
    auto* recv_meta = recv_payload + static_cast<int64_t>(num_queue_slots) * row_ints;
    auto* recv_dst_token = recv_meta + num_queue_slots;
    auto* send_payload = recv_dst_token + num_queue_slots;
    auto* send_meta = send_payload + static_cast<int64_t>(num_channels) * row_ints;
    auto* send_dst_token = send_meta + num_channels;
    const int output_cols = 8;
    const int init_range =
        std::max({num_queue_slots * row_ints, num_queue_slots, num_channels * row_ints, num_channels, num_channels * output_cols});

    DebugChannelPutInit init_kernel{output,
                                    recv_payload,
                                    recv_meta,
                                    recv_dst_token,
                                    send_payload,
                                    send_meta,
                                    send_dst_token,
                                    num_queue_slots,
                                    row_ints,
                                    num_channels,
                                    output_cols,
                                    rank};
    queue.submit([&](sycl::handler& cgh) { cgh.parallel_for<DebugChannelPutInitKernel>(sycl::range<1>(init_range), init_kernel); });
    queue.wait();
    internode::mpi_barrier();

    for (int channel = 0; channel < num_channels; ++channel) {
        const int reset_range = std::max(num_queue_slots * row_ints, num_queue_slots);
        DebugChannelPutReset reset_kernel{recv_payload, recv_meta, recv_dst_token, num_queue_slots, row_ints};
        queue.submit([&](sycl::handler& cgh) { cgh.parallel_for<DebugChannelPutResetKernel>(sycl::range<1>(reset_range), reset_kernel); });
        queue.wait();
        internode::mpi_barrier();

        DebugChannelPutPost post_kernel{recv_payload,
                                        recv_meta,
                                        recv_dst_token,
                                        send_payload,
                                        send_meta,
                                        send_dst_token,
                                        row_ints,
                                        num_channels,
                                        queue_stride,
                                        rank,
                                        channel};
        queue.submit([&](sycl::handler& cgh) { cgh.single_task<DebugChannelPutKernel>(post_kernel); });
        queue.wait();
        internode::mpi_barrier();

        DebugChannelPutValidate validate_kernel{
            output, recv_payload, recv_meta, recv_dst_token, row_ints, num_channels, queue_stride, output_cols, rank, channel};
        queue.submit([&](sycl::handler& cgh) { cgh.single_task<DebugChannelPutValidateKernel>(validate_kernel); });
        queue.wait();
        internode::mpi_barrier();
    }
#else
    TORCH_CHECK(false, "debug_channel_put requires iSHMEM support");
#endif
}

}  // namespace internode
}  // namespace deep_ep
