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

class DispatchInitKernel;
class DispatchCountExchangeKernel;
class DispatchPackKernel;
class DispatchOffsetComputeKernel;
class DispatchPayloadKernel;

class DispatchQueueResetKernel;
class DispatchQueueCopyKernel;
class DispatchCopyKernel;
class DebugChannelPutInitKernel;
class DebugChannelPutResetKernel;
class DebugChannelPutKernel;
class DebugChannelPutValidateKernel;

// NVL-only internode dispatch kernels
class NvlDispatchInitKernel;
class NvlDispatchCountWriteKernel;
class NvlDispatchCountBarrierKernel;
class NvlDispatchOffsetKernel;
class NvlDispatchPackKernel;
class NvlDispatchPackBarrierKernel;
class NvlDispatchReceiveKernel;
class NvlDispatchCopyKernel;

class CombinedDispatchInitKernel;
class CombinedDispatchPackKernel;
class CombinedDispatchPackBarrierKernel;
class CombinedDispatchRdmaSendKernel;
class CombinedDispatchRdmaPutKernel;
class CombinedDispatchFwdWriteKernel;
class CombinedDispatchFwdBarrierKernel;
class CombinedDispatchAssembleKernel;
class CombinedDispatchHeadKernel;

// NVL-only internode combine kernels
template <typename dtype_t>
class NvlCombineInitKernel;
template <typename dtype_t>
class NvlCombineCountWriteKernel;
template <typename dtype_t>
class NvlCombinePackKernel;
template <typename dtype_t>
class NvlCombinePackBarrierKernel;
template <typename dtype_t>
class NvlCombineReduceKernel;
template <typename dtype_t>
class NvlCombineBiasKernel;

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
        ishmem_putmem_nbi(dst_payload, src_payload, static_cast<size_t>(row_ints) * sizeof(int), peer);
        ishmem_putmem_nbi(dst_token, src_token, sizeof(int), peer);
        ishmem_putmem_nbi(dst_meta, src_meta, sizeof(int), peer);
        ishmem_quiet();
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

namespace {
class QpWarmupKernel;
}  // namespace

// Warm up the IBGDA QPs / RC connections to every other RDMA peer before the
// first real dispatch.  On a cold QP the first device-issued RDMA write can be
// dropped (no PCIe TLP reaches the NIC until the connection is established),
// which surfaced as an iter-0 RDMA-half undercount / DEVICE_LOST.  Issuing a
// few small blocking puts (poll-to-CQE) here establishes the connection state
// so the first dispatch delivers reliably.
void warmup_qps(void* rdma_buffer_ptr,
                int my_rdma_rank,
                int num_rdma_ranks,
                int num_nvl_ranks,
                int nvl_rank,
                sycl::queue& queue) {
    if (num_rdma_ranks <= 1 || rdma_buffer_ptr == nullptr) {
        return;
    }
    // ONE round is sufficient: round 0 establishes the IBGDA RC connection
    // (the cold-QP handshake) and exchanges the first WQE/CQE so subsequent
    // dispatch puts find a warm QP.  Empirically (see [warmup_qps round N]
    // timings) round 0 takes ~5.6s on first init while rounds 1-3 each cost
    // ~700ms purely on ishmemx_barrier_all_work_group + kernel launch
    // overhead with no additional QP-warming benefit.
    constexpr int kWarmupRounds = 1;
    auto* base = static_cast<uint8_t*>(rdma_buffer_ptr);
    const int rdma_ranks = num_rdma_ranks;
    const int nvl_ranks = num_nvl_ranks;
    const int my_rdma = my_rdma_rank;
    const int my_nvl = nvl_rank;
    const char* twenv = std::getenv("DEEP_EP_TIME_WARMUP");
    const bool log = (my_rdma == 0 && my_nvl == 0) && (twenv != nullptr && twenv[0] != '\0');

    for (int round = 0; round < kWarmupRounds; ++round) {
        auto t0 = std::chrono::steady_clock::now();
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<QpWarmupKernel>(
                sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)),
                [=](sycl::nd_item<1> item) {
                    auto group = item.get_group();
                    if (my_nvl == 0 && group.get_local_linear_id() == 0) {
                        // Use the tail of the RDMA buffer as scratch so the
                        // warmup never collides with real dispatch regions.
                        auto* src = base;
                        auto* dst = base;
                        for (int dst_rdma = 0; dst_rdma < rdma_ranks; ++dst_rdma) {
                            if (dst_rdma == my_rdma) continue;
                            const int dst_pe = dst_rdma * nvl_ranks;
                            // Use non-blocking put: a device-side BLOCKING
                            // ishmem_putmem deadlocks here because its completion
                            // path needs host-proxy progress while the host
                            // thread is parked in queue.wait().  NBI rings the
                            // doorbell directly (ISHMEM_IBGDA_DIRECT_DOORBELL) and
                            // the work-group barrier below device-quiets to drain
                            // it — same pattern as the LL kernels.
                            ishmem_putmem_nbi(dst, src, 128, dst_pe);
                        }
                    }
                    sycl::group_barrier(group);
                    // Use the work-group collective barrier (every WI participates)
                    // rather than a single-WI ishmem_barrier_all().  On the cold-QP
                    // first init the single-WI device-wide ishmem_barrier_all()
                    // spins forever (it needs host-proxy progress while the host is
                    // parked in queue.wait()), deadlocking the warmup.  The
                    // work-group barrier is the proven path the LL/dispatch kernels
                    // use and it both cross-PE syncs and device-quiets the NBI puts.
                    ishmemx_barrier_all_work_group(group);
                    sycl::group_barrier(group);
                });
        });
        queue.wait();
        auto t1 = std::chrono::steady_clock::now();
        if (log) {
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::fprintf(stderr, "[warmup_qps round %d] %.3f ms\n", round, ms);
        }
    }
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

    for (int channel = 0; channel < num_channels; ++channel) {
        const int reset_range = std::max(num_queue_slots * row_ints, num_queue_slots);
        DebugChannelPutReset reset_kernel{recv_payload, recv_meta, recv_dst_token, num_queue_slots, row_ints};
        queue.submit([&](sycl::handler& cgh) { cgh.parallel_for<DebugChannelPutResetKernel>(sycl::range<1>(reset_range), reset_kernel); });
        queue.wait();

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

        DebugChannelPutValidate validate_kernel{
            output, recv_payload, recv_meta, recv_dst_token, row_ints, num_channels, queue_stride, output_cols, rank, channel};
        queue.submit([&](sycl::handler& cgh) { cgh.single_task<DebugChannelPutValidateKernel>(validate_kernel); });
        queue.wait();
    }
#else
    TORCH_CHECK(false, "debug_channel_put requires iSHMEM support");
#endif
}

// ===========================================================================
// NVL-only internode dispatch (no iSHMEM required)
// Used when num_nvl_bytes > 0 and num_rdma_ranks == 1 (single node).
// All communication via direct IPC buffer reads/writes with device barriers.
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

void dispatch_nvl(void* recv_x,
                  float* recv_x_scales,
                  topk_idx_t* recv_topk_idx,
                  float* recv_topk_weights,
                  void* recv_src_meta,
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
                  void** buffer_ptrs_gpu,
                  int** barrier_signal_ptrs_gpu,
                  int nvl_rank,
                  int num_nvl_ranks,
                  int barrier_signal_base,
                  int rank,
                  int num_ranks,
                  sycl::queue& queue) {
    TORCH_CHECK(recv_x != nullptr && x != nullptr, "NVL dispatch requires input and output tensors");
    TORCH_CHECK(buffer_ptrs_gpu != nullptr, "NVL dispatch requires NVL buffer pointers");
    TORCH_CHECK(barrier_signal_ptrs_gpu != nullptr, "NVL dispatch requires barrier signal pointers");
    TORCH_CHECK(is_token_in_rank != nullptr, "NVL dispatch requires is_token_in_rank");
    TORCH_CHECK(num_channels > 0, "NVL dispatch requires a positive channel count");
    TORCH_CHECK(element_size > 0, "NVL dispatch element size must be positive");
    TORCH_CHECK(num_nvl_ranks > 0 && num_nvl_ranks <= NUM_MAX_NVL_PEERS, "NVL peer count out of range");

    const size_t row_bytes = static_cast<size_t>(hidden) * element_size;
    const auto* src = static_cast<const uint8_t*>(x);
    auto* dst = static_cast<uint8_t*>(recv_x);
    auto* meta = static_cast<SourceMeta*>(recv_src_meta);

    NvlBufferLayout layout(num_tokens, num_ranks, num_channels, row_bytes, num_topk, num_scales);

    // Phase 1: Initialize receive buffers
    const size_t total_recv = static_cast<size_t>(num_recv_tokens);
    const size_t init_range = std::max({total_recv * (row_bytes / sizeof(uint8_t)),
                                        total_recv * sizeof(SourceMeta),
                                        static_cast<size_t>(num_ranks) * num_channels,
                                        static_cast<size_t>(num_tokens) * num_ranks,
                                        static_cast<size_t>(1)});
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NvlDispatchInitKernel>(sycl::range<1>(init_range), [=](sycl::id<1> id) {
            const size_t linear = id[0];
            if (linear < total_recv) {
                auto* meta_ptr = reinterpret_cast<SourceMeta*>(meta);
                if (meta_ptr) {
                    meta_ptr[linear].src_rdma_rank = -1;
                    meta_ptr[linear].is_token_in_nvl_rank_bits = 0;
                    meta_ptr[linear].src_nvl_rank = -1;
                }
            }
            if (linear < static_cast<size_t>(num_ranks) * num_channels && gbl_channel_prefix_matrix) {
                gbl_channel_prefix_matrix[linear] = 0;
            }
            if (send_rdma_head && linear < static_cast<size_t>(num_tokens) * num_ranks) {
                send_rdma_head[linear] = -1;
            }
            if (send_nvl_head && linear < total_recv * NUM_MAX_NVL_PEERS) {
                send_nvl_head[linear] = -1;
            }
        });
    });
    queue.wait();

    // Phase 2: Write count data to own NVL buffer, then device barrier
    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<NvlDispatchCountWriteKernel>([=]() {
            auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
            auto* my_counts = reinterpret_cast<int*>(my_buf + layout.count_offset);
            auto* my_channel_counts = reinterpret_cast<int*>(my_buf + layout.channel_count_offset);

            // Write per-destination token counts
            for (int d = 0; d < num_ranks; ++d) {
                if (num_tokens_per_rank != nullptr) {
                    my_counts[d] = num_tokens_per_rank[d];
                } else {
                    int count = 0;
                    for (int t = 0; t < num_tokens; ++t) {
                        count += is_token_in_rank[t * num_ranks + d] ? 1 : 0;
                    }
                    my_counts[d] = count;
                }
            }

            // Write per-destination per-channel counts
            for (int d = 0; d < num_ranks; ++d) {
                for (int c = 0; c < num_channels; ++c) {
                    const int ch_start = (static_cast<int64_t>(num_tokens) * c) / num_channels;
                    const int ch_end = (static_cast<int64_t>(num_tokens) * (c + 1)) / num_channels;
                    int count = 0;
                    for (int t = ch_start; t < ch_end; ++t) {
                        count += is_token_in_rank[t * num_ranks + d] ? 1 : 0;
                    }
                    my_channel_counts[d * num_channels + c] = count;
                }
            }
        });
    });
    queue.wait();

    // Device barrier: all NVL peers see count data
    const int barrier_count = barrier_signal_base;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NvlDispatchCountBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_count, num_nvl_ranks, item); });
    });
    queue.wait();

    // Phase 3: Read counts from all NVL peers and compute prefix sums
    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<NvlDispatchOffsetKernel>([=]() {
            // Read counts from all peers
            for (int peer = 0; peer < num_ranks; ++peer) {
                const int peer_nvl = peer % NUM_MAX_NVL_PEERS;
                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer_nvl]);
                auto* peer_channel_counts = reinterpret_cast<int*>(peer_buf + layout.channel_count_offset);

                for (int d = 0; d < num_ranks; ++d) {
                    for (int c = 0; c < num_channels; ++c) {
                        // In NVL-only mode, peer == peer_nvl, and we only read the row
                        // that this peer sends to destination d.
                        // For gbl_channel_prefix_matrix: [src_rank * num_channels + channel]
                        // But the CUDA layout is [global_rank * num_channels + channel] = count
                    }
                }
            }

            // For NVL-only, rdma_rank = 0 for all ranks, num_rdma_ranks = 1.
            // gbl_channel_prefix_matrix[src_rank * num_channels + c] = cumulative count from src to this rank
            // recv_gbl_rank_prefix_sum[src_rank] = cumulative total from src to this rank
            int gbl_prefix = 0;
            for (int src = 0; src < num_ranks; ++src) {
                const int src_nvl = src % NUM_MAX_NVL_PEERS;
                auto* src_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[src_nvl]);
                auto* src_channel_counts = reinterpret_cast<int*>(src_buf + layout.channel_count_offset);

                for (int c = 0; c < num_channels; ++c) {
                    int count = src_channel_counts[rank * num_channels + c];
                    gbl_prefix += count;
                    gbl_channel_prefix_matrix[src * num_channels + c] = gbl_prefix;
                }
                recv_gbl_rank_prefix_sum[src] = gbl_prefix;
            }

            // In NVL-only mode, rdma_channel_prefix_matrix is [1, num_channels] (one RDMA rank)
            // It stores cumulative tokens received from all ranks combined per channel
            int rdma_prefix = 0;
            for (int c = 0; c < num_channels; ++c) {
                int channel_total = 0;
                for (int src = 0; src < num_ranks; ++src) {
                    const int src_nvl = src % NUM_MAX_NVL_PEERS;
                    auto* src_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[src_nvl]);
                    auto* src_channel_counts = reinterpret_cast<int*>(src_buf + layout.channel_count_offset);
                    channel_total += src_channel_counts[rank * num_channels + c];
                }
                rdma_prefix += channel_total;
                rdma_channel_prefix_matrix[c] = rdma_prefix;
            }
            recv_rdma_rank_prefix_sum[0] = rdma_prefix;

            // recv_rdma_channel_prefix_matrix: [1, num_channels]
            if (recv_rdma_channel_prefix_matrix) {
                for (int c = 0; c < num_channels; ++c) {
                    recv_rdma_channel_prefix_matrix[c] = rdma_channel_prefix_matrix[c];
                }
            }

            // recv_gbl_channel_prefix_matrix: [num_ranks, num_channels]
            if (recv_gbl_channel_prefix_matrix) {
                for (int src = 0; src < num_ranks; ++src) {
                    for (int c = 0; c < num_channels; ++c) {
                        recv_gbl_channel_prefix_matrix[src * num_channels + c] = gbl_channel_prefix_matrix[src * num_channels + c];
                    }
                }
            }
        });
    });
    queue.wait();

    // Phase 4: Pack tokens into own NVL buffer
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NvlDispatchPackKernel>(sycl::range<1>(std::max(num_tokens, 1)), [=](sycl::id<1> id) {
            const int token = static_cast<int>(id[0]);
            if (token >= num_tokens)
                return;

            auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
            auto* my_send_x = my_buf + layout.send_x_offset;
            auto* my_send_meta = reinterpret_cast<SourceMeta*>(my_buf + layout.send_meta_offset);
            auto* my_send_topk_idx = reinterpret_cast<topk_idx_t*>(my_buf + layout.send_topk_idx_offset);
            auto* my_send_topk_weights = reinterpret_cast<float*>(my_buf + layout.send_topk_weights_offset);
            auto* my_send_x_scales = reinterpret_cast<float*>(my_buf + layout.send_x_scales_offset);
            auto* my_send_dst_token = reinterpret_cast<int*>(my_buf + layout.send_dst_token_offset);
            auto* my_send_routing_bits = reinterpret_cast<int*>(my_buf + layout.send_routing_bits_offset);

            // Copy payload
            const auto* src_row = src + static_cast<size_t>(token) * row_bytes;
            auto* dst_row = my_send_x + static_cast<size_t>(token) * row_bytes;
            for (size_t b = 0; b < row_bytes; ++b) {
                dst_row[b] = src_row[b];
            }

            // Write source metadata: {src_rank, src_token} — same format as RDMA dispatch
            my_send_meta[token] = SourceMeta{rank, token, nvl_rank};

            // Write per-token routing bitmask separately for the receive kernel
            int bits = 0;
            for (int nvl = 0; nvl < num_ranks && nvl < NUM_MAX_NVL_PEERS; ++nvl) {
                bits |= (is_token_in_rank[token * num_ranks + nvl] ? 1 : 0) << nvl;
            }
            my_send_routing_bits[token] = bits;

            // Copy topk
            if (topk_idx && topk_weights) {
                for (int k = 0; k < num_topk; ++k) {
                    my_send_topk_idx[token * num_topk + k] = topk_idx[token * num_topk + k];
                    my_send_topk_weights[token * num_topk + k] = topk_weights[token * num_topk + k];
                }
            }

            // Copy scales
            if (x_scales) {
                for (int s = 0; s < num_scales; ++s) {
                    my_send_x_scales[token * num_scales + s] = x_scales[token * num_scales + s];
                }
            }
        });
    });
    queue.wait();

    // Device barrier: all NVL peers see packed data
    const int barrier_pack = barrier_signal_base + 1;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NvlDispatchPackBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_pack, num_nvl_ranks, item); });
    });
    queue.wait();

    // Phase 5: Read from all peers' NVL buffers and copy to final recv tensor
    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<NvlDispatchReceiveKernel>([=]() {
            // For each source rank, iterate its tokens and copy those destined for us
            int recv_offset = 0;
            for (int src = 0; src < num_ranks; ++src) {
                const int src_nvl = src % NUM_MAX_NVL_PEERS;
                auto* src_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[src_nvl]);
                auto* src_counts = reinterpret_cast<int*>(src_buf + layout.count_offset);
                auto* src_send_x = src_buf + layout.send_x_offset;
                auto* src_send_meta = reinterpret_cast<SourceMeta*>(src_buf + layout.send_meta_offset);
                auto* src_send_topk_idx = reinterpret_cast<topk_idx_t*>(src_buf + layout.send_topk_idx_offset);
                auto* src_send_topk_weights = reinterpret_cast<float*>(src_buf + layout.send_topk_weights_offset);
                auto* src_send_x_scales = reinterpret_cast<float*>(src_buf + layout.send_x_scales_offset);
                auto* src_routing_bits = reinterpret_cast<int*>(src_buf + layout.send_routing_bits_offset);

                // Scan source tokens using the routing bitmask
                for (int t = 0; t < num_tokens; ++t) {
                    bool is_for_me = (src_routing_bits[t] >> nvl_rank) & 1;
                    if (!is_for_me)
                        continue;

                    // Copy to recv position
                    const int recv_idx = recv_offset;
                    recv_offset++;

                    // Copy payload
                    auto* src_row = src_send_x + static_cast<size_t>(t) * row_bytes;
                    auto* dst_row = dst + static_cast<size_t>(recv_idx) * row_bytes;
                    for (size_t b = 0; b < row_bytes; ++b) {
                        dst_row[b] = src_row[b];
                    }

                    // Copy metadata
                    if (meta) {
                        meta[recv_idx] = src_send_meta[t];
                    }

                    // Copy topk_idx
                    if (recv_topk_idx) {
                        for (int k = 0; k < num_topk; ++k) {
                            recv_topk_idx[recv_idx * num_topk + k] = src_send_topk_idx[t * num_topk + k];
                        }
                    }
                    if (recv_topk_weights) {
                        for (int k = 0; k < num_topk; ++k) {
                            recv_topk_weights[recv_idx * num_topk + k] = src_send_topk_weights[t * num_topk + k];
                        }
                    }

                    // Copy scales
                    if (recv_x_scales) {
                        for (int s = 0; s < num_scales; ++s) {
                            recv_x_scales[recv_idx * num_scales + s] = src_send_x_scales[t * num_scales + s];
                        }
                    }
                }
            }
        });
    });
    queue.wait();

    // Phase 6: Compute send_rdma_head from deterministic receive ordering
    // Each rank computes where its own tokens were placed on each destination rank.
    // The receive ordering is deterministic: destinations iterate sources 0..num_ranks-1
    // and within each source, tokens are iterated in original order.
    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<NvlDispatchCopyKernel>([=]() {
            for (int dst = 0; dst < num_ranks; ++dst) {
                // Compute prefix: number of tokens placed on dst before source rank
                int prefix = 0;
                for (int prior_src = 0; prior_src < rank; ++prior_src) {
                    const int prior_nvl = prior_src % NUM_MAX_NVL_PEERS;
                    auto* prior_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[prior_nvl]);
                    auto* prior_counts = reinterpret_cast<int*>(prior_buf + layout.count_offset);
                    prefix += prior_counts[dst];
                }

                // Iterate our own tokens, compute ordinal for those going to dst
                int ordinal = 0;
                for (int t = 0; t < num_tokens; ++t) {
                    if (!is_token_in_rank[t * num_ranks + dst])
                        continue;
                    if (send_rdma_head) {
                        send_rdma_head[t * num_ranks + dst] = prefix + ordinal;
                    }
                    ordinal++;
                }
            }
        });
    });
    queue.wait();
}

// Functor structs for NVL combine kernels (avoids oneAPI "Unexpected kernel lambda size")
template <typename dtype_t>
struct NvlCombineInitFunctor {
    dtype_t* dst;
    float* combined_topk_weights;
    size_t total_combined;
    size_t total_topk;

    void operator()(sycl::id<1> id) const {
        const size_t linear = id[0];
        if (linear < total_combined) {
            dst[linear] = static_cast<dtype_t>(0);
        }
        if (linear < total_topk && combined_topk_weights) {
            combined_topk_weights[linear] = 0.0f;
        }
    }
};

template <typename dtype_t>
struct NvlCombinePackFunctor {
    void** buffer_ptrs_gpu;
    const dtype_t* src;
    const float* topk_weights;
    int nvl_rank;
    int num_tokens;
    int hidden;
    int num_topk;
    size_t combine_x_offset;
    size_t combine_topk_offset;
    size_t combine_src_token_offset;

    void operator()(sycl::id<1> id) const {
        const int token = static_cast<int>(id[0]);
        if (token >= num_tokens)
            return;

        auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
        auto* combine_x = reinterpret_cast<dtype_t*>(my_buf + combine_x_offset);
        auto* combine_topk = reinterpret_cast<float*>(my_buf + combine_topk_offset);
        auto* combine_src_token = reinterpret_cast<int*>(my_buf + combine_src_token_offset);

        for (int h = 0; h < hidden; ++h) {
            combine_x[token * hidden + h] = src[token * hidden + h];
        }
        for (int k = 0; k < num_topk; ++k) {
            combine_topk[token * num_topk + k] = topk_weights ? topk_weights[token * num_topk + k] : 0.0f;
        }
        combine_src_token[token] = token;
    }
};

template <typename dtype_t>
struct NvlCombineCountWriteFunctor {
    void** buffer_ptrs_gpu;
    int nvl_rank;
    int num_tokens;
    size_t combine_count_offset;

    void operator()() const {
        auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
        auto* combine_count = reinterpret_cast<int*>(my_buf + combine_count_offset);
        *combine_count = num_tokens;
    }
};

template <typename dtype_t>
struct NvlCombineReduceFunctor {
    dtype_t* dst;
    float* combined_topk_weights;
    const bool* is_combined_token_in_rank;
    const int* combined_rdma_head;
    void** buffer_ptrs_gpu;
    int num_tokens;
    int num_combined_tokens;
    int hidden;
    int num_topk;
    int num_ranks;
    size_t combine_x_offset;
    size_t combine_topk_offset;
    size_t combine_count_offset;

    void operator()() const {
        for (int ct = 0; ct < num_combined_tokens; ++ct) {
            for (int peer = 0; peer < num_ranks; ++peer) {
                if (is_combined_token_in_rank && !is_combined_token_in_rank[ct * num_ranks + peer]) {
                    continue;
                }

                const int peer_nvl = peer % NUM_MAX_NVL_PEERS;
                auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[peer_nvl]);
                auto* peer_combine_x = reinterpret_cast<dtype_t*>(peer_buf + combine_x_offset);
                auto* peer_combine_topk = reinterpret_cast<float*>(peer_buf + combine_topk_offset);
                auto* peer_combine_count = reinterpret_cast<int*>(peer_buf + combine_count_offset);

                int peer_num_tokens = *peer_combine_count;

                const int peer_recv_pos = combined_rdma_head[ct * num_ranks + peer];
                if (peer_recv_pos < 0 || peer_recv_pos >= peer_num_tokens)
                    continue;

                for (int h = 0; h < hidden; ++h) {
                    float val = static_cast<float>(dst[ct * hidden + h]);
                    val += static_cast<float>(peer_combine_x[peer_recv_pos * hidden + h]);
                    dst[ct * hidden + h] = static_cast<dtype_t>(val);
                }
                if (combined_topk_weights) {
                    for (int k = 0; k < num_topk; ++k) {
                        combined_topk_weights[ct * num_topk + k] += peer_combine_topk[peer_recv_pos * num_topk + k];
                    }
                }
            }
        }
    }
};

template <typename dtype_t>
struct NvlCombineBiasFunctor {
    dtype_t* dst;
    const dtype_t* b0;
    const dtype_t* b1;
    int num_combined_tokens;
    int hidden;

    void operator()(sycl::id<1> id) const {
        const int ct = static_cast<int>(id[0]);
        if (ct >= num_combined_tokens)
            return;
        for (int h = 0; h < hidden; ++h) {
            float val = static_cast<float>(dst[ct * hidden + h]);
            if (b0)
                val += static_cast<float>(b0[ct * hidden + h]);
            if (b1)
                val += static_cast<float>(b1[ct * hidden + h]);
            dst[ct * hidden + h] = static_cast<dtype_t>(val);
        }
    }
};

template <typename dtype_t>
void launch_combine_nvl(void* combined_x,
                        float* combined_topk_weights,
                        const bool* is_combined_token_in_rank,
                        const void* x,
                        const float* topk_weights,
                        const void* bias_0,
                        const void* bias_1,
                        const int* combined_rdma_head,
                        const int* combined_nvl_head,
                        const void* src_meta_void,
                        const int* rdma_channel_prefix_matrix,
                        const int* rdma_rank_prefix_sum,
                        const int* gbl_channel_prefix_matrix,
                        int num_tokens,
                        int num_combined_tokens,
                        int hidden,
                        int num_topk,
                        void** buffer_ptrs_gpu,
                        int** barrier_signal_ptrs_gpu,
                        int nvl_rank,
                        int num_nvl_ranks,
                        int barrier_signal_base,
                        int rank,
                        int num_ranks,
                        sycl::queue& queue) {
    const auto* src = static_cast<const dtype_t*>(x);
    auto* dst = static_cast<dtype_t*>(combined_x);

    const size_t combine_base_offset = 4096;
    const size_t combine_x_offset = combine_base_offset;
    const size_t combine_topk_offset =
        align_offset(combine_x_offset + static_cast<size_t>(num_tokens) * hidden * sizeof(dtype_t), alignof(float));
    const size_t combine_src_token_offset =
        align_offset(combine_topk_offset + static_cast<size_t>(num_tokens) * num_topk * sizeof(float), alignof(int));
    const size_t combine_count_offset =
        align_offset(combine_src_token_offset + static_cast<size_t>(num_tokens) * sizeof(int), alignof(int));

    // Phase 1: Initialize combined output
    const size_t total_combined = static_cast<size_t>(num_combined_tokens) * hidden;
    const size_t total_topk = static_cast<size_t>(num_combined_tokens) * num_topk;
    const size_t init_range = std::max({total_combined, total_topk, static_cast<size_t>(1)});

    queue.submit([&](sycl::handler& cgh) {
        NvlCombineInitFunctor<dtype_t> fn{dst, combined_topk_weights, total_combined, total_topk};
        cgh.parallel_for<NvlCombineInitKernel<dtype_t>>(sycl::range<1>(init_range), fn);
    });
    queue.wait();

    // Phase 2: Each rank packs its combine contributions into its NVL buffer
    queue.submit([&](sycl::handler& cgh) {
        NvlCombinePackFunctor<dtype_t> fn{buffer_ptrs_gpu,
                                          src,
                                          topk_weights,
                                          nvl_rank,
                                          num_tokens,
                                          hidden,
                                          num_topk,
                                          combine_x_offset,
                                          combine_topk_offset,
                                          combine_src_token_offset};
        cgh.parallel_for<NvlCombinePackKernel<dtype_t>>(sycl::range<1>(std::max(num_tokens, 1)), fn);
    });
    queue.wait();

    // Write combine count to NVL buffer
    queue.submit([&](sycl::handler& cgh) {
        NvlCombineCountWriteFunctor<dtype_t> fn{buffer_ptrs_gpu, nvl_rank, num_tokens, combine_count_offset};
        cgh.single_task<NvlCombineCountWriteKernel<dtype_t>>(fn);
    });
    queue.wait();

    // Device barrier: all NVL peers see combine data
    const int barrier_combine = barrier_signal_base;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<NvlCombinePackBarrierKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_combine, num_nvl_ranks, item); });
    });
    queue.wait();

    // Phase 3: Each rank reads combine contributions from all peers and reduces
    queue.submit([&](sycl::handler& cgh) {
        NvlCombineReduceFunctor<dtype_t> fn{dst,
                                            combined_topk_weights,
                                            is_combined_token_in_rank,
                                            combined_rdma_head,
                                            buffer_ptrs_gpu,
                                            num_tokens,
                                            num_combined_tokens,
                                            hidden,
                                            num_topk,
                                            num_ranks,
                                            combine_x_offset,
                                            combine_topk_offset,
                                            combine_count_offset};
        cgh.single_task<NvlCombineReduceKernel<dtype_t>>(fn);
    });
    queue.wait();

    // Phase 4: Apply bias if present
    if (bias_0 || bias_1) {
        const auto* b0 = static_cast<const dtype_t*>(bias_0);
        const auto* b1 = static_cast<const dtype_t*>(bias_1);
        queue.submit([&](sycl::handler& cgh) {
            NvlCombineBiasFunctor<dtype_t> fn{dst, b0, b1, num_combined_tokens, hidden};
            cgh.parallel_for<NvlCombineBiasKernel<dtype_t>>(sycl::range<1>(std::max(num_combined_tokens, 1)), fn);
        });
        queue.wait();
    }
}

void combine_nvl(DataType type,
                 void* combined_x,
                 float* combined_topk_weights,
                 const bool* is_combined_token_in_rank,
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
                 void** buffer_ptrs_gpu,
                 int** barrier_signal_ptrs_gpu,
                 int nvl_rank,
                 int num_nvl_ranks,
                 int barrier_signal_base,
                 int rank,
                 int num_ranks,
                 sycl::queue& queue) {
    TORCH_CHECK(combined_x != nullptr && x != nullptr, "NVL combine requires input and output tensors");
    TORCH_CHECK(buffer_ptrs_gpu != nullptr, "NVL combine requires NVL buffer pointers");
    TORCH_CHECK(barrier_signal_ptrs_gpu != nullptr, "NVL combine requires barrier signal pointers");

    if (type == DataType::kBFloat16) {
        launch_combine_nvl<sycl::ext::oneapi::bfloat16>(combined_x,
                                                        combined_topk_weights,
                                                        is_combined_token_in_rank,
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
                                                        buffer_ptrs_gpu,
                                                        barrier_signal_ptrs_gpu,
                                                        nvl_rank,
                                                        num_nvl_ranks,
                                                        barrier_signal_base,
                                                        rank,
                                                        num_ranks,
                                                        queue);
    } else {
        TORCH_CHECK(false, "NVL combine only supports BFloat16 for now");
    }
}

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
//   Region A: NVL-local tokens (same NvlBufferLayout as NVL-only dispatch)
//   Region B: RDMA-forwarded tokens
struct NvlForwardLayout {
    size_t fwd_x_offset = 0;     // uint8[max_fwd_tokens * row_bytes]
    size_t fwd_meta_offset = 0;  // SourceMeta[max_fwd_tokens]
    size_t fwd_topk_idx_offset = 0;
    size_t fwd_topk_weights_offset = 0;
    size_t fwd_x_scales_offset = 0;
    size_t fwd_count_offset = 0;  // int[num_rdma_ranks]: per-source-rdma-rank count
    size_t total_bytes = 0;

    NvlForwardLayout(int max_fwd_tokens, size_t row_bytes, int num_topk, int num_scales, int num_rdma_ranks) {
        size_t off = 0;
        auto add_aligned = [&](size_t bytes, size_t align = 128) {
            off = align_offset(off, align);
            size_t result = off;
            off += bytes;
            return result;
        };
        fwd_x_offset = add_aligned(static_cast<size_t>(max_fwd_tokens) * row_bytes, 128);
        fwd_meta_offset = add_aligned(static_cast<size_t>(max_fwd_tokens) * sizeof(SourceMeta), alignof(SourceMeta));
        fwd_topk_idx_offset = add_aligned(static_cast<size_t>(max_fwd_tokens) * num_topk * sizeof(topk_idx_t), alignof(topk_idx_t));
        fwd_topk_weights_offset = add_aligned(static_cast<size_t>(max_fwd_tokens) * num_topk * sizeof(float), alignof(float));
        fwd_x_scales_offset = add_aligned(static_cast<size_t>(max_fwd_tokens) * num_scales * sizeof(float), alignof(float));
        fwd_count_offset = add_aligned(static_cast<size_t>(num_rdma_ranks) * sizeof(int), alignof(int));
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

    const auto* src = static_cast<const uint8_t*>(x);
    auto* dst = static_cast<uint8_t*>(recv_x);
    auto* meta = static_cast<SourceMeta*>(recv_src_meta);
    auto* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);

    NvlBufferLayout layout(num_tokens, num_ranks, 1, row_bytes, num_topk, num_scales);
    NvlForwardLayout fwd_layout(num_recv_tokens, row_bytes, num_topk, num_scales, num_rdma_ranks);
    const size_t fwd_base_offset = align_offset(layout.total_bytes, 128);

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
    const size_t rdma_region_bytes = align_offset(rdma_count_offset + sizeof(int), 128);
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
    queue.wait();
    ddbg_stage("1-Init");

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
    queue.wait();
    ddbg_stage("2-Pack");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchPackBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base, num_nvl_ranks, item); });
    });
    queue.wait();
    ddbg_stage("3-PackBarrier");

    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<CombinedDispatchRdmaSendKernel>([=]() {
            if (nvl_rank != 0) {
                return;
            }
            // Acquire fence: order reads of the NVL peers' packed send buffers
            // (IPC-mapped remote GPU memory) after their Pack release fence.
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
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
                    auto* peer_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[src_nvl]);
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
    queue.wait();
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
    queue.wait();
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
    queue.wait();
    ddbg_stage("6-FwdWrite");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedDispatchFwdBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 1, num_nvl_ranks, item); });
    });
    queue.wait();
    ddbg_stage("7-FwdBarrier");

    queue.submit([&](sycl::handler& cgh) {
        cgh.single_task<CombinedDispatchAssembleKernel>([=]() {
            // Acquire fence: invalidate any stale local cache and order all
            // subsequent reads of the NVL peers' forwarded/send buffers
            // (IPC-mapped remote GPU memory) after the leader's release fence,
            // mirroring CUDA's ld_acquire_sys_global on the NVL channel tail.
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            constexpr int kMaxRanks = 64;
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
            for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                fwd_total += my_fwd_counts[src_rdma];
            }
            int fwd_off_scan = 0;
            for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                const int count = my_fwd_counts[src_rdma];
                for (int i = 0; i < count; ++i) {
                    const SourceMeta sm = my_fwd_m[fwd_off_scan + i];
                    const int src_rank = sm.src_rdma_rank * num_nvl_ranks + sm.src_nvl_rank;
                    per_src_count[src_rank] += 1;
                }
                fwd_off_scan += count;
            }

            // Build exclusive prefix into cursors[] and inclusive prefix into output prefix-sum arrays.
            int total = 0;
            for (int s = 0; s < num_ranks; ++s) {
                cursors[s] = total;
                total += per_src_count[s];
            }
            if (recv_gbl_rank_prefix_sum != nullptr) {
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
            if (recv_rdma_rank_prefix_sum != nullptr) {
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
                    for (size_t b = 0; b < row_bytes; ++b) {
                        dst_row[b] = src_row[b];
                    }
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
            int fwd_offset = 0;
            for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                const int count = my_fwd_counts[src_rdma];
                for (int i = 0; i < count; ++i) {
                    const int idx = fwd_offset + i;
                    const SourceMeta sm = my_fwd_m[idx];
                    const int src_rank = sm.src_rdma_rank * num_nvl_ranks + sm.src_nvl_rank;
                    const int pos = cursors[src_rank]++;
                    auto* src_row = my_fwd_x + static_cast<size_t>(idx) * row_bytes;
                    auto* dst_row = dst + static_cast<size_t>(pos) * row_bytes;
                    for (size_t b = 0; b < row_bytes; ++b) {
                        dst_row[b] = src_row[b];
                    }
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
                fwd_offset += count;
            }

            // Zero-fill leftover rows beyond the actual receive count (capacity may exceed actual).
            for (int idx = total; idx < num_recv_tokens; ++idx) {
                auto* dst_row = dst + static_cast<size_t>(idx) * row_bytes;
                for (size_t b = 0; b < row_bytes; ++b) {
                    dst_row[b] = 0;
                }
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
        });
    });
    queue.wait();
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
                int scan = 0;
                for (int src_rdma = 0; src_rdma < num_rdma_ranks; ++src_rdma) {
                    const int n = dst_fwd_counts[src_rdma];
                    for (int i = 0; i < n; ++i) {
                        const SourceMeta sm = dst_fwd_m[scan + i];
                        const int src_rank = sm.src_rdma_rank * num_nvl_ranks + sm.src_nvl_rank;
                        per_src_per_dst_count[src_rank][dst_nvl] += 1;
                    }
                    scan += n;
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
                        if (is_my_token && send_rdma_head != nullptr) {
                            send_rdma_head[token * num_rdma_ranks + dst_rdma] = rdma_head[dst_rdma];
                        }
                        rdma_head[dst_rdma] += 1;
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
    const size_t rdma_region_bytes = align_offset(rdma_count_offset + sizeof(int), 128);
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
    dbg_stage("0-entry");

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
    queue.wait();
    dbg_stage("1-Init");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombinePackKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
            auto group = item.get_group();
            const int local_id = static_cast<int>(item.get_local_id(0));
            auto* my_buf = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]);
            auto* combine_count = reinterpret_cast<int*>(my_buf + layout.count_offset);
            auto* combine_x = reinterpret_cast<dtype_t*>(my_buf + layout.send_x_offset);
            auto* combine_meta = reinterpret_cast<SourceMeta*>(my_buf + layout.send_meta_offset);
            auto* combine_topk = reinterpret_cast<float*>(my_buf + layout.send_topk_weights_offset);
            if (local_id == 0) {
                *combine_count = num_tokens;
            }
            // Bulk payload copy parallelized across the work-group: each
            // work-item strides over the hidden dimension. Scalar per-token
            // metadata is written by work-item 0. All work-items execute the
            // same (deterministic) token loop so control flow stays uniform.
            for (int t = 0; t < num_tokens; ++t) {
                for (int h = local_id; h < hidden; h += kIshmemWGSize) {
                    combine_x[t * hidden + h] = src[t * hidden + h];
                }
                if (num_topk > 0) {
                    for (int k = local_id; k < num_topk; k += kIshmemWGSize) {
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
    queue.wait();
    dbg_stage("2-Pack");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombinePackBarrierKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base, num_nvl_ranks, item); });
    });
    queue.wait();
    dbg_stage("3-PackBarrier");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombineRdmaSendKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
            const int local_id = static_cast<int>(item.get_local_id(0));
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

            for (int ct = 0; ct < num_combined_tokens; ++ct) {
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
                    for (int h = local_id; h < hidden; h += kIshmemWGSize) {
                        float value = static_cast<float>(dst[ct * hidden + h]);
                        value += static_cast<float>(peer_x[peer_recv_pos * hidden + h]);
                        dst[ct * hidden + h] = static_cast<dtype_t>(value);
                    }
                    if (combined_topk_weights != nullptr) {
                        auto* peer_topk = reinterpret_cast<float*>(peer_buf + peer_topk_offset(peer_count));
                        for (int k = local_id; k < num_topk; k += kIshmemWGSize) {
                            combined_topk_weights[ct * num_topk + k] += peer_topk[peer_recv_pos * num_topk + k];
                        }
                    }
                }
            }
        });
    });
    queue.wait();
    dbg_stage("4-RdmaSend");

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
    queue.wait();
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
    queue.wait();
    dbg_stage("6-FwdWrite");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombineFwdBarrierKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(std::max(num_nvl_ranks, 32)), sycl::range<1>(std::max(num_nvl_ranks, 32))),
            [=](sycl::nd_item<1> item) { nvl_barrier(barrier_signal_ptrs_gpu, nvl_rank, barrier_signal_base + 1, num_nvl_ranks, item); });
    });
    queue.wait();
    dbg_stage("7-FwdBarrier");

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CombinedCombineReduceKernel<dtype_t>>(
            sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
            const int local_id = static_cast<int>(item.get_local_id(0));
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
            // Capacity of this rank's forwarded buffer, in tokens (must match
            // the max_fwd_tokens used to size fwd_layout and the FwdWrite guard).
            // The sum of all per-source counts can never legitimately exceed
            // this; a larger value is a corrupt/stale count that would drive an
            // unbounded, out-of-bounds reduction loop and wedge the GPU
            // (DEVICE_LOST). When counts are valid this clamp is a no-op.
            const int fwd_capacity = num_rdma_ranks * num_combined_tokens;
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
                for (int i = 0; i < count; ++i) {
                    const int idx = fwd_offset + i;
                    const int recv_pos = fwd_meta[idx].is_token_in_nvl_rank_bits;
                    for (int ct = 0; ct < num_combined_tokens; ++ct) {
                        if (combined_rdma_head[ct * num_rdma_ranks + src_rdma] != recv_pos) {
                            continue;
                        }
                        for (int h = local_id; h < hidden; h += kIshmemWGSize) {
                            float value = static_cast<float>(dst[ct * hidden + h]);
                            value += static_cast<float>(fwd_x[idx * hidden + h]);
                            dst[ct * hidden + h] = static_cast<dtype_t>(value);
                        }
                        if (combined_topk_weights != nullptr) {
                            for (int k = local_id; k < num_topk; k += kIshmemWGSize) {
                                combined_topk_weights[ct * num_topk + k] += fwd_topk[idx * num_topk + k];
                            }
                        }
                        break;
                    }
                }
                fwd_offset += count;
            }

            for (int ct = 0; ct < num_combined_tokens; ++ct) {
                for (int h = local_id; h < hidden; h += kIshmemWGSize) {
                    float value = static_cast<float>(dst[ct * hidden + h]);
                    if (b0 != nullptr) {
                        value += static_cast<float>(b0[ct * hidden + h]);
                    }
                    if (b1 != nullptr) {
                        value += static_cast<float>(b1[ct * hidden + h]);
                    }
                    dst[ct * hidden + h] = static_cast<dtype_t>(value);
                }
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
