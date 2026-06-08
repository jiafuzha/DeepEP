#include <c10/util/Float8_e4m3fn.h>

#include "xpu_kernels.hpp"

#ifdef DEEP_EP_ENABLE_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

namespace deep_ep {
namespace internode_ll {
namespace {

class CleanLowLatencyBufferKernel;
class UpdateMaskBufferKernel;
class QueryMaskBufferKernel;
class CleanMaskBufferKernel;
class LowLatencyDispatchMergedKernel;
class LowLatencyCastFp8Kernel;
class LowLatencyCombineMergedKernel;

struct LowLatencyLayout {
    size_t dispatch_data_bytes;
    size_t dispatch_src_bytes;
    size_t dispatch_count_bytes;
    size_t send_data_bytes;
    size_t send_src_bytes;
    size_t send_count_bytes;
    size_t combine_data_bytes;
    size_t combine_flag_bytes;
    size_t mask_bytes;

    size_t dispatch_data_offset;
    size_t dispatch_src_offset;
    size_t dispatch_count_offset;
    size_t send_data_offset;
    size_t send_src_offset;
    size_t send_count_offset;
    size_t combine_data_offset;
    size_t combine_flag_offset;
    size_t mask_offset;
    size_t sync_offset;
    size_t total_bytes;
};

inline LowLatencyLayout make_layout(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    const int num_local_experts = num_experts / num_ranks;
    LowLatencyLayout l{};
    const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    const size_t num_dispatch_slots = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
    const size_t num_send_slots = static_cast<size_t>(num_ranks) * num_local_experts * num_max_dispatch_tokens_per_rank;
    const size_t num_combine_slots = static_cast<size_t>(num_experts) * num_max_dispatch_tokens_per_rank;
    l.dispatch_data_bytes = num_dispatch_slots * hidden_bytes;
    l.dispatch_src_bytes = num_dispatch_slots * sizeof(int);
    l.dispatch_count_bytes = static_cast<size_t>(num_local_experts) * num_ranks * sizeof(int);
    l.send_data_bytes = num_send_slots * hidden_bytes;
    l.send_src_bytes = num_send_slots * sizeof(int);
    l.send_count_bytes = static_cast<size_t>(num_ranks) * num_local_experts * sizeof(int);
    l.combine_data_bytes = num_combine_slots * hidden_bytes;
    l.combine_flag_bytes = static_cast<size_t>(num_experts) * sizeof(uint64_t);
    l.mask_bytes = static_cast<size_t>(num_ranks) * sizeof(int);

    size_t offset = 0;
    auto add = [&](size_t bytes) {
        const size_t old = offset;
        offset = align_up<size_t>(offset + bytes, NUM_BUFFER_ALIGNMENT_BYTES);
        return old;
    };
    l.dispatch_data_offset = add(l.dispatch_data_bytes);
    l.dispatch_src_offset = add(l.dispatch_src_bytes);
    l.dispatch_count_offset = add(l.dispatch_count_bytes);
    l.send_data_offset = add(l.send_data_bytes);
    l.send_src_offset = add(l.send_src_bytes);
    l.send_count_offset = add(l.send_count_bytes);
    l.combine_data_offset = add(l.combine_data_bytes);
    l.combine_flag_offset = add(l.combine_flag_bytes);
    l.mask_offset = add(l.mask_bytes);
    l.sync_offset = add(static_cast<size_t>(num_ranks) * sizeof(int));
    l.total_bytes = offset;
    return l;
}

inline uint64_t pack_range(int count, int begin) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(begin)) << 32) | static_cast<uint32_t>(count);
}

inline void unpack_range(int64_t packed, int& count, int& begin) {
    uint64_t u = static_cast<uint64_t>(packed);
    count = static_cast<int>(u & 0xffffffffu);
    begin = static_cast<int>(u >> 32);
}

inline bool ll_rank_masked(int* mask_buffer_ptr, int rank) {
    return mask_buffer_ptr != nullptr && plain_load(mask_buffer_ptr + rank) != 0;
}

inline sycl::ext::oneapi::bfloat16 bf16_from_float(float value) {
    return sycl::ext::oneapi::bfloat16(value);
}

inline uint8_t ue8m0_from_float(float value) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(value);
    return static_cast<uint8_t>(bits >> 23);
}

constexpr int kLowLatencyMergedGroupSize = 256;

}  // namespace

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer_ptr,
                              int* sync_buffer_ptr,
                              sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    queue.wait();
    if (sync_buffer_ptr == nullptr) {
        internode::barrier();
    }
#endif
    const int total = num_clean_int_0 + num_clean_int_1;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CleanLowLatencyBufferKernel>(sycl::range<1>(total), [=](sycl::id<1> id) {
            int i = static_cast<int>(id[0]);
            if (i < num_clean_int_0) {
                clean_0[i] = 0;
            } else if (clean_1 != nullptr) {
                clean_1[i - num_clean_int_0] = 0;
            }
        });
    });
#ifdef DEEP_EP_ENABLE_ISHMEM
    queue.wait();
    if (sync_buffer_ptr == nullptr) {
        internode::barrier();
    }
#else
    (void)rank;
    (void)num_ranks;
    (void)mask_buffer_ptr;
    (void)sync_buffer_ptr;
#endif
}

void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask, sycl::queue& queue) {
    queue.submit(
        [&](sycl::handler& cgh) { cgh.single_task<UpdateMaskBufferKernel>([=]() { mask_buffer_ptr[rank_to_mask] = mask ? 1 : 0; }); });
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* output_mask_tensor, sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<QueryMaskBufferKernel>(sycl::range<1>(num_ranks), [=](sycl::id<1> id) {
            int rank = static_cast<int>(id[0]);
            output_mask_tensor[rank] = mask_buffer_ptr[rank];
        });
    });
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CleanMaskBufferKernel>(sycl::range<1>(num_ranks),
                                                [=](sycl::id<1> id) { mask_buffer_ptr[static_cast<int>(id[0])] = 0; });
    });
}

void dispatch_bf16(void* packed_recv_x,
                   int* packed_recv_src_info,
                   int64_t* packed_recv_layout_range,
                   int* packed_recv_count,
                   int* cumulative_local_expert_recv_stats,
                   int64_t* dispatch_wait_recv_cost_stats,
                   void* rdma_buffer,
                   int* mask_buffer_ptr,
                   const void* x,
                   const topk_idx_t* topk_idx,
                   int num_tokens,
                   int hidden,
                   int num_max_dispatch_tokens_per_rank,
                   int num_topk,
                   int num_experts,
                   int rank,
                   int num_ranks,
                   sycl::queue& queue) {
#ifndef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(false, "XPU low-latency dispatch requires iSHMEM support");
#else
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    auto layout = make_layout(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    const int num_local_experts = num_experts / num_ranks;
    const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    auto* base = static_cast<uint8_t*>(rdma_buffer);
    auto* dispatch_data = base + layout.dispatch_data_offset;
    auto* dispatch_src = reinterpret_cast<int*>(base + layout.dispatch_src_offset);
    auto* dispatch_count = reinterpret_cast<int*>(base + layout.dispatch_count_offset);
    auto* send_data = base + layout.send_data_offset;
    auto* send_src = reinterpret_cast<int*>(base + layout.send_src_offset);
    auto* send_count = reinterpret_cast<int*>(base + layout.send_count_offset);

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<LowLatencyDispatchMergedKernel>(
            sycl::nd_range<1>(sycl::range<1>(kLowLatencyMergedGroupSize), sycl::range<1>(kLowLatencyMergedGroupSize)),
            [=](sycl::nd_item<1> item) {
                auto group = item.get_group();
                const int local_id = static_cast<int>(item.get_local_id(0));
                const int local_size = static_cast<int>(item.get_local_range(0));
                const size_t send_count_elems = static_cast<size_t>(num_ranks) * num_local_experts;
                const size_t slot_elems = send_count_elems * num_max_dispatch_tokens_per_rank;

                // Zero only LOCAL staging and output tensors.
                // Symmetric buffer regions (dispatch_count, dispatch_src, dispatch_data)
                // are already zeroed by clean_low_latency_buffer with cross-PE barriers.
                for (size_t i = local_id; i < send_count_elems; i += local_size) {
                    send_count[i] = 0;
                }
                for (size_t i = local_id; i < slot_elems; i += local_size) {
                    send_src[i] = -1;
                    packed_recv_src_info[i] = -1;
                }
                for (int local_expert = local_id; local_expert < num_local_experts; local_expert += local_size) {
                    packed_recv_count[local_expert] = 0;
                }
                sycl::group_barrier(group);

                // No barrier_all needed here — clean_low_latency_buffer guarantees
                // all PEs have zeroed the symmetric receive buffers before dispatch.

                // Route tokens into local send staging
                for (int token_idx = local_id; token_idx < num_tokens; token_idx += local_size) {
                    for (int k = 0; k < num_topk; ++k) {
                        const int expert = static_cast<int>(topk_idx[token_idx * num_topk + k]);
                        if (expert < 0 || expert >= num_experts) {
                            continue;
                        }
                        const int dst_rank = expert / num_local_experts;
                        if (ll_rank_masked(mask_buffer_ptr, dst_rank)) {
                            continue;
                        }
                        const int local_expert = expert - dst_rank * num_local_experts;
                        sycl::atomic_ref<int,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::global_space>
                            count_ref(send_count[dst_rank * num_local_experts + local_expert]);
                        const int slot = count_ref.fetch_add(1);
                        if (slot >= num_max_dispatch_tokens_per_rank) {
                            continue;
                        }
                        const size_t packed_slot =
                            (static_cast<size_t>(dst_rank) * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank + slot;
                        auto* dst = send_data + packed_slot * hidden_bytes;
                        auto* src = static_cast<const uint8_t*>(x) + static_cast<size_t>(token_idx) * hidden_bytes;
                        for (size_t b = 0; b < hidden_bytes; ++b) {
                            dst[b] = src[b];
                        }
                        send_src[packed_slot] = token_idx;
                    }
                }
                sycl::group_barrier(group);

                // Phase 1: Issue data + src puts (NBI) to all remote PEs.
                // Local copies go directly into dispatch buffers.
                // Slots are contiguous in both send and dispatch layouts, so we
                // batch all slots per channel into a single large put.
                for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                    for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                        const size_t src_slot =
                            (static_cast<size_t>(dst_rank) * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank;
                        const size_t dst_slot = (static_cast<size_t>(local_expert) * num_ranks + rank) * num_max_dispatch_tokens_per_rank;
                        auto* src_count = send_count + dst_rank * num_local_experts + local_expert;
                        auto* dst_src = dispatch_src + dst_slot;
                        auto* src_src = send_src + src_slot;

                        if (dst_rank == rank) {
                            const int count = sycl::min(*src_count, num_max_dispatch_tokens_per_rank);
                            if (local_id == 0) {
                                dispatch_count[local_expert * num_ranks + rank] = count;
                            }
                            for (int slot = local_id; slot < num_max_dispatch_tokens_per_rank; slot += local_size) {
                                dst_src[slot] = src_src[slot];
                            }
                            auto* dst_ptr = dispatch_data + dst_slot * hidden_bytes;
                            auto* src_ptr = send_data + src_slot * hidden_bytes;
                            const size_t bytes = static_cast<size_t>(count) * hidden_bytes;
                            for (size_t b = local_id; b < bytes; b += local_size) {
                                dst_ptr[b] = src_ptr[b];
                            }
                        } else {
                            if (local_id == 0) {
                                *src_count = sycl::min(*src_count, num_max_dispatch_tokens_per_rank);
                            }
                            sycl::group_barrier(group);
                            const int count = *src_count;
                            // Put count to remote
                            auto* dst_count = dispatch_count + local_expert * num_ranks + rank;
                            ishmemx_int_put_nbi_work_group(dst_count, src_count, 1, dst_rank, group);
                            if (count > 0) {
                                // Batch src info into one put
                                ishmemx_putmem_nbi_work_group(
                                    dst_src, src_src, static_cast<size_t>(count) * sizeof(int), dst_rank, group);
                                // Batch all data slots into one put
                                auto* dst_ptr = dispatch_data + dst_slot * hidden_bytes;
                                auto* src_ptr = send_data + src_slot * hidden_bytes;
                                ishmemx_putmem_nbi_work_group(
                                    dst_ptr, src_ptr, static_cast<size_t>(count) * hidden_bytes, dst_rank, group);
                            }
                        }
                        sycl::group_barrier(group);
                    }
                }

                // Phase 2: Barrier ensures all PEs have completed their puts
                // (barrier_all internally does quiet + cross-PE synchronization)
                ishmemx_barrier_all_work_group(group);

                // Phase 3: Pack received data — batch contiguous ranges per channel
                for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                    int begin = 0;
                    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                        const int clamped_count =
                            sycl::min(dispatch_count[local_expert * num_ranks + src_rank], num_max_dispatch_tokens_per_rank);
                        packed_recv_layout_range[local_expert * num_ranks + src_rank] =
                            static_cast<int64_t>(pack_range(clamped_count, begin));
                        if (local_id == 0) {
                            packed_recv_count[local_expert] += clamped_count;
                            if (cumulative_local_expert_recv_stats != nullptr) {
                                cumulative_local_expert_recv_stats[local_expert] += clamped_count;
                            }
                            if (dispatch_wait_recv_cost_stats != nullptr && local_expert == 0) {
                                dispatch_wait_recv_cost_stats[src_rank] += 0;
                            }
                        }
                        if (clamped_count > 0) {
                            // Batch copy all data for this channel
                            const size_t src_base =
                                (static_cast<size_t>(local_expert) * num_ranks + src_rank) * num_max_dispatch_tokens_per_rank;
                            const size_t dst_base =
                                static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin;
                            auto* src = dispatch_data + src_base * hidden_bytes;
                            auto* dst = static_cast<uint8_t*>(packed_recv_x) + dst_base * hidden_bytes;
                            const size_t total_bytes = static_cast<size_t>(clamped_count) * hidden_bytes;
                            for (size_t b = local_id; b < total_bytes; b += local_size) {
                                dst[b] = src[b];
                            }
                            // Copy src_info for this channel
                            for (int slot = local_id; slot < clamped_count; slot += local_size) {
                                packed_recv_src_info[dst_base + slot] = dispatch_src[src_base + slot];
                            }
                        }
                        sycl::group_barrier(group);
                        begin += clamped_count;
                    }
                }
            });
    });
#endif
}

void cast_bf16_to_fp8(void* packed_recv_x,
                      void* packed_recv_x_scales,
                      const void* packed_recv_bf16,
                      const int* packed_recv_src_info,
                      int num_rows,
                      int hidden,
                      bool round_scale,
                      bool use_ue8m0,
                      sycl::queue& queue) {
    TORCH_CHECK(hidden % 128 == 0, "FP8 low-latency dispatch requires hidden to be divisible by 128");
    const int num_scales = hidden / 128;
    const int scale_packs = use_ue8m0 ? (num_scales + 3) / 4 : num_scales;
    auto* dst_fp8 = static_cast<uint8_t*>(packed_recv_x);
    auto* src_bf16 = static_cast<const sycl::ext::oneapi::bfloat16*>(packed_recv_bf16);
    auto* dst_scale_float = static_cast<float*>(packed_recv_x_scales);
    auto* dst_scale_int = static_cast<int32_t*>(packed_recv_x_scales);

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<LowLatencyCastFp8Kernel>(sycl::range<1>(static_cast<size_t>(num_rows) * num_scales), [=](sycl::id<1> id) {
            const int linear = static_cast<int>(id[0]);
            const int row = linear / num_scales;
            const int scale_idx = linear - row * num_scales;
            if (packed_recv_src_info[row] < 0) {
                return;
            }
            const int base_h = scale_idx * 128;
            float amax = 1.0e-4f;
            for (int i = 0; i < 128; ++i) {
                const float value = static_cast<float>(src_bf16[static_cast<size_t>(row) * hidden + base_h + i]);
                amax = sycl::fmax(amax, sycl::fabs(value));
            }

            float scale;
            float scale_inv;
            if (round_scale) {
                const float exp_scale_inv = sycl::ceil(sycl::log2(amax / 448.0f));
                scale = sycl::exp2(-exp_scale_inv);
                scale_inv = sycl::exp2(exp_scale_inv);
            } else {
                scale_inv = amax / 448.0f;
                scale = 448.0f / amax;
            }

            for (int i = 0; i < 128; ++i) {
                const float value = static_cast<float>(src_bf16[static_cast<size_t>(row) * hidden + base_h + i]) * scale;
                dst_fp8[static_cast<size_t>(row) * hidden + base_h + i] = c10::Float8_e4m3fn(value).x;
            }

            if (use_ue8m0) {
                const int pack_idx = scale_idx / 4;
                const int pack_shift = (scale_idx % 4) * 8;
                const int32_t scale_byte = static_cast<int32_t>(ue8m0_from_float(scale_inv)) << pack_shift;
                sycl::
                    atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>
                        scale_pack(dst_scale_int[static_cast<size_t>(row) * scale_packs + pack_idx]);
                scale_pack.fetch_or(scale_byte);
            } else {
                dst_scale_float[static_cast<size_t>(row) * num_scales + scale_idx] = scale_inv;
            }
        });
    });
}

void combine_bf16(void* combined_x,
                  void* rdma_buffer,
                  int* mask_buffer_ptr,
                  const void* x,
                  const topk_idx_t* topk_idx,
                  const float* topk_weights,
                  const int* src_info,
                  const int64_t* layout_range,
                  int64_t* combine_wait_recv_cost_stats,
                  int num_combined_tokens,
                  int hidden,
                  int num_max_dispatch_tokens_per_rank,
                  int num_topk,
                  int num_experts,
                  int rank,
                  int num_ranks,
                  sycl::queue& queue,
                  bool zero_copy) {
#ifndef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(false, "XPU low-latency combine requires iSHMEM support");
#else
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    (void)zero_copy;
    auto layout = make_layout(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    const int num_local_experts = num_experts / num_ranks;
    const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    auto* base = static_cast<uint8_t*>(rdma_buffer);
    auto* send_data = base + layout.send_data_offset;
    auto* combine_data = base + layout.combine_data_offset;
    (void)layout.combine_flag_offset;  // combine_flag zeroed by clean_low_latency_buffer

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<LowLatencyCombineMergedKernel>(
            sycl::nd_range<1>(sycl::range<1>(kLowLatencyMergedGroupSize), sycl::range<1>(kLowLatencyMergedGroupSize)),
            [=](sycl::nd_item<1> item) {
                auto group = item.get_group();
                const int local_id = static_cast<int>(item.get_local_id(0));
                const int local_size = static_cast<int>(item.get_local_range(0));
                const size_t send_elems = static_cast<size_t>(num_ranks) * num_local_experts * num_max_dispatch_tokens_per_rank;
                auto* send_bf16 = reinterpret_cast<sycl::ext::oneapi::bfloat16*>(send_data);

                // Zero only local send staging. combine_data and combine_flag are
                // already zeroed by clean_low_latency_buffer with cross-PE barriers.
                for (size_t i = local_id; i < send_elems * hidden; i += local_size) {
                    send_bf16[i] = sycl::ext::oneapi::bfloat16(0.0f);
                }
                sycl::group_barrier(group);

                // No barrier_all needed here — clean_low_latency_buffer guarantees
                // all PEs have zeroed combine_data/combine_flag before dispatch+combine.

                for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                        if (ll_rank_masked(mask_buffer_ptr, src_rank)) {
                            continue;
                        }
                        int count = 0, begin = 0;
                        unpack_range(layout_range[local_expert * num_ranks + src_rank], count, begin);
                        const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                        for (int slot = 0; slot < clamped_count; ++slot) {
                            const int original_token =
                                src_info[static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin + slot];
                            if (original_token < 0 || original_token >= num_max_dispatch_tokens_per_rank) {
                                continue;
                            }
                            auto* staged_dst = send_data +
                                (static_cast<size_t>(src_rank * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank +
                                 original_token) *
                                    hidden_bytes;
                            auto* src = static_cast<const uint8_t*>(x) +
                                (static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin + slot) *
                                    hidden_bytes;
                            for (size_t b = local_id; b < hidden_bytes; b += local_size) {
                                staged_dst[b] = src[b];
                            }
                        }
                    }
                }
                sycl::group_barrier(group);

                for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                    for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                        const int global_expert = rank * num_local_experts + local_expert;
                        int count = 0, begin = 0;
                        unpack_range(layout_range[local_expert * num_ranks + dst_rank], count, begin);
                        const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                        int min_token = num_max_dispatch_tokens_per_rank;
                        int max_token = -1;
                        for (int slot = 0; slot < clamped_count; ++slot) {
                            const int original_token =
                                src_info[static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin + slot];
                            if (original_token >= 0 && original_token < num_max_dispatch_tokens_per_rank) {
                                min_token = sycl::min(min_token, original_token);
                                max_token = sycl::max(max_token, original_token);
                            }
                        }
                        if (max_token < min_token) {
                            continue;
                        }
                        auto* src = send_data +
                            (static_cast<size_t>(dst_rank * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank +
                             min_token) *
                                hidden_bytes;
                        auto* dst = combine_data +
                            (static_cast<size_t>(global_expert) * num_max_dispatch_tokens_per_rank + min_token) * hidden_bytes;
                        const size_t bytes = static_cast<size_t>(max_token - min_token + 1) * hidden_bytes;
                        if (dst_rank == rank) {
                            for (size_t b = local_id; b < bytes; b += local_size) {
                                dst[b] = src[b];
                            }
                        } else {
                            ishmemx_putmem_nbi_work_group(dst, src, bytes, dst_rank, group);
                        }
                        sycl::group_barrier(group);
                    }
                }

                ishmemx_barrier_all_work_group(group);

                auto* out = static_cast<sycl::ext::oneapi::bfloat16*>(combined_x);
                const size_t reduce_work = static_cast<size_t>(num_combined_tokens) * hidden;
                for (size_t idx = local_id; idx < reduce_work; idx += local_size) {
                    const int token_idx = static_cast<int>(idx / hidden);
                    const int h = static_cast<int>(idx % hidden);
                    float acc = 0.0f;
                    for (int k = 0; k < num_topk; ++k) {
                        const int expert = static_cast<int>(topk_idx[token_idx * num_topk + k]);
                        if (expert < 0 || expert >= num_experts) {
                            continue;
                        }
                        const int src_rank = expert / num_local_experts;
                        if (ll_rank_masked(mask_buffer_ptr, src_rank)) {
                            continue;
                        }
                        const auto* value = reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(
                            combine_data + (static_cast<size_t>(expert) * num_max_dispatch_tokens_per_rank + token_idx) * hidden_bytes);
                        acc += static_cast<float>(value[h]) * topk_weights[token_idx * num_topk + k];
                    }
                    out[static_cast<size_t>(token_idx) * hidden + h] = bf16_from_float(acc);
                }
                if (combine_wait_recv_cost_stats != nullptr) {
                    for (int i = local_id; i < num_ranks; i += local_size) {
                        combine_wait_recv_cost_stats[i] += 0;
                    }
                }
            });
    });
#endif
}

}  // namespace internode_ll
}  // namespace deep_ep
