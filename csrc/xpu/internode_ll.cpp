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
class LowLatencyCombineMergedKernel;

// Tier-1 multi-work-group low-latency kernels (multi-WG grid + sub-group
// collectives). The merged single-256-WI-work-group kernels above are kept for
// reference / fallback; the active dispatch/combine paths now use these.
class LowLatencyDispatchRouteKernel;
class LowLatencyDispatchPutKernel;
class LowLatencyDispatchBarrierKernel;
class LowLatencyDispatchPackKernel;
class LowLatencyCombineScatterKernel;
class LowLatencyCombinePutKernel;
class LowLatencyCombineBarrierKernel;
class LowLatencyCombineReduceKernel;

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

// Tier-1 multi-WG tuning.
constexpr int kLLWGSize = 256;     // work-items per work-group
constexpr int kLLMaxWGs = 256;     // cap on grid size for grid-stride phases

// Maximum bytes per single iSHMEM NBI put. On this BMG + mlx5 IBGDA stack a
// single ishmem_putmem_nbi larger than ~256 KiB falls into a pathologically
// slow transport path (RERING / landing-spin) that costs a fixed ~0.5 s per
// call regardless of size. Splitting large puts into <=192 KiB chunks keeps
// every put on the fast path. Overridable via DEEP_EP_LL_MAX_PUT_KB.
inline size_t ll_max_put_bytes() {
    const char* env = std::getenv("DEEP_EP_LL_MAX_PUT_KB");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0) return static_cast<size_t>(v) * 1024;
    }
    return static_cast<size_t>(64) * 1024;
}

inline int ll_num_wgs(size_t work_units, int wg_size, int cap) {
    const char* env = std::getenv("DEEP_EP_LL_NUM_WGS");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0) return v;
    }
    size_t n = (work_units + static_cast<size_t>(wg_size) - 1) / static_cast<size_t>(wg_size);
    if (n < 1) n = 1;
    if (n > static_cast<size_t>(cap)) n = static_cast<size_t>(cap);
    return static_cast<int>(n);
}

// Cooperative copy of `n` bytes from src to dst using a strided set of
// cooperating lanes (lane in [0, lanes)). Uses 16-byte vector chunks when both
// pointers are 16-byte aligned (always true for the 128-byte-aligned symmetric
// heap regions and bf16/fp8 payloads here), falling back to bytes for any tail.
inline void coop_copy_bytes(uint8_t* dst, const uint8_t* src, size_t n, int lane, int lanes) {
    size_t done = 0;
    if ((reinterpret_cast<uintptr_t>(dst) & 0xF) == 0 && (reinterpret_cast<uintptr_t>(src) & 0xF) == 0) {
        const size_t n16 = n >> 4;
        auto* d16 = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst);
        auto* s16 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src);
        for (size_t j = static_cast<size_t>(lane); j < n16; j += static_cast<size_t>(lanes)) {
            d16[j] = s16[j];
        }
        done = n16 << 4;
    }
    for (size_t b = done + static_cast<size_t>(lane); b < n; b += static_cast<size_t>(lanes)) {
        dst[b] = src[b];
    }
}

#ifdef DEEP_EP_ENABLE_ISHMEM
// Issue an iSHMEM NBI put of `n` bytes from a single work-item, split into
// chunks no larger than `max_chunk` to stay on the IBGDA fast path (see
// ll_max_put_bytes). Chunk boundaries are 16-byte aligned for safety.
inline void chunked_put_nbi(uint8_t* dst, const uint8_t* src, size_t n, int dst_pe, size_t max_chunk) {
    if (max_chunk == 0 || n <= max_chunk) {
        ishmem_putmem_nbi(dst, src, n, dst_pe);
        return;
    }
    const size_t step = max_chunk & ~static_cast<size_t>(0xF);
    size_t off = 0;
    while (off < n) {
        const size_t this_bytes = sycl::min(step, n - off);
        ishmem_putmem_nbi(dst + off, src + off, this_bytes, dst_pe);
        off += this_bytes;
    }
}
#endif

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
                   void* packed_recv_x_scales,
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
                   bool use_fp8,
                   bool round_scale,
                   bool use_ue8m0,
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

    const size_t send_count_elems = static_cast<size_t>(num_ranks) * num_local_experts;
    const size_t slot_elems = send_count_elems * num_max_dispatch_tokens_per_rank;
    const size_t recv_src_elems = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
    const size_t max_put = ll_max_put_bytes();

    // --- Stage 0: zero LOCAL staging + output tensors (multi-WG via memset).
    // Symmetric receive buffers (dispatch_*) are zeroed by clean_low_latency_buffer
    // with cross-PE barriers. send_src / packed_recv_src_info use -1 (0xFF bytes).
    queue.memset(send_count, 0, send_count_elems * sizeof(int));
    queue.memset(packed_recv_count, 0, static_cast<size_t>(num_local_experts) * sizeof(int));
    queue.memset(send_src, 0xFF, slot_elems * sizeof(int));
    queue.memset(packed_recv_src_info, 0xFF, recv_src_elems * sizeof(int));

    // --- Stage 1: route tokens into local send staging (multi-WG, one sub-group
    // per (token, k) pair; the sub-group cooperatively copies the token payload).
    {
        const size_t num_pairs = static_cast<size_t>(num_tokens) * num_topk;
        const int num_wgs = ll_num_wgs(num_pairs, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchRouteKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    auto sg = item.get_sub_group();
                    const int sg_local = static_cast<int>(sg.get_local_linear_id());
                    const int sg_size = static_cast<int>(sg.get_local_range()[0]);
                    const int sgs_per_wg = static_cast<int>(sg.get_group_range()[0]);
                    const int global_sg =
                        static_cast<int>(item.get_group_linear_id()) * sgs_per_wg + static_cast<int>(sg.get_group_linear_id());
                    const int num_global_sgs = static_cast<int>(item.get_group_range(0)) * sgs_per_wg;

                    for (int p = global_sg; p < static_cast<int>(num_pairs); p += num_global_sgs) {
                        const int token_idx = p / num_topk;
                        const int k = p - token_idx * num_topk;
                        const int expert = static_cast<int>(topk_idx[token_idx * num_topk + k]);
                        if (expert < 0 || expert >= num_experts) {
                            continue;
                        }
                        const int dst_rank = expert / num_local_experts;
                        if (ll_rank_masked(mask_buffer_ptr, dst_rank)) {
                            continue;
                        }
                        const int local_expert = expert - dst_rank * num_local_experts;
                        // Single lane reserves the slot; broadcast to the sub-group.
                        int slot = -1;
                        if (sg_local == 0) {
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                count_ref(send_count[dst_rank * num_local_experts + local_expert]);
                            slot = count_ref.fetch_add(1);
                        }
                        slot = sycl::group_broadcast(sg, slot, 0);
                        if (slot >= num_max_dispatch_tokens_per_rank) {
                            continue;
                        }
                        const size_t packed_slot =
                            (static_cast<size_t>(dst_rank) * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank + slot;
                        coop_copy_bytes(send_data + packed_slot * hidden_bytes,
                                        static_cast<const uint8_t*>(x) + static_cast<size_t>(token_idx) * hidden_bytes,
                                        hidden_bytes, sg_local, sg_size);
                        if (sg_local == 0) {
                            send_src[packed_slot] = token_idx;
                        }
                    }
                });
        });
    }

    // --- Stage 2: local self-copy + remote NBI puts (multi-WG).
    // Local self-copy is heavily data-parallel across the whole grid; the remote
    // NBI put issuing is intentionally bounded to one work-item per channel
    // (num_experts channels) to avoid oversubscribing the single QP per PE.
    {
        const size_t self_bytes = static_cast<size_t>(num_local_experts) * num_max_dispatch_tokens_per_rank * hidden_bytes;
        const int num_wgs = ll_num_wgs(self_bytes >> 4, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchPutKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    const int gid = static_cast<int>(item.get_global_id(0));
                    const int gsize = static_cast<int>(item.get_global_range(0));

                    // Local (self) copy of dispatch_count / dispatch_src / dispatch_data.
                    for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                        const size_t src_slot =
                            (static_cast<size_t>(rank) * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank;
                        const size_t dst_slot =
                            (static_cast<size_t>(local_expert) * num_ranks + rank) * num_max_dispatch_tokens_per_rank;
                        const int count =
                            sycl::min(send_count[rank * num_local_experts + local_expert], num_max_dispatch_tokens_per_rank);
                        if (gid == 0) {
                            dispatch_count[local_expert * num_ranks + rank] = count;
                        }
                        for (int slot = gid; slot < num_max_dispatch_tokens_per_rank; slot += gsize) {
                            dispatch_src[dst_slot + slot] = send_src[src_slot + slot];
                        }
                        coop_copy_bytes(dispatch_data + dst_slot * hidden_bytes,
                                        send_data + src_slot * hidden_bytes,
                                        static_cast<size_t>(count) * hidden_bytes, gid, gsize);
                    }

                    // Remote NBI puts: one work-item per channel.
                    int ch = 0;
                    for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                        if (dst_rank == rank) {
                            continue;
                        }
                        for (int le = 0; le < num_local_experts; ++le) {
                            if (ch == gid) {
                                const int sc_idx = dst_rank * num_local_experts + le;
                                send_count[sc_idx] = sycl::min(send_count[sc_idx], num_max_dispatch_tokens_per_rank);
                                const int count = send_count[sc_idx];
                                const size_t src_slot = static_cast<size_t>(sc_idx) * num_max_dispatch_tokens_per_rank;
                                const size_t dst_slot =
                                    (static_cast<size_t>(le) * num_ranks + rank) * num_max_dispatch_tokens_per_rank;
                                ishmem_putmem_nbi(dispatch_count + le * num_ranks + rank, send_count + sc_idx, sizeof(int), dst_rank);
                                if (count > 0) {
                                    ishmem_putmem_nbi(dispatch_src + dst_slot, send_src + src_slot,
                                                      static_cast<size_t>(count) * sizeof(int), dst_rank);
                                    chunked_put_nbi(dispatch_data + dst_slot * hidden_bytes, send_data + src_slot * hidden_bytes,
                                                    static_cast<size_t>(count) * hidden_bytes, dst_rank, max_put);
                                }
                            }
                            ch++;
                        }
                    }
                });
        });
    }

    // --- Stage 3: cross-PE barrier (single WG). Drains the NBI puts (quiet) and
    // synchronizes all PEs before the receive/pack stage. Runs after Stage 2
    // completes on the in-order queue, so all puts have been issued.
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<LowLatencyDispatchBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(kLLWGSize), sycl::range<1>(kLLWGSize)),
            [=](sycl::nd_item<1> item) { ishmemx_barrier_all_work_group(item.get_group()); });
    });

    // --- Stage 4: pack received data. Each local expert is handled by a cohort of
    // `pack_wgs_per_expert` work-groups; the per-rank prefix (begin offsets) is
    // recomputed (cheaply) by every work-group, only the cohort leader writes the
    // per-channel layout/stats, and the payload copy / src_info / FP8 conversion are
    // partitioned across the whole cohort for parallelism.
    // When use_fp8 is set, the BF16 payload from the RDMA staging is converted to
    // FP8 (with per-128-channel scales) directly here — the FP8 cast is fused into
    // dispatch rather than performed as a separate pass over the packed output.
    {
        const int num_scales = (hidden % 128 == 0) ? hidden / 128 : 0;
        const int scale_packs = use_ue8m0 ? (num_scales + 3) / 4 : num_scales;
        auto* dst_fp8 = static_cast<uint8_t*>(packed_recv_x);
        auto* dst_scale_float = static_cast<float*>(packed_recv_x_scales);
        auto* dst_scale_int = static_cast<int32_t*>(packed_recv_x_scales);
        const int local_experts = num_local_experts > 0 ? num_local_experts : 1;
        const int pack_wgs_per_expert = sycl::max(1, sycl::min(kLLMaxWGs / local_experts, 32));
        const int num_wgs = local_experts * pack_wgs_per_expert;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchPackKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    const int local_size = static_cast<int>(item.get_local_range(0));
                    const int wg = static_cast<int>(item.get_group_linear_id());
                    const int local_expert = wg / pack_wgs_per_expert;
                    const int sub = wg % pack_wgs_per_expert;
                    if (local_expert >= num_local_experts) {
                        return;
                    }
                    const bool leader = (sub == 0 && local_id == 0);
                    // Cooperating-lane identity across the whole cohort for this expert.
                    const int cohort_id = sub * local_size + local_id;
                    const int cohort_size = pack_wgs_per_expert * local_size;

                    int begin = 0;
                    int total = 0;
                    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                        const int clamped_count =
                            sycl::min(dispatch_count[local_expert * num_ranks + src_rank], num_max_dispatch_tokens_per_rank);
                        if (leader) {
                            packed_recv_layout_range[local_expert * num_ranks + src_rank] =
                                static_cast<int64_t>(pack_range(clamped_count, begin));
                            if (cumulative_local_expert_recv_stats != nullptr) {
                                cumulative_local_expert_recv_stats[local_expert] += clamped_count;
                            }
                            if (dispatch_wait_recv_cost_stats != nullptr && local_expert == 0) {
                                dispatch_wait_recv_cost_stats[src_rank] += 0;
                            }
                        }
                        if (clamped_count > 0) {
                            const size_t src_base =
                                (static_cast<size_t>(local_expert) * num_ranks + src_rank) * num_max_dispatch_tokens_per_rank;
                            const size_t dst_base =
                                static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin;
                            if (use_fp8) {
                                // Per-(row, 128-channel block) BF16->FP8 conversion. Each lane
                                // owns one (row, scale block): compute amax, scale, write 128 FP8
                                // values and the per-block scale (float, or packed UE8M0 byte).
                                const size_t work = static_cast<size_t>(clamped_count) * num_scales;
                                for (size_t w = cohort_id; w < work; w += cohort_size) {
                                    const int local_row = static_cast<int>(w / num_scales);
                                    const int scale_idx = static_cast<int>(w % num_scales);
                                    const auto* src_bf16 = reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(
                                        dispatch_data + (src_base + local_row) * hidden_bytes);
                                    const size_t dst_row = dst_base + local_row;
                                    const int base_h = scale_idx * 128;
                                    float amax = 1.0e-4f;
                                    for (int i = 0; i < 128; ++i) {
                                        amax = sycl::fmax(amax, sycl::fabs(static_cast<float>(src_bf16[base_h + i])));
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
                                        const float value = static_cast<float>(src_bf16[base_h + i]) * scale;
                                        dst_fp8[dst_row * hidden + base_h + i] = c10::Float8_e4m3fn(value).x;
                                    }
                                    if (use_ue8m0) {
                                        const int pack_idx = scale_idx / 4;
                                        const int pack_shift = (scale_idx % 4) * 8;
                                        const int32_t scale_byte = static_cast<int32_t>(ue8m0_from_float(scale_inv)) << pack_shift;
                                        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            scale_pack(dst_scale_int[dst_row * scale_packs + pack_idx]);
                                        scale_pack.fetch_or(scale_byte);
                                    } else {
                                        dst_scale_float[dst_row * num_scales + scale_idx] = scale_inv;
                                    }
                                }
                            } else {
                                coop_copy_bytes(static_cast<uint8_t*>(packed_recv_x) + dst_base * hidden_bytes,
                                                dispatch_data + src_base * hidden_bytes,
                                                static_cast<size_t>(clamped_count) * hidden_bytes, cohort_id, cohort_size);
                            }
                            for (int slot = cohort_id; slot < clamped_count; slot += cohort_size) {
                                packed_recv_src_info[dst_base + slot] = dispatch_src[src_base + slot];
                            }
                        }
                        begin += clamped_count;
                        total += clamped_count;
                    }
                    if (leader) {
                        packed_recv_count[local_expert] = total;
                    }
                });
        });
    }
#endif
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

    const size_t send_elems = static_cast<size_t>(num_ranks) * num_local_experts * num_max_dispatch_tokens_per_rank;
    const size_t max_put = ll_max_put_bytes();

    // --- Stage 0: zero local send staging (bf16 zero == 0x0000). combine_data and
    // combine_flag are already zeroed by clean_low_latency_buffer (cross-PE barrier).
    queue.memset(send_data, 0, send_elems * hidden_bytes);

    // --- Stage 1: scatter the received expert payload (x) into the per-source send
    // staging slots indexed by (src_rank, local_expert, original_token). One
    // sub-group per (local_expert, src_rank, slot) triple cooperatively copies a row.
    {
        const size_t num_rows = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
        const int num_wgs = ll_num_wgs(num_rows, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombineScatterKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    auto sg = item.get_sub_group();
                    const int sg_local = static_cast<int>(sg.get_local_linear_id());
                    const int sg_size = static_cast<int>(sg.get_local_range()[0]);
                    const int sgs_per_wg = static_cast<int>(sg.get_group_range()[0]);
                    const int global_sg =
                        static_cast<int>(item.get_group_linear_id()) * sgs_per_wg + static_cast<int>(sg.get_group_linear_id());
                    const int num_global_sgs = static_cast<int>(item.get_group_range(0)) * sgs_per_wg;
                    const size_t rows_per_expert = static_cast<size_t>(num_ranks) * num_max_dispatch_tokens_per_rank;

                    for (size_t r = global_sg; r < num_rows; r += num_global_sgs) {
                        const int local_expert = static_cast<int>(r / rows_per_expert);
                        const int rem = static_cast<int>(r % rows_per_expert);
                        const int src_rank = rem / num_max_dispatch_tokens_per_rank;
                        const int slot = rem % num_max_dispatch_tokens_per_rank;
                        if (ll_rank_masked(mask_buffer_ptr, src_rank)) {
                            continue;
                        }
                        int count = 0, begin = 0;
                        unpack_range(layout_range[local_expert * num_ranks + src_rank], count, begin);
                        const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                        if (slot >= clamped_count) {
                            continue;
                        }
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
                        coop_copy_bytes(staged_dst, src, hidden_bytes, sg_local, sg_size);
                    }
                });
        });
    }

    // --- Stage 2: combine put — local self-copy (grid-parallel) + bounded remote
    // NBI puts (one work-item per channel) of the per-destination min/max token span.
    {
        const size_t self_bytes = static_cast<size_t>(num_local_experts) * num_max_dispatch_tokens_per_rank * hidden_bytes;
        const int num_wgs = ll_num_wgs(self_bytes >> 4, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombinePutKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    const int gid = static_cast<int>(item.get_global_id(0));
                    const int gsize = static_cast<int>(item.get_global_range(0));

                    // Step 1: local self-copy (dst_rank == rank), grid-parallel.
                    for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                        const int global_expert = rank * num_local_experts + local_expert;
                        int count = 0, begin = 0;
                        unpack_range(layout_range[local_expert * num_ranks + rank], count, begin);
                        const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                        int min_token = num_max_dispatch_tokens_per_rank;
                        int max_token = -1;
                        for (int slot = 0; slot < clamped_count; ++slot) {
                            const int original_token =
                                src_info[static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin +
                                         slot];
                            if (original_token >= 0 && original_token < num_max_dispatch_tokens_per_rank) {
                                min_token = sycl::min(min_token, original_token);
                                max_token = sycl::max(max_token, original_token);
                            }
                        }
                        if (max_token >= min_token) {
                            auto* src_ptr = send_data +
                                (static_cast<size_t>(rank * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank +
                                 min_token) *
                                    hidden_bytes;
                            auto* dst_ptr = combine_data +
                                (static_cast<size_t>(global_expert) * num_max_dispatch_tokens_per_rank + min_token) * hidden_bytes;
                            const size_t bytes = static_cast<size_t>(max_token - min_token + 1) * hidden_bytes;
                            coop_copy_bytes(dst_ptr, src_ptr, bytes, gid, gsize);
                        }
                    }

                    // Step 2: remote NBI puts, one work-item per channel.
                    int ch = 0;
                    for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                        if (dst_rank == rank) {
                            continue;
                        }
                        for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                            if (ch == gid) {
                                const int global_expert = rank * num_local_experts + local_expert;
                                int count = 0, begin = 0;
                                unpack_range(layout_range[local_expert * num_ranks + dst_rank], count, begin);
                                const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                                int min_token = num_max_dispatch_tokens_per_rank;
                                int max_token = -1;
                                for (int slot = 0; slot < clamped_count; ++slot) {
                                    const int original_token =
                                        src_info[static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank +
                                                 begin + slot];
                                    if (original_token >= 0 && original_token < num_max_dispatch_tokens_per_rank) {
                                        min_token = sycl::min(min_token, original_token);
                                        max_token = sycl::max(max_token, original_token);
                                    }
                                }
                                if (max_token >= min_token) {
                                    auto* src_ptr = send_data +
                                        (static_cast<size_t>(dst_rank * num_local_experts + local_expert) *
                                             num_max_dispatch_tokens_per_rank +
                                         min_token) *
                                            hidden_bytes;
                                    auto* dst_ptr = combine_data +
                                        (static_cast<size_t>(global_expert) * num_max_dispatch_tokens_per_rank + min_token) *
                                            hidden_bytes;
                                    const size_t bytes = static_cast<size_t>(max_token - min_token + 1) * hidden_bytes;
                                    chunked_put_nbi(dst_ptr, src_ptr, bytes, dst_rank, max_put);
                                }
                            }
                            ch++;
                        }
                    }
                });
        });
    }

    // --- Stage 3: cross-PE barrier (single WG). Drains NBI puts (quiet) + syncs PEs.
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<LowLatencyCombineBarrierKernel>(
            sycl::nd_range<1>(sycl::range<1>(kLLWGSize), sycl::range<1>(kLLWGSize)),
            [=](sycl::nd_item<1> item) { ishmemx_barrier_all_work_group(item.get_group()); });
    });

    // --- Stage 4: weighted reduction over top-k into combined_x (grid-parallel).
    {
        const size_t reduce_work = static_cast<size_t>(num_combined_tokens) * hidden;
        const int num_wgs = ll_num_wgs(reduce_work, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombineReduceKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    const int gid = static_cast<int>(item.get_global_id(0));
                    const int gsize = static_cast<int>(item.get_global_range(0));
                    auto* out = static_cast<sycl::ext::oneapi::bfloat16*>(combined_x);
                    for (size_t idx = gid; idx < reduce_work; idx += gsize) {
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
                                combine_data +
                                (static_cast<size_t>(expert) * num_max_dispatch_tokens_per_rank + token_idx) * hidden_bytes);
                            acc += static_cast<float>(value[h]) * topk_weights[token_idx * num_topk + k];
                        }
                        out[static_cast<size_t>(token_idx) * hidden + h] = bf16_from_float(acc);
                    }
                    if (combine_wait_recv_cost_stats != nullptr && gid < num_ranks) {
                        combine_wait_recv_cost_stats[gid] += 0;
                    }
                });
        });
    }
#endif
}

}  // namespace internode_ll
}  // namespace deep_ep
