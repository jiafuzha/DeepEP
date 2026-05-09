#if defined(DEEPEP_USE_XPU)
#include "configs.cuh"
#include "exception.cuh"

#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <vector>
#else
#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#endif

namespace deep_ep {

namespace intranode {

#if defined(DEEPEP_USE_XPU)

namespace {

struct XpuIntranodeNotifyArrival {
    runtime_stream_t stream;
    int rank;
    int num_ranks;
    int num_channels;
    int num_experts;
    int expert_alignment;
    std::vector<int> num_tokens_per_rank;
    std::vector<int> num_tokens_per_expert;
    std::vector<uint8_t> is_token_in_rank;
    int* moe_recv_counter_mapped;
    int* moe_recv_expert_counter_mapped;
    int* channel_prefix_matrix;
    int* rank_prefix_matrix_copy;
};

struct XpuIntranodeDispatchArrival {
    runtime_stream_t stream;
    int rank;
    int num_ranks;
    int num_channels;
    int num_tokens;
    int hidden_int4;
    int num_topk;
    int num_scales;
    int scale_token_stride;
    int scale_hidden_stride;
    void* recv_x;
    float* recv_x_scales;
    int* recv_src_idx;
    topk_idx_t* recv_topk_idx;
    float* recv_topk_weights;
    int* recv_channel_offset;
    int* send_head;
    std::vector<uint8_t> x;
    std::vector<uint8_t> is_token_in_rank;
    std::vector<float> x_scales;
    std::vector<topk_idx_t> topk_idx;
    std::vector<float> topk_weights;
};

struct XpuIntranodeCombineArrival {
    runtime_stream_t stream;
    int rank;
    int num_ranks;
    int num_tokens;
    int num_recv_tokens;
    int hidden;
    int num_topk;
    void* recv_x;
    float* recv_topk_weights;
    std::vector<uint8_t> x;
    std::vector<int> src_idx;
    std::vector<int> rank_prefix_matrix;
    std::vector<float> topk_weights;
    std::vector<uint8_t> bias_0;
    std::vector<uint8_t> bias_1;
};

template <typename Arrival>
struct XpuIntranodeRendezvousState {
    std::mutex mu;
    std::condition_variable cv;
    size_t generation = 0;
    int arrived = 0;
    std::vector<std::optional<Arrival>> arrivals;
};

template <typename Arrival>
std::shared_ptr<XpuIntranodeRendezvousState<Arrival>> get_xpu_intranode_state(
    std::unordered_map<uintptr_t, std::shared_ptr<XpuIntranodeRendezvousState<Arrival>>>& states,
    std::mutex& states_mu,
    uintptr_t key,
    int num_ranks) {
    std::lock_guard<std::mutex> guard(states_mu);
    auto& state = states[key];
    if (state == nullptr)
        state = std::make_shared<XpuIntranodeRendezvousState<Arrival>>();
    if (state->arrivals.empty())
        state->arrivals.resize(static_cast<size_t>(num_ranks));
    return state;
}

std::mutex xpu_intranode_notify_states_mu;
std::unordered_map<uintptr_t, std::shared_ptr<XpuIntranodeRendezvousState<XpuIntranodeNotifyArrival>>> xpu_intranode_notify_states;
std::mutex xpu_intranode_dispatch_states_mu;
std::unordered_map<uintptr_t, std::shared_ptr<XpuIntranodeRendezvousState<XpuIntranodeDispatchArrival>>> xpu_intranode_dispatch_states;
std::mutex xpu_intranode_combine_states_mu;
std::unordered_map<uintptr_t, std::shared_ptr<XpuIntranodeRendezvousState<XpuIntranodeCombineArrival>>> xpu_intranode_combine_states;
std::mutex xpu_intranode_memcpy_mu;

inline void xpu_blocking_memcpy(runtime_stream_t stream, void* dst, const void* src, size_t bytes) {
    if (bytes == 0)
        return;
    std::lock_guard<std::mutex> guard(xpu_intranode_memcpy_mu);
    auto event = stream.queue().memcpy(dst, src, bytes);
    event.wait_and_throw();
}

inline void get_channel_task_range_host(int num_tokens, int num_channels, int channel_id, int& token_start_idx, int& token_end_idx) {
    const int num_tokens_per_channel = (num_tokens + num_channels - 1) / num_channels;
    token_start_idx = std::min(num_tokens_per_channel * channel_id, num_tokens);
    token_end_idx = std::min(token_start_idx + num_tokens_per_channel, num_tokens);
}

inline int get_source_rank_from_rank_prefix(const std::vector<int>& rank_prefix_matrix, int num_ranks, int dst_rank, int row_idx) {
    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
        const int prefix = rank_prefix_matrix[static_cast<size_t>(src_rank) * static_cast<size_t>(num_ranks) + static_cast<size_t>(dst_rank)];
        if (row_idx < prefix)
            return src_rank;
    }
    return -1;
}

inline uintptr_t get_xpu_ipc_group_key(void** buffer_ptrs, int num_ranks) {
    if (buffer_ptrs == nullptr || num_ranks <= 0)
        return static_cast<uintptr_t>(num_ranks);

    // Hash imported peer pointer table so staged rendezvous follows IPC topology
    // instead of globally aliasing by world-size only.
    uintptr_t hash = static_cast<uintptr_t>(1469598103934665603ull);
    constexpr uintptr_t kFnvPrime = static_cast<uintptr_t>(1099511628211ull);
    for (int i = 0; i < num_ranks; ++i) {
        const uintptr_t value = reinterpret_cast<uintptr_t>(buffer_ptrs[i]);
        hash ^= value + static_cast<uintptr_t>(0x9e3779b97f4a7c15ull) + (hash << 6) + (hash >> 2);
        hash *= kFnvPrime;
    }
    return hash;
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
                     int,
                     int expert_alignment,
                     void** buffer_ptrs,
                     int**,
                     int rank,
                     runtime_stream_t stream,
                     int num_channels) {
    EP_HOST_ASSERT(num_experts > 0 and expert_alignment > 0 and num_channels > 0);
    EP_HOST_ASSERT(buffer_ptrs != nullptr);

    if (num_ranks == 1 and rank == 0) {
        int host_recv_tokens = 0;
        xpu_blocking_memcpy(stream, &host_recv_tokens, num_tokens_per_rank, sizeof(int));
        *moe_recv_counter_mapped = host_recv_tokens;

        std::vector<int> host_expert_counts(static_cast<size_t>(num_experts), 0);
        xpu_blocking_memcpy(stream, host_expert_counts.data(), num_tokens_per_expert, static_cast<size_t>(num_experts) * sizeof(int));
        for (int i = 0; i < num_experts; ++i) {
            const int aligned = ((host_expert_counts[i] + expert_alignment - 1) / expert_alignment) * expert_alignment;
            moe_recv_expert_counter_mapped[i] = aligned;
        }

        int host_rank_prefix = host_recv_tokens;
        xpu_blocking_memcpy(stream, rank_prefix_matrix_copy, &host_rank_prefix, sizeof(int));

        std::vector<uint8_t> host_is_token_in_rank(static_cast<size_t>(num_tokens), 0);
        if (num_tokens > 0)
            xpu_blocking_memcpy(stream, host_is_token_in_rank.data(), is_token_in_rank, static_cast<size_t>(num_tokens) * sizeof(uint8_t));

        std::vector<int> host_channel_prefix(static_cast<size_t>(num_channels), 0);
        for (int c = 0; c < num_channels; ++c) {
            int token_start = 0, token_end = 0;
            get_channel_task_range_host(num_tokens, num_channels, c, token_start, token_end);
            int count = 0;
            for (int t = token_start; t < token_end; ++t)
                count += static_cast<int>(host_is_token_in_rank[t] != 0);
            host_channel_prefix[c] = count;
        }
        for (int c = 1; c < num_channels; ++c)
            host_channel_prefix[c] += host_channel_prefix[c - 1];

        xpu_blocking_memcpy(stream, channel_prefix_matrix, host_channel_prefix.data(), static_cast<size_t>(num_channels) * sizeof(int));
        return;
    }

    EP_HOST_ASSERT(rank >= 0 and rank < num_ranks);

    std::vector<int> host_num_tokens_per_rank(static_cast<size_t>(num_ranks), 0);
    xpu_blocking_memcpy(stream, host_num_tokens_per_rank.data(), num_tokens_per_rank, static_cast<size_t>(num_ranks) * sizeof(int));

    std::vector<int> host_num_tokens_per_expert(static_cast<size_t>(num_experts), 0);
    xpu_blocking_memcpy(stream, host_num_tokens_per_expert.data(), num_tokens_per_expert, static_cast<size_t>(num_experts) * sizeof(int));

    std::vector<uint8_t> host_is_token_in_rank(static_cast<size_t>(num_tokens) * static_cast<size_t>(num_ranks), 0);
    if (num_tokens > 0)
        xpu_blocking_memcpy(stream, host_is_token_in_rank.data(), is_token_in_rank, host_is_token_in_rank.size() * sizeof(uint8_t));

    const uintptr_t group_key = get_xpu_ipc_group_key(buffer_ptrs, num_ranks);
    auto state = get_xpu_intranode_state(xpu_intranode_notify_states, xpu_intranode_notify_states_mu, group_key, num_ranks);
    std::unique_lock<std::mutex> lock(state->mu);
    const size_t generation = state->generation;
    state->arrivals[static_cast<size_t>(rank)] = XpuIntranodeNotifyArrival{stream,
                                                                           rank,
                                                                           num_ranks,
                                                                           num_channels,
                                                                           num_experts,
                                                                           expert_alignment,
                                                                           std::move(host_num_tokens_per_rank),
                                                                           std::move(host_num_tokens_per_expert),
                                                                           std::move(host_is_token_in_rank),
                                                                           moe_recv_counter_mapped,
                                                                           moe_recv_expert_counter_mapped,
                                                                           channel_prefix_matrix,
                                                                           rank_prefix_matrix_copy};
    state->arrived++;

    if (state->arrived < num_ranks) {
        state->cv.wait(lock, [&] { return state->generation != generation; });
        return;
    }

    const int num_experts_per_rank = num_experts / num_ranks;
    EP_HOST_ASSERT(num_experts_per_rank > 0 and num_experts % num_ranks == 0);

    std::vector<XpuIntranodeNotifyArrival> arrivals;
    arrivals.reserve(static_cast<size_t>(num_ranks));
    for (auto& arrival_opt : state->arrivals) {
        EP_HOST_ASSERT(arrival_opt.has_value());
        arrivals.push_back(std::move(arrival_opt.value()));
        arrival_opt.reset();
    }
    state->arrived = 0;

    std::vector<int> rank_prefix_matrix(static_cast<size_t>(num_ranks) * static_cast<size_t>(num_ranks), 0);
    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
        for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
            rank_prefix_matrix[static_cast<size_t>(src_rank) * static_cast<size_t>(num_ranks) + static_cast<size_t>(dst_rank)] =
                arrivals[static_cast<size_t>(src_rank)].num_tokens_per_rank[static_cast<size_t>(dst_rank)];
        }
    }
    for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
        for (int src_rank = 1; src_rank < num_ranks; ++src_rank) {
            rank_prefix_matrix[static_cast<size_t>(src_rank) * static_cast<size_t>(num_ranks) + static_cast<size_t>(dst_rank)] +=
                rank_prefix_matrix[static_cast<size_t>(src_rank - 1) * static_cast<size_t>(num_ranks) + static_cast<size_t>(dst_rank)];
        }
    }

    for (int recv_rank = 0; recv_rank < num_ranks; ++recv_rank) {
        auto& arrival = arrivals[static_cast<size_t>(recv_rank)];
        arrival.moe_recv_counter_mapped[0] = rank_prefix_matrix[static_cast<size_t>(num_ranks - 1) * static_cast<size_t>(num_ranks) +
                                                                static_cast<size_t>(recv_rank)];
        for (int local_expert = 0; local_expert < num_experts_per_rank; ++local_expert) {
            int sum = 0;
            const int global_expert = recv_rank * num_experts_per_rank + local_expert;
            for (int src_rank = 0; src_rank < num_ranks; ++src_rank)
                sum += arrivals[static_cast<size_t>(src_rank)].num_tokens_per_expert[static_cast<size_t>(global_expert)];
            arrival.moe_recv_expert_counter_mapped[local_expert] =
                ((sum + arrival.expert_alignment - 1) / arrival.expert_alignment) * arrival.expert_alignment;
        }

        xpu_blocking_memcpy(
            arrival.stream, arrival.rank_prefix_matrix_copy, rank_prefix_matrix.data(), rank_prefix_matrix.size() * sizeof(int));
    }

    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
        auto& arrival = arrivals[static_cast<size_t>(src_rank)];
        std::vector<int> channel_prefix(static_cast<size_t>(num_ranks) * static_cast<size_t>(arrival.num_channels), 0);
        const int src_num_tokens = static_cast<int>(arrival.is_token_in_rank.size() / static_cast<size_t>(num_ranks));
        for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
            for (int channel = 0; channel < arrival.num_channels; ++channel) {
                int token_start = 0, token_end = 0;
                get_channel_task_range_host(src_num_tokens, arrival.num_channels, channel, token_start, token_end);
                int count = 0;
                for (int token = token_start; token < token_end; ++token) {
                    count += static_cast<int>(
                        arrival.is_token_in_rank[static_cast<size_t>(token) * static_cast<size_t>(num_ranks) + static_cast<size_t>(dst_rank)] != 0);
                }
                channel_prefix[static_cast<size_t>(dst_rank) * static_cast<size_t>(arrival.num_channels) + static_cast<size_t>(channel)] = count;
            }
            for (int channel = 1; channel < arrival.num_channels; ++channel) {
                channel_prefix[static_cast<size_t>(dst_rank) * static_cast<size_t>(arrival.num_channels) + static_cast<size_t>(channel)] +=
                    channel_prefix[static_cast<size_t>(dst_rank) * static_cast<size_t>(arrival.num_channels) + static_cast<size_t>(channel - 1)];
            }
        }
        xpu_blocking_memcpy(
            arrival.stream, arrival.channel_prefix_matrix, channel_prefix.data(), channel_prefix.size() * sizeof(int));
    }

    state->generation++;
    lock.unlock();
    state->cv.notify_all();
}

void cached_notify_dispatch(const int* rank_prefix_matrix, int, void**, int**, int rank, int num_ranks, runtime_stream_t stream) {
    EP_HOST_ASSERT(rank >= 0 and rank < num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix != nullptr);

    // Host-staged XPU dispatch uses cached prefix matrices directly and does not
    // consume in-buffer queue metadata. Keep stream ordering consistent.
    auto noop = stream.queue().memcpy(const_cast<int*>(rank_prefix_matrix), rank_prefix_matrix, sizeof(int));
    noop.wait_and_throw();
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
              const int*,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_topk,
              int,
              int num_scales,
              int scale_token_stride,
              int scale_hidden_stride,
              void** buffer_ptrs,
              int rank,
              int num_ranks,
              runtime_stream_t stream,
              int num_sms,
              int,
              int) {
    EP_HOST_ASSERT(recv_x != nullptr and recv_src_idx != nullptr and recv_channel_offset != nullptr and send_head != nullptr);
    EP_HOST_ASSERT(x != nullptr and is_token_in_rank != nullptr);
    EP_HOST_ASSERT(buffer_ptrs != nullptr);
    EP_HOST_ASSERT(num_tokens >= 0 and hidden_int4 >= 0 and num_sms > 0);
    EP_HOST_ASSERT(num_sms % 2 == 0 and "num_sms must be even for intranode dispatch");

    const int num_channels = num_sms / 2;

    const size_t row_bytes = static_cast<size_t>(hidden_int4) * sizeof(int4);
    const size_t x_bytes = static_cast<size_t>(num_tokens) * row_bytes;
    std::vector<uint8_t> host_x(x_bytes);
    if (x_bytes > 0)
        xpu_blocking_memcpy(stream, host_x.data(), x, x_bytes);

    const size_t is_token_in_rank_elems = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_ranks);
    std::vector<uint8_t> host_is_token_in_rank(is_token_in_rank_elems, 0);
    if (is_token_in_rank_elems > 0)
        xpu_blocking_memcpy(stream, host_is_token_in_rank.data(), is_token_in_rank, is_token_in_rank_elems * sizeof(uint8_t));

    std::vector<float> host_x_scales;
    if (x_scales != nullptr and recv_x_scales != nullptr and num_scales > 0) {
        const size_t scale_src_elems = static_cast<size_t>(num_tokens) * static_cast<size_t>(std::max(1, scale_token_stride));
        host_x_scales.resize(scale_src_elems);
        xpu_blocking_memcpy(stream, host_x_scales.data(), x_scales, scale_src_elems * sizeof(float));
    }

    std::vector<topk_idx_t> host_topk_idx;
    if (topk_idx != nullptr and recv_topk_idx != nullptr and num_topk > 0) {
        const size_t topk_elems = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_topk);
        host_topk_idx.resize(topk_elems);
        xpu_blocking_memcpy(stream, host_topk_idx.data(), topk_idx, topk_elems * sizeof(topk_idx_t));
    }

    std::vector<float> host_topk_weights;
    if (topk_weights != nullptr and recv_topk_weights != nullptr and num_topk > 0) {
        const size_t topk_weight_elems = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_topk);
        host_topk_weights.resize(topk_weight_elems);
        xpu_blocking_memcpy(stream, host_topk_weights.data(), topk_weights, topk_weight_elems * sizeof(float));
    }

    if (num_ranks == 1 and rank == 0) {
        std::vector<int> selected_src_indices;
        selected_src_indices.reserve(static_cast<size_t>(num_tokens));
        for (int t = 0; t < num_tokens; ++t) {
            if (host_is_token_in_rank[t] != 0)
                selected_src_indices.push_back(t);
        }

        int num_recv_tokens = num_worst_tokens > 0 ? num_worst_tokens : static_cast<int>(selected_src_indices.size());
        if (num_worst_tokens > 0 and static_cast<int>(selected_src_indices.size()) > num_recv_tokens)
            selected_src_indices.resize(static_cast<size_t>(num_recv_tokens));

        std::vector<uint8_t> host_recv_x(static_cast<size_t>(num_recv_tokens) * row_bytes, 0);
        std::vector<int> host_recv_src_idx(static_cast<size_t>(num_recv_tokens), -1);
        std::vector<float> host_recv_x_scales;
        if (recv_x_scales != nullptr and num_scales > 0)
            host_recv_x_scales.assign(static_cast<size_t>(num_recv_tokens) * static_cast<size_t>(num_scales), 0.f);
        std::vector<topk_idx_t> host_recv_topk_idx;
        if (recv_topk_idx != nullptr and num_topk > 0)
            host_recv_topk_idx.assign(static_cast<size_t>(num_recv_tokens) * static_cast<size_t>(num_topk), static_cast<topk_idx_t>(-1));
        std::vector<float> host_recv_topk_weights;
        if (recv_topk_weights != nullptr and num_topk > 0)
            host_recv_topk_weights.assign(static_cast<size_t>(num_recv_tokens) * static_cast<size_t>(num_topk), 0.f);

        for (int out_idx = 0; out_idx < static_cast<int>(selected_src_indices.size()); ++out_idx) {
            const int src_idx = selected_src_indices[static_cast<size_t>(out_idx)];
            host_recv_src_idx[static_cast<size_t>(out_idx)] = src_idx;

            if (row_bytes > 0) {
                std::memcpy(host_recv_x.data() + static_cast<size_t>(out_idx) * row_bytes,
                            host_x.data() + static_cast<size_t>(src_idx) * row_bytes,
                            row_bytes);
            }

            if (recv_x_scales != nullptr and num_scales > 0 and !host_x_scales.empty()) {
                for (int s = 0; s < num_scales; ++s) {
                    host_recv_x_scales[static_cast<size_t>(out_idx) * static_cast<size_t>(num_scales) + static_cast<size_t>(s)] =
                        host_x_scales[static_cast<size_t>(src_idx) * static_cast<size_t>(scale_token_stride) +
                                      static_cast<size_t>(s) * static_cast<size_t>(scale_hidden_stride)];
                }
            }

            if (recv_topk_idx != nullptr and num_topk > 0 and !host_topk_idx.empty()) {
                std::memcpy(host_recv_topk_idx.data() + static_cast<size_t>(out_idx) * static_cast<size_t>(num_topk),
                            host_topk_idx.data() + static_cast<size_t>(src_idx) * static_cast<size_t>(num_topk),
                            static_cast<size_t>(num_topk) * sizeof(topk_idx_t));
            }

            if (recv_topk_weights != nullptr and num_topk > 0 and !host_topk_weights.empty()) {
                std::memcpy(host_recv_topk_weights.data() + static_cast<size_t>(out_idx) * static_cast<size_t>(num_topk),
                            host_topk_weights.data() + static_cast<size_t>(src_idx) * static_cast<size_t>(num_topk),
                            static_cast<size_t>(num_topk) * sizeof(float));
            }
        }

        std::vector<int> host_recv_channel_offset(static_cast<size_t>(num_channels), 0);
        for (int c = 0; c < num_channels; ++c) {
            int token_start = 0, token_end = 0;
            get_channel_task_range_host(num_tokens, num_channels, c, token_start, token_end);
            int count = 0;
            for (const int src_idx : selected_src_indices)
                count += (src_idx >= token_start and src_idx < token_end) ? 1 : 0;
            host_recv_channel_offset[static_cast<size_t>(c)] = count;
        }
        for (int c = 1; c < num_channels; ++c)
            host_recv_channel_offset[static_cast<size_t>(c)] += host_recv_channel_offset[static_cast<size_t>(c - 1)];

        std::vector<int> host_send_head(static_cast<size_t>(num_tokens), 0);
        int sent = 0;
        for (int t = 0; t < num_tokens; ++t) {
            if (host_is_token_in_rank[t] != 0)
                ++sent;
            host_send_head[static_cast<size_t>(t)] = sent;
        }

        if (!host_recv_x.empty()) {
            auto recv_x_event = stream.queue().memcpy(recv_x, host_recv_x.data(), host_recv_x.size());
            recv_x_event.wait_and_throw();
        }
        if (!host_recv_src_idx.empty()) {
            auto recv_src_event = stream.queue().memcpy(
                recv_src_idx, host_recv_src_idx.data(), host_recv_src_idx.size() * sizeof(int));
            recv_src_event.wait_and_throw();
        }
        if (!host_recv_x_scales.empty()) {
            auto recv_scale_event = stream.queue().memcpy(
                recv_x_scales, host_recv_x_scales.data(), host_recv_x_scales.size() * sizeof(float));
            recv_scale_event.wait_and_throw();
        }
        if (!host_recv_topk_idx.empty()) {
            auto recv_topk_event = stream.queue().memcpy(
                recv_topk_idx, host_recv_topk_idx.data(), host_recv_topk_idx.size() * sizeof(topk_idx_t));
            recv_topk_event.wait_and_throw();
        }
        if (!host_recv_topk_weights.empty()) {
            auto recv_topk_w_event = stream.queue().memcpy(
                recv_topk_weights, host_recv_topk_weights.data(), host_recv_topk_weights.size() * sizeof(float));
            recv_topk_w_event.wait_and_throw();
        }
        if (!host_recv_channel_offset.empty()) {
            auto recv_ch_event = stream.queue().memcpy(
                recv_channel_offset, host_recv_channel_offset.data(), host_recv_channel_offset.size() * sizeof(int));
            recv_ch_event.wait_and_throw();
        }
        if (!host_send_head.empty()) {
            auto send_head_event = stream.queue().memcpy(send_head, host_send_head.data(), host_send_head.size() * sizeof(int));
            send_head_event.wait_and_throw();
        }
        return;
    }

    EP_HOST_ASSERT(rank >= 0 and rank < num_ranks);
    EP_HOST_ASSERT(recv_topk_idx == nullptr || num_topk > 0);

    const uintptr_t group_key = get_xpu_ipc_group_key(buffer_ptrs, num_ranks);
    auto state = get_xpu_intranode_state(xpu_intranode_dispatch_states, xpu_intranode_dispatch_states_mu, group_key, num_ranks);
    std::unique_lock<std::mutex> lock(state->mu);
    const size_t generation = state->generation;
    state->arrivals[static_cast<size_t>(rank)] = XpuIntranodeDispatchArrival{stream,
                                                                             rank,
                                                                             num_ranks,
                                                                             num_channels,
                                                                             num_tokens,
                                                                             hidden_int4,
                                                                             num_topk,
                                                                             num_scales,
                                                                             scale_token_stride,
                                                                             scale_hidden_stride,
                                                                             recv_x,
                                                                             recv_x_scales,
                                                                             recv_src_idx,
                                                                             recv_topk_idx,
                                                                             recv_topk_weights,
                                                                             recv_channel_offset,
                                                                             send_head,
                                                                             std::move(host_x),
                                                                             std::move(host_is_token_in_rank),
                                                                             std::move(host_x_scales),
                                                                             std::move(host_topk_idx),
                                                                             std::move(host_topk_weights)};
    state->arrived++;

    if (state->arrived < num_ranks) {
        state->cv.wait(lock, [&] { return state->generation != generation; });
        return;
    }

    std::vector<XpuIntranodeDispatchArrival> arrivals;
    arrivals.reserve(static_cast<size_t>(num_ranks));
    for (auto& arrival_opt : state->arrivals) {
        EP_HOST_ASSERT(arrival_opt.has_value());
        arrivals.push_back(std::move(arrival_opt.value()));
        arrival_opt.reset();
    }
    state->arrived = 0;

    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
        auto& arrival = arrivals[static_cast<size_t>(src_rank)];
        std::vector<int> host_send_head(static_cast<size_t>(arrival.num_tokens) * static_cast<size_t>(num_ranks), 0);
        std::vector<int> sent_per_rank(static_cast<size_t>(num_ranks), 0);
        for (int token = 0; token < arrival.num_tokens; ++token) {
            for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                if (arrival.is_token_in_rank[static_cast<size_t>(token) * static_cast<size_t>(num_ranks) + static_cast<size_t>(dst_rank)] != 0)
                    sent_per_rank[static_cast<size_t>(dst_rank)]++;
                host_send_head[static_cast<size_t>(token) * static_cast<size_t>(num_ranks) + static_cast<size_t>(dst_rank)] =
                    sent_per_rank[static_cast<size_t>(dst_rank)];
            }
        }
        xpu_blocking_memcpy(arrival.stream, arrival.send_head, host_send_head.data(), host_send_head.size() * sizeof(int));
    }

    for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
        auto& dst = arrivals[static_cast<size_t>(dst_rank)];
        std::vector<std::pair<int, int>> selected_tokens;
        for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
            const auto& src = arrivals[static_cast<size_t>(src_rank)];
            for (int token = 0; token < src.num_tokens; ++token) {
                if (src.is_token_in_rank[static_cast<size_t>(token) * static_cast<size_t>(num_ranks) + static_cast<size_t>(dst_rank)] != 0)
                    selected_tokens.emplace_back(src_rank, token);
            }
        }
        if (num_worst_tokens > 0 && static_cast<int>(selected_tokens.size()) > num_worst_tokens)
            selected_tokens.resize(static_cast<size_t>(num_worst_tokens));

        std::vector<uint8_t> host_recv_x(static_cast<size_t>(selected_tokens.size()) * row_bytes, 0);
        std::vector<int> host_recv_src_idx(static_cast<size_t>(selected_tokens.size()), -1);
        std::vector<float> host_recv_x_scales;
        if (dst.recv_x_scales != nullptr && dst.num_scales > 0)
            host_recv_x_scales.assign(static_cast<size_t>(selected_tokens.size()) * static_cast<size_t>(dst.num_scales), 0.f);
        std::vector<topk_idx_t> host_recv_topk_idx;
        if (dst.recv_topk_idx != nullptr && dst.num_topk > 0)
            host_recv_topk_idx.assign(static_cast<size_t>(selected_tokens.size()) * static_cast<size_t>(dst.num_topk), static_cast<topk_idx_t>(-1));
        std::vector<float> host_recv_topk_weights;
        if (dst.recv_topk_weights != nullptr && dst.num_topk > 0)
            host_recv_topk_weights.assign(static_cast<size_t>(selected_tokens.size()) * static_cast<size_t>(dst.num_topk), 0.f);

        for (size_t out_idx = 0; out_idx < selected_tokens.size(); ++out_idx) {
            const auto [src_rank, token] = selected_tokens[out_idx];
            const auto& src = arrivals[static_cast<size_t>(src_rank)];
            host_recv_src_idx[out_idx] = token;
            if (row_bytes > 0) {
                std::memcpy(host_recv_x.data() + out_idx * row_bytes,
                            src.x.data() + static_cast<size_t>(token) * row_bytes,
                            row_bytes);
            }
            if (dst.recv_x_scales != nullptr && dst.num_scales > 0 && !src.x_scales.empty()) {
                for (int s = 0; s < dst.num_scales; ++s) {
                    host_recv_x_scales[out_idx * static_cast<size_t>(dst.num_scales) + static_cast<size_t>(s)] =
                        src.x_scales[static_cast<size_t>(token) * static_cast<size_t>(src.scale_token_stride) +
                                     static_cast<size_t>(s) * static_cast<size_t>(src.scale_hidden_stride)];
                }
            }
            if (dst.recv_topk_idx != nullptr && dst.num_topk > 0 && !src.topk_idx.empty()) {
                std::memcpy(host_recv_topk_idx.data() + out_idx * static_cast<size_t>(dst.num_topk),
                            src.topk_idx.data() + static_cast<size_t>(token) * static_cast<size_t>(src.num_topk),
                            static_cast<size_t>(dst.num_topk) * sizeof(topk_idx_t));
            }
            if (dst.recv_topk_weights != nullptr && dst.num_topk > 0 && !src.topk_weights.empty()) {
                std::memcpy(host_recv_topk_weights.data() + out_idx * static_cast<size_t>(dst.num_topk),
                            src.topk_weights.data() + static_cast<size_t>(token) * static_cast<size_t>(src.num_topk),
                            static_cast<size_t>(dst.num_topk) * sizeof(float));
            }
        }

        std::vector<int> host_recv_channel_offset(static_cast<size_t>(num_ranks) * static_cast<size_t>(dst.num_channels), 0);
        for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
            const auto& src = arrivals[static_cast<size_t>(src_rank)];
            for (int channel = 0; channel < dst.num_channels; ++channel) {
                int token_start = 0, token_end = 0;
                get_channel_task_range_host(src.num_tokens, dst.num_channels, channel, token_start, token_end);
                int count = 0;
                for (const auto& selected : selected_tokens) {
                    if (selected.first != src_rank)
                        continue;
                    const int token = selected.second;
                    count += static_cast<int>(token >= token_start and token < token_end);
                }
                host_recv_channel_offset[static_cast<size_t>(src_rank) * static_cast<size_t>(dst.num_channels) + static_cast<size_t>(channel)] = count;
            }
            for (int channel = 1; channel < dst.num_channels; ++channel) {
                host_recv_channel_offset[static_cast<size_t>(src_rank) * static_cast<size_t>(dst.num_channels) + static_cast<size_t>(channel)] +=
                    host_recv_channel_offset[static_cast<size_t>(src_rank) * static_cast<size_t>(dst.num_channels) + static_cast<size_t>(channel - 1)];
            }
        }

        if (!host_recv_x.empty())
            xpu_blocking_memcpy(dst.stream, dst.recv_x, host_recv_x.data(), host_recv_x.size());
        if (!host_recv_src_idx.empty())
            xpu_blocking_memcpy(dst.stream, dst.recv_src_idx, host_recv_src_idx.data(), host_recv_src_idx.size() * sizeof(int));
        if (!host_recv_x_scales.empty())
            xpu_blocking_memcpy(dst.stream, dst.recv_x_scales, host_recv_x_scales.data(), host_recv_x_scales.size() * sizeof(float));
        if (!host_recv_topk_idx.empty())
            xpu_blocking_memcpy(dst.stream, dst.recv_topk_idx, host_recv_topk_idx.data(), host_recv_topk_idx.size() * sizeof(topk_idx_t));
        if (!host_recv_topk_weights.empty())
            xpu_blocking_memcpy(dst.stream, dst.recv_topk_weights, host_recv_topk_weights.data(), host_recv_topk_weights.size() * sizeof(float));
        if (!host_recv_channel_offset.empty())
            xpu_blocking_memcpy(dst.stream, dst.recv_channel_offset, host_recv_channel_offset.data(), host_recv_channel_offset.size() * sizeof(int));
    }

    state->generation++;
    lock.unlock();
    state->cv.notify_all();
}

void cached_notify_combine(void**, int* send_head, int, int, int, int**, int rank, int num_ranks, runtime_stream_t stream) {
    EP_HOST_ASSERT(rank >= 0 and rank < num_ranks);
    // Single-rank XPU path does not use queue heads in buffer memory. Keep API side effects minimal.
    if (send_head != nullptr) {
        // Ensure memory dependency on communication stream.
        auto noop = stream.queue().memcpy(send_head, send_head, sizeof(int));
        noop.wait_and_throw();
    }
}

void combine(runtime_data_type_t type,
             void* recv_x,
             float* recv_topk_weights,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* src_idx,
             const int* rank_prefix_matrix,
             const int*,
             int*,
             int num_tokens,
             int num_recv_tokens,
             int hidden,
             int num_topk,
             void** buffer_ptrs,
             int rank,
             int num_ranks,
             runtime_stream_t stream,
             int,
             int,
             int) {
    const bool is_bf16 = (type == RUNTIME_R_16BF);
    const bool is_f32 = (type == static_cast<runtime_data_type_t>(c10::ScalarType::Float));
    const bool is_f16 = (type == static_cast<runtime_data_type_t>(c10::ScalarType::Half));
    if (!is_bf16 and !is_f32 and !is_f16)
        EP_UNSUPPORTED_XPU("intranode combine unsupported dtype on XPU");

    EP_HOST_ASSERT(recv_x != nullptr and x != nullptr and src_idx != nullptr);
    EP_HOST_ASSERT(num_tokens >= 0 and num_recv_tokens >= 0 and hidden >= 0 and num_topk >= 0);

    EP_HOST_ASSERT(rank >= 0 and rank < num_ranks);

    const size_t elem_size = is_bf16 ? sizeof(nv_bfloat16) : (is_f16 ? sizeof(c10::Half) : sizeof(float));
    const size_t row_bytes = static_cast<size_t>(hidden) * elem_size;
    const size_t src_bytes = static_cast<size_t>(num_tokens) * row_bytes;

    if (num_ranks == 1 and rank == 0) {
        std::vector<uint8_t> host_x(src_bytes);
        if (src_bytes > 0) {
            auto x_event = stream.queue().memcpy(host_x.data(), x, src_bytes);
            x_event.wait_and_throw();
        }

        std::vector<int> host_src_idx(static_cast<size_t>(num_tokens), -1);
        if (num_tokens > 0) {
            auto src_idx_event = stream.queue().memcpy(host_src_idx.data(), src_idx, static_cast<size_t>(num_tokens) * sizeof(int));
            src_idx_event.wait_and_throw();
        }

        std::vector<float> host_recv_x_accum(static_cast<size_t>(num_recv_tokens) * static_cast<size_t>(hidden), 0.f);
        const auto* host_x_bf16 = is_bf16 ? reinterpret_cast<const nv_bfloat16*>(host_x.data()) : nullptr;
        const auto* host_x_f32 = is_f32 ? reinterpret_cast<const float*>(host_x.data()) : nullptr;
        const auto* host_x_f16 = is_f16 ? reinterpret_cast<const c10::Half*>(host_x.data()) : nullptr;
        for (int i = 0; i < num_tokens; ++i) {
            const int dst_idx = host_src_idx[static_cast<size_t>(i)];
            if (dst_idx < 0 or dst_idx >= num_recv_tokens)
                continue;
            for (int h = 0; h < hidden; ++h) {
                const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(hidden) + static_cast<size_t>(h);
                const float x_value = is_bf16 ? static_cast<float>(host_x_bf16[idx]) :
                                     (is_f16 ? static_cast<float>(host_x_f16[idx]) : host_x_f32[idx]);
                host_recv_x_accum[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)] += x_value;
            }
        }

        const nv_bfloat16* host_bias_0_bf16 = nullptr;
        const nv_bfloat16* host_bias_1_bf16 = nullptr;
        const c10::Half* host_bias_0_f16 = nullptr;
        const c10::Half* host_bias_1_f16 = nullptr;
        const float* host_bias_0_f32 = nullptr;
        const float* host_bias_1_f32 = nullptr;
        std::vector<uint8_t> host_bias_0;
        std::vector<uint8_t> host_bias_1;
        if (bias_0 != nullptr) {
            host_bias_0.resize(static_cast<size_t>(num_recv_tokens) * row_bytes);
            auto bias0_event = stream.queue().memcpy(host_bias_0.data(), bias_0, host_bias_0.size());
            bias0_event.wait_and_throw();
            if (is_bf16)
                host_bias_0_bf16 = reinterpret_cast<const nv_bfloat16*>(host_bias_0.data());
            else if (is_f16)
                host_bias_0_f16 = reinterpret_cast<const c10::Half*>(host_bias_0.data());
            else
                host_bias_0_f32 = reinterpret_cast<const float*>(host_bias_0.data());
        }
        if (bias_1 != nullptr) {
            host_bias_1.resize(static_cast<size_t>(num_recv_tokens) * row_bytes);
            auto bias1_event = stream.queue().memcpy(host_bias_1.data(), bias_1, host_bias_1.size());
            bias1_event.wait_and_throw();
            if (is_bf16)
                host_bias_1_bf16 = reinterpret_cast<const nv_bfloat16*>(host_bias_1.data());
            else if (is_f16)
                host_bias_1_f16 = reinterpret_cast<const c10::Half*>(host_bias_1.data());
            else
                host_bias_1_f32 = reinterpret_cast<const float*>(host_bias_1.data());
        }

        if (is_bf16) {
            std::vector<nv_bfloat16> host_recv_x(static_cast<size_t>(num_recv_tokens) * static_cast<size_t>(hidden));
            for (int dst_idx = 0; dst_idx < num_recv_tokens; ++dst_idx) {
                for (int h = 0; h < hidden; ++h) {
                    float value = host_recv_x_accum[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)];
                    if (host_bias_0_bf16 != nullptr)
                        value += static_cast<float>(host_bias_0_bf16[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)]);
                    if (host_bias_1_bf16 != nullptr)
                        value += static_cast<float>(host_bias_1_bf16[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)]);
                    host_recv_x[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)] = static_cast<nv_bfloat16>(value);
                }
            }
            if (!host_recv_x.empty()) {
                auto recv_x_event = stream.queue().memcpy(recv_x, host_recv_x.data(), host_recv_x.size() * sizeof(nv_bfloat16));
                recv_x_event.wait_and_throw();
            }
        } else if (is_f16) {
            std::vector<c10::Half> host_recv_x(static_cast<size_t>(num_recv_tokens) * static_cast<size_t>(hidden));
            for (int dst_idx = 0; dst_idx < num_recv_tokens; ++dst_idx) {
                for (int h = 0; h < hidden; ++h) {
                    float value = host_recv_x_accum[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)];
                    if (host_bias_0_f16 != nullptr)
                        value += static_cast<float>(host_bias_0_f16[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)]);
                    if (host_bias_1_f16 != nullptr)
                        value += static_cast<float>(host_bias_1_f16[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)]);
                    host_recv_x[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)] = static_cast<c10::Half>(value);
                }
            }
            if (!host_recv_x.empty()) {
                auto recv_x_event = stream.queue().memcpy(recv_x, host_recv_x.data(), host_recv_x.size() * sizeof(c10::Half));
                recv_x_event.wait_and_throw();
            }
        } else {
            std::vector<float> host_recv_x(static_cast<size_t>(num_recv_tokens) * static_cast<size_t>(hidden), 0.f);
            for (int dst_idx = 0; dst_idx < num_recv_tokens; ++dst_idx) {
                for (int h = 0; h < hidden; ++h) {
                    float value = host_recv_x_accum[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)];
                    if (host_bias_0_f32 != nullptr)
                        value += host_bias_0_f32[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)];
                    if (host_bias_1_f32 != nullptr)
                        value += host_bias_1_f32[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)];
                    host_recv_x[static_cast<size_t>(dst_idx) * static_cast<size_t>(hidden) + static_cast<size_t>(h)] = value;
                }
            }
            if (!host_recv_x.empty()) {
                auto recv_x_event = stream.queue().memcpy(recv_x, host_recv_x.data(), host_recv_x.size() * sizeof(float));
                recv_x_event.wait_and_throw();
            }
        }

        if (recv_topk_weights != nullptr and topk_weights != nullptr and num_topk > 0) {
            std::vector<float> host_topk_weights(static_cast<size_t>(num_tokens) * static_cast<size_t>(num_topk), 0.f);
            auto topk_event = stream.queue().memcpy(
                host_topk_weights.data(), topk_weights, host_topk_weights.size() * sizeof(float));
            topk_event.wait_and_throw();

            std::vector<float> host_recv_topk_weights(static_cast<size_t>(num_recv_tokens) * static_cast<size_t>(num_topk), 0.f);
            for (int i = 0; i < num_tokens; ++i) {
                const int dst_idx = host_src_idx[static_cast<size_t>(i)];
                if (dst_idx < 0 or dst_idx >= num_recv_tokens)
                    continue;
                for (int k = 0; k < num_topk; ++k) {
                    host_recv_topk_weights[static_cast<size_t>(dst_idx) * static_cast<size_t>(num_topk) + static_cast<size_t>(k)] +=
                        host_topk_weights[static_cast<size_t>(i) * static_cast<size_t>(num_topk) + static_cast<size_t>(k)];
                }
            }

            auto recv_topk_event = stream.queue().memcpy(
                recv_topk_weights, host_recv_topk_weights.data(), host_recv_topk_weights.size() * sizeof(float));
            recv_topk_event.wait_and_throw();
        }
        return;
    }

    std::vector<uint8_t> host_x(src_bytes);
    if (src_bytes > 0)
        xpu_blocking_memcpy(stream, host_x.data(), x, src_bytes);

    std::vector<int> host_src_idx(static_cast<size_t>(num_tokens), -1);
    if (num_tokens > 0)
        xpu_blocking_memcpy(stream, host_src_idx.data(), src_idx, static_cast<size_t>(num_tokens) * sizeof(int));

    std::vector<int> host_rank_prefix_matrix(static_cast<size_t>(num_ranks) * static_cast<size_t>(num_ranks), 0);
    if (num_ranks > 0)
        xpu_blocking_memcpy(stream,
                           host_rank_prefix_matrix.data(),
                           rank_prefix_matrix,
                           host_rank_prefix_matrix.size() * sizeof(int));

    std::vector<float> host_topk_weights;
    if (recv_topk_weights != nullptr and topk_weights != nullptr and num_topk > 0) {
        host_topk_weights.resize(static_cast<size_t>(num_tokens) * static_cast<size_t>(num_topk), 0.f);
        xpu_blocking_memcpy(stream,
                           host_topk_weights.data(),
                           topk_weights,
                           host_topk_weights.size() * sizeof(float));
    }

    std::vector<uint8_t> host_bias_0;
    std::vector<uint8_t> host_bias_1;
    if (bias_0 != nullptr) {
        host_bias_0.resize(static_cast<size_t>(num_recv_tokens) * row_bytes);
        xpu_blocking_memcpy(stream, host_bias_0.data(), bias_0, host_bias_0.size());
    }
    if (bias_1 != nullptr) {
        host_bias_1.resize(static_cast<size_t>(num_recv_tokens) * row_bytes);
        xpu_blocking_memcpy(stream, host_bias_1.data(), bias_1, host_bias_1.size());
    }

    const uintptr_t group_key = get_xpu_ipc_group_key(buffer_ptrs, num_ranks);
    auto state = get_xpu_intranode_state(xpu_intranode_combine_states, xpu_intranode_combine_states_mu, group_key, num_ranks);
    std::unique_lock<std::mutex> lock(state->mu);
    const size_t generation = state->generation;
    state->arrivals[static_cast<size_t>(rank)] = XpuIntranodeCombineArrival{stream,
                                                                            rank,
                                                                            num_ranks,
                                                                            num_tokens,
                                                                            num_recv_tokens,
                                                                            hidden,
                                                                            num_topk,
                                                                            recv_x,
                                                                            recv_topk_weights,
                                                                            std::move(host_x),
                                                                            std::move(host_src_idx),
                                                                            std::move(host_rank_prefix_matrix),
                                                                            std::move(host_topk_weights),
                                                                            std::move(host_bias_0),
                                                                            std::move(host_bias_1)};
    state->arrived++;

    if (state->arrived < num_ranks) {
        state->cv.wait(lock, [&] { return state->generation != generation; });
        return;
    }

    std::vector<XpuIntranodeCombineArrival> arrivals;
    arrivals.reserve(static_cast<size_t>(num_ranks));
    for (auto& arrival_opt : state->arrivals) {
        EP_HOST_ASSERT(arrival_opt.has_value());
        arrivals.push_back(std::move(arrival_opt.value()));
        arrival_opt.reset();
    }
    state->arrived = 0;

    std::vector<std::vector<float>> recv_x_accum_per_rank(static_cast<size_t>(num_ranks));
    std::vector<std::vector<float>> recv_topk_accum_per_rank(static_cast<size_t>(num_ranks));
    for (int target_rank = 0; target_rank < num_ranks; ++target_rank) {
        const auto& target = arrivals[static_cast<size_t>(target_rank)];
        recv_x_accum_per_rank[static_cast<size_t>(target_rank)].assign(
            static_cast<size_t>(target.num_recv_tokens) * static_cast<size_t>(target.hidden), 0.f);
        if (target.recv_topk_weights != nullptr && target.num_topk > 0) {
            recv_topk_accum_per_rank[static_cast<size_t>(target_rank)].assign(
                static_cast<size_t>(target.num_recv_tokens) * static_cast<size_t>(target.num_topk), 0.f);
        }
    }

    for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
        const auto& arrival = arrivals[static_cast<size_t>(dst_rank)];
        const auto* host_x_bf16 = is_bf16 ? reinterpret_cast<const nv_bfloat16*>(arrival.x.data()) : nullptr;
        const auto* host_x_f32 = is_f32 ? reinterpret_cast<const float*>(arrival.x.data()) : nullptr;
        const auto* host_x_f16 = is_f16 ? reinterpret_cast<const c10::Half*>(arrival.x.data()) : nullptr;
        for (int row = 0; row < arrival.num_tokens; ++row) {
            const int src_token_idx = arrival.src_idx[static_cast<size_t>(row)];
            const int src_rank = get_source_rank_from_rank_prefix(arrival.rank_prefix_matrix, num_ranks, dst_rank, row);
            if (src_rank < 0 || src_rank >= num_ranks)
                continue;
            auto& target_accum = recv_x_accum_per_rank[static_cast<size_t>(src_rank)];
            const auto& target = arrivals[static_cast<size_t>(src_rank)];
            if (src_token_idx < 0 || src_token_idx >= target.num_recv_tokens)
                continue;

            for (int h = 0; h < arrival.hidden; ++h) {
                const size_t idx = static_cast<size_t>(row) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h);
                const float x_value = is_bf16 ? static_cast<float>(host_x_bf16[idx]) :
                                     (is_f16 ? static_cast<float>(host_x_f16[idx]) : host_x_f32[idx]);
                target_accum[static_cast<size_t>(src_token_idx) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)] += x_value;
            }

            if (!arrival.topk_weights.empty() &&
                arrivals[static_cast<size_t>(src_rank)].recv_topk_weights != nullptr &&
                arrivals[static_cast<size_t>(src_rank)].num_topk > 0) {
                auto& target_topk_accum = recv_topk_accum_per_rank[static_cast<size_t>(src_rank)];
                for (int k = 0; k < arrival.num_topk; ++k) {
                    target_topk_accum[static_cast<size_t>(src_token_idx) * static_cast<size_t>(arrival.num_topk) + static_cast<size_t>(k)] +=
                        arrival.topk_weights[static_cast<size_t>(row) * static_cast<size_t>(arrival.num_topk) + static_cast<size_t>(k)];
                }
            }
        }
    }

    for (int target_rank = 0; target_rank < num_ranks; ++target_rank) {
        const auto& arrival = arrivals[static_cast<size_t>(target_rank)];
        const auto* bias0_bf16 = (is_bf16 && !arrival.bias_0.empty()) ? reinterpret_cast<const nv_bfloat16*>(arrival.bias_0.data()) : nullptr;
        const auto* bias1_bf16 = (is_bf16 && !arrival.bias_1.empty()) ? reinterpret_cast<const nv_bfloat16*>(arrival.bias_1.data()) : nullptr;
        const auto* bias0_f16 = (is_f16 && !arrival.bias_0.empty()) ? reinterpret_cast<const c10::Half*>(arrival.bias_0.data()) : nullptr;
        const auto* bias1_f16 = (is_f16 && !arrival.bias_1.empty()) ? reinterpret_cast<const c10::Half*>(arrival.bias_1.data()) : nullptr;
        const auto* bias0_f32 = (is_f32 && !arrival.bias_0.empty()) ? reinterpret_cast<const float*>(arrival.bias_0.data()) : nullptr;
        const auto* bias1_f32 = (is_f32 && !arrival.bias_1.empty()) ? reinterpret_cast<const float*>(arrival.bias_1.data()) : nullptr;

        const auto& accum = recv_x_accum_per_rank[static_cast<size_t>(target_rank)];
        if (is_bf16) {
            std::vector<nv_bfloat16> host_recv_x(static_cast<size_t>(arrival.num_recv_tokens) * static_cast<size_t>(arrival.hidden));
            for (int i = 0; i < arrival.num_recv_tokens; ++i) {
                for (int h = 0; h < arrival.hidden; ++h) {
                    float value = accum[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)];
                    if (bias0_bf16 != nullptr)
                        value += static_cast<float>(bias0_bf16[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)]);
                    if (bias1_bf16 != nullptr)
                        value += static_cast<float>(bias1_bf16[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)]);
                    host_recv_x[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)] = static_cast<nv_bfloat16>(value);
                }
            }
            if (!host_recv_x.empty())
                xpu_blocking_memcpy(arrival.stream, arrival.recv_x, host_recv_x.data(), host_recv_x.size() * sizeof(nv_bfloat16));
        } else if (is_f16) {
            std::vector<c10::Half> host_recv_x(static_cast<size_t>(arrival.num_recv_tokens) * static_cast<size_t>(arrival.hidden));
            for (int i = 0; i < arrival.num_recv_tokens; ++i) {
                for (int h = 0; h < arrival.hidden; ++h) {
                    float value = accum[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)];
                    if (bias0_f16 != nullptr)
                        value += static_cast<float>(bias0_f16[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)]);
                    if (bias1_f16 != nullptr)
                        value += static_cast<float>(bias1_f16[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)]);
                    host_recv_x[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)] = static_cast<c10::Half>(value);
                }
            }
            if (!host_recv_x.empty())
                xpu_blocking_memcpy(arrival.stream, arrival.recv_x, host_recv_x.data(), host_recv_x.size() * sizeof(c10::Half));
        } else {
            std::vector<float> host_recv_x(static_cast<size_t>(arrival.num_recv_tokens) * static_cast<size_t>(arrival.hidden), 0.f);
            for (int i = 0; i < arrival.num_recv_tokens; ++i) {
                for (int h = 0; h < arrival.hidden; ++h) {
                    float value = accum[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)];
                    if (bias0_f32 != nullptr)
                        value += bias0_f32[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)];
                    if (bias1_f32 != nullptr)
                        value += bias1_f32[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)];
                    host_recv_x[static_cast<size_t>(i) * static_cast<size_t>(arrival.hidden) + static_cast<size_t>(h)] = value;
                }
            }
            if (!host_recv_x.empty())
                xpu_blocking_memcpy(arrival.stream, arrival.recv_x, host_recv_x.data(), host_recv_x.size() * sizeof(float));
        }

        if (arrival.recv_topk_weights != nullptr && arrival.num_topk > 0) {
            const auto& topk_accum = recv_topk_accum_per_rank[static_cast<size_t>(target_rank)];
            if (!topk_accum.empty()) {
                xpu_blocking_memcpy(arrival.stream,
                                   arrival.recv_topk_weights,
                                   topk_accum.data(),
                                   topk_accum.size() * sizeof(float));
            }
        }
    }

    state->generation++;
    lock.unlock();
    state->cv.notify_all();
}

#else

template <int kNumRanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,
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
                                int rank) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32;

    if (sm_id == 0) {
        // Barrier first
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        int *per_rank_buffer, *per_expert_buffer;
        if (thread_id < kNumRanks) {
            per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]);
            per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
        }

        // After this loop:
        //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
        //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
        int num_experts_per_rank = num_experts / kNumRanks;
        if (thread_id < kNumRanks) {
            per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];
            #pragma unroll
            for (int i = 0; i < num_experts_per_rank; ++i)
                per_expert_buffer[rank * num_experts_per_rank + i] = num_tokens_per_expert[thread_id * num_experts_per_rank + i];
        }

        // Wait for all ranks to be finished
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

        // Sum per-rank counts and return to CPU
        // Also pre-compute the prefix sum for data sending
        auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);
        if (thread_id < kNumRanks) {
            #pragma unroll
            for (int i = 1; i < kNumRanks; ++i)
                local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
            if (thread_id == rank)
                *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
        }

        // Sum per-experts counts and return to CPU
        auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
        if (thread_id < num_experts_per_rank) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i)
                sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            moe_recv_expert_counter_mapped[thread_id] = sum;
        }
        __syncthreads();

        // Copy rank size prefix matrix to another tensor
        #pragma unroll
        for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
            rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];

        // Extra memset for later communication queue
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            local_per_expert_buffer[i] = 0;

        // Barrier
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {
        int dst_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            int count = 0;
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
                count += is_token_in_rank[i * kNumRanks + dst_rank];
            count = warp_reduce_sum(count);
            if (elect_one_sync())
                channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
        }
        __syncthreads();

        // Pre-compute prefix sum for all channels
        if (thread_id == 0) {
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];
        }
    }
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
                     runtime_stream_t stream,
                     int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)        \
    LAUNCH_KERNEL(&cfg,                           \
                  notify_dispatch<ranks>,         \
                  num_tokens_per_rank,            \
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
                  rank);                          \
    break

    constexpr int kNumThreads = 128;
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);

    SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
    SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_dispatch(
    const int* rank_prefix_matrix, int num_memset_int, void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
    // A simplified version for cached handles
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    // Copy and clean
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto ptr = static_cast<int*>(buffer_ptrs[rank]);
    #pragma unroll
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
        ptr[i] = rank_prefix_matrix[i];
    #pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads)
        ptr[kNumRanks * kNumRanks + i] = 0;

    // Barrier after cleaning
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void cached_notify_dispatch(const int* rank_prefix_matrix,
                            int num_memset_int,
                            void** buffer_ptrs,
                            int** barrier_signal_ptrs,
                            int rank,
                            int num_ranks,
                            runtime_stream_t stream) {
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                                                   \
    LAUNCH_KERNEL(&cfg, cached_notify_dispatch<ranks>, rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 128, stream);
    SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch(int4* recv_x,
                                                           float* recv_x_scales,
                                                           int* recv_src_idx,
                                                           topk_idx_t* recv_topk_idx,
                                                           float* recv_topk_weights,
                                                           int* recv_channel_offset,
                                                           int* send_head,
                                                           const int4* x,
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
                                                           int rank,
                                                           int num_max_send_tokens,
                                                           int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    // Several warps are response for a single rank
    const auto num_threads_per_rank = kNumThreads / kNumRanks;
    const auto num_channels = num_sms / 2;
    const auto responsible_rank = (static_cast<int>(thread_id)) / num_threads_per_rank;
    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    const auto responsible_channel = sm_id / 2;

    int num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    // Calculate pointers by the specific layout
    // `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[is_sender ? responsible_rank : rank]) +
                                       kNumRanks * kNumRanks * sizeof(int));
    int target_rank = is_sender ? rank : responsible_rank;
    auto num_channels_total = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    // Channel buffer metadata
    // Senders are responsible for tails, and receivers are responsible for heads
    // Stored on the receiver side
    // The retired signals are actually boolean flags, but to align with 16 bytes, we make it `int64_t`
    // `start_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `end_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);

    // Channel data buffers, stored on the receiver side
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
    // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
    // `topk_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(topk_idx_t)
    // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
    // `x_scales_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_scales * sizeof(float)
    auto channel_x_buffers = Buffer<int4>(
        ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers =
        Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_idx_buffers = Buffer<topk_idx_t>(
        ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_topk_weights_buffers =
        Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_x_scales_buffers = Buffer<float>(
        ptr, num_channels_total * num_recv_buffer_tokens * num_scales, channel_rank_offset * num_recv_buffer_tokens * num_scales);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto half_hidden_int4 = hidden_int4 / 2;
    auto half_hidden_bytes = half_hidden_int4 * static_cast<int>(sizeof(int4));
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + half_hidden_bytes);
    uint32_t tma_phase = 0;
    if (elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(hidden_int4 % 2 == 0 and half_hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();
#endif

    if (is_sender) {
        // Workers for sending
        constexpr int num_send_warps = kNumThreads / 32;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        const auto send_thread_id = thread_id;
        const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
        // NOTES: this is for distinguishing zero tokens
        if (send_warp_id_in_rank == 0 and elect_one_sync()) {
            int value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;
            st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
            value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
            st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);
        }
        __syncwarp();

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

        // Iterate over all tokens and send by chunks
        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            // NOTES: the head index received by different warps may not be the same
            auto start_time = clock64();
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for dispatch senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();

            int chunk_token_idx = 0;
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                // NOTES: for the same token, the warp assigned to save `send_head` may be different from the warp assigned to send the
                // following data
                if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and elect_one_sync())
                    send_head[token_idx * kNumRanks + responsible_rank] =
                        is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;

                // Skip if not selected
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                // Get an empty slot
                int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    // Copy data
                    auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                    auto shifted_x = x + token_idx * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);

                    // Copy source index
                    if (elect_one_sync())
                        channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

                    // Copy `topk_idx` and `topk_weights` with transformed index
                    if (lane_id < num_topk) {
                        // Top-k index
                        int recv_expert_begin = responsible_rank * num_experts_per_rank,
                            recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;
                        auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id);
                        idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;
                        channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

                        // Top-k weights
                        auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                        weight_value = (idx_value >= 0) ? weight_value : 0.0f;
                        channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = weight_value;
                    }

                    // Copy `x_scales`
                    #pragma unroll
                    for (int i = lane_id; i < num_scales; i += 32) {
                        auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                        channel_x_scales_buffers[dst_slot_idx * num_scales + i] = __ldg(x_scales + offset);
                    }
                }

                // Move token index
                chunk_token_idx++, token_idx++;
            }

            // Move tail index
            // NOTES: here all warps should share the same new tail
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
        }
    } else {
        // Workers for receiving and copying into buffer
        constexpr int num_recv_warps = kNumThreads / 32;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        const auto recv_thread_id = thread_id;
        const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
        const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // Calculate offset first
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;

        // Receive channel offset
        int total_offset, num_tokens_to_recv;
        if (elect_one_sync()) {
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
                ;
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
                ;
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            if (recv_warp_id_in_rank == 0)
                recv_channel_offset[responsible_rank * num_channels + responsible_channel] = total_offset;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(0xffffffff, total_offset, 0);
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

        // Shared tail indices for different warps
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];

        auto start_time = clock64();
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // NOTES: unlike the sender, the receiver must ensure that the tail indices hold by different warps are the same
            while (recv_thread_id_in_rank == 0) {
                cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());

                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx) {
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = %d, tokens remained: %d\n",
                           rank,
                           responsible_channel,
                           num_tokens_to_recv);
                    trap();
                }
            }

            // Synchronize queue tail
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) {
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
                auto shifted_recv_x_int4 = recv_x + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
#ifndef DISABLE_SM90_FEATURES
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    tma_store_wait<0>();
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer, shifted_buffer_x_int4 + i * half_hidden_int4, tma_mbarrier, half_hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
                        mbarrier_wait(tma_mbarrier, tma_phase);
                        tma_store_1d(tma_buffer, shifted_recv_x_int4 + i * half_hidden_int4, half_hidden_bytes, false);
                    }
                }
                __syncwarp();
#else
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4, shifted_buffer_x_int4, ld_nc_global, st_na_global);
#endif
            }

            // Copy `src_idx`
            #pragma unroll 4
            for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx;
                 chunk_idx += 32 * num_recv_warps_per_rank)
                recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] =
                    ld_nc_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);

            // Copy `topk_idx` and `topk_weights`
            #pragma unroll 4
            for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk; idx += 32 * num_recv_warps_per_rank) {
                int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                auto recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx;
                auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
                recv_topk_idx[recv_idx] = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
                recv_topk_weights[recv_idx] = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
            }

            // Copy `x_scales`
            #pragma unroll 4
            for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales; i += 32 * num_recv_warps_per_rank) {
                int chunk_idx = i / num_scales, scales_idx = i % num_scales;
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scales_idx] =
                    ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_buffer * num_scales + scales_idx);
            }

            // Move queue
            cached_channel_head_idx += num_recv_tokens;
            total_offset += num_recv_tokens;
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

            // Exit
            num_tokens_to_recv -= num_recv_tokens;
        }
    }

    // Clean unused `recv_topk_idx` as -1
    if (num_worst_tokens > 0) {
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        const auto num_recv_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
        const auto clean_start = num_recv_tokens * num_topk + sm_id * kNumThreads;
        const auto clean_end = num_worst_tokens * num_topk;
        const auto clean_stride = num_sms * kNumThreads;
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
            recv_topk_idx[i] = -1;
    }
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
              int rank,
              int num_ranks,
              runtime_stream_t stream,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 8192;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks)                                      \
    {                                                                    \
        auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                               \
        LAUNCH_KERNEL(&cfg,                                              \
                      kernel,                                            \
                      reinterpret_cast<int4*>(recv_x),                   \
                      recv_x_scales,                                     \
                      recv_src_idx,                                      \
                      recv_topk_idx,                                     \
                      recv_topk_weights,                                 \
                      recv_channel_offset,                               \
                      send_head,                                         \
                      reinterpret_cast<const int4*>(x),                  \
                      x_scales,                                          \
                      topk_idx,                                          \
                      topk_weights,                                      \
                      is_token_in_rank,                                  \
                      channel_prefix_matrix,                             \
                      num_tokens,                                        \
                      num_worst_tokens,                                  \
                      hidden_int4,                                       \
                      num_topk,                                          \
                      num_experts,                                       \
                      num_scales,                                        \
                      scale_token_stride,                                \
                      scale_hidden_stride,                               \
                      buffer_ptrs,                                       \
                      rank,                                              \
                      num_max_send_tokens,                               \
                      num_recv_buffer_tokens);                           \
    }                                                                    \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_combine(
    void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens, int num_memset_int, int** barrier_signal_ptrs, int rank) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    if (sm_id == 0) {
        // Barrier before cleaning
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        // Clean
        auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
        auto ptr = static_cast<int*>(buffer_ptrs[rank]);
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            ptr[i] = 0;

        // Barrier after cleaning
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {
        const auto channel_id = sm_id - 1;
        const auto thread_id = static_cast<int>(threadIdx.x);
        const auto rank_id = thread_id / 32;
        const auto lane_id = thread_id % 32;
        if (rank_id >= kNumRanks)
            return;

        int token_start_idx, token_end_idx;
        get_channel_task_range(num_recv_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // NOTES: `1 << 25` is a heuristic large number
        int last_head = 1 << 25;
        #pragma unroll
        for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= 32) {
            int token_idx = token_idx_tail - lane_id, expected_head = 0;
            auto current_head = (token_idx >= token_start_idx) ? __ldg(send_head + token_idx * kNumRanks + rank_id) : -1;
            for (int i = 0; i < min(32, token_idx_tail - token_start_idx + 1); ++i) {
                const int head = __shfl_sync(0xffffffff, current_head, i);
                if (head < 0) {
                    if (lane_id == i)
                        expected_head = -last_head - 1;
                } else {
                    last_head = head;
                }
            }
            if (current_head < 0 and token_idx >= token_start_idx)
                send_head[token_idx * kNumRanks + rank_id] = expected_head;
        }
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
                           runtime_stream_t stream) {
#define CACHED_NOTIFY_COMBINE(ranks)            \
    LAUNCH_KERNEL(&cfg,                         \
                  cached_notify_combine<ranks>, \
                  buffer_ptrs,                  \
                  send_head,                    \
                  num_channels,                 \
                  num_recv_tokens,              \
                  num_memset_int,               \
                  barrier_signal_ptrs,          \
                  rank);                        \
    break

    const int num_threads = std::max(128, 32 * num_ranks);
    EP_HOST_ASSERT(num_ranks <= num_threads);
    EP_HOST_ASSERT(num_threads <= 1024);
    EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);
    SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
    SWITCH_RANKS(CACHED_NOTIFY_COMBINE);
#undef CACHED_NOTIFY_COMBINE
}

template <typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) combine(dtype_t* recv_x,
                                                          float* recv_topk_weights,
                                                          const dtype_t* x,
                                                          const float* topk_weights,
                                                          const dtype_t* bias_0,
                                                          const dtype_t* bias_1,
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
                                                          int num_max_send_tokens,
                                                          int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto sm_id = static_cast<int>(blockIdx.x), lane_id = get_lane_id();
    const auto num_channels = num_sms / 2;
    const bool is_sender = sm_id % 2 == 0;
    const int responsible_channel = sm_id / 2;
    EP_DEVICE_ASSERT(num_topk <= 32);

    constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
    int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
    int hidden_int4_aligned = align_down(hidden_int4, 32);
    auto x_int4 = reinterpret_cast<const int4*>(x);
    auto bias_0_int4 = reinterpret_cast<const int4*>(bias_0);
    auto bias_1_int4 = reinterpret_cast<const int4*>(bias_1);
    auto recv_int4 = reinterpret_cast<int4*>(recv_x);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
#endif

    if (is_sender) {
        // Workers for sending
        // Several warps are responsible for a single rank
        constexpr int num_send_warps_per_rank = (kNumThreads / 32) / kNumRanks;
        constexpr int num_send_warps = num_send_warps_per_rank * kNumRanks;
        const auto num_threads_per_rank = num_send_warps_per_rank * 32;
        const auto send_thread_id = thread_id;
        const auto send_warp_id = send_thread_id / 32;
        const auto send_rank_id = (responsible_channel + send_warp_id) % kNumRanks;
        const auto send_warp_id_in_rank = send_warp_id / kNumRanks;
        EP_STATIC_ASSERT(num_send_warps * 32 == kNumThreads, "Invalid warp count");

        // Calculate pointers by the specific layout
        auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[send_rank_id]));
        auto num_channels_total = num_channels * kNumRanks;
        auto channel_rank_offset = responsible_channel * kNumRanks + rank;

        // Channel meta data
        // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_x_buffers = Buffer<int4>(
            ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
        auto channel_src_idx_buffers =
            Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
        auto channel_topk_weights_buffers = Buffer<float>(
            ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);

        // Get tasks
        // NOTES: `channel_offset` is already shifted
        int rank_offset = send_rank_id > 0 ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank] : 0;
        int num_rank_tokens = rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
        int channel_offset = channel_prefix_matrix[send_rank_id * num_channels + responsible_channel];
        int num_channel_tokens =
            (responsible_channel == num_channels - 1 ? num_rank_tokens
                                                     : channel_prefix_matrix[send_rank_id * num_channels + responsible_channel + 1]) -
            channel_offset;
        int token_start_idx = rank_offset + channel_offset, token_end_idx = rank_offset + channel_offset + num_channel_tokens;

        // Iterate over all tokens and send by chunks
        int current_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            auto start_time = clock64();
            int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = current_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();

            // Send by chunk
            #pragma unroll
            for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) {
                // Get an empty slot
                int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;

                // Copy data
                auto shifted_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                auto shifted_x = x_int4 + (token_idx + i) * hidden_int4;
                UNROLLED_WARP_COPY(4, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);

                // Send source index
                if (elect_one_sync())
                    channel_src_idx_buffers[dst_slot_idx] = __ldg(src_idx + token_idx + i);

                // Send `topk_weights`
                if (num_topk > 0 and lane_id < num_topk)
                    channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
                        __ldg(topk_weights + (token_idx + i) * num_topk + lane_id);
            }
            token_idx += num_round_tokens;
            current_channel_tail_idx += num_round_tokens;

            // Move tail index
            asm volatile("bar.sync %0, %1;" ::"r"(send_rank_id), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), current_channel_tail_idx);
        }
    } else {
        // Workers for receiving
        // One warp for moving the queue head, others for reduction
        constexpr int num_recv_warps = kNumThreads / 32;
        const auto recv_warp_id = thread_id / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32 and kNumThreads > 32);
        EP_DEVICE_ASSERT(thread_id >= 0 and kNumThreads % 32 == 0);

        // Shared head, tail and retired flags for receiver warps
        __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
        __shared__ volatile int channel_tail_idx[kNumRanks];
        __shared__ volatile bool warp_retired[num_recv_warps];
        if (thread_id < num_recv_warps)
            warp_retired[thread_id] = false;
        if (lane_id < kNumRanks)
            warp_channel_head_idx[recv_warp_id][lane_id] = 0;
        if (thread_id < kNumRanks)
            channel_tail_idx[thread_id] = 0;
        asm volatile("bar.sync 0, %0;" ::"r"(kNumThreads));

        if (thread_id < 32) {
            int* channel_head_idx_ptr = static_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks + lane_id;
            int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;

            // Queue head updater
            int last_head = 0;
            while (lane_id < kNumRanks) {
                // Check retired
                bool retired = true;
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    retired = retired and warp_retired[i];
                if (retired)
                    break;

                // Update queue tail
                channel_tail_idx[lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);

                // Update minimum head
                int min_head = std::numeric_limits<int>::max();
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    if (not warp_retired[i])
                        min_head = min(min_head, warp_channel_head_idx[i][lane_id]);
                if (min_head != std::numeric_limits<int>::max() and min_head > last_head)
                    st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head);
            }
        } else {
            // Receivers
            // Channel metadata
            // All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`
            Buffer<int4> channel_x_buffers[kNumRanks];
            Buffer<float> channel_topk_weights_buffers[kNumRanks];

            // Calculate pointers by the specific layout
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i) {
                auto channel_rank_offset = responsible_channel * kNumRanks + i;
                auto num_channels_total = num_channels * kNumRanks;
                // `head_idx` & `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
                auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[rank]) + 2 * num_channels * kNumRanks * sizeof(int));

                // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
                channel_x_buffers[i] = Buffer<int4>(ptr,
                                                    num_channels_total * num_recv_buffer_tokens * hidden_int4,
                                                    channel_rank_offset * num_recv_buffer_tokens * hidden_int4);

                // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
                ptr = reinterpret_cast<void*>(static_cast<int8_t*>(ptr) + num_channels_total * num_recv_buffer_tokens * sizeof(int));

                // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
                channel_topk_weights_buffers[i] = Buffer<float>(
                    ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
            }

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            for (int64_t token_idx = token_start_idx + recv_warp_id - 1; token_idx < token_end_idx; token_idx += num_recv_warps - 1) {
                // Read expected head
                int expected_head = -1;
                if (lane_id < kNumRanks)
                    expected_head = ld_nc_global(send_head + token_idx * kNumRanks + lane_id);

                auto start_time = clock64();
                while (__any_sync(0xffffffff, channel_tail_idx[lane_id] <= expected_head and expected_head >= 0)) {
                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine receivers, rank %d, responsible_channel = %d, expect = %d\n",
                               rank,
                               responsible_channel,
                               expected_head);
                        trap();
                    }
                }
                __syncwarp();

                // Broadcast current heads
                int num_topk_ranks = 0, topk_ranks[kNumRanks], slot_indices[kNumRanks];
                #pragma unroll
                for (int i = 0; i < kNumRanks; ++i) {
                    auto expected_head_i = __shfl_sync(0xffffffff, expected_head, i);
                    if (expected_head_i >= 0) {
                        slot_indices[num_topk_ranks] = expected_head_i % num_recv_buffer_tokens;
                        topk_ranks[num_topk_ranks++] = i;
                    }
                }

                // Wait shared memory release
#ifndef DISABLE_SM90_FEATURES
                tma_store_wait<0>();
                __syncwarp();
#endif

                // Reduce data with pipeline
                constexpr int kNumStages = 8;
                EP_STATIC_ASSERT(kNumStages * 32 * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid count");
                #pragma unroll
                for (int i = lane_id; i < hidden_int4; i += 32) {
                    // Read bias
                    // TODO: make it as a template
                    int4 bias_0_value_int4 =
                        bias_0_int4 != nullptr ? __ldg(bias_0_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);
                    int4 bias_1_value_int4 =
                        bias_1_int4 != nullptr ? __ldg(bias_1_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);

                    // Read buffers
                    int4 recv_value_int4[kNumRanks];
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j)
                        recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);

                    // Reduce bias
                    float values[kDtypePerInt4];
                    auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
                    auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);

                    // Reduce all-to-all results
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j) {
                        auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                        #pragma unroll
                        for (int k = 0; k < kDtypePerInt4; ++k)
                            values[k] += static_cast<float>(recv_value_dtypes[k]);
                    }

                    // Cast back to `dtype_t`
                    int4 out_int4;
                    auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        out_dtypes[j] = static_cast<dtype_t>(values[j]);

#ifndef DISABLE_SM90_FEATURES
                    if (i < hidden_int4_aligned) {
                        // Wait TMA arrival
                        tma_store_wait<kNumStages - 1>();
                        __syncwarp();

                        // Write into TMA buffer
                        auto tma_stage_idx = (i / 32) % kNumStages;
                        reinterpret_cast<int4*>(tma_buffer)[tma_stage_idx * 32 + lane_id] = out_int4;

                        // Issue TMA
                        tma_store_fence();
                        __syncwarp();
                        if (elect_one_sync()) {
                            auto tma_bytes = min(32, hidden_int4 - i) * static_cast<int>(sizeof(int4));
                            tma_store_1d(reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * 32,
                                         recv_int4 + token_idx * hidden_int4 + i,
                                         tma_bytes,
                                         false);
                        }
                        __syncwarp();
                    } else {
#endif
                        recv_int4[token_idx * hidden_int4 + i] = out_int4;
#ifndef DISABLE_SM90_FEATURES
                    }
#endif
                }

                // Reduce `topk_weights`
                if (lane_id < num_topk) {
                    float value = 0;
                    #pragma unroll
                    for (int i = 0; i < num_topk_ranks; ++i)
                        value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);
                    recv_topk_weights[token_idx * num_topk + lane_id] = value;
                }

                // Update head
                if (lane_id < kNumRanks)
                    warp_channel_head_idx[recv_warp_id][lane_id] = (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
            }

            // Retired
            __syncwarp();
            if (elect_one_sync())
                warp_retired[recv_warp_id] = true;
        }
    }
}

void combine(runtime_data_type_t type,
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
             runtime_stream_t stream,
             int num_sms,
             int num_max_send_tokens,
             int num_recv_buffer_tokens) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 4096;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

#define COMBINE_LAUNCH_CASE(dtype, ranks)                                      \
    {                                                                          \
        auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                     \
        LAUNCH_KERNEL(&cfg,                                                    \
                      kernel,                                                  \
                      reinterpret_cast<dtype*>(recv_x),                        \
                      recv_topk_weights,                                       \
                      reinterpret_cast<const dtype*>(x),                       \
                      topk_weights,                                            \
                      reinterpret_cast<const dtype*>(bias_0),                  \
                      reinterpret_cast<const dtype*>(bias_1),                  \
                      src_idx,                                                 \
                      rank_prefix_matrix,                                      \
                      channel_prefix_matrix,                                   \
                      send_head,                                               \
                      num_tokens,                                              \
                      num_recv_tokens,                                         \
                      hidden,                                                  \
                      num_topk,                                                \
                      buffer_ptrs,                                             \
                      rank,                                                    \
                      num_max_send_tokens,                                     \
                      num_recv_buffer_tokens);                                 \
    }                                                                          \
    break
#define COMBINE_DTYPE_LAUNCH_CASE(dtype)                 \
    SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving
    EP_HOST_ASSERT(num_sms % 2 == 0);
    EP_HOST_ASSERT(kNumThreads >= num_ranks * 32);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
}

#endif

}  // namespace intranode

}  // namespace deep_ep
