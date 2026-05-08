#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"

#include <algorithm>
#include <vector>

namespace deep_ep {

namespace layout {

#if defined(DEEPEP_USE_XPU)

void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         runtime_stream_t stream) {
    EP_HOST_ASSERT(topk_idx != nullptr and "topk_idx must not be null");
    EP_HOST_ASSERT(num_tokens_per_rank != nullptr and "num_tokens_per_rank must not be null");
    EP_HOST_ASSERT(num_tokens_per_expert != nullptr and "num_tokens_per_expert must not be null");
    EP_HOST_ASSERT(is_token_in_rank != nullptr and "is_token_in_rank must not be null");
    EP_HOST_ASSERT(num_tokens >= 0 and num_topk >= 0 and num_ranks > 0 and num_experts > 0);

    if (num_tokens_per_rdma_rank != nullptr)
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);

    const int num_experts_per_rank = num_experts / num_ranks;
    EP_HOST_ASSERT(num_experts_per_rank > 0 and "num_experts must be at least num_ranks");

    const size_t topk_count = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_topk);
    std::vector<topk_idx_t> host_topk(topk_count);
    if (topk_count > 0) {
        auto copy_event = stream.queue().memcpy(host_topk.data(), topk_idx, topk_count * sizeof(topk_idx_t));
        copy_event.wait_and_throw();
    }

    std::vector<int> host_num_tokens_per_rank(static_cast<size_t>(num_ranks), 0);
    std::vector<int> host_num_tokens_per_expert(static_cast<size_t>(num_experts), 0);
    std::vector<uint8_t> host_is_token_in_rank(static_cast<size_t>(num_tokens) * static_cast<size_t>(num_ranks), 0);
    std::vector<int> host_num_tokens_per_rdma_rank;
    const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    if (num_tokens_per_rdma_rank != nullptr)
        host_num_tokens_per_rdma_rank.assign(static_cast<size_t>(num_rdma_ranks), 0);

    std::vector<uint8_t> token_seen_rank(static_cast<size_t>(num_ranks));
    std::vector<uint8_t> token_seen_rdma_rank;
    if (num_tokens_per_rdma_rank != nullptr)
        token_seen_rdma_rank.resize(static_cast<size_t>(num_rdma_ranks));

    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        std::fill(token_seen_rank.begin(), token_seen_rank.end(), 0);
        if (num_tokens_per_rdma_rank != nullptr)
            std::fill(token_seen_rdma_rank.begin(), token_seen_rdma_rank.end(), 0);

        const auto* token_topk = host_topk.data() + static_cast<size_t>(token_idx) * static_cast<size_t>(num_topk);
        for (int k = 0; k < num_topk; ++k) {
            const int expert_idx = static_cast<int>(token_topk[k]);
            if (expert_idx < 0 or expert_idx >= num_experts)
                continue;

            host_num_tokens_per_expert[expert_idx]++;
            const int rank_idx = expert_idx / num_experts_per_rank;
            if (rank_idx < 0 or rank_idx >= num_ranks)
                continue;

            token_seen_rank[rank_idx] = 1;
            if (num_tokens_per_rdma_rank != nullptr)
                token_seen_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS] = 1;
        }

        auto* token_flags = host_is_token_in_rank.data() + static_cast<size_t>(token_idx) * static_cast<size_t>(num_ranks);
        for (int rank_idx = 0; rank_idx < num_ranks; ++rank_idx) {
            token_flags[rank_idx] = token_seen_rank[rank_idx];
            host_num_tokens_per_rank[rank_idx] += static_cast<int>(token_seen_rank[rank_idx] != 0);
        }

        if (num_tokens_per_rdma_rank != nullptr) {
            for (int rdma_rank = 0; rdma_rank < num_rdma_ranks; ++rdma_rank)
                host_num_tokens_per_rdma_rank[rdma_rank] += static_cast<int>(token_seen_rdma_rank[rdma_rank] != 0);
        }
    }

    auto rank_copy_event = stream.queue().memcpy(
        num_tokens_per_rank, host_num_tokens_per_rank.data(), static_cast<size_t>(num_ranks) * sizeof(int));
    auto expert_copy_event = stream.queue().memcpy(
        num_tokens_per_expert, host_num_tokens_per_expert.data(), static_cast<size_t>(num_experts) * sizeof(int));
    auto token_copy_event = stream.queue().memcpy(
        is_token_in_rank, host_is_token_in_rank.data(), static_cast<size_t>(num_tokens) * static_cast<size_t>(num_ranks) * sizeof(uint8_t));
    rank_copy_event.wait_and_throw();
    expert_copy_event.wait_and_throw();
    token_copy_event.wait_and_throw();

    if (num_tokens_per_rdma_rank != nullptr) {
        auto rdma_copy_event = stream.queue().memcpy(
            num_tokens_per_rdma_rank, host_num_tokens_per_rdma_rank.data(), static_cast<size_t>(num_rdma_ranks) * sizeof(int));
        rdma_copy_event.wait_and_throw();
    }
}

#else

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(const topk_idx_t* topk_idx,
                                    int* num_tokens_per_rank,
                                    int* num_tokens_per_rdma_rank,
                                    int* num_tokens_per_expert,
                                    bool* is_token_in_rank,
                                    int num_tokens,
                                    int num_topk,
                                    int num_ranks,
                                    int num_experts) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);

    // Count expert statistics
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
    int expert_begin_idx = sm_id * kNumExpertsPerSM, expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
    if (expert_begin_idx < expert_end_idx) {
        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumExpertsPerSM; ++i)
            num_tokens_per_expert_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
            #pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
            }
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
        if (expert_begin_idx + thread_id < expert_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
        }
        return;
    }

    if (num_tokens_per_rdma_rank != nullptr)
        EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);

    // Count rank statistics
    constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
    auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
    int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM, rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
    int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS, rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;
    if (rank_begin_idx < rank_end_idx) {
        const auto num_expert_per_rank = num_experts / num_ranks;
        auto expert_begin = rank_begin_idx * num_expert_per_rank;
        auto expert_end = rank_end_idx * num_expert_per_rank;

        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
            int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
            #pragma unroll
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin <= expert_idx and expert_idx < expert_end) {
                    // Count single rank
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
                    is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
                }
            }

            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
            #pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
            }

            #pragma unroll
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
        if (rank_begin_idx + thread_id < rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
        }

        if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
        }
    }
}

void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         runtime_stream_t stream) {
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
    int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
    EP_STATIC_ASSERT(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg,
                  (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
                  topk_idx,
                  num_tokens_per_rank,
                  num_tokens_per_rdma_rank,
                  num_tokens_per_expert,
                  is_token_in_rank,
                  num_tokens,
                  num_topk,
                  num_ranks,
                  num_experts);
}

#endif

}  // namespace layout

}  // namespace deep_ep
