#include "xpu_runtime.hpp"

namespace deep_ep {

namespace {

constexpr int kLayoutThreads = 256;

template <typename T>
T* optional_ptr(T* ptr) {
    return ptr;
}

class LayoutExpertCountKernel;
class LayoutRankCountKernel;
class LayoutRDMARankCountKernel;

}  // namespace

void launch_get_dispatch_layout(const topk_idx_t* topk_idx,
                                int* num_tokens_per_rank,
                                int* num_tokens_per_rdma_rank,
                                int* num_tokens_per_expert,
                                bool* is_token_in_rank,
                                int num_tokens,
                                int num_topk,
                                int num_ranks,
                                int num_experts,
                                sycl::queue& queue) {
    TORCH_CHECK(num_tokens >= 0, "num_tokens must be non-negative");
    TORCH_CHECK(num_topk > 0, "num_topk must be positive");
    TORCH_CHECK(num_ranks > 0, "num_ranks must be positive");
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");

    const int num_experts_per_rank = num_experts / num_ranks;
    const int num_rdma_ranks = num_tokens_per_rdma_rank == nullptr ? 0 : std::max(1, num_ranks / NUM_MAX_NVL_PEERS);

    auto zero_rank_event = queue.memset(num_tokens_per_rank, 0, sizeof(int) * num_ranks);
    auto zero_expert_event = queue.memset(num_tokens_per_expert, 0, sizeof(int) * num_experts);
    auto zero_token_event = queue.memset(is_token_in_rank, 0, sizeof(bool) * num_tokens * num_ranks);
    std::vector<sycl::event> zero_events{zero_rank_event, zero_expert_event, zero_token_event};
    if (num_tokens_per_rdma_rank != nullptr) {
        zero_events.emplace_back(queue.memset(num_tokens_per_rdma_rank, 0, sizeof(int) * num_rdma_ranks));
    }

    const size_t topk_work_items = static_cast<size_t>(num_tokens) * num_topk;
    if (topk_work_items > 0) {
        const size_t global = align_up<size_t>(topk_work_items, kLayoutThreads);
        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(zero_events);
            cgh.parallel_for<LayoutExpertCountKernel>(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kLayoutThreads)), [=](sycl::nd_item<1> item) {
                    const size_t linear_idx = item.get_global_linear_id();
                    if (linear_idx >= topk_work_items) {
                        return;
                    }
                    const int expert_idx = static_cast<int>(topk_idx[linear_idx]);
                    if (expert_idx < 0) {
                        return;
                    }
                    if (expert_idx >= num_experts) {
                        return;
                    }
                    sycl::
                        atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>
                            expert_counter(num_tokens_per_expert[expert_idx]);
                    expert_counter.fetch_add(1);
                });
        });
    }

    const size_t rank_work_items = static_cast<size_t>(num_tokens) * num_ranks;
    if (rank_work_items > 0) {
        const size_t global = align_up<size_t>(rank_work_items, kLayoutThreads);
        queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(zero_events);
            cgh.parallel_for<LayoutRankCountKernel>(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kLayoutThreads)),
                                                    [=](sycl::nd_item<1> item) {
                                                        const size_t linear_idx = item.get_global_linear_id();
                                                        if (linear_idx >= rank_work_items) {
                                                            return;
                                                        }

                                                        const int token_idx = static_cast<int>(linear_idx / num_ranks);
                                                        const int rank_idx = static_cast<int>(linear_idx % num_ranks);
                                                        const int expert_begin = rank_idx * num_experts_per_rank;
                                                        const int expert_end = expert_begin + num_experts_per_rank;
                                                        bool in_rank = false;

                                                        for (int topk = 0; topk < num_topk; ++topk) {
                                                            const int expert_idx = static_cast<int>(topk_idx[token_idx * num_topk + topk]);
                                                            in_rank |= expert_begin <= expert_idx && expert_idx < expert_end;
                                                        }

                                                        is_token_in_rank[token_idx * num_ranks + rank_idx] = in_rank;
                                                        if (in_rank) {
                                                            sycl::atomic_ref<int,
                                                                             sycl::memory_order::relaxed,
                                                                             sycl::memory_scope::device,
                                                                             sycl::access::address_space::global_space>
                                                                rank_counter(num_tokens_per_rank[rank_idx]);
                                                            rank_counter.fetch_add(1);
                                                        }
                                                    });
        });
    }

    if (num_tokens_per_rdma_rank != nullptr) {
        const size_t rdma_work_items = static_cast<size_t>(num_tokens) * num_rdma_ranks;
        if (rdma_work_items > 0) {
            const size_t global = align_up<size_t>(rdma_work_items, kLayoutThreads);
            queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(zero_events);
                cgh.parallel_for<LayoutRDMARankCountKernel>(
                    sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kLayoutThreads)), [=](sycl::nd_item<1> item) {
                        const size_t linear_idx = item.get_global_linear_id();
                        if (linear_idx >= rdma_work_items) {
                            return;
                        }

                        const int token_idx = static_cast<int>(linear_idx / num_rdma_ranks);
                        const int rdma_rank_idx = static_cast<int>(linear_idx % num_rdma_ranks);
                        const int expert_begin = rdma_rank_idx * NUM_MAX_NVL_PEERS * num_experts_per_rank;
                        const int expert_end = std::min(expert_begin + NUM_MAX_NVL_PEERS * num_experts_per_rank, num_experts);
                        bool in_rdma_rank = false;

                        for (int topk = 0; topk < num_topk; ++topk) {
                            const int expert_idx = static_cast<int>(topk_idx[token_idx * num_topk + topk]);
                            in_rdma_rank |= expert_begin <= expert_idx && expert_idx < expert_end;
                        }

                        if (in_rdma_rank) {
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                rdma_counter(num_tokens_per_rdma_rank[rdma_rank_idx]);
                            rdma_counter.fetch_add(1);
                        }
                    });
            });
        }
    }
}

}  // namespace deep_ep
