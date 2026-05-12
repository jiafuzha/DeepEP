#include <sycl/sycl.hpp>
#include <cstring>
#include <vector>

#include "configs.dp.hpp"
#include "exception.dp.hpp"
#include "launch.dp.hpp"
#include "utils.dp.hpp"

#ifndef DISABLE_NVSHMEM
#include "ibgda_device.dp.hpp"
#endif

namespace deep_ep {

namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, dpct::queue_ptr stream) {
    auto launch_barrier = [&](auto ranks_tag) {
        constexpr int ranks = decltype(ranks_tag)::value;
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)), [=](sycl::nd_item<1>) {
                barrier_block<ranks>(barrier_signal_ptrs, rank);
            });
        });
    };

    switch (num_ranks) {
        case 1:
            launch_barrier(std::integral_constant<int, 1>{});
            break;
        case 2:
            launch_barrier(std::integral_constant<int, 2>{});
            break;
        case 4:
            launch_barrier(std::integral_constant<int, 4>{});
            break;
        case 8:
            launch_barrier(std::integral_constant<int, 8>{});
            break;
        default:
            EP_HOST_ASSERT(false && "Unsupported number of ranks for intranode barrier");
    }
}

}  // namespace intranode

namespace internode {

#ifndef DISABLE_NVSHMEM
nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                                  rank % NUM_MAX_NVL_PEERS,
                                                  NUM_MAX_NVL_PEERS,
                                                  num_ranks / NUM_MAX_NVL_PEERS,
                                                  &cpu_rdma_team_config,
                                                  0,
                                                  &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }

    nvshmem_barrier_all();
    return nvshmem_my_pe();
}

void* alloc(size_t size, size_t alignment) {
    return nvshmem_align(alignment, size);
}

void free(void* ptr) {
    nvshmem_free(ptr);
}

void barrier() {
    nvshmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
}
#endif

}  // namespace internode

}  // namespace deep_ep
