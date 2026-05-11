#include <cstring>
#include <cstdlib>
#include <vector>

#if defined(DEEPEP_USE_XPU)
#include "configs.cuh"
#include "exception.cuh"

#include <c10/xpu/XPUCachingAllocator.h>
#else
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#ifndef DISABLE_NVSHMEM
#include "ibgda_device.cuh"
#include "nvshmem.h"
#endif
#endif

namespace deep_ep {

namespace intranode {

#if defined(DEEPEP_USE_XPU)

void barrier(int**, int, int, runtime_stream_t) {
    // Staged XPU migration: no-op barrier for single-process bring-up.
}

#else

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, runtime_stream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                  \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 32, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

#endif

}  // namespace intranode

namespace internode {

#if defined(DEEPEP_USE_XPU)

std::vector<uint8_t> get_unique_id() {
    // Keep a stable payload size to match callsites expecting an opaque ID blob.
    std::vector<uint8_t> unique_id(16, 0);
    unique_id[0] = 0x58;  // 'X'
    unique_id[1] = 0x50;  // 'P'
    unique_id[2] = 0x55;  // 'U'
    return unique_id;
}

int init(const std::vector<uint8_t>&, int rank, int, bool) {
    // Staged XPU migration: treat init as successful local initialization.
    return rank;
}

void* alloc(size_t size, size_t alignment) {
    if (size == 0)
        return nullptr;
    (void)alignment;
    return c10::xpu::XPUCachingAllocator::raw_alloc(size);
}

void free(void* ptr) {
    if (ptr != nullptr)
        c10::xpu::XPUCachingAllocator::raw_delete(ptr);
}

void barrier() {
    // Staged XPU migration: no-op barrier.
}

void finalize() {
    // Staged XPU migration: nothing to release in fallback runtime.
}

#else

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

#endif

}  // namespace internode

}  // namespace deep_ep
