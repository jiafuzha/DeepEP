#include <cstring>
#include <vector>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "transport.hpp"
#include "utils.cuh"

#if defined(DEEPEP_TRANSPORT_USE_NVSHMEM) && !defined(DISABLE_NVSHMEM)
#include "ibgda_device.cuh"
#include "nvshmem.h"
#elif defined(DEEPEP_TRANSPORT_USE_ISHMEM)
#include "ishmemx.h"
#endif

namespace deep_ep {

namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                  \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 32, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace intranode

namespace transport {

#if defined(DEEPEP_TRANSPORT_USE_ISHMEM)
transport::TeamHandle cpu_rdma_team = ISHMEM_TEAM_INVALID;
ishmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    ishmemx_uniqueid_t unique_id;
    EP_HOST_ASSERT(ishmemx_get_uniqueid(&unique_id) == 0);
    std::vector<uint8_t> result(sizeof(ishmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(ishmemx_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    ishmemx_uniqueid_t root_unique_id;
    ishmemx_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(ishmemx_uniqueid_t));
    attr.use_uid = true;
    attr.uid = &root_unique_id;
    attr.rank = rank;
    attr.nranks = num_ranks;
    ishmemx_init_attr(&attr);

    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(cpu_rdma_team == ISHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(ishmem_team_split_strided(ISHMEM_TEAM_WORLD,
                                                 rank % NUM_MAX_NVL_PEERS,
                                                 NUM_MAX_NVL_PEERS,
                                                 num_ranks / NUM_MAX_NVL_PEERS,
                                                 &cpu_rdma_team_config,
                                                 0,
                                                 &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != ISHMEM_TEAM_INVALID);
    }

    ishmem_barrier_all();
    return ishmem_my_pe();
}

void* alloc(size_t size, size_t alignment) {
    return ishmem_align(alignment, size);
}

void free(void* ptr) {
    ishmem_free(ptr);
}

void barrier() {
    ishmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != ISHMEM_TEAM_INVALID) {
        ishmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = ISHMEM_TEAM_INVALID;
    }
    ishmem_finalize();
}
#elif defined(DEEPEP_TRANSPORT_USE_NVSHMEM) && !defined(DISABLE_NVSHMEM)
transport::TeamHandle cpu_rdma_team = NVSHMEM_TEAM_INVALID;
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
#else
std::vector<uint8_t> get_unique_id() {
    EP_HOST_ASSERT(false and "NVSHMEM transport is disabled during compilation");
    return {};
}

int init(const std::vector<uint8_t>&, int, int, bool) {
    EP_HOST_ASSERT(false and "NVSHMEM transport is disabled during compilation");
    return -1;
}

void* alloc(size_t, size_t) {
    EP_HOST_ASSERT(false and "NVSHMEM transport is disabled during compilation");
    return nullptr;
}

void free(void*) {
    EP_HOST_ASSERT(false and "NVSHMEM transport is disabled during compilation");
}

void barrier() {
    EP_HOST_ASSERT(false and "NVSHMEM transport is disabled during compilation");
}

void finalize() {
    EP_HOST_ASSERT(false and "NVSHMEM transport is disabled during compilation");
}
#endif

}  // namespace transport

namespace internode {

std::vector<uint8_t> get_unique_id() {
    return transport::get_unique_id();
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    return transport::init(root_unique_id_val, rank, num_ranks, low_latency_mode);
}

void* alloc(size_t size, size_t alignment) {
    return transport::alloc(size, alignment);
}

void free(void* ptr) {
    transport::free(ptr);
}

void barrier() {
    transport::barrier();
}

void finalize() {
    transport::finalize();
}

}  // namespace internode

}  // namespace deep_ep
