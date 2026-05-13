#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(DEEPEP_USE_ISHMEM) && __has_include(<ishmem.h>)
#include <ishmem.h>
#define DEEPEP_TRANSPORT_USE_ISHMEM 1
#elif !defined(DISABLE_NVSHMEM)
#include <nvshmem.h>
#define DEEPEP_TRANSPORT_USE_NVSHMEM 1
#endif

namespace deep_ep::transport {

#if defined(DEEPEP_TRANSPORT_USE_ISHMEM)
using TeamHandle = ishmem_team_t;
#elif defined(DEEPEP_TRANSPORT_USE_NVSHMEM)
using TeamHandle = nvshmem_team_t;
#else
using TeamHandle = int;
#endif

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode);

void* alloc(size_t size, size_t alignment);

void free(void* ptr);

void barrier();

void finalize();

}  // namespace deep_ep::transport
