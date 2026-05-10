#include "deep_ep.hpp"

#if defined(DEEPEP_USE_XPU)
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/PinnedMemoryAllocator.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <sycl/sycl.hpp>
#else
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#endif
#include <pybind11/functional.h>
#include <torch/python.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <unistd.h>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"

namespace shared_memory {

#if defined(DEEPEP_USE_XPU)
namespace {

struct XpuLocalMemHandle {
    uint32_t magic;
    uint16_t version;
    uint16_t kind;
    uint16_t flags;
    uint16_t reserved0;
    uint64_t generation;
    uint64_t ptr_value;
    uint64_t size;
    uint64_t pid;
    uint64_t checksum;
};

constexpr uint32_t kXpuLocalMemHandleMagic = 0x58495043u;  // "XIPC"
constexpr uint16_t kXpuLocalMemHandleVersion = 1;
constexpr uint16_t kXpuLocalMemHandleKindDeviceAllocation = 1;
constexpr uint16_t kXpuLocalMemHandleKindFabricHint = 2;

inline uint64_t fnv1a64_mix(uint64_t seed, uint64_t value) {
    constexpr uint64_t kFnvOffset = 1469598103934665603ull;
    constexpr uint64_t kFnvPrime = 1099511628211ull;
    uint64_t hash = seed == 0 ? kFnvOffset : seed;
    for (int i = 0; i < 8; ++i) {
        hash ^= static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
        hash *= kFnvPrime;
    }
    return hash;
}

inline uint64_t compute_xpu_local_handle_checksum(const XpuLocalMemHandle& local) {
    uint64_t hash = 0;
    hash = fnv1a64_mix(hash, local.magic);
    hash = fnv1a64_mix(hash, local.version);
    hash = fnv1a64_mix(hash, local.kind);
    hash = fnv1a64_mix(hash, local.flags);
    hash = fnv1a64_mix(hash, local.reserved0);
    hash = fnv1a64_mix(hash, local.generation);
    hash = fnv1a64_mix(hash, local.ptr_value);
    hash = fnv1a64_mix(hash, local.size);
    hash = fnv1a64_mix(hash, local.pid);
    return hash;
}

struct XpuLiveAllocationMeta {
    size_t size;
    uint64_t generation;
};

std::mutex xpu_live_allocations_mu;
std::unordered_map<void*, XpuLiveAllocationMeta> xpu_live_allocations;
uint64_t xpu_next_allocation_generation = 1;

inline uint64_t register_xpu_live_allocation(void* ptr, size_t size) {
    std::lock_guard<std::mutex> guard(xpu_live_allocations_mu);
    EP_HOST_ASSERT(ptr != nullptr and size > 0 and "Invalid XPU live allocation registration");
    const uint64_t generation = xpu_next_allocation_generation++;
    const auto [it, inserted] = xpu_live_allocations.emplace(ptr, XpuLiveAllocationMeta{size, generation});
    EP_HOST_ASSERT(inserted and "Duplicate XPU live allocation registration");
    return it->second.generation;
}

inline void unregister_xpu_live_allocation(void* ptr, size_t size, uint64_t generation) {
    std::lock_guard<std::mutex> guard(xpu_live_allocations_mu);
    const auto it = xpu_live_allocations.find(ptr);
    EP_HOST_ASSERT(it != xpu_live_allocations.end() and "Unknown XPU live allocation unregistration");
    EP_HOST_ASSERT(it->second.size == size and "XPU live allocation size mismatch on unregistration");
    EP_HOST_ASSERT(it->second.generation == generation and "XPU live allocation generation mismatch on unregistration");
    xpu_live_allocations.erase(it);
}

inline void validate_xpu_live_allocation(void* ptr, size_t size, uint64_t generation) {
    std::lock_guard<std::mutex> guard(xpu_live_allocations_mu);
    const auto it = xpu_live_allocations.find(ptr);
    EP_HOST_ASSERT(it != xpu_live_allocations.end() and "XPU IPC handle points to non-live allocation");
    EP_HOST_ASSERT(it->second.size == size and "XPU IPC handle size mismatch with live allocation");
    EP_HOST_ASSERT(it->second.generation == generation and "XPU IPC handle generation is stale");
}

static_assert(sizeof(XpuLocalMemHandle) <= sizeof(XpuIpcMemHandle::opaque),
              "XpuLocalMemHandle must fit inside XpuIpcMemHandle opaque payload");

inline void encode_xpu_local_handle(XpuIpcMemHandle* handle,
                                    void* ptr,
                                    size_t size,
                                    uint64_t generation,
                                    uint16_t kind) {
    XpuLocalMemHandle local{};
    local.magic = kXpuLocalMemHandleMagic;
    local.version = kXpuLocalMemHandleVersion;
    local.kind = kind;
    local.flags = 0;
    local.reserved0 = 0;
    local.generation = generation;
    local.ptr_value = reinterpret_cast<uint64_t>(ptr);
    local.size = static_cast<uint64_t>(size);
    local.pid = static_cast<uint64_t>(getpid());
    local.checksum = compute_xpu_local_handle_checksum(local);
    std::memset(handle->opaque, 0, sizeof(handle->opaque));
    std::memcpy(handle->opaque, &local, sizeof(local));
}

inline XpuLocalMemHandle decode_xpu_local_handle(const XpuIpcMemHandle& handle) {
    XpuLocalMemHandle local{};
    std::memcpy(&local, handle.opaque, sizeof(local));
    EP_HOST_ASSERT(local.magic == kXpuLocalMemHandleMagic and "Invalid XPU local IPC handle magic");
    EP_HOST_ASSERT(local.version == kXpuLocalMemHandleVersion and "Unsupported XPU local IPC handle version");
    EP_HOST_ASSERT((local.kind == kXpuLocalMemHandleKindDeviceAllocation or
                    local.kind == kXpuLocalMemHandleKindFabricHint) and
                   "Unsupported XPU local IPC handle kind");
    EP_HOST_ASSERT(local.flags == 0 and "Unsupported XPU local IPC handle flags");
    EP_HOST_ASSERT(local.reserved0 == 0 and "Invalid XPU local IPC handle reserved payload");
    EP_HOST_ASSERT(local.generation > 0 and "XPU local IPC handle has invalid generation payload");
    EP_HOST_ASSERT(local.ptr_value != 0 and "XPU local IPC handle has null pointer payload");
    EP_HOST_ASSERT(local.size > 0 and "XPU local IPC handle has invalid size payload");
    EP_HOST_ASSERT(local.checksum == compute_xpu_local_handle_checksum(local) and
                   "Invalid XPU local IPC handle checksum");
    return local;
}

inline void* build_xpu_foreign_handle_token(const XpuLocalMemHandle& local) {
    uint64_t hash = 0;
    hash = fnv1a64_mix(hash, local.magic);
    hash = fnv1a64_mix(hash, local.version);
    hash = fnv1a64_mix(hash, local.kind);
    hash = fnv1a64_mix(hash, local.generation);
    hash = fnv1a64_mix(hash, local.ptr_value);
    hash = fnv1a64_mix(hash, local.size);
    hash = fnv1a64_mix(hash, local.pid);
    hash = fnv1a64_mix(hash, local.checksum);
    const uintptr_t token = static_cast<uintptr_t>((hash << 1) | 1ull);
    return reinterpret_cast<void*>(token == 0 ? 1ull : token);
}

}  // namespace
#endif

#if !defined(DEEPEP_USE_XPU)
void cu_mem_set_access_all(void* ptr, size_t size) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    CUmemAccessDesc access_desc[device_count];
    for (int idx = 0; idx < device_count; ++idx) {
        access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[idx].location.id = idx;
        access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    CU_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, access_desc, device_count));
}

void cu_mem_free(void* ptr) {
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(handle));
}

size_t get_size_align_to_granularity(size_t size_raw, size_t granularity) {
    size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
    if (size == 0)
        size = granularity;
    return size;
}
#endif

SharedMemoryAllocator::SharedMemoryAllocator(bool use_fabric) : use_fabric(use_fabric) {}

void SharedMemoryAllocator::malloc(void** ptr, size_t size_raw) {
#if defined(DEEPEP_USE_XPU)
    EP_HOST_ASSERT(ptr != nullptr and "XPU malloc got null output pointer");
    const auto alloc_size = size_raw == 0 ? 1 : size_raw;
    *ptr = c10::xpu::XPUCachingAllocator::raw_alloc(alloc_size);
    EP_HOST_ASSERT(*ptr != nullptr and "XPU shared memory allocation failed");
    const auto generation = register_xpu_live_allocation(*ptr, alloc_size);
    {
        std::lock_guard<std::mutex> guard(allocation_sizes_mu);
        allocation_sizes[*ptr] = {alloc_size, generation};
    }
#else
    if (use_fabric) {
        CUdevice device;
        CU_CHECK(cuCtxGetDevice(&device));

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.id = device;

        size_t granularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        size_t size = get_size_align_to_granularity(size_raw, granularity);

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemCreate(&handle, size, &prop, 0));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaMalloc(ptr, size_raw));
    }
#endif
}

void SharedMemoryAllocator::free(void* ptr) {
#if defined(DEEPEP_USE_XPU)
    if (ptr != nullptr) {
        std::pair<size_t, uint64_t> meta{};
        {
            std::lock_guard<std::mutex> guard(allocation_sizes_mu);
            const auto it = allocation_sizes.find(ptr);
            EP_HOST_ASSERT(it != allocation_sizes.end() and "XPU shared memory free got unknown pointer");
            meta = it->second;
            allocation_sizes.erase(it);
        }
        unregister_xpu_live_allocation(ptr, meta.first, meta.second);
        c10::xpu::XPUCachingAllocator::raw_delete(ptr);
    }
#else
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }
#endif
}

void SharedMemoryAllocator::get_mem_handle(MemHandle* mem_handle, void* ptr) {
#if defined(DEEPEP_USE_XPU)
    EP_HOST_ASSERT(mem_handle != nullptr and ptr != nullptr and "XPU IPC handle export got null input");
    std::pair<size_t, uint64_t> meta{};
    {
        std::lock_guard<std::mutex> guard(allocation_sizes_mu);
        const auto it = allocation_sizes.find(ptr);
        EP_HOST_ASSERT(it != allocation_sizes.end() and "XPU IPC handle export got unknown pointer");
        meta = it->second;
    }
    mem_handle->size = meta.first;
    const auto handle_kind = use_fabric ? kXpuLocalMemHandleKindFabricHint : kXpuLocalMemHandleKindDeviceAllocation;
    encode_xpu_local_handle(&mem_handle->inner.xpu_ipc_mem_handle, ptr, meta.first, meta.second, handle_kind);
#else
    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    mem_handle->size = size;

    if (use_fabric) {
        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

        CU_CHECK(cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    } else {
        CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }
#endif
}

void SharedMemoryAllocator::open_mem_handle(void** ptr, MemHandle* mem_handle) {
#if defined(DEEPEP_USE_XPU)
    EP_HOST_ASSERT(ptr != nullptr and mem_handle != nullptr and "XPU IPC handle import got null input");
    const auto local = decode_xpu_local_handle(mem_handle->inner.xpu_ipc_mem_handle);
    const auto expected_kind = use_fabric ? kXpuLocalMemHandleKindFabricHint : kXpuLocalMemHandleKindDeviceAllocation;
    EP_HOST_ASSERT(local.kind == expected_kind and
                   "XPU IPC handle kind mismatch between exporter/importer transport mode");
    EP_HOST_ASSERT(local.size > 0 and "XPU IPC handle import got invalid allocation size");
    EP_HOST_ASSERT(mem_handle->size == static_cast<size_t>(local.size) and
                   "XPU IPC handle import size mismatch");
    const bool same_process = (local.pid == static_cast<uint64_t>(getpid()));
    if (same_process)
        validate_xpu_live_allocation(reinterpret_cast<void*>(local.ptr_value), static_cast<size_t>(local.size), local.generation);

    void* opened_ptr = same_process ? reinterpret_cast<void*>(local.ptr_value) : build_xpu_foreign_handle_token(local);
    EP_HOST_ASSERT(opened_ptr != nullptr and "XPU IPC handle import decoded null pointer");
    {
        std::lock_guard<std::mutex> guard(imported_allocations_mu);
        const auto [it, inserted] = imported_allocations.emplace(
            opened_ptr,
            SharedMemoryAllocator::ImportedAllocationMeta{static_cast<size_t>(local.size), local.generation, same_process});
        EP_HOST_ASSERT(inserted and "XPU IPC handle import got duplicate pointer without close");
        EP_HOST_ASSERT(it->second.size == static_cast<size_t>(local.size) and "XPU IPC handle import got inconsistent size");
        EP_HOST_ASSERT(it->second.generation == local.generation and "XPU IPC handle import got inconsistent generation");
        EP_HOST_ASSERT(it->second.require_live_validation == same_process and
                       "XPU IPC handle import got inconsistent process-local validation requirement");
    }
    *ptr = opened_ptr;
#else
    if (use_fabric) {
        size_t size = mem_handle->size;

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
    }
#endif
}

void SharedMemoryAllocator::close_mem_handle(void* ptr) {
#if defined(DEEPEP_USE_XPU)
    if (ptr == nullptr)
        return;
    SharedMemoryAllocator::ImportedAllocationMeta imported_meta{};
    {
        std::lock_guard<std::mutex> guard(imported_allocations_mu);
        const auto it = imported_allocations.find(ptr);
        EP_HOST_ASSERT(it != imported_allocations.end() and "XPU IPC handle close got unknown pointer");
        imported_meta = it->second;
        imported_allocations.erase(it);
    }

    if (not imported_meta.require_live_validation)
        return;

    // Teardown can legitimately happen after the exporting buffer has already
    // released its allocation. Keep close order-independent across local ranks.
    std::lock_guard<std::mutex> live_guard(xpu_live_allocations_mu);
    const auto live_it = xpu_live_allocations.find(ptr);
    if (live_it == xpu_live_allocations.end())
        return;
    EP_HOST_ASSERT(live_it->second.size == imported_meta.size and
                   "XPU IPC handle close size mismatch with live allocation");
    EP_HOST_ASSERT(live_it->second.generation == imported_meta.generation and
                   "XPU IPC handle close generation mismatch with live allocation");
#else
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
    }
#endif
}

#if defined(DEEPEP_USE_XPU)
bool SharedMemoryAllocator::requires_live_validation(void* ptr) const {
    if (ptr == nullptr)
        return false;
    std::lock_guard<std::mutex> guard(imported_allocations_mu);
    const auto it = imported_allocations.find(ptr);
    EP_HOST_ASSERT(it != imported_allocations.end() and "XPU IPC handle query got unknown pointer");
    return it->second.require_live_validation;
}
#endif
}  // namespace shared_memory

namespace deep_ep {

#if defined(DEEPEP_USE_XPU)
namespace {
std::mutex xpu_host_fallback_alloc_mu;

enum class XpuHostFallbackAllocationKind : uint8_t {
    kMalloc,
    kSyclShared,
};

std::unordered_map<void*, XpuHostFallbackAllocationKind> xpu_host_fallback_allocations;
}  // namespace
#endif

inline runtime_data_type_t get_runtime_dtype(torch::ScalarType scalar_type) {
#if defined(DEEPEP_USE_XPU)
    return scalar_type == torch::kBFloat16 ? RUNTIME_R_16BF : static_cast<runtime_data_type_t>(scalar_type);
#else
    return at::cuda::ScalarTypeToCudaDataType(scalar_type);
#endif
}

inline void runtime_device_synchronize() {
#if defined(DEEPEP_USE_XPU)
    c10::xpu::syncStreamsOnDevice();
#else
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

inline void runtime_memcpy_h2d(void* dst, const void* src, size_t bytes) {
#if defined(DEEPEP_USE_XPU)
    EP_HOST_ASSERT(dst != nullptr and src != nullptr and "XPU memcpy got null pointer");
    auto event = backend::get_current_stream().queue().memcpy(dst, src, bytes);
    event.wait_and_throw();
#else
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#endif
}

inline void runtime_memset_async(void* ptr, int value, size_t bytes, const backend::Stream& stream) {
#if defined(DEEPEP_USE_XPU)
    EP_HOST_ASSERT(ptr != nullptr and "XPU memset_async got null pointer");
    stream.queue().memset(ptr, value, bytes);
#else
    CUDA_CHECK(cudaMemsetAsync(ptr, value, bytes, stream));
#endif
}

inline void runtime_memset(void* ptr, int value, size_t bytes) {
#if defined(DEEPEP_USE_XPU)
    EP_HOST_ASSERT(ptr != nullptr and "XPU memset got null pointer");
    auto event = backend::get_current_stream().queue().memset(ptr, value, bytes);
    event.wait_and_throw();
#else
    CUDA_CHECK(cudaMemset(ptr, value, bytes));
#endif
}

inline void runtime_malloc(void** ptr, size_t bytes) {
#if defined(DEEPEP_USE_XPU)
    EP_HOST_ASSERT(ptr != nullptr and "XPU malloc got null output pointer");
    *ptr = c10::xpu::XPUCachingAllocator::raw_alloc(bytes == 0 ? 1 : bytes);
    EP_HOST_ASSERT(*ptr != nullptr and "XPU malloc failed");
#else
    CUDA_CHECK(cudaMalloc(ptr, bytes));
#endif
}

inline void runtime_free(void* ptr) {
#if defined(DEEPEP_USE_XPU)
    if (ptr != nullptr)
        c10::xpu::XPUCachingAllocator::raw_delete(ptr);
#else
    CUDA_CHECK(cudaFree(ptr));
#endif
}

inline void runtime_malloc_host_mapped(void** host_ptr, void** device_ptr, size_t bytes) {
#if defined(DEEPEP_USE_XPU)
    EP_HOST_ASSERT(host_ptr != nullptr and device_ptr != nullptr and "XPU mapped allocation got null pointer");
    const auto alloc_size = bytes == 0 ? 1 : bytes;
    auto* allocator = at::xpu::getPinnedMemoryAllocator();
    if (allocator != nullptr and allocator->raw_deleter() != nullptr) {
        *host_ptr = allocator->raw_allocate(alloc_size);
    } else {
        // Prefer USM shared allocations so fallback pointers remain device accessible.
        try {
            *host_ptr = sycl::malloc_shared(alloc_size, backend::get_current_stream().queue());
        } catch (...) {
            *host_ptr = nullptr;
        }
        if (*host_ptr != nullptr) {
            std::lock_guard<std::mutex> guard(xpu_host_fallback_alloc_mu);
            xpu_host_fallback_allocations.emplace(*host_ptr, XpuHostFallbackAllocationKind::kSyclShared);
        } else {
            // Fall back to host heap only as a last resort.
            *host_ptr = std::malloc(alloc_size);
            if (*host_ptr != nullptr) {
                std::lock_guard<std::mutex> guard(xpu_host_fallback_alloc_mu);
                xpu_host_fallback_allocations.emplace(*host_ptr, XpuHostFallbackAllocationKind::kMalloc);
            }
        }
    }
    EP_HOST_ASSERT(*host_ptr != nullptr and "XPU mapped host allocation failed");
    *device_ptr = *host_ptr;
#else
    CUDA_CHECK(cudaMallocHost(host_ptr, bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(device_ptr, *host_ptr, 0));
#endif
}

inline void runtime_free_host(void* ptr) {
#if defined(DEEPEP_USE_XPU)
    if (ptr != nullptr) {
        {
            std::lock_guard<std::mutex> guard(xpu_host_fallback_alloc_mu);
            const auto it = xpu_host_fallback_allocations.find(ptr);
            if (it != xpu_host_fallback_allocations.end()) {
                const auto alloc_kind = it->second;
                xpu_host_fallback_allocations.erase(it);
                if (alloc_kind == XpuHostFallbackAllocationKind::kSyclShared) {
                    sycl::free(ptr, backend::get_current_stream().queue());
                } else {
                    std::free(ptr);
                }
                return;
            }
        }
        auto* allocator = at::xpu::getPinnedMemoryAllocator();
        EP_HOST_ASSERT(allocator != nullptr and allocator->raw_deleter() != nullptr and
                       "XPU pinned host allocator is unavailable");
        allocator->raw_deallocate(ptr);
    }
#else
    CUDA_CHECK(cudaFreeHost(ptr));
#endif
}

inline void runtime_get_device(int* device_id) {
#if defined(DEEPEP_USE_XPU)
    EP_HOST_ASSERT(device_id != nullptr and "XPU get-device got null pointer");
    *device_id = static_cast<int>(c10::xpu::current_device());
#else
    CUDA_CHECK(cudaGetDevice(device_id));
#endif
}

inline int runtime_get_num_sms(int device_id) {
#if defined(DEEPEP_USE_XPU)
    auto* device_prop = at::xpu::getDeviceProperties(device_id);
    EP_HOST_ASSERT(device_prop != nullptr and "XPU device properties are unavailable");
    if (device_prop->max_compute_units > 0)
        return static_cast<int>(device_prop->max_compute_units);
    return 128;
#else
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    return device_prop.multiProcessorCount;
#endif
}

#if defined(DEEPEP_USE_XPU)
inline uint64_t xpu_hash_bytes_fnv1a(const uint8_t* data, size_t size) {
    constexpr uint64_t kFnvOffset = 1469598103934665603ull;
    constexpr uint64_t kFnvPrime = 1099511628211ull;
    uint64_t hash = kFnvOffset;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= kFnvPrime;
    }
    return hash;
}

inline void build_xpu_ipc_peer_tokens(const shared_memory::MemHandle* handles, int num_ranks, void** out_tokens) {
    EP_HOST_ASSERT(handles != nullptr and out_tokens != nullptr and num_ranks > 0);
    for (int i = 0; i < num_ranks; ++i) {
        auto hash = xpu_hash_bytes_fnv1a(handles[i].inner.xpu_ipc_mem_handle.opaque,
                                         sizeof(handles[i].inner.xpu_ipc_mem_handle.opaque));
        hash ^= static_cast<uint64_t>(handles[i].size) + static_cast<uint64_t>(0x9e3779b97f4a7c15ull) +
                (hash << 6) + (hash >> 2);
        hash ^= static_cast<uint64_t>(i + 1) * static_cast<uint64_t>(0x9e3779b97f4a7c15ull);
        const uintptr_t token = static_cast<uintptr_t>((hash << 1) | 1ull);
        out_tokens[i] = reinterpret_cast<void*>(token);
    }
}
#endif

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               bool explicitly_destroy,
               bool enable_shrink,
               bool use_fabric)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      enable_shrink(enable_shrink),
      low_latency_mode(low_latency_mode),
      explicitly_destroy(explicitly_destroy),
      comm_stream(backend::get_stream_from_pool(true)),
      shared_memory_allocator(use_fabric) {
    // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);
    bool xpu_cross_process_local_only = false;
    auto local_num_tokens_per_rank = std::optional<torch::Tensor>();
    auto local_num_tokens_per_expert = std::optional<torch::Tensor>();
    auto local_is_token_in_rank = torch::Tensor();

#if defined(DEEPEP_USE_XPU)
    if (num_rdma_bytes > 0) {
        static std::once_flag warned_once;
        std::call_once(warned_once, [this, low_latency_mode]() {
            if (low_latency_mode) {
                printf("WARNING: XPU low-latency mode uses staged local-only buffers; RDMA/NVSHMEM transport is not implemented yet.\n");
            } else {
                printf("WARNING: XPU internode (RDMA/NVSHMEM) path is not implemented yet; num_rdma_bytes will be ignored.\n");
            }
            fflush(stdout);
        });
        if (not low_latency_mode) {
            this->num_rdma_bytes = 0;
            num_rdma_bytes = 0;
        }
    }
#endif

    // Common checks
    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);
    // Get ranks
    runtime_get_device(&device_id);
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
#ifdef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");
#endif

    // Get device info
    num_device_sms = runtime_get_num_sms(device_id);

    // Number of per-channel bytes cannot be large
    EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handles
        shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                                       num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);
        shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
        buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        // Set barrier signals
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu =
            reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        runtime_memset_async(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream);
    }

    // Create 32 MiB workspace
    runtime_malloc(&workspace, NUM_WORKSPACE_BYTES);
    runtime_memset_async(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream);

    // MoE counter
    runtime_malloc_host_mapped(reinterpret_cast<void**>(const_cast<int**>(&moe_recv_counter)),
                               reinterpret_cast<void**>(&moe_recv_counter_mapped),
                               sizeof(int64_t));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    runtime_malloc_host_mapped(reinterpret_cast<void**>(const_cast<int**>(&moe_recv_expert_counter)),
                               reinterpret_cast<void**>(&moe_recv_expert_counter_mapped),
                               sizeof(int) * NUM_MAX_LOCAL_EXPERTS);
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_bytes > 0) {
        runtime_malloc_host_mapped(reinterpret_cast<void**>(const_cast<int**>(&moe_recv_rdma_counter)),
                       reinterpret_cast<void**>(&moe_recv_rdma_counter_mapped),
                       sizeof(int));
        *moe_recv_rdma_counter = -1;
    }
}

Buffer::~Buffer() noexcept(false) {
    if (not explicitly_destroy) {
        if (not destroyed)
            destroy();
    } else if (not destroyed) {
        printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_internode_available() const {
#if defined(DEEPEP_USE_XPU)
    return false;
#else
    return is_available() and num_rdma_bytes > 0;
#endif
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? rdma_rank * num_nvl_ranks : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    const shared_memory::MemHandle& handle = ipc_handles[nvl_rank];
    return {reinterpret_cast<const char*>(&handle), sizeof(handle)};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
#if defined(DEEPEP_USE_XPU)
    return {};
#elif !defined(DISABLE_NVSHMEM)
    EP_HOST_ASSERT(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(backend::kDeviceType));
}

torch::Stream Buffer::get_comm_stream() const {
    return comm_stream;
}

void Buffer::destroy() {
    if (destroyed)
        return;

    // Synchronize
    runtime_device_synchronize();

    if (num_nvl_bytes > 0) {
        // Barrier
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);
        runtime_device_synchronize();

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++i)
                if (i != nvl_rank)
                    shared_memory_allocator.close_mem_handle(buffer_ptrs[i]);
        }

        // Free local buffer and error flag
        shared_memory_allocator.free(buffer_ptrs[nvl_rank]);
    }

    // Free NVSHMEM
#if defined(DEEPEP_USE_XPU)
    if (rdma_buffer_ptr != nullptr)
        runtime_free(rdma_buffer_ptr);
    if (mask_buffer_ptr != nullptr)
        runtime_free(mask_buffer_ptr);
    if (sync_buffer_ptr != nullptr)
        runtime_free(sync_buffer_ptr);
#elif !defined(DISABLE_NVSHMEM)
    if (is_available() and num_rdma_bytes > 0) {
        runtime_device_synchronize();
        internode::barrier();
        internode::free(rdma_buffer_ptr);
        if (enable_shrink) {
            internode::free(mask_buffer_ptr);
            internode::free(sync_buffer_ptr);
        }
        internode::finalize();
    }
#endif

    // Free workspace and MoE counter
    runtime_free(workspace);
    runtime_free_host(const_cast<int*>(moe_recv_counter));

    // Free chunked mode staffs
    runtime_free_host(const_cast<int*>(moe_recv_expert_counter));
    if (moe_recv_rdma_counter != nullptr)
        runtime_free_host(const_cast<int*>(moe_recv_rdma_counter));

    destroyed = true;
    available = false;
}

void Buffer::sync(const std::vector<int>& device_ids,
                  const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());
    has_foreign_ipc_peers = false;

    // Sync IPC handles
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(num_ranks == device_ids.size());
        EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == shared_memory::HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE);
                shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
#if defined(DEEPEP_USE_XPU)
                has_foreign_ipc_peers = has_foreign_ipc_peers or
                                        (not shared_memory_allocator.requires_live_validation(buffer_ptrs[i]));
#endif
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
            } else {
#if defined(DEEPEP_USE_XPU)
                // In staged XPU mode, allow a foreign self-rank payload during cross-process handle exchange.
                // Keep local buffer ownership for this rank and treat the mismatch as foreign-peer presence.
                if (std::memcmp(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE) != 0)
                    has_foreign_ipc_peers = true;
#else
                EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE) == 0);
#endif
            }
        }

        // Copy all buffer and barrier signal pointers to GPU
        runtime_memcpy_h2d(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS);
        runtime_memcpy_h2d(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS);
        runtime_device_synchronize();
    }

    // Sync NVSHMEM handles and allocate memory
#if defined(DEEPEP_USE_XPU)
    if (num_rdma_bytes > 0) {
        runtime_malloc(&rdma_buffer_ptr, num_rdma_bytes);
        runtime_memset(rdma_buffer_ptr, 0, num_rdma_bytes);

        if (enable_shrink) {
            int num_mask_buffer_bytes = num_ranks * sizeof(int);
            int num_sync_buffer_bytes = num_ranks * sizeof(int);
            runtime_malloc(reinterpret_cast<void**>(&mask_buffer_ptr), num_mask_buffer_bytes);
            runtime_malloc(reinterpret_cast<void**>(&sync_buffer_ptr), num_sync_buffer_bytes);
            runtime_memset(mask_buffer_ptr, 0, num_mask_buffer_bytes);
            runtime_memset(sync_buffer_ptr, 0, num_sync_buffer_bytes);
        }
    }
#elif !defined(DISABLE_NVSHMEM)
    if (num_rdma_bytes > 0) {
        // Initialize NVSHMEM
        EP_HOST_ASSERT(root_unique_id_opt.has_value());
        std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
        auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
        std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());
        auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
        auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
        EP_HOST_ASSERT(nvshmem_rank == internode::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks, low_latency_mode));
        internode::barrier();

        // Allocate
        rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

        // Clean buffer (mainly for low-latency mode)
        runtime_memset(rdma_buffer_ptr, 0, num_rdma_bytes);

        // Allocate and clean shrink buffer
        if (enable_shrink) {
            int num_mask_buffer_bytes = num_ranks * sizeof(int);
            int num_sync_buffer_bytes = num_ranks * sizeof(int);
            mask_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_mask_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            sync_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_sync_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            runtime_memset(mask_buffer_ptr, 0, num_mask_buffer_bytes);
            runtime_memset(sync_buffer_ptr, 0, num_sync_buffer_bytes);
        }

        // Barrier
        internode::barrier();
        runtime_device_synchronize();
    }
#endif

    // Ready to use
    available = true;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(
    const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = backend::get_current_stream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        backend::set_current_stream(comm_stream);
    }

    // Wait previous tasks to be finished.
#if defined(DEEPEP_USE_XPU)
    {
        pybind11::gil_scoped_release no_gil;
#endif
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }
#if defined(DEEPEP_USE_XPU)
    }
#endif

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(backend::kDeviceType));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(backend::kDeviceType));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(backend::kDeviceType));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(backend::kDeviceType));

    layout::get_dispatch_layout(topk_idx.data_ptr<topk_idx_t>(),
                                num_tokens_per_rank.data_ptr<int>(),
                                num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                num_tokens_per_expert.data_ptr<int>(),
                                is_token_in_rank.data_ptr<bool>(),
                                num_tokens,
                                num_topk,
                                num_ranks,
                                num_experts,
                                comm_stream);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {num_tokens_per_rdma_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        backend::set_current_stream(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::vector<int>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<EventHandle>>
Buffer::intranode_dispatch(const torch::Tensor& x,
                           const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx,
                           const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank,
                           const torch::Tensor& is_token_in_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                           int expert_alignment,
                           int num_worst_tokens,
                           const Config& config,
                           std::optional<EventHandle>& previous_event,
                           bool async,
                           bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();
    bool xpu_cross_process_local_only = false;
    auto local_num_tokens_per_rank = std::optional<torch::Tensor>();
    auto local_num_tokens_per_expert = std::optional<torch::Tensor>();
    auto local_is_token_in_rank = torch::Tensor();
    auto local_cached_rank_prefix_matrix = torch::Tensor();
    auto local_cached_channel_prefix_matrix = torch::Tensor();

#if defined(DEEPEP_USE_XPU)
    if (num_ranks > 1 and has_foreign_ipc_peers) {
        if (cached_mode) {
            auto host_is_token_in_rank = is_token_in_rank.to(torch::kCPU).contiguous();
            auto* host_is_token_in_rank_ptr = host_is_token_in_rank.data_ptr<bool>();
            const auto num_tokens = static_cast<int>(x.size(0));
            bool only_local_rank = true;
            for (int token = 0; token < num_tokens and only_local_rank; ++token) {
                for (int r = 0; r < num_ranks; ++r) {
                    const bool in_rank =
                        host_is_token_in_rank_ptr[static_cast<size_t>(token) * static_cast<size_t>(num_ranks) + static_cast<size_t>(r)];
                    if (r == rank)
                        only_local_rank = only_local_rank and in_rank;
                    else
                        only_local_rank = only_local_rank and (not in_rank);
                    if (not only_local_rank)
                        break;
                }
            }

            if (not only_local_rank) {
                EP_UNSUPPORTED_XPU("intranode dispatch cached multi-rank across processes via PCIe IPC transport is pending");
            }

            xpu_cross_process_local_only = true;
            local_is_token_in_rank = is_token_in_rank.select(1, rank).contiguous().view({x.size(0), 1});
            local_cached_rank_prefix_matrix =
                cached_rank_prefix_matrix.value().select(0, rank).select(0, rank).contiguous().view({static_cast<int64_t>(1), static_cast<int64_t>(1)});
            local_cached_channel_prefix_matrix =
                cached_channel_prefix_matrix.value().select(0, rank).contiguous().view({static_cast<int64_t>(1), cached_channel_prefix_matrix->size(1)});
        }

        if (not cached_mode) {
            EP_HOST_ASSERT(num_tokens_per_rank.has_value() and num_tokens_per_expert.has_value());
            auto host_is_token_in_rank = is_token_in_rank.to(torch::kCPU).contiguous();
            auto* host_is_token_in_rank_ptr = host_is_token_in_rank.data_ptr<bool>();
            const auto num_tokens = static_cast<int>(x.size(0));
            bool only_local_rank = true;
            for (int token = 0; token < num_tokens and only_local_rank; ++token) {
                for (int r = 0; r < num_ranks; ++r) {
                    const bool in_rank =
                        host_is_token_in_rank_ptr[static_cast<size_t>(token) * static_cast<size_t>(num_ranks) + static_cast<size_t>(r)];
                    if (r == rank)
                        only_local_rank = only_local_rank and in_rank;
                    else
                        only_local_rank = only_local_rank and (not in_rank);
                    if (not only_local_rank)
                        break;
                }
            }

            if (not only_local_rank) {
                EP_UNSUPPORTED_XPU("intranode dispatch multi-rank across processes via PCIe IPC transport is pending");
            }

            xpu_cross_process_local_only = true;
            auto host_num_tokens_per_rank = num_tokens_per_rank->to(torch::kCPU).contiguous();
            const int local_recv_tokens = host_num_tokens_per_rank.data_ptr<int>()[rank];
            local_num_tokens_per_rank = torch::tensor({local_recv_tokens}, num_tokens_per_rank->options());

            const int num_experts = static_cast<int>(num_tokens_per_expert->size(0));
            const int local_num_experts = num_experts / num_ranks;
            auto host_num_tokens_per_expert = num_tokens_per_expert->to(torch::kCPU).contiguous();
            auto* host_num_tokens_per_expert_ptr = host_num_tokens_per_expert.data_ptr<int>();
            std::vector<int> local_expert_counts(static_cast<size_t>(local_num_experts), 0);
            for (int i = 0; i < local_num_experts; ++i)
                local_expert_counts[static_cast<size_t>(i)] = host_num_tokens_per_expert_ptr[rank * local_num_experts + i];
            local_num_tokens_per_expert =
                torch::tensor(local_expert_counts, num_tokens_per_expert->options().device(torch::kCPU)).to(num_tokens_per_expert->device());

            local_is_token_in_rank = is_token_in_rank.select(1, rank).contiguous().view({x.size(0), 1});
        }
    }
#endif

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = backend::get_current_stream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        backend::set_current_stream(comm_stream);
    }

    // Wait previous tasks to be finished.
#if defined(DEEPEP_USE_XPU)
    {
        pybind11::gil_scoped_release no_gil;
#endif
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }
#if defined(DEEPEP_USE_XPU)
    }
#endif

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    const int dispatch_num_ranks = xpu_cross_process_local_only ? 1 : num_ranks;
    const int dispatch_rank = xpu_cross_process_local_only ? 0 : rank;
    int num_memset_int = num_channels * dispatch_num_ranks * 4;
    void** intranode_buffer_ptrs = buffer_ptrs_gpu;
    int** intranode_barrier_signal_ptrs = barrier_signal_ptrs_gpu;
#if defined(DEEPEP_USE_XPU)
    // Use IPC-handle-derived peer tokens for staged XPU rendezvous grouping.
    // This avoids dependence on process-local pointer identity.
    void* intranode_ipc_peer_tokens[NUM_MAX_NVL_PEERS] = {nullptr};
    build_xpu_ipc_peer_tokens(ipc_handles, num_ranks, intranode_ipc_peer_tokens);
    intranode_buffer_ptrs = intranode_ipc_peer_tokens;
    intranode_barrier_signal_ptrs = barrier_signal_ptrs;
#endif
#if defined(DEEPEP_USE_XPU)
    {
        pybind11::gil_scoped_release no_gil;
#endif
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        rank_prefix_matrix = xpu_cross_process_local_only ? local_cached_rank_prefix_matrix : cached_rank_prefix_matrix.value();
        channel_prefix_matrix = xpu_cross_process_local_only ? local_cached_channel_prefix_matrix : cached_channel_prefix_matrix.value();

        // Copy rank prefix matrix and clean flags
        intranode::cached_notify_dispatch(
            rank_prefix_matrix.data_ptr<int>(), num_memset_int, intranode_buffer_ptrs, intranode_barrier_signal_ptrs, dispatch_rank, dispatch_num_ranks,
            comm_stream);
    } else {
        rank_prefix_matrix = xpu_cross_process_local_only
                                 ? torch::zeros({num_ranks, num_ranks}, dtype(torch::kInt32).device(backend::kDeviceType))
                                 : torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(backend::kDeviceType));
        channel_prefix_matrix = xpu_cross_process_local_only
                                    ? torch::zeros({num_ranks, num_channels}, dtype(torch::kInt32).device(backend::kDeviceType))
                                    : torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(backend::kDeviceType));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        const int dispatch_num_experts = xpu_cross_process_local_only ? num_local_experts : num_experts;
        const int dispatch_num_local_experts = dispatch_num_experts / dispatch_num_ranks;
        EP_HOST_ASSERT(dispatch_num_ranks * (dispatch_num_ranks + dispatch_num_local_experts) * sizeof(int) <= num_nvl_bytes);
        intranode::notify_dispatch((xpu_cross_process_local_only ? local_num_tokens_per_rank.value() : num_tokens_per_rank.value()).data_ptr<int>(),
                                   moe_recv_counter_mapped,
                       dispatch_num_ranks,
                       (xpu_cross_process_local_only ? local_num_tokens_per_expert.value() : num_tokens_per_expert.value()).data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                       dispatch_num_experts,
                                   num_tokens,
                       (xpu_cross_process_local_only ? local_is_token_in_rank : is_token_in_rank).data_ptr<bool>(),
                                   channel_prefix_matrix.data_ptr<int>(),
                                   rank_prefix_matrix.data_ptr<int>(),
                                   num_memset_int,
                                   expert_alignment,
                                   intranode_buffer_ptrs,
                                   intranode_barrier_signal_ptrs,
                       dispatch_rank,
                                   comm_stream,
                                   num_channels);

        if (num_worst_tokens > 0) {
            // No CPU sync, just allocate the worst case
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        } else {
            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    NUM_CPU_TIMEOUT_SECS)
                    throw std::runtime_error("DeepEP error: CPU recv timeout");
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }
#if defined(DEEPEP_USE_XPU)
    }
#endif

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(backend::kDeviceType));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = xpu_cross_process_local_only
                                          ? torch::zeros({num_ranks, num_channels}, dtype(torch::kInt32).device(backend::kDeviceType))
                                          : torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(backend::kDeviceType));
    auto send_head = xpu_cross_process_local_only
                         ? torch::zeros({num_tokens, num_ranks}, dtype(torch::kInt32).device(backend::kDeviceType))
                         : torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(backend::kDeviceType));

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Dispatch
    EP_HOST_ASSERT(
        dispatch_num_ranks * dispatch_num_ranks * sizeof(int) +                                                                 // Size prefix matrix
            num_channels * dispatch_num_ranks * sizeof(int) +                                                                  // Channel start offset
            num_channels * dispatch_num_ranks * sizeof(int) +                                                                  // Channel end offset
            num_channels * dispatch_num_ranks * sizeof(int) * 2 +                                                              // Queue head and tail
            num_channels * dispatch_num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +     // Data buffer
            num_channels * dispatch_num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                        // Source index buffer
            num_channels * dispatch_num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +      // Top-k index buffer
            num_channels * dispatch_num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +           // Top-k weight buffer
            num_channels * dispatch_num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales           // FP8 scale buffer
        <= num_nvl_bytes);
#if defined(DEEPEP_USE_XPU)
    {
        pybind11::gil_scoped_release no_gil;
#endif
    intranode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_src_idx.data_ptr<int>(),
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        (xpu_cross_process_local_only ? local_is_token_in_rank : is_token_in_rank).data_ptr<bool>(),
                        channel_prefix_matrix.data_ptr<int>(),
                        num_tokens,
                        num_worst_tokens,
                        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                        num_topk,
                        (xpu_cross_process_local_only ? num_local_experts : num_experts),
                        num_scales,
                        scale_token_stride,
                        scale_hidden_stride,
                        intranode_buffer_ptrs,
                        dispatch_rank,
                        dispatch_num_ranks,
                        comm_stream,
                        config.num_sms,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens);
#if defined(DEEPEP_USE_XPU)
    }
#endif

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        is_token_in_rank,
                        rank_prefix_matrix,
                        channel_prefix_matrix,
                        recv_x,
                        recv_src_idx,
                        recv_channel_prefix_matrix,
                        send_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {x_scales,
                         topk_idx,
                         topk_weights,
                         num_tokens_per_rank,
                         num_tokens_per_expert,
                         cached_channel_prefix_matrix,
                         cached_rank_prefix_matrix,
                         recv_topk_idx,
                         recv_topk_weights,
                         recv_x_scales}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        backend::set_current_stream(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::intranode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& bias_0,
    const std::optional<torch::Tensor>& bias_1,
    const torch::Tensor& src_idx,
    const torch::Tensor& rank_prefix_matrix,
    const torch::Tensor& channel_prefix_matrix,
    const torch::Tensor& send_head,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                   rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                   channel_prefix_matrix.scalar_type() == torch::kInt32);

    int combine_num_ranks = num_ranks;
    int combine_rank = rank;
#if defined(DEEPEP_USE_XPU)
    if (num_ranks > 1 and has_foreign_ipc_peers) {
        auto host_channel_prefix_matrix = channel_prefix_matrix.to(torch::kCPU).contiguous();
        auto* host_channel_prefix_matrix_ptr = host_channel_prefix_matrix.data_ptr<int>();
        bool local_only = true;
        const int host_num_channels = static_cast<int>(channel_prefix_matrix.size(1));
        for (int r = 1; r < num_ranks and local_only; ++r) {
            for (int c = 0; c < host_num_channels; ++c) {
                const int value = host_channel_prefix_matrix_ptr[
                    static_cast<size_t>(r) * static_cast<size_t>(host_num_channels) + static_cast<size_t>(c)];
                local_only = local_only and (value == 0);
                if (not local_only)
                    break;
            }
        }

        if (local_only) {
            combine_num_ranks = 1;
            combine_rank = 0;
        } else {
            EP_UNSUPPORTED_XPU("intranode combine multi-rank across processes via PCIe IPC transport is pending");
        }
    }
#endif

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = backend::get_current_stream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        backend::set_current_stream(comm_stream);
    }

    // Wait previous tasks to be finished.
    // internode_dispatch already releases the GIL at entry on XPU.
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    void** intranode_buffer_ptrs = buffer_ptrs_gpu;
    int** intranode_barrier_signal_ptrs = barrier_signal_ptrs_gpu;
#if defined(DEEPEP_USE_XPU)
    // Use IPC-handle-derived peer tokens for staged XPU rendezvous grouping.
    // This avoids dependence on process-local pointer identity.
    void* intranode_ipc_peer_tokens[NUM_MAX_NVL_PEERS] = {nullptr};
    build_xpu_ipc_peer_tokens(ipc_handles, num_ranks, intranode_ipc_peer_tokens);
    intranode_buffer_ptrs = intranode_ipc_peer_tokens;
    intranode_barrier_signal_ptrs = barrier_signal_ptrs;
#endif

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * combine_num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
#if defined(DEEPEP_USE_XPU)
    {
        pybind11::gil_scoped_release no_gil;
#endif
    intranode::cached_notify_combine(intranode_buffer_ptrs,
                                     send_head.data_ptr<int>(),
                                     num_channels,
                                     num_recv_tokens,
                                     num_channels * combine_num_ranks * 2,
                                     intranode_barrier_signal_ptrs,
                                     combine_rank,
                                     combine_num_ranks,
                                     comm_stream);
#if defined(DEEPEP_USE_XPU)
    }
#endif

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    // Combine data
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    EP_HOST_ASSERT(num_channels * combine_num_ranks * sizeof(int) * 2 +  // Queue head and tail
                       num_channels * combine_num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * x.element_size() +  // Data buffer
                       num_channels * combine_num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +             // Source index buffer
                       num_channels * combine_num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)  // Top-k weight buffer
                   <= num_nvl_bytes);
 #if defined(DEEPEP_USE_XPU)
    {
        pybind11::gil_scoped_release no_gil;
#endif
    intranode::combine(get_runtime_dtype(x.scalar_type()),
                       recv_x.data_ptr(),
                       recv_topk_weights_ptr,
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       src_idx.data_ptr<int>(),
                       rank_prefix_matrix.data_ptr<int>(),
                       channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(),
                       num_tokens,
                       num_recv_tokens,
                       hidden,
                       num_topk,
                       intranode_buffer_ptrs,
                       combine_rank,
                       combine_num_ranks,
                       comm_stream,
                       config.num_sms,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens);
#if defined(DEEPEP_USE_XPU)
    }
#endif

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {topk_weights, recv_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        backend::set_current_stream(compute_stream);

    return {recv_x, recv_topk_weights, event};
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::vector<int>,
           torch::Tensor,
           torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<EventHandle>>
Buffer::internode_dispatch(const torch::Tensor& x,
                           const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx,
                           const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                           const torch::Tensor& is_token_in_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens,
                           int cached_num_rdma_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                           const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                           int expert_alignment,
                           int num_worst_tokens,
                           const Config& config,
                           std::optional<EventHandle>& previous_event,
                           bool async,
                           bool allocate_on_comm_stream) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_bytes > 0 and "internode_dispatch requires RDMA buffer bytes > 0");
                            bool xpu_cross_process_local_only = false;
                            auto local_num_tokens_per_rank = std::optional<torch::Tensor>();
                            auto local_num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
                            auto local_num_tokens_per_expert = std::optional<torch::Tensor>();
                            auto local_is_token_in_rank = torch::Tensor();

    // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
    // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
    // unless we release GIL here.
    pybind11::gil_scoped_release release;

    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

#if defined(DEEPEP_USE_XPU)
    if (num_ranks > 1 and has_foreign_ipc_peers) {
        if (cached_mode) {
            EP_HOST_ASSERT(num_rdma_ranks == 1 and "cross-process local-only internode dispatch currently requires a single RDMA rank");

            auto host_is_token_in_rank = is_token_in_rank.to(torch::kCPU).contiguous();
            auto* host_is_token_in_rank_ptr = host_is_token_in_rank.data_ptr<bool>();
            const auto num_tokens = static_cast<int>(x.size(0));
            bool only_local_rank = true;
            for (int token = 0; token < num_tokens and only_local_rank; ++token) {
                for (int r = 0; r < num_ranks; ++r) {
                    const bool in_rank =
                        host_is_token_in_rank_ptr[static_cast<size_t>(token) * static_cast<size_t>(num_ranks) + static_cast<size_t>(r)];
                    if (r == rank)
                        only_local_rank = only_local_rank and in_rank;
                    else
                        only_local_rank = only_local_rank and (not in_rank);
                    if (not only_local_rank)
                        break;
                }
            }

            if (not only_local_rank) {
                EP_UNSUPPORTED_XPU("internode dispatch cached multi-rank across processes via PCIe IPC transport is pending");
            }

            xpu_cross_process_local_only = true;
            local_is_token_in_rank = is_token_in_rank.select(1, rank).contiguous().view({x.size(0), 1});
        }

        if (not cached_mode) {
            EP_HOST_ASSERT(num_tokens_per_rank.has_value() and num_tokens_per_rdma_rank.has_value() and num_tokens_per_expert.has_value());
            EP_HOST_ASSERT(num_rdma_ranks == 1 and "cross-process local-only internode dispatch currently requires a single RDMA rank");

            auto host_is_token_in_rank = is_token_in_rank.to(torch::kCPU).contiguous();
            auto* host_is_token_in_rank_ptr = host_is_token_in_rank.data_ptr<bool>();
            const auto num_tokens = static_cast<int>(x.size(0));
            bool only_local_rank = true;
            for (int token = 0; token < num_tokens and only_local_rank; ++token) {
                for (int r = 0; r < num_ranks; ++r) {
                    const bool in_rank =
                        host_is_token_in_rank_ptr[static_cast<size_t>(token) * static_cast<size_t>(num_ranks) + static_cast<size_t>(r)];
                    if (r == rank)
                        only_local_rank = only_local_rank and in_rank;
                    else
                        only_local_rank = only_local_rank and (not in_rank);
                    if (not only_local_rank)
                        break;
                }
            }

            if (not only_local_rank) {
                EP_UNSUPPORTED_XPU("internode dispatch multi-rank across processes via PCIe IPC transport is pending");
            }

            xpu_cross_process_local_only = true;
            auto host_num_tokens_per_rank = num_tokens_per_rank->to(torch::kCPU).contiguous();
            const int local_recv_tokens = host_num_tokens_per_rank.data_ptr<int>()[rank];
            local_num_tokens_per_rank = torch::tensor({local_recv_tokens}, num_tokens_per_rank->options());

            auto host_num_tokens_per_rdma_rank = num_tokens_per_rdma_rank->to(torch::kCPU).contiguous();
            local_num_tokens_per_rdma_rank = torch::tensor({host_num_tokens_per_rdma_rank.data_ptr<int>()[0]}, num_tokens_per_rdma_rank->options());

            const int num_experts = static_cast<int>(num_tokens_per_expert->size(0));
            const int local_num_experts = num_experts / num_ranks;
            auto host_num_tokens_per_expert = num_tokens_per_expert->to(torch::kCPU).contiguous();
            auto* host_num_tokens_per_expert_ptr = host_num_tokens_per_expert.data_ptr<int>();
            std::vector<int> local_expert_counts(static_cast<size_t>(local_num_experts), 0);
            for (int i = 0; i < local_num_experts; ++i)
                local_expert_counts[static_cast<size_t>(i)] = host_num_tokens_per_expert_ptr[rank * local_num_experts + i];
            local_num_tokens_per_expert =
                torch::tensor(local_expert_counts, num_tokens_per_expert->options().device(torch::kCPU)).to(num_tokens_per_expert->device());

            local_is_token_in_rank = is_token_in_rank.select(1, rank).contiguous().view({x.size(0), 1});
        }
    }
#endif

    // Type checks
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                       cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                       cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;
        const int dispatch_num_ranks = xpu_cross_process_local_only ? 1 : num_ranks;
        const int dispatch_rank = xpu_cross_process_local_only ? 0 : rank;
        const int dispatch_num_experts = xpu_cross_process_local_only ? num_local_experts : num_experts;

    // Top-k checks
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = backend::get_current_stream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        backend::set_current_stream(comm_stream);
    }

    // Wait previous tasks to be finished.
    // NOTE: internode_dispatch already releases GIL at function entry.
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        if (xpu_cross_process_local_only) {
            gbl_channel_prefix_matrix =
                cached_gbl_channel_prefix_matrix.value().select(0, rank).contiguous().view({1, cached_gbl_channel_prefix_matrix->size(1)});
            recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value().select(0, rank).contiguous().view({1});
        } else {
            gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
            recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();
        }

        // Just a barrier and clean flags
        internode::cached_notify(hidden_int4,
                                 num_scales,
                                 num_topk,
                                 num_topk,
                                 num_ranks,
                                 num_channels,
                                 0,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 rdma_buffer_ptr,
                                 config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu,
                                 config.num_max_nvl_chunked_recv_tokens,
                                 barrier_signal_ptrs_gpu,
                                 rank,
                                 comm_stream,
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                 num_nvl_bytes,
                                 true,
                                 low_latency_mode);
    } else {
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(backend::kDeviceType));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(backend::kDeviceType));
        gbl_channel_prefix_matrix = xpu_cross_process_local_only
                                        ? torch::zeros({num_ranks, num_channels}, dtype(torch::kInt32).device(backend::kDeviceType))
                                        : torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(backend::kDeviceType));
        recv_gbl_rank_prefix_sum = xpu_cross_process_local_only
                                       ? torch::zeros({num_ranks}, dtype(torch::kInt32).device(backend::kDeviceType))
                                       : torch::empty({num_ranks}, dtype(torch::kInt32).device(backend::kDeviceType));

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch((xpu_cross_process_local_only ? local_num_tokens_per_rank.value() : num_tokens_per_rank.value()).data_ptr<int>(),
                                   moe_recv_counter_mapped,
                       dispatch_num_ranks,
                       (xpu_cross_process_local_only ? local_num_tokens_per_rdma_rank.value() : num_tokens_per_rdma_rank.value()).data_ptr<int>(),
                                   moe_recv_rdma_counter_mapped,
                       (xpu_cross_process_local_only ? local_num_tokens_per_expert.value() : num_tokens_per_expert.value()).data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                       dispatch_num_experts,
                       (xpu_cross_process_local_only ? local_is_token_in_rank : is_token_in_rank).data_ptr<bool>(),
                                   num_tokens,
                                   num_worst_tokens,
                                   num_channels,
                                   hidden_int4,
                                   num_scales,
                                   num_topk,
                                   expert_alignment,
                                   rdma_channel_prefix_matrix.data_ptr<int>(),
                                   recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                   gbl_channel_prefix_matrix.data_ptr<int>(),
                                   recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                   rdma_buffer_ptr,
                                   config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu,
                                   config.num_max_nvl_chunked_recv_tokens,
                                   barrier_signal_ptrs_gpu,
                                   dispatch_rank,
                                   comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), dispatch_num_ranks),
                                   num_nvl_bytes,
                                   low_latency_mode);

        // Synchronize total received tokens and tokens per expert
        if (num_worst_tokens > 0) {
            num_recv_tokens = num_worst_tokens;
            num_rdma_recv_tokens = num_worst_tokens;
        } else {
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);
                num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    NUM_CPU_TIMEOUT_SECS) {
                    printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
                    for (int i = 0; i < num_local_experts; ++i)
                        printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                    throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
                }
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_nvl_head = std::optional<torch::Tensor>();
    if (not cached_mode) {
        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(backend::kDeviceType));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(backend::kDeviceType));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(backend::kDeviceType));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(backend::kDeviceType));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(backend::kDeviceType));
    }

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    internode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        cached_mode ? nullptr : recv_src_meta->data_ptr(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
                        cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
                        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                        rdma_channel_prefix_matrix.data_ptr<int>(),
                        recv_rdma_rank_prefix_sum.data_ptr<int>(),
                        gbl_channel_prefix_matrix.data_ptr<int>(),
                        recv_gbl_rank_prefix_sum.data_ptr<int>(),
                        (xpu_cross_process_local_only ? local_is_token_in_rank : is_token_in_rank).data_ptr<bool>(),
                        num_tokens,
                        num_worst_tokens,
                        hidden_int4,
                        num_scales,
                        num_topk,
                        dispatch_num_experts,
                        scale_token_stride,
                        scale_hidden_stride,
                        rdma_buffer_ptr,
                        config.num_max_rdma_chunked_send_tokens,
                        config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens,
                        dispatch_rank,
                        dispatch_num_ranks,
                        cached_mode,
                        comm_stream,
                        num_channels,
                        low_latency_mode);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        is_token_in_rank,
                        recv_x,
                        rdma_channel_prefix_matrix,
                        recv_rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {x_scales,
                         topk_idx,
                         topk_weights,
                         num_tokens_per_rank,
                         num_tokens_per_rdma_rank,
                         num_tokens_per_expert,
                         cached_rdma_channel_prefix_matrix,
                         cached_recv_rdma_rank_prefix_sum,
                         cached_gbl_channel_prefix_matrix,
                         cached_recv_gbl_rank_prefix_sum,
                         recv_topk_idx,
                         recv_topk_weights,
                         recv_x_scales,
                         recv_rdma_channel_prefix_matrix,
                         recv_gbl_channel_prefix_matrix,
                         send_rdma_head,
                         send_nvl_head,
                         recv_src_meta}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        backend::set_current_stream(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
            event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::internode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& bias_0,
    const std::optional<torch::Tensor>& bias_1,
    const torch::Tensor& src_meta,
    const torch::Tensor& is_combined_token_in_rank,
    const torch::Tensor& rdma_channel_prefix_matrix,
    const torch::Tensor& rdma_rank_prefix_sum,
    const torch::Tensor& gbl_channel_prefix_matrix,
    const torch::Tensor& combined_rdma_head,
    const torch::Tensor& combined_nvl_head,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_bytes > 0 and "internode_combine requires RDMA buffer bytes > 0");
        int combine_num_ranks = num_ranks;
        int combine_rank = rank;
    bool xpu_cross_process_local_only = false;
        auto local_is_combined_token_in_rank = torch::Tensor();
#if defined(DEEPEP_USE_XPU)
    if (num_ranks > 1 and has_foreign_ipc_peers) {
            EP_HOST_ASSERT(num_rdma_ranks == 1 and "cross-process local-only internode combine currently requires a single RDMA rank");
            bool local_only = false;
            const int host_num_channels = static_cast<int>(gbl_channel_prefix_matrix.size(1));
            if (gbl_channel_prefix_matrix.size(0) == 1) {
                // Cached local-only dispatch may already provide a collapsed single-rank prefix view.
                local_only = true;
            } else {
                EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and
                               "cross-process internode combine expects full-rank or collapsed local-only global prefixes");
                auto host_gbl_channel_prefix_matrix = gbl_channel_prefix_matrix.to(torch::kCPU).contiguous();
                auto* host_gbl_channel_prefix_matrix_ptr = host_gbl_channel_prefix_matrix.data_ptr<int>();
                local_only = true;
                for (int r = 1; r < num_ranks and local_only; ++r) {
                    for (int c = 0; c < host_num_channels; ++c) {
                        const int value = host_gbl_channel_prefix_matrix_ptr[
                            static_cast<size_t>(r) * static_cast<size_t>(host_num_channels) + static_cast<size_t>(c)];
                        local_only = local_only and (value == 0);
                        if (not local_only)
                            break;
                    }
                }
            }

            if (local_only) {
                xpu_cross_process_local_only = true;
                combine_num_ranks = 1;
                combine_rank = 0;
                local_is_combined_token_in_rank = is_combined_token_in_rank.select(1, rank).contiguous().view({is_combined_token_in_rank.size(0), 1});
            } else {
                EP_UNSUPPORTED_XPU("internode combine multi-rank across processes via PCIe IPC transport is pending");
            }
    }
#endif

    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and
                   is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and
                   rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                   rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and
                   gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                   combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    if (xpu_cross_process_local_only) {
        EP_HOST_ASSERT((gbl_channel_prefix_matrix.size(0) == combine_num_ranks or gbl_channel_prefix_matrix.size(0) == num_ranks) and
                       gbl_channel_prefix_matrix.size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == combine_num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    }
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                   combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = backend::get_current_stream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        backend::set_current_stream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Top-k checks
    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());

    // Launch barrier and reset queue head and tail, then combine.
#if defined(DEEPEP_USE_XPU)
    {
        pybind11::gil_scoped_release no_gil;
#endif
        internode::cached_notify(hidden_int4,
                                 0,
                                 0,
                                 num_topk,
                                 combine_num_ranks,
                                 num_channels,
                                 num_combined_tokens,
                                 combined_rdma_head.data_ptr<int>(),
                                 rdma_channel_prefix_matrix.data_ptr<int>(),
                                 rdma_rank_prefix_sum.data_ptr<int>(),
                                 combined_nvl_head.data_ptr<int>(),
                                 rdma_buffer_ptr,
                                 config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu,
                                 config.num_max_nvl_chunked_recv_tokens,
                                 barrier_signal_ptrs_gpu,
                                   combine_rank,
                                 comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), combine_num_ranks),
                                 num_nvl_bytes,
                                 false,
                                 low_latency_mode);

        internode::combine(get_runtime_dtype(x.scalar_type()),
                           combined_x.data_ptr(),
                           combined_topk_weights_ptr,
                           (xpu_cross_process_local_only ? local_is_combined_token_in_rank : is_combined_token_in_rank).data_ptr<bool>(),
                           x.data_ptr(),
                           topk_weights_ptr,
                           bias_ptrs[0],
                           bias_ptrs[1],
                           combined_rdma_head.data_ptr<int>(),
                           combined_nvl_head.data_ptr<int>(),
                           src_meta.data_ptr(),
                           rdma_channel_prefix_matrix.data_ptr<int>(),
                           rdma_rank_prefix_sum.data_ptr<int>(),
                           gbl_channel_prefix_matrix.data_ptr<int>(),
                           num_tokens,
                           num_combined_tokens,
                           hidden,
                           num_topk,
                           rdma_buffer_ptr,
                           config.num_max_rdma_chunked_send_tokens,
                           config.num_max_rdma_chunked_recv_tokens,
                           buffer_ptrs_gpu,
                           config.num_max_nvl_chunked_send_tokens,
                           config.num_max_nvl_chunked_recv_tokens,
                           combine_rank,
                           combine_num_ranks,
                           comm_stream,
                           num_channels,
                           low_latency_mode);
#if defined(DEEPEP_USE_XPU)
    }
#endif

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        src_meta,
                        is_combined_token_in_rank,
                        rdma_channel_prefix_matrix,
                        rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        combined_x,
                        combined_rdma_head,
                        combined_nvl_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {topk_weights, combined_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        backend::set_current_stream(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
    check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

    internode_ll::clean_low_latency_buffer(clean_meta_0.first,
                                           clean_meta_0.second,
                                           clean_meta_1.first,
                                           clean_meta_1.second,
                                           rank,
                                           num_ranks,
                                           mask_buffer_ptr,
                                           sync_buffer_ptr,
                                           backend::get_current_stream());
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor& x,
                             const torch::Tensor& topk_idx,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                             int num_max_dispatch_tokens_per_rank,
                             int num_experts,
                             bool use_fp8,
                             bool round_scale,
                             bool use_ue8m0,
                             bool async,
                             bool return_recv_hook) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_bytes > 0 and "low_latency_dispatch requires RDMA buffer bytes > 0");
    int dispatch_num_ranks = num_ranks;
    int dispatch_rank = rank;
    int dispatch_num_experts = num_experts;
    auto dispatch_topk_idx = topk_idx;
#if defined(DEEPEP_USE_XPU)
    if (num_ranks > 1 and has_foreign_ipc_peers) {
        EP_HOST_ASSERT(num_experts % num_ranks == 0);
        const int num_local_experts = num_experts / num_ranks;
        auto host_topk_idx = topk_idx.to(torch::kCPU).contiguous();
        auto* host_topk_idx_ptr = host_topk_idx.data_ptr<topk_idx_t>();
        bool local_only = true;
        const int lower = rank * num_local_experts;
        const int upper = (rank + 1) * num_local_experts;
        const int num_tokens = static_cast<int>(topk_idx.size(0));
        const int num_topk = static_cast<int>(topk_idx.size(1));
        for (int token = 0; token < num_tokens and local_only; ++token) {
            for (int k = 0; k < num_topk; ++k) {
                const int expert = static_cast<int>(host_topk_idx_ptr[token * num_topk + k]);
                local_only = (expert >= lower and expert < upper);
                if (not local_only)
                    break;
            }
        }

        if (local_only) {
            auto local_host_topk_idx = host_topk_idx.clone();
            auto* local_host_topk_idx_ptr = local_host_topk_idx.data_ptr<topk_idx_t>();
            for (int token = 0; token < num_tokens; ++token) {
                for (int k = 0; k < num_topk; ++k) {
                    const int idx = token * num_topk + k;
                    local_host_topk_idx_ptr[idx] = static_cast<topk_idx_t>(static_cast<int>(local_host_topk_idx_ptr[idx]) - lower);
                }
            }
            dispatch_topk_idx = local_host_topk_idx.to(topk_idx.device(), topk_idx.scalar_type(), false, true);
            dispatch_num_ranks = 1;
            dispatch_rank = 0;
            dispatch_num_experts = num_local_experts;
        } else {
            EP_UNSUPPORTED_XPU("low-latency dispatch multi-rank across processes via PCIe IPC transport is pending");
        }
    }
#endif

    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    // Diagnosis tensors
    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks;

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = backend::get_current_stream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));

    // Allocate packed tensors
    // Staged XPU low-latency path keeps BF16 payload even when use_fp8=True,
    // mirroring the Python fallback contract while still returning scale tensors.
#if defined(DEEPEP_USE_XPU)
    auto packed_recv_x = torch::empty({num_local_experts, dispatch_num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                      x.options().dtype(torch::kBFloat16));
#else
    auto packed_recv_x = torch::empty({num_local_experts, dispatch_num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
#endif
    auto packed_recv_src_info =
        torch::empty({num_local_experts, dispatch_num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(backend::kDeviceType));
    auto packed_recv_layout_range = torch::empty({num_local_experts, dispatch_num_ranks}, torch::dtype(torch::kInt64).device(backend::kDeviceType));
    auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(backend::kDeviceType));

    // Allocate column-majored scales
    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((dispatch_num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, dispatch_num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kFloat32).device(backend::kDeviceType));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, dispatch_num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kInt).device(backend::kDeviceType));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::dispatch(
            packed_recv_x.data_ptr(),
            packed_recv_x_scales_ptr,
            packed_recv_src_info.data_ptr<int>(),
            packed_recv_layout_range.data_ptr<int64_t>(),
            packed_recv_count.data_ptr<int>(),
            mask_buffer_ptr,
            cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
            dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
            buffer.dispatch_rdma_recv_data_buffer,
            buffer.dispatch_rdma_recv_count_buffer,
            buffer.dispatch_rdma_send_buffer,
            x.data_ptr(),
            dispatch_topk_idx.data_ptr<topk_idx_t>(),
            next_clean_meta.first,
            next_clean_meta.second,
            num_tokens,
            hidden,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            dispatch_num_experts,
            dispatch_rank,
            dispatch_num_ranks,
            use_fp8,
            round_scale,
            use_ue8m0,
            workspace,
            num_device_sms,
            launch_stream,
            phases);
    };

#if defined(DEEPEP_USE_XPU)
    {
        pybind11::gil_scoped_release no_gil;
        if (not return_recv_hook)
            stream_wait(launch_stream, compute_stream);
        launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));
    }
#else
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));
#endif

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
#if defined(DEEPEP_USE_XPU)
        pybind11::gil_scoped_release no_gil;
#endif
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() {
#if defined(DEEPEP_USE_XPU)
            pybind11::gil_scoped_release no_gil;
#endif
            launcher(LOW_LATENCY_RECV_PHASE);
        };

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& src_info,
    const torch::Tensor& layout_range,
    const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
    int num_max_dispatch_tokens_per_rank,
    int num_experts,
    bool use_logfmt,
    bool zero_copy,
    bool async,
    bool return_recv_hook,
    const std::optional<torch::Tensor>& out) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_bytes > 0 and "low_latency_combine requires RDMA buffer bytes > 0");
    int combine_num_ranks = num_ranks;
    int combine_rank = rank;
    int combine_num_experts = num_experts;
    auto combine_x = x;
    auto combine_topk_idx = topk_idx;
    auto combine_src_info = src_info;
    auto combine_layout_range = layout_range;
#if defined(DEEPEP_USE_XPU)
    if (num_ranks > 1 and has_foreign_ipc_peers) {
        EP_HOST_ASSERT(num_experts % num_ranks == 0);
        const int num_local_experts = num_experts / num_ranks;
        auto host_topk_idx = topk_idx.to(torch::kCPU).contiguous();
        auto* host_topk_idx_ptr = host_topk_idx.data_ptr<topk_idx_t>();
        bool local_only = true;
        const int lower = rank * num_local_experts;
        const int upper = (rank + 1) * num_local_experts;
        const int num_tokens = static_cast<int>(topk_idx.size(0));
        const int num_topk = static_cast<int>(topk_idx.size(1));
        for (int token = 0; token < num_tokens and local_only; ++token) {
            for (int k = 0; k < num_topk; ++k) {
                const int expert = static_cast<int>(host_topk_idx_ptr[token * num_topk + k]);
                local_only = (expert >= lower and expert < upper);
                if (not local_only)
                    break;
            }
        }

        if (local_only) {
            auto local_host_topk_idx = host_topk_idx.clone();
            auto* local_host_topk_idx_ptr = local_host_topk_idx.data_ptr<topk_idx_t>();
            for (int token = 0; token < num_tokens; ++token) {
                for (int k = 0; k < num_topk; ++k) {
                    const int idx = token * num_topk + k;
                    local_host_topk_idx_ptr[idx] = static_cast<topk_idx_t>(static_cast<int>(local_host_topk_idx_ptr[idx]) - lower);
                }
            }

            combine_topk_idx = local_host_topk_idx.to(topk_idx.device(), topk_idx.scalar_type(), false, true);
            combine_num_ranks = 1;
            combine_rank = 0;
            combine_num_experts = num_local_experts;

            if (combine_x.size(1) == static_cast<int64_t>(num_ranks * num_max_dispatch_tokens_per_rank)) {
                combine_x = combine_x.slice(1,
                                            static_cast<int64_t>(rank * num_max_dispatch_tokens_per_rank),
                                            static_cast<int64_t>((rank + 1) * num_max_dispatch_tokens_per_rank))
                                .contiguous();
            }

            if (combine_src_info.size(1) == static_cast<int64_t>(num_ranks * num_max_dispatch_tokens_per_rank)) {
                combine_src_info = combine_src_info.slice(1,
                                                          static_cast<int64_t>(rank * num_max_dispatch_tokens_per_rank),
                                                          static_cast<int64_t>((rank + 1) * num_max_dispatch_tokens_per_rank))
                                      .contiguous();
            }

            if (combine_layout_range.size(1) == num_ranks) {
                combine_layout_range =
                    combine_layout_range.select(1, rank).contiguous().view({combine_layout_range.size(0), static_cast<int64_t>(1)});
            }
        } else {
            EP_UNSUPPORTED_XPU("low-latency combine multi-rank across processes via PCIe IPC transport is pending");
        }
    }
#endif

    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    EP_HOST_ASSERT(combine_x.dim() == 3 and combine_x.is_contiguous() and combine_x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(combine_x.size(0) == combine_num_experts / combine_num_ranks);
    EP_HOST_ASSERT(combine_x.size(1) == combine_num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(combine_x.size(2) % sizeof(int4) == 0 and combine_x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(combine_src_info.dim() == 2 and combine_src_info.is_contiguous());
    EP_HOST_ASSERT(combine_src_info.scalar_type() == torch::kInt32 and combine_x.size(0) == combine_src_info.size(0));
    EP_HOST_ASSERT(combine_layout_range.dim() == 2 and combine_layout_range.is_contiguous());
    EP_HOST_ASSERT(combine_layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(combine_layout_range.size(0) == combine_num_experts / combine_num_ranks and combine_layout_range.size(1) == combine_num_ranks);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == combine_num_ranks);
    }

    auto hidden = static_cast<int>(combine_x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, combine_num_ranks, combine_num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = backend::get_current_stream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == combine_x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, combine_x.options());
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::combine(combined_x.data_ptr(),
                              buffer.combine_rdma_recv_data_buffer,
                              buffer.combine_rdma_recv_flag_buffer,
                              buffer.combine_rdma_send_buffer,
                              combine_x.data_ptr(),
                              combine_topk_idx.data_ptr<topk_idx_t>(),
                              topk_weights.data_ptr<float>(),
                              combine_src_info.data_ptr<int>(),
                              combine_layout_range.data_ptr<int64_t>(),
                              mask_buffer_ptr,
                              combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                              next_clean_meta.first,
                              next_clean_meta.second,
                              num_combined_tokens,
                              hidden,
                              num_max_dispatch_tokens_per_rank,
                              num_topk,
                              combine_num_experts,
                              combine_rank,
                              combine_num_ranks,
                              use_logfmt,
                              workspace,
                              num_device_sms,
                              launch_stream,
                              phases,
                              zero_copy);
    };

#if defined(DEEPEP_USE_XPU)
    {
        pybind11::gil_scoped_release no_gil;
        if (not return_recv_hook)
            stream_wait(launch_stream, compute_stream);
        launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));
    }
#else
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));
#endif

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
#if defined(DEEPEP_USE_XPU)
        pybind11::gil_scoped_release no_gil;
#endif
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() {
#if defined(DEEPEP_USE_XPU)
            pybind11::gil_scoped_release no_gil;
#endif
            launcher(LOW_LATENCY_RECV_PHASE);
        };

    // Return values
    return {combined_x, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
#ifndef DISABLE_NVSHMEM
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                            {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(backend::kDeviceType));
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);
    internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, backend::get_current_stream());
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);

    internode_ll::query_mask_buffer(
        mask_buffer_ptr, num_ranks, reinterpret_cast<int*>(mask_status.data_ptr()), backend::get_current_stream());
}

void Buffer::low_latency_clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, backend::get_current_stream());
}

}  // namespace deep_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6,
             py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6,
             py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("is_internode_available", &deep_ep::Buffer::is_internode_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("low_latency_update_mask_buffer", &deep_ep::Buffer::low_latency_update_mask_buffer)
        .def("low_latency_query_mask_buffer", &deep_ep::Buffer::low_latency_query_mask_buffer)
        .def("low_latency_clean_mask_buffer", &deep_ep::Buffer::low_latency_clean_mask_buffer)
        .def("get_next_low_latency_combine_buffer", &deep_ep::Buffer::get_next_low_latency_combine_buffer);

    m.def("is_sm90_compiled", deep_ep::is_sm90_compiled);
    auto torch_mod = py::module::import("torch");
    m.attr("topk_idx_t") = (sizeof(deep_ep::topk_idx_t) == 4) ? torch_mod.attr("int32") : torch_mod.attr("int64");
}
