#include "xpu_native_runtime.hpp"

#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <torch/python.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

#include "config.hpp"
#include "kernels/api.cuh"
#include "kernels/transport.hpp"

#if defined(DEEPEP_USE_ISHMEM)
#include <ishmemx.h>
#endif

namespace {

struct XpuIpcMemHandle {
    int pid = 0;
    ze_ipc_mem_handle_t handle = {};
};

void ze_check(ze_result_t result, const char* expr, const char* file, int line) {
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("Level Zero call failed: ") + expr + " at " + file + ":" + std::to_string(line) +
                                 " result=" + std::to_string(static_cast<int>(result)));
    }
}

#define ZE_CHECK(expr) ze_check((expr), #expr, __FILE__, __LINE__)

void ensure_level_zero_initialized() {
    static const bool initialized = [] {
        ZE_CHECK(zeInit(0));
        return true;
    }();
    (void)initialized;
}

sycl::queue make_init_queue_for_device(int device_index) {
    ensure_level_zero_initialized();
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    EP_HOST_ASSERT(device_index >= 0 && device_index < static_cast<int>(devices.size()));
    auto& device = devices[device_index];
    EP_HOST_ASSERT(device.get_backend() == sycl::backend::ext_oneapi_level_zero);
    return sycl::queue(device);
}

void zero_symmetric_buffer(void* ptr, size_t num_bytes, int device_index) {
    if (ptr == nullptr || num_bytes == 0) {
        return;
    }
    auto queue = make_init_queue_for_device(device_index);
    queue.memset(ptr, 0, num_bytes);
    queue.wait_and_throw();
}

int duplicate_peer_fd(int pid, const ze_ipc_mem_handle_t& handle) {
    if (pid == static_cast<int>(getpid())) {
        return *reinterpret_cast<const int*>(handle.data);
    }

#if defined(__linux__)
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif
    int pidfd = static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
    if (pidfd < 0) {
        throw std::runtime_error("Failed to open peer pidfd for Level Zero IPC handle: errno=" + std::to_string(errno));
    }
    int peer_fd = *reinterpret_cast<const int*>(handle.data);
    int fd = static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, peer_fd, 0));
    close(pidfd);
    if (fd < 0) {
        throw std::runtime_error("Failed to duplicate peer Level Zero IPC file descriptor: errno=" + std::to_string(errno));
    }
    return fd;
#else
    throw std::runtime_error("Level Zero IPC handle import requires Linux pidfd support");
#endif
}

void* allocate_device_memory(ze_context_handle_t context, ze_device_handle_t device, size_t num_bytes, size_t alignment) {
    ensure_level_zero_initialized();
    ze_device_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    desc.ordinal = 0;
    void* ptr = nullptr;
    ZE_CHECK(zeMemAllocDevice(context, &desc, num_bytes, alignment, device, &ptr));
    return ptr;
}

int tensor_first_int_cpu(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.to(torch::kCPU);
    return cpu_tensor.data_ptr<int>()[0];
}

std::vector<int> tensor_to_int_vector_cpu(const torch::Tensor& tensor, int count) {
    auto cpu_tensor = tensor.to(torch::kCPU);
    auto* ptr = cpu_tensor.data_ptr<int>();
    return {ptr, ptr + count};
}

void compute_dispatch_layout_cpu(const torch::Tensor& topk_idx,
                                 int num_experts,
                                 int num_ranks,
                                 int num_rdma_ranks,
                                 int* num_tokens_per_rank,
                                 int* num_tokens_per_rdma_rank,
                                 int* num_tokens_per_expert,
                                 bool* is_token_in_rank) {
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    const int num_tokens = static_cast<int>(topk_idx.size(0));
    const int num_topk = static_cast<int>(topk_idx.size(1));
    const int num_experts_per_rank = num_experts / num_ranks;
    const auto topk_idx_cpu = topk_idx.to(torch::kCPU);
    const auto* topk_idx_ptr = topk_idx_cpu.data_ptr<deep_ep::topk_idx_t>();

    std::vector<int> tokens_per_rank(num_ranks, 0);
    std::vector<int> tokens_per_rdma_rank(num_rdma_ranks, 0);
    std::vector<int> tokens_per_expert(num_experts, 0);
    std::vector<uint8_t> token_in_rank(static_cast<size_t>(num_tokens) * num_ranks, 0);

    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        std::vector<uint8_t> rank_seen(num_ranks, 0);
        std::vector<uint8_t> rdma_rank_seen(num_rdma_ranks, 0);
        for (int topk = 0; topk < num_topk; ++topk) {
            const int expert_idx = static_cast<int>(topk_idx_ptr[token_idx * num_topk + topk]);
            if (expert_idx < 0 || expert_idx >= num_experts) {
                continue;
            }
            ++tokens_per_expert[expert_idx];
            const int rank_idx = expert_idx / num_experts_per_rank;
            rank_seen[rank_idx] = 1;
            if (num_rdma_ranks > 1) {
                rdma_rank_seen[rank_idx / NUM_MAX_NVL_PEERS] = 1;
            }
        }
        for (int rank_idx = 0; rank_idx < num_ranks; ++rank_idx) {
            token_in_rank[static_cast<size_t>(token_idx) * num_ranks + rank_idx] = rank_seen[rank_idx];
            tokens_per_rank[rank_idx] += rank_seen[rank_idx];
        }
        for (int rdma_rank_idx = 0; rdma_rank_idx < num_rdma_ranks; ++rdma_rank_idx) {
            tokens_per_rdma_rank[rdma_rank_idx] += rdma_rank_seen[rdma_rank_idx];
        }
    }

    std::memcpy(num_tokens_per_rank, tokens_per_rank.data(), tokens_per_rank.size() * sizeof(int));
    if (num_tokens_per_rdma_rank != nullptr) {
        std::memcpy(num_tokens_per_rdma_rank, tokens_per_rdma_rank.data(), tokens_per_rdma_rank.size() * sizeof(int));
    }
    std::memcpy(num_tokens_per_expert, tokens_per_expert.data(), tokens_per_expert.size() * sizeof(int));
    std::memcpy(is_token_in_rank, token_in_rank.data(), token_in_rank.size() * sizeof(uint8_t));
}

size_t native_intranode_staging_bytes(int num_ranks, int num_slots, int hidden_int4, int num_topk, int num_scales) {
    size_t cursor = static_cast<size_t>(num_ranks) * num_ranks * sizeof(int);
    cursor = deep_ep::align_up(cursor, alignof(int4));
    cursor += static_cast<size_t>(num_slots) * sizeof(int);
    cursor = deep_ep::align_up(cursor, alignof(deep_ep::topk_idx_t));
    cursor += static_cast<size_t>(num_slots) * std::max(num_topk, 1) * sizeof(deep_ep::topk_idx_t);
    cursor = deep_ep::align_up(cursor, alignof(float));
    cursor += static_cast<size_t>(num_slots) * std::max(num_topk, 1) * sizeof(float);
    cursor = deep_ep::align_up(cursor, alignof(float));
    cursor += static_cast<size_t>(num_slots) * std::max(num_scales, 1) * sizeof(float);
    cursor = deep_ep::align_up(cursor, alignof(int4));
    cursor += static_cast<size_t>(num_slots) * hidden_int4 * sizeof(int4);
    return cursor;
}

bool experimental_native_intranode_enabled() {
    const char* value = std::getenv("DEEPEP_XPU_ENABLE_EXPERIMENTAL_INTRANODE");
    return value != nullptr && std::string(value) == "1";
}

void debug_intranode_log(int rank, const char* message) {
    if (std::getenv("DEEPEP_XPU_DEBUG_INTRANODE") != nullptr) {
        fprintf(stderr, "[deepep:xpu:rank%d] %s\n", rank, message);
        fflush(stderr);
    }
}

}  // namespace

namespace deep_ep {

namespace internode {

#if defined(DEEPEP_USE_ISHMEM)
namespace {
ishmem_team_t cpu_rdma_team = ISHMEM_TEAM_INVALID;
ishmem_team_config_t cpu_rdma_team_config;
}
#endif

std::vector<uint8_t> get_unique_id() {
#if defined(DEEPEP_USE_ISHMEM)
    ishmemx_uniqueid_t unique_id;
    EP_HOST_ASSERT(ishmemx_get_uniqueid(&unique_id) == 0);
    std::vector<uint8_t> result(sizeof(ishmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(ishmemx_uniqueid_t));
    return result;
#else
    EP_HOST_ASSERT(false and "iSHMEM transport is disabled during compilation");
    return {};
#endif
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
#if defined(DEEPEP_USE_ISHMEM)
    ishmemx_uniqueid_t root_unique_id;
    ishmemx_attr_t attr{};
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(ishmemx_uniqueid_t));
    attr.runtime = ISHMEMX_RUNTIME_MPI;
    attr.initialize_runtime = true;
    attr.gpu = true;
    attr.device_idx = c10::xpu::current_device();
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
#else
    EP_HOST_ASSERT(false and "iSHMEM transport is disabled during compilation");
    return -1;
#endif
}

void* alloc(size_t size, size_t alignment) {
#if defined(DEEPEP_USE_ISHMEM)
    return ishmem_align(alignment, size);
#else
    EP_HOST_ASSERT(false and "iSHMEM transport is disabled during compilation");
    return nullptr;
#endif
}

void free(void* ptr) {
#if defined(DEEPEP_USE_ISHMEM)
    ishmem_free(ptr);
#else
    EP_HOST_ASSERT(false and "iSHMEM transport is disabled during compilation");
#endif
}

void barrier() {
#if defined(DEEPEP_USE_ISHMEM)
    ishmem_barrier_all();
#else
    EP_HOST_ASSERT(false and "iSHMEM transport is disabled during compilation");
#endif
}

void finalize() {
#if defined(DEEPEP_USE_ISHMEM)
    if (cpu_rdma_team != ISHMEM_TEAM_INVALID) {
        ishmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = ISHMEM_TEAM_INVALID;
    }
    ishmem_finalize();
#else
    EP_HOST_ASSERT(false and "iSHMEM transport is disabled during compilation");
#endif
}

}  // namespace internode

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               bool explicitly_destroy,
               bool enable_shrink,
               bool use_fabric)
    : low_latency_mode(low_latency_mode),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      enable_shrink(enable_shrink),
      rank(rank),
      num_ranks(num_ranks),
      explicitly_destroy(explicitly_destroy) {
    (void)use_fabric;

    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0);
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0);
    EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(0 <= rank && rank < num_ranks);
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS || num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0) {
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS || low_latency_mode);
    }

    device_id = c10::xpu::current_device();
    rdma_rank = rank / NUM_MAX_NVL_PEERS;
    nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
    num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    auto queue = make_init_queue_for_device(device_id);
    num_device_sms = static_cast<int>(queue.get_device().get_info<sycl::info::device::max_compute_units>());
    EP_HOST_ASSERT(num_device_sms > 0);
    comm_stream = get_stream_from_pool(true);
    auto& comm_queue = c10::xpu::XPUStream(comm_stream.value()).queue();
    EP_HOST_ASSERT(comm_queue.get_backend() == sycl::backend::ext_oneapi_level_zero);
    ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(comm_queue.get_context());
    ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(comm_queue.get_device());

    EP_HOST_ASSERT(ceil_div<int64_t>(std::max<int64_t>(num_nvl_bytes, 1), std::max(1, num_device_sms / 2)) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(std::max<int64_t>(num_rdma_bytes, 1), std::max(1, num_device_sms / 2)) < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        auto barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
        nvl_allocation_bytes = static_cast<size_t>(num_nvl_bytes) + barrier_signal_bytes;
        buffer_ptrs[nvl_rank] = allocate_device_memory(ze_context, ze_device, nvl_allocation_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        comm_queue.memset(buffer_ptrs[nvl_rank], 0, nvl_allocation_bytes);

        buffer_ptrs_gpu = static_cast<void**>(sycl::malloc_device(sizeof(void*) * NUM_MAX_NVL_PEERS, comm_queue));
        barrier_signal_ptrs_gpu = static_cast<int**>(sycl::malloc_device(sizeof(int*) * NUM_MAX_NVL_PEERS, comm_queue));
        EP_HOST_ASSERT(buffer_ptrs_gpu != nullptr && barrier_signal_ptrs_gpu != nullptr);
        comm_queue.memset(buffer_ptrs_gpu, 0, sizeof(void*) * NUM_MAX_NVL_PEERS);
        comm_queue.memset(barrier_signal_ptrs_gpu, 0, sizeof(int*) * NUM_MAX_NVL_PEERS);
        comm_queue.wait_and_throw();

        ZE_CHECK(zeMemGetIpcHandle(ze_context, buffer_ptrs[nvl_rank], &nvl_ipc_handle));
        nvl_ipc_handle_ready = true;
    }
}

Buffer::~Buffer() noexcept(false) {
    if (!explicitly_destroy) {
        destroy();
    } else if (!destroyed) {
        printf("WARNING: destroy() was not called before DeepEP XPU buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::object Buffer::get_local_ipc_handle() const {
    if (num_nvl_bytes == 0) {
        return pybind11::none();
    }
    EP_HOST_ASSERT(nvl_ipc_handle_ready);
    XpuIpcMemHandle handle{};
    handle.pid = static_cast<int>(getpid());
    handle.handle = nvl_ipc_handle;
    return pybind11::bytearray(reinterpret_cast<const char*>(&handle), sizeof(handle));
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
    EP_HOST_ASSERT(rdma_rank == 0 && "Only RDMA rank 0 can get iSHMEM unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto* base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]);
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    EP_HOST_ASSERT(base_ptr != nullptr && "Requested buffer is not allocated");
    EP_HOST_ASSERT(offset >= 0 && offset <= num_bytes);
    EP_HOST_ASSERT(offset % element_bytes == 0);
    return torch::from_blob(base_ptr + offset, (num_bytes - offset) / element_bytes, runtime_tensor_options(casted_dtype));
}

torch::Stream Buffer::get_comm_stream() const {
    EP_HOST_ASSERT(comm_stream.has_value());
    return comm_stream.value();
}

void Buffer::sync_internode_runtime(const pybind11::object& root_unique_id_obj) {
    if (num_rdma_bytes == 0) {
        return;
    }

    EP_HOST_ASSERT(!root_unique_id_obj.is_none());
    auto root_unique_id_bytes = root_unique_id_obj.cast<pybind11::bytearray>();
    auto root_unique_id_str = root_unique_id_bytes.cast<std::string>();
    std::vector<uint8_t> root_unique_id(root_unique_id_bytes.size());
    std::memcpy(root_unique_id.data(), root_unique_id_str.data(), root_unique_id_bytes.size());
    auto transport_rank = low_latency_mode ? rank : rdma_rank;
    auto transport_num_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
    EP_HOST_ASSERT(transport_rank == internode::init(root_unique_id, transport_rank, transport_num_ranks, low_latency_mode));
    transport_initialized = true;
    internode::barrier();

    rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
    EP_HOST_ASSERT(rdma_buffer_ptr != nullptr);
    zero_symmetric_buffer(rdma_buffer_ptr, num_rdma_bytes, device_id);

    if (enable_shrink) {
        int num_mask_buffer_bytes = num_ranks * static_cast<int>(sizeof(int));
        int num_sync_buffer_bytes = num_ranks * static_cast<int>(sizeof(int));
        mask_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_mask_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
        sync_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_sync_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
        EP_HOST_ASSERT(mask_buffer_ptr != nullptr && sync_buffer_ptr != nullptr);
        zero_symmetric_buffer(mask_buffer_ptr, num_mask_buffer_bytes, device_id);
        zero_symmetric_buffer(sync_buffer_ptr, num_sync_buffer_bytes, device_id);
    }

    internode::barrier();
    comm_stream = get_stream_from_pool(true);
}

void Buffer::sync(const std::vector<int>& device_ids,
                  const std::vector<pybind11::object>& all_gathered_handles,
                  const pybind11::object& root_unique_id_obj) {
    EP_HOST_ASSERT(!available);
    sync_intranode_handles(device_ids, all_gathered_handles);
    sync_internode_runtime(root_unique_id_obj);
    available = true;
}

void Buffer::sync_intranode_handles(const std::vector<int>& device_ids, const std::vector<pybind11::object>& all_gathered_handles) {
    if (num_nvl_bytes == 0) {
        return;
    }

    EP_HOST_ASSERT(device_ids.size() == static_cast<size_t>(num_ranks));
    EP_HOST_ASSERT(all_gathered_handles.size() == static_cast<size_t>(num_ranks));
    auto& queue = c10::xpu::XPUStream(get_comm_stream()).queue();

    const int group_offset = rdma_rank * num_nvl_ranks;
    for (int local_rank = 0; local_rank < num_nvl_ranks; ++local_rank) {
        const int global_rank = group_offset + local_rank;
        EP_HOST_ASSERT(global_rank >= 0 && global_rank < num_ranks);
        EP_HOST_ASSERT(device_ids[global_rank] >= 0);
        EP_HOST_ASSERT(!all_gathered_handles[global_rank].is_none());

        auto handle_bytes = all_gathered_handles[global_rank].cast<pybind11::bytearray>().cast<std::string>();
        EP_HOST_ASSERT(handle_bytes.size() == sizeof(XpuIpcMemHandle));
        XpuIpcMemHandle remote_handle{};
        std::memcpy(&remote_handle, handle_bytes.data(), sizeof(remote_handle));

        if (global_rank == rank) {
            EP_HOST_ASSERT(std::memcmp(&remote_handle.handle, &nvl_ipc_handle, sizeof(ze_ipc_mem_handle_t)) == 0);
            buffer_ptrs[local_rank] = buffer_ptrs[nvl_rank];
        } else {
            ze_ipc_mem_handle_t import_handle = remote_handle.handle;
            int fd = duplicate_peer_fd(remote_handle.pid, remote_handle.handle);
            *reinterpret_cast<int*>(import_handle.data) = fd;
            ZE_CHECK(zeMemOpenIpcHandle(ze_context, ze_device, import_handle, ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED, &buffer_ptrs[local_rank]));
        }
        barrier_signal_ptrs[local_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[local_rank]) + num_nvl_bytes);
    }

    queue.memcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS);
    queue.memcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS);
    queue.wait_and_throw();
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(
    const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    auto compute_stream = get_current_stream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() && async);
        set_current_stream(comm_stream.value());
    }

    if (previous_event.has_value()) {
        stream_wait(comm_stream.value(), previous_event.value());
    } else {
        stream_wait(comm_stream.value(), compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, runtime_tensor_options(torch::kInt32));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, runtime_tensor_options(torch::kInt32));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, runtime_tensor_options(torch::kBool));
    if (num_rdma_ranks > 1) {
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, runtime_tensor_options(torch::kInt32));
    }

    auto num_tokens_per_rank_cpu = torch::empty({num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto num_tokens_per_rdma_rank_cpu = std::optional<torch::Tensor>();
    if (num_tokens_per_rdma_rank.has_value()) {
        num_tokens_per_rdma_rank_cpu = torch::empty({num_rdma_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    }
    auto num_tokens_per_expert_cpu = torch::empty({num_experts}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto is_token_in_rank_cpu = torch::empty({num_tokens, num_ranks}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    compute_dispatch_layout_cpu(topk_idx,
                                num_experts,
                                num_ranks,
                                num_rdma_ranks,
                                num_tokens_per_rank_cpu.data_ptr<int>(),
                                num_tokens_per_rdma_rank_cpu.has_value() ? num_tokens_per_rdma_rank_cpu->data_ptr<int>() : nullptr,
                                num_tokens_per_expert_cpu.data_ptr<int>(),
                                is_token_in_rank_cpu.data_ptr<bool>());

    auto& queue = c10::xpu::XPUStream(comm_stream.value()).queue();
    queue.memcpy(num_tokens_per_rank.data_ptr<int>(), num_tokens_per_rank_cpu.data_ptr<int>(), num_tokens_per_rank_cpu.numel() * sizeof(int));
    if (num_tokens_per_rdma_rank.has_value()) {
        queue.memcpy(num_tokens_per_rdma_rank->data_ptr<int>(),
                     num_tokens_per_rdma_rank_cpu->data_ptr<int>(),
                     num_tokens_per_rdma_rank_cpu->numel() * sizeof(int));
    }
    queue.memcpy(num_tokens_per_expert.data_ptr<int>(), num_tokens_per_expert_cpu.data_ptr<int>(), num_tokens_per_expert_cpu.numel() * sizeof(int));
    queue.memcpy(is_token_in_rank.data_ptr<bool>(), is_token_in_rank_cpu.data_ptr<bool>(), is_token_in_rank_cpu.numel() * sizeof(uint8_t));
    queue.wait_and_throw();

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream.value());
        for (auto& t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream.value());
            if (allocate_on_comm_stream) {
                t.record_stream(compute_stream);
            }
        }
        if (num_tokens_per_rdma_rank.has_value()) {
            num_tokens_per_rdma_rank->record_stream(comm_stream.value());
            if (allocate_on_comm_stream) {
                num_tokens_per_rdma_rank->record_stream(compute_stream);
            }
        }
    } else {
        stream_wait(compute_stream, comm_stream.value());
    }

    if (allocate_on_comm_stream) {
        set_current_stream(compute_stream);
    }

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
    if (!experimental_native_intranode_enabled()) {
        throw std::runtime_error(
            "DeepEP XPU native intranode dispatch is compiled but remains behind "
            "DEEPEP_XPU_ENABLE_EXPERIMENTAL_INTRANODE=1 because the current Level Zero IPC device-side "
            "cross-rank synchronization path is not validated yet.");
    }
    EP_HOST_ASSERT(available && num_nvl_bytes > 0);
    EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 && is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) && is_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    const bool cached_mode = cached_rank_prefix_matrix.has_value();
    const int num_channels = config.num_sms / 2;
    const int num_tokens = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    const int hidden_int4 = static_cast<int>(hidden * x.element_size() / sizeof(int4));
    const int num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0));
    const int num_local_experts = cached_mode ? 0 : num_experts / num_ranks;
    const int num_memset_int = num_channels * num_ranks * 4;
    const int num_staging_slots = std::max(1, num_tokens * num_nvl_ranks);

    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    if (topk_idx.has_value()) {
        EP_HOST_ASSERT(topk_idx->dim() == 2 && topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_idx->size(0) == num_tokens && topk_weights->size(0) == num_tokens);
        num_topk = static_cast<int>(topk_idx->size(1));
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    float* x_scales_ptr = nullptr;
    int num_scales = 0;
    int scale_token_stride = 0;
    int scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x_scales->is_contiguous());
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->stride(1));
    }
    EP_HOST_ASSERT(native_intranode_staging_bytes(num_nvl_ranks, num_staging_slots, hidden_int4, num_topk, num_scales) <=
                   static_cast<size_t>(num_nvl_bytes));

    auto compute_stream = get_current_stream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() && async);
        set_current_stream(comm_stream.value());
    }
    if (previous_event.has_value()) {
        stream_wait(comm_stream.value(), previous_event.value());
    } else {
        stream_wait(comm_stream.value(), compute_stream);
    }

    int num_recv_tokens = cached_num_recv_tokens;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    if (cached_mode) {
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
        rank_prefix_matrix = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();
        debug_intranode_log(rank, "cached_notify_dispatch begin");
        intranode::cached_notify_dispatch(
            rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu, barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream.value());
        debug_intranode_log(rank, "cached_notify_dispatch end");
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value() && num_tokens_per_expert.has_value());
        rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, runtime_tensor_options(torch::kInt32));
        channel_prefix_matrix = torch::empty({num_ranks, num_channels}, runtime_tensor_options(torch::kInt32));
        auto moe_recv_counter = torch::empty({1}, runtime_tensor_options(torch::kInt32));
        auto moe_recv_expert_counter = torch::empty({num_local_experts}, runtime_tensor_options(torch::kInt32));
        moe_recv_counter.fill_(-1);
        moe_recv_expert_counter.fill_(-1);

        debug_intranode_log(rank, "notify_dispatch begin");
        intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter.data_ptr<int>(),
                                   num_ranks,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter.data_ptr<int>(),
                                   num_experts,
                                   num_tokens,
                                   is_token_in_rank.data_ptr<bool>(),
                                   channel_prefix_matrix.data_ptr<int>(),
                                   rank_prefix_matrix.data_ptr<int>(),
                                   num_memset_int,
                                   expert_alignment,
                                   buffer_ptrs_gpu,
                                   barrier_signal_ptrs_gpu,
                                   nvl_rank,
                                   comm_stream.value(),
                                   num_channels);

        c10::xpu::XPUStream(comm_stream.value()).queue().wait_and_throw();
        debug_intranode_log(rank, "notify_dispatch end");
        if (num_worst_tokens > 0) {
            num_recv_tokens = num_worst_tokens;
        } else {
            num_recv_tokens = tensor_first_int_cpu(moe_recv_counter);
            num_recv_tokens_per_expert_list = tensor_to_int_vector_cpu(moe_recv_expert_counter, num_local_experts);
        }
    }

    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, runtime_tensor_options(torch::kInt32));
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, runtime_tensor_options(torch::kInt32));
    auto send_head = torch::empty({num_tokens, num_ranks}, runtime_tensor_options(torch::kInt32));
    auto recv_topk_idx = std::optional<torch::Tensor>();
    auto recv_topk_weights = std::optional<torch::Tensor>();
    auto recv_x_scales = std::optional<torch::Tensor>();

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

    debug_intranode_log(rank, "dispatch begin");
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
                        is_token_in_rank.data_ptr<bool>(),
                        channel_prefix_matrix.data_ptr<int>(),
                        num_tokens,
                        num_worst_tokens,
                        hidden_int4,
                        num_topk,
                        num_experts,
                        num_scales,
                        scale_token_stride,
                         scale_hidden_stride,
                         buffer_ptrs_gpu,
                         barrier_signal_ptrs_gpu,
                         nvl_rank,
                         num_nvl_ranks,
                        comm_stream.value(),
                        config.num_sms,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens);
    c10::xpu::XPUStream(comm_stream.value()).queue().wait_and_throw();
    debug_intranode_log(rank, "dispatch end");

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream.value());
    } else {
        stream_wait(compute_stream, comm_stream.value());
    }
    if (allocate_on_comm_stream) {
        set_current_stream(compute_stream);
    }

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
    if (!experimental_native_intranode_enabled()) {
        throw std::runtime_error(
            "DeepEP XPU native intranode combine is compiled but remains behind "
            "DEEPEP_XPU_ENABLE_EXPERIMENTAL_INTRANODE=1 because the current Level Zero IPC device-side "
            "cross-rank synchronization path is not validated yet.");
    }
    EP_HOST_ASSERT(available && num_nvl_bytes > 0);
    EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 && src_idx.is_contiguous());
    EP_HOST_ASSERT(send_head.dim() == 2 && send_head.is_contiguous());
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 && rank_prefix_matrix.is_contiguous());
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 && channel_prefix_matrix.is_contiguous());
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    const int num_channels = config.num_sms / 2;
    const int num_tokens = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    const int num_recv_tokens = static_cast<int>(send_head.size(0));
    const int hidden_int4 = static_cast<int>(hidden * x.element_size() / sizeof(int4));
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    auto compute_stream = get_current_stream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() && async);
        set_current_stream(comm_stream.value());
    }
    if (previous_event.has_value()) {
        stream_wait(comm_stream.value(), previous_event.value());
    } else {
        stream_wait(comm_stream.value(), compute_stream);
    }

    int num_topk = 0;
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 && topk_weights->is_contiguous());
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    const int num_staging_slots = std::max(1, num_recv_tokens * num_nvl_ranks);
    EP_HOST_ASSERT(native_intranode_staging_bytes(num_nvl_ranks, num_staging_slots, hidden_int4, num_topk, 0) <=
                   static_cast<size_t>(num_nvl_bytes));

    intranode::cached_notify_combine(buffer_ptrs_gpu,
                                     const_cast<int*>(send_head.data_ptr<int>()),
                                     num_channels,
                                     num_recv_tokens,
                                     num_channels * num_ranks * 2,
                                     barrier_signal_ptrs_gpu,
                                     nvl_rank,
                                     num_nvl_ranks,
                                     comm_stream.value());

    const void* bias_0_ptr = bias_0.has_value() ? bias_0->data_ptr() : nullptr;
    const void* bias_1_ptr = bias_1.has_value() ? bias_1->data_ptr() : nullptr;
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    intranode::combine(runtime_scalar_type_to_data_type(x.scalar_type()),
                       recv_x.data_ptr(),
                       recv_topk_weights_ptr,
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_0_ptr,
                       bias_1_ptr,
                       src_idx.data_ptr<int>(),
                       rank_prefix_matrix.data_ptr<int>(),
                       channel_prefix_matrix.data_ptr<int>(),
                       const_cast<int*>(send_head.data_ptr<int>()),
                       num_tokens,
                       num_recv_tokens,
                       hidden,
                        num_topk,
                        buffer_ptrs_gpu,
                        barrier_signal_ptrs_gpu,
                        nvl_rank,
                        num_nvl_ranks,
                       comm_stream.value(),
                       config.num_sms,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens);

    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream.value());
    } else {
        stream_wait(compute_stream, comm_stream.value());
    }
    if (allocate_on_comm_stream) {
        set_current_stream(compute_stream);
    }
    return {recv_x, recv_topk_weights, event};
}

void Buffer::destroy_internode_resources() {
    if (!available || num_rdma_bytes == 0 || !transport_initialized) {
        return;
    }

    internode::barrier();
    if (rdma_buffer_ptr != nullptr) {
        internode::free(rdma_buffer_ptr);
        rdma_buffer_ptr = nullptr;
    }
    if (mask_buffer_ptr != nullptr) {
        internode::free(mask_buffer_ptr);
        mask_buffer_ptr = nullptr;
    }
    if (sync_buffer_ptr != nullptr) {
        internode::free(sync_buffer_ptr);
        sync_buffer_ptr = nullptr;
    }
    internode::finalize();
    transport_initialized = false;
}

void Buffer::destroy_intranode_resources() {
    if (num_nvl_bytes == 0) {
        return;
    }

    if (comm_stream.has_value()) {
        c10::xpu::XPUStream(comm_stream.value()).queue().wait_and_throw();
    }

    for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) {
        if (i != nvl_rank && buffer_ptrs[i] != nullptr) {
            ZE_CHECK(zeMemCloseIpcHandle(ze_context, buffer_ptrs[i]));
            buffer_ptrs[i] = nullptr;
            barrier_signal_ptrs[i] = nullptr;
        }
    }

    if (comm_stream.has_value()) {
        auto& queue = c10::xpu::XPUStream(comm_stream.value()).queue();
        if (buffer_ptrs_gpu != nullptr) {
            sycl::free(buffer_ptrs_gpu, queue);
            buffer_ptrs_gpu = nullptr;
        }
        if (barrier_signal_ptrs_gpu != nullptr) {
            sycl::free(barrier_signal_ptrs_gpu, queue);
            barrier_signal_ptrs_gpu = nullptr;
        }
    }

    if (buffer_ptrs[nvl_rank] != nullptr) {
        ZE_CHECK(zeMemFree(ze_context, buffer_ptrs[nvl_rank]));
        buffer_ptrs[nvl_rank] = nullptr;
        barrier_signal_ptrs[nvl_rank] = nullptr;
    }
    nvl_ipc_handle_ready = false;
    nvl_allocation_bytes = 0;
}

void Buffer::destroy() {
    EP_HOST_ASSERT(!destroyed);
    destroy_intranode_resources();
    destroy_internode_resources();
    destroyed = true;
    available = false;
}

void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr && "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 && rank_to_mask < num_ranks);
    internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, get_comm_stream());
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr && "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);
    internode_ll::query_mask_buffer(mask_buffer_ptr, num_ranks, reinterpret_cast<int*>(mask_status.data_ptr()), get_comm_stream());
}

void Buffer::low_latency_clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr && "Shrink mode must be enabled");
    internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, get_comm_stream());
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
    EP_HOST_ASSERT(low_latency_mode);
    EP_HOST_ASSERT(rdma_buffer_ptr != nullptr);

    auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [this](const int* ptr, int num_ints) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        auto num_bytes = static_cast<int64_t>(num_ints) * static_cast<int64_t>(sizeof(int));
        EP_HOST_ASSERT(0 <= offset && offset + num_bytes <= num_rdma_bytes);
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second);
    check_boundary(clean_meta_1.first, clean_meta_1.second);

    auto& queue = c10::xpu::XPUStream(get_comm_stream()).queue();
    internode::barrier();
    queue.memset(clean_meta_0.first, 0, static_cast<size_t>(clean_meta_0.second) * sizeof(int));
    queue.memset(clean_meta_1.first, 0, static_cast<size_t>(clean_meta_1.second) * sizeof(int));
    queue.wait_and_throw();
    internode::barrier();
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
    EP_HOST_ASSERT(rdma_buffer_ptr != nullptr);
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto buffer = layout.buffers[0];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));
    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                            {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            runtime_tensor_options(dtype));
}

}  // namespace deep_ep
