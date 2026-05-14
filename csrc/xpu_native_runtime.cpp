#include "xpu_native_runtime.hpp"

#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/python.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "config.hpp"
#include "kernels/api.cuh"
#include "kernels/transport.hpp"

#if defined(DEEPEP_USE_ISHMEM)
#include <ishmemx.h>
#endif

namespace {

sycl::queue make_init_queue_for_device(int device_index) {
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

    if (num_nvl_bytes != 0) {
        throw std::runtime_error(
            "DeepEP XPU native runtime does not implement the NVLink/IPC buffer path yet: "
            "this still needs Level Zero memory IPC handle support plus host-side handle exchange.");
    }

    device_id = c10::xpu::current_device();
    rdma_rank = rank / NUM_MAX_NVL_PEERS;
    nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS);
    num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    auto queue = make_init_queue_for_device(device_id);
    num_device_sms = static_cast<int>(queue.get_device().get_info<sycl::info::device::max_compute_units>());
    EP_HOST_ASSERT(num_device_sms > 0);

    EP_HOST_ASSERT(ceil_div<int64_t>(std::max<int64_t>(num_nvl_bytes, 1), std::max(1, num_device_sms / 2)) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(std::max<int64_t>(num_rdma_bytes, 1), std::max(1, num_device_sms / 2)) < std::numeric_limits<int>::max());
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
    return pybind11::none();
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
    EP_HOST_ASSERT(rdma_rank == 0 && "Only RDMA rank 0 can get iSHMEM unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto* base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : nullptr);
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
    (void)device_ids;
    (void)all_gathered_handles;
    sync_internode_runtime(root_unique_id_obj);
    available = true;
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

void Buffer::destroy() {
    EP_HOST_ASSERT(!destroyed);
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
