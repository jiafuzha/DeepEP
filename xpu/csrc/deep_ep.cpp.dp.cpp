#include "deep_ep.hpp"

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <level_zero/ze_api.h>
#include <pybind11/functional.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <torch/python.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "kernels/api.dp.hpp"
#include "kernels/configs.dp.hpp"

namespace {

using XPUStream = c10::xpu::XPUStream;

constexpr const char* kXpuLowLatencyUnsupportedMessage =
    "DeepEP XPU low-latency kernels are intentionally unsupported: the mirrored API surface is kept for compatibility, "
    "but a portable SYCL/iSHMEM implementation is not available yet.";

void ze_check(ze_result_t status, const char* expr) {
    if (status != ZE_RESULT_SUCCESS) {
        std::ostringstream oss;
        oss << "Level Zero call failed: " << expr << " returned 0x" << std::hex << static_cast<uint32_t>(status);
        throw std::runtime_error(oss.str());
    }
}

#define ZE_CHECK(expr) ze_check((expr), #expr)

torch::TensorOptions xpu_tensor_options(torch::ScalarType dtype, c10::DeviceIndex device_index) {
    return torch::TensorOptions().dtype(dtype).device(c10::Device(c10::kXPU, device_index));
}

torch::ScalarType topk_idx_scalar_type() {
    return sizeof(deep_ep::topk_idx_t) == sizeof(int64_t) ? torch::kInt64 : torch::kInt32;
}

int64_t align_up_bytes(int64_t value, int64_t alignment = 64) {
    return ((value + alignment - 1) / alignment) * alignment;
}

dpct::library_data_t scalar_type_to_library_data(torch::ScalarType scalar_type) {
    switch (scalar_type) {
        case torch::kBFloat16:
            return dpct::library_data_t::real_bfloat16;
        case torch::kFloat16:
            return dpct::library_data_t::real_half;
        case torch::kFloat32:
            return dpct::library_data_t::real_float;
        default:
            EP_HOST_ASSERT(false and "Unsupported scalar type");
            return dpct::library_data_t::real_float;
    }
}

ze_context_handle_t get_ze_context() {
    static std::once_flag ze_init_once;
    std::call_once(ze_init_once, []() { ZE_CHECK(zeInit(0)); });
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c10::xpu::get_device_context());
}

ze_device_handle_t get_ze_device(c10::DeviceIndex device_index = -1) {
    static std::once_flag ze_init_once;
    std::call_once(ze_init_once, []() { ZE_CHECK(zeInit(0)); });
    if (device_index < 0)
        device_index = c10::xpu::current_device();
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(c10::xpu::get_raw_device(device_index));
}

void device_synchronize(c10::DeviceIndex device_index) {
    c10::xpu::syncStreamsOnDevice(device_index);
}

template <typename T>
T* malloc_shared_or_throw(size_t count, const XPUStream& stream) {
    auto* ptr = sycl::malloc_shared<T>(count, stream.queue());
    EP_HOST_ASSERT(ptr != nullptr);
    return ptr;
}

template <typename T>
T* malloc_host_or_throw(size_t count, const XPUStream& stream) {
    auto* ptr = sycl::malloc_host<T>(count, stream.queue());
    EP_HOST_ASSERT(ptr != nullptr);
    return ptr;
}

[[noreturn]] void throw_xpu_low_latency_unsupported(const char* api) {
    throw EPException("XPU", __FILE__, __LINE__, std::string(api) + ": " + kXpuLowLatencyUnsupportedMessage);
}

int create_ipc_socket(char* socket_path, size_t socket_path_size, int rank) {
    const char* master_port = std::getenv("MASTER_PORT");
    int written = std::snprintf(socket_path,
                                socket_path_size,
                                "/tmp/deepep-xpu-ipc-%d-%s-%d.sock",
                                static_cast<int>(geteuid()),
                                master_port == nullptr ? "0" : master_port,
                                rank);
    EP_HOST_ASSERT(written > 0 and static_cast<size_t>(written) < socket_path_size);

    int sockfd = socket(AF_UNIX, SOCK_DGRAM, 0);
    EP_HOST_ASSERT(sockfd >= 0);

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    unlink(socket_path);
    EP_HOST_ASSERT(bind(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    return sockfd;
}

void close_ipc_socket(int sockfd, const char* socket_path) {
    if (sockfd >= 0)
        close(sockfd);
    if (socket_path[0] != '\0')
        unlink(socket_path);
}

void send_fd_no_connection(int sockfd, const char* remote_socket_path, int fd_to_send, int rank) {
    sockaddr_un remote_addr{};
    remote_addr.sun_family = AF_UNIX;
    std::strncpy(remote_addr.sun_path, remote_socket_path, sizeof(remote_addr.sun_path) - 1);

    int payload = rank;
    iovec iov{.iov_base = &payload, .iov_len = sizeof(payload)};

    char control[CMSG_SPACE(sizeof(int))] = {};
    msghdr msg{};
    msg.msg_name = &remote_addr;
    msg.msg_namelen = sizeof(remote_addr);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));

    EP_HOST_ASSERT(sendmsg(sockfd, &msg, 0) == static_cast<ssize_t>(sizeof(payload)));
}

std::pair<int, int> recv_fd_no_connection(int sockfd) {
    int payload = -1;
    iovec iov{.iov_base = &payload, .iov_len = sizeof(payload)};

    char control[CMSG_SPACE(sizeof(int))] = {};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    EP_HOST_ASSERT(recvmsg(sockfd, &msg, 0) == static_cast<ssize_t>(sizeof(payload)));

    cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    EP_HOST_ASSERT(cmsg != nullptr);
    EP_HOST_ASSERT(cmsg->cmsg_level == SOL_SOCKET and cmsg->cmsg_type == SCM_RIGHTS);

    int received_fd = -1;
    std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(received_fd));
    EP_HOST_ASSERT(received_fd >= 0);
    return {received_fd, payload};
}

size_t get_xpu_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    (void)num_max_dispatch_tokens_per_rank;
    (void)hidden;
    (void)num_ranks;
    (void)num_experts;
    throw_xpu_low_latency_unsupported("get_low_latency_rdma_size_hint");
}

}  // namespace

namespace shared_memory {

SharedMemoryAllocator::SharedMemoryAllocator(bool use_fabric) : use_fabric(use_fabric) {}

void SharedMemoryAllocator::malloc(void** ptr, size_t size_raw) {
    auto size = deep_ep::align_up<size_t>(size_raw, NUM_BUFFER_ALIGNMENT_BYTES);
    auto* alloc_ptr = sycl::aligned_alloc_device(NUM_BUFFER_ALIGNMENT_BYTES, size, c10::xpu::getCurrentXPUStream().queue());
    EP_HOST_ASSERT(alloc_ptr != nullptr);
    *ptr = alloc_ptr;
    allocation_sizes[*ptr] = size;
}

void SharedMemoryAllocator::free(void* ptr) {
    allocation_sizes.erase(ptr);
    sycl::free(ptr, c10::xpu::get_device_context());
}

void SharedMemoryAllocator::get_mem_handle(MemHandle* mem_handle, void* ptr) {
    auto alloc_it = allocation_sizes.find(ptr);
    EP_HOST_ASSERT(alloc_it != allocation_sizes.end());
    mem_handle->size = alloc_it->second;
    std::memset(mem_handle->inner.socket_path, 0, sizeof(mem_handle->inner.socket_path));
    ZE_CHECK(zeMemGetIpcHandle(get_ze_context(), ptr, &mem_handle->inner.ze_ipc_mem_handle));
}

void SharedMemoryAllocator::open_mem_handle(void** ptr, MemHandle* mem_handle, int remote_device_index) {
    auto local_device = get_ze_device();
    ze_bool_t can_access_peer = false;
    ZE_CHECK(zeDeviceCanAccessPeer(local_device, get_ze_device(remote_device_index), &can_access_peer));
    EP_HOST_ASSERT(can_access_peer);
    ZE_CHECK(zeMemOpenIpcHandle(get_ze_context(), local_device, mem_handle->inner.ze_ipc_mem_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, ptr));
    void* base_ptr = nullptr;
    size_t base_size = 0;
    ZE_CHECK(zeMemGetAddressRange(get_ze_context(), *ptr, &base_ptr, &base_size));
    EP_HOST_ASSERT(base_ptr == *ptr);
    EP_HOST_ASSERT(base_size >= mem_handle->size);
}

void SharedMemoryAllocator::close_mem_handle(void* ptr) {
    ZE_CHECK(zeMemCloseIpcHandle(get_ze_context(), ptr));
}
}  // namespace shared_memory

namespace deep_ep {

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
      comm_stream(c10::xpu::getStreamFromPool(true)),
      shared_memory_allocator(use_fabric) {
    if (low_latency_mode)
        throw_xpu_low_latency_unsupported("Buffer(low_latency_mode=True)");

    // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

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
    device_id = c10::xpu::current_device();
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
#ifdef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");
#endif

    // Get device info
    auto* device_prop = at::xpu::getDeviceProperties(device_id);
    num_device_sms = std::max(2, static_cast<int>(device_prop->gpu_eu_count));

    // Number of per-channel bytes cannot be large
    EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handles
        shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                                       num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);
        shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
        ipc_socket_fd = create_ipc_socket(ipc_socket_path, sizeof(ipc_socket_path), rank);
        std::strncpy(ipc_handles[nvl_rank].inner.socket_path, ipc_socket_path, sizeof(ipc_handles[nvl_rank].inner.socket_path) - 1);
        buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        // Set barrier signals
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu =
            reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        comm_stream.queue().memset(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes);
    }

    // Create 32 MiB workspace
    workspace = sycl::malloc_device(NUM_WORKSPACE_BYTES, comm_stream.queue());
    EP_HOST_ASSERT(workspace != nullptr);
    comm_stream.queue().memset(workspace, 0, NUM_WORKSPACE_BYTES);

    // MoE counter
    moe_recv_counter = malloc_host_or_throw<int>(1, comm_stream);
    moe_recv_counter_mapped = const_cast<int*>(moe_recv_counter);
    *moe_recv_counter = -1;

    // MoE expert-level counter
    moe_recv_expert_counter = malloc_host_or_throw<int>(NUM_MAX_LOCAL_EXPERTS, comm_stream);
    moe_recv_expert_counter_mapped = const_cast<int*>(moe_recv_expert_counter);
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        moe_recv_rdma_counter = malloc_host_or_throw<int>(1, comm_stream);
        moe_recv_rdma_counter_mapped = const_cast<int*>(moe_recv_rdma_counter);
        *moe_recv_rdma_counter = -1;
    }
}

Buffer::~Buffer() noexcept(false) {
    if (not explicitly_destroy) {
        destroy();
    } else if (not destroyed) {
        const char* use_collective_fallback = std::getenv("DEEP_EP_XPU_USE_COLLECTIVE_FALLBACK");
        if (use_collective_fallback != nullptr and std::strcmp(use_collective_fallback, "1") == 0)
            return;
        printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_internode_available() const {
    return is_available() and num_ranks > NUM_MAX_NVL_PEERS;
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

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    const shared_memory::MemHandle& handle = ipc_handles[nvl_rank];
    return {reinterpret_cast<const char*>(&handle), sizeof(handle)};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
#ifndef DISABLE_NVSHMEM
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
    return torch::from_blob(base_ptr, num_bytes / element_bytes, xpu_tensor_options(casted_dtype, device_id));
}

torch::Stream Buffer::get_comm_stream() const {
    return comm_stream;
}

uint8_t* Buffer::ensure_host_staging_slab(int64_t bytes) {
    EP_HOST_ASSERT(bytes >= 0);
    if (bytes == 0)
        return nullptr;
    if (host_staging_slab == nullptr || host_staging_slab_bytes < bytes) {
        if (host_staging_slab != nullptr)
            sycl::free(host_staging_slab, c10::xpu::get_device_context());
        host_staging_slab = malloc_host_or_throw<uint8_t>(static_cast<size_t>(bytes), comm_stream);
        host_staging_slab_bytes = bytes;
    }
    return host_staging_slab;
}

torch::Tensor Buffer::ensure_pack_scratch_x(const torch::Tensor& like, int64_t rows, int64_t hidden) {
    EP_HOST_ASSERT(rows >= 0 and hidden >= 0);
    if (rows == 0 || hidden == 0)
        return torch::Tensor();
    if (not pack_scratch_x.defined() || pack_scratch_x.scalar_type() != like.scalar_type() || pack_scratch_x.device() != like.device() ||
        pack_scratch_x.dim() != 2 || pack_scratch_x.size(0) < rows || pack_scratch_x.size(1) != hidden) {
        pack_scratch_x = torch::empty({rows, hidden}, like.options());
    }
    return pack_scratch_x;
}

torch::Tensor Buffer::ensure_pack_scratch_topk_weights(const torch::Tensor& like, int64_t rows, int64_t num_topk) {
    EP_HOST_ASSERT(rows >= 0 and num_topk >= 0);
    if (rows == 0 || num_topk == 0)
        return torch::Tensor();
    if (not pack_scratch_topk_weights.defined() || pack_scratch_topk_weights.scalar_type() != like.scalar_type() ||
        pack_scratch_topk_weights.device() != like.device() || pack_scratch_topk_weights.dim() != 2 ||
        pack_scratch_topk_weights.size(0) < rows || pack_scratch_topk_weights.size(1) != num_topk) {
        pack_scratch_topk_weights = torch::empty({rows, num_topk}, like.options());
    }
    return pack_scratch_topk_weights;
}

void Buffer::destroy() {
    EP_HOST_ASSERT(not destroyed);

    // Synchronize
    device_synchronize(device_id);

    if (num_nvl_bytes > 0) {
        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++i)
                if (i != nvl_rank)
                    shared_memory_allocator.close_mem_handle(buffer_ptrs[i]);
        }

        // Free local buffer and error flag
        shared_memory_allocator.free(buffer_ptrs[nvl_rank]);
        close_ipc_socket(ipc_socket_fd, ipc_socket_path);
        ipc_socket_fd = -1;
        ipc_socket_path[0] = '\0';
    }

    // Free NVSHMEM
#ifndef DISABLE_NVSHMEM
    if (is_available() and num_rdma_bytes > 0) {
        device_synchronize(device_id);
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
    sycl::free(workspace, c10::xpu::get_device_context());
    if (host_staging_slab != nullptr) {
        sycl::free(host_staging_slab, c10::xpu::get_device_context());
        host_staging_slab = nullptr;
        host_staging_slab_bytes = 0;
    }
    sycl::free(const_cast<int*>(moe_recv_counter), c10::xpu::get_device_context());

    // Free chunked mode staffs
    sycl::free(const_cast<int*>(moe_recv_expert_counter), c10::xpu::get_device_context());
    if (moe_recv_rdma_counter != nullptr)
        sycl::free(const_cast<int*>(moe_recv_rdma_counter), c10::xpu::get_device_context());

    destroyed = true;
    available = false;
}

void Buffer::sync(const std::vector<int>& device_ids,
                  const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());

    // Sync IPC handles
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(num_ranks == device_ids.size());
        EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
        int offset = rdma_rank * num_nvl_ranks;
        for (int i = 0; i < num_nvl_ranks; ++i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == shared_memory::HANDLE_SIZE);
            std::memcpy(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE);
            if (offset + i == rank)
                EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE) == 0);
        }

        int local_fd = -1;
        std::memcpy(&local_fd, ipc_handles[nvl_rank].inner.ze_ipc_mem_handle.data, sizeof(local_fd));
        EP_HOST_ASSERT(local_fd >= 0);

        for (int step = 1; step < num_nvl_ranks; ++step) {
            int send_idx = (nvl_rank + step) % num_nvl_ranks;
            int recv_idx = (nvl_rank - step + num_nvl_ranks) % num_nvl_ranks;
            send_fd_no_connection(ipc_socket_fd, ipc_handles[send_idx].inner.socket_path, local_fd, rank);

            auto [remote_fd, remote_rank] = recv_fd_no_connection(ipc_socket_fd);
            EP_HOST_ASSERT(remote_rank == offset + recv_idx);

            auto remote_handle = ipc_handles[recv_idx];
            std::memcpy(remote_handle.inner.ze_ipc_mem_handle.data, &remote_fd, sizeof(remote_fd));
            shared_memory_allocator.open_mem_handle(&buffer_ptrs[recv_idx], &remote_handle, device_ids[offset + recv_idx]);
            close(remote_fd);
            barrier_signal_ptrs[recv_idx] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[recv_idx]) + num_nvl_bytes);
        }
        for (int i = 0; i < num_nvl_ranks; ++i) {
            EP_HOST_ASSERT(buffer_ptrs[i] != nullptr);
            EP_HOST_ASSERT(barrier_signal_ptrs[i] != nullptr);
        }

        // Copy all buffer and barrier signal pointers to GPU
        comm_stream.queue().memcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS);
        comm_stream.queue().memcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS);
        device_synchronize(device_id);
    }

    // Sync NVSHMEM handles and allocate memory
#ifndef DISABLE_NVSHMEM
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
        comm_stream.queue().memset(rdma_buffer_ptr, 0, num_rdma_bytes).wait();

        // Allocate and clean shrink buffer
        if (enable_shrink) {
            int num_mask_buffer_bytes = num_ranks * sizeof(int);
            int num_sync_buffer_bytes = num_ranks * sizeof(int);
            mask_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_mask_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            sync_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_sync_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            comm_stream.queue().memset(mask_buffer_ptr, 0, num_mask_buffer_bytes).wait();
            comm_stream.queue().memset(sync_buffer_ptr, 0, num_sync_buffer_bytes).wait();
        }

        // Barrier
        internode::barrier();
        device_synchronize(device_id);
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
    auto compute_stream = c10::xpu::getCurrentXPUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        c10::xpu::setCurrentXPUStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, xpu_tensor_options(torch::kInt32, device_id));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, xpu_tensor_options(torch::kInt32, device_id));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, xpu_tensor_options(torch::kBool, device_id));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, xpu_tensor_options(torch::kInt32, device_id));

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
        c10::xpu::setCurrentXPUStream(compute_stream);

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
                           const std::optional<torch::Tensor>& cached_send_pos,
                           int expert_alignment,
                           int num_worst_tokens,
                           const Config& config,
                           std::optional<EventHandle>& previous_event,
                           bool async,
                           bool allocate_on_comm_stream) {
    bool precomputed_layout_mode = cached_rank_prefix_matrix.has_value();

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (precomputed_layout_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_num_recv_tokens >= 0);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (precomputed_layout_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
        if (cached_send_pos.has_value())
            EP_HOST_ASSERT(cached_send_pos->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    if (precomputed_layout_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
        if (cached_send_pos.has_value()) {
            EP_HOST_ASSERT(cached_send_pos->dim() == 2 and cached_send_pos->is_contiguous());
            EP_HOST_ASSERT(cached_send_pos->size(0) == num_ranks and cached_send_pos->size(1) == x.size(0));
        }
    }
    if (num_tokens_per_expert.has_value()) {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }
    if (not precomputed_layout_mode) {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = num_tokens_per_expert.has_value() ? static_cast<int>(num_tokens_per_expert->size(0)) : 0;
    auto num_local_experts = num_experts / num_ranks;

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
    auto compute_stream = c10::xpu::getCurrentXPUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        c10::xpu::setCurrentXPUStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    int actual_num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;
    auto recv_tokens_per_expert = torch::Tensor();

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (precomputed_layout_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        actual_num_recv_tokens = cached_num_recv_tokens;
        rank_prefix_matrix = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();
        auto local_buffer_ptr = static_cast<int*>(buffer_ptrs[rank]);
        comm_stream.queue().memcpy(local_buffer_ptr, rank_prefix_matrix.data_ptr<int>(), num_ranks * num_ranks * sizeof(int));
        comm_stream.queue().memset(local_buffer_ptr + num_ranks * num_ranks, 0, num_memset_int * sizeof(int));
        comm_stream.queue().wait();
        if (num_worst_tokens > 0 and topk_idx.has_value())
            num_recv_tokens = num_worst_tokens;
    } else {
        rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, xpu_tensor_options(torch::kInt32, device_id));
        channel_prefix_matrix = torch::empty({num_ranks, num_channels}, xpu_tensor_options(torch::kInt32, device_id));
        recv_tokens_per_expert = torch::empty({num_local_experts}, xpu_tensor_options(torch::kInt32, device_id));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
        intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   num_ranks,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   recv_tokens_per_expert.data_ptr<int>(),
                                   num_experts,
                                   num_tokens,
                                   is_token_in_rank.data_ptr<bool>(),
                                   channel_prefix_matrix.data_ptr<int>(),
                                   rank_prefix_matrix.data_ptr<int>(),
                                   num_memset_int,
                                   expert_alignment,
                                   buffer_ptrs_gpu,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   comm_stream,
                                   num_channels);

        if (num_worst_tokens > 0) {
            // No CPU sync, just allocate the worst case
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        } else {
            device_synchronize(device_id);
            auto rank_prefix_matrix_cpu = rank_prefix_matrix.to(torch::kCPU);
            auto recv_tokens_per_expert_cpu = recv_tokens_per_expert.to(torch::kCPU);
            num_recv_tokens = rank_prefix_matrix_cpu.data_ptr<int>()[(num_ranks - 1) * num_ranks + rank];
            actual_num_recv_tokens = num_recv_tokens;
            num_recv_tokens_per_expert_list = std::vector<int>(recv_tokens_per_expert_cpu.data_ptr<int>(),
                                                               recv_tokens_per_expert_cpu.data_ptr<int>() + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, xpu_tensor_options(torch::kInt32, device_id));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, xpu_tensor_options(torch::kInt32, device_id));
    auto send_head = torch::empty({num_tokens, num_ranks}, xpu_tensor_options(torch::kInt32, device_id));
    std::vector<torch::Tensor> dispatch_send_slices;

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    auto dummy_topk_idx = torch::Tensor();
    auto dummy_topk_weights = torch::Tensor();
    auto dummy_x_scales = torch::Tensor();
    auto dummy_recv_topk_idx = torch::Tensor();
    auto dummy_recv_topk_weights = torch::Tensor();
    auto dummy_recv_x_scales = torch::Tensor();
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
    if (topk_idx_ptr == nullptr) {
        dummy_topk_idx = torch::empty({1}, xpu_tensor_options(topk_idx_scalar_type(), device_id));
        topk_idx_ptr = dummy_topk_idx.data_ptr<topk_idx_t>();
        dummy_topk_weights = torch::empty({1}, xpu_tensor_options(torch::kFloat32, device_id));
        topk_weights_ptr = dummy_topk_weights.data_ptr<float>();
        dummy_recv_topk_idx = torch::empty({1}, xpu_tensor_options(topk_idx_scalar_type(), device_id));
        recv_topk_idx_ptr = dummy_recv_topk_idx.data_ptr<topk_idx_t>();
        dummy_recv_topk_weights = torch::empty({1}, xpu_tensor_options(torch::kFloat32, device_id));
        recv_topk_weights_ptr = dummy_recv_topk_weights.data_ptr<float>();
    }
    if (x_scales_ptr == nullptr) {
        dummy_x_scales = torch::empty({1}, xpu_tensor_options(torch::kFloat32, device_id));
        x_scales_ptr = dummy_x_scales.data_ptr<float>();
        dummy_recv_x_scales = torch::empty({1}, xpu_tensor_options(torch::kFloat32, device_id));
        recv_x_scales_ptr = dummy_recv_x_scales.data_ptr<float>();
    }

    if (false and precomputed_layout_mode) {
        EP_HOST_ASSERT(not x_scales.has_value() and "The simple XPU intranode path currently supports tensor inputs only");
        auto rank_prefix_matrix_cpu = rank_prefix_matrix.to(torch::kCPU);
        auto* rank_prefix_ptr = rank_prefix_matrix_cpu.data_ptr<int>();
        std::vector<int> recv_tokens_per_rank(num_ranks);
        for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank)
            recv_tokens_per_rank[dst_rank] = rank_prefix_ptr[(num_ranks - 1) * num_ranks + dst_rank];
        int max_recv_tokens = *std::max_element(recv_tokens_per_rank.begin(), recv_tokens_per_rank.end());

        int row_bytes = hidden * static_cast<int>(recv_x.element_size());
        int64_t x_stage_offset = 0;
        int64_t x_stage_bytes = static_cast<int64_t>(max_recv_tokens) * row_bytes;
        int64_t src_idx_stage_offset = align_up_bytes(x_stage_offset + x_stage_bytes);
        int64_t src_idx_stage_bytes = static_cast<int64_t>(max_recv_tokens) * sizeof(int);
        int64_t topk_idx_stage_offset = align_up_bytes(src_idx_stage_offset + src_idx_stage_bytes);
        int64_t topk_idx_stage_bytes = static_cast<int64_t>(max_recv_tokens) * num_topk * sizeof(topk_idx_t);
        int64_t topk_weight_stage_offset = align_up_bytes(topk_idx_stage_offset + topk_idx_stage_bytes);
        int64_t topk_weight_stage_bytes = static_cast<int64_t>(max_recv_tokens) * num_topk * sizeof(float);
        int64_t signal_offset = align_up_bytes(topk_weight_stage_offset + topk_weight_stage_bytes);
        EP_HOST_ASSERT(signal_offset + static_cast<int64_t>(num_ranks) * sizeof(int) <= num_nvl_bytes);

        auto* local_base_ptr = static_cast<uint8_t*>(buffer_ptrs[rank]);
        auto* local_signal_ptr = reinterpret_cast<int*>(local_base_ptr + signal_offset);
        comm_stream.queue().memset(local_base_ptr + x_stage_offset, 0, signal_offset - x_stage_offset);
        comm_stream.queue().memset(local_signal_ptr, 0, num_ranks * sizeof(int)).wait();
        int phase_base = static_cast<int>(++intranode_simple_phase);
        int ready_phase = phase_base * 2;
        int done_phase = ready_phase + 1;
        for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
            if (dst_rank == rank)
                continue;
            auto* remote_signal_ptr = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[dst_rank]) + signal_offset) + rank;
            comm_stream.queue().memcpy(remote_signal_ptr, &ready_phase, sizeof(ready_phase));
        }
        comm_stream.queue().memcpy(local_signal_ptr + rank, &ready_phase, sizeof(ready_phase)).wait();
        std::vector<int> signal_values(num_ranks, 0);
        while (true) {
            comm_stream.queue().memcpy(signal_values.data(), local_signal_ptr, num_ranks * sizeof(int)).wait();
            bool ready = true;
            for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                if (src_rank == rank)
                    continue;
                if (signal_values[src_rank] < ready_phase) {
                    ready = false;
                    break;
                }
            }
            if (ready)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        std::vector<int> next_offsets(num_ranks);
        for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
            next_offsets[dst_rank] = (rank == 0) ? 0 : rank_prefix_ptr[(rank - 1) * num_ranks + dst_rank];
        }

        auto self_topk_idx = std::optional<torch::Tensor>();
        auto self_topk_weights = std::optional<torch::Tensor>();
        int self_base_offset = 0;
        int self_send_count = 0;
        if (not topk_idx.has_value() and cached_send_pos.has_value()) {
            auto previous_stream = c10::xpu::getCurrentXPUStream();
            c10::xpu::setCurrentXPUStream(comm_stream);
            auto pack_scratch = ensure_pack_scratch_x(x, max_recv_tokens, hidden);
            auto* host_packed = ensure_host_staging_slab(static_cast<int64_t>(max_recv_tokens) * row_bytes);
            for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                int sender_end_offset = rank_prefix_ptr[rank * num_ranks + dst_rank];
                int send_count = sender_end_offset - next_offsets[dst_rank];
                if (send_count <= 0)
                    continue;
                int base_offset = next_offsets[dst_rank];
                auto* remote_base_ptr = static_cast<uint8_t*>(buffer_ptrs[dst_rank]);
                auto send_indices = cached_send_pos->select(0, dst_rank).slice(0, 0, send_count);
                auto packed_slice = pack_scratch.narrow(0, 0, send_count);
                packed_slice.copy_(torch::index_select(x, 0, send_indices.to(torch::kInt64)));
                comm_stream.queue().memcpy(host_packed, packed_slice.data_ptr(), static_cast<int64_t>(send_count) * row_bytes).wait();
                comm_stream.queue()
                    .memcpy(remote_base_ptr + x_stage_offset + static_cast<int64_t>(base_offset) * row_bytes,
                            host_packed,
                            static_cast<int64_t>(send_count) * row_bytes)
                    .wait();
            }
            comm_stream.queue().wait();
            c10::xpu::setCurrentXPUStream(previous_stream);
        } else {
            auto previous_stream = c10::xpu::getCurrentXPUStream();
            c10::xpu::setCurrentXPUStream(comm_stream);
            auto pack_scratch = ensure_pack_scratch_x(x, max_recv_tokens, hidden);
            int64_t staging_bytes = std::max<int64_t>(
                static_cast<int64_t>(max_recv_tokens) * row_bytes,
                std::max<int64_t>(static_cast<int64_t>(max_recv_tokens) * sizeof(int),
                                  static_cast<int64_t>(max_recv_tokens) * num_topk *
                                      std::max<int64_t>(static_cast<int64_t>(sizeof(topk_idx_t)), static_cast<int64_t>(sizeof(float)))));
            auto* host_packed = ensure_host_staging_slab(staging_bytes);
            for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                int sender_end_offset = rank_prefix_ptr[rank * num_ranks + dst_rank];
                int send_count = sender_end_offset - next_offsets[dst_rank];
                if (send_count <= 0)
                    continue;
                int base_offset = next_offsets[dst_rank];
                next_offsets[dst_rank] = sender_end_offset;
                auto* remote_base_ptr = static_cast<uint8_t*>(buffer_ptrs[dst_rank]);
                auto send_indices = cached_send_pos->select(0, dst_rank).slice(0, 0, send_count).contiguous();
                auto send_indices64 = send_indices.to(torch::kInt64);

                auto packed_x = pack_scratch.narrow(0, 0, send_count);
                packed_x.copy_(torch::index_select(x, 0, send_indices64));
                comm_stream.queue().memcpy(host_packed, packed_x.data_ptr(), static_cast<int64_t>(send_count) * row_bytes).wait();
                comm_stream.queue()
                    .memcpy(remote_base_ptr + x_stage_offset + static_cast<int64_t>(base_offset) * row_bytes,
                            host_packed,
                            static_cast<int64_t>(send_count) * row_bytes)
                    .wait();

                comm_stream.queue().memcpy(host_packed, send_indices.data_ptr(), static_cast<int64_t>(send_count) * sizeof(int)).wait();
                comm_stream.queue()
                    .memcpy(remote_base_ptr + src_idx_stage_offset + static_cast<int64_t>(base_offset) * sizeof(int),
                            host_packed,
                            static_cast<int64_t>(send_count) * sizeof(int))
                    .wait();

                int local_expert_begin = dst_rank * num_local_experts;
                int local_expert_end = local_expert_begin + num_local_experts;
                auto packed_topk_idx_cpu = torch::index_select(*topk_idx, 0, send_indices64).to(torch::TensorOptions().device(torch::kCPU));
                auto packed_topk_weights_cpu =
                    torch::index_select(*topk_weights, 0, send_indices64).to(torch::TensorOptions().device(torch::kCPU));
                auto* packed_topk_idx_ptr = packed_topk_idx_cpu.data_ptr<topk_idx_t>();
                auto* packed_topk_weights_ptr = packed_topk_weights_cpu.data_ptr<float>();
                for (int token_idx = 0; token_idx < send_count; ++token_idx) {
                    for (int k_idx = 0; k_idx < num_topk; ++k_idx) {
                        auto& expert_idx = packed_topk_idx_ptr[token_idx * num_topk + k_idx];
                        if (expert_idx >= local_expert_begin and expert_idx < local_expert_end) {
                            expert_idx -= local_expert_begin;
                        } else {
                            expert_idx = static_cast<topk_idx_t>(-1);
                            packed_topk_weights_ptr[token_idx * num_topk + k_idx] = 0.0f;
                        }
                    }
                }
                std::memcpy(host_packed, packed_topk_idx_ptr, static_cast<int64_t>(send_count) * num_topk * sizeof(topk_idx_t));
                comm_stream.queue()
                    .memcpy(remote_base_ptr + topk_idx_stage_offset + static_cast<int64_t>(base_offset) * num_topk * sizeof(topk_idx_t),
                            host_packed,
                            static_cast<int64_t>(send_count) * num_topk * sizeof(topk_idx_t))
                    .wait();
                std::memcpy(host_packed, packed_topk_weights_ptr, static_cast<int64_t>(send_count) * num_topk * sizeof(float));
                comm_stream.queue()
                    .memcpy(remote_base_ptr + topk_weight_stage_offset + static_cast<int64_t>(base_offset) * num_topk * sizeof(float),
                            host_packed,
                            static_cast<int64_t>(send_count) * num_topk * sizeof(float))
                    .wait();
                if (dst_rank == rank) {
                    self_base_offset = base_offset;
                    self_send_count = send_count;
                    auto local_mask = torch::index_select(*topk_idx, 0, send_indices64)
                                          .ge(local_expert_begin)
                                          .logical_and(torch::index_select(*topk_idx, 0, send_indices64).lt(local_expert_end));
                    self_topk_idx = torch::index_select(*topk_idx, 0, send_indices64);
                    self_topk_idx->sub_(local_expert_begin);
                    self_topk_idx->masked_fill_(local_mask.logical_not(), static_cast<topk_idx_t>(-1));
                    self_topk_weights = torch::index_select(*topk_weights, 0, send_indices64);
                    self_topk_weights->masked_fill_(local_mask.logical_not(), 0.0f);
                }
            }
            comm_stream.queue().wait();
            c10::xpu::setCurrentXPUStream(previous_stream);
        }
        for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
            if (dst_rank == rank)
                continue;
            auto* remote_signal_ptr = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[dst_rank]) + signal_offset) + rank;
            comm_stream.queue().memcpy(remote_signal_ptr, &done_phase, sizeof(done_phase));
        }
        comm_stream.queue().memcpy(local_signal_ptr + rank, &done_phase, sizeof(done_phase));
        comm_stream.queue().wait();
        while (true) {
            comm_stream.queue().memcpy(signal_values.data(), local_signal_ptr, num_ranks * sizeof(int)).wait();
            bool ready = true;
            for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                if (src_rank == rank)
                    continue;
                if (signal_values[src_rank] < done_phase) {
                    ready = false;
                    break;
                }
            }
            if (ready)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        comm_stream.queue().memcpy(
            recv_x.data_ptr(), local_base_ptr + x_stage_offset, static_cast<int64_t>(actual_num_recv_tokens) * row_bytes);
        comm_stream.queue().memcpy(recv_src_idx.data_ptr<int>(),
                                   local_base_ptr + src_idx_stage_offset,
                                   static_cast<int64_t>(actual_num_recv_tokens) * sizeof(int));
        if (topk_idx.has_value()) {
            comm_stream.queue().memcpy(recv_topk_idx_ptr,
                                       local_base_ptr + topk_idx_stage_offset,
                                       static_cast<int64_t>(actual_num_recv_tokens) * num_topk * sizeof(topk_idx_t));
            comm_stream.queue().memcpy(recv_topk_weights_ptr,
                                       local_base_ptr + topk_weight_stage_offset,
                                       static_cast<int64_t>(actual_num_recv_tokens) * num_topk * sizeof(float));
        }
        comm_stream.queue().wait();
        if (self_send_count > 0) {
            recv_topk_idx->narrow(0, self_base_offset, self_send_count).copy_(*self_topk_idx);
            recv_topk_weights->narrow(0, self_base_offset, self_send_count).copy_(*self_topk_weights);
            comm_stream.queue().wait();
        }

        if (actual_num_recv_tokens < num_recv_tokens) {
            recv_x.slice(0, actual_num_recv_tokens, num_recv_tokens).zero_();
            recv_src_idx.slice(0, actual_num_recv_tokens, num_recv_tokens).zero_();
            if (recv_topk_idx.has_value())
                recv_topk_idx->slice(0, actual_num_recv_tokens, num_recv_tokens).fill_(-1);
            if (recv_topk_weights.has_value())
                recv_topk_weights->slice(0, actual_num_recv_tokens, num_recv_tokens).zero_();
        }
        recv_channel_prefix_matrix.copy_(channel_prefix_matrix);
        send_head.zero_();

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
                             cached_send_pos,
                             recv_topk_idx,
                             recv_topk_weights,
                             recv_x_scales}) {
                to.has_value() ? to->record_stream(comm_stream) : void();
                if (allocate_on_comm_stream)
                    to.has_value() ? to->record_stream(compute_stream) : void();
            }
            for (auto& t : dispatch_send_slices) {
                t.record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        if (allocate_on_comm_stream)
            c10::xpu::setCurrentXPUStream(compute_stream);

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

    // Dispatch
    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +                                                                     // Size prefix matrix
            num_channels * num_ranks * sizeof(int) +                                                              // Channel start offset
            num_channels * num_ranks * sizeof(int) +                                                              // Channel end offset
            num_channels * num_ranks * sizeof(int) * 2 +                                                          // Queue head and tail
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +  // Data buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                     // Source index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +   // Top-k index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +        // Top-k weight buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales        // FP8 scale buffer
        <= num_nvl_bytes);
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
                        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                        num_topk,
                        num_experts,
                        num_scales,
                        scale_token_stride,
                        scale_hidden_stride,
                        buffer_ptrs_gpu,
                        rank,
                        num_ranks,
                        comm_stream,
                        config.num_sms,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens);

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
        for (auto& t :
             {dummy_topk_idx, dummy_topk_weights, dummy_x_scales, dummy_recv_topk_idx, dummy_recv_topk_weights, dummy_recv_x_scales}) {
            if (t.defined()) {
                t.record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);
            }
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
        c10::xpu::setCurrentXPUStream(compute_stream);

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
    const std::optional<torch::Tensor>& row_src_rank,
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
    if (row_src_rank.has_value())
        EP_HOST_ASSERT(row_src_rank->dim() == 1 and row_src_rank->is_contiguous() and row_src_rank->scalar_type() == torch::kInt32);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    if (row_src_rank.has_value())
        EP_HOST_ASSERT(row_src_rank->size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = c10::xpu::getCurrentXPUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        c10::xpu::setCurrentXPUStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    auto dummy_topk_weights = torch::Tensor();
    auto dummy_recv_topk_weights = torch::Tensor();
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    } else {
        dummy_topk_weights = torch::empty({1}, xpu_tensor_options(torch::kFloat32, device_id));
        topk_weights_ptr = dummy_topk_weights.data_ptr<float>();
        dummy_recv_topk_weights = torch::empty({1}, xpu_tensor_options(torch::kFloat32, device_id));
        recv_topk_weights_ptr = dummy_recv_topk_weights.data_ptr<float>();
    }

    auto rank_prefix_matrix_cpu = rank_prefix_matrix.to(torch::kCPU);
    auto* rank_prefix_ptr = rank_prefix_matrix_cpu.data_ptr<int>();
    auto simple_bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    std::vector<torch::Tensor> combine_send_slices;

    int row_bytes = hidden * static_cast<int>(x.element_size());
    int64_t x_stage_offset = 0;
    int64_t x_stage_bytes = static_cast<int64_t>(num_ranks) * num_recv_tokens * row_bytes;
    int64_t topk_stage_offset = align_up_bytes(x_stage_offset + x_stage_bytes);
    int64_t topk_stage_bytes = static_cast<int64_t>(num_ranks) * num_recv_tokens * num_topk * sizeof(float);
    int64_t signal_offset = align_up_bytes(topk_stage_offset + topk_stage_bytes);
    EP_HOST_ASSERT(signal_offset + static_cast<int64_t>(num_ranks) * sizeof(int) <= num_nvl_bytes);

    auto* local_base_ptr = static_cast<uint8_t*>(buffer_ptrs[rank]);
    auto* local_signal_ptr = reinterpret_cast<int*>(local_base_ptr + signal_offset);
    comm_stream.queue().memset(local_base_ptr + x_stage_offset, 0, signal_offset - x_stage_offset);
    comm_stream.queue().memset(local_signal_ptr, 0, num_ranks * sizeof(int)).wait();
    int phase_base = static_cast<int>(++intranode_simple_phase);
    int ready_phase = phase_base * 2;
    int done_phase = ready_phase + 1;
    for (int src_rank_id = 0; src_rank_id < num_ranks; ++src_rank_id) {
        if (src_rank_id == rank)
            continue;
        auto* remote_signal_ptr = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[src_rank_id]) + signal_offset) + rank;
        comm_stream.queue().memcpy(remote_signal_ptr, &ready_phase, sizeof(ready_phase));
    }
    comm_stream.queue().memcpy(local_signal_ptr + rank, &ready_phase, sizeof(ready_phase)).wait();
    std::vector<int> signal_values(num_ranks, 0);
    while (true) {
        comm_stream.queue().memcpy(signal_values.data(), local_signal_ptr, num_ranks * sizeof(int)).wait();
        bool ready = true;
        for (int src_rank_id = 0; src_rank_id < num_ranks; ++src_rank_id) {
            if (src_rank_id == rank)
                continue;
            if (signal_values[src_rank_id] < ready_phase) {
                ready = false;
                break;
            }
        }
        if (ready)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    bool use_fast_no_topk_combine = false;
    auto self_packed_topk_cpu = std::optional<torch::Tensor>();
    if (use_fast_no_topk_combine) {
        auto previous_stream = c10::xpu::getCurrentXPUStream();
        c10::xpu::setCurrentXPUStream(comm_stream);
        auto pack_scratch = ensure_pack_scratch_x(x, static_cast<int64_t>(num_ranks) * num_recv_tokens, hidden);
        auto* host_packed = ensure_host_staging_slab(static_cast<int64_t>(num_recv_tokens) * row_bytes);
        int start = 0;
        for (int src_rank_id = 0; src_rank_id < num_ranks; ++src_rank_id) {
            int end = rank_prefix_ptr[src_rank_id * num_ranks + rank];
            int row_count = end - start;
            if (row_count > 0) {
                auto* remote_base_ptr = static_cast<uint8_t*>(buffer_ptrs[src_rank_id]);
                auto row_slot = static_cast<int64_t>(rank) * num_recv_tokens;
                auto packed_slice = pack_scratch.narrow(0, src_rank_id * num_recv_tokens, num_recv_tokens);
                packed_slice.zero_();
                auto dst_indices = src_idx.slice(0, start, end).to(torch::kInt64);
                auto src_rows = x.slice(0, start, end);
                packed_slice.index_copy_(0, dst_indices, src_rows);
                auto* packed_slice_ptr = static_cast<uint8_t*>(packed_slice.data_ptr());
                comm_stream.queue().memcpy(host_packed, packed_slice_ptr, static_cast<int64_t>(num_recv_tokens) * row_bytes).wait();
                comm_stream.queue()
                    .memcpy(remote_base_ptr + x_stage_offset + row_slot * row_bytes,
                            host_packed,
                            static_cast<int64_t>(num_recv_tokens) * row_bytes)
                    .wait();
            }
            start = end;
        }
        comm_stream.queue().wait();
        c10::xpu::setCurrentXPUStream(previous_stream);
    } else {
        auto previous_stream = c10::xpu::getCurrentXPUStream();
        c10::xpu::setCurrentXPUStream(comm_stream);
        auto packed_x = ensure_pack_scratch_x(x, num_recv_tokens, hidden);
        auto packed_topk =
            topk_weights.has_value() ? ensure_pack_scratch_topk_weights(topk_weights.value(), num_recv_tokens, num_topk) : torch::Tensor();
        auto* host_packed =
            ensure_host_staging_slab(std::max(static_cast<int64_t>(num_recv_tokens) * row_bytes,
                                              static_cast<int64_t>(num_recv_tokens) * num_topk * static_cast<int64_t>(sizeof(float))));
        int start = 0;
        for (int src_rank_id = 0; src_rank_id < num_ranks; ++src_rank_id) {
            int end = rank_prefix_ptr[src_rank_id * num_ranks + rank];
            int row_count = end - start;
            auto* remote_base_ptr = static_cast<uint8_t*>(buffer_ptrs[src_rank_id]);
            if (row_count > 0) {
                auto dst_indices = src_idx.slice(0, start, end).to(torch::kInt64);
                auto row_slot = static_cast<int64_t>(rank) * num_recv_tokens;
                packed_x.zero_();
                packed_x.index_copy_(0, dst_indices, x.slice(0, start, end));
                comm_stream.queue().memcpy(host_packed, packed_x.data_ptr(), static_cast<int64_t>(num_recv_tokens) * row_bytes).wait();
                comm_stream.queue()
                    .memcpy(remote_base_ptr + x_stage_offset + row_slot * row_bytes,
                            host_packed,
                            static_cast<int64_t>(num_recv_tokens) * row_bytes)
                    .wait();
                if (topk_weights.has_value()) {
                    packed_topk.zero_();
                    packed_topk.index_copy_(0, dst_indices, topk_weights->slice(0, start, end));
                    if (src_rank_id == rank)
                        self_packed_topk_cpu = packed_topk.to(torch::kCPU);
                    comm_stream.queue()
                        .memcpy(host_packed, packed_topk.data_ptr(), static_cast<int64_t>(num_recv_tokens) * num_topk * sizeof(float))
                        .wait();
                    comm_stream.queue()
                        .memcpy(remote_base_ptr + topk_stage_offset + row_slot * num_topk * sizeof(float),
                                host_packed,
                                static_cast<int64_t>(num_recv_tokens) * num_topk * sizeof(float))
                        .wait();
                }
            }
            start = end;
        }
        comm_stream.queue().wait();
        c10::xpu::setCurrentXPUStream(previous_stream);
    }

    for (int src_rank_id = 0; src_rank_id < num_ranks; ++src_rank_id) {
        if (src_rank_id == rank)
            continue;
        auto* remote_signal_ptr = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[src_rank_id]) + signal_offset) + rank;
        comm_stream.queue().memcpy(remote_signal_ptr, &done_phase, sizeof(done_phase));
    }
    comm_stream.queue().memcpy(local_signal_ptr + rank, &done_phase, sizeof(done_phase));
    comm_stream.queue().wait();

    while (true) {
        comm_stream.queue().memcpy(signal_values.data(), local_signal_ptr, num_ranks * sizeof(int)).wait();
        bool ready = true;
        for (int src_rank_id = 0; src_rank_id < num_ranks; ++src_rank_id) {
            if (src_rank_id == rank)
                continue;
            if (signal_values[src_rank_id] < done_phase) {
                ready = false;
                break;
            }
        }
        if (ready)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    torch::Tensor simple_recv_x;
    if (use_fast_no_topk_combine) {
        auto previous_stream = c10::xpu::getCurrentXPUStream();
        c10::xpu::setCurrentXPUStream(comm_stream);
        auto stage_x = torch::from_blob(local_base_ptr + x_stage_offset, {num_ranks, num_recv_tokens, hidden}, x.options());
        simple_recv_x = stage_x.to(torch::kFloat32).sum(0).to(x.options());
        c10::xpu::setCurrentXPUStream(previous_stream);
    } else {
        std::vector<uint8_t> host_stage_x(x_stage_bytes);
        comm_stream.queue().memcpy(host_stage_x.data(), local_base_ptr + x_stage_offset, x_stage_bytes).wait();
        auto cpu_stage_x = torch::from_blob(host_stage_x.data(),
                                            {num_ranks, num_recv_tokens, hidden},
                                            torch::TensorOptions().dtype(x.scalar_type()).device(torch::kCPU))
                               .clone();
        simple_recv_x = cpu_stage_x.to(torch::kFloat32).sum(0).to(x.options());
    }
    if (topk_weights.has_value()) {
        std::vector<float> host_stage_topk(num_ranks * num_recv_tokens * num_topk);
        comm_stream.queue().memcpy(host_stage_topk.data(), local_base_ptr + topk_stage_offset, topk_stage_bytes).wait();
        if (self_packed_topk_cpu.has_value()) {
            std::memcpy(host_stage_topk.data() + static_cast<int64_t>(rank) * num_recv_tokens * num_topk,
                        self_packed_topk_cpu->data_ptr<float>(),
                        static_cast<int64_t>(num_recv_tokens) * num_topk * sizeof(float));
        }
        auto cpu_stage_topk = torch::from_blob(host_stage_topk.data(),
                                               {num_ranks, num_recv_tokens, num_topk},
                                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                                  .clone();
        recv_topk_weights = cpu_stage_topk.sum(0).to(topk_weights->options());
    }

    for (int i = 0; i < 2; ++i) {
        if (simple_bias_opts[i].has_value())
            simple_recv_x.add_(simple_bias_opts[i].value());
    }

    std::optional<EventHandle> simple_event;
    if (async) {
        simple_event = EventHandle(comm_stream);
        for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, simple_recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {topk_weights, recv_topk_weights, bias_0, bias_1, row_src_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
        for (auto& t : combine_send_slices) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    if (allocate_on_comm_stream)
        c10::xpu::setCurrentXPUStream(compute_stream);

    return {simple_recv_x, recv_topk_weights, simple_event};

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu,
                                     send_head.data_ptr<int>(),
                                     num_channels,
                                     num_recv_tokens,
                                     num_channels * num_ranks * 2,
                                     barrier_signal_ptrs_gpu,
                                     rank,
                                     num_ranks,
                                     comm_stream);

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
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * x.element_size() +  // Data buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +             // Source index buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)  // Top-k weight buffer
                   <= num_nvl_bytes);
    intranode::combine(scalar_type_to_library_data(x.scalar_type()),
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
                       buffer_ptrs_gpu,
                       rank,
                       num_ranks,
                       comm_stream,
                       config.num_sms,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& t : {dummy_topk_weights, dummy_recv_topk_weights}) {
            if (t.defined()) {
                t.record_stream(comm_stream);
                if (allocate_on_comm_stream)
                    t.record_stream(compute_stream);
            }
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
        c10::xpu::setCurrentXPUStream(compute_stream);

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
    auto compute_stream = c10::xpu::getCurrentXPUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        c10::xpu::setCurrentXPUStream(comm_stream);
    }

    // Wait previous tasks to be finished
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
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

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
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, xpu_tensor_options(torch::kInt32, device_id));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, xpu_tensor_options(torch::kInt32, device_id));
        gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, xpu_tensor_options(torch::kInt32, device_id));
        recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, xpu_tensor_options(torch::kInt32, device_id));

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_rdma_rank->data_ptr<int>(),
                                   moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                                   num_experts,
                                   is_token_in_rank.data_ptr<bool>(),
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
                                   rank,
                                   comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
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
        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, xpu_tensor_options(torch::kByte, device_id));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, xpu_tensor_options(torch::kInt32, device_id));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, xpu_tensor_options(torch::kInt32, device_id));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, xpu_tensor_options(torch::kInt32, device_id));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, xpu_tensor_options(torch::kInt32, device_id));
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
                        is_token_in_rank.data_ptr<bool>(),
                        num_tokens,
                        num_worst_tokens,
                        hidden_int4,
                        num_scales,
                        num_topk,
                        num_experts,
                        scale_token_stride,
                        scale_hidden_stride,
                        rdma_buffer_ptr,
                        config.num_max_rdma_chunked_send_tokens,
                        config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens,
                        rank,
                        num_ranks,
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
        c10::xpu::setCurrentXPUStream(compute_stream);

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
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                   combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = c10::xpu::getCurrentXPUStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        c10::xpu::setCurrentXPUStream(comm_stream);
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

    // Launch barrier and reset queue head and tail
    internode::cached_notify(hidden_int4,
                             0,
                             0,
                             num_topk,
                             num_ranks,
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
                             rank,
                             comm_stream,
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_nvl_bytes,
                             false,
                             low_latency_mode);

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

    // Launch data combine
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    internode::combine(scalar_type_to_library_data(x.scalar_type()),
                       combined_x.data_ptr(),
                       combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_ptr<bool>(),
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
                       rank,
                       num_ranks,
                       comm_stream,
                       num_channels,
                       low_latency_mode);

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
        c10::xpu::setCurrentXPUStream(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
    (void)num_max_dispatch_tokens_per_rank;
    (void)hidden;
    (void)num_experts;
    throw_xpu_low_latency_unsupported("Buffer::clean_low_latency_buffer");
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
    (void)x;
    (void)topk_idx;
    (void)cumulative_local_expert_recv_stats;
    (void)dispatch_wait_recv_cost_stats;
    (void)num_max_dispatch_tokens_per_rank;
    (void)num_experts;
    (void)use_fp8;
    (void)round_scale;
    (void)use_ue8m0;
    (void)async;
    (void)return_recv_hook;
    throw_xpu_low_latency_unsupported("Buffer::low_latency_dispatch");
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
    (void)x;
    (void)topk_idx;
    (void)topk_weights;
    (void)src_info;
    (void)layout_range;
    (void)combine_wait_recv_cost_stats;
    (void)num_max_dispatch_tokens_per_rank;
    (void)num_experts;
    (void)use_logfmt;
    (void)zero_copy;
    (void)async;
    (void)return_recv_hook;
    (void)out;
    throw_xpu_low_latency_unsupported("Buffer::low_latency_combine");
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
    (void)num_max_dispatch_tokens_per_rank;
    (void)hidden;
    (void)num_experts;
    throw_xpu_low_latency_unsupported("Buffer::get_next_low_latency_combine_buffer");
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
    (void)rank_to_mask;
    (void)mask;
    throw_xpu_low_latency_unsupported("Buffer::low_latency_update_mask_buffer");
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
    (void)mask_status;
    throw_xpu_low_latency_unsupported("Buffer::low_latency_query_mask_buffer");
}

void Buffer::low_latency_clean_mask_buffer() {
    throw_xpu_low_latency_unsupported("Buffer::low_latency_clean_mask_buffer");
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
        .def_readwrite("num_sms", &deep_ep::Config::num_sms)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &get_xpu_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
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
    m.attr("topk_idx_t") =
        py::reinterpret_borrow<py::object>((PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value));
}
