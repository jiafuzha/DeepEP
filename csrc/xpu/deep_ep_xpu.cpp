#include <c10/xpu/XPUEvent.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <level_zero/ze_api.h>
#include <torch/python.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#ifdef DEEP_EP_ENABLE_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

#include "xpu_runtime.hpp"

namespace py = pybind11;

namespace deep_ep {

namespace {

constexpr int kNumMaxTopK = 128;
constexpr int kNumMaxScales = 128;
constexpr size_t kNumWorkspaceBytes = 32 * 1024 * 1024;
constexpr ze_external_memory_type_flags_t kIpcExternalMemoryType = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;

#define ZE_CHECK(cmd)                                                                                                     \
    do {                                                                                                                  \
        ze_result_t result = (cmd);                                                                                       \
        TORCH_CHECK(result == ZE_RESULT_SUCCESS, "Level Zero call failed: " #cmd " returned ", static_cast<int>(result)); \
    } while (0)

struct XpuIpcHandle {
    ze_ipc_mem_handle_t handle{};
    uint64_t offset = 0;
    uint64_t allocation_size = 0;
    uint64_t requested_size = 0;
    int rank = -1;
    int device_id = -1;
    int pid = 0;
    int fd = -1;
};

void check_xpu_tensor(const torch::Tensor& tensor, const char* name);

struct LowLatencyBufferLayout {
    size_t mask_offset = 0;
    size_t sync_offset = 0;
    size_t total_bytes = 0;
};

LowLatencyBufferLayout get_low_latency_buffer_layout(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    const int num_local_experts = num_experts / num_ranks;
    const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    const size_t num_dispatch_slots = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
    const size_t num_send_slots = static_cast<size_t>(num_ranks) * num_local_experts * num_max_dispatch_tokens_per_rank;
    const size_t num_combine_slots = static_cast<size_t>(num_experts) * num_max_dispatch_tokens_per_rank;
    size_t offset = 0;
    auto add = [&](size_t bytes) {
        const size_t old = offset;
        offset = align_up<size_t>(offset + bytes, NUM_BUFFER_ALIGNMENT_BYTES);
        return old;
    };
    add(num_dispatch_slots * hidden_bytes);
    add(num_dispatch_slots * sizeof(int));
    add(static_cast<size_t>(num_local_experts) * num_ranks * sizeof(int));
    add(num_send_slots * hidden_bytes);
    add(num_send_slots * sizeof(int));
    add(static_cast<size_t>(num_ranks) * num_local_experts * sizeof(int));
    add(num_combine_slots * hidden_bytes);
    add(static_cast<size_t>(num_experts) * sizeof(int));
    LowLatencyBufferLayout layout;
    layout.mask_offset = add(static_cast<size_t>(num_ranks) * sizeof(int));
    layout.sync_offset = add(static_cast<size_t>(num_ranks) * sizeof(int));
    layout.total_bytes = offset;
    return layout;
}

DataType scalar_type_to_data_type(c10::ScalarType scalar_type) {
    if (scalar_type == torch::kBFloat16) {
        return DataType::kBFloat16;
    }
    if (scalar_type == torch::kInt32) {
        return DataType::kInt32;
    }
    TORCH_CHECK(false, "XPU intranode combine supports only bfloat16 and int32 tensors for now");
}

void ze_init_once() {
    static std::once_flag once;
    std::call_once(once, []() { ZE_CHECK(zeInit(0)); });
}

ze_context_handle_t get_native_context(const sycl::queue& queue) {
    ze_init_once();
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
}

ze_device_handle_t get_native_device(const sycl::queue& queue) {
    ze_init_once();
    return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
}

void check_xpu_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::XPU, name, " must be an XPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

XpuIpcHandle unpack_ipc_handle(const pybind11::bytearray& bytes) {
    std::string handle_str = bytes.cast<std::string>();
    TORCH_CHECK(handle_str.size() == sizeof(XpuIpcHandle), "unexpected XPU IPC handle size");
    XpuIpcHandle handle{};
    std::memcpy(&handle, handle_str.data(), sizeof(handle));
    return handle;
}

}  // namespace

namespace internode {

#ifdef DEEP_EP_ENABLE_ISHMEM
std::vector<uint8_t> get_unique_id() {
    ishmemx_uniqueid_t unique_id{};
    int result = ishmemx_get_uniqueid(&unique_id);
    TORCH_CHECK(result == 0, "ishmemx_get_uniqueid failed with ", result);
    std::vector<uint8_t> bytes(sizeof(unique_id));
    std::memcpy(bytes.data(), &unique_id, sizeof(unique_id));
    return bytes;
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool) {
    TORCH_CHECK(root_unique_id_val.size() == sizeof(ishmemx_uniqueid_t), "unexpected iSHMEM unique ID size");

    int initialized = 0;
    ishmemx_query_initialized(&initialized);
    if (initialized) {
        TORCH_CHECK(ishmem_my_pe() == rank, "iSHMEM was already initialized with an unexpected PE rank");
        TORCH_CHECK(ishmem_n_pes() == num_ranks, "iSHMEM was already initialized with an unexpected PE count");
        return ishmem_my_pe();
    }

    ishmemx_uniqueid_t root_unique_id{};
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(root_unique_id));
    ishmemx_attr_t attr{};
    attr.runtime = ISHMEMX_RUNTIME_MPI;
    attr.initialize_runtime = true;
    attr.gpu = true;
    attr.device_idx = c10::xpu::current_device();
    attr.use_uid = true;
    attr.nranks = num_ranks;
    attr.rank = rank;
    attr.uid = &root_unique_id;
    ishmemx_init_attr(&attr);
    TORCH_CHECK(ishmem_my_pe() == rank, "iSHMEM initialized with PE ", ishmem_my_pe(), ", expected ", rank);
    TORCH_CHECK(ishmem_n_pes() == num_ranks, "iSHMEM initialized with ", ishmem_n_pes(), " PEs, expected ", num_ranks);
    ishmem_barrier_all();
    return ishmem_my_pe();
}

void* alloc(size_t size, size_t alignment) {
    void* ptr = ishmem_align(alignment, size);
    TORCH_CHECK(ptr != nullptr, "ishmem_align failed for ", size, " bytes");
    return ptr;
}

void free(void* ptr) {
    if (ptr != nullptr) {
        ishmem_free(ptr);
    }
}

void barrier() {
    ishmem_barrier_all();
}

void finalize() {
    int initialized = 0;
    ishmemx_query_initialized(&initialized);
    if (initialized) {
        ishmem_barrier_all();
        ishmem_finalize();
    }
}
#else
std::vector<uint8_t> get_unique_id() {
    TORCH_CHECK(false, "XPU iSHMEM support is not enabled in this build");
}

int init(const std::vector<uint8_t>&, int, int, bool) {
    TORCH_CHECK(false, "XPU iSHMEM support is not enabled in this build");
}

void* alloc(size_t, size_t) {
    TORCH_CHECK(false, "XPU iSHMEM support is not enabled in this build");
}

void free(void*) {}

void barrier() {
    TORCH_CHECK(false, "XPU iSHMEM support is not enabled in this build");
}

void finalize() {}
#endif

int get_source_meta_bytes() {
    return sizeof(SourceMeta);
}

}  // namespace internode

size_t Config::get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
    TORCH_CHECK(num_ranks > 0, "num_ranks must be positive");
    TORCH_CHECK(num_ranks < NUM_MAX_NVL_PEERS || num_ranks % NUM_MAX_NVL_PEERS == 0,
                "num_ranks must be less than NUM_MAX_NVL_PEERS or divisible by NUM_MAX_NVL_PEERS");
    const int num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
    const int num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
    const int num_channels = std::max(num_sms / 2, 1);

    size_t num_bytes = 0;
    num_bytes += num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * hidden_bytes;
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t);
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(float);
    num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxScales * sizeof(float);
    return align_up<size_t>(num_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
}

size_t Config::get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
    TORCH_CHECK(num_ranks > NUM_MAX_NVL_PEERS, "RDMA buffer is only required for internode ranks");
    TORCH_CHECK(num_ranks % NUM_MAX_NVL_PEERS == 0, "internode num_ranks must be divisible by NUM_MAX_NVL_PEERS");
    const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    const int num_channels = std::max(num_sms / 2, 1);

    size_t num_bytes = 0;
    num_bytes += num_channels * num_rdma_ranks * (NUM_MAX_NVL_PEERS * 2 + 2) * 2 * sizeof(int);
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t) * 2;
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(float) * 2;
    num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxScales * sizeof(float) * 2;
    return align_up<size_t>(num_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
}

EventHandle::EventHandle() : event(std::make_shared<c10::xpu::XPUEvent>()) {
    event->record(c10::xpu::getCurrentXPUStream());
}

EventHandle::EventHandle(const c10::xpu::XPUStream& stream) : event(std::make_shared<c10::xpu::XPUEvent>()) {
    event->record(stream);
}

void EventHandle::current_stream_wait() const {
    event->block(c10::xpu::getCurrentXPUStream());
}

void stream_wait(const c10::xpu::XPUStream& dst, const c10::xpu::XPUStream& src) {
    auto event = c10::xpu::XPUEvent();
    event.record(src);
    event.block(dst);
}

void stream_wait(const c10::xpu::XPUStream& dst, const EventHandle& event) {
    event.event->block(dst);
}

size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    const int num_scales = hidden / 128;
    const size_t num_bytes_per_dispatch_msg =
        sizeof(int32_t) * 4 + std::max<size_t>(hidden * sizeof(sycl::ext::oneapi::bfloat16), hidden + num_scales * sizeof(float));
    const size_t num_bytes_per_combine_msg =
        num_scales * sizeof(sycl::ext::oneapi::bfloat16) * 2 + hidden * sizeof(sycl::ext::oneapi::bfloat16);
    const size_t dispatch_send_buffer_bytes = static_cast<size_t>(num_max_dispatch_tokens_per_rank) * num_bytes_per_dispatch_msg;
    const size_t combine_send_buffer_bytes =
        static_cast<size_t>(num_experts) * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;
    const size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
    const size_t dispatch_recv_data_buffer_bytes =
        static_cast<size_t>(num_experts) * num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
    const size_t combine_recv_buffer_bytes =
        static_cast<size_t>(num_experts) * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;
    const size_t recv_buffer_bytes = std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);
    const size_t signaling_buffer_bytes = static_cast<size_t>(num_experts) * sizeof(int);
    const size_t signaling_buffer_bytes_aligned = align_up<size_t>(signaling_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
    const size_t legacy_hint =
        align_up<size_t>((send_buffer_bytes + recv_buffer_bytes + signaling_buffer_bytes_aligned) * 2, NUM_BUFFER_ALIGNMENT_BYTES);
    const size_t xpu_layout_bytes =
        get_low_latency_buffer_layout(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts).total_bytes;
    return std::max(legacy_hint, xpu_layout_bytes);
}

struct Buffer {
    int rank;
    int num_ranks;
    int rdma_rank;
    int nvl_rank;
    int num_rdma_ranks;
    int num_nvl_ranks;
    int device_id;
    int64_t num_nvl_bytes;
    int64_t num_rdma_bytes;
    bool low_latency_mode;
    bool explicitly_destroy;
    bool destroyed = false;
    bool available = false;
    c10::xpu::XPUStream comm_stream;
    ze_context_handle_t ze_context = nullptr;
    ze_device_handle_t ze_device = nullptr;

    torch::Tensor local_buffer;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;
    int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;

    mutable XpuIpcHandle local_ipc_handle{};
    mutable bool local_ipc_handle_valid = false;
    mutable void* local_ipc_base = nullptr;
    ze_ipc_mem_handle_t imported_ipc_handles[NUM_MAX_NVL_PEERS] = {};
    bool imported_ipc_handle_valid[NUM_MAX_NVL_PEERS] = {false};
    void* imported_ipc_bases[NUM_MAX_NVL_PEERS] = {nullptr};
    bool imported_external_allocation_valid[NUM_MAX_NVL_PEERS] = {false};
    int barrier_signal_counter = 1;

    void* workspace = nullptr;
    void* rdma_buffer_ptr = nullptr;
    int* low_latency_mask_buffer_ptr = nullptr;
    int* low_latency_sync_buffer_ptr = nullptr;
    int low_latency_num_max_dispatch_tokens_per_rank = 0;
    int low_latency_hidden = 0;
    int low_latency_num_experts = 0;
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;
    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;
    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;

    Buffer(
        int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode, bool explicitly_destroy, bool, bool)
        : rank(rank),
          num_ranks(num_ranks),
          rdma_rank(rank / NUM_MAX_NVL_PEERS),
          nvl_rank(rank % NUM_MAX_NVL_PEERS),
          num_rdma_ranks(low_latency_mode ? num_ranks : std::max(1, num_ranks / NUM_MAX_NVL_PEERS)),
          num_nvl_ranks(std::min(num_ranks, NUM_MAX_NVL_PEERS)),
          device_id(c10::xpu::current_device()),
          num_nvl_bytes(num_nvl_bytes),
          num_rdma_bytes(num_rdma_bytes),
          low_latency_mode(low_latency_mode),
          explicitly_destroy(explicitly_destroy),
          comm_stream(c10::xpu::getStreamFromPool(true)) {
        TORCH_CHECK(rank >= 0 && rank < num_ranks, "rank must be in [0, num_ranks)");
        TORCH_CHECK(num_ranks > 0, "num_ranks must be positive");
        TORCH_CHECK(num_ranks <= NUM_MAX_NVL_PEERS || num_ranks % NUM_MAX_NVL_PEERS == 0,
                    "XPU internode ranks must be divisible by ",
                    NUM_MAX_NVL_PEERS);
        TORCH_CHECK(num_rdma_bytes == 0 || num_ranks > NUM_MAX_NVL_PEERS || low_latency_mode,
                    "XPU RDMA buffer is only valid for internode or low-latency ranks");
        TORCH_CHECK(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0, "num_nvl_bytes must be aligned to ", NUM_BUFFER_ALIGNMENT_BYTES);
        TORCH_CHECK(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0, "num_rdma_bytes must be aligned to ", NUM_BUFFER_ALIGNMENT_BYTES);
        TORCH_CHECK(num_rdma_ranks == 1 || num_rdma_bytes > 0, "XPU internode mode requires a non-empty RDMA buffer");

        auto& queue = comm_stream.queue();
        ze_context = get_native_context(queue);
        ze_device = get_native_device(queue);

        if (num_nvl_bytes > 0) {
            const int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * static_cast<int64_t>(sizeof(int));
            const int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * static_cast<int64_t>(sizeof(void*));
            const int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * static_cast<int64_t>(sizeof(int*));
            const int64_t total_bytes = num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes;

            ze_external_memory_export_desc_t export_desc = {};
            export_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC;
            export_desc.flags = kIpcExternalMemoryType;
            ze_device_mem_alloc_desc_t alloc_desc = {};
            alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
            alloc_desc.pNext = &export_desc;
            ZE_CHECK(zeMemAllocDevice(ze_context, &alloc_desc, total_bytes, NUM_BUFFER_ALIGNMENT_BYTES, ze_device, &buffer_ptrs[nvl_rank]));
            local_buffer = torch::from_blob(
                buffer_ptrs[nvl_rank], {total_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(c10::kXPU, device_id));
            buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);
            barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
            barrier_signal_ptrs_gpu = reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes +
                                                              barrier_signal_bytes + buffer_ptr_bytes);
            queue.memset(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes).wait();
        }

        workspace = sycl::malloc_device(kNumWorkspaceBytes, queue);
        TORCH_CHECK(workspace != nullptr, "Failed to allocate XPU workspace");
        queue.memset(workspace, 0, kNumWorkspaceBytes).wait();

        moe_recv_counter_mapped = sycl::malloc_shared<int>(1, queue);
        moe_recv_expert_counter_mapped = sycl::malloc_shared<int>(NUM_MAX_LOCAL_EXPERTS, queue);
        TORCH_CHECK(moe_recv_counter_mapped != nullptr && moe_recv_expert_counter_mapped != nullptr,
                    "Failed to allocate XPU shared counters");
        moe_recv_counter = moe_recv_counter_mapped;
        moe_recv_expert_counter = moe_recv_expert_counter_mapped;
        if (num_rdma_ranks > 1) {
            moe_recv_rdma_counter_mapped = sycl::malloc_shared<int>(1, queue);
            TORCH_CHECK(moe_recv_rdma_counter_mapped != nullptr, "Failed to allocate XPU shared RDMA counter");
            moe_recv_rdma_counter = moe_recv_rdma_counter_mapped;
            *moe_recv_rdma_counter_mapped = -1;
        }
        *moe_recv_counter_mapped = -1;
        for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i) {
            moe_recv_expert_counter_mapped[i] = -1;
        }
    }

    ~Buffer() noexcept(false) {
        if (!explicitly_destroy) {
            destroy();
        } else if (!destroyed) {
            std::printf("WARNING: destroy() was not called before DeepEP XPU buffer destruction, which can leak resources.\n");
            std::fflush(stdout);
        }
    }

    bool is_available() const { return available; }

    int get_num_rdma_ranks() const { return num_rdma_ranks; }

    int get_rdma_rank() const { return rdma_rank; }

    int get_root_rdma_rank(bool global) const { return global ? nvl_rank : 0; }

    int get_local_device_id() const { return device_id; }

    pybind11::bytearray get_local_ipc_handle() const {
        if (num_nvl_bytes == 0) {
            return {};
        }
        if (!local_ipc_handle_valid) {
            void* base = nullptr;
            size_t allocation_size = 0;
            uint64_t fd = 0;
            ZE_CHECK(zeMemGetAddressRange(ze_context, buffer_ptrs[nvl_rank], &base, &allocation_size));
            ZE_CHECK(zeMemGetIpcHandle(ze_context, base, &local_ipc_handle.handle));
            if (num_nvl_ranks > 1) {
                ze_external_memory_export_fd_t export_fd = {};
                export_fd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD;
                export_fd.flags = kIpcExternalMemoryType;
                ze_memory_allocation_properties_t alloc_props = {};
                alloc_props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
                alloc_props.pNext = &export_fd;
                ZE_CHECK(zeMemGetAllocProperties(ze_context, base, &alloc_props, nullptr));
                fd = static_cast<uint64_t>(export_fd.fd);
            }
            local_ipc_base = base;
            local_ipc_handle.offset =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer_ptrs[nvl_rank]) - reinterpret_cast<uintptr_t>(base));
            local_ipc_handle.allocation_size = allocation_size;
            local_ipc_handle.requested_size = static_cast<uint64_t>(local_buffer.numel());
            local_ipc_handle.rank = rank;
            local_ipc_handle.device_id = device_id;
            local_ipc_handle.pid = static_cast<int>(getpid());
            local_ipc_handle.fd = num_nvl_ranks > 1 ? static_cast<int>(fd) : -1;
            local_ipc_handle_valid = true;
        }
        return {reinterpret_cast<const char*>(&local_ipc_handle), sizeof(local_ipc_handle)};
    }

    pybind11::bytearray get_local_nvshmem_unique_id() const {
        TORCH_CHECK(rdma_rank == 0, "Only XPU RDMA rank 0 can get an iSHMEM unique ID");
#ifdef DEEP_EP_ENABLE_ISHMEM
        const char* port_env = std::getenv("I_MPI_MPCP_SERVER_PORT");
        int base_port = port_env == nullptr || port_env[0] == '\0' ? 35555 : std::stoi(port_env);
        const char* master_port_env = std::getenv("MASTER_PORT");
        int master_port = master_port_env == nullptr || master_port_env[0] == '\0' ? -1 : std::stoi(master_port_env);
        int nvl_port = base_port + nvl_rank;
        if (nvl_port == master_port) {
            nvl_port += NUM_MAX_NVL_PEERS;
        }
        TORCH_CHECK(::setenv("I_MPI_MPCP_SERVER_PORT", std::to_string(nvl_port).c_str(), 1) == 0,
                    "failed to set I_MPI_MPCP_SERVER_PORT for XPU iSHMEM");
#endif
        auto unique_id = internode::get_unique_id();
        return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
    }

    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
        torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
        auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
        auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
        auto base = use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank];
        TORCH_CHECK(base != nullptr,
                    use_rdma_buffer ? "XPU RDMA buffer is not allocated" : "XPU local communication buffer is not allocated");
        TORCH_CHECK(offset >= 0 && offset <= num_bytes, "invalid XPU buffer offset");
        TORCH_CHECK((num_bytes - offset) % element_bytes == 0, "XPU buffer size is not divisible by dtype size");
        auto base_ptr = static_cast<uint8_t*>(base) + offset;
        return torch::from_blob(
            base_ptr, (num_bytes - offset) / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(c10::kXPU, device_id));
    }

    torch::Stream get_comm_stream() const { return comm_stream.unwrap(); }

    void sync(const std::vector<int>& device_ids,
              const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
              const std::optional<pybind11::bytearray>& root_unique_id_opt) {
        TORCH_CHECK(!is_available(), "XPU Buffer::sync called twice");

        if (num_nvl_bytes > 0) {
            TORCH_CHECK(static_cast<int>(device_ids.size()) == num_ranks, "device ID list size mismatch");
            TORCH_CHECK(static_cast<int>(all_gathered_handles.size()) == num_ranks, "IPC handle list size mismatch");

            const int offset = rdma_rank * num_nvl_ranks;
            for (int i = 0; i < num_nvl_ranks; ++i) {
                const int global_rank = offset + i;
                TORCH_CHECK(all_gathered_handles[global_rank].has_value(), "missing XPU IPC handle for rank ", global_rank);
                XpuIpcHandle handle = unpack_ipc_handle(*all_gathered_handles[global_rank]);
                TORCH_CHECK(handle.rank == global_rank, "XPU IPC handle rank mismatch");

                if (global_rank == rank) {
                    TORCH_CHECK(buffer_ptrs[i] == buffer_ptrs[nvl_rank], "local XPU buffer pointer mismatch");
                    barrier_signal_ptrs[i] = barrier_signal_ptrs[nvl_rank];
                } else {
                    TORCH_CHECK(
                        handle.fd >= 0, "XPU Level Zero IPC for rank ", global_rank, " is missing a received SCM_RIGHTS file descriptor");
                    void* imported_base = nullptr;
                    ze_external_memory_import_fd_t import_fd = {};
                    import_fd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
                    import_fd.flags = kIpcExternalMemoryType;
                    import_fd.fd = handle.fd;
                    ze_device_mem_alloc_desc_t import_desc = {};
                    import_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
                    import_desc.pNext = &import_fd;
                    ze_result_t import_result = zeMemAllocDevice(
                        ze_context, &import_desc, handle.allocation_size, NUM_BUFFER_ALIGNMENT_BYTES, ze_device, &imported_base);
                    ze_ipc_mem_handle_t ipc_handle = handle.handle;
                    ze_result_t ipc_result = ZE_RESULT_SUCCESS;
                    ze_result_t open_result = ZE_RESULT_SUCCESS;
                    if (import_result == ZE_RESULT_SUCCESS) {
                        imported_external_allocation_valid[i] = true;
                    } else {
                        ipc_result = zeMemGetIpcHandleFromFileDescriptorExp(ze_context, static_cast<uint64_t>(handle.fd), &ipc_handle);
                        open_result = zeMemOpenIpcHandle(ze_context, ze_device, ipc_handle, 0, &imported_base);
                        if (ipc_result == ZE_RESULT_SUCCESS && open_result != ZE_RESULT_SUCCESS) {
                            ZE_CHECK(zeMemPutIpcHandle(ze_context, ipc_handle));
                            ipc_handle = handle.handle;
                            open_result = zeMemOpenIpcHandle(ze_context, ze_device, ipc_handle, 0, &imported_base);
                        }
                    }
                    int close_result = ::close(handle.fd);
                    TORCH_CHECK(close_result == 0, "failed to close received XPU IPC file descriptor for rank ", global_rank);
                    TORCH_CHECK(import_result == ZE_RESULT_SUCCESS || open_result == ZE_RESULT_SUCCESS,
                                "Level Zero IPC import failed, zeMemAllocDevice import returned ",
                                static_cast<int>(import_result),
                                ", zeMemGetIpcHandleFromFileDescriptorExp returned ",
                                static_cast<int>(ipc_result),
                                " and zeMemOpenIpcHandle returned ",
                                static_cast<int>(open_result));
                    if (!imported_external_allocation_valid[i]) {
                        imported_ipc_handles[i] = ipc_handle;
                        imported_ipc_handle_valid[i] = true;
                    }
                    imported_ipc_bases[i] = imported_base;
                    buffer_ptrs[i] = static_cast<uint8_t*>(imported_base) + handle.offset;
                    barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
                }
            }

            auto& queue = comm_stream.queue();
            queue.memcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS);
            queue.memcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS);
            queue.wait();
        }
        if (num_rdma_bytes > 0) {
            TORCH_CHECK(root_unique_id_opt.has_value(), "missing XPU iSHMEM root unique ID");
            auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
            std::vector<uint8_t> root_unique_id(root_unique_id_str.size());
            std::memcpy(root_unique_id.data(), root_unique_id_str.data(), root_unique_id.size());
            const int ishmem_rank = low_latency_mode ? rank : rdma_rank;
            const int num_ishmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
            TORCH_CHECK(internode::init(root_unique_id, ishmem_rank, num_ishmem_ranks, low_latency_mode) == ishmem_rank,
                        "XPU iSHMEM initialized with an unexpected rank");
            internode::barrier();
            rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
            auto& queue = comm_stream.queue();
            queue.memset(rdma_buffer_ptr, 0, num_rdma_bytes).wait();
            internode::barrier();
        }
        available = true;
    }

    int reserve_barrier_signals(int count) {
        int base = barrier_signal_counter;
        barrier_signal_counter += count;
        return base;
    }

    void destroy() {
        if (destroyed) {
            return;
        }
        auto& queue = comm_stream.queue();
        queue.wait();

        if (num_nvl_bytes > 0 && available && barrier_signal_ptrs_gpu != nullptr) {
            intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, reserve_barrier_signals(1), queue);
            queue.wait();
        }
        if (rdma_buffer_ptr != nullptr) {
            queue.wait();
            internode::barrier();
            internode::free(rdma_buffer_ptr);
            rdma_buffer_ptr = nullptr;
            internode::finalize();
        }
        if (local_ipc_handle_valid) {
            if (local_ipc_handle.fd >= 0) {
                ::close(local_ipc_handle.fd);
                local_ipc_handle.fd = -1;
            }
            ZE_CHECK(zeMemPutIpcHandle(ze_context, local_ipc_handle.handle));
            local_ipc_handle_valid = false;
        }
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) {
            if (imported_ipc_bases[i] != nullptr) {
                if (imported_external_allocation_valid[i]) {
                    ZE_CHECK(zeMemFree(ze_context, imported_ipc_bases[i]));
                    imported_external_allocation_valid[i] = false;
                } else {
                    ZE_CHECK(zeMemCloseIpcHandle(ze_context, imported_ipc_bases[i]));
                }
                imported_ipc_bases[i] = nullptr;
            }
            if (imported_ipc_handle_valid[i]) {
                ZE_CHECK(zeMemPutIpcHandle(ze_context, imported_ipc_handles[i]));
                imported_ipc_handle_valid[i] = false;
            }
        }
        void* local_allocation = buffer_ptrs[nvl_rank];
        if (workspace != nullptr) {
            sycl::free(workspace, queue);
            workspace = nullptr;
        }
        if (moe_recv_counter_mapped != nullptr) {
            sycl::free(moe_recv_counter_mapped, queue);
            moe_recv_counter_mapped = nullptr;
            moe_recv_counter = nullptr;
        }
        if (moe_recv_rdma_counter_mapped != nullptr) {
            sycl::free(moe_recv_rdma_counter_mapped, queue);
            moe_recv_rdma_counter_mapped = nullptr;
            moe_recv_rdma_counter = nullptr;
        }
        if (moe_recv_expert_counter_mapped != nullptr) {
            sycl::free(moe_recv_expert_counter_mapped, queue);
            moe_recv_expert_counter_mapped = nullptr;
            moe_recv_expert_counter = nullptr;
        }
        local_buffer = torch::Tensor();
        if (local_allocation != nullptr) {
            ZE_CHECK(zeMemFree(ze_context, local_allocation));
        }
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) {
            buffer_ptrs[i] = nullptr;
            barrier_signal_ptrs[i] = nullptr;
        }
        buffer_ptrs_gpu = nullptr;
        barrier_signal_ptrs_gpu = nullptr;
        destroyed = true;
        available = false;
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>> get_dispatch_layout(
        const torch::Tensor& topk_idx,
        int num_experts,
        std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream) {
        check_xpu_tensor(topk_idx, "topk_idx");
        TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must be 2D");
        TORCH_CHECK(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value, "topk_idx has an unexpected dtype");
        TORCH_CHECK(num_experts > 0, "num_experts must be positive");

        auto compute_stream = c10::xpu::getCurrentXPUStream();
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else if (comm_stream != compute_stream) {
            stream_wait(comm_stream, compute_stream);
        }

        const int num_tokens = static_cast<int>(topk_idx.size(0));
        const int num_topk = static_cast<int>(topk_idx.size(1));
        auto int_options = topk_idx.options().dtype(torch::kInt32);
        auto bool_options = topk_idx.options().dtype(torch::kBool);
        auto num_tokens_per_rank = torch::empty({num_ranks}, int_options);
        auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
        if (num_rdma_ranks > 1) {
            num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, int_options);
        }
        auto num_tokens_per_expert = torch::empty({num_experts}, int_options);
        auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, bool_options);

        launch_get_dispatch_layout(topk_idx.data_ptr<topk_idx_t>(),
                                   num_tokens_per_rank.data_ptr<int>(),
                                   num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank->data_ptr<int>() : nullptr,
                                   num_tokens_per_expert.data_ptr<int>(),
                                   is_token_in_rank.data_ptr<bool>(),
                                   num_tokens,
                                   num_topk,
                                   num_ranks,
                                   num_experts,
                                   comm_stream.queue());

        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
        } else if (comm_stream != compute_stream) {
            stream_wait(compute_stream, comm_stream);
        }

        if (allocate_on_comm_stream) {
            c10::xpu::setCurrentXPUStream(compute_stream);
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
    intranode_dispatch(const torch::Tensor& x,
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
        TORCH_CHECK(is_available(), "XPU Buffer must be synced before intranode_dispatch");
        TORCH_CHECK(num_nvl_bytes > 0, "XPU intranode_dispatch requires a non-empty NVL buffer");
        TORCH_CHECK(num_ranks == num_nvl_ranks, "XPU intranode_dispatch supports intranode ranks only");
        bool cached_mode = cached_rank_prefix_matrix.has_value();
        TORCH_CHECK(config.num_sms % 2 == 0, "config.num_sms must be even");
        int num_channels = config.num_sms / 2;
        if (cached_mode) {
            TORCH_CHECK(cached_channel_prefix_matrix.has_value(), "cached channel prefix matrix is required in cached mode");
        } else {
            TORCH_CHECK(num_tokens_per_rank.has_value() && num_tokens_per_expert.has_value(),
                        "num_tokens_per_rank and num_tokens_per_expert are required in non-cached mode");
        }

        check_xpu_tensor(x, "x");
        check_xpu_tensor(is_token_in_rank, "is_token_in_rank");
        TORCH_CHECK(is_token_in_rank.scalar_type() == torch::kBool, "is_token_in_rank must be bool");
        TORCH_CHECK(x.dim() == 2, "x must be 2D");
        TORCH_CHECK((x.size(1) * x.element_size()) % sizeof(int4) == 0, "hidden bytes must be divisible by int4");
        TORCH_CHECK(is_token_in_rank.dim() == 2 && is_token_in_rank.size(0) == x.size(0) && is_token_in_rank.size(1) == num_ranks,
                    "is_token_in_rank shape mismatch");
        if (cached_mode) {
            check_xpu_tensor(*cached_rank_prefix_matrix, "cached_rank_prefix_matrix");
            check_xpu_tensor(*cached_channel_prefix_matrix, "cached_channel_prefix_matrix");
            TORCH_CHECK(
                cached_rank_prefix_matrix->scalar_type() == torch::kInt32 && cached_channel_prefix_matrix->scalar_type() == torch::kInt32,
                "cached prefix matrices must be int32");
        } else {
            check_xpu_tensor(*num_tokens_per_rank, "num_tokens_per_rank");
            check_xpu_tensor(*num_tokens_per_expert, "num_tokens_per_expert");
            TORCH_CHECK(num_tokens_per_rank->scalar_type() == torch::kInt32 && num_tokens_per_expert->scalar_type() == torch::kInt32,
                        "token count tensors must be int32");
        }

        auto num_tokens = static_cast<int>(x.size(0));
        auto hidden = static_cast<int>(x.size(1));
        auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0));
        auto num_local_experts = cached_mode ? 0 : num_experts / num_ranks;

        int num_topk = 0;
        topk_idx_t* topk_idx_ptr = nullptr;
        float* topk_weights_ptr = nullptr;
        TORCH_CHECK(topk_idx.has_value() == topk_weights.has_value(), "topk_idx and topk_weights must be both set or both unset");
        if (topk_idx.has_value()) {
            check_xpu_tensor(*topk_idx, "topk_idx");
            check_xpu_tensor(*topk_weights, "topk_weights");
            TORCH_CHECK(topk_idx->dim() == 2 && topk_weights->dim() == 2, "top-k tensors must be 2D");
            TORCH_CHECK(topk_idx->size(0) == num_tokens && topk_weights->size(0) == num_tokens, "top-k tensor shape mismatch");
            TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32, "topk_weights must be float32");
            num_topk = static_cast<int>(topk_idx->size(1));
            TORCH_CHECK(num_topk == topk_weights->size(1), "top-k width mismatch");
            topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        float* x_scales_ptr = nullptr;
        int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
        if (x_scales.has_value()) {
            check_xpu_tensor(*x_scales, "x_scales");
            TORCH_CHECK(x_scales->scalar_type() == torch::kFloat32 || x_scales->scalar_type() == torch::kInt32,
                        "x_scales must be float32 or int32");
            num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
            x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
            scale_token_stride = static_cast<int>(x_scales->stride(0));
            scale_hidden_stride = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->stride(1));
        }

        auto compute_stream = c10::xpu::getCurrentXPUStream();
        if (allocate_on_comm_stream) {
            TORCH_CHECK(previous_event.has_value() && async, "allocate_on_comm_stream requires previous_event and async");
            c10::xpu::setCurrentXPUStream(comm_stream);
        }
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else if (comm_stream != compute_stream) {
            stream_wait(comm_stream, compute_stream);
        }

        int num_recv_tokens = -1;
        auto rank_prefix_matrix = torch::Tensor();
        auto channel_prefix_matrix = torch::Tensor();
        std::vector<int> num_recv_tokens_per_expert_list;
        int num_memset_int = num_channels * num_ranks * 4;
        auto int_options = x.options().dtype(torch::kInt32);

        if (cached_mode) {
            num_recv_tokens = cached_num_recv_tokens;
            rank_prefix_matrix = cached_rank_prefix_matrix.value();
            channel_prefix_matrix = cached_channel_prefix_matrix.value();
            intranode::cached_notify_dispatch(rank_prefix_matrix.data_ptr<int>(),
                                              num_memset_int,
                                              buffer_ptrs_gpu,
                                              barrier_signal_ptrs_gpu,
                                              rank,
                                              num_ranks,
                                              reserve_barrier_signals(2),
                                              comm_stream.queue());
            comm_stream.queue().wait_and_throw();
        } else {
            TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
            TORCH_CHECK(num_local_experts <= NUM_MAX_LOCAL_EXPERTS, "too many local experts");
            rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, int_options);
            channel_prefix_matrix = torch::empty({num_ranks, num_channels}, int_options);
            *moe_recv_counter_mapped = -1;
            for (int i = 0; i < num_local_experts; ++i) {
                moe_recv_expert_counter_mapped[i] = -1;
            }
            TORCH_CHECK(num_ranks * (num_ranks + num_local_experts) * static_cast<int64_t>(sizeof(int)) <= num_nvl_bytes,
                        "XPU NVL buffer is too small for dispatch metadata");
            intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                       moe_recv_counter_mapped,
                                       num_ranks,
                                       num_tokens_per_expert->data_ptr<int>(),
                                       moe_recv_expert_counter_mapped,
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
                                       reserve_barrier_signals(3),
                                       comm_stream.queue(),
                                       num_channels);
            comm_stream.queue().wait_and_throw();
            if (num_worst_tokens > 0) {
                num_recv_tokens = num_worst_tokens;
                TORCH_CHECK(topk_idx.has_value() && topk_weights.has_value(), "num_worst_tokens requires top-k tensors");
            } else {
                auto start_time = std::chrono::high_resolution_clock::now();
                while (true) {
                    num_recv_tokens = *moe_recv_counter_mapped;
                    bool ready = (num_recv_tokens >= 0);
                    for (int i = 0; i < num_local_experts && ready; ++i) {
                        ready &= moe_recv_expert_counter_mapped[i] >= 0;
                    }
                    if (ready) {
                        break;
                    }
                    TORCH_CHECK(
                        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() <=
                            NUM_CPU_TIMEOUT_SECS,
                        "DeepEP XPU error: CPU recv timeout");
                }
                num_recv_tokens_per_expert_list =
                    std::vector<int>(moe_recv_expert_counter_mapped, moe_recv_expert_counter_mapped + num_local_experts);
            }
        }

        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
        auto recv_src_idx = torch::empty({num_recv_tokens}, int_options);
        auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, int_options);
        auto send_head = torch::empty({num_tokens, num_ranks}, int_options);
        std::optional<torch::Tensor> recv_topk_idx, recv_topk_weights, recv_x_scales;
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

        TORCH_CHECK(
            num_ranks * num_ranks * static_cast<int64_t>(sizeof(int)) + num_channels * num_ranks * static_cast<int64_t>(sizeof(int)) * 4 +
                    num_channels * num_ranks * static_cast<int64_t>(config.num_max_nvl_chunked_recv_tokens) * hidden *
                        recv_x.element_size() +
                    num_channels * num_ranks * static_cast<int64_t>(config.num_max_nvl_chunked_recv_tokens) * sizeof(int) +
                    num_channels * num_ranks * static_cast<int64_t>(config.num_max_nvl_chunked_recv_tokens) * num_topk *
                        sizeof(topk_idx_t) +
                    num_channels * num_ranks * static_cast<int64_t>(config.num_max_nvl_chunked_recv_tokens) * num_topk * sizeof(float) +
                    num_channels * num_ranks * static_cast<int64_t>(config.num_max_nvl_chunked_recv_tokens) * num_scales * sizeof(float) <=
                num_nvl_bytes,
            "XPU NVL buffer is too small for dispatch");
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
                            comm_stream.queue(),
                            config.num_sms,
                            config.num_max_nvl_chunked_send_tokens,
                            config.num_max_nvl_chunked_recv_tokens);

        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            auto stream = comm_stream.unwrap();
            for (auto& t : {x,
                            is_token_in_rank,
                            rank_prefix_matrix,
                            channel_prefix_matrix,
                            recv_x,
                            recv_src_idx,
                            recv_channel_prefix_matrix,
                            send_head}) {
                t.record_stream(stream);
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
                if (to.has_value()) {
                    to->record_stream(stream);
                }
            }
        } else if (comm_stream != compute_stream) {
            stream_wait(compute_stream, comm_stream);
        }
        if (allocate_on_comm_stream) {
            c10::xpu::setCurrentXPUStream(compute_stream);
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

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> intranode_combine(
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
        TORCH_CHECK(is_available(), "XPU Buffer must be synced before intranode_combine");
        check_xpu_tensor(x, "x");
        check_xpu_tensor(src_idx, "src_idx");
        check_xpu_tensor(rank_prefix_matrix, "rank_prefix_matrix");
        check_xpu_tensor(channel_prefix_matrix, "channel_prefix_matrix");
        check_xpu_tensor(send_head, "send_head");
        TORCH_CHECK(src_idx.scalar_type() == torch::kInt32 && rank_prefix_matrix.scalar_type() == torch::kInt32 &&
                        channel_prefix_matrix.scalar_type() == torch::kInt32 && send_head.scalar_type() == torch::kInt32,
                    "combine metadata tensors must be int32");
        TORCH_CHECK(config.num_sms % 2 == 0, "config.num_sms must be even");
        int num_channels = config.num_sms / 2;
        auto num_tokens = static_cast<int>(x.size(0));
        auto hidden = static_cast<int>(x.size(1));
        auto num_recv_tokens = static_cast<int>(send_head.size(0));
        TORCH_CHECK(src_idx.size(0) == num_tokens && send_head.size(1) == num_ranks, "combine metadata shape mismatch");

        auto compute_stream = c10::xpu::getCurrentXPUStream();
        if (allocate_on_comm_stream) {
            TORCH_CHECK(previous_event.has_value() && async, "allocate_on_comm_stream requires previous_event and async");
            c10::xpu::setCurrentXPUStream(comm_stream);
        }
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else if (comm_stream != compute_stream) {
            stream_wait(comm_stream, compute_stream);
        }

        int num_topk = 0;
        std::optional<torch::Tensor> recv_topk_weights;
        float* topk_weights_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            check_xpu_tensor(*topk_weights, "topk_weights");
            TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32 && topk_weights->dim() == 2, "topk_weights must be 2D float32");
            num_topk = static_cast<int>(topk_weights->size(1));
            topk_weights_ptr = topk_weights->data_ptr<float>();
            recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
        }

        TORCH_CHECK(num_channels * num_ranks * static_cast<int64_t>(sizeof(int)) * 2 <= num_nvl_bytes,
                    "XPU NVL buffer is too small for combine metadata");
        intranode::cached_notify_combine(buffer_ptrs_gpu,
                                         send_head.data_ptr<int>(),
                                         num_channels,
                                         num_recv_tokens,
                                         num_channels * num_ranks * 2,
                                         barrier_signal_ptrs_gpu,
                                         rank,
                                         num_ranks,
                                         reserve_barrier_signals(2),
                                         comm_stream.queue());
        comm_stream.queue().wait_and_throw();

        void* bias_ptrs[2] = {nullptr, nullptr};
        if (bias_0.has_value()) {
            check_xpu_tensor(*bias_0, "bias_0");
            bias_ptrs[0] = bias_0->data_ptr();
        }
        if (bias_1.has_value()) {
            check_xpu_tensor(*bias_1, "bias_1");
            bias_ptrs[1] = bias_1->data_ptr();
        }

        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
        TORCH_CHECK(
            num_channels * num_ranks * static_cast<int64_t>(sizeof(int)) * 2 +
                    num_channels * num_ranks * static_cast<int64_t>(config.num_max_nvl_chunked_recv_tokens) * hidden * x.element_size() +
                    num_channels * num_ranks * static_cast<int64_t>(config.num_max_nvl_chunked_recv_tokens) * sizeof(int) +
                    num_channels * num_ranks * static_cast<int64_t>(config.num_max_nvl_chunked_recv_tokens) * num_topk * sizeof(float) <=
                num_nvl_bytes,
            "XPU NVL buffer is too small for combine");
        intranode::combine(scalar_type_to_data_type(x.scalar_type()),
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
                           comm_stream.queue(),
                           config.num_sms,
                           config.num_max_nvl_chunked_send_tokens,
                           config.num_max_nvl_chunked_recv_tokens);

        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            auto stream = comm_stream.unwrap();
            for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
                t.record_stream(stream);
            }
            for (auto& to : {topk_weights, recv_topk_weights, bias_0, bias_1}) {
                if (to.has_value()) {
                    to->record_stream(stream);
                }
            }
        } else if (comm_stream != compute_stream) {
            stream_wait(compute_stream, comm_stream);
        }
        if (allocate_on_comm_stream) {
            c10::xpu::setCurrentXPUStream(compute_stream);
        }
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
    internode_dispatch(const torch::Tensor& x,
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
                       int,
                       int num_worst_tokens,
                       const Config& config,
                       std::optional<EventHandle>& previous_event,
                       bool async,
                       bool allocate_on_comm_stream) {
        pybind11::gil_scoped_release release;
        TORCH_CHECK(is_available(), "XPU Buffer must be synced before internode_dispatch");
        TORCH_CHECK(!low_latency_mode, "XPU internode_dispatch requires a high-throughput buffer, not a low-latency buffer");
        TORCH_CHECK(num_rdma_bytes > 0 && rdma_buffer_ptr != nullptr, "XPU internode_dispatch requires an iSHMEM RDMA buffer");
        TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "XPU internode_dispatch currently supports BF16 tensors only");
        TORCH_CHECK(!x_scales.has_value(), "XPU internode_dispatch FP8/x_scales path is not migrated yet");
        TORCH_CHECK(!topk_idx.has_value() && !topk_weights.has_value(), "XPU internode_dispatch top-k return path is not migrated yet");
        check_xpu_tensor(x, "x");
        check_xpu_tensor(is_token_in_rank, "is_token_in_rank");
        TORCH_CHECK(config.num_sms % 2 == 0, "config.num_sms must be even");
        bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
        if (cached_mode) {
            TORCH_CHECK(cached_recv_rdma_rank_prefix_sum.has_value() && cached_gbl_channel_prefix_matrix.has_value() &&
                            cached_recv_gbl_rank_prefix_sum.has_value(),
                        "cached internode dispatch requires all cached prefix tensors");
        } else {
            TORCH_CHECK(num_tokens_per_rank.has_value() && num_tokens_per_rdma_rank.has_value() && num_tokens_per_expert.has_value(),
                        "non-cached internode dispatch requires token count tensors");
            TORCH_CHECK(num_worst_tokens > 0,
                        "XPU internode_dispatch non-cached metadata exchange is not complete; pass num_worst_tokens for the "
                        "correctness-first path");
        }

        auto compute_stream = c10::xpu::getCurrentXPUStream();
        if (allocate_on_comm_stream) {
            TORCH_CHECK(previous_event.has_value() && async, "allocate_on_comm_stream requires previous_event and async");
            c10::xpu::setCurrentXPUStream(comm_stream);
        }
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else if (comm_stream != compute_stream) {
            stream_wait(comm_stream, compute_stream);
        }

        const int num_tokens = static_cast<int>(x.size(0));
        const int hidden = static_cast<int>(x.size(1));
        const int num_channels = config.num_sms / 2;
        const int num_recv_tokens = cached_mode ? cached_num_recv_tokens : num_worst_tokens;
        const int num_rdma_recv_tokens = cached_mode ? cached_num_rdma_recv_tokens : num_worst_tokens;
        auto int_options = x.options().dtype(torch::kInt32);
        auto byte_options = x.options().dtype(torch::kUInt8);

        torch::Tensor rdma_channel_prefix_matrix =
            cached_mode ? cached_rdma_channel_prefix_matrix.value() : torch::zeros({num_rdma_ranks, num_channels}, int_options);
        torch::Tensor recv_rdma_rank_prefix_sum =
            cached_mode ? cached_recv_rdma_rank_prefix_sum.value() : torch::zeros({num_rdma_ranks}, int_options);
        torch::Tensor gbl_channel_prefix_matrix =
            cached_mode ? cached_gbl_channel_prefix_matrix.value() : torch::zeros({num_ranks, num_channels}, int_options);
        torch::Tensor recv_gbl_rank_prefix_sum =
            cached_mode ? cached_recv_gbl_rank_prefix_sum.value() : torch::zeros({num_ranks}, int_options);

        auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
        auto recv_src_meta = cached_mode
            ? std::optional<torch::Tensor>()
            : std::optional<torch::Tensor>(torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, byte_options));
        auto recv_rdma_channel_prefix_matrix = cached_mode
            ? std::optional<torch::Tensor>()
            : std::optional<torch::Tensor>(torch::empty({num_rdma_ranks, num_channels}, int_options));
        auto recv_gbl_channel_prefix_matrix = cached_mode
            ? std::optional<torch::Tensor>()
            : std::optional<torch::Tensor>(torch::empty({num_ranks, num_channels}, int_options));
        auto send_rdma_head = cached_mode ? std::optional<torch::Tensor>()
                                          : std::optional<torch::Tensor>(torch::empty({num_tokens, num_rdma_ranks}, int_options));
        auto send_nvl_head = cached_mode
            ? std::optional<torch::Tensor>()
            : std::optional<torch::Tensor>(torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, int_options));

        const size_t copy_rows = static_cast<size_t>(std::min(num_tokens, num_recv_tokens));
        if (copy_rows > 0) {
            comm_stream.queue().memcpy(recv_x.data_ptr(), x.data_ptr(), copy_rows * hidden * x.element_size());
        }
        internode::dispatch(recv_x.data_ptr(),
                            nullptr,
                            nullptr,
                            nullptr,
                            cached_mode ? nullptr : recv_src_meta->data_ptr(),
                            x.data_ptr(),
                            nullptr,
                            nullptr,
                            nullptr,
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
                            num_recv_tokens,
                            hidden,
                            0,
                            rank,
                            num_ranks,
                            comm_stream.queue());

        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            auto stream = comm_stream.unwrap();
            for (auto& t : {x,
                            is_token_in_rank,
                            recv_x,
                            rdma_channel_prefix_matrix,
                            recv_rdma_rank_prefix_sum,
                            gbl_channel_prefix_matrix,
                            recv_gbl_rank_prefix_sum}) {
                t.record_stream(stream);
            }
            for (auto& to : {num_tokens_per_rank,
                             num_tokens_per_rdma_rank,
                             num_tokens_per_expert,
                             cached_rdma_channel_prefix_matrix,
                             cached_recv_rdma_rank_prefix_sum,
                             cached_gbl_channel_prefix_matrix,
                             cached_recv_gbl_rank_prefix_sum,
                             recv_rdma_channel_prefix_matrix,
                             recv_gbl_channel_prefix_matrix,
                             send_rdma_head,
                             send_nvl_head,
                             recv_src_meta}) {
                if (to.has_value()) {
                    to->record_stream(stream);
                }
            }
        } else if (comm_stream != compute_stream) {
            stream_wait(compute_stream, comm_stream);
        }
        if (allocate_on_comm_stream) {
            c10::xpu::setCurrentXPUStream(compute_stream);
        }

        return {recv_x,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                {},
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
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> internode_combine(
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
        TORCH_CHECK(is_available(), "XPU Buffer must be synced before internode_combine");
        TORCH_CHECK(!low_latency_mode, "XPU internode_combine requires a high-throughput buffer, not a low-latency buffer");
        TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "XPU internode_combine currently supports BF16 tensors only");
        check_xpu_tensor(x, "x");
        check_xpu_tensor(src_meta, "src_meta");
        check_xpu_tensor(is_combined_token_in_rank, "is_combined_token_in_rank");
        check_xpu_tensor(rdma_channel_prefix_matrix, "rdma_channel_prefix_matrix");
        check_xpu_tensor(rdma_rank_prefix_sum, "rdma_rank_prefix_sum");
        check_xpu_tensor(gbl_channel_prefix_matrix, "gbl_channel_prefix_matrix");
        check_xpu_tensor(combined_rdma_head, "combined_rdma_head");
        check_xpu_tensor(combined_nvl_head, "combined_nvl_head");
        TORCH_CHECK(config.num_sms % 2 == 0, "config.num_sms must be even");
        TORCH_CHECK(src_meta.size(1) == internode::get_source_meta_bytes(), "src_meta shape mismatch");

        auto compute_stream = c10::xpu::getCurrentXPUStream();
        if (allocate_on_comm_stream) {
            TORCH_CHECK(previous_event.has_value() && async, "allocate_on_comm_stream requires previous_event and async");
            c10::xpu::setCurrentXPUStream(comm_stream);
        }
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else if (comm_stream != compute_stream) {
            stream_wait(comm_stream, compute_stream);
        }

        const int num_tokens = static_cast<int>(x.size(0));
        const int hidden = static_cast<int>(x.size(1));
        const int num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
        int num_topk = 0;
        float* topk_weights_ptr = nullptr;
        std::optional<torch::Tensor> combined_topk_weights;
        float* combined_topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            check_xpu_tensor(*topk_weights, "topk_weights");
            TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32 && topk_weights->dim() == 2, "topk_weights must be 2D float32");
            num_topk = static_cast<int>(topk_weights->size(1));
            topk_weights_ptr = topk_weights->data_ptr<float>();
            combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
            combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
        }
        void* bias_ptrs[2] = {nullptr, nullptr};
        if (bias_0.has_value()) {
            check_xpu_tensor(*bias_0, "bias_0");
            TORCH_CHECK(bias_0->scalar_type() == x.scalar_type(), "bias_0 dtype mismatch");
            bias_ptrs[0] = bias_0->data_ptr();
        }
        if (bias_1.has_value()) {
            check_xpu_tensor(*bias_1, "bias_1");
            TORCH_CHECK(bias_1->scalar_type() == x.scalar_type(), "bias_1 dtype mismatch");
            bias_ptrs[1] = bias_1->data_ptr();
        }

        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
        internode::combine(scalar_type_to_data_type(x.scalar_type()),
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
                           rank,
                           num_ranks,
                           comm_stream.queue());

        std::optional<EventHandle> event;
        if (async) {
            event = EventHandle(comm_stream);
            auto stream = comm_stream.unwrap();
            for (auto& t : {x,
                            src_meta,
                            is_combined_token_in_rank,
                            rdma_channel_prefix_matrix,
                            rdma_rank_prefix_sum,
                            gbl_channel_prefix_matrix,
                            combined_rdma_head,
                            combined_nvl_head,
                            combined_x}) {
                t.record_stream(stream);
            }
            for (auto& to : {topk_weights, combined_topk_weights, bias_0, bias_1}) {
                if (to.has_value()) {
                    to->record_stream(stream);
                }
            }
        } else if (comm_stream != compute_stream) {
            stream_wait(compute_stream, comm_stream);
        }
        if (allocate_on_comm_stream) {
            c10::xpu::setCurrentXPUStream(compute_stream);
        }
        return {combined_x, combined_topk_weights, event};
    }

    void configure_low_latency_layout(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
        TORCH_CHECK(low_latency_mode, "XPU low-latency APIs require a low-latency buffer");
        TORCH_CHECK(rdma_buffer_ptr != nullptr, "XPU low-latency APIs require an RDMA/iSHMEM buffer");
        auto layout = get_low_latency_buffer_layout(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
        TORCH_CHECK(layout.total_bytes <= static_cast<size_t>(num_rdma_bytes),
                    "XPU RDMA buffer is too small for low-latency mode: need ",
                    layout.total_bytes,
                    " bytes, got ",
                    num_rdma_bytes);
        low_latency_mask_buffer_ptr = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_buffer_ptr) + layout.mask_offset);
        low_latency_sync_buffer_ptr = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_buffer_ptr) + layout.sync_offset);
        low_latency_num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank;
        low_latency_hidden = hidden;
        low_latency_num_experts = num_experts;
    }

    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
        configure_low_latency_layout(num_max_dispatch_tokens_per_rank, hidden, num_experts);
        internode_ll::clean_low_latency_buffer(static_cast<int*>(rdma_buffer_ptr),
                                               static_cast<int>(num_rdma_bytes / sizeof(int)),
                                               nullptr,
                                               0,
                                               rank,
                                               num_ranks,
                                               low_latency_mask_buffer_ptr,
                                               low_latency_sync_buffer_ptr,
                                               comm_stream.queue());
        comm_stream.queue().wait_and_throw();
    }

    void low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
        TORCH_CHECK(low_latency_mask_buffer_ptr != nullptr,
                    "low-latency buffer layout is not configured; call clean_low_latency_buffer first");
        TORCH_CHECK(rank_to_mask >= 0 && rank_to_mask < num_ranks, "rank_to_mask out of range");
        internode_ll::update_mask_buffer(low_latency_mask_buffer_ptr, rank_to_mask, mask, comm_stream.queue());
    }

    void low_latency_query_mask_buffer(const torch::Tensor& output_mask_tensor) {
        TORCH_CHECK(low_latency_mask_buffer_ptr != nullptr,
                    "low-latency buffer layout is not configured; call clean_low_latency_buffer first");
        check_xpu_tensor(output_mask_tensor, "output_mask_tensor");
        TORCH_CHECK(output_mask_tensor.scalar_type() == torch::kInt32 && output_mask_tensor.numel() >= num_ranks,
                    "output_mask_tensor must be int32 with at least num_ranks elements");
        internode_ll::query_mask_buffer(low_latency_mask_buffer_ptr, num_ranks, output_mask_tensor.data_ptr<int>(), comm_stream.queue());
    }

    void low_latency_clean_mask_buffer() {
        TORCH_CHECK(low_latency_mask_buffer_ptr != nullptr,
                    "low-latency buffer layout is not configured; call clean_low_latency_buffer first");
        internode_ll::clean_mask_buffer(low_latency_mask_buffer_ptr, num_ranks, comm_stream.queue());
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, EventHandle, pybind11::object>
    low_latency_dispatch(const torch::Tensor& x,
                         const torch::Tensor& topk_idx,
                         const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                         const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                         int num_max_dispatch_tokens_per_rank,
                         int num_experts,
                         bool use_fp8,
                         bool,
                         bool use_ue8m0,
                         bool async_finish,
                         bool return_recv_hook) {
        TORCH_CHECK(is_available(), "XPU Buffer must be synced before low_latency_dispatch");
        TORCH_CHECK(low_latency_mode, "low_latency_dispatch requires a low-latency buffer");
        TORCH_CHECK(!use_fp8 && !use_ue8m0, "XPU kernel-backed low_latency_dispatch currently supports BF16 output only");
        TORCH_CHECK(!(async_finish && return_recv_hook), "async_finish and return_recv_hook cannot both be true");
        check_xpu_tensor(x, "x");
        check_xpu_tensor(topk_idx, "topk_idx");
        TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
        TORCH_CHECK(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value, "topk_idx dtype mismatch");
        TORCH_CHECK(x.dim() == 2 && topk_idx.dim() == 2 && topk_idx.size(0) == x.size(0), "low-latency dispatch shape mismatch");
        if (cumulative_local_expert_recv_stats.has_value()) {
            check_xpu_tensor(*cumulative_local_expert_recv_stats, "cumulative_local_expert_recv_stats");
            TORCH_CHECK(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt32, "stats must be int32");
        }
        if (dispatch_wait_recv_cost_stats.has_value()) {
            check_xpu_tensor(*dispatch_wait_recv_cost_stats, "dispatch_wait_recv_cost_stats");
            TORCH_CHECK(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64, "wait stats must be int64");
        }
        const int num_tokens = static_cast<int>(x.size(0));
        const int hidden = static_cast<int>(x.size(1));
        const int num_topk = static_cast<int>(topk_idx.size(1));
        const int num_local_experts = num_experts / num_ranks;
        configure_low_latency_layout(num_max_dispatch_tokens_per_rank, hidden, num_experts);

        auto compute_stream = c10::xpu::getCurrentXPUStream();
        if (comm_stream != compute_stream) {
            stream_wait(comm_stream, compute_stream);
        }
        auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden}, x.options());
        auto int_options = x.options().dtype(torch::kInt32);
        auto long_options = x.options().dtype(torch::kInt64);
        auto packed_recv_count = torch::empty({num_local_experts}, int_options);
        auto packed_recv_src_info = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, int_options);
        auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, long_options);

        internode_ll::dispatch_bf16(
            packed_recv_x.data_ptr(),
            packed_recv_src_info.data_ptr<int>(),
            packed_recv_layout_range.data_ptr<int64_t>(),
            packed_recv_count.data_ptr<int>(),
            cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
            dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
            rdma_buffer_ptr,
            low_latency_mask_buffer_ptr,
            x.data_ptr(),
            topk_idx.data_ptr<topk_idx_t>(),
            num_tokens,
            hidden,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            num_experts,
            rank,
            num_ranks,
            comm_stream.queue());

        EventHandle event(comm_stream);
        if (!async_finish && comm_stream != compute_stream) {
            stream_wait(compute_stream, comm_stream);
        }
        pybind11::object hook = pybind11::none();
        if (return_recv_hook) {
            hook = pybind11::cpp_function([]() {});
        }
        return {packed_recv_x, std::nullopt, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, hook};
    }

    std::tuple<torch::Tensor, EventHandle, pybind11::object> low_latency_combine(
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
        bool async_finish,
        bool return_recv_hook,
        const std::optional<torch::Tensor>& out) {
        TORCH_CHECK(is_available(), "XPU Buffer must be synced before low_latency_combine");
        TORCH_CHECK(low_latency_mode, "low_latency_combine requires a low-latency buffer");
        TORCH_CHECK(!use_logfmt, "XPU kernel-backed low_latency_combine does not support LogFMT");
        TORCH_CHECK(!(async_finish && return_recv_hook), "async_finish and return_recv_hook cannot both be true");
        check_xpu_tensor(x, "x");
        check_xpu_tensor(topk_idx, "topk_idx");
        check_xpu_tensor(topk_weights, "topk_weights");
        check_xpu_tensor(src_info, "src_info");
        check_xpu_tensor(layout_range, "layout_range");
        TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
        TORCH_CHECK(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value, "topk_idx dtype mismatch");
        TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32, "topk_weights must be float32");
        TORCH_CHECK(src_info.scalar_type() == torch::kInt32 && layout_range.scalar_type() == torch::kInt64,
                    "combine handle tensors have invalid dtypes");
        if (combine_wait_recv_cost_stats.has_value()) {
            check_xpu_tensor(*combine_wait_recv_cost_stats, "combine_wait_recv_cost_stats");
            TORCH_CHECK(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64, "wait stats must be int64");
        }
        const int hidden = static_cast<int>(x.size(2));
        const int num_combined_tokens = static_cast<int>(topk_idx.size(0));
        const int num_topk = static_cast<int>(topk_idx.size(1));
        configure_low_latency_layout(num_max_dispatch_tokens_per_rank, hidden, num_experts);

        auto compute_stream = c10::xpu::getCurrentXPUStream();
        if (comm_stream != compute_stream) {
            stream_wait(comm_stream, compute_stream);
        }
        torch::Tensor combined_x;
        if (out.has_value()) {
            check_xpu_tensor(*out, "out");
            TORCH_CHECK(
                out->scalar_type() == x.scalar_type() && out->dim() == 2 && out->size(0) == num_combined_tokens && out->size(1) == hidden,
                "out shape or dtype mismatch");
            combined_x = *out;
        } else {
            combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
        }
        internode_ll::combine_bf16(combined_x.data_ptr(),
                                   rdma_buffer_ptr,
                                   low_latency_mask_buffer_ptr,
                                   x.data_ptr(),
                                   topk_idx.data_ptr<topk_idx_t>(),
                                   topk_weights.data_ptr<float>(),
                                   src_info.data_ptr<int>(),
                                   layout_range.data_ptr<int64_t>(),
                                   combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                                   num_combined_tokens,
                                   hidden,
                                   num_max_dispatch_tokens_per_rank,
                                   num_topk,
                                   num_experts,
                                   rank,
                                   num_ranks,
                                   comm_stream.queue(),
                                   zero_copy);
        EventHandle event(comm_stream);
        if (!async_finish && comm_stream != compute_stream) {
            stream_wait(compute_stream, comm_stream);
        }
        pybind11::object hook = pybind11::none();
        if (return_recv_hook) {
            hook = pybind11::cpp_function([]() {});
        }
        return {combined_x, event, hook};
    }

    torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
        configure_low_latency_layout(num_max_dispatch_tokens_per_rank, hidden, num_experts);
        TORCH_CHECK(
            false,
            "XPU get_next_low_latency_combine_buffer zero-copy path is not implemented; call low_latency_combine with zero_copy=False");
    }
};

bool is_sm90_compiled() {
    return false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             pybind11::arg("num_sms"),
             pybind11::arg("num_max_nvl_chunked_send_tokens"),
             pybind11::arg("num_max_nvl_chunked_recv_tokens"),
             pybind11::arg("num_max_rdma_chunked_send_tokens") = 6,
             pybind11::arg("num_max_rdma_chunked_recv_tokens") = 128)
        .def("get_nvl_buffer_size_hint", &Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &get_low_latency_rdma_size_hint);
    m.def("_xpu_get_ipc_handle_fd", [](const pybind11::bytearray& handle_bytes) { return unpack_ipc_handle(handle_bytes).fd; });
    m.def("_xpu_set_ipc_handle_fd", [](const pybind11::bytearray& handle_bytes, int fd) {
        XpuIpcHandle handle = unpack_ipc_handle(handle_bytes);
        handle.fd = fd;
        return pybind11::bytearray(reinterpret_cast<const char*>(&handle), sizeof(handle));
    });

    pybind11::class_<EventHandle>(m, "EventHandle").def(pybind11::init<>()).def("current_stream_wait", &EventHandle::current_stream_wait);

    pybind11::class_<Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())
        .def("is_available", &Buffer::is_available)
        .def("get_num_rdma_ranks", &Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &Buffer::get_comm_stream)
        .def("sync", &Buffer::sync)
        .def("destroy", &Buffer::destroy)
        .def("get_dispatch_layout", &Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &Buffer::intranode_dispatch)
        .def("intranode_combine", &Buffer::intranode_combine)
        .def("internode_dispatch", &Buffer::internode_dispatch)
        .def("internode_combine", &Buffer::internode_combine)
        .def("clean_low_latency_buffer", &Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &Buffer::low_latency_dispatch)
        .def("low_latency_combine", &Buffer::low_latency_combine)
        .def("low_latency_update_mask_buffer", &Buffer::low_latency_update_mask_buffer)
        .def("low_latency_query_mask_buffer", &Buffer::low_latency_query_mask_buffer)
        .def("low_latency_clean_mask_buffer", &Buffer::low_latency_clean_mask_buffer)
        .def("get_next_low_latency_combine_buffer", &Buffer::get_next_low_latency_combine_buffer);

    m.def("is_sm90_compiled", is_sm90_compiled);
    m.attr("topk_idx_t") = py::reinterpret_borrow<py::object>((PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<topk_idx_t>::value));
}

}  // namespace deep_ep
