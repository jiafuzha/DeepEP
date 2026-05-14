#include "api.cuh"
#include "utils.cuh"

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace deep_ep {

namespace internode_ll {

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor, cudaStream_t stream) {
    auto& queue = c10::xpu::XPUStream(stream).queue();
    queue.memcpy(mask_tensor, mask_buffer_ptr, static_cast<size_t>(num_ranks) * sizeof(int));
}

void update_mask_buffer(int* mask_buffer_ptr, int rank, bool mask, cudaStream_t stream) {
    auto& queue = c10::xpu::XPUStream(stream).queue();
    queue.fill(mask_buffer_ptr + rank, mask ? 1 : 0, 1);
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream) {
    auto& queue = c10::xpu::XPUStream(stream).queue();
    queue.memset(mask_buffer_ptr, 0, static_cast<size_t>(num_ranks) * sizeof(int));
}

}  // namespace internode_ll

}  // namespace deep_ep
