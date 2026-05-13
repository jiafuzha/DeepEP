#include "api.cuh"
#include "utils.cuh"

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace deep_ep {

namespace internode_ll {

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor, cudaStream_t stream) {
    auto& queue = c10::xpu::XPUStream(stream).queue();
    constexpr int kNumThreads = 1024;
    const size_t global_size = static_cast<size_t>(align_up(num_ranks, kNumThreads));
    queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kNumThreads)),
                       [=](sycl::nd_item<1> item) {
                           const int idx = static_cast<int>(item.get_global_linear_id());
                           if (idx < num_ranks) {
                               mask_tensor[idx] = mask_buffer_ptr[idx];
                           }
                       });
}

void update_mask_buffer(int* mask_buffer_ptr, int rank, bool mask, cudaStream_t stream) {
    auto& queue = c10::xpu::XPUStream(stream).queue();
    queue.single_task([=]() {
        mask_buffer_ptr[rank] = mask ? 1 : 0;
    });
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream) {
    auto& queue = c10::xpu::XPUStream(stream).queue();
    constexpr int kNumThreads = 32;
    const size_t global_size = static_cast<size_t>(align_up(num_ranks, kNumThreads));
    queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kNumThreads)),
                       [=](sycl::nd_item<1> item) {
                           const int idx = static_cast<int>(item.get_global_linear_id());
                           if (idx < num_ranks) {
                               mask_buffer_ptr[idx] = 0;
                           }
                       });
}

}  // namespace internode_ll

}  // namespace deep_ep
