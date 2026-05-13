#include <stdexcept>

#include "kernels/api.cuh"

namespace {

[[noreturn]] void throw_xpu_unported(const char* name) {
    throw std::runtime_error(std::string("DeepEP XPU native kernel is not implemented yet: ") + name);
}

}  // namespace

namespace deep_ep {

namespace intranode {

void barrier(int**, int, int, cudaStream_t) { throw_xpu_unported("intranode::barrier"); }

void notify_dispatch(const int*, int*, int, const int*, int*, int, int, const bool*, int*, int*, int, int, void**, int**, int, cudaStream_t, int) {
    throw_xpu_unported("intranode::notify_dispatch");
}

void cached_notify_dispatch(const int*, int, void**, int**, int, int, cudaStream_t) {
    throw_xpu_unported("intranode::cached_notify_dispatch");
}

void dispatch(void*, float*, int*, topk_idx_t*, float*, int*, int*, const void*, const float*, const topk_idx_t*, const float*, const bool*,
              const int*, int, int, int, int, int, int, int, int, void**, int, int, cudaStream_t, int, int, int) {
    throw_xpu_unported("intranode::dispatch");
}

void cached_notify_combine(void**, int*, int, int, int, int**, int, int, cudaStream_t) {
    throw_xpu_unported("intranode::cached_notify_combine");
}

void combine(cudaDataType_t, void*, float*, const void*, const float*, const void*, const void*, const int*, const int*, const int*, int*,
             int, int, int, int, void**, int, int, cudaStream_t, int, int, int) {
    throw_xpu_unported("intranode::combine");
}

}  // namespace intranode

namespace internode {

int get_source_meta_bytes() {
    return 2 * static_cast<int>(sizeof(int));
}

void notify_dispatch(const int*, int*, int, const int*, int*, const int*, int*, int, const bool*, int, int, int, int, int, int, int*, int*,
                     int*, int*, void*, int, void**, int, int**, int, cudaStream_t, int64_t, int64_t, bool) {
    throw_xpu_unported("internode::notify_dispatch");
}

void dispatch(void*, float*, topk_idx_t*, float*, void*, const void*, const float*, const topk_idx_t*, const float*, int*, int*, int*, int*,
              const int*, const int*, const int*, const int*, const bool*, int, int, int, int, int, int, int, int, void*, int, int, void**,
              int, int, int, int, bool, cudaStream_t, int, bool) {
    throw_xpu_unported("internode::dispatch");
}

void cached_notify(int, int, int, int, int, int, int, int*, const int*, const int*, int*, void*, int, void**, int, int**, int, cudaStream_t,
                   int64_t, int64_t, bool, bool) {
    throw_xpu_unported("internode::cached_notify");
}

void combine(cudaDataType_t, void*, float*, const bool*, const void*, const float*, const void*, const void*, const int*, const int*, const void*,
             const int*, const int*, const int*, int, int, int, int, void*, int, int, void**, int, int, int, int, cudaStream_t, int, bool) {
    throw_xpu_unported("internode::combine");
}

}  // namespace internode

namespace internode_ll {

void clean_low_latency_buffer(int*, int, int*, int, int, int, int*, int*, cudaStream_t) {
    throw_xpu_unported("internode_ll::clean_low_latency_buffer");
}

void dispatch(void*, void*, int*, int64_t*, int*, int*, int*, int64_t*, void*, int*, void*, const void*, const topk_idx_t*, int*, int, int, int,
              int, int, int, int, int, bool, bool, bool, void*, int, cudaStream_t, int) {
    throw_xpu_unported("internode_ll::dispatch");
}

void combine(void*, void*, int*, void*, const void*, const topk_idx_t*, const float*, const int*, const int64_t*, int*, int64_t*, int*, int, int,
             int, int, int, int, int, bool, void*, int, cudaStream_t, int, bool) {
    throw_xpu_unported("internode_ll::combine");
}

}  // namespace internode_ll

}  // namespace deep_ep
