#include <sycl/sycl.hpp>
#include <ishmem.h>
#include <ishmemx.h>
#include <cstdio>

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
struct __namedBarrier;
extern SYCL_EXTERNAL __namedBarrier __attribute__((opencl_local)) *
named_barrier_init(int count);
extern SYCL_EXTERNAL void work_group_named_barrier(
    __namedBarrier __attribute__((opencl_local)) *, unsigned int);
#define HAVE_NB 1
#endif

class ReproKernel;

int main() {
    ishmemx_attr_t attr{};
    attr.runtime = ISHMEMX_RUNTIME_MPI;
    ishmemx_init_attr(&attr);
    int me = ishmem_my_pe(), np = ishmem_n_pes();
    sycl::queue q(sycl::gpu_selector_v);
    int* sym = (int*)ishmem_align(64, 1024 * sizeof(int));
    int* src = sycl::malloc_device<int>(1024, q);
    q.memset(src, 1, 1024 * sizeof(int)).wait();
    q.memset(sym, 0, 1024 * sizeof(int)).wait();
    ishmem_barrier_all();

    int peer = (me + 1) % np;
    int* out = sycl::malloc_shared<int>(4, q);
    out[0] = out[1] = -1;

    const int kSubGroups = 8;      // 8 sub-groups of 32 = 256 WI
    const int kThreads = kSubGroups * 32;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<ReproKernel>(
            sycl::nd_range<1>(sycl::range<1>(kThreads), sycl::range<1>(kThreads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                auto sg = it.get_sub_group();
                int wid = sg.get_group_linear_id();
                int lane = sg.get_local_linear_id();
#if defined(HAVE_NB)
                auto* bar_a = named_barrier_init(4);   // sub-groups 0..3
                auto* bar_b = named_barrier_init(4);   // sub-groups 4..7
#endif
                if (wid < 4) {
                    // "sender" role: issue an iSHMEM put (non-inlined lib call)
                    if (wid == 0 && lane == 0)
                        ishmem_putmem_nbi(sym, src, 256 * sizeof(int), peer);
#if defined(HAVE_NB)
                    work_group_named_barrier(bar_a, 0x3);
                    work_group_named_barrier(bar_a, 0x1);
#else
                    it.barrier(sycl::access::fence_space::global_and_local);
#endif
                    if (it.get_local_id(0) == 0) out[0] = 42;
                } else {
                    // "receiver" role: must NOT enter bar_a
#if defined(HAVE_NB)
                    work_group_named_barrier(bar_b, 0x1);
                    work_group_named_barrier(bar_b, 0x3);
#else
                    it.barrier(sycl::access::fence_space::global_and_local);
#endif
                    if (wid == 4 && lane == 0) out[1] = 43;
                }
            });
    }).wait();
    ishmem_quiet();
    ishmem_barrier_all();
    printf("[pe %d] out=%d,%d (expect 42,43) HAVE_NB_host=%s\n", me, out[0], out[1],
#if defined(HAVE_NB)
           "yes"
#else
           "no(host-pass-only)"
#endif
    );
    fflush(stdout);
    ishmem_barrier_all();
    return 0;
}
