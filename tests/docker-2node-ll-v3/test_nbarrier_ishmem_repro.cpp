// test_nbarrier_ishmem_repro.cpp — verifies that the SPIR-V NamedBarrier
// primitive coexists with iSHMEM device calls in the same SYCL kernel.
//
// Per csrc/xpu/named_barrier_usage.md, we want to prove that:
//   1. A SYCL kernel can call a non-inlined iSHMEM device function
//      (e.g. ishmem_putmem_nbi) AND named_barrier_init /
//      work_group_named_barrier in the SAME translation unit / kernel.
//   2. The compiler/JIT/AOT pipeline accepts the combination
//      (this is what previously miscompiled at JIT time).
//   3. The kernel executes and produces the expected data movement.
//
// Build (spir64 JIT, matches the guide):
//   icpx -fsycl -fsycl-rdc -fsycl-targets=spir64 \
//       $(pkg-config --cflags ishmem) test_nbarrier_ishmem_repro.cpp \
//       $(pkg-config --libs ishmem) -o test_nbarrier_ishmem_repro
//
// Run (two PEs via mpirun; iSHMEM auto-selects the NIC by PCIe topology):
//   mpirun -n 2 -ppn 1 ./test_nbarrier_ishmem_repro

#include <sycl/sycl.hpp>
#include <ishmem.h>
#include <ishmemx.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// SPIR-V NamedBarrier declarations — MUST be at global scope, MUST NOT sit
// inside any namespace (matches csrc/xpu/xpu_kernels.hpp).
// ---------------------------------------------------------------------------
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
struct __namedBarrier;
extern SYCL_EXTERNAL __namedBarrier __attribute__((opencl_local)) *
named_barrier_init(int count);
extern SYCL_EXTERNAL void work_group_named_barrier(
    __namedBarrier __attribute__((opencl_local)) *, unsigned int);
#endif

// NamedBarrier flag: global memory + local memory fence (matches xpu_kernels.hpp default)
static constexpr unsigned int kNamedBarrierGlobalFence = 0x2u;

// ---------------------------------------------------------------------------
// A tiny helper class that mirrors the shape used in xpu_kernels.hpp.
// ---------------------------------------------------------------------------
class NamedBarrier {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    ::__namedBarrier __attribute__((opencl_local)) * handle_ = nullptr;
#endif
public:
    void init(int num_subgroups) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        handle_ = ::named_barrier_init(num_subgroups);
#else
        (void)num_subgroups;
#endif
    }
    void sync(unsigned int flags = kNamedBarrierGlobalFence) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        ::work_group_named_barrier(handle_, flags);
#else
        (void)flags;
#endif
    }
};

// ---------------------------------------------------------------------------
// Main — mpirun-launched, two PE program.
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // iSHMEM MPI-runtime init (bootstraps MPI internally under
    // ISHMEM_DEFAULT_RUNTIME=MPI, as our build uses).
    ishmem_init();

    int my_pe = ishmem_my_pe();
    int n_pes = ishmem_n_pes();
    std::fprintf(stderr, "[nb-repro] pe=%d/%d init OK\n", my_pe, n_pes);

    if (n_pes < 2) {
        std::fprintf(stderr,
            "[nb-repro] need >=2 PEs, got %d — skipping data transfer test.\n",
            n_pes);
    }

    // Symmetric buffers: 64 int slots on each PE.
    constexpr size_t N = 64;
    int* src = static_cast<int*>(ishmem_malloc(N * sizeof(int)));
    int* dst = static_cast<int*>(ishmem_malloc(N * sizeof(int)));
    if (!src || !dst) {
        std::fprintf(stderr, "[nb-repro] pe=%d ishmem_malloc FAILED\n", my_pe);
        ishmem_finalize();
        return 2;
    }

    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order()});
    std::fprintf(stderr, "[nb-repro] pe=%d device=%s\n", my_pe,
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    // Initialize src[i] = my_pe*1000 + i, dst[i] = -1  (device init to keep
    // everything on-GPU, matches DeepEP kernel patterns).
    int me = my_pe;
    std::fprintf(stderr, "[nb-repro] pe=%d before init kernel\n", my_pe); std::fflush(stderr);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            src[i] = me * 1000 + static_cast<int>(i);
            dst[i] = -1;
        });
    }).wait_and_throw();
    std::fprintf(stderr, "[nb-repro] pe=%d after init kernel\n", my_pe); std::fflush(stderr);

    ishmem_barrier_all();
    std::fprintf(stderr, "[nb-repro] pe=%d after first barrier\n", my_pe); std::fflush(stderr);

    // The core coexistence kernel:
    //   - launched as a work-group of 32 sub-groups (matches LL kernel shape:
    //     num_warps_per_group == 32);
    //   - the first sub-group ("caster") calls ishmem_putmem_nbi;
    //   - all sub-groups meet at a NamedBarrier.
    // We use nd_range to get real sub-groups (single_task would collapse to 1
    // work-item and hide the interesting JIT path).
    const int peer = (my_pe + 1) % n_pes;
    const size_t sg_size = 32;      // sub-group size
    const size_t num_sgs = 32;      // 32 sub-groups per WG (LL shape)
    const size_t wg_size = sg_size * num_sgs;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(wg_size), sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
#ifndef DISABLE_NBARRIER
                NamedBarrier nb;
                // "Must be called uniformly by all participants." (xpu_kernels.hpp)
                nb.init(static_cast<int>(num_sgs));
#endif

                auto sg = it.get_sub_group();
                if (sg.get_group_id()[0] == 0 && sg.get_local_id()[0] == 0) {
#ifndef DISABLE_ISHMEM_CALL
                    // Non-inlined iSHMEM device call — the historical JIT
                    // miscompile root cause when combined with NamedBarrier.
                    ishmem_putmem_nbi(dst, src, N * sizeof(int), peer);
                    ishmem_quiet();
#else
                    // Fallback: plain memcpy so peer sees data in the
                    // no-ishmem-device-call variant (host-verified separately).
                    for (size_t i = 0; i < N; ++i) dst[i] = src[i];
#endif
                }

#ifndef DISABLE_NBARRIER
                // Sub-group-subset barrier across all sub-groups. Even though
                // we sync ALL sub-groups here (== whole WG), the interesting
                // property is that the SPIR-V NamedBarrier + non-inlined
                // iSHMEM call coexist in the same kernel.
                nb.sync(kNamedBarrierGlobalFence);
#endif
            });
    }).wait_and_throw();
    std::fprintf(stderr, "[nb-repro] pe=%d after main kernel\n", my_pe); std::fflush(stderr);

    // Cross-PE synchronization AFTER the putmem_nbi + quiet so the receiver
    // sees the data.
    ishmem_barrier_all();

    // Verify: dst on my_pe should contain peer's src pattern (or own src in the
    // no-ishmem-call variant).
#ifdef DISABLE_ISHMEM_CALL
    int expected_peer = my_pe;
#else
    int expected_peer = (my_pe + n_pes - 1) % n_pes;   // whoever wrote into us
#endif
    int errors = 0;
    // Copy back to host with a small kernel to avoid direct-USM host reads on XPU.
    int* host_dst = static_cast<int*>(std::malloc(N * sizeof(int)));
    q.memcpy(host_dst, dst, N * sizeof(int)).wait_and_throw();
    for (size_t i = 0; i < N; ++i) {
        int want = expected_peer * 1000 + static_cast<int>(i);
        if (host_dst[i] != want) {
            if (errors < 4) {
                std::fprintf(stderr,
                    "[nb-repro] pe=%d dst[%zu]=%d expected %d\n",
                    my_pe, i, host_dst[i], want);
            }
            ++errors;
        }
    }
    std::free(host_dst);

    if (errors == 0) {
        std::fprintf(stderr,
            "[nb-repro] pe=%d PASS: NamedBarrier + iSHMEM device call OK\n",
            my_pe);
    } else {
        std::fprintf(stderr,
            "[nb-repro] pe=%d FAIL: %d/%zu mismatches\n",
            my_pe, errors, N);
    }

    ishmem_free(src);
    ishmem_free(dst);
    ishmem_finalize();
    return errors == 0 ? 0 : 1;
}
