// -----------------------------------------------------------------------------
// FAITHFUL reproducer: SPIR-V NamedBarrier vs the REAL iSHMEM IBGDA device
// library, device-linked exactly the way DeepEP builds it (-fsycl-rdc, iSHMEM
// static archive merged into the SYCL device link).
//
// This kernel:
//   (a) uses the SPIR-V named barrier (work_group_named_barrier), and
//   (b) calls a real iSHMEM device function (ishmem_my_pe) so the iSHMEM RDC
//       device subroutines are retained and stitched into the module.
//
// Expected on BMG (Arc Pro B60, oneAPI 2025.3 + iSHMEM IBGDA archive):
//   at first-kernel JIT/AOT, IGC stamps `.kernel_attr NBarrierCnt=1` on the
//   kernel AND on every non-inlined iSHMEM device subroutine, and vISA fails:
//       "More than 1 kernel attribute defined NBarrierCnt"
//
// NOTE: we do NOT initialize the iSHMEM runtime -- the NBarrierCnt failure is a
// COMPILE/JIT-time error that occurs before the kernel is ever launched, so no
// iSHMEM bootstrap (MPI/NIC) is required to reproduce it. The ishmem_my_pe call
// only needs to survive dead-code elimination into device codegen.
// -----------------------------------------------------------------------------
#include <cstdio>
#include <sycl/sycl.hpp>
#include <ishmem.h>

// --- SPIR-V NamedBarrier builtins (cl_khr_subgroup_named_barrier) -------------
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
struct __namedBarrier;
extern SYCL_EXTERNAL __namedBarrier __attribute__((opencl_local)) *
named_barrier_init(int count);
extern SYCL_EXTERNAL void work_group_named_barrier(
    __namedBarrier __attribute__((opencl_local)) *, unsigned int);
#endif

constexpr unsigned int CLK_GLOBAL_MEM_FENCE = 0x2;

class NamedBarrier {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    __namedBarrier __attribute__((opencl_local)) * handle_ = nullptr;
#endif
public:
    inline void init([[maybe_unused]] int count) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        handle_ = named_barrier_init(count);
#endif
    }
    inline void sync([[maybe_unused]] unsigned int flags = CLK_GLOBAL_MEM_FENCE) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        work_group_named_barrier(handle_, flags);
#endif
    }
};

// Toggle the iSHMEM device call. USE_ISHMEM=0 builds the control (named barrier
// only, no iSHMEM subroutines) which is expected to compile and run.
#ifndef USE_ISHMEM
#define USE_ISHMEM 1
#endif

constexpr unsigned SG = 32;
constexpr unsigned NSG = 4;
constexpr unsigned PART = 3;
constexpr unsigned WI = SG * NSG;

class ReproIshmemKernel;

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("mode: USE_ISHMEM=%d (1 = named barrier + real iSHMEM device fn)\n",
                USE_ISHMEM);

    int* out = sycl::malloc_shared<int>(1, q);
    *out = 0;

    q.submit([&](sycl::handler& h) {
         h.parallel_for<ReproIshmemKernel>(
             sycl::nd_range<1>{WI, WI},
             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                 const unsigned sg_id = it.get_sub_group().get_group_linear_id();

                 NamedBarrier bar;
                 bar.init(PART);
                 if (sg_id < PART) {
                     bar.sync(CLK_GLOBAL_MEM_FENCE);
                 }

#if USE_ISHMEM
                 // Reference the REAL iSHMEM IBGDA RDMA device path so the large
                 // non-inlined iSHMEM device subroutines (ishmemi_ibgda_*_poll_cq,
                 // emit_*_wqe_nbi, ring_doorbell, ...) are linked and stitched
                 // into this module -- these are what IGC stamps NBarrierCnt onto.
                 //
                 // `*out` is 0 at launch, so this branch NEVER executes at runtime
                 // (iSHMEM is not initialized here); but the compiler cannot prove
                 // that, so the RDMA subroutines survive into device codegen and
                 // the NBarrierCnt vISA error triggers at JIT-compile time -- before
                 // the kernel body runs.
                 if (*out < 0) {
                     ishmem_putmem(out, out, sizeof(int), 0);
                     ishmem_quiet();
                 }
                 // A small always-safe iSHMEM call so `out` gets a defined value.
                 if (it.get_local_linear_id() == 0) {
                     sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>
                         o{*out};
                     o.fetch_add(ishmem_my_pe());
                 }
#endif
             });
     }).wait();

    std::printf("PASS: kernel compiled and ran; out=%d\n", *out);
    sycl::free(out, q);
    return 0;
}
