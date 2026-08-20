// Step-1 de-risking proof for the fused internode-NORMAL dispatch rewrite.
//
// Reproduces the exact work-group shape and NamedBarrier counts that the fused
// CUDA `dispatch` kernel (csrc/cuda_kernels/internode.cu:447) needs on XPU:
//
//   work-group = (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32
//              = (7 + 1 + 8) * 32 = 512 work-items = 16 sub-groups
//
//   barrier 0 (`sync_rdma_sender_smem`, internode.cu:563)
//       CUDA: barrier.sync 0, (kNumDispatchRDMASenderWarps + 1) * 32 = 256
//       SYCL: named_barrier_init(8)     // 8 sub-groups
//   barrier 1 (`sync_forwarder_smem`, internode.cu:580)
//       CUDA: barrier.sync 1, (NUM_MAX_NVL_PEERS + 1) * 32 = 288
//       SYCL: named_barrier_init(9)     // 9 sub-groups
//
// ... while ALSO calling iSHMEM device functions from inside the same kernel.
//
// Build (inside the container, after `source /opt/intel/oneapi/setvars.sh`):
//   icpx -fsycl -fsycl-targets=spir64 -O2 \
//     -I$ISHMEM_DIR/include -I$I_MPI_ROOT/include \
//     csrc/xpu/tools/test_nbarrier_dispatch_shape.cpp \
//     -L$ISHMEM_DIR/lib -l:libishmem.a -lmpi -lze_loader -lhwloc \
//     -o /tmp/nbshape
// Run with 2 ranks via mpirun.

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

class DispatchShapeKernel;

static constexpr int kNumDispatchRDMASenderWarps = 7;
static constexpr int NUM_MAX_NVL_PEERS = 8;
static constexpr int kNumWarps = kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS;  // 16
static constexpr int kNumThreads = kNumWarps * 32;                                     // 512

int main() {
    ishmemx_attr_t attr{};
    attr.runtime = ISHMEMX_RUNTIME_MPI;
    ishmemx_init_attr(&attr);
    int me = ishmem_my_pe(), np = ishmem_n_pes();
    sycl::queue q(sycl::gpu_selector_v);

    int* sym = (int*)ishmem_align(64, 4096 * sizeof(int));
    int* src = sycl::malloc_device<int>(4096, q);
    q.memset(src, 1, 4096 * sizeof(int)).wait();
    q.memset(sym, 0, 4096 * sizeof(int)).wait();
    ishmem_barrier_all();

    int peer = (me + 1) % np;

    // Report the device's max work-group size for this kernel shape.
    auto dev = q.get_device();
    size_t dev_max_wg = dev.get_info<sycl::info::device::max_work_group_size>();
    printf("[pe %d] device max_work_group_size = %zu (need %d)\n", me, dev_max_wg, kNumThreads);
    fflush(stdout);

    // Shared "SLM state" mirroring rdma_send_channel_tail / forward_channel_head.
    constexpr int kNumRDMARanks = 2;

    int* out = sycl::malloc_shared<int>(64, q);
    for (int i = 0; i < 64; ++i) out[i] = -1;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> smem_sender(sycl::range<1>(kNumRDMARanks), h);
        sycl::local_accessor<int, 1> smem_fwd(sycl::range<1>(NUM_MAX_NVL_PEERS * kNumRDMARanks), h);
        h.parallel_for<DispatchShapeKernel>(
            sycl::nd_range<1>(sycl::range<1>(kNumThreads * 2), sycl::range<1>(kNumThreads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                auto sg = it.get_sub_group();
                const int warp_id = sg.get_group_linear_id();
                const int lane_id = sg.get_local_linear_id();
                const int sm_id = it.get_group(0);
                const bool is_forwarder = (sm_id % 2 == 0);

                // IGC rule: both handles must be plain SSA locals of THIS function.
                // Both are initialized unconditionally by every sub-group (init is
                // just a handle materialization, not a barrier).
#if defined(HAVE_NB)
                auto* sync_rdma_sender_smem = named_barrier_init(kNumDispatchRDMASenderWarps + 1);  // 8
                auto* sync_forwarder_smem = named_barrier_init(NUM_MAX_NVL_PEERS + 1);              // 9
#endif

                if (is_forwarder) {
                    // warps 0..7  -> kRDMAAndNVLForwarder
                    // warp  8     -> kForwarderCoordinator (participates in barrier 1)
                    // warps 9..15 -> idle coordinator warps (never enter barrier 1)
                    if (warp_id < NUM_MAX_NVL_PEERS) {
                        if (lane_id < kNumRDMARanks)
                            smem_fwd[warp_id * kNumRDMARanks + lane_id] = 0;
#if defined(HAVE_NB)
                        work_group_named_barrier(sync_forwarder_smem, 0x3);
#endif
                        // an iSHMEM device call from a barrier participant
                        if (warp_id == 0 && lane_id == 0)
                            ishmem_putmem_nbi(sym, src, 128 * sizeof(int), peer);
#if defined(HAVE_NB)
                        work_group_named_barrier(sync_forwarder_smem, 0x3);
#endif
                        if (lane_id == 0) out[warp_id] = 100 + warp_id;
                    } else if (warp_id == NUM_MAX_NVL_PEERS) {
#if defined(HAVE_NB)
                        work_group_named_barrier(sync_forwarder_smem, 0x3);
#endif
                        if (lane_id == 0) out[8] = 108;
#if defined(HAVE_NB)
                        work_group_named_barrier(sync_forwarder_smem, 0x3);
#endif
                        // verify all forwarders published their SLM slot
                        int sum = 0;
                        for (int i = 0; i < NUM_MAX_NVL_PEERS * kNumRDMARanks; ++i) sum += smem_fwd[i];
                        if (lane_id == 0) out[9] = sum;  // expect 0
                    } else {
                        // non-participants: must NOT enter barrier 1
                        if (lane_id == 0) out[10 + (warp_id - NUM_MAX_NVL_PEERS - 1)] = 200 + warp_id;
                    }
                } else {
                    // warps 0..6  -> kRDMASender
                    // warp  7     -> kRDMASenderCoordinator
                    // warps 8..15 -> kNVLReceivers (never enter barrier 0)
                    if (warp_id < kNumDispatchRDMASenderWarps) {
#if defined(HAVE_NB)
                        work_group_named_barrier(sync_rdma_sender_smem, 0x3);
#endif
                        if (warp_id == 0 && lane_id == 0)
                            ishmem_putmem_nbi(sym + 256, src, 128 * sizeof(int), peer);
                        if (lane_id == 0) out[20 + warp_id] = 300 + warp_id;
                    } else if (warp_id == kNumDispatchRDMASenderWarps) {
                        if (lane_id < kNumRDMARanks) smem_sender[lane_id] = 0;
#if defined(HAVE_NB)
                        work_group_named_barrier(sync_rdma_sender_smem, 0x3);
#endif
                        if (lane_id == 0) out[27] = 307;
                    } else {
                        if (lane_id == 0) out[28 + (warp_id - kNumDispatchRDMASenderWarps - 1)] = 400 + warp_id;
                    }
                }
            });
    }).wait();
    ishmem_quiet();
    ishmem_barrier_all();

    bool ok = true;
    for (int i = 0; i < 9; ++i) ok &= (out[i] == 100 + i);
    ok &= (out[9] == 0);
    for (int i = 0; i < 7; ++i) ok &= (out[10 + i] == 200 + 9 + i);
    for (int i = 0; i < 7; ++i) ok &= (out[20 + i] == 300 + i);
    ok &= (out[27] == 307);
    for (int i = 0; i < 8; ++i) ok &= (out[28 + i] == 400 + 8 + i);
    printf("[pe %d] fwd=", me);
    for (int i = 0; i < 10; ++i) printf("%d,", out[i]);
    printf(" idle=");
    for (int i = 10; i < 17; ++i) printf("%d,", out[i]);
    printf(" send=");
    for (int i = 20; i < 28; ++i) printf("%d,", out[i]);
    printf(" recv=");
    for (int i = 28; i < 36; ++i) printf("%d,", out[i]);
    printf("  => %s HAVE_NB_host=%s\n", ok ? "OK" : "FAIL",
#if defined(HAVE_NB)
           "yes"
#else
           "no"
#endif
    );
    fflush(stdout);
    ishmem_barrier_all();
    return ok ? 0 : 1;
}
