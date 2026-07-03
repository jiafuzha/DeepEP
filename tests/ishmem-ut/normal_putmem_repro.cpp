/* Minimal iSHMEM reproducer for the DeepEP internode NORMAL-path hang.
 *
 * WHY THIS EXISTS
 * ---------------
 * The DeepEP internode low-latency path (dispatch_bf16 in internode_ll.cpp)
 * uses ONLY device-initiated non-blocking puts + work-group collectives:
 *     ishmem_putmem_nbi(...)                 // ring doorbell, return immediately
 *     ishmemx_barrier_all_work_group(group)  // every WI participates
 *     ishmemx_quiet_work_group(group)        // work-group drains the SQ
 * ll_put_repro.cpp already covers that path and PASSES on the fixed iSHMEM.
 *
 * The internode NORMAL path (CombinedDispatchRdmaPutKernel in internode.cpp)
 * ADDITIONALLY relies on two primitives the LL path never uses:
 *
 *     if (group.leader()) ishmem_barrier_all();   // SINGLE work-item,
 *                                                  // DEVICE-WIDE barrier
 *     ...
 *     if (group.leader()) ishmem_putmem(...);      // SCALAR BLOCKING put:
 *                                                  // polls the IBGDA CQ until
 *                                                  // the NIC ACKs the write
 *
 * Both of these complete on the DEVICE while the HOST thread is parked in
 * queue.wait(). Their completion needs host-proxy progress; internode.cpp's
 * own comments (lines ~293, ~304-307, ~2480-2499) warn that on a cold QP the
 * single-WI device-wide ishmem_barrier_all() / blocking ishmem_putmem "spins
 * forever ... deadlocking". On the suspect iSHMEM build the real test hangs at
 * exactly the first BF16 dispatch after the QP warmup.
 *
 * This program reproduces that hang in isolation, WITHOUT torch/DeepEP:
 *
 *   REPRO_MODE=nbi_wg   (control)  ishmem_putmem_nbi + ishmemx_*_work_group
 *                                  -> the LL-style path; must COMPLETE.
 *   REPRO_MODE=barrier             group.leader() -> ishmem_barrier_all()
 *                                  (single-WI device-wide barrier), host in
 *                                  queue.wait(). Mirrors internode.cpp:2510.
 *   REPRO_MODE=putmem              group.leader() -> blocking ishmem_putmem to
 *                                  every peer, then ishmem_barrier_all().
 *   REPRO_MODE=exact               faithful CombinedDispatchRdmaPutKernel:
 *                                  rdma/nvl rank split, sentinel + fence,
 *                                  barrier -> nvl0-only SPLIT blocking put ->
 *                                  barrier. Mirrors internode.cpp:2500-2542.
 *   REPRO_MODE=full   (REPRODUCER) QpWarmupKernel pattern (NBI + wg-barrier) in
 *                                  a SEPARATE kernel + queue.wait() FIRST, then
 *                                  the `exact` RdmaPut kernel on the now-warm QP
 *                                  -- reproducing the exact dispatch_nvl_rdma
 *                                  sequence (warmup_qps -> CombinedDispatchRdmaPut).
 *
 * FINDING (validated on real 2-node b70-hq-1 / b70-hq-2, 2 GPUs each):
 *   The real internode-normal test hangs in CombinedDispatchRdmaPutKernel
 *   (localized via DEEP_EP_KERNEL_TIMING=1: Init/Pack/PackBarrier/RdmaSend all
 *   complete, RdmaPut never returns). The ISOLATED primitives (barrier / putmem
 *   / exact) all PASS on BOTH iSHMEM builds -- so payload size and the primitive
 *   in isolation are NOT the trigger. Only the `full` mode reproduces the hang,
 *   and ONLY on the suspect build:
 *
 *       mode=exact (no warmup)   good=PASS   suspect=PASS
 *       mode=full  (warmup+put)  good=PASS   suspect=HANG   <-- reproduces
 *
 *   Root cause: after a preceding warmup kernel + queue.wait(), a subsequent
 *   DEVICE-side BLOCKING ishmem_putmem (and/or single-WI ishmem_barrier_all)
 *   never completes on the suspect build -- its completion needs host-proxy
 *   progress while the host is parked in queue.wait(), exactly the hazard the
 *   warmup_qps comments (internode.cpp:293-296, 304-307) warn about. The LL path
 *   never uses these primitives (NBI + work-group collectives only), so it
 *   passes while the normal path hangs.
 *
 * On a healthy iSHMEM every mode COMPLETES and prints PASS. On the broken build
 * `nbi_wg` completes but `barrier` / `putmem` HANG (the process never prints
 * "[normrepro] ... completed"); the launcher's timeout then flags the hang.
 *
 * A per-stage host-side watchdog prints how far each PE got before a hang, so
 * the failure is observable as a partial-progress log rather than a silent stall.
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <ishmem.h>
#include <ishmemx.h>
#include <sycl/sycl.hpp>

namespace {

// Mirror internode.cpp's kIshmemWGSize single-work-group launch shape: one
// work-group, group.leader() issues the blocking op, all WIs hit the barrier.
// nbi_wg/barrier/putmem use kWGSize; the `exact` mode uses the real value.
constexpr size_t kWGSize = 512;
constexpr size_t kIshmemWGSize = 32;

const char* env_str(const char* name, const char* def) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? v : def;
}
int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : def;
}

// Run `fn` (which submits a kernel and calls queue.wait()) on a helper thread,
// with the host watchdog reporting if it does not return within `timeout_s`.
// Reports the elapsed time on completion so a catastrophically-SLOW primitive
// (the degraded-IBGDA signature: ~11s where healthy is ~ms) is distinguished
// from a true deadlock. Returns true if the stage completed, false if it hung.
template <typename Fn>
bool run_stage_with_watchdog(const char* stage, int my_pe, int timeout_s, Fn&& fn) {
    std::atomic<bool> done{false};
    const auto start = std::chrono::steady_clock::now();
    std::thread worker([&]() {
        fn();
        done.store(true, std::memory_order_release);
    });

    while (!done.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed >= timeout_s) {
            std::printf("[normrepro] PE=%d stage=%s HANG (no completion after %ds; host parked in queue.wait())\n",
                        my_pe,
                        stage,
                        timeout_s);
            std::fflush(stdout);
            // The device op is wedged; the worker thread is stuck in queue.wait()
            // and cannot be safely joined. Detach and let the launcher's timeout
            // tear the whole process down (mirrors the real test's mpirun timeout).
            worker.detach();
            return false;
        }
    }
    worker.join();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::printf("[normrepro] PE=%d stage=%s completed in %.1f ms\n", my_pe, stage, ms);
    std::fflush(stdout);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    ishmem_init();
    const int my_pe = ishmem_my_pe();
    const int npes = ishmem_n_pes();

    const std::string mode = env_str("REPRO_MODE", "full");
    const int stage_timeout = env_int("REPRO_STAGE_TIMEOUT", 30);
    const int payload_bytes = env_int("REPRO_PAYLOAD_BYTES", 131072);

    sycl::queue q;
    if (my_pe == 0) {
        std::printf("[normrepro] npes=%d mode=%s payload_bytes=%d stage_timeout=%ds device=%s\n",
                    npes,
                    mode.c_str(),
                    payload_bytes,
                    stage_timeout,
                    q.get_device().get_info<sycl::info::device::name>().c_str());
        std::fflush(stdout);
    }

    // Symmetric buffers: a send region and a receive region per PE.
    uint8_t* send = static_cast<uint8_t*>(ishmem_malloc(static_cast<size_t>(payload_bytes)));
    uint8_t* recv = static_cast<uint8_t*>(ishmem_malloc(static_cast<size_t>(payload_bytes) * npes));
    q.memset(send, static_cast<int>(my_pe & 0xFF), payload_bytes).wait();
    q.memset(recv, 0, static_cast<size_t>(payload_bytes) * npes).wait();
    ishmem_barrier_all();

    const int me = my_pe;
    const int n = npes;
    const size_t pbytes = static_cast<size_t>(payload_bytes);

    bool ok = true;

    if (mode == "nbi_wg") {
        // CONTROL: the LL-style path. Must complete on a healthy fix.
        ok = run_stage_with_watchdog("nbi_wg", my_pe, stage_timeout, [&]() {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kWGSize), sycl::range<1>(kWGSize)), [=](sycl::nd_item<1> item) {
                     auto group = item.get_group();
                     if (group.get_local_linear_id() == 0) {
                         for (int dst = 0; dst < n; ++dst) {
                             if (dst == me) continue;
                             ishmem_putmem_nbi(recv + static_cast<size_t>(me) * pbytes, send, pbytes, dst);
                         }
                     }
                     sycl::group_barrier(group);
                     ishmemx_quiet_work_group(group);
                     ishmemx_barrier_all_work_group(group);
                     sycl::group_barrier(group);
                 });
             }).wait();
        });
    } else if (mode == "barrier") {
        // Single-WI device-wide ishmem_barrier_all() from the group leader,
        // host parked in queue.wait(). Mirrors internode.cpp:2510/2541.
        ok = run_stage_with_watchdog("barrier", my_pe, stage_timeout, [&]() {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kWGSize), sycl::range<1>(kWGSize)), [=](sycl::nd_item<1> item) {
                     auto group = item.get_group();
                     sycl::group_barrier(group);
                     if (group.leader()) ishmem_barrier_all();
                     sycl::group_barrier(group);
                 });
             }).wait();
        });
    } else if (mode == "full") {
        // FULL dispatch_nvl_rdma reproduction: run the QpWarmupKernel pattern
        // (device NBI put + work-group barrier -- the "safe" primitives the
        // warmup deliberately uses) in a SEPARATE kernel + queue.wait() FIRST,
        // exactly as warmup_qps() does, THEN run the RdmaPut kernel that uses
        // the "forbidden" blocking ishmem_putmem + single-WI ishmem_barrier_all.
        //
        // internode.cpp's own warmup comments (293-296, 304-307) state those
        // blocking primitives "deadlock ... needs host-proxy progress while the
        // host thread is parked in queue.wait()". This mode checks whether it is
        // the QP/CQ state left by the preceding warmup (host parked across two
        // back-to-back queue.wait()s) that wedges the subsequent blocking put --
        // the real hang that `exact` alone (fresh QP) does not trigger.
        constexpr int kNvlPerNode = 2;
        const int nvl_rank = me % kNvlPerNode;
        const int rdma_rank = me / kNvlPerNode;
        const int num_rdma = (n + kNvlPerNode - 1) / kNvlPerNode;
        const size_t count_off = pbytes - sizeof(int);

        // ---- Stage A: QpWarmupKernel pattern (must complete; NBI + wg barrier).
        bool warm_ok = run_stage_with_watchdog("full:warmup", my_pe, stage_timeout, [&]() {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
                     auto group = item.get_group();
                     if (nvl_rank == 0 && group.get_local_linear_id() == 0) {
                         for (int dst_rdma = 0; dst_rdma < num_rdma; ++dst_rdma) {
                             if (dst_rdma == rdma_rank) continue;
                             ishmem_putmem_nbi(send, send, 128, dst_rdma * kNvlPerNode);
                         }
                     }
                     sycl::group_barrier(group);
                     ishmemx_barrier_all_work_group(group);
                     sycl::group_barrier(group);
                 });
             }).wait();
        });

        // ---- Stage B: the exact RdmaPut sequence on the now-warm QP.
        bool put_ok = false;
        if (warm_ok) {
            put_ok = run_stage_with_watchdog("full:rdmaput", my_pe, stage_timeout, [&]() {
                q.submit([&](sycl::handler& cgh) {
                     cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
                         auto group = item.get_group();
                         constexpr int kSentinel = -424242;
                         if (nvl_rank == 0 && group.get_local_linear_id() == 0) {
                             for (int src_rdma = 0; src_rdma < num_rdma; ++src_rdma) {
                                 if (src_rdma == rdma_rank) continue;
                                 *reinterpret_cast<int*>(recv + static_cast<size_t>(src_rdma) * pbytes + count_off) = kSentinel;
                             }
                             sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                         }
                         sycl::group_barrier(group);
                         if (group.leader()) ishmem_barrier_all();
                         sycl::group_barrier(group);

                         if (nvl_rank == 0 && group.get_local_linear_id() == 0) {
                             for (int dst_rdma = 0; dst_rdma < num_rdma; ++dst_rdma) {
                                 if (dst_rdma == rdma_rank) continue;
                                 const int dst_pe = dst_rdma * kNvlPerNode;
                                 uint8_t* dst_region = recv + static_cast<size_t>(rdma_rank) * pbytes;
                                 ishmem_putmem(dst_region, send, count_off, dst_pe);
                                 ishmem_putmem(dst_region + count_off, send + count_off, sizeof(int), dst_pe);
                             }
                         }
                         sycl::group_barrier(group);
                         if (group.leader()) ishmem_barrier_all();
                         sycl::group_barrier(group);
                     });
                 }).wait();
            });
        }
        ok = warm_ok && put_ok;
    } else if (mode == "exact") {
        // FAITHFUL mirror of CombinedDispatchRdmaPutKernel (internode.cpp
        // 2500-2542): emulate the rdma/nvl rank split (2 nvl ranks per node),
        // sentinel uc_store + release fence, leading device-wide barrier, then
        // ONLY the nvl_rank==0 PE issues the two-part SPLIT blocking put (data
        // region, then the 4-byte count word) on the same QP, then a trailing
        // device-wide barrier. All PEs' leaders participate in both barriers.
        constexpr int kNvlPerNode = 2;
        const int nvl_rank = me % kNvlPerNode;
        const int rdma_rank = me / kNvlPerNode;
        const int num_rdma = (n + kNvlPerNode - 1) / kNvlPerNode;
        const size_t count_off = pbytes - sizeof(int);  // data region | count word
        ok = run_stage_with_watchdog("exact", my_pe, stage_timeout, [&]() {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
                     auto group = item.get_group();
                     constexpr int kSentinel = -424242;
                     if (nvl_rank == 0 && group.get_local_linear_id() == 0) {
                         // Stamp MY receive count slots with a sentinel.
                         for (int src_rdma = 0; src_rdma < num_rdma; ++src_rdma) {
                             if (src_rdma == rdma_rank) continue;
                             *reinterpret_cast<int*>(recv + static_cast<size_t>(src_rdma) * pbytes + count_off) = kSentinel;
                         }
                         sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                     }
                     sycl::group_barrier(group);
                     if (group.leader()) ishmem_barrier_all();
                     sycl::group_barrier(group);

                     if (nvl_rank == 0 && group.get_local_linear_id() == 0) {
                         for (int dst_rdma = 0; dst_rdma < num_rdma; ++dst_rdma) {
                             if (dst_rdma == rdma_rank) continue;
                             const int dst_pe = dst_rdma * kNvlPerNode;
                             uint8_t* dst_region = recv + static_cast<size_t>(rdma_rank) * pbytes;
                             // Split-put, both blocking, same QP: data then count.
                             ishmem_putmem(dst_region, send, count_off, dst_pe);
                             ishmem_putmem(dst_region + count_off, send + count_off, sizeof(int), dst_pe);
                         }
                     }
                     sycl::group_barrier(group);
                     if (group.leader()) ishmem_barrier_all();
                     sycl::group_barrier(group);
                 });
             }).wait();
        });
    } else {
        // DEFAULT "putmem": blocking scalar ishmem_putmem to every peer from the
        // group leader, then a single-WI device-wide ishmem_barrier_all().
        // This is the exact primitive sequence of internode.cpp:2534-2541.
        ok = run_stage_with_watchdog("putmem", my_pe, stage_timeout, [&]() {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kWGSize), sycl::range<1>(kWGSize)), [=](sycl::nd_item<1> item) {
                     auto group = item.get_group();
                     if (group.leader()) {
                         for (int dst = 0; dst < n; ++dst) {
                             if (dst == me) continue;
                             // Blocking put: polls the IBGDA CQ until the NIC ACKs.
                             ishmem_putmem(recv + static_cast<size_t>(me) * pbytes, send, pbytes, dst);
                         }
                     }
                     sycl::group_barrier(group);
                     if (group.leader()) ishmem_barrier_all();
                     sycl::group_barrier(group);
                 });
             }).wait();
        });
    }

    if (ok) {
        std::printf("[normrepro] PE=%d mode=%s completed -> PASS\n", my_pe, mode.c_str());
        std::fflush(stdout);
        ishmem_barrier_all();
        if (my_pe == 0) {
            std::printf("[normrepro] ===== ALL PEs completed mode=%s : PASS =====\n", mode.c_str());
            std::fflush(stdout);
        }
    } else {
        std::printf("[normrepro] PE=%d mode=%s DID NOT COMPLETE -> FAIL (hang reproduced)\n", my_pe, mode.c_str());
        std::fflush(stdout);
        // Do NOT attempt further ishmem calls / clean shutdown: the device is
        // wedged. Exit non-zero and let the launcher tear the job down.
        std::_Exit(EXIT_FAILURE);
    }

    ishmem_free(send);
    ishmem_free(recv);
    ishmem_finalize();
    return EXIT_SUCCESS;
}
