/* Minimal iSHMEM reproducer for the DeepEP internode teardown ABORT.
 *
 * WHY THIS EXISTS
 * ---------------
 * After the internode NORMAL test passes ALL correctness checks and both ranks
 * print "[teardown] all done, exiting cleanly", the processes still abort:
 *
 *     [teardown] all done, exiting cleanly
 *     terminate called without an active exception   (x N)
 *     ... KILLED BY SIGNAL: 6 (Aborted)
 *
 * ROOT CAUSE (confirmed by disassembling libishmem.a proxy.cpp.o):
 *   - ishmemi_proxy_init() launches a global  std::thread proxy_thread  that
 *     runs host_proxy_thread() -- the IBGDA host-proxy progress loop that
 *     drives blocking ishmem_putmem / device-wide ishmem_barrier_all
 *     completion (the very mechanism the normal path depends on).
 *   - ishmemi_proxy_fini() -- reached ONLY through ishmem_finalize() -- is the
 *     sole place that join()s proxy_thread.
 *   - DeepEP's internode::finalize() (csrc/xpu/deep_ep_xpu.cpp) SKIPS
 *     ishmem_finalize() by default (it "can hang after a long workload"), so
 *     proxy_thread is never joined.
 *   - At process exit the global std::thread's destructor runs while the thread
 *     is still joinable. std::thread::~thread() on a joinable thread calls
 *     std::terminate() -> "terminate called without an active exception" ->
 *     SIGABRT. Exactly the teardown crash.
 *
 * WHAT THIS PROGRAM DOES
 * ----------------------
 * Mirror DeepEP's lifecycle in isolation, WITHOUT torch/DeepEP:
 *   ishmem_init  -> (proxy_thread starts)
 *   blocking ishmem_putmem to every peer + ishmem_barrier_all
 *                -> exercise the proxy so it is definitely running
 *   then RETURN FROM main according to REPRO_FINALIZE:
 *
 *     REPRO_FINALIZE=0  (default)  -> do NOT call ishmem_finalize()  (DeepEP's
 *                                     default). EXPECT: "terminate called
 *                                     without an active exception" + SIGABRT
 *                                     (process exit code 134) -- reproduces.
 *     REPRO_FINALIZE=1             -> call ishmem_finalize() (joins proxy_thread)
 *                                     before returning. EXPECT: clean exit rc=0
 *                                     (this is the fix -- unless finalize itself
 *                                     hangs, which is the reason DeepEP skips it).
 *
 * The abort happens during C++ runtime teardown (global/atexit destructors),
 * AFTER main returns and AFTER our "exiting cleanly" print -- so a passing
 * PASS/FAIL log line followed by SIGABRT is the reproduction, matching the real
 * test's "[teardown] all done, exiting cleanly" then abort.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <ishmem.h>
#include <ishmemx.h>
#include <sycl/sycl.hpp>

namespace {

constexpr size_t kIshmemWGSize = 32;

int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : def;
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    ishmem_init();
    const int my_pe = ishmem_my_pe();
    const int npes = ishmem_n_pes();

    const int do_finalize = env_int("REPRO_FINALIZE", 0);
    const int payload_bytes = env_int("REPRO_PAYLOAD_BYTES", 4096);

    sycl::queue q;
    if (my_pe == 0) {
        std::printf("[finrepro] npes=%d REPRO_FINALIZE=%d payload_bytes=%d device=%s\n",
                    npes,
                    do_finalize,
                    payload_bytes,
                    q.get_device().get_info<sycl::info::device::name>().c_str());
        std::fflush(stdout);
    }

    const size_t pbytes = static_cast<size_t>(payload_bytes);
    uint8_t* send = static_cast<uint8_t*>(ishmem_malloc(pbytes));
    uint8_t* recv = static_cast<uint8_t*>(ishmem_malloc(pbytes * npes));
    q.memset(send, my_pe & 0xFF, pbytes).wait();
    q.memset(recv, 0, pbytes * npes).wait();
    ishmem_barrier_all();

    const int me = my_pe;
    const int n = npes;

    // Exercise the IBGDA host-proxy progress thread with a blocking put to every
    // peer + a device-wide barrier -- the same primitives the normal-path
    // CombinedDispatchRdmaPutKernel uses, guaranteeing proxy_thread is running.
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kIshmemWGSize), sycl::range<1>(kIshmemWGSize)), [=](sycl::nd_item<1> item) {
             auto group = item.get_group();
             if (group.leader()) {
                 for (int dst = 0; dst < n; ++dst) {
                     if (dst == me) continue;
                     ishmem_putmem(recv + static_cast<size_t>(me) * pbytes, send, pbytes, dst);
                 }
             }
             sycl::group_barrier(group);
             if (group.leader()) ishmem_barrier_all();
             sycl::group_barrier(group);
         });
     }).wait();

    ishmem_barrier_all();
    std::printf("[finrepro] PE=%d workload done -> exiting cleanly (finalize=%s)\n",
                my_pe,
                do_finalize ? "yes" : "no");
    std::fflush(stdout);

    ishmem_free(send);
    ishmem_free(recv);

    if (do_finalize) {
        // The fix: ishmem_finalize() -> ishmemi_proxy_fini() -> proxy_thread.join().
        // With the proxy thread joined, its global std::thread destructor at exit
        // is a no-op and the process exits rc=0.
        int initialized = 0;
        ishmemx_query_initialized(&initialized);
        if (initialized) {
            ishmem_finalize();
        }
        if (my_pe == 0) {
            std::printf("[finrepro] ===== ishmem_finalize() returned : clean shutdown expected (rc=0) =====\n");
            std::fflush(stdout);
        }
    } else {
        // DeepEP's default: skip ishmem_finalize(). proxy_thread stays joinable;
        // its global std::thread destructor at process exit calls std::terminate()
        // -> "terminate called without an active exception" -> SIGABRT (rc=134).
        if (my_pe == 0) {
            std::printf("[finrepro] ===== NO ishmem_finalize(): expect 'terminate called without an "
                        "active exception' + SIGABRT at exit (rc=134) =====\n");
            std::fflush(stdout);
        }
    }

    // main returns -> C++ runtime destroys global proxy_thread here.
    return EXIT_SUCCESS;
}
