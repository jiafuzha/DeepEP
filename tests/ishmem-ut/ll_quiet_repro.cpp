/* Minimal iSHMEM reproducer for the DeepEP low-latency IBGDA "native_flush"
 * REGRESSION introduced by the upstream iSHMEM-IBGDA fix that repaired the
 * internode-NORMAL hang.
 *
 * WHAT BROKE
 * ----------
 * On the real 2-node internode LOW-LATENCY test, every run now aborts on the
 * very first dispatch, before any bandwidth number is printed:
 *
 *     [IBGDA native_flush CQE_ERROR] wc[0].status=9      (IBV_WC_REM_OP_ERR)
 *     IBGDA native_flush: completion failed status=9
 *     ERROR: poll -- native_flush failed in quiet_flag path        (flag path)
 *     ... or ...
 *     ibgda native_flush failed for ordering op barrier_all        (barrier path)
 *     Abort(-1) ... MPI_Abort(MPI_COMM_WORLD, -1)
 *
 * The failing primitive is IBGDA `native_flush`, which iSHMEM invokes to drain
 * this PE's outbound RDMA send queue whenever the device code QUIETS or
 * BARRIERS after issuing device-initiated `ishmem_putmem_nbi`. That is exactly
 * the DeepEP low-latency dispatch/combine sequence (csrc/xpu/internode_ll.cpp):
 *
 *   Stage 2 (per remote channel, one work-item):
 *       ishmem_putmem_nbi(dispatch_data + slot, send_data + slot, bytes, dst);
 *       ishmem_putmem_nbi(dispatch_count + slot, &count,          4,     dst);
 *   Stage 3 drain, ONE of:
 *       - BARRIER PATH (DEEP_EP_LL_FLAG_PROGRESS=0, default):
 *             ishmemx_barrier_all_work_group(group);   // internal quiet+flush
 *       - FLAG PATH    (DEEP_EP_LL_FLAG_PROGRESS=1):
 *             ishmemx_quiet_work_group(group);          // explicit quiet+flush
 *
 * BOTH drains call IBGDA native_flush internally, so BOTH abort with status=9
 * on the broken build -- confirming the failure is in the flush path, not in
 * the flag-progress logic (the env flag is NOT the trigger).
 *
 * WHAT THIS PROGRAM DOES
 * ----------------------
 * No DeepEP / torch. Every PE issues one device-initiated `ishmem_putmem_nbi`
 * to every OTHER PE (one work-item per destination, mirroring the LL dispatch's
 * one-WI-per-channel remote put), then drains via the selected primitive. On a
 * healthy iSHMEM the drain returns and the program prints PASS + exits 0. On the
 * broken build the drain aborts with `native_flush ... status=9`.
 *
 * ENV KNOBS
 *   REPRO_SYNC = quiet (default) | barrier | host_quiet | host_barrier
 *       quiet        -> ishmexx_quiet_work_group  (device)   == FLAG path
 *                       reproduces "native_flush failed in quiet_flag path"
 *       barrier      -> ishmemx_barrier_all_work_group (device) == BARRIER path
 *                       reproduces "native_flush failed for ordering op barrier_all"
 *       host_quiet   -> ishmem_quiet()            (host)  -- isolates device vs host
 *       host_barrier -> ishmem_barrier_all()      (host)
 *   REPRO_PAYLOAD_INTS  (default 4096)  ints put per (src -> dst) slot
 *   REPRO_ITERS         (default 1)     back-to-back dispatch-like iterations
 *
 * EXPECTED RESULT (this ISHMEM build): ABORT with IBGDA native_flush status=9
 * on the first drain (any device REPRO_SYNC mode), rc!=0. A fixed ISHMEM build
 * prints "[llquiet] ... PASS" on every PE and exits 0.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <ishmem.h>
#include <ishmemx.h>
#include <sycl/sycl.hpp>

namespace {

int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return def;
    return std::atoi(v);
}

std::string env_str(const char* name, const char* def) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return std::string(def);
    return std::string(v);
}

// Deterministic per-(source, index) payload so any receiver can independently
// recompute what it should have received from a given source PE.
inline int encode(int src_pe, int idx) { return src_pe * 1000003 + idx * 7 + 1; }

constexpr int kSelfCount = 123;

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    ishmem_init();
    const int my_pe = ishmem_my_pe();
    const int npes = ishmem_n_pes();

    const int payload_ints = env_int("REPRO_PAYLOAD_INTS", 4096);
    const int iters = env_int("REPRO_ITERS", 1);
    const std::string sync_mode = env_str("REPRO_SYNC", "quiet");
    const bool valid_mode = (sync_mode == "quiet" || sync_mode == "barrier" ||
                             sync_mode == "host_quiet" || sync_mode == "host_barrier");

    sycl::queue q;
    if (my_pe == 0) {
        std::printf("[llquiet] npes=%d payload_ints=%d iters=%d sync=%s%s device=%s\n",
                    npes, payload_ints, iters, sync_mode.c_str(),
                    valid_mode ? "" : " (INVALID->falling back to quiet)",
                    q.get_device().get_info<sycl::info::device::name>().c_str());
        std::printf("[llquiet] ===== EXPECT: broken ISHMEM aborts here with "
                    "'IBGDA native_flush ... status=9'; fixed ISHMEM prints PASS =====\n");
        std::fflush(stdout);
    }

    // Symmetric receive region: one per-source slot for every PE.
    const size_t count_elems = static_cast<size_t>(npes);
    const size_t data_elems = static_cast<size_t>(npes) * payload_ints;
    int* recv_count = static_cast<int*>(ishmem_malloc(count_elems * sizeof(int)));
    int* recv_data = static_cast<int*>(ishmem_malloc(data_elems * sizeof(int)));

    // Symmetric send staging.
    int* send_data = static_cast<int*>(ishmem_malloc(static_cast<size_t>(payload_ints) * sizeof(int)));
    int* send_count = static_cast<int*>(ishmem_malloc(sizeof(int)));

    const int me = my_pe;
    q.parallel_for(sycl::range<1>(payload_ints), [=](sycl::id<1> i) {
         send_data[i] = encode(me, static_cast<int>(i));
     }).wait();
    q.single_task([=]() { send_count[0] = kSelfCount; }).wait();

    int total_fail = 0;

    for (int it = 0; it < iters; ++it) {
        q.fill(recv_count, -1, count_elems).wait();
        q.fill(recv_data, -1, data_elems).wait();
        ishmem_barrier_all();  // clean, agreed start state

        // Stage 1: device-initiated remote NBI puts, one work-item per dst
        // (mirrors LL dispatch's one-WI-per-channel remote put that loads the
        // single per-PE IBGDA QP send queue that native_flush must later drain).
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(npes), sycl::range<1>(npes)),
                              [=](sycl::nd_item<1> item) {
                                  const int dst = static_cast<int>(item.get_global_id(0));
                                  if (dst == me) {
                                      for (int i = 0; i < payload_ints; ++i) {
                                          recv_data[static_cast<size_t>(me) * payload_ints + i] = send_data[i];
                                      }
                                      recv_count[me] = send_count[0];
                                      return;
                                  }
                                  ishmem_putmem_nbi(&recv_data[static_cast<size_t>(me) * payload_ints],
                                                    send_data,
                                                    static_cast<size_t>(payload_ints) * sizeof(int),
                                                    dst);
                                  ishmem_putmem_nbi(&recv_count[me], send_count, sizeof(int), dst);
                              });
         }).wait();

        // Stage 2: DRAIN the outbound puts. This is where IBGDA native_flush
        // runs -- and where the broken build aborts with status=9.
        if (sync_mode == "barrier") {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(npes), sycl::range<1>(npes)),
                                  [=](sycl::nd_item<1> item) { ishmemx_barrier_all_work_group(item.get_group()); });
             }).wait();
        } else if (sync_mode == "host_quiet") {
            ishmem_quiet();
            ishmem_barrier_all();  // make peers' puts visible before we verify
        } else if (sync_mode == "host_barrier") {
            ishmem_barrier_all();
        } else {  // "quiet" (default) == DEEP_EP_LL_FLAG_PROGRESS=1 flag path
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(npes), sycl::range<1>(npes)),
                                  [=](sycl::nd_item<1> item) { ishmemx_quiet_work_group(item.get_group()); });
             }).wait();
            // The flag path relies on the receiver polling per-source flags; a
            // barrier here just lets this isolation UT verify delivery cleanly.
            ishmem_barrier_all();
        }

        // Stage 3: verify delivery from every source (self + all peers).
        int* fail = sycl::malloc_host<int>(npes, q);
        for (int s = 0; s < npes; ++s) fail[s] = 0;
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::range<1>(npes), [=](sycl::id<1> sid) {
                 const int s = static_cast<int>(sid);
                 int bad = 0;
                 if (recv_count[s] != kSelfCount) bad = 1;
                 for (int i = 0; i < payload_ints; ++i) {
                     if (recv_data[static_cast<size_t>(s) * payload_ints + i] != encode(s, i)) {
                         bad = 1;
                         break;
                     }
                 }
                 fail[s] = bad;
             });
         }).wait();

        int delivered = 0;
        std::string missing;
        for (int s = 0; s < npes; ++s) {
            if (fail[s] == 0) {
                ++delivered;
            } else {
                if (!missing.empty()) missing += ",";
                missing += std::to_string(s);
            }
        }
        const bool ok = (delivered == npes);
        if (!ok) ++total_fail;
        std::printf("[llquiet] iter=%d PE=%d sync=%s delivered=%d/%d %s missing_sources=[%s]\n",
                    it, my_pe, sync_mode.c_str(), delivered, npes,
                    ok ? "PASS" : "FAIL", missing.c_str());
        std::fflush(stdout);
        sycl::free(fail, q);
        ishmem_barrier_all();
    }

    ishmem_barrier_all();
    if (my_pe == 0) {
        std::printf("[llquiet] ===== PE0 local result: %s (drain survived; native_flush OK) =====\n",
                    total_fail == 0 ? "PASS" : "FAIL");
        std::fflush(stdout);
    }

    ishmem_free(recv_count);
    ishmem_free(recv_data);
    ishmem_free(send_data);
    ishmem_free(send_count);
    ishmem_finalize();

    return total_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
