/* Minimal iSHMEM device-initiated PUT reproducer for the DeepEP low-latency
 * "token-delivery" correctness bug.
 *
 * WHY THIS EXISTS
 * ---------------
 * The DeepEP low-latency dispatch kernel (csrc/xpu/internode_ll.cpp,
 * dispatch_bf16) delivers a token's payload from the sending PE to the
 * owning expert's PE using DEVICE-INITIATED RDMA:
 *
 *     Stage 2 (per remote channel, one work-item):
 *         ishmem_putmem_nbi(dispatch_data + slot, send_data + slot, bytes, dst_rank);
 *         ishmem_putmem_nbi(dispatch_count + slot, &count,           4,     dst_rank);
 *     Stage 3 (default DEEP_EP_LL_FLAG_PROGRESS=0, "barrier path"):
 *         ishmemx_barrier_all_work_group(group);
 *     Stage 4:
 *         read dispatch_count[le*num_ranks + src_rank] to learn how many
 *         tokens each source PE sent to this local expert.
 *
 * With a broken iSHMEM build, the device-initiated remote NBI puts (and/or the
 * device barrier that is supposed to make them globally visible) never land on
 * the peer PE. The receiver therefore only ever sees the ONE slot it filled
 * locally (its own rank), so every expert counts ~1/num_ranks of the tokens it
 * should have received -- exactly the observed failures like "8 != 27".
 *
 * This program reproduces that failure in isolation, WITHOUT any DeepEP / torch
 * machinery: every PE puts a uniquely-encoded payload + a count SET-value into
 * every OTHER PE's per-source slot, does the same device barrier, then verifies
 * that it received correct data from ALL peers. A healthy iSHMEM delivers all
 * num_pes slots; the broken iSHMEM delivers only the self slot (1/num_pes).
 *
 * ENV KNOBS
 *   REPRO_BARRIER = device (default) | host
 *       device -> ishmemx_barrier_all_work_group (exactly what dispatch uses)
 *       host   -> ishmem_barrier_all (isolates "is the device barrier the
 *                 culprit, or the NBI put itself?")
 *   REPRO_PAYLOAD_INTS  (default 4096)  ints put per (src -> dst) slot
 *   REPRO_ITERS         (default 1)     repeat count (matches back-to-back
 *                                       dispatches in the real test)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <ishmem.h>
#include <ishmemx.h>
#include <sycl/sycl.hpp>

namespace {

int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return def;
    return std::atoi(v);
}

// Deterministic per-(source, index) payload value. Depends ONLY on the sender's
// PE id and the element index, so any receiver can independently recompute the
// value it should have received from a given source.
inline int encode(int src_pe, int idx) { return src_pe * 1000003 + idx * 7 + 1; }

constexpr int kSelfCount = 123;  // count value each PE advertises to its peers

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    ishmem_init();
    const int my_pe = ishmem_my_pe();
    const int npes = ishmem_n_pes();

    const int payload_ints = env_int("REPRO_PAYLOAD_INTS", 4096);
    const int iters = env_int("REPRO_ITERS", 1);
    const std::string barrier_mode = std::getenv("REPRO_BARRIER") ? std::getenv("REPRO_BARRIER") : "device";
    const bool device_barrier = (barrier_mode != "host");

    sycl::queue q;
    if (my_pe == 0) {
        std::printf("[repro] npes=%d payload_ints=%d iters=%d barrier=%s device=%s\n",
                    npes,
                    payload_ints,
                    iters,
                    barrier_mode.c_str(),
                    q.get_device().get_info<sycl::info::device::name>().c_str());
        std::fflush(stdout);
    }

    // Symmetric receive region: one per-source slot for every PE.
    //   recv_count[s] : count advertised by source PE s (SET value)
    //   recv_data[s * payload_ints + i] : payload element i from source PE s
    const size_t count_elems = static_cast<size_t>(npes);
    const size_t data_elems = static_cast<size_t>(npes) * payload_ints;
    int* recv_count = static_cast<int*>(ishmem_malloc(count_elems * sizeof(int)));
    int* recv_data = static_cast<int*>(ishmem_malloc(data_elems * sizeof(int)));

    // Symmetric send staging (source of the puts). Every PE stages ONE payload
    // (encoded with its own PE id) plus one count value.
    int* send_data = static_cast<int*>(ishmem_malloc(static_cast<size_t>(payload_ints) * sizeof(int)));
    int* send_count = static_cast<int*>(ishmem_malloc(sizeof(int)));

    const int me = my_pe;
    q.parallel_for(sycl::range<1>(payload_ints), [=](sycl::id<1> i) { send_data[i] = encode(me, static_cast<int>(i)); }).wait();
    q.single_task([=]() { send_count[0] = kSelfCount; }).wait();

    int total_fail = 0;

    for (int it = 0; it < iters; ++it) {
        // Stage 0: reset the receive region to a sentinel so a "not delivered"
        // slot is distinguishable from a legitimately delivered value.
        q.fill(recv_count, -1, count_elems).wait();
        q.fill(recv_data, -1, data_elems).wait();
        ishmem_barrier_all();  // everyone starts from a clean, agreed state

        // Stage 1 (mirror of dispatch Stage 2): device-initiated puts.
        // One work-item per destination channel, exactly like the dispatch
        // kernel bounds remote puts to one work-item per (dst_rank, local_expert)
        // channel to avoid oversubscribing the single QP per PE.
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(npes), sycl::range<1>(npes)), [=](sycl::nd_item<1> item) {
                 const int dst = static_cast<int>(item.get_global_id(0));
                 if (dst == me) {
                     // Self slot: filled locally, never over the NIC -- this is
                     // the slot that "survives" when remote delivery is broken.
                     for (int i = 0; i < payload_ints; ++i) {
                         recv_data[static_cast<size_t>(me) * payload_ints + i] = send_data[i];
                     }
                     recv_count[me] = send_count[0];
                     return;
                 }
                 // Remote slot on peer `dst`, at this PE's source index `me`.
                 ishmem_putmem_nbi(&recv_data[static_cast<size_t>(me) * payload_ints],
                                   send_data,
                                   static_cast<size_t>(payload_ints) * sizeof(int),
                                   dst);
                 ishmem_putmem_nbi(&recv_count[me], send_count, sizeof(int), dst);
             });
         }).wait();

        // Stage 2 (mirror of dispatch Stage 3): make the puts globally visible.
        if (device_barrier) {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(npes), sycl::range<1>(npes)),
                                  [=](sycl::nd_item<1> item) { ishmemx_barrier_all_work_group(item.get_group()); });
             }).wait();
        } else {
            ishmem_barrier_all();
        }

        // Stage 3: verify. Every PE must have received a correct slot from EVERY
        // source (self + all peers).
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
        int self_ok = 0;
        std::string missing;
        for (int s = 0; s < npes; ++s) {
            if (fail[s] == 0) {
                ++delivered;
                if (s == my_pe) self_ok = 1;
            } else {
                if (!missing.empty()) missing += ",";
                missing += std::to_string(s);
            }
        }
        const bool ok = (delivered == npes);
        if (!ok) ++total_fail;
        std::printf("[repro] iter=%d PE=%d delivered=%d/%d self_ok=%d %s missing_sources=[%s]\n",
                    it,
                    my_pe,
                    delivered,
                    npes,
                    self_ok,
                    ok ? "PASS" : "FAIL",
                    missing.c_str());
        std::fflush(stdout);
        sycl::free(fail, q);
        ishmem_barrier_all();
    }

    // Global pass/fail: each PE returns non-zero on local failure and mpirun
    // propagates any non-zero child exit, so no cross-PE reduction is needed
    // (a device-side gather here is both unnecessary and a teardown hazard).
    // A single clean barrier makes the per-PE PASS/FAIL lines above flush in a
    // deterministic order before finalize.
    ishmem_barrier_all();
    if (my_pe == 0) {
        std::printf("[repro] ===== PE0 local result: %s (see per-PE lines above; "
                    "launcher greps FAIL / checks exit code) =====\n",
                    total_fail == 0 ? "PASS" : "FAIL");
        std::fflush(stdout);
    }

    ishmem_free(recv_count);
    ishmem_free(recv_data);
    ishmem_free(send_data);
    ishmem_free(send_count);
    ishmem_finalize();

    // Every PE returns non-zero on local failure; mpirun returns non-zero if ANY
    // rank fails, and the launcher also greps the per-PE output for "FAIL".
    return total_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
