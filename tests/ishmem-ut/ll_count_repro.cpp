/* Standalone iSHMEM reproducer for the DeepEP low-latency DATA-CORRECTNESS
 * REGRESSION introduced by the *newer* iSHMEM-IBGDA build.
 *
 * WHAT CHANGED
 * -----------
 * The first suspect iSHMEM build aborted the LL dispatch on the very first
 * device drain with `IBGDA native_flush ... status=9` (see ll_quiet_repro.cpp).
 * The freshly rebuilt iSHMEM (ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install)
 * NO LONGER aborts -- native_flush now returns cleanly -- but the LL test then
 * fails a CORRECTNESS assertion instead:
 *
 *     [rank0]: AssertionError: 42 != 56      (received != expected tokens)
 *     [rank2]: AssertionError: 44 != 56
 *     [rank3]: AssertionError: 31 != 64
 *     [rank1]: AssertionError: Error: diff=0.49...   (payload bytes wrong)
 *
 * i.e. some device-initiated `ishmem_putmem_nbi` transfers are SILENTLY DROPPED
 * or their per-channel completion count is delivered WRONG. Fewer tokens land on
 * the receiver than the sender put, so packed_recv_count / cumulative stats /
 * ground-truth topk counts disagree, and the payload bytes for the surviving
 * rows can be stale. This is a data-plane regression, distinct from the earlier
 * native_flush crash.
 *
 * WHY ll_quiet_repro.cpp DOES NOT CATCH IT
 * ----------------------------------------
 * ll_quiet_repro issues exactly ONE fixed-size put per (src->dst) pair and only
 * asks "did the drain survive?" (native_flush). The real dispatch issues, from
 * ONE work-item per channel sharing the single per-PE QP:
 *     (a) a VARIABLE-length payload put   (count rows x hidden bytes), then
 *     (b) a 4-byte per-expert COUNT/flag put (SET = -count-1)
 * across (num_ranks-1) x num_local_experts channels concurrently. The dropped
 * transfers only show up when you (1) send variable counts per channel and
 * (2) verify BOTH the delivered count flag AND every payload row. This UT does
 * exactly that, with no DeepEP / torch dependency.
 *
 * WHAT THIS PROGRAM DOES  (mirrors csrc/xpu/internode_ll.cpp Stage 2/3)
 * --------------------------------------------------------------------
 * Receiver symmetric buffers, per (local_expert le, source PE src):
 *     dispatch_count[le*npes + src]                         : int flag (=-count-1)
 *     dispatch_data [(le*npes + src)*max_tokens + row]*rowI : payload rows
 * Every PE, one work-item per (dst, le) channel:
 *     count = channel_count(me, dst, le)   (deterministic, variable, some 0)
 *     ishmem_putmem_nbi(payload rows -> dst slot)      // (a) variable-size
 *     ishmem_putmem_nbi(-count-1 flag  -> dst slot)    // (b) 4-byte SET
 * (dst==me is a local copy). Then DRAIN via REPRO_SYNC, then the receiver
 * checks, for every (le, src): decoded flag count == expected count, and every
 * payload row == encode(src, le, row, i). Any mismatch => token loss/corruption.
 *
 * ENV KNOBS
 *   REPRO_SYNC          quiet (default) | barrier
 *       quiet   -> FLAG-POLL path (== DEEP_EP_LL_FLAG_PROGRESS=1): sender drains
 *                  its OWN queue (ishmemx_quiet_work_group); receiver SPIN-POLLS
 *                  each per-source count flag (NO global barrier) then reads the
 *                  payload -- exactly reproducing the receiver-poll that surfaces
 *                  a dropped/late flag (undercount) or flag-before-payload
 *                  reordering (stale payload) on the broken ISHMEM.
 *       barrier -> BARRIER path (LL default): global ishmemx_barrier_all_work_group
 *                  between put and verify (drains + synchronizes all PEs).
 *   REPRO_MAX_TOKENS    per-channel max rows            (default 64)
 *   REPRO_LOCAL_EXPERTS local experts per PE            (default 2)
 *   REPRO_ROW_INTS      ints per payload row (row_bytes=4*this, default 3584
 *                       -> 14 KiB/row, matching hidden=7168 bf16)
 *   REPRO_ITERS         back-to-back dispatch iterations (default 4)
 *   REPRO_MAX_PUT_KB    chunk payload puts to <= this KiB (default 0 = one put)
 *   REPRO_POLL_CAP      flag-poll spin cap per slot     (default 1000000)
 *   REPRO_SENDER_FENCE  1 = release(system) fence before flag put (default 1)
 *   REPRO_RECV_ACQ      1 = acquire(system) fence after flag seen (default 1)
 *
 * EXPECTED RESULT
 *   Fixed iSHMEM : every PE prints "[llcount] ... PASS", exit 0.
 *   Broken iSHMEM: REPRO_SYNC=quiet (flag-poll) shows at least one (le,src) with
 *                  got=0 exp>0 -> "[llcount] ... FAIL ... got=X exp=Y", exit != 0,
 *                  reproducing the "X != Y" token-count regression.
 *
 * CONFIRMED ISOLATION (new iSHMEM /root/jiafuzha/ishmem_ibgda, 2x2 PEs)
 *   REPRO_SYNC=barrier -> ALL PASS  (a global barrier forces completion, so the
 *                          payload AND the counts all arrive: data is NOT corrupt)
 *   REPRO_SYNC=quiet    -> FAIL     (got=0 exp=16, tokens got=19 exp=227, ...):
 *                          the small 4-byte SET count-flag put is NOT made visible
 *                          to a device-side spin-poll within POLL_CAP, so those
 *                          sources are counted as 0 tokens -> undercount.
 *   => The regression is specifically that the flag-progress path's per-expert
 *      4-byte flag RDMA WRITE is not delivered/ordered to a polling receiver on
 *      the new iSHMEM; the barrier path (global quiet+sync) still works.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>

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

// Deterministic per-channel token count in [0, max_tokens]. Varies with the
// (source, destination, local-expert) triple so different channels carry
// different amounts -- including some zero-length channels -- exactly the
// irregular traffic that exposes dropped puts.
inline int channel_count(int src, int dst, int le, int max_tokens, int it) {
    unsigned h = static_cast<unsigned>(src) * 2654435761u + static_cast<unsigned>(dst) * 40503u +
                 static_cast<unsigned>(le) * 2246822519u + static_cast<unsigned>(it) * 3266489917u;
    return static_cast<int>(h % static_cast<unsigned>(max_tokens + 1));
}

// Deterministic payload value: any receiver can recompute what row i from a
// given (source, local-expert, row) must contain, so a stale/dropped payload
// is detected byte-for-byte.
inline int encode(int src, int le, int row, int i) {
    return src * 1000003 + le * 100003 + row * 131 + i * 7 + 1;
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    ishmem_init();
    const int my_pe = ishmem_my_pe();
    const int npes = ishmem_n_pes();

    const int max_tokens = env_int("REPRO_MAX_TOKENS", 64);
    const int local_experts = env_int("REPRO_LOCAL_EXPERTS", 2);
    const int row_ints = env_int("REPRO_ROW_INTS", 3584);
    const int iters = env_int("REPRO_ITERS", 4);
    const size_t max_put_kb = static_cast<size_t>(env_int("REPRO_MAX_PUT_KB", 0));
    const uint64_t poll_cap = static_cast<uint64_t>(env_int("REPRO_POLL_CAP", 1000000));
    const int sender_fence = env_int("REPRO_SENDER_FENCE", 1);
    const int recv_acq = env_int("REPRO_RECV_ACQ", 1);
    // Per-iteration cross-PE sync AFTER clearing the receive buffers and BEFORE
    // the puts. The real flag-progress (DEEP_EP_LL_FLAG_PROGRESS=1) path has NO
    // such global barrier between iterations -- it relies on next_clean ordering
    // alone. Set REPRO_RESET_BARRIER=0 to drop it and expose the barrier-free
    // race the real perf path actually runs (a fast peer's put can land in the
    // window where its target is still verifying / re-clearing the prior iter).
    const int reset_barrier = env_int("REPRO_RESET_BARRIER", 1);
    // Per-PE host skew (microseconds) injected before each iteration's put stage
    // to DESYNCHRONIZE the PEs (= my_pe * skew_us). Widens the barrier-free
    // cross-iteration clear/put race window the real flag path is exposed to.
    const int skew_us = env_int("REPRO_SKEW_US", 0);
    std::string sync_mode = env_str("REPRO_SYNC", "quiet");
    if (sync_mode != "quiet" && sync_mode != "barrier") sync_mode = "quiet";

    const size_t row_bytes = static_cast<size_t>(row_ints) * sizeof(int);
    const size_t max_put_bytes = max_put_kb ? max_put_kb * 1024 : 0;

    sycl::queue q;
    if (my_pe == 0) {
        std::printf("[llcount] npes=%d local_experts=%d max_tokens=%d row_ints=%d (row=%zuB) "
                    "iters=%d sync=%s max_put_kb=%zu device=%s\n",
                    npes, local_experts, max_tokens, row_ints, row_bytes, iters, sync_mode.c_str(),
                    max_put_kb, q.get_device().get_info<sycl::info::device::name>().c_str());
        std::printf("[llcount] ===== EXPECT: fixed ISHMEM -> all PASS; broken ISHMEM -> at least one "
                    "'delivered_count != expected' or payload mismatch (token loss) =====\n");
        std::fflush(stdout);
    }

    // Receiver symmetric buffers: slot per (local_expert, source PE).
    const size_t nslots = static_cast<size_t>(local_experts) * npes;
    const size_t data_ints = nslots * static_cast<size_t>(max_tokens) * row_ints;
    int* dispatch_count = static_cast<int*>(ishmem_malloc(nslots * sizeof(int)));
    int* dispatch_data = static_cast<int*>(ishmem_malloc(data_ints * sizeof(int)));

    // Sender symmetric staging: per (dst, local_expert) channel payload + flag.
    const size_t nchan = static_cast<size_t>(npes) * local_experts;
    const size_t send_data_ints = nchan * static_cast<size_t>(max_tokens) * row_ints;
    int* send_data = static_cast<int*>(ishmem_malloc(send_data_ints * sizeof(int)));
    int* send_flag = static_cast<int*>(ishmem_malloc(nchan * sizeof(int)));

    // Host-visible per-slot failure map (indexed by le*npes + src).
    int* fail = sycl::malloc_host<int>(static_cast<int>(nslots), q);
    int* got = sycl::malloc_host<int>(static_cast<int>(nslots), q);
    int* exp = sycl::malloc_host<int>(static_cast<int>(nslots), q);

    const int me = my_pe;
    int total_fail = 0;

    for (int it = 0; it < iters; ++it) {
        // Inject per-PE host skew so the PEs fall out of lockstep: a lagging
        // receiver runs its buffer clear (fill below) LATE, after a fast peer has
        // already put its flag -> the clear clobbers the delivered flag = token
        // loss, the barrier-free hazard the real flag path is exposed to.
        if (skew_us > 0 && me > 0)
            std::this_thread::sleep_for(std::chrono::microseconds(skew_us * me));
        // Stage the payload + flag this PE will send on every channel.
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::range<1>(nchan), [=](sycl::id<1> cid) {
                 const int ch = static_cast<int>(cid);
                 const int dst = ch / local_experts;
                 const int le = ch % local_experts;
                 const int count = channel_count(me, dst, le, max_tokens, it);
                 send_flag[ch] = -count - 1;  // SET encoding, matches internode_ll
                 const size_t base = static_cast<size_t>(ch) * max_tokens * row_ints;
                 for (int r = 0; r < count; ++r)
                     for (int i = 0; i < row_ints; ++i)
                         send_data[base + static_cast<size_t>(r) * row_ints + i] = encode(me, le, r, i);
             });
         }).wait();

        // Reset receiver buffers to a sentinel so dropped puts are visible.
        q.fill(dispatch_count, 0, nslots).wait();      // 0 == not-arrived (flag encoding)
        q.fill(dispatch_data, -777, data_ints).wait();
        if (reset_barrier)
            ishmem_barrier_all();

        // Stage 2: one work-item per (dst, le) channel. Payload put first, then
        // the 4-byte flag put -- same ordering as the flag path in internode_ll.
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::range<1>(nchan), [=](sycl::id<1> cid) {
                 const int ch = static_cast<int>(cid);
                 const int dst = ch / local_experts;
                 const int le = ch % local_experts;
                 const int count = -send_flag[ch] - 1;
                 const size_t src_base = static_cast<size_t>(ch) * max_tokens * row_ints;
                 // Destination slot uses THIS PE's rank as the source index.
                 const int dslot = le * npes + me;
                 const size_t dst_base = static_cast<size_t>(dslot) * max_tokens * row_ints;

                 if (dst == me) {
                     for (int r = 0; r < count; ++r)
                         for (int i = 0; i < row_ints; ++i)
                             dispatch_data[dst_base + static_cast<size_t>(r) * row_ints + i] =
                                 send_data[src_base + static_cast<size_t>(r) * row_ints + i];
                     dispatch_count[dslot] = send_flag[ch];
                     return;
                 }
                 if (count > 0) {
                     const size_t bytes = static_cast<size_t>(count) * row_bytes;
                     uint8_t* dptr = reinterpret_cast<uint8_t*>(dispatch_data + dst_base);
                     const uint8_t* sptr = reinterpret_cast<const uint8_t*>(send_data + src_base);
                     if (max_put_bytes == 0 || bytes <= max_put_bytes) {
                         ishmem_putmem_nbi(dptr, sptr, bytes, dst);
                     } else {
                         size_t off = 0;
                         while (off < bytes) {
                             size_t chunk = bytes - off;
                             if (chunk > max_put_bytes) chunk = max_put_bytes;
                             ishmem_putmem_nbi(dptr + off, sptr + off, chunk, dst);
                             off += chunk;
                         }
                     }
                 }
                 // Per-expert completion flag (SET), delivered after the payload
                 // on the same in-order RC QP (QPS_PER_PE=1). A release(system)
                 // fence mirrors ll_sender_flush / DEEP_EP_LL_FLAG_SENDER_FENCE=1
                 // so the NIC DMA-reads the freshly-staged flag, not a stale value.
                 if (sender_fence)
                     sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                 ishmem_putmem_nbi(&dispatch_count[dslot], &send_flag[ch], sizeof(int), dst);
             });
         }).wait();

        // Stage 3: drain + make peers' puts visible. Two paths mirror internode_ll.
        if (sync_mode == "barrier") {
            // BARRIER path: a global device barrier drains all outbound NBI puts
            // and synchronizes every PE before verify.
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(npes), sycl::range<1>(npes)),
                                  [=](sycl::nd_item<1> item) { ishmemx_barrier_all_work_group(item.get_group()); });
             }).wait();
        } else {
            // FLAG-POLL path (== DEEP_EP_LL_FLAG_PROGRESS=1). NO global barrier:
            // (a) drain THIS PE's own outbound puts, then (b) spin-poll each of
            // this receiver's per-(le, src) flag slots until it lands (or times
            // out -> left 0 -> treated as 0 tokens, i.e. token loss is recorded).
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(
                     sycl::nd_range<1>(sycl::range<1>(npes), sycl::range<1>(npes)), [=](sycl::nd_item<1> item) {
                         auto group = item.get_group();
                         ishmemx_quiet_work_group(group);
                         sycl::group_barrier(group);
                         const int lid = static_cast<int>(item.get_local_id(0));
                         const int lsize = static_cast<int>(item.get_local_range(0));
                         for (int s = lid; s < static_cast<int>(nslots); s += lsize) {
                             const int src = s % npes;
                             if (src == me) continue;  // self slot written locally
                             uint64_t spins = 0;
			     /* Use system-scope atomic load to bypass GPU L3 cache.
                              * NIC DMA writes bypass L3; without invalidation the
                              * GPU keeps reading stale cached zeros forever. */
                             sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                              sycl::memory_scope::system,
                                              sycl::access::address_space::global_space>
                                 flag_ref(dispatch_count[s]);
                             while (flag_ref.load(sycl::memory_order::acquire) == 0) {
                             //while (dispatch_count[s] == 0) {
                                 if (++spins >= poll_cap) break;  // timeout -> 0 tokens
                             }
                         }
                     });
             }).wait();
        }

        // Stage 4: verify every (local_expert, source) slot on this receiver.
        // REPRO_VERIFY selects the receiver-read STRUCTURE:
        //   perslot  (default) - one work-item per slot; the acquire happens right
        //                        before that slot's payload read (this is coherent).
        //   packlike           - mirrors csrc/xpu/internode_ll.cpp PACK kernel: a
        //                        SINGLE work-item does the acquire ONCE at entry,
        //                        then loops over ALL slots reading the flag
        //                        (atomic-acquire) + payload (cached). This exposes
        //                        the real-code staleness: the once-at-entry acquire
        //                        does NOT keep later slots' payload L3 lines fresh.
        //   packlike_fix       - packlike but re-issues the acquire before EACH
        //                        slot's payload read (candidate fix).
        for (size_t s = 0; s < nslots; ++s) {
            fail[s] = 0;
            got[s] = 0;
            exp[s] = 0;
        }
        const std::string verify_mode = env_str("REPRO_VERIFY", "perslot");
        auto read_flag_atomic = [](int* p) {
            sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                             sycl::access::address_space::global_space>
                r(*p);
            return r.load(sycl::memory_order::acquire);
        };
        if (verify_mode == "packlike" || verify_mode == "packlike_fix") {
            const bool per_slot_acq = (verify_mode == "packlike_fix");
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                     if (recv_acq)  // ll_recv_acquire ONCE at "kernel entry"
                         sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                     for (int slot = 0; slot < static_cast<int>(nslots); ++slot) {
                         const int le = slot / npes;
                         const int src = slot % npes;
                         const int expected = channel_count(src, me, le, max_tokens, it);
                         if (per_slot_acq)
                             sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                         const int raw = read_flag_atomic(&dispatch_count[slot]);
                         const int delivered = (raw == 0) ? 0 : (-raw - 1);
                         got[slot] = delivered;
                         exp[slot] = expected;
                         int bad = (delivered != expected) ? 1 : 0;
                         const size_t base = static_cast<size_t>(slot) * max_tokens * row_ints;
                         for (int r = 0; r < delivered && bad == 0; ++r)
                             for (int i = 0; i < row_ints; ++i)
                                 if (dispatch_data[base + static_cast<size_t>(r) * row_ints + i] != encode(src, le, r, i)) {
                                     bad = 1;
                                     break;
                                 }
                         fail[slot] = bad;
                     }
                 });
             }).wait();
        } else {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::range<1>(nslots), [=](sycl::id<1> sid) {
                     const int slot = static_cast<int>(sid);
                     const int le = slot / npes;
                     const int src = slot % npes;
                     const int expected = channel_count(src, me, le, max_tokens, it);
                     // Acquire(system) so the CACHED payload reads below observe the
                     // NIC-delivered bytes (mirrors ll_recv_acquire / RECV_ACQ=1).
                     if (recv_acq)
                         sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                     const int raw = dispatch_count[slot];
                     const int delivered = (raw == 0) ? 0 : (-raw - 1);  // flag decode
                     got[slot] = delivered;
                     exp[slot] = expected;
                     int bad = (delivered != expected) ? 1 : 0;
                     // Verify payload bytes for the rows the flag says arrived. A
                     // flag-before-payload reordering yields delivered==expected but
                     // STALE payload (sentinel -777) -> caught here.
                     const int rows = delivered;
                     const size_t base = static_cast<size_t>(slot) * max_tokens * row_ints;
                     for (int r = 0; r < rows && bad == 0; ++r) {
                         for (int i = 0; i < row_ints; ++i) {
                             if (dispatch_data[base + static_cast<size_t>(r) * row_ints + i] != encode(src, le, r, i)) {
                                 bad = 1;
                                 break;
                             }
                         }
                     }
                     fail[slot] = bad;
                 });
             }).wait();
        }

        int slots_ok = 0, tok_exp = 0, tok_got = 0, bad_slots = 0;
        std::string first_bad;
        for (size_t s = 0; s < nslots; ++s) {
            tok_exp += exp[s];
            tok_got += got[s];
            if (fail[s] == 0) {
                ++slots_ok;
            } else {
                ++bad_slots;
                if (first_bad.empty()) {
                    const int le = static_cast<int>(s) / npes;
                    const int src = static_cast<int>(s) % npes;
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "le=%d src=%d got=%d exp=%d", le, src, got[s], exp[s]);
                    first_bad = buf;
                }
            }
        }
        const bool ok = (bad_slots == 0);
        if (!ok) ++total_fail;
        std::printf("[llcount] iter=%d PE=%d sync=%s slots_ok=%d/%zu tokens got=%d exp=%d %s%s%s\n",
                    it, my_pe, sync_mode.c_str(), slots_ok, nslots, tok_got, tok_exp,
                    ok ? "PASS" : "FAIL",
                    ok ? "" : " first_bad_slot=", ok ? "" : first_bad.c_str());
        std::fflush(stdout);
        // Per-iteration cross-PE barrier. The real flag-progress path has NONE;
        // gating it under reset_barrier lets the PEs run free so a fast peer's
        // next-iter put can race this receiver's clear (token loss).
        if (reset_barrier)
            ishmem_barrier_all();
    }

    ishmem_barrier_all();
    if (my_pe == 0) {
        std::printf("[llcount] ===== PE0 overall: %s =====\n", total_fail == 0 ? "PASS" : "FAIL");
        std::fflush(stdout);
    }

    sycl::free(fail, q);
    sycl::free(got, q);
    sycl::free(exp, q);
    ishmem_free(dispatch_count);
    ishmem_free(dispatch_data);
    ishmem_free(send_data);
    ishmem_free(send_flag);
    ishmem_finalize();

    return total_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
