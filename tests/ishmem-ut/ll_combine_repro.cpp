/* Standalone iSHMEM reproducer for the DeepEP low-latency COMBINE-path payload
 * staleness regression (the residual `diff=...` assertion that survives the
 * dispatch flag-read fix).
 *
 * WHY ll_count_repro.cpp DOES NOT CATCH THE COMBINE `diff`
 * -------------------------------------------------------
 * ll_count_repro faithfully models the DISPATCH flag exchange and PASSES with
 * the system-scope acquire flag read. But the real LL test still reports
 *   [rankN]: AssertionError: Error: diff=0.58...
 * which is a COMBINE-stage numeric mismatch, not a dispatch count loss. The
 * combine RECEIVE path (csrc/xpu/internode_ll.cpp Stage 3/4) differs from
 * dispatch in the one way that matters for L3 coherence:
 *
 *   - Stage 3 (WAIT): a SEPARATE kernel spin-polls the per-global-expert
 *     combine_flag with an acquire atomic load. Good -- the flags become
 *     coherent.
 *   - Stage 4 (REDUCE): a DIFFERENT, GRID-PARALLEL kernel. Each work-item does
 *     ONE ll_recv_acquire (a sycl::atomic_fence(acquire, system)) at kernel
 *     entry, then strides over MANY combine_data elements reading them CACHED
 *     (flag_recv_acq != 0, the default). There is NO per-element / per-expert
 *     atomic-acquire LOAD touching the payload region.
 *
 * On BMG + mlx5 IBGDA a NIC RDMA-WRITE lands in VRAM but BYPASSES GPU L3. A
 * plain acquire FENCE orders memory ops but does NOT evict already-cached L3
 * lines. So when the receiver's L3 already holds a stale copy of combine_data
 * (from the cached sentinel clear of this iter, or the cached reduce reads of a
 * PRIOR iter reusing the same symmetric addresses), the fence-guarded CACHED
 * read in Stage 4 returns the STALE bytes -> the weighted reduction is slightly
 * off -> `diff`. Only an actual atomic-ACQUIRE LOAD (or an uncached load)
 * invalidates the line and returns the NIC-delivered value.
 *
 * WHAT THIS PROGRAM DOES  (mirrors internode_ll.cpp combine Stage 2/3/4)
 * ---------------------------------------------------------------------
 * num_experts = npes * local_experts; expert ge is OWNED by rank ge/local_experts.
 * The owner combines and SENDS its expert's rows to every dst_rank:
 *     combine_data[ge][token][i]   payload rows (variable per (ge,dst))
 *     combine_flag[ge]             4-byte completion flag (SET = 1)
 * Receiver:
 *     Stage 3: quiet + spin-poll combine_flag[ge] (acquire atomic load).
 *     Stage 4: GRID-PARALLEL, per-work-item ONE up-front acquire, then strided
 *              CACHED reads of combine_data verified byte-for-byte.
 * The cached-sentinel clear + no-barrier + cached reduce reads reproduce the
 * fence-does-not-invalidate-L3 staleness. REPRO_RECV_ACQ selects the read mode:
 *     0 = per-element uncached uc_load           (coherent -> PASS)
 *     1 = one acquire FENCE + cached reads        (DEFAULT real path -> FAIL)
 *     3 = per-element system-scope acquire LOAD   (candidate FIX -> PASS)
 *
 * ENV KNOBS
 *   REPRO_SYNC          quiet (default) | barrier
 *   REPRO_RECV_ACQ      0 | 1 (default) | 3           (see above)
 *   REPRO_MAX_TOKENS    per-(ge,dst) max rows          (default 64)
 *   REPRO_LOCAL_EXPERTS experts owned per PE            (default 2)
 *   REPRO_ROW_INTS      ints per row (~hidden/2)        (default 3584)
 *   REPRO_ITERS         iterations                      (default 8)
 *   REPRO_MAX_PUT_KB    chunk size for payload puts     (default 64, 0=one put)
 *   REPRO_POLL_CAP      flag spin cap                   (default 1000000)
 *   REPRO_SENDER_FENCE  release fence before flag put   (default 1)
 *   REPRO_RESET_BARRIER per-iter cross-PE barrier       (default 0 = expose bug)
 *   REPRO_CLEAR         cached (default) | uc           receiver buffer clear
 */

#include <cstdint>
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

// Deterministic per-channel token count in [0, max_tokens], varying with
// (expert, destination, iteration) so channels carry irregular amounts.
inline int channel_count(int ge, int dst, int max_tokens, int it) {
    unsigned h = static_cast<unsigned>(ge) * 2654435761u + static_cast<unsigned>(dst) * 40503u +
                 static_cast<unsigned>(it) * 3266489917u + 101u;
    return static_cast<int>(h % static_cast<unsigned>(max_tokens + 1));
}

// Deterministic payload value for (expert, token, i, iteration).
inline int encode(int ge, int token, int i, int it) {
    return ge * 1000003 + token * 131 + i * 7 + it * 2246822519u + 1;
}

// Uncached (L3-bypassing) 32-bit load (mirrors csrc/xpu/xpu_kernels.hpp uc_load).
inline int uc_load_i32(const int* ptr) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t v = *reinterpret_cast<const volatile uint32_t*>(
        __builtin_intel_sycl_ptr_annotation(reinterpret_cast<uint32_t*>(const_cast<int*>(ptr)), "sycl-cache-read-hint", 0x7));
    int r;
    __builtin_memcpy(&r, &v, 4);
    return r;
#else
    return *ptr;
#endif
}

// Uncached (write-through) 32-bit store.
inline void uc_store_i32(int* ptr, int value) {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t v;
    __builtin_memcpy(&v, &value, 4);
    *reinterpret_cast<volatile uint32_t*>(
        __builtin_intel_sycl_ptr_annotation(reinterpret_cast<uint32_t*>(ptr), "sycl-cache-write-hint", 0x7)) = v;
#else
    *ptr = value;
#endif
}

inline int acq_load_i32(int* p) {
    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        r(*p);
    return r.load(sycl::memory_order::acquire);
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    ishmem_init();
    const int me = ishmem_my_pe();
    const int npes = ishmem_n_pes();

    const int max_tokens = env_int("REPRO_MAX_TOKENS", 64);
    const int local_experts = env_int("REPRO_LOCAL_EXPERTS", 2);
    const int row_ints = env_int("REPRO_ROW_INTS", 3584);
    const int iters = env_int("REPRO_ITERS", 8);
    const size_t max_put_kb = static_cast<size_t>(env_int("REPRO_MAX_PUT_KB", 64));
    const uint64_t poll_cap = static_cast<uint64_t>(env_int("REPRO_POLL_CAP", 1000000));
    const int sender_fence = env_int("REPRO_SENDER_FENCE", 1);
    const int recv_acq = env_int("REPRO_RECV_ACQ", 1);
    const int reset_barrier = env_int("REPRO_RESET_BARRIER", 0);
    std::string sync_mode = env_str("REPRO_SYNC", "quiet");
    if (sync_mode != "quiet" && sync_mode != "barrier") sync_mode = "quiet";
    std::string clear_mode = env_str("REPRO_CLEAR", "cached");

    const int num_experts = npes * local_experts;
    const size_t row_bytes = static_cast<size_t>(row_ints) * sizeof(int);
    const size_t max_put_bytes = max_put_kb ? max_put_kb * 1024 : 0;

    sycl::queue q;
    if (me == 0) {
        std::printf("[llcombine] npes=%d local_experts=%d num_experts=%d max_tokens=%d row_ints=%d "
                    "iters=%d sync=%s recv_acq=%d clear=%s max_put_kb=%zu device=%s\n",
                    npes, local_experts, num_experts, max_tokens, row_ints, iters, sync_mode.c_str(), recv_acq,
                    clear_mode.c_str(), max_put_kb, q.get_device().get_info<sycl::info::device::name>().c_str());
        std::printf("[llcombine] ===== EXPECT: recv_acq=0/3 -> PASS; recv_acq=1 (fence+cached) -> "
                    "payload staleness (diff) on broken stack =====\n");
        std::fflush(stdout);
    }

    // Receiver symmetric combine buffers: per global expert.
    const size_t cdata_ints = static_cast<size_t>(num_experts) * max_tokens * row_ints;
    int* combine_data = static_cast<int*>(ishmem_malloc(cdata_ints * sizeof(int)));
    int* combine_flag = static_cast<int*>(ishmem_malloc(static_cast<size_t>(num_experts) * sizeof(int)));

    // Sender symmetric staging: per (dst_rank, local_expert) channel payload + flag.
    const size_t nchan = static_cast<size_t>(npes) * local_experts;
    const size_t sdata_ints = nchan * static_cast<size_t>(max_tokens) * row_ints;
    int* send_data = static_cast<int*>(ishmem_malloc(sdata_ints * sizeof(int)));
    int* send_flag = static_cast<int*>(ishmem_malloc(nchan * sizeof(int)));

    // Host-visible per-expert mismatch map + element mismatch counter.
    int* efail = sycl::malloc_host<int>(num_experts, q);
    int* egot = sycl::malloc_host<int>(num_experts, q);
    int* mism = sycl::malloc_shared<int>(1, q);

    int total_fail = 0;

    for (int it = 0; it < iters; ++it) {
        // Stage the payload + flag this owner PE sends on each channel.
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::range<1>(nchan), [=](sycl::id<1> cid) {
                 const int ch = static_cast<int>(cid);
                 const int dst = ch / local_experts;
                 const int le = ch % local_experts;
                 const int ge = me * local_experts + le;
                 const int count = channel_count(ge, dst, max_tokens, it);
                 send_flag[ch] = 1;
                 const size_t base = static_cast<size_t>(ch) * max_tokens * row_ints;
                 for (int r = 0; r < count; ++r)
                     for (int i = 0; i < row_ints; ++i)
                         send_data[base + static_cast<size_t>(r) * row_ints + i] = encode(ge, r, i, it);
             });
         }).wait();

        // Clear receiver combine buffers to a sentinel. A CACHED clear (default)
        // populates L3 with the sentinel so a later fence-guarded cached reduce
        // read can return it (the staleness we reproduce). REPRO_CLEAR=uc clears
        // write-through (mirrors the real uc_store next_clean) as an A/B.
        if (clear_mode == "uc") {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::range<1>(cdata_ints), [=](sycl::id<1> i) { uc_store_i32(&combine_data[i], -777); });
             }).wait();
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::range<1>(num_experts), [=](sycl::id<1> i) { uc_store_i32(&combine_flag[i], 0); });
             }).wait();
        } else {
            q.fill(combine_data, -777, cdata_ints).wait();
            q.fill(combine_flag, 0, num_experts).wait();
        }
        mism[0] = 0;
        if (reset_barrier) ishmem_barrier_all();

        // Stage 2: combine put. One work-item per (dst_rank, le) channel: payload
        // put first, then the 4-byte completion flag (SET) on the same in-order QP.
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::range<1>(nchan), [=](sycl::id<1> cid) {
                 const int ch = static_cast<int>(cid);
                 const int dst = ch / local_experts;
                 const int le = ch % local_experts;
                 const int ge = me * local_experts + le;
                 const int count = channel_count(ge, dst, max_tokens, it);
                 const size_t src_base = static_cast<size_t>(ch) * max_tokens * row_ints;
                 const size_t dst_base = static_cast<size_t>(ge) * max_tokens * row_ints;

                 if (dst == me) {
                     for (int r = 0; r < count; ++r)
                         for (int i = 0; i < row_ints; ++i)
                             combine_data[dst_base + static_cast<size_t>(r) * row_ints + i] =
                                 send_data[src_base + static_cast<size_t>(r) * row_ints + i];
                     uc_store_i32(&combine_flag[ge], 1);  // self-owned flag set locally
                     return;
                 }
                 if (count > 0) {
                     const size_t bytes = static_cast<size_t>(count) * row_bytes;
                     uint8_t* dptr = reinterpret_cast<uint8_t*>(combine_data + dst_base);
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
                 // Per-expert completion flag (SET), delivered after the payload on
                 // the same in-order RC QP. Release fence mirrors ll_sender_flush.
                 if (sender_fence)
                     sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                 ishmem_putmem_nbi(&combine_flag[ge], &send_flag[ch], sizeof(int), dst);
             });
         }).wait();

        // Stage 3: drain + wait. Mirrors combine Stage 3.
        if (sync_mode == "barrier") {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(npes), sycl::range<1>(npes)),
                                  [=](sycl::nd_item<1> item) { ishmemx_barrier_all_work_group(item.get_group()); });
             }).wait();
        } else {
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(
                     sycl::nd_range<1>(sycl::range<1>(npes), sycl::range<1>(npes)), [=](sycl::nd_item<1> item) {
                         auto group = item.get_group();
                         ishmemx_quiet_work_group(group);
                         sycl::group_barrier(group);
                         const int lid = static_cast<int>(item.get_local_id(0));
                         const int lsize = static_cast<int>(item.get_local_range(0));
                         for (int ge = lid; ge < num_experts; ge += lsize) {
                             const int owner = ge / local_experts;
                             if (owner == me) continue;  // self-owned flag set locally
                             uint64_t spins = 0;
                             // System-scope acquire load so the flag becomes coherent
                             // (the dispatch fix; combine uses the same primitive).
                             sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                              sycl::access::address_space::global_space>
                                 fref(combine_flag[ge]);
                             while (fref.load(sycl::memory_order::acquire) == 0) {
                                 if (++spins >= poll_cap) break;  // timeout -> reduce reads whatever landed
                             }
                         }
                     });
             }).wait();
        }

        // Stage 4: GRID-PARALLEL reduce read, mirroring internode_ll Stage 4. Each
        // work-item does ONE up-front acquire (recv_acq==1), then strides over many
        // combine_data elements reading them CACHED -- the exact structure where a
        // fence that does not invalidate L3 yields stale payload. recv_acq==0 reads
        // uncached; recv_acq==3 issues a per-element acquire LOAD (candidate fix).
        for (int e = 0; e < num_experts; ++e) { efail[e] = 0; egot[e] = 0; }
        int* mism_ptr = mism;
        {
            const size_t work = cdata_ints;
            const int wg = 256;
            const int nwg = 512;  // fixed grid, strided loop (like ll_num_wgs cap)
            q.submit([&](sycl::handler& cgh) {
                 cgh.parallel_for(
                     sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(nwg) * wg), sycl::range<1>(wg)),
                     [=](sycl::nd_item<1> item) {
                         const size_t gid = item.get_global_linear_id();
                         const size_t gsize = item.get_global_range(0);
                         // ll_recv_acquire ONCE at kernel entry (per work-item).
                         if (recv_acq == 1)
                             sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                         const size_t per_expert = static_cast<size_t>(max_tokens) * row_ints;
                         for (size_t idx = gid; idx < work; idx += gsize) {
                             const int ge = static_cast<int>(idx / per_expert);
                             const int rem = static_cast<int>(idx % per_expert);
                             const int token = rem / row_ints;
                             const int i = rem % row_ints;
                             const int owner = ge / local_experts;
                             if (owner == me) continue;  // self-owned verified trivially
                             const int count = channel_count(ge, me, max_tokens, it);
                             if (token >= count) continue;  // only rows the owner sent
                             int v;
                             if (recv_acq == 0)
                                 v = uc_load_i32(&combine_data[idx]);
                             else if (recv_acq == 3)
                                 v = acq_load_i32(&combine_data[idx]);
                             else
                                 v = combine_data[idx];  // cached (recv_acq==1)
                             if (v != encode(ge, token, i, it)) {
                                 sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                  sycl::access::address_space::global_space>
                                     mref(*mism_ptr);
                                 mref.fetch_add(1);
                                 sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                  sycl::access::address_space::global_space>
                                     eref(efail[ge]);
                                 eref.store(1);
                             }
                         }
                     });
             }).wait();
        }

        int bad_experts = 0;
        std::string first_bad;
        for (int e = 0; e < num_experts; ++e) {
            if (efail[e]) {
                ++bad_experts;
                if (first_bad.empty()) {
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "ge=%d owner=%d", e, e / local_experts);
                    first_bad = buf;
                }
            }
        }
        const bool ok = (mism[0] == 0);
        if (!ok) ++total_fail;
        std::printf("[llcombine] iter=%d PE=%d sync=%s recv_acq=%d bad_experts=%d mismatched_ints=%d %s%s%s\n",
                    it, me, sync_mode.c_str(), recv_acq, bad_experts, mism[0], ok ? "PASS" : "FAIL",
                    ok ? "" : " first_bad=", ok ? "" : first_bad.c_str());
        std::fflush(stdout);
        if (reset_barrier) ishmem_barrier_all();
    }

    ishmem_barrier_all();
    if (me == 0) {
        std::printf("[llcombine] ===== PE0 overall: %s =====\n", total_fail == 0 ? "PASS" : "FAIL");
        std::fflush(stdout);
    }

    sycl::free(efail, q);
    sycl::free(egot, q);
    sycl::free(mism, q);
    ishmem_free(combine_data);
    ishmem_free(combine_flag);
    ishmem_free(send_data);
    ishmem_free(send_flag);
    ishmem_finalize();

    return total_fail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
