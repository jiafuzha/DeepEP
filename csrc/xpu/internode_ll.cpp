#include <c10/util/Float8_e4m3fn.h>

#include "xpu_kernels.hpp"

#ifdef DEEP_EP_ENABLE_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

// ============================================================================
// Two receiver-progress paths (dispatch & combine). Both move the same data via
// ishmem_putmem_nbi over RDMA; they differ ONLY in how the receiver learns the
// RDMA writes have landed before reading them. Selected by ll_flag_progress()
// (env DEEP_EP_LL_FLAG_PROGRESS, default 0 = barrier path).
//
//   1. BARRIER PATH  (DEFAULT, DEEP_EP_LL_FLAG_PROGRESS unset/0)
//      - Sync: one global ishmemx_barrier_all_work_group between the put stage
//        and the pack/reduce stage.
//      - The barrier doubles as a system-scope acquire/invalidate, making GPU L2
//        coherent with NIC RDMA writes -> receiver reads delivered data with
//        FAST CACHED loads.
//      - Latency is gated by the slowest peer (all-to-all sync); cheap at low PE
//        counts, grows with scale. Proven-stable, including true 2-node.
//
//   2. FLAG PATH  (opt-in, DEEP_EP_LL_FLAG_PROGRESS=1)
//      - Sync: NO global barrier. Instead (a) a LOCAL ishmemx_quiet_work_group
//        drains only this PE's outbound NBI puts, then (b) per-expert completion
//        flags/counts that the receiver SPIN-POLLS (bounded by DEEP_EP_LL_POLL_CAP).
//      - With no barrier there is no acquire/invalidate and BMG's L2 is NOT
//        coherent with NIC RDMA writes, so EVERY delivered byte (flag + payload)
//        must be read with L2-bypassing uncached uc_load (RECEIVER side).
//      - SENDER side: the flag VALUE is staged via uc_store then RDMA-WRITE-put;
//        uc_store's write-through hint only bypasses L1, so the NIC could DMA-read
//        a STALE L3-resident 0 and transmit it -> receiver spins forever ->
//        watchdog DEVICE_LOST on 2-node. A sender-side release/flush fence between
//        the uc_store and the put (ll_sender_flush, DEEP_EP_LL_FLAG_SENDER_FENCE,
//        DEFAULT 1 = sycl::atomic_fence(release,system); 2 = LSC evict.sysrel)
//        evicts the value to the memory domain the NIC reads. With this flush the
//        flag path PASSES true 2-node (avg ~1.15-1.24 ms, no DEVICE_LOST); it was
//        the missing piece (NOT the receiver read primitive, NOT the NIC PCIe
//        domain, NOT transport latency).
//      - Removes the all-to-all sync so latency no longer tracks the slowest peer
//        (mirrors CUDA's amo_nonfetch_add + receiver-poll). Still pays an
//        uncached-read tax per byte (loopback ~1861 vs barrier ~1130 us/iter).
//
// uc_load is confirmed to be a genuine uncached load (equivalent to explicit LSC
// .uc.uc asm and the only correct way to read post-RDMA data without a barrier);
// see ll_flag_progress() below for the full rationale and the flip-to-flags
// criteria. Tier-2's job is to close the flag path's per-byte read-cost gap.
// ============================================================================

namespace deep_ep {
namespace internode_ll {
namespace {

class CleanLowLatencyBufferKernel;
class UpdateMaskBufferKernel;
class QueryMaskBufferKernel;
class CleanMaskBufferKernel;
class LowLatencyDispatchMergedKernel;
class LowLatencyCombineMergedKernel;

// Tier-1 multi-work-group low-latency kernels (multi-WG grid + sub-group
// collectives). The merged single-256-WI-work-group kernels above are kept for
// reference / fallback; the active dispatch/combine paths now use these.
class LowLatencyDispatchRouteKernel;
class LowLatencyDispatchPutKernel;
class LowLatencyDispatchBarrierKernel;
class LowLatencyDispatchWaitKernel;
class LowLatencyDispatchPackKernel;
class LowLatencyCombineScatterKernel;
class LowLatencyCombinePutKernel;
class LowLatencyCombineBarrierKernel;
class LowLatencyCombineWaitKernel;
class LowLatencyCombineReduceKernel;
class LowLatencyDispatchCleanFlagKernel;
class LowLatencyCombineCleanCountKernel;

struct LowLatencyLayout {
    size_t dispatch_data_bytes;
    size_t dispatch_src_bytes;
    size_t dispatch_count_bytes;
    size_t send_data_bytes;
    size_t send_src_bytes;
    size_t send_count_bytes;
    size_t combine_data_bytes;
    size_t combine_flag_bytes;
    size_t mask_bytes;

    size_t dispatch_data_offset;
    size_t dispatch_src_offset;
    size_t dispatch_count_offset;
    size_t send_data_offset;
    size_t send_src_offset;
    size_t send_count_offset;
    size_t combine_data_offset;
    size_t combine_flag_offset;
    size_t mask_offset;
    size_t sync_offset;
    size_t total_bytes;
};

inline LowLatencyLayout make_layout(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    const int num_local_experts = num_experts / num_ranks;
    LowLatencyLayout l{};
    const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    const size_t num_dispatch_slots = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
    const size_t num_send_slots = static_cast<size_t>(num_ranks) * num_local_experts * num_max_dispatch_tokens_per_rank;
    const size_t num_combine_slots = static_cast<size_t>(num_experts) * num_max_dispatch_tokens_per_rank;
    l.dispatch_data_bytes = num_dispatch_slots * hidden_bytes;
    l.dispatch_src_bytes = num_dispatch_slots * sizeof(int);
    l.dispatch_count_bytes = static_cast<size_t>(num_local_experts) * num_ranks * sizeof(int);
    l.send_data_bytes = num_send_slots * hidden_bytes;
    l.send_src_bytes = num_send_slots * sizeof(int);
    l.send_count_bytes = static_cast<size_t>(num_ranks) * num_local_experts * sizeof(int);
    l.combine_data_bytes = num_combine_slots * hidden_bytes;
    l.combine_flag_bytes = static_cast<size_t>(num_experts) * sizeof(uint64_t);
    l.mask_bytes = static_cast<size_t>(num_ranks) * sizeof(int);

    size_t offset = 0;
    auto add = [&](size_t bytes) {
        const size_t old = offset;
        offset = align_up<size_t>(offset + bytes, NUM_BUFFER_ALIGNMENT_BYTES);
        return old;
    };
    l.dispatch_data_offset = add(l.dispatch_data_bytes);
    l.dispatch_src_offset = add(l.dispatch_src_bytes);
    l.dispatch_count_offset = add(l.dispatch_count_bytes);
    l.send_data_offset = add(l.send_data_bytes);
    l.send_src_offset = add(l.send_src_bytes);
    l.send_count_offset = add(l.send_count_bytes);
    l.combine_data_offset = add(l.combine_data_bytes);
    l.combine_flag_offset = add(l.combine_flag_bytes);
    l.mask_offset = add(l.mask_bytes);
    l.sync_offset = add(static_cast<size_t>(num_ranks) * sizeof(int));
    l.total_bytes = offset;
    return l;
}

inline uint64_t pack_range(int count, int begin) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(begin)) << 32) | static_cast<uint32_t>(count);
}

inline void unpack_range(int64_t packed, int& count, int& begin) {
    uint64_t u = static_cast<uint64_t>(packed);
    count = static_cast<int>(u & 0xffffffffu);
    begin = static_cast<int>(u >> 32);
}

inline bool ll_rank_masked(int* mask_buffer_ptr, int rank) {
    return mask_buffer_ptr != nullptr && plain_load(mask_buffer_ptr + rank) != 0;
}

inline sycl::ext::oneapi::bfloat16 bf16_from_float(float value) {
    return sycl::ext::oneapi::bfloat16(value);
}

inline uint8_t ue8m0_from_float(float value) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(value);
    return static_cast<uint8_t>(bits >> 23);
}

constexpr int kLowLatencyMergedGroupSize = 256;

// Tier-1 multi-WG tuning.
constexpr int kLLWGSize = 256;  // work-items per work-group
constexpr int kLLMaxWGs = 256;  // cap on grid size for grid-stride phases

// Maximum bytes per single iSHMEM NBI put. On this BMG + mlx5 IBGDA stack a
// single ishmem_putmem_nbi larger than ~256 KiB falls into a pathologically
// slow transport path (RERING / landing-spin) that costs a fixed ~0.5 s per
// call regardless of size. Splitting large puts into <=192 KiB chunks keeps
// every put on the fast path. Overridable via DEEP_EP_LL_MAX_PUT_KB.
inline size_t ll_max_put_bytes() {
    const char* env = std::getenv("DEEP_EP_LL_MAX_PUT_KB");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0)
            return static_cast<size_t>(v) * 1024;
    }
    return static_cast<size_t>(64) * 1024;
}

inline int ll_num_wgs(size_t work_units, int wg_size, int cap) {
    const char* env = std::getenv("DEEP_EP_LL_NUM_WGS");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0)
            return v;
    }
    size_t n = (work_units + static_cast<size_t>(wg_size) - 1) / static_cast<size_t>(wg_size);
    if (n < 1)
        n = 1;
    if (n > static_cast<size_t>(cap))
        n = static_cast<size_t>(cap);
    return static_cast<int>(n);
}

// Tier-2: flag-based asynchronous progress. When enabled (default), the global
// `ishmemx_barrier_all_work_group` between the put and pack/reduce stages is
// replaced by (a) a local `ishmemx_quiet_work_group` to drain this PE's outbound
// NBI puts (keeps the shared send_data staging safe to reuse) plus (b) per-expert
// RDMA completion flags that the receiver spin-polls. This mirrors the CUDA
// internode_ll design (amo_nonfetch_add completion signal + receiver poll) and
// removes the all-to-all synchronization point so latency no longer grows with
// the slowest peer.
//
// DEFAULT: OFF (opt-in via DEEP_EP_LL_FLAG_PROGRESS=1). The flag path is
// validated-correct on single-node loopback, but on this BMG+mlx5 stack it is
// NOT a net win and is unstable at multi-node scale:
//   * Small-scale loopback: ~1861 us/iter vs the barrier path's ~1130 us/iter.
//     BMG's GPU L2 is not coherent with NIC RDMA writes, so the flag path must
//     use L2-bypassing uncached (uc_load) reads for every RDMA-delivered byte.
//     The removed global barrier_all previously supplied the acquire/invalidate
//     that enabled fast CACHED reads, and at low PE counts that barrier is cheap
//     -- so the uncached-read cost dominates and the flag path is slower. There
//     is no portable BMG L2-invalidate primitive to permit cached post-RDMA
//     reads (confirmed: only uc_load/uc_store work; fences do not).
//   * True 2-node: the flag path COULD hit UR_RESULT_ERROR_DEVICE_LOST. When a
//     cross-node flag was delayed, the bounded uncached spin ran long enough
//     (with the old 2e9 / 50M cap) to trip the GPU hang-check watchdog before
//     the poll cap was reached -> hard device wedge. FIXED by lowering the
//     default poll cap to 1,000,000 (see ll_poll_cap() below): the worst-case
//     spin is now a few ms, safely under the watchdog, so a delayed flag
//     degrades to "0 tokens" (a soft, test-caught failure) instead of a wedge.
//     With a CLEAN driver/env, the flag path is reliable at 32 & 64 tokens; the
//     residual DEVICE_LOST seen historically was accumulated NIC/QP wedge state
//     across runs, not a flag-path bug (reset the igub driver between runs to
//     avoid it -- DEEP_EP_LL_RESET_DRIVER=1 in the docker-2node-ll harness).
// The barrier path (this default) is the proven-stable route. Flip to flags only
// once scale makes the all-to-all barrier the dominant cost AND the cross-node
// flag-landing/transport stability is resolved.
inline bool ll_flag_progress() {
    const char* env = std::getenv("DEEP_EP_LL_FLAG_PROGRESS");
    if (env != nullptr && env[0] != '\0') {
        return std::atoi(env) != 0;
    }
    return false;
}

// Independent combine-side toggle (defaults to the dispatch setting). Lets us A/B
// isolate a fault to the dispatch vs combine flag path: DEEP_EP_LL_COMBINE_FLAG=0
// forces combine back onto the proven barrier path while dispatch stays on flags.
inline bool ll_flag_progress_combine() {
    const char* env = std::getenv("DEEP_EP_LL_COMBINE_FLAG");
    if (env != nullptr && env[0] != '\0') {
        return std::atoi(env) != 0;
    }
    return ll_flag_progress();
}

// DIAGNOSTIC: when set, the flag-path Stage-3 WaitKernel does a global
// ishmemx_barrier_all_work_group before the spin (isolates "SET flag lands" from
// "quiet+spin waits correctly"). Remove after debugging.
inline bool ll_flag_diag_barrier() {
    const char* env = std::getenv("DEEP_EP_LL_FLAG_DIAG_BARRIER");
    return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
}

// A/B toggle for the flag-path dispatch_count read primitive. Default 0 keeps
// the hint-based uc_load (sycl-cache-read-hint 0x7, L1-uncached). Set
// DEEP_EP_LL_FLAG_LSC=1 to read the per-expert flag via explicit LSC
// `lsc_load.ugm.uc.uc` (L1+L3 uncached at the GenISA message level) to test
// whether 2-node correctness/staleness differs from the hint. Optionally also
// issues an `lsc_fence.ugm.invalidate.sysacq` before each read when value >= 2.
inline int ll_flag_lsc_mode() {
    const char* env = std::getenv("DEEP_EP_LL_FLAG_LSC");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0)
            return v;
    }
    return 0;
}

// Sender-side flush fence mode for the flag path. After uc_store-ing the encoded
// flag value into the symmetric send staging, the NIC DMA-reads that staging to
// RDMA-WRITE it into the receiver's slot. If the GPU has not flushed the store
// past L3 to the system memory domain the NIC reads, the NIC can transmit a STALE
// value (e.g. 0), so the receiver spins forever (-> watchdog DEVICE_LOST on
// 2-node) even with a perfect uncached read. This emits a release/flush fence
// between the uc_store and the flag put:
//   0 = none   1 = sycl::atomic_fence(release, system) [DEFAULT, portable]
//   2 = LSC lsc_fence.ugm.evict.sysrel (explicit L3 evict to memory, ~1% faster)
// DEFAULT 1: the flag path is INCORRECT on 2-node without this flush (empirically
// the NIC reads a stale 0); set DEEP_EP_LL_FLAG_SENDER_FENCE=0 only to reproduce
// the broken behavior, =2 to use the explicit LSC evict.
inline int ll_flag_sender_fence() {
    const char* env = std::getenv("DEEP_EP_LL_FLAG_SENDER_FENCE");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v >= 0)
            return v;
    }
    return 1;
}

// Design-(b) cooperative sub-group aggregated NBI put path (opt-in,
// DEEP_EP_LL_COOP_PUT=1, default 0). When enabled (and flag_progress is on) the
// dispatch/combine remote puts are issued via ishmemx_putmem_nbi_sub_group so a
// single leader per (sub-group, QP) publishes SND_DBR monotonically for all its
// lanes — removing the concurrent-producer doorbell race. Requires the ishmem
// build that exports ishmemx_putmem_nbi_sub_group. Keep OFF until validated on HW.
inline bool ll_coop_put() {
    const char* env = std::getenv("DEEP_EP_LL_COOP_PUT");
    return env != nullptr && env[0] == '1';
}

// Device-side flag read used by the flag path. lsc_mode 0 = hint-based uc_load
// (L1 uncached); 1 = explicit LSC `lsc_load.ugm.uc.uc` (L1+L3 uncached); >=2 =
// LSC load preceded by an `invalidate.sysacq` cache-invalidate fence.
inline int ll_read_flag(const int* p, int lsc_mode) {
#ifdef __SYCL_DEVICE_ONLY__
    if (lsc_mode >= 2) {
        lsc_fence_sysacq();
    }
    if (lsc_mode >= 1) {
        return lsc_uc_load_i32(p);
    }
#endif
    return uc_load(p);
}

// Device-side sender-side flush fence (see ll_flag_sender_fence). Forces freshly
// uc_store-d staging bytes out to the memory domain the NIC DMA-reads.
inline void ll_sender_flush(int mode) {
#ifdef __SYCL_DEVICE_ONLY__
    if (mode == 1) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    } else if (mode >= 2) {
        lsc_fence_sysrel();
    }
#endif
    (void)mode;
}

// Receiver-side bulk-read acquire mode for the flag path. The default flag path
// USED to read EVERY delivered payload byte with an uncached uc_load (L2-bypassing),
// so each byte paid a memory round-trip. Better: once the per-expert flags confirm
// the RDMA payload has landed, issue a SINGLE system-scope acquire / cache-invalidate
// at the start of the read kernel so the GPU L2 becomes coherent with the NIC writes,
// then read the bulk payload with FAST CACHED loads (the same mechanism the barrier
// path relies on). Trades a per-byte uncached tax for one invalidate + cached reads.
//   0 = per-byte uncached uc_load (fallback; ~15% slower on loopback)
//   1 = one sycl::atomic_fence(acquire, system) at kernel entry + cached reads (DEFAULT)
//   2 = one LSC lsc_fence.ugm.invalidate.sysacq at kernel entry + cached reads
//       (UNSTABLE on 2-node: reproducibly trips UR_RESULT_ERROR_DEVICE_LOST; do NOT use)
// Measured (flag path on): loopback ~1581 us (mode 1) vs ~1870 us (mode 0); true 2-node
// is transport-bound so all modes ~1150 us. Mode 1 is correct on both and faster on
// loopback, so it is the default; mode 0 stays as an opt-out fallback.
inline int ll_flag_recv_acq() {
    const char* env = std::getenv("DEEP_EP_LL_FLAG_RECV_ACQ");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v >= 0)
            return v;
    }
    return 1;
}

// Device-side receiver acquire/invalidate (see ll_flag_recv_acq). Issued ONCE per
// work-item at the start of the payload-read kernel, before any cached load.
inline void ll_recv_acquire(int mode) {
#ifdef __SYCL_DEVICE_ONLY__
    if (mode == 1) {
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    } else if (mode >= 2) {
        lsc_fence_sysacq();
    }
#endif
    (void)mode;
}

// Spin-poll iteration cap for flag waits (BMG has no portable cycle counter, so
// a bounded busy-spin replaces CUDA's clock64 timeout). Overridable via
// DEEP_EP_LL_POLL_CAP. On timeout the slot is treated as "0 tokens" (graceful
// degradation), matching the rank-mask-on-timeout intent of the CUDA path.
//
// DEFAULT = 1,000,000. This is deliberately LOW. The receiver's flag wait is a
// tight loop of L2-bypassing uncached uc_load reads; each read is a full VRAM
// round-trip (~µs). A warm-QP RDMA flag lands within a handful of µs, so 1M
// iterations (a few ms of spin) has ample margin. Crucially, 1M keeps the
// WORST-CASE spin (a delayed / never-landing flag under a marginal env) well
// under the GPU hang-check/heartbeat watchdog interval. A large cap (the old
// 2e9 / the harness's 50M) lets a single delayed flag spin for tens of seconds,
// tripping the watchdog -> UR_RESULT_ERROR_DEVICE_LOST (a hard device wedge that
// also corrupts NIC/QP state and cascades into later iterations). With the low
// cap the same delayed flag instead breaks out and degrades to "0 tokens" -- a
// soft, recoverable failure the test's correctness check catches, never a wedge.
// This is the flag-path DEVICE_LOST fix: bound the spin below the watchdog.
inline uint64_t ll_poll_cap() {
    const char* env = std::getenv("DEEP_EP_LL_POLL_CAP");
    if (env != nullptr && env[0] != '\0') {
        long long v = std::atoll(env);
        if (v > 0)
            return static_cast<uint64_t>(v);
    }
    return static_cast<uint64_t>(1) * 1000 * 1000;
}

// Cooperative copy of `n` bytes from src to dst using a strided set of
// cooperating lanes (lane in [0, lanes)). Uses 16-byte vector chunks when both
// pointers are 16-byte aligned (always true for the 128-byte-aligned symmetric
// heap regions and bf16/fp8 payloads here), falling back to bytes for any tail.
inline void coop_copy_bytes(uint8_t* dst, const uint8_t* src, size_t n, int lane, int lanes) {
    size_t done = 0;
    if ((reinterpret_cast<uintptr_t>(dst) & 0xF) == 0 && (reinterpret_cast<uintptr_t>(src) & 0xF) == 0) {
        const size_t n16 = n >> 4;
        auto* d16 = reinterpret_cast<sycl::vec<uint32_t, 4>*>(dst);
        auto* s16 = reinterpret_cast<const sycl::vec<uint32_t, 4>*>(src);
        for (size_t j = static_cast<size_t>(lane); j < n16; j += static_cast<size_t>(lanes)) {
            d16[j] = s16[j];
        }
        done = n16 << 4;
    }
    for (size_t b = done + static_cast<size_t>(lane); b < n; b += static_cast<size_t>(lanes)) {
        dst[b] = src[b];
    }
}

// Uncached variant of coop_copy_bytes: reads the SOURCE through uc_load (the GPU
// L2 is not coherent with external PCIe-P2P RDMA writes on this BMG+mlx5 stack,
// and the symmetric-heap addresses are reused each iteration, so a cached load
// can return stale lines). Used by the flag-progress path where the global
// barrier_all (which previously supplied the cross-PCIe acquire) is removed.
inline void coop_copy_bytes_uc(uint8_t* dst, const uint8_t* src, size_t n, int lane, int lanes) {
    size_t done = 0;
    if ((reinterpret_cast<uintptr_t>(dst) & 0x7) == 0 && (reinterpret_cast<uintptr_t>(src) & 0x7) == 0) {
        const size_t n8 = n >> 3;
        auto* d8 = reinterpret_cast<uint64_t*>(dst);
        auto* s8 = reinterpret_cast<const uint64_t*>(src);
        for (size_t j = static_cast<size_t>(lane); j < n8; j += static_cast<size_t>(lanes)) {
            d8[j] = uc_load(&s8[j]);
        }
        done = n8 << 3;
    }
    for (size_t b = done + static_cast<size_t>(lane); b < n; b += static_cast<size_t>(lanes)) {
        dst[b] = uc_load(&src[b]);
    }
}

// Write-through variant: reads SOURCE cached (local, coherent) and writes DST via
// uc_store (write-through to memory). Used by the flag-progress path for the LOCAL
// self-copy into the symmetric receive region (dispatch_data / combine_data), so a
// later uncached reader (uc_load, which bypasses L2) observes the bytes in memory
// rather than missing a still-cached store.
inline void coop_copy_bytes_store_uc(uint8_t* dst, const uint8_t* src, size_t n, int lane, int lanes) {
    size_t done = 0;
    if ((reinterpret_cast<uintptr_t>(dst) & 0x7) == 0 && (reinterpret_cast<uintptr_t>(src) & 0x7) == 0) {
        const size_t n8 = n >> 3;
        auto* d8 = reinterpret_cast<uint64_t*>(dst);
        auto* s8 = reinterpret_cast<const uint64_t*>(src);
        for (size_t j = static_cast<size_t>(lane); j < n8; j += static_cast<size_t>(lanes)) {
            uc_store(&d8[j], s8[j]);
        }
        done = n8 << 3;
    }
    for (size_t b = done + static_cast<size_t>(lane); b < n; b += static_cast<size_t>(lanes)) {
        uc_store(&dst[b], src[b]);
    }
}

#ifdef DEEP_EP_ENABLE_ISHMEM
// Issue an iSHMEM NBI put of `n` bytes from a single work-item, split into
// chunks no larger than `max_chunk` to stay on the IBGDA fast path (see
// ll_max_put_bytes). Chunk boundaries are 16-byte aligned for safety.
inline void chunked_put_nbi(uint8_t* dst, const uint8_t* src, size_t n, int dst_pe, size_t max_chunk) {
    if (max_chunk == 0 || n <= max_chunk) {
        ishmem_putmem_nbi(dst, src, n, dst_pe);
        return;
    }
    const size_t step = max_chunk & ~static_cast<size_t>(0xF);
    size_t off = 0;
    while (off < n) {
        const size_t this_bytes = sycl::min(step, n - off);
        ishmem_putmem_nbi(dst + off, src + off, this_bytes, dst_pe);
        off += this_bytes;
    }
}
#endif

}  // namespace

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer_ptr,
                              int* sync_buffer_ptr,
                              sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    queue.wait();
    if (sync_buffer_ptr == nullptr) {
        internode::barrier();
    }
#endif
    const int total = num_clean_int_0 + num_clean_int_1;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CleanLowLatencyBufferKernel>(sycl::range<1>(total), [=](sycl::id<1> id) {
            int i = static_cast<int>(id[0]);
            if (i < num_clean_int_0) {
                clean_0[i] = 0;
            } else if (clean_1 != nullptr) {
                clean_1[i - num_clean_int_0] = 0;
            }
        });
    });
#ifdef DEEP_EP_ENABLE_ISHMEM
    queue.wait();
    if (sync_buffer_ptr == nullptr) {
        internode::barrier();
    }
#else
    (void)rank;
    (void)num_ranks;
    (void)mask_buffer_ptr;
    (void)sync_buffer_ptr;
#endif
}

void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask, sycl::queue& queue) {
    queue.submit(
        [&](sycl::handler& cgh) { cgh.single_task<UpdateMaskBufferKernel>([=]() { mask_buffer_ptr[rank_to_mask] = mask ? 1 : 0; }); });
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* output_mask_tensor, sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<QueryMaskBufferKernel>(sycl::range<1>(num_ranks), [=](sycl::id<1> id) {
            int rank = static_cast<int>(id[0]);
            output_mask_tensor[rank] = mask_buffer_ptr[rank];
        });
    });
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, sycl::queue& queue) {
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<CleanMaskBufferKernel>(sycl::range<1>(num_ranks),
                                                [=](sycl::id<1> id) { mask_buffer_ptr[static_cast<int>(id[0])] = 0; });
    });
}

void dispatch_bf16(void* packed_recv_x,
                   void* packed_recv_x_scales,
                   int* packed_recv_src_info,
                   int64_t* packed_recv_layout_range,
                   int* packed_recv_count,
                   int* cumulative_local_expert_recv_stats,
                   int64_t* dispatch_wait_recv_cost_stats,
                   void* rdma_buffer,
                   int* mask_buffer_ptr,
                   const void* x,
                   const topk_idx_t* topk_idx,
                   int num_tokens,
                   int hidden,
                   int num_max_dispatch_tokens_per_rank,
                   int num_topk,
                   int num_experts,
                   int rank,
                   int num_ranks,
                   bool use_fp8,
                   bool round_scale,
                   bool use_ue8m0,
                   sycl::queue& queue) {
#ifndef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(false, "XPU low-latency dispatch requires iSHMEM support");
#else
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    auto layout = make_layout(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    const int num_local_experts = num_experts / num_ranks;
    const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    auto* base = static_cast<uint8_t*>(rdma_buffer);
    auto* dispatch_data = base + layout.dispatch_data_offset;
    auto* dispatch_src = reinterpret_cast<int*>(base + layout.dispatch_src_offset);
    auto* dispatch_count = reinterpret_cast<int*>(base + layout.dispatch_count_offset);
    auto* send_data = base + layout.send_data_offset;
    auto* send_src = reinterpret_cast<int*>(base + layout.send_src_offset);
    auto* send_count = reinterpret_cast<int*>(base + layout.send_count_offset);

    const size_t send_count_elems = static_cast<size_t>(num_ranks) * num_local_experts;
    const size_t slot_elems = send_count_elems * num_max_dispatch_tokens_per_rank;
    const size_t recv_src_elems = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
    const size_t max_put = ll_max_put_bytes();
    const bool flag_progress = ll_flag_progress();
    const uint64_t poll_cap = ll_poll_cap();
    const bool flag_diag_barrier = ll_flag_diag_barrier();
    const int flag_lsc_mode = ll_flag_lsc_mode();
    const int flag_sender_fence = ll_flag_sender_fence();
    const int flag_recv_acq = ll_flag_recv_acq();
    const bool coop_put = ll_coop_put();
    // When the receiver does a single acquire/invalidate up front, the bulk payload
    // is read with CACHED loads instead of per-byte uncached uc_load.
    const bool recv_uncached = flag_progress && (flag_recv_acq == 0);
    auto* combine_flag = base + layout.combine_flag_offset;

    // --- Stage 0: zero LOCAL staging + output tensors (multi-WG via memset).
    // Symmetric receive buffers (dispatch_*) are zeroed by clean_low_latency_buffer
    // with cross-PE barriers. send_src / packed_recv_src_info use -1 (0xFF bytes).
    queue.memset(send_count, 0, send_count_elems * sizeof(int));
    queue.memset(packed_recv_count, 0, static_cast<size_t>(num_local_experts) * sizeof(int));
    queue.memset(send_src, 0xFF, slot_elems * sizeof(int));
    queue.memset(packed_recv_src_info, 0xFF, recv_src_elems * sizeof(int));
    // Flag-progress cross-cleaning: this dispatch zeros the LOCAL combine_flag
    // buffer (the slot combine RECEIVES completion signals into), so the next
    // combine starts from a clean flag state without a global clean_low_latency
    // barrier on the perf path. Mirrors CUDA's next_clean (dispatch clears the
    // combine flag region). dispatch_count (this stage's receive slot) is zeroed
    // by the prior combine's Stage 0.
    if (flag_progress) {
        // Write-through zero of the LOCAL combine_flag slots (the region combine
        // RECEIVES completion signals into). A cached queue.memset would leave the
        // zeros in L2; the combine receiver reads these via uc_load (bypassing L2),
        // and a lazily-flushed cached zero could also clobber an RDMA-delivered flag
        // in memory. Zeroing through uc_store puts the zeros in memory deterministically.
        const int n_flag_ints = num_experts * 2;
        const int wg = 256;
        const int wgs = (n_flag_ints + wg - 1) / wg > 0 ? (n_flag_ints + wg - 1) / wg : 1;
        auto* combine_flag_ints = reinterpret_cast<int*>(combine_flag);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchCleanFlagKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(wgs) * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> item) {
                    const int i = static_cast<int>(item.get_global_linear_id());
                    if (i < n_flag_ints) {
                        uc_store(&combine_flag_ints[i], 0);
                    }
                });
        });
    }

    // --- Stage 1: route tokens into local send staging (multi-WG, one sub-group
    // per (token, k) pair; the sub-group cooperatively copies the token payload).
    {
        const size_t num_pairs = static_cast<size_t>(num_tokens) * num_topk;
        const int num_wgs = ll_num_wgs(num_pairs, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchRouteKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    auto sg = item.get_sub_group();
                    const int sg_local = static_cast<int>(sg.get_local_linear_id());
                    const int sg_size = static_cast<int>(sg.get_local_range()[0]);
                    const int sgs_per_wg = static_cast<int>(sg.get_group_range()[0]);
                    const int global_sg =
                        static_cast<int>(item.get_group_linear_id()) * sgs_per_wg + static_cast<int>(sg.get_group_linear_id());
                    const int num_global_sgs = static_cast<int>(item.get_group_range(0)) * sgs_per_wg;

                    for (int p = global_sg; p < static_cast<int>(num_pairs); p += num_global_sgs) {
                        const int token_idx = p / num_topk;
                        const int k = p - token_idx * num_topk;
                        const int expert = static_cast<int>(topk_idx[token_idx * num_topk + k]);
                        if (expert < 0 || expert >= num_experts) {
                            continue;
                        }
                        const int dst_rank = expert / num_local_experts;
                        if (ll_rank_masked(mask_buffer_ptr, dst_rank)) {
                            continue;
                        }
                        const int local_expert = expert - dst_rank * num_local_experts;
                        // Single lane reserves the slot; broadcast to the sub-group.
                        int slot = -1;
                        if (sg_local == 0) {
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                count_ref(send_count[dst_rank * num_local_experts + local_expert]);
                            slot = count_ref.fetch_add(1);
                        }
                        slot = sycl::group_broadcast(sg, slot, 0);
                        if (slot >= num_max_dispatch_tokens_per_rank) {
                            continue;
                        }
                        const size_t packed_slot =
                            (static_cast<size_t>(dst_rank) * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank + slot;
                        coop_copy_bytes(send_data + packed_slot * hidden_bytes,
                                        static_cast<const uint8_t*>(x) + static_cast<size_t>(token_idx) * hidden_bytes,
                                        hidden_bytes,
                                        sg_local,
                                        sg_size);
                        if (sg_local == 0) {
                            send_src[packed_slot] = token_idx;
                        }
                    }
                });
        });
    }

    // --- Stage 2: local self-copy + remote NBI puts (multi-WG).
    // Local self-copy is heavily data-parallel across the whole grid; the remote
    // NBI put issuing is intentionally bounded to one work-item per channel
    // (num_experts channels) to avoid oversubscribing the single QP per PE.
    {
        const size_t self_bytes = static_cast<size_t>(num_local_experts) * num_max_dispatch_tokens_per_rank * hidden_bytes;
        const int num_wgs = ll_num_wgs(self_bytes >> 4, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchPutKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    const int gid = static_cast<int>(item.get_global_id(0));
                    const int gsize = static_cast<int>(item.get_global_range(0));

                    // Local (self) copy of dispatch_count / dispatch_src / dispatch_data.
                    for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                        const size_t src_slot =
                            (static_cast<size_t>(rank) * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank;
                        const size_t dst_slot = (static_cast<size_t>(local_expert) * num_ranks + rank) * num_max_dispatch_tokens_per_rank;
                        const int count = sycl::min(send_count[rank * num_local_experts + local_expert], num_max_dispatch_tokens_per_rank);
                        if (gid == 0) {
                            if (flag_progress) {
                                // Self path: encode as -count-1 so 0 tokens (-1) is
                                // distinguishable from not-arrived (0). Written via
                                // uc_store so the Wait/Pack uc_load reader observes it.
                                uc_store(&dispatch_count[local_expert * num_ranks + rank], -count - 1);
                            } else {
                                dispatch_count[local_expert * num_ranks + rank] = count;
                            }
                        }
                        for (int slot = gid; slot < num_max_dispatch_tokens_per_rank; slot += gsize) {
                            if (flag_progress) {
                                uc_store(&dispatch_src[dst_slot + slot], send_src[src_slot + slot]);
                            } else {
                                dispatch_src[dst_slot + slot] = send_src[src_slot + slot];
                            }
                        }
                        if (flag_progress) {
                            coop_copy_bytes_store_uc(dispatch_data + dst_slot * hidden_bytes,
                                                     send_data + src_slot * hidden_bytes,
                                                     static_cast<size_t>(count) * hidden_bytes,
                                                     gid,
                                                     gsize);
                        } else {
                            coop_copy_bytes(dispatch_data + dst_slot * hidden_bytes,
                                            send_data + src_slot * hidden_bytes,
                                            static_cast<size_t>(count) * hidden_bytes,
                                            gid,
                                            gsize);
                        }
                    }

                    // Remote NBI puts.
                    if (coop_put && flag_progress) {
                        // Design-(b) cooperative path: the WHOLE sub-group enters each
                        // put convergently; each lane owns channel = gid (if in range).
                        // Payload+flag share one affinity QP (aff = gid) so in-order
                        // delivery still guarantees "flag lands after payload", while a
                        // single leader per (sub-group, QP) publishes SND_DBR.
                        auto sg = item.get_sub_group();
                        const int num_channels = (num_ranks - 1) * num_local_experts;
                        bool ch_active = (gid < num_channels);
                        int my_dst_rank = 0, my_le = 0, my_sc_idx = 0, my_count = 0;
                        size_t my_src_slot = 0, my_dst_slot = 0;
                        if (ch_active) {
                            const int dr = gid / num_local_experts;   // index among non-self ranks
                            my_le = gid % num_local_experts;
                            my_dst_rank = (dr < rank) ? dr : dr + 1;   // skip self
                            my_sc_idx = my_dst_rank * num_local_experts + my_le;
                            send_count[my_sc_idx] = sycl::min(send_count[my_sc_idx], num_max_dispatch_tokens_per_rank);
                            my_count = send_count[my_sc_idx];
                            my_src_slot = static_cast<size_t>(my_sc_idx) * num_max_dispatch_tokens_per_rank;
                            my_dst_slot = (static_cast<size_t>(my_le) * num_ranks + rank) * num_max_dispatch_tokens_per_rank;
                        }
                        const unsigned int aff = static_cast<unsigned int>(gid);
                        const bool pay = ch_active && (my_count > 0);
                        // 1) payload: source token indices
                        ishmemx_putmem_nbi_sub_group(dispatch_src + my_dst_slot, send_src + my_src_slot,
                                                     static_cast<size_t>(my_count) * sizeof(int),
                                                     my_dst_rank, pay, aff, sg);
                        // 2) payload: token data (cooperative emit chunks internally)
                        ishmemx_putmem_nbi_sub_group(dispatch_data + my_dst_slot * hidden_bytes,
                                                     send_data + my_src_slot * hidden_bytes,
                                                     static_cast<size_t>(my_count) * hidden_bytes,
                                                     my_dst_rank, pay, aff, sg);
                        // 3) local flag store + sender flush, then the completion flag put
                        if (ch_active) uc_store(&send_count[my_sc_idx], -my_count - 1);
                        ll_sender_flush(flag_sender_fence);
                        ishmemx_putmem_nbi_sub_group(dispatch_count + my_le * num_ranks + rank,
                                                     send_count + my_sc_idx, sizeof(int),
                                                     my_dst_rank, ch_active, aff, sg);
                    } else {
                    // one work-item per channel.
                    int ch = 0;
                    for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                        if (dst_rank == rank) {
                            continue;
                        }
                        for (int le = 0; le < num_local_experts; ++le) {
                            if (ch == gid) {
                                const int sc_idx = dst_rank * num_local_experts + le;
                                send_count[sc_idx] = sycl::min(send_count[sc_idx], num_max_dispatch_tokens_per_rank);
                                const int count = send_count[sc_idx];
                                const size_t src_slot = static_cast<size_t>(sc_idx) * num_max_dispatch_tokens_per_rank;
                                const size_t dst_slot = (static_cast<size_t>(le) * num_ranks + rank) * num_max_dispatch_tokens_per_rank;
                                if (flag_progress) {
                                    // Send payload first, then the per-expert completion flag via a
                                    // 4-byte RDMA WRITE (SET semantics) of the encoded value -count-1
                                    // into the receiver's count slot. SET (not atomic-add) is required
                                    // because with GPU_IPC=0 every peer goes over the NIC: the test may
                                    // issue two identical dispatches back-to-back without an intervening
                                    // combine (which would reset the slot), so an accumulating add would
                                    // double the count. A WRITE overwrites, matching CUDA's intra-node
                                    // st_release SET path. With QPS_PER_PE=1 the payload puts and this
                                    // WRITE share the same RC QP, so in-order delivery guarantees the
                                    // flag lands AFTER the payload. 0 tokens -> -1 (still != 0).
                                    if (count > 0) {
                                        ishmem_putmem_nbi(dispatch_src + dst_slot,
                                                          send_src + src_slot,
                                                          static_cast<size_t>(count) * sizeof(int),
                                                          dst_rank);
                                        chunked_put_nbi(dispatch_data + dst_slot * hidden_bytes,
                                                        send_data + src_slot * hidden_bytes,
                                                        static_cast<size_t>(count) * hidden_bytes,
                                                        dst_rank,
                                                        max_put);
                                    }
                                    uc_store(&send_count[sc_idx], -count - 1);  // write-through so NIC reads fresh
                                    // Optional sender-side release/flush so the NIC DMA-reads the
                                    // freshly-stored flag (not a stale cached value). See
                                    // ll_flag_sender_fence / DEEP_EP_LL_FLAG_SENDER_FENCE.
                                    ll_sender_flush(flag_sender_fence);
                                    ishmem_putmem_nbi(dispatch_count + le * num_ranks + rank, send_count + sc_idx, sizeof(int), dst_rank);
                                } else {
                                    ishmem_putmem_nbi(dispatch_count + le * num_ranks + rank, send_count + sc_idx, sizeof(int), dst_rank);
                                    if (count > 0) {
                                        ishmem_putmem_nbi(dispatch_src + dst_slot,
                                                          send_src + src_slot,
                                                          static_cast<size_t>(count) * sizeof(int),
                                                          dst_rank);
                                        chunked_put_nbi(dispatch_data + dst_slot * hidden_bytes,
                                                        send_data + src_slot * hidden_bytes,
                                                        static_cast<size_t>(count) * hidden_bytes,
                                                        dst_rank,
                                                        max_put);
                                    }
                                }
                            }
                            ch++;
                        }
                    }
                    }  // end else (non-cooperative per-channel path)
                });
        });
    }

    // --- Stage 3: synchronize PEs before the receive/pack stage. Runs after
    // Stage 2 completes on the in-order queue, so all puts have been issued.
    if (flag_progress) {
        // Flag-based progress: NO global all-to-all barrier. Each PE (a) drains its
        // own outbound NBI puts via ishmemx_quiet_work_group (so the shared
        // send_data staging is safe to reuse next iter), then (b) spin-polls its OWN
        // per-(local_expert, src_rank) dispatch_count flag slots until the remote
        // RDMA atomic-add lands (slot != 0), with a bounded spin cap. Mirrors the
        // CUDA receiver poll; latency no longer grows with the slowest peer.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchWaitKernel>(sycl::nd_range<1>(sycl::range<1>(kLLWGSize), sycl::range<1>(kLLWGSize)),
                                                           [=](sycl::nd_item<1> item) {
                                                               auto group = item.get_group();
                                                               if (flag_diag_barrier) {
                                                                   // DIAGNOSTIC: proven barrier sync only (no quiet, no spin).
                                                                   // Tests SET-flag landing + uc_load decode in isolation.
                                                                   ishmemx_barrier_all_work_group(group);
                                                                   return;
                                                               }
                                                               ishmemx_quiet_work_group(group);
                                                               sycl::group_barrier(group);
                                                               const int local_id = static_cast<int>(item.get_local_id(0));
                                                               const int local_size = static_cast<int>(item.get_local_range(0));
                                                               const int num_slots = num_local_experts * num_ranks;
                                                               for (int s = local_id; s < num_slots; s += local_size) {
                                                                   const int le = s / num_ranks;
                                                                   const int src_rank = s - le * num_ranks;
                                                                   if (src_rank == rank) {
                                                                       continue;  // self slot written locally (uc_store)
                                                                   }
                                                                   if (ll_rank_masked(mask_buffer_ptr, src_rank)) {
                                                                       continue;  // masked peer never sends; leave 0 -> 0 tokens
                                                                   }
                                                                   uint64_t spins = 0;
                                                                   while (ll_read_flag(&dispatch_count[le * num_ranks + src_rank], flag_lsc_mode) == 0) {
                                                                       if (++spins >= poll_cap) {
                                                                           break;  // timeout -> leave 0 -> treated as 0 tokens
                                                                       }
                                                                   }
                                                               }
                                                           });
        });
    } else {
        // Barrier path: single-WG cross-PE barrier drains the NBI puts (quiet) and
        // synchronizes all PEs before the receive/pack stage.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchBarrierKernel>(
                sycl::nd_range<1>(sycl::range<1>(kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) { ishmemx_barrier_all_work_group(item.get_group()); });
        });
    }

    // --- Stage 4: pack received data. Each local expert is handled by a cohort of
    // `pack_wgs_per_expert` work-groups; the per-rank prefix (begin offsets) is
    // recomputed (cheaply) by every work-group, only the cohort leader writes the
    // per-channel layout/stats, and the payload copy / src_info / FP8 conversion are
    // partitioned across the whole cohort for parallelism.
    // When use_fp8 is set, the BF16 payload from the RDMA staging is converted to
    // FP8 (with per-128-channel scales) directly here — the FP8 cast is fused into
    // dispatch rather than performed as a separate pass over the packed output.
    {
        const int num_scales = (hidden % 128 == 0) ? hidden / 128 : 0;
        const int scale_packs = use_ue8m0 ? (num_scales + 3) / 4 : num_scales;
        auto* dst_fp8 = static_cast<uint8_t*>(packed_recv_x);
        auto* dst_scale_float = static_cast<float*>(packed_recv_x_scales);
        auto* dst_scale_int = static_cast<int32_t*>(packed_recv_x_scales);
        const int local_experts = num_local_experts > 0 ? num_local_experts : 1;
        const int pack_wgs_per_expert = sycl::max(1, sycl::min(kLLMaxWGs / local_experts, 32));
        const int num_wgs = local_experts * pack_wgs_per_expert;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchPackKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    const int local_size = static_cast<int>(item.get_local_range(0));
                    const int wg = static_cast<int>(item.get_group_linear_id());
                    const int local_expert = wg / pack_wgs_per_expert;
                    const int sub = wg % pack_wgs_per_expert;
                    if (local_expert >= num_local_experts) {
                        return;
                    }
                    const bool leader = (sub == 0 && local_id == 0);
                    // Cooperating-lane identity across the whole cohort for this expert.
                    const int cohort_id = sub * local_size + local_id;
                    const int cohort_size = pack_wgs_per_expert * local_size;

                    // Flag path: the per-expert flags already confirmed (Stage 3) that
                    // the RDMA payload has landed in memory. Issue a SINGLE system-scope
                    // acquire/invalidate here so the subsequent CACHED payload reads see
                    // the NIC-delivered bytes (when flag_recv_acq != 0). No-op otherwise.
                    if (flag_progress && flag_recv_acq != 0) {
                        ll_recv_acquire(flag_recv_acq);
                    }

                    int begin = 0;
                    int total = 0;
                    for (int src_rank = 0; src_rank < num_ranks; ++src_rank) {
                        int clamped_count;
                        if (flag_progress) {
                            // Flag encoding: 0 = not-arrived/timeout -> 0 tokens;
                            // otherwise raw = -count-1 -> count = -raw-1. uc_load
                            // forces a fetch of the RDMA-delivered (NIC-written) slot.
                            const int raw = ll_read_flag(&dispatch_count[local_expert * num_ranks + src_rank], flag_lsc_mode);
                            clamped_count = (raw == 0) ? 0 : sycl::min(-raw - 1, num_max_dispatch_tokens_per_rank);
                        } else {
                            clamped_count =
                                sycl::min(dispatch_count[local_expert * num_ranks + src_rank], num_max_dispatch_tokens_per_rank);
                        }
                        if (leader) {
                            packed_recv_layout_range[local_expert * num_ranks + src_rank] =
                                static_cast<int64_t>(pack_range(clamped_count, begin));
                            if (cumulative_local_expert_recv_stats != nullptr) {
                                cumulative_local_expert_recv_stats[local_expert] += clamped_count;
                            }
                            if (dispatch_wait_recv_cost_stats != nullptr && local_expert == 0) {
                                dispatch_wait_recv_cost_stats[src_rank] += 0;
                            }
                        }
                        if (clamped_count > 0) {
                            const size_t src_base =
                                (static_cast<size_t>(local_expert) * num_ranks + src_rank) * num_max_dispatch_tokens_per_rank;
                            const size_t dst_base =
                                static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin;
                            if (use_fp8) {
                                // Per-(row, 128-channel block) BF16->FP8 conversion. Each lane
                                // owns one (row, scale block): compute amax, scale, write 128 FP8
                                // values and the per-block scale (float, or packed UE8M0 byte).
                                const size_t work = static_cast<size_t>(clamped_count) * num_scales;
                                for (size_t w = cohort_id; w < work; w += cohort_size) {
                                    const int local_row = static_cast<int>(w / num_scales);
                                    const int scale_idx = static_cast<int>(w % num_scales);
                                    const auto* src_bf16 = reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(
                                        dispatch_data + (src_base + local_row) * hidden_bytes);
                                    const size_t dst_row = dst_base + local_row;
                                    const int base_h = scale_idx * 128;
                                    // Read the 128-channel block ONCE (uncached on the flag path so the
                                    // RDMA-delivered payload is observed, not a stale L2 line) into a
                                    // local buffer, then run the amax and scale passes from the local
                                    // copy. Avoids a second uncached pass over the same data.
                                    float blk[128];
                                    float amax = 1.0e-4f;
                                    for (int i = 0; i < 128; ++i) {
                                        const float fv = recv_uncached ? static_cast<float>(uc_load(&src_bf16[base_h + i]))
                                                                       : static_cast<float>(src_bf16[base_h + i]);
                                        blk[i] = fv;
                                        amax = sycl::fmax(amax, sycl::fabs(fv));
                                    }
                                    float scale;
                                    float scale_inv;
                                    if (round_scale) {
                                        const float exp_scale_inv = sycl::ceil(sycl::log2(amax / 448.0f));
                                        scale = sycl::exp2(-exp_scale_inv);
                                        scale_inv = sycl::exp2(exp_scale_inv);
                                    } else {
                                        scale_inv = amax / 448.0f;
                                        scale = 448.0f / amax;
                                    }
                                    for (int i = 0; i < 128; ++i) {
                                        const float value = blk[i] * scale;
                                        dst_fp8[dst_row * hidden + base_h + i] = c10::Float8_e4m3fn(value).x;
                                    }
                                    if (use_ue8m0) {
                                        const int pack_idx = scale_idx / 4;
                                        const int pack_shift = (scale_idx % 4) * 8;
                                        const int32_t scale_byte = static_cast<int32_t>(ue8m0_from_float(scale_inv)) << pack_shift;
                                        sycl::atomic_ref<int32_t,
                                                         sycl::memory_order::relaxed,
                                                         sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>
                                            scale_pack(dst_scale_int[dst_row * scale_packs + pack_idx]);
                                        scale_pack.fetch_or(scale_byte);
                                    } else {
                                        dst_scale_float[dst_row * num_scales + scale_idx] = scale_inv;
                                    }
                                }
                            } else {
                                if (recv_uncached) {
                                    coop_copy_bytes_uc(static_cast<uint8_t*>(packed_recv_x) + dst_base * hidden_bytes,
                                                       dispatch_data + src_base * hidden_bytes,
                                                       static_cast<size_t>(clamped_count) * hidden_bytes,
                                                       cohort_id,
                                                       cohort_size);
                                } else {
                                    coop_copy_bytes(static_cast<uint8_t*>(packed_recv_x) + dst_base * hidden_bytes,
                                                    dispatch_data + src_base * hidden_bytes,
                                                    static_cast<size_t>(clamped_count) * hidden_bytes,
                                                    cohort_id,
                                                    cohort_size);
                                }
                            }
                            for (int slot = cohort_id; slot < clamped_count; slot += cohort_size) {
                                packed_recv_src_info[dst_base + slot] =
                                    recv_uncached ? uc_load(&dispatch_src[src_base + slot]) : dispatch_src[src_base + slot];
                            }
                        }
                        begin += clamped_count;
                        total += clamped_count;
                    }
                    if (leader) {
                        packed_recv_count[local_expert] = total;
                    }
                });
        });
    }
#endif
}

void combine_bf16(void* combined_x,
                  void* rdma_buffer,
                  int* mask_buffer_ptr,
                  const void* x,
                  const topk_idx_t* topk_idx,
                  const float* topk_weights,
                  const int* src_info,
                  const int64_t* layout_range,
                  int64_t* combine_wait_recv_cost_stats,
                  int num_combined_tokens,
                  int hidden,
                  int num_max_dispatch_tokens_per_rank,
                  int num_topk,
                  int num_experts,
                  int rank,
                  int num_ranks,
                  sycl::queue& queue,
                  bool zero_copy) {
#ifndef DEEP_EP_ENABLE_ISHMEM
    TORCH_CHECK(false, "XPU low-latency combine requires iSHMEM support");
#else
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    (void)zero_copy;
    auto layout = make_layout(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    const int num_local_experts = num_experts / num_ranks;
    const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    auto* base = static_cast<uint8_t*>(rdma_buffer);
    auto* send_data = base + layout.send_data_offset;
    auto* combine_data = base + layout.combine_data_offset;
    auto* combine_flag_i = reinterpret_cast<int*>(base + layout.combine_flag_offset);
    auto* dispatch_count = reinterpret_cast<int*>(base + layout.dispatch_count_offset);
    auto* send_count = reinterpret_cast<int*>(base + layout.send_count_offset);

    const size_t send_elems = static_cast<size_t>(num_ranks) * num_local_experts * num_max_dispatch_tokens_per_rank;
    const size_t max_put = ll_max_put_bytes();
    const bool flag_progress = ll_flag_progress_combine();
    const uint64_t poll_cap = ll_poll_cap();
    const bool flag_diag_barrier = ll_flag_diag_barrier();
    const int flag_lsc_mode = ll_flag_lsc_mode();
    const int flag_sender_fence = ll_flag_sender_fence();
    const int flag_recv_acq = ll_flag_recv_acq();
    const bool coop_put = ll_coop_put();
    const bool recv_uncached = flag_progress && (flag_recv_acq == 0);

    // --- Stage 0: zero local send staging (bf16 zero == 0x0000). combine_data and
    // combine_flag are already zeroed by clean_low_latency_buffer (cross-PE barrier).
    queue.memset(send_data, 0, send_elems * hidden_bytes);
    // Flag-progress cross-cleaning: this combine zeros the LOCAL dispatch_count
    // buffer (the slot the NEXT dispatch receives its per-expert count flags into),
    // so the next dispatch starts clean without a global clean_low_latency barrier.
    // Mirrors CUDA's next_clean (combine clears the dispatch count region).
    // combine_flag (this stage's receive slot) was zeroed by this iter's dispatch.
    if (flag_progress) {
        // Write-through zero of the LOCAL dispatch_count slots (the region the NEXT
        // dispatch RECEIVES its per-expert count flags into). See the matching
        // rationale in dispatch Stage 0: cached memset zeros may not reach memory
        // before the next dispatch's RDMA flag write / uc_load, so zero via uc_store.
        const int n_count_ints = num_local_experts * num_ranks;
        const int wg = 256;
        const int wgs = (n_count_ints + wg - 1) / wg > 0 ? (n_count_ints + wg - 1) / wg : 1;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombineCleanCountKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(wgs) * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> item) {
                    const int i = static_cast<int>(item.get_global_linear_id());
                    if (i < n_count_ints) {
                        uc_store(&dispatch_count[i], 0);
                    }
                });
        });
    }

    // --- Stage 1: scatter the received expert payload (x) into the per-source send
    // staging slots indexed by (src_rank, local_expert, original_token). One
    // sub-group per (local_expert, src_rank, slot) triple cooperatively copies a row.
    {
        const size_t num_rows = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
        const int num_wgs = ll_num_wgs(num_rows, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombineScatterKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    auto sg = item.get_sub_group();
                    const int sg_local = static_cast<int>(sg.get_local_linear_id());
                    const int sg_size = static_cast<int>(sg.get_local_range()[0]);
                    const int sgs_per_wg = static_cast<int>(sg.get_group_range()[0]);
                    const int global_sg =
                        static_cast<int>(item.get_group_linear_id()) * sgs_per_wg + static_cast<int>(sg.get_group_linear_id());
                    const int num_global_sgs = static_cast<int>(item.get_group_range(0)) * sgs_per_wg;
                    const size_t rows_per_expert = static_cast<size_t>(num_ranks) * num_max_dispatch_tokens_per_rank;

                    for (size_t r = global_sg; r < num_rows; r += num_global_sgs) {
                        const int local_expert = static_cast<int>(r / rows_per_expert);
                        const int rem = static_cast<int>(r % rows_per_expert);
                        const int src_rank = rem / num_max_dispatch_tokens_per_rank;
                        const int slot = rem % num_max_dispatch_tokens_per_rank;
                        if (ll_rank_masked(mask_buffer_ptr, src_rank)) {
                            continue;
                        }
                        int count = 0, begin = 0;
                        unpack_range(layout_range[local_expert * num_ranks + src_rank], count, begin);
                        const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                        if (slot >= clamped_count) {
                            continue;
                        }
                        const int original_token =
                            src_info[static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin + slot];
                        if (original_token < 0 || original_token >= num_max_dispatch_tokens_per_rank) {
                            continue;
                        }
                        auto* staged_dst = send_data +
                            (static_cast<size_t>(src_rank * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank +
                             original_token) *
                                hidden_bytes;
                        auto* src = static_cast<const uint8_t*>(x) +
                            (static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin + slot) *
                                hidden_bytes;
                        coop_copy_bytes(staged_dst, src, hidden_bytes, sg_local, sg_size);
                    }
                });
        });
    }

    // --- Stage 2: combine put — local self-copy (grid-parallel) + bounded remote
    // NBI puts (one work-item per channel) of the per-destination min/max token span.
    {
        const size_t self_bytes = static_cast<size_t>(num_local_experts) * num_max_dispatch_tokens_per_rank * hidden_bytes;
        const int num_wgs = ll_num_wgs(self_bytes >> 4, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombinePutKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    const int gid = static_cast<int>(item.get_global_id(0));
                    const int gsize = static_cast<int>(item.get_global_range(0));

                    // Step 1: local self-copy (dst_rank == rank), grid-parallel.
                    for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                        const int global_expert = rank * num_local_experts + local_expert;
                        int count = 0, begin = 0;
                        unpack_range(layout_range[local_expert * num_ranks + rank], count, begin);
                        const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                        int min_token = num_max_dispatch_tokens_per_rank;
                        int max_token = -1;
                        for (int slot = 0; slot < clamped_count; ++slot) {
                            const int original_token =
                                src_info[static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin + slot];
                            if (original_token >= 0 && original_token < num_max_dispatch_tokens_per_rank) {
                                min_token = sycl::min(min_token, original_token);
                                max_token = sycl::max(max_token, original_token);
                            }
                        }
                        if (max_token >= min_token) {
                            auto* src_ptr = send_data +
                                (static_cast<size_t>(rank * num_local_experts + local_expert) * num_max_dispatch_tokens_per_rank +
                                 min_token) *
                                    hidden_bytes;
                            auto* dst_ptr = combine_data +
                                (static_cast<size_t>(global_expert) * num_max_dispatch_tokens_per_rank + min_token) * hidden_bytes;
                            const size_t bytes = static_cast<size_t>(max_token - min_token + 1) * hidden_bytes;
                            if (flag_progress) {
                                coop_copy_bytes_store_uc(dst_ptr, src_ptr, bytes, gid, gsize);
                            } else {
                                coop_copy_bytes(dst_ptr, src_ptr, bytes, gid, gsize);
                            }
                        }
                        if (flag_progress && gid == 0) {
                            // Self-owned expert: set the local completion flag (UNCONDITIONALLY,
                            // even 0 tokens). Written via uc_store so the Reduce/Wait uc_load
                            // reader observes it. combine_flag is uint64_t[num_experts]; the int*
                            // alias indexes the little-endian low word at [global_expert*2].
                            uc_store(&combine_flag_i[global_expert * 2], 1);
                        }
                    }

                    // Step 2: remote NBI puts.
                    if (coop_put && flag_progress) {
                        // Design-(b) cooperative path (see dispatch stage 2 for rationale).
                        auto sg = item.get_sub_group();
                        const int num_channels = (num_ranks - 1) * num_local_experts;
                        bool ch_active = (gid < num_channels);
                        int my_dst_rank = 0, my_local_expert = 0, my_global_expert = 0, my_sc_idx = 0;
                        bool has_payload = false;
                        size_t my_bytes = 0;
                        uint8_t* my_src_ptr = send_data;
                        uint8_t* my_dst_ptr = combine_data;
                        if (ch_active) {
                            const int dr = gid / num_local_experts;
                            my_local_expert = gid % num_local_experts;
                            my_dst_rank = (dr < rank) ? dr : dr + 1;
                            my_global_expert = rank * num_local_experts + my_local_expert;
                            my_sc_idx = my_dst_rank * num_local_experts + my_local_expert;
                            int count = 0, begin = 0;
                            unpack_range(layout_range[my_local_expert * num_ranks + my_dst_rank], count, begin);
                            const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                            int min_token = num_max_dispatch_tokens_per_rank;
                            int max_token = -1;
                            for (int slot = 0; slot < clamped_count; ++slot) {
                                const int original_token =
                                    src_info[static_cast<size_t>(my_local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin +
                                             slot];
                                if (original_token >= 0 && original_token < num_max_dispatch_tokens_per_rank) {
                                    min_token = sycl::min(min_token, original_token);
                                    max_token = sycl::max(max_token, original_token);
                                }
                            }
                            if (max_token >= min_token) {
                                has_payload = true;
                                my_src_ptr = send_data +
                                    (static_cast<size_t>(my_dst_rank * num_local_experts + my_local_expert) *
                                         num_max_dispatch_tokens_per_rank +
                                     min_token) *
                                        hidden_bytes;
                                my_dst_ptr = combine_data +
                                    (static_cast<size_t>(my_global_expert) * num_max_dispatch_tokens_per_rank + min_token) * hidden_bytes;
                                my_bytes = static_cast<size_t>(max_token - min_token + 1) * hidden_bytes;
                            }
                        }
                        const unsigned int aff = static_cast<unsigned int>(gid);
                        const bool pay = ch_active && has_payload;
                        // payload put (cooperative emit chunks internally)
                        ishmemx_putmem_nbi_sub_group(my_dst_ptr, my_src_ptr, my_bytes, my_dst_rank, pay, aff, sg);
                        // local flag store + flush, then completion flag put
                        if (ch_active) uc_store(&send_count[my_sc_idx], 1);
                        ll_sender_flush(flag_sender_fence);
                        ishmemx_putmem_nbi_sub_group(combine_flag_i + my_global_expert * 2,
                                                     send_count + my_sc_idx, sizeof(int),
                                                     my_dst_rank, ch_active, aff, sg);
                    } else {
                    // one work-item per channel.
                    int ch = 0;
                    for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
                        if (dst_rank == rank) {
                            continue;
                        }
                        for (int local_expert = 0; local_expert < num_local_experts; ++local_expert) {
                            if (ch == gid) {
                                const int global_expert = rank * num_local_experts + local_expert;
                                int count = 0, begin = 0;
                                unpack_range(layout_range[local_expert * num_ranks + dst_rank], count, begin);
                                const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                                int min_token = num_max_dispatch_tokens_per_rank;
                                int max_token = -1;
                                for (int slot = 0; slot < clamped_count; ++slot) {
                                    const int original_token =
                                        src_info[static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin +
                                                 slot];
                                    if (original_token >= 0 && original_token < num_max_dispatch_tokens_per_rank) {
                                        min_token = sycl::min(min_token, original_token);
                                        max_token = sycl::max(max_token, original_token);
                                    }
                                }
                                if (max_token >= min_token) {
                                    auto* src_ptr = send_data +
                                        (static_cast<size_t>(dst_rank * num_local_experts + local_expert) *
                                             num_max_dispatch_tokens_per_rank +
                                         min_token) *
                                            hidden_bytes;
                                    auto* dst_ptr = combine_data +
                                        (static_cast<size_t>(global_expert) * num_max_dispatch_tokens_per_rank + min_token) * hidden_bytes;
                                    const size_t bytes = static_cast<size_t>(max_token - min_token + 1) * hidden_bytes;
                                    chunked_put_nbi(dst_ptr, src_ptr, bytes, dst_rank, max_put);
                                }
                                if (flag_progress) {
                                    // Per-expert completion flag SET (not atomic-add) via a 4-byte
                                    // RDMA WRITE of 1 into the receiver's combine_flag slot for this
                                    // global_expert. Each global_expert is owned by exactly ONE rank,
                                    // so SET (overwrite) is sufficient and avoids the atomic-add path
                                    // (which faults on this BMG+mlx5 IBGDA stack). The value 1 is
                                    // staged via uc_store into send_count[sc_idx] (one scratch int per
                                    // channel, unused by combine) so the NIC reads a fresh source.
                                    // Same-QP RC ordering (QPS_PER_PE=1) lands it after the payload put.
                                    const int sc_idx = dst_rank * num_local_experts + local_expert;
                                    uc_store(&send_count[sc_idx], 1);
                                    // Flush the staged flag past L3 so the NIC DMA-reads the fresh
                                    // value (same hazard as dispatch; see ll_flag_sender_fence).
                                    ll_sender_flush(flag_sender_fence);
                                    ishmem_putmem_nbi(combine_flag_i + global_expert * 2, send_count + sc_idx, sizeof(int), dst_rank);
                                }
                            }
                            ch++;
                        }
                    }
                    }  // end else (non-cooperative per-channel path)
                });
        });
    }

    // --- Stage 3: synchronize PEs before the reduction stage.
    if (flag_progress) {
        // Flag-based progress: drain outbound puts (quiet), then spin-poll the
        // per-global-expert combine_flag slots until each owner's RDMA atomic-add
        // lands. Skip self-owned experts (set locally) and masked owners. No global
        // all-to-all barrier.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombineWaitKernel>(sycl::nd_range<1>(sycl::range<1>(kLLWGSize), sycl::range<1>(kLLWGSize)),
                                                          [=](sycl::nd_item<1> item) {
                                                              auto group = item.get_group();
                                                              if (flag_diag_barrier) {
                                                                  ishmemx_barrier_all_work_group(group);
                                                                  return;
                                                              }
                                                              ishmemx_quiet_work_group(group);
                                                              sycl::group_barrier(group);
                                                              const int local_id = static_cast<int>(item.get_local_id(0));
                                                              const int local_size = static_cast<int>(item.get_local_range(0));
                                                              for (int ge = local_id; ge < num_experts; ge += local_size) {
                                                                  const int owner = ge / num_local_experts;
                                                                  if (owner == rank) {
                                                                      continue;  // self-owned: flag set locally
                                                                  }
                                                                  if (ll_rank_masked(mask_buffer_ptr, owner)) {
                                                                      continue;  // masked owner never sends
                                                                  }
                                                                  uint64_t spins = 0;
                                                                  while (ll_read_flag(&combine_flag_i[ge * 2], flag_lsc_mode) == 0) {
                                                                      if (++spins >= poll_cap) {
                                                                          break;  // timeout -> proceed; Reduce reads zeros
                                                                      }
                                                                  }
                                                              }
                                                          });
        });
    } else {
        // Barrier path: single-WG cross-PE barrier drains NBI puts (quiet) + syncs PEs.
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombineBarrierKernel>(
                sycl::nd_range<1>(sycl::range<1>(kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) { ishmemx_barrier_all_work_group(item.get_group()); });
        });
    }

    // --- Stage 4: weighted reduction over top-k into combined_x (grid-parallel).
    {
        const size_t reduce_work = static_cast<size_t>(num_combined_tokens) * hidden;
        const int num_wgs = ll_num_wgs(reduce_work, kLLWGSize, kLLMaxWGs);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombineReduceKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    const int gid = static_cast<int>(item.get_global_id(0));
                    const int gsize = static_cast<int>(item.get_global_range(0));
                    auto* out = static_cast<sycl::ext::oneapi::bfloat16*>(combined_x);
                    // Flag path: one acquire/invalidate up front (Stage 3 confirmed the
                    // combine payload landed) so the reduction reads CACHED instead of
                    // per-byte uncached, when flag_recv_acq != 0.
                    if (flag_progress && flag_recv_acq != 0) {
                        ll_recv_acquire(flag_recv_acq);
                    }
                    for (size_t idx = gid; idx < reduce_work; idx += gsize) {
                        const int token_idx = static_cast<int>(idx / hidden);
                        const int h = static_cast<int>(idx % hidden);
                        float acc = 0.0f;
                        for (int k = 0; k < num_topk; ++k) {
                            const int expert = static_cast<int>(topk_idx[token_idx * num_topk + k]);
                            if (expert < 0 || expert >= num_experts) {
                                continue;
                            }
                            const int src_rank = expert / num_local_experts;
                            if (ll_rank_masked(mask_buffer_ptr, src_rank)) {
                                continue;
                            }
                            const auto* value = reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(
                                combine_data + (static_cast<size_t>(expert) * num_max_dispatch_tokens_per_rank + token_idx) * hidden_bytes);
                            const float fv = recv_uncached ? static_cast<float>(uc_load(&value[h])) : static_cast<float>(value[h]);
                            acc += fv * topk_weights[token_idx * num_topk + k];
                        }
                        out[static_cast<size_t>(token_idx) * hidden + h] = bf16_from_float(acc);
                    }
                    if (combine_wait_recv_cost_stats != nullptr && gid < num_ranks) {
                        combine_wait_recv_cost_stats[gid] += 0;
                    }
                });
        });
    }
#endif
}

}  // namespace internode_ll
}  // namespace deep_ep
