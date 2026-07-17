#include <c10/util/Float8_e4m3fn.h>

#include "xpu_kernels.hpp"

#ifdef DEEP_EP_ENABLE_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

// ============================================================================
// Low-latency dispatch & combine: a SINGLE fused kernel each (CUDA-parity with
// the one-kernel LL design). There is no longer a runtime-selectable "barrier"
// vs "flag" path -- the earlier two-path split (env DEEP_EP_LL_FLAG_PROGRESS,
// selected by a now-removed ll_flag_progress()) is gone; only the fused kernel
// remains.
//
// Each fused kernel does put -> grid-sync -> recv in ONE launch (see the detailed
// per-phase comment at each parallel_for below). Synchronization is:
//   - ONE device-side GridBarrier == CUDA cg::this_grid().sync(), standing in for
//     the single grid barrier the CUDA LL kernel uses between the put and recv
//     phases. It syncs only the co-resident work-groups, NOT remote PEs.
//   - ONE cross-PE host ishmem_barrier_all() issued right before the launch. It
//     (a) supplies a cross-PCIe acquire so peer RDMA writes are observed and
//     (b) forces global iteration alignment (the count flags are NOT epoch-tagged,
//     so a stale flag from a prior iteration must not race a fresh one).
//   - Payload-before-flag ordering INSIDE the kernel is enforced without an extra
//     grid barrier via a per-send-channel atomic finish-counter (CUDA parity with
//     atomic_finish_counter_per_expert); the receiver spin-polls a per-channel
//     cross-PE arrival flag (bounded by DEEP_EP_LL_POLL_CAP).
//
// Fine-grained memory-ordering of the RDMA flag/payload traffic is tunable inside
// the fused kernel (these are knobs, NOT separate paths):
//   - DEEP_EP_LL_FLAG_SENDER_FENCE : flush uc_store-d flag bytes out to the domain
//     the NIC DMA-reads, so it never transmits a stale value (ll_flag_sender_fence).
//   - DEEP_EP_LL_FLAG_RECV_ACQ     : receiver acquire mode -- one system-scope
//     acquire + cached bulk reads vs per-byte uncached reads (ll_flag_recv_acq).
//   - DEEP_EP_LL_FLAG_LSC          : flag-read primitive, hint-based uc_load vs
//     explicit LSC uncached load (ll_flag_lsc_mode / ll_read_flag).
// uc_load is a genuine uncached load (equivalent to explicit LSC .uc.uc asm), the
// correct way to read a freshly RDMA-delivered flag without relying on the barrier.
// ============================================================================

namespace deep_ep {
namespace internode_ll {
namespace {

class CleanLowLatencyBufferKernel;
class UpdateMaskBufferKernel;
class QueryMaskBufferKernel;
class CleanMaskBufferKernel;

// Tier-1 multi-work-group low-latency kernels (multi-WG grid + sub-group
// collectives). The merged single-256-WI-work-group kernels above are kept for
// reference / fallback; the active dispatch/combine paths now use these.
class LowLatencyDispatchRouteKernel;
class LowLatencyDispatchFusedKernel;
class LowLatencyCombineScatterKernel;
class LowLatencyCombineFusedKernel;
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
    size_t barrier_offset;  // 2 x uint32_t: GridBarrier {counter, sense} scratch
    size_t finish_offset;   // 2*(num_ranks-1)*num_local_experts ints: per-send-channel
                            // atomic finish-counter + uc_store ready flag (CUDA
                            // atomic_finish_counter_per_expert parity). Shared by
                            // dispatch & combine (separate launches).
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
    // Double-buffered: 2 parity slots per (local_expert, src_rank). The flag for call
    // parity p lives at index (le*num_ranks+src)*2 + p; the opposite slot is cleaned at
    // the next call's send-start (CUDA next_clean parity), making the barrier-free
    // atomic-add epoch-safe for back-to-back dispatch.
    l.dispatch_count_bytes = static_cast<size_t>(num_local_experts) * num_ranks * 2 * sizeof(long);
    l.send_data_bytes = num_send_slots * hidden_bytes;
    l.send_src_bytes = num_send_slots * sizeof(int);
    l.send_count_bytes = static_cast<size_t>(num_ranks) * num_local_experts * sizeof(int);
    l.combine_data_bytes = num_combine_slots * hidden_bytes;
    l.combine_flag_bytes = static_cast<size_t>(num_experts) * 2 * sizeof(long);
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
    // GridBarrier scratch: 2 zero-initialized uint32_t (counter + sense) used by
    // the DEEP_EP_LL_FUSED single-kernel dispatch. Must mirror the identical
    // add() in deep_ep_xpu.cpp::get_low_latency_buffer_layout so total_bytes (and
    // therefore every offset) stays consistent between the allocator and here.
    l.barrier_offset = add(2 * sizeof(uint32_t));
    // Per-send-channel finish-counter (CUDA parity for atomic_finish_counter_per_expert):
    // TWO ints per remote send channel = 2*(num_ranks-1)*num_local_experts. The
    // first n ints are the atomic post-counter; the next n are a uc_store'd
    // "ready" flag (set by the last incrementer, uc_load-polled by the flag
    // sender) -- mirroring GridBarrier's atomic-count-then-uc-publish pattern so
    // the completion is observed cross-work-group on BMG (a plain/atomic load of
    // the counter is NOT cross-WG coherent for spinning). Lets the fused kernels
    // order payload-before-flag WITHOUT a grid barrier. Must mirror the identical
    // add() in deep_ep_xpu.cpp::get_low_latency_buffer_layout.
    l.finish_offset = add(static_cast<size_t>(2 * (num_ranks - 1) * num_local_experts) * sizeof(int));
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
//
// DEFAULT = 192 KiB. Measured (2-node, HIDDEN=7168, 2048 tokens):
// 64 KiB -> 1.67 GB/s (53.2 ms/iter), 128 KiB -> 1.85 GB/s, 192 KiB -> 2.00 GB/s
// (44.4 ms/iter, ~17% faster), 256 KiB -> 1.83 GB/s (regresses, nearing the slow
// path). Larger chunks post fewer WQEs per channel, and since Stage 2 issues the
// remote puts from a single work-item per channel, WQE-posting overhead is on the
// critical path. 192 KiB is the measured sweet spot below the slow-path cliff.
// Small token counts fit in one put regardless, so this never hurts them.
inline size_t ll_max_put_bytes() {
    const char* env = std::getenv("DEEP_EP_LL_MAX_PUT_KB");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0)
            return static_cast<size_t>(v) * 1024;
    }
    return static_cast<size_t>(192) * 1024;
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

// A/B toggle for the fused kernel's dispatch_count flag read primitive. Default 0 keeps
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

// Sender-side flush fence mode (fused kernel). After uc_store-ing the encoded
// flag value into the symmetric send staging, the NIC DMA-reads that staging to
// RDMA-WRITE it into the receiver's slot. If the GPU has not flushed the store
// past L3 to the system memory domain the NIC reads, the NIC can transmit a STALE
// value (e.g. 0), so the receiver spins forever (-> watchdog DEVICE_LOST on
// 2-node) even with a perfect uncached read. This emits a release/flush fence
// between the uc_store and the flag put:
//   0 = none   1 = sycl::atomic_fence(release, system) [DEFAULT, portable]
//   2 = LSC lsc_fence.ugm.evict.sysrel (explicit L3 evict to memory, ~1% faster)
// DEFAULT 1: dispatch/combine are INCORRECT on 2-node without this flush (empirically
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

// Number of work-groups for the coop warp-put payload kernel. The ordered
// commit gate in ishmemx_putmem_nbi_warp requires the producing sub-groups to be
// CO-RESIDENT (CUDA sizes its grid to num_sms for the same reason). Default to
// the device compute-unit count; override with DEEP_EP_LL_PUT_WGS.
inline int ll_put_wgs(sycl::queue& q) {
    const char* env = std::getenv("DEEP_EP_LL_PUT_WGS");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0)
            return v;
    }
    int cu = static_cast<int>(q.get_device().get_info<sycl::info::device::max_compute_units>());
    if (cu < 1) cu = 1;
    return cu;
}

// Grid size (work-groups) for the fused dispatch kernel. GridBarrier DEADLOCKS
// unless every launched work-group is concurrently resident, so this is bounded
// to the empirically-determined max co-resident count for a 256-work-item WG on
// this BMG. The header documents G=24 WGs of 256 as validated co-resident; the
// fused kernel is heavier (registers/SLM) than a trivial barrier probe, so we
// keep the default at the validated-safe 24 (>= any test's recv-channel count)
// rather than ll_put_wgs' 160 (which is NOT all co-resident and would hang).
// Override with DEEP_EP_LL_FUSED_WGS. Must be >= num_local_experts*num_ranks.
constexpr int kLLFusedMaxCoresidentWGs = 24;
inline int ll_fused_wgs(sycl::queue& q, int min_wgs) {
    int wgs;
    const char* env = std::getenv("DEEP_EP_LL_FUSED_WGS");
    if (env != nullptr && env[0] != '\0' && std::atoi(env) > 0) {
        wgs = std::atoi(env);
    } else {
        wgs = std::min(ll_put_wgs(q), kLLFusedMaxCoresidentWGs);
    }
    // Recv phase needs one WG per (local_expert, src_rank) channel.
    if (wgs < min_wgs) {
        wgs = min_wgs;
    }
    return wgs;
}

// Device-side flag read used by the fused kernel. lsc_mode 0 = hint-based uc_load
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

// 64-bit variant of ll_read_flag for the int64/long dispatch_count & combine_flag
// slots (posted with the native single-FADD ishmemx_long_atomic_add_qp). Each flag
// is a full 8-byte word, so no lsc_uc_load_i32 subword path -- a hint-based uncached
// uc_load<long> (optionally preceded by the sysacq invalidate fence) suffices.
inline long ll_read_flag64(const long* p, int lsc_mode) {
#ifdef __SYCL_DEVICE_ONLY__
    if (lsc_mode >= 2) {
        lsc_fence_sysacq();
    }
#endif
    return uc_load<long>(p);
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

// Receiver-side bulk-read acquire mode (fused kernel). The receiver originally
// read EVERY delivered payload byte with an uncached uc_load (L2-bypassing),
// so each byte paid a memory round-trip. Better: once the per-expert flags confirm
// the RDMA payload has landed, issue a SINGLE system-scope acquire / cache-invalidate
// at the start of the read phase so the GPU L2 becomes coherent with the NIC writes,
// then read the bulk payload with FAST CACHED loads. Trades a per-byte uncached tax
// for one invalidate + cached reads.
//   0 = per-byte uncached uc_load (fallback; ~15% slower on loopback)
//   1 = one sycl::atomic_fence(acquire, system) at kernel entry + cached reads (DEFAULT)
//   2 = one LSC lsc_fence.ugm.invalidate.sysacq at kernel entry + cached reads
//       (UNSTABLE on 2-node: reproducibly trips UR_RESULT_ERROR_DEVICE_LOST; do NOT use)
// Measured: loopback ~1581 us (mode 1) vs ~1870 us (mode 0); true 2-node
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
// DEFAULT = 50,000,000. The flag is posted with ishmemx_long_atomic_add_qp, which
// now BLOCKS to CQE completion on the sender (ibgda_device_impl.h rdma_atomic64),
// so cross-PE flag DELIVERY is reliable -- a polled flag is guaranteed to land, the
// only question is WHEN. The receiver reaches its recv-poll only after finishing its
// OWN send phase (payload puts + per-QP quiet + atomic flags to every peer), and the
// senders it waits on may still be draining their cold QPs, so on the FIRST /
// cold-start dispatch the sender->receiver skew is large: measured worst-case
// ~4.3M spins (steady-state is ~10-300K). The OLD default of 1,000,000 sat in the
// MIDDLE of that cold-start distribution, so ~2/3 of freshly-reset runs had at least
// one (le,src) flag arrive after the cap -> silently decoded as 0 tokens -> a small
// VARIABLE undercount (the "27!=31" LL flag-path bug). 50M gives ~11x margin over the
// observed cold-start worst case; the loop ALWAYS exits at the real arrival (a few
// million spins) in normal operation, so the cap is only ever approached by a genuine
// never-landing flag (which, given the blocking-to-CQE atomic, does not occur in a
// healthy env). Validated 5/5 on freshly-reset HW with 0 DEVICE_LOST. Lower it only
// to fail-fast harder on a suspected wedge; raise it if a slower fabric needs more
// cold-start headroom.
inline uint64_t ll_poll_cap() {
    const char* env = std::getenv("DEEP_EP_LL_POLL_CAP");
    if (env != nullptr && env[0] != '\0') {
        long long v = std::atoll(env);
        if (v > 0)
            return static_cast<uint64_t>(v);
    }
    return static_cast<uint64_t>(50) * 1000 * 1000;
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
                   int cur_parity,
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
    auto* dispatch_count = reinterpret_cast<long*>(base + layout.dispatch_count_offset);
    auto* send_data = base + layout.send_data_offset;
    auto* send_src = reinterpret_cast<int*>(base + layout.send_src_offset);
    auto* send_count = reinterpret_cast<int*>(base + layout.send_count_offset);

    const size_t send_count_elems = static_cast<size_t>(num_ranks) * num_local_experts;
    const size_t slot_elems = send_count_elems * num_max_dispatch_tokens_per_rank;
    const size_t recv_src_elems = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
    const size_t max_put = ll_max_put_bytes();
    const uint64_t poll_cap = ll_poll_cap();
    const int flag_lsc_mode = ll_flag_lsc_mode();
    const int flag_sender_fence = ll_flag_sender_fence();
    const int flag_recv_acq = ll_flag_recv_acq();
    // When the receiver does a single acquire/invalidate up front, the bulk payload
    // is read with CACHED loads instead of per-byte uncached uc_load.
    const bool recv_uncached = (flag_recv_acq == 0);
    // GridBarrier scratch: {counter, sense} live at layout.barrier_offset. Zeroed
    // right before the fused kernel launches (below); reusable across barriers.
    auto* barrier_scratch = reinterpret_cast<uint32_t*>(base + layout.barrier_offset);
    auto* combine_flag = base + layout.combine_flag_offset;
    (void)combine_flag;
    const int dispatch_parity = cur_parity & 1;

    // --- Stage 0: zero LOCAL staging + output tensors (multi-WG via memset).
    // Symmetric receive buffers (dispatch_*) are zeroed by clean_low_latency_buffer
    // with cross-PE barriers. send_src / packed_recv_src_info use -1 (0xFF bytes).
    queue.memset(send_count, 0, send_count_elems * sizeof(int));
    queue.memset(packed_recv_count, 0, static_cast<size_t>(num_local_experts) * sizeof(int));
    queue.memset(send_src, 0xFF, slot_elems * sizeof(int));
    queue.memset(packed_recv_src_info, 0xFF, recv_src_elems * sizeof(int));
    // NOTE: dispatch_count is double-buffered (2 parity slots per (le, src)). The flag
    // for THIS call lives at slot cur_parity; the OPPOSITE slot (1-cur_parity) is
    // cross-cleaned at the fused kernel's send-start (CUDA next_clean parity) so the
    // barrier-free atomic-add is epoch-safe for back-to-back dispatch. Both slots are
    // zeroed together only by clean_low_latency_buffer.

    // Pre-launch next_clean (double-buffer): zero the OPPOSITE parity slot of every
    // dispatch_count receive slot (all le x src). Runs as its own kernel on the in-order
    // queue, so it fully retires BEFORE the fused kernel's flag send -- giving the
    // causality (my clean < my flag send < peer recv < peer next-call add into this
    // slot) that makes the barrier-free atomic-add epoch-safe. Written via uc_store so
    // the peer's later RDMA add lands on a deterministically-zeroed slot.
    {
        const int n = num_local_experts * num_ranks;
        const int wg = 256;
        const int wgs = (n + wg - 1) / wg > 0 ? (n + wg - 1) / wg : 1;
        const int clean_parity = (cur_parity & 1) ^ 1;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyDispatchCleanFlagKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(wgs) * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> item) {
                    const int i = static_cast<int>(item.get_global_linear_id());
                    if (i < n) {
                        uc_store<long>(&dispatch_count[i * 2 + clean_parity], 0L);
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

    // --- FUSED single-kernel dispatch. CUDA-parity for the one-kernel LL dispatch:
    // send-put phase -> cg::this_grid().sync() -> recv. CoopPut + CoopFlag + quiet
    // + Recv are ONE launch, with a single GridBarrier standing in for CUDA's one
    // cg::this_grid().sync(). Grid = fused_wgs WGs of 256, fused_wgs co-residency-
    // bounded (see ll_fused_wgs) and >= num_recv_channels.
    //
    // Phases (per work-item) -- exactly ONE grid barrier (CUDA parity):
    //   0. Self-copy      : local (dst_rank==rank) send_*->dispatch_* copy + self
    //                       count flag (the receiver reads its own slot directly).
    //   1. Payload put    : coop warp-put of every remote channel's contiguous
    //                       payload block, grid-strided over all sub-groups. After
    //                       each put the sub-group leader bumps a per-channel atomic
    //                       finish-counter (CUDA atomic_finish_counter_per_expert).
    //   2. Flag put       : one sub-group per send channel spins on its finish-
    //                       counter "ready" flag (payload-before-flag ordering, no
    //                       grid barrier) then emits the count flag on qp=le.
    //   3. Quiet          : WG0 drains this PE's outbound NBI puts (send_data reuse).
    //   4. GridBarrier    : THE single grid sync (== CUDA this_grid().sync()).
    //   5. Recv/pack      : WG c owns recv channel c (< num_recv_channels); it
    //                       spin-polls its cross-PE arrival flag (GridBarrier does
    //                       NOT sync remote PEs, so the poll MUST stay), reserves an
    //                       output range, and packs the payload. Extra WGs return.
    {
        const int num_send_channels = (num_ranks - 1) * num_local_experts;
        const int num_recv_channels = num_local_experts * num_ranks;
        const int fused_wgs = ll_fused_wgs(queue, num_recv_channels);
        const size_t put_chunk_host = static_cast<size_t>(max_put);
        const int num_scales = (hidden % 128 == 0) ? hidden / 128 : 0;
        const int scale_packs = use_ue8m0 ? (num_scales + 3) / 4 : num_scales;
        auto* dst_fp8 = static_cast<uint8_t*>(packed_recv_x);
        auto* dst_scale_float = static_cast<float*>(packed_recv_x_scales);
        auto* dst_scale_int = static_cast<int32_t*>(packed_recv_x_scales);
        auto* barrier_counter = barrier_scratch;
        auto* barrier_sense = barrier_scratch + 1;
        // Per-send-channel atomic finish-counter (CUDA atomic_finish_counter_per_expert
        // parity): orders payload-before-flag WITHOUT a grid barrier. See Phase 1/2.
        // 2 ints/channel: [0..n) atomic post-counter, [n..2n) uc_store ready flag.
        auto* finish_counter = reinterpret_cast<int*>(base + layout.finish_offset);
        auto* finish_ready = finish_counter + num_send_channels;

        // Zero the {counter, sense} GridBarrier scratch + the finish counter/ready
        // arrays before the launch. memset on the in-order queue completes first.
        queue.memset(barrier_scratch, 0, 2 * sizeof(uint32_t));
        queue.memset(finish_counter, 0, static_cast<size_t>(2 * num_send_channels) * sizeof(int));

        // NO cross-PE host barrier. Barrier-free CUDA-parity: the flag is posted with
        // ishmemx_int_atomic_add_qp on the payload's QP (RC in-order => flag lands after
        // payload), each work-group drains only its own QP with ishmemx_quiet_qp, and
        // every dispatch cleans its own dispatch_count receive slots at send-start
        // (CUDA next_clean parity) so back-to-back calls never accumulate the atomic
        // add. Peer RDMA writes are observed via the recv-phase per-iteration
        // system-scope acquire fence + uncached flag load (no barrier acquire needed).
        // The accumulated HW/QP wedge that previously forced the barrier is now cleared
        // by the Python signal-handler resource cleanup.

        queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<int, 1> shared(sycl::range<1>(2), cgh);  // [0]=count, [1]=begin
            cgh.parallel_for<LowLatencyDispatchFusedKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(fused_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    GridBarrier gb(barrier_counter, barrier_sense, static_cast<uint32_t>(fused_wgs));
                    auto group = item.get_group();
                    auto sg = item.get_sub_group();
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    const int local_size = static_cast<int>(item.get_local_range(0));
                    const int wg = static_cast<int>(item.get_group_linear_id());
                    const int sgs_per_wg = static_cast<int>(sg.get_group_linear_range());
                    const int global_sg = wg * sgs_per_wg + static_cast<int>(sg.get_group_linear_id());
                    const int num_sgs = fused_wgs * sgs_per_wg;
                    const bool leader = (sg.get_local_id()[0] == 0);
                    const int global_id = static_cast<int>(item.get_global_id(0));
                    const int global_size = static_cast<int>(item.get_global_range(0));

                    // -------- Phase 0: local self-copy (dst_rank == rank) ----------
                    // The remote send loop below skips self; the receiver reads its
                    // OWN slot from dispatch_* directly. Copy send_* -> dispatch_* for
                    // the self slot and set the self count flag (encoded -count-1),
                    // grid-parallel. uc_store so the recv-phase uc_load observes it.
                    // (This is the self-copy that used to live in Stage 2's
                    // LowLatencyDispatchPutKernel, now folded into the fused kernel.)
                    for (int le = 0; le < num_local_experts; ++le) {
                        const int self_count = sycl::min(send_count[rank * num_local_experts + le],
                                                         num_max_dispatch_tokens_per_rank);
                        const size_t self_src_slot =
                            (static_cast<size_t>(rank) * num_local_experts + le) * num_max_dispatch_tokens_per_rank;
                        const size_t self_dst_slot =
                            (static_cast<size_t>(le) * num_ranks + rank) * num_max_dispatch_tokens_per_rank;
                        if (global_id == 0) {
                            uc_store<long>(&dispatch_count[(le * num_ranks + rank) * 2 + dispatch_parity], static_cast<long>(-self_count - 1));
                        }
                        for (int slot = global_id; slot < self_count; slot += global_size) {
                            uc_store(&dispatch_src[self_dst_slot + slot], send_src[self_src_slot + slot]);
                        }
                        coop_copy_bytes_store_uc(dispatch_data + self_dst_slot * hidden_bytes,
                                                 send_data + self_src_slot * hidden_bytes,
                                                 static_cast<size_t>(self_count) * hidden_bytes,
                                                 global_id, global_size);
                    }

                    // -------- Phase 1-3: per-channel payload + per-QP drain + flag ---
                    // Barrier-free CUDA-parity send. Each work-group owns a stride of
                    // channels (ch = wg, wg+fused_wgs, ...). For its channel it posts the
                    // payload on QP=le (batched, force_db=false), then the leader drains
                    // ONLY that QP with ishmemx_quiet_qp (concurrent-safe across work-
                    // groups -- unlike the old WG0-confined ishmemx_quiet_work_group) so
                    // every payload WQE has landed on the peer, and finally posts the
                    // count flag with ishmemx_int_atomic_add_qp on the SAME QP. RC in-
                    // order delivery on QP=le => the flag lands after the payload, with
                    // no host barrier, no WG0 confinement, and no finish-counter spin.
                    // The atomic-add lands on the send-start-cleaned dispatch_count slot
                    // (CUDA next_clean parity) so the receiver decodes -value-1 == count.
                    const size_t put_chunk = put_chunk_host;
                    const int sg_in_wg = static_cast<int>(sg.get_group_linear_id());
                    for (int ch = wg; ch < num_send_channels; ch += fused_wgs) {
                        const int dr = ch / num_local_experts;
                        const int le = ch % num_local_experts;
                        const int dst_rank = (dr < rank) ? dr : dr + 1;  // skip self
                        const int sc_idx = dst_rank * num_local_experts + le;
                        int count = 0;
                        if (leader) count = sycl::min(send_count[sc_idx], num_max_dispatch_tokens_per_rank);
                        count = sycl::group_broadcast(sg, count, 0);
                        const size_t src_slot = static_cast<size_t>(sc_idx) * num_max_dispatch_tokens_per_rank;
                        const size_t dst_slot = (static_cast<size_t>(le) * num_ranks + rank) * num_max_dispatch_tokens_per_rank;
                        if (count > 0) {
                            // src-index put: one sub-group per channel posts it.
                            if (sg_in_wg == 0) {
                                ishmemx_putmem_nbi_warp(dispatch_src + dst_slot, send_src + src_slot,
                                                        static_cast<size_t>(count) * sizeof(int), dst_rank,
                                                        static_cast<unsigned int>(le), true, sg, /*force_db=*/false);
                            }
                            // Payload chunks distributed across this WG's sub-groups.
                            const size_t total = static_cast<size_t>(count) * hidden_bytes;
                            const size_t nchunks = (total + put_chunk - 1) / put_chunk;
                            uint8_t* src_base = send_data + src_slot * hidden_bytes;
                            uint8_t* dst_base = dispatch_data + dst_slot * hidden_bytes;
                            for (size_t c = sg_in_wg; c < nchunks; c += sgs_per_wg) {
                                const size_t off = c * put_chunk;
                                const size_t this_bytes = sycl::min(put_chunk, total - off);
                                ishmemx_putmem_nbi_warp(dst_base + off, src_base + off, this_bytes,
                                                        dst_rank, static_cast<unsigned int>(le), true, sg, /*force_db=*/false);
                            }
                        }
                        // All sub-groups of this WG have posted their puts on QP=le.
                        sycl::group_barrier(group);
                        // Leader drains QP=le (payload delivered) then posts the count
                        // flag atomic-add on the SAME QP (RC in-order => flag after
                        // payload). count<=0 still posts the flag (encodes -1) so the
                        // receiver observes an arrival.
                        if (local_id == 0) {
                            ishmemx_quiet_qp(dst_rank, static_cast<unsigned int>(le));
                            ll_sender_flush(flag_sender_fence);
                            ishmemx_long_atomic_add_qp(dispatch_count + (le * num_ranks + rank) * 2 + dispatch_parity,
                                                      static_cast<long>(-count - 1), dst_rank, static_cast<unsigned int>(le));
                        }
                        // Hold the WG until the leader's quiet + flag completes before
                        // reusing shared state for the next channel iteration.
                        sycl::group_barrier(group);
                    }

                    // -------- Phase 4: THE SINGLE grid-sync (== CUDA this_grid().sync)
                    gb.arrive_and_wait(item);

                    // -------- Phase 5: recv/pack (RecvKernel logic) ---------------
                    // WGs beyond the recv-channel count did their share of the send
                    // and both barriers; they have nothing to receive.
                    if (wg >= num_recv_channels) {
                        return;
                    }
                    const int local_expert = wg / num_ranks;
                    const int src_rank = wg % num_ranks;
                    const int sg_id = static_cast<int>(sg.get_group_linear_id());

                    if (sg_id == 0) {
                        int count = 0;
                        if (local_id == 0) {
                            if (ll_rank_masked(mask_buffer_ptr, src_rank)) {
                                count = 0;
                            } else if (src_rank == rank) {
                                const long raw = ll_read_flag64(
                                    &dispatch_count[(local_expert * num_ranks + src_rank) * 2 + dispatch_parity],
                                    flag_lsc_mode);
                                count = (raw == 0) ? 0 : sycl::min(static_cast<int>(-raw - 1), num_max_dispatch_tokens_per_rank);
                            } else {
                                // Cross-PE arrival poll (barrier-free, double-buffered).
                                // The sender atomic-adds -(count+1) onto this call's parity
                                // slot, which was zeroed by the previous opposite-parity
                                // call's send-start clean (race-free by causality). Decode
                                // -raw-1. Follow the Intranode IPC polling rule: system-
                                // scope ACQUIRE fence + fresh uncached load + spin hint
                                // each iteration so the RDMA-delivered add is observed.
                                const int slot = (local_expert * num_ranks + src_rank) * 2 + dispatch_parity;
                                uint64_t spins = 0;
                                long raw = 0;
                                while (true) {
                                    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                                    raw = ll_read_flag64(&dispatch_count[slot], flag_lsc_mode);
                                    if (raw != 0) {
                                        break;
                                    }
                                    if (++spins >= poll_cap) {
                                        break;
                                    }
                                    visa_spin_hint();
                                }
                                count = (raw == 0) ? 0 : sycl::min(static_cast<int>(-raw - 1), num_max_dispatch_tokens_per_rank);
                            }
                            int begin = 0;
                            {
                                sycl::atomic_ref<int,
                                                 sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    recv_count(packed_recv_count[local_expert]);
                                begin = (count > 0) ? recv_count.fetch_add(count) : recv_count.load();
                            }
                            packed_recv_layout_range[local_expert * num_ranks + src_rank] =
                                static_cast<int64_t>(pack_range(count, begin));
                            if (cumulative_local_expert_recv_stats != nullptr && count > 0) {
                                sycl::atomic_ref<int,
                                                 sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>
                                    stat(cumulative_local_expert_recv_stats[local_expert]);
                                stat.fetch_add(count);
                            }
                            shared[0] = count;
                            shared[1] = begin;
                        }
                    }
                    sycl::group_barrier(group);
                    const int count = shared[0];
                    const int begin = shared[1];
                    if (count <= 0) {
                        return;
                    }
                    if (flag_recv_acq != 0) {
                        ll_recv_acquire(flag_recv_acq);
                    }
                    const int cohort_id = local_id;
                    const int cohort_size = local_size;
                    const size_t src_base =
                        (static_cast<size_t>(local_expert) * num_ranks + src_rank) * num_max_dispatch_tokens_per_rank;
                    const size_t dst_base =
                        static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin;
                    if (use_fp8) {
                        const size_t work = static_cast<size_t>(count) * num_scales;
                        for (size_t w = cohort_id; w < work; w += cohort_size) {
                            const int local_row = static_cast<int>(w / num_scales);
                            const int scale_idx = static_cast<int>(w % num_scales);
                            const auto* src_bf16 = reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(
                                dispatch_data + (src_base + local_row) * hidden_bytes);
                            const size_t dst_row = dst_base + local_row;
                            const int base_h = scale_idx * 128;
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
                                               static_cast<size_t>(count) * hidden_bytes,
                                               cohort_id,
                                               cohort_size);
                        } else {
                            coop_copy_bytes(static_cast<uint8_t*>(packed_recv_x) + dst_base * hidden_bytes,
                                            dispatch_data + src_base * hidden_bytes,
                                            static_cast<size_t>(count) * hidden_bytes,
                                            cohort_id,
                                            cohort_size);
                        }
                    }
                    for (int slot = cohort_id; slot < count; slot += cohort_size) {
                        packed_recv_src_info[dst_base + slot] =
                            recv_uncached ? uc_load(&dispatch_src[src_base + slot]) : dispatch_src[src_base + slot];
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
                  int cur_parity,
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
    auto* combine_flag_i = reinterpret_cast<long*>(base + layout.combine_flag_offset);
    auto* dispatch_count = reinterpret_cast<long*>(base + layout.dispatch_count_offset);
    auto* send_count = reinterpret_cast<int*>(base + layout.send_count_offset);
    (void)dispatch_count;
    const int combine_parity = cur_parity & 1;

    const size_t send_elems = static_cast<size_t>(num_ranks) * num_local_experts * num_max_dispatch_tokens_per_rank;
    const size_t max_put = ll_max_put_bytes();
    const uint64_t poll_cap = ll_poll_cap();
    const int flag_lsc_mode = ll_flag_lsc_mode();
    const int flag_sender_fence = ll_flag_sender_fence();
    const int flag_recv_acq = ll_flag_recv_acq();
    const bool recv_uncached = (flag_recv_acq == 0);
    // GridBarrier scratch (reuses the same layout.barrier_offset added for the
    // fused dispatch). Re-zeroed right before the fused combine launch below.
    auto* barrier_scratch = reinterpret_cast<uint32_t*>(base + layout.barrier_offset);

    // --- Stage 0: zero local send staging (bf16 zero == 0x0000). combine_data and
    // combine_flag are already zeroed by clean_low_latency_buffer (cross-PE barrier).
    queue.memset(send_data, 0, send_elems * hidden_bytes);
    // Pre-launch next_clean (double-buffer): zero the OPPOSITE parity slot of every
    // combine_flag receive slot (all global experts). combine_flag has 2 ints/expert,
    // used as the 2 parity slots [ge*2+p]. Runs as its own kernel on the in-order queue
    // so it retires before the fused flag send (same causality as dispatch).
    {
        const int n = num_experts;
        const int wg = 256;
        const int wgs = (n + wg - 1) / wg > 0 ? (n + wg - 1) / wg : 1;
        const int clean_parity = (cur_parity & 1) ^ 1;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombineCleanCountKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(wgs) * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> item) {
                    const int ge = static_cast<int>(item.get_global_linear_id());
                    if (ge < n) {
                        uc_store<long>(&combine_flag_i[ge * 2 + clean_parity], 0L);
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



    // --- FUSED single-kernel combine (DEEP_EP_LL_FUSED). CUDA-parity for the
    // one-kernel LL combine: send-put -> cg::this_grid().sync() -> reduce. Here
    // CombineCoopPut + CombineCoopFlag + quiet + flag-poll + Reduce are ONE launch
    // with a GridBarrier standing in for CUDA's this_grid().sync(). Grid =
    // fused_wgs WGs of 256, co-residency-bounded (ll_fused_wgs).
    //
    // Phases (per work-item):
    //   1. Payload put   : coop warp-put of each channel's contiguous token span,
    //                      per-channel atomic finish-counter + uc_store ready flag.
    //   2. Flag put      : one sub-group per send channel spins on its ready flag
    //                      (payload-before-flag ordering, no grid barrier) then
    //                      sets the remote flag.
    //   3. Quiet         : every WG drains this PE's outbound NBI puts.
    //   4. Flag poll     : grid-strided spin-poll of every remote-owned expert flag
    //                      (cross-PE arrival sync -- GridBarrier does NOT sync PEs,
    //                      so the poll is mandatory before the reduce).
    //   5. GridBarrier    : THE single grid-sync (== CUDA this_grid().sync()) --
    //                      every expert flag observed grid-wide before ANY reduce.
    //   6. Reduce        : grid-strided weighted top-k reduction into combined_x.
    {
        const int num_channels = (num_ranks - 1) * num_local_experts;
        const int num_send_channels = num_channels;
        const int fused_wgs = ll_fused_wgs(queue, num_local_experts * num_ranks);
        const size_t put_chunk_host = static_cast<size_t>(max_put);
        const size_t reduce_work = static_cast<size_t>(num_combined_tokens) * hidden;
        auto* barrier_counter = barrier_scratch;
        auto* barrier_sense = barrier_scratch + 1;
        // Per-send-channel atomic finish-counter (CUDA parity for
        // atomic_finish_counter_per_expert): orders payload-before-flag per channel
        // WITHOUT a grid barrier. Reuses the same layout.finish_offset as dispatch
        // (both are separate in-order launches, sized identically). 2 ints/channel:
        // [0..n) atomic post-counter, [n..2n) uc_store ready flag. Re-zeroed here.
        auto* finish_counter = reinterpret_cast<int*>(base + layout.finish_offset);
        auto* finish_ready = finish_counter + num_send_channels;

        // Zero the {counter, sense} GridBarrier scratch + finish arrays before the
        // launch (dispatch and combine are separate calls, so re-zero fresh).
        queue.memset(barrier_scratch, 0, 2 * sizeof(uint32_t));
        queue.memset(finish_counter, 0, static_cast<size_t>(2 * num_send_channels) * sizeof(int));
        // NO cross-PE host barrier (see dispatch). Barrier-free: the arrival flag is
        // posted with ishmemx_int_atomic_add_qp on the payload's QP (RC in-order =>
        // flag after payload), each work-group drains only its own QP with
        // ishmemx_quiet_qp, and every combine cleans its own combine_flag receive slots
        // at send-start (CUDA next_clean parity) so a stale nonzero flag from a prior
        // call is never mistaken for a fresh arrival.

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LowLatencyCombineFusedKernel>(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(fused_wgs) * kLLWGSize), sycl::range<1>(kLLWGSize)),
                [=](sycl::nd_item<1> item) {
                    GridBarrier gb(barrier_counter, barrier_sense, static_cast<uint32_t>(fused_wgs));
                    auto group = item.get_group();
                    auto sg = item.get_sub_group();
                    const int wg = static_cast<int>(item.get_group_linear_id());
                    const int sgs_per_wg = static_cast<int>(sg.get_group_linear_range());
                    const int global_sg = wg * sgs_per_wg + static_cast<int>(sg.get_group_linear_id());
                    const int num_sgs = fused_wgs * sgs_per_wg;
                    const int global_id = static_cast<int>(item.get_global_id(0));
                    const int global_size = static_cast<int>(item.get_global_range(0));

                    // NOTE: no send-start next_clean of combine_flag. Flags are now
                    // MONOTONIC atomic-add counters; the receiver decodes arrival from a
                    // LOCAL per-slot baseline (see the flag poll below), and the total is
                    // zeroed only by clean_low_latency_buffer (together with the baseline).

                    // -------- Phase 0: local self-copy (dst_rank == rank) ----------
                    // The remote send loop skips self-owned experts; the reduce reads
                    // self combine_data directly and the flag poll skips self owners.
                    // Copy the self min/max token span send_data -> combine_data and
                    // set the self combine flag (=1) locally, grid-parallel via
                    // uc_store. (Folds in Stage 2's LowLatencyCombinePutKernel self-copy.)
                    const size_t put_chunk = put_chunk_host;
                    const bool leader = (sg.get_local_id()[0] == 0);
                    for (int le = 0; le < num_local_experts; ++le) {
                        const int self_ge = rank * num_local_experts + le;
                        int scount = 0, sbegin = 0;
                        unpack_range(layout_range[le * num_ranks + rank], scount, sbegin);
                        const int sclamped = sycl::min(scount, num_max_dispatch_tokens_per_rank);
                        int smin = num_max_dispatch_tokens_per_rank;
                        int smax = -1;
                        for (int slot = 0; slot < sclamped; ++slot) {
                            const int ot =
                                src_info[static_cast<size_t>(le) * num_ranks * num_max_dispatch_tokens_per_rank + sbegin + slot];
                            if (ot >= 0 && ot < num_max_dispatch_tokens_per_rank) {
                                smin = sycl::min(smin, ot);
                                smax = sycl::max(smax, ot);
                            }
                        }
                        if (smax >= smin) {
                            auto* s_src = send_data +
                                (static_cast<size_t>(rank * num_local_experts + le) * num_max_dispatch_tokens_per_rank + smin) *
                                hidden_bytes;
                            auto* s_dst = combine_data +
                                (static_cast<size_t>(self_ge) * num_max_dispatch_tokens_per_rank + smin) * hidden_bytes;
                            coop_copy_bytes_store_uc(s_dst, s_src,
                                                     static_cast<size_t>(smax - smin + 1) * hidden_bytes,
                                                     global_id, global_size);
                        }
                        if (global_id == 0) {
                            uc_store<long>(&combine_flag_i[self_ge * 2 + combine_parity], 1L);
                        }
                    }

                    // -------- Phase 1-3: per-channel payload + per-QP drain + flag ---
                    // Barrier-free CUDA-parity (see dispatch). Each work-group owns a
                    // stride of channels (ch = wg, wg+fused_wgs, ...), posts that
                    // channel's reduced payload span on QP=le (batched, force_db=false),
                    // drains ONLY that QP with ishmemx_quiet_qp (concurrent-safe across
                    // work-groups, unlike the old WG0-confined quiet_work_group), then
                    // the leader posts the arrival flag with ishmemx_int_atomic_add_qp
                    // (+1) on the SAME QP (RC in-order => flag after payload). The flag
                    // lands on the send-start-cleaned combine_flag slot so the receiver
                    // reads nonzero == arrived-this-call.
                    const int local_id = static_cast<int>(item.get_local_id(0));
                    const int sg_in_wg = static_cast<int>(sg.get_group_linear_id());
                    for (int ch = wg; ch < num_channels; ch += fused_wgs) {
                        const int dr = ch / num_local_experts;
                        const int le = ch % num_local_experts;
                        const int dst_rank = (dr < rank) ? dr : dr + 1;
                        const int global_expert = rank * num_local_experts + le;
                        int count = 0, begin = 0;
                        unpack_range(layout_range[le * num_ranks + dst_rank], count, begin);
                        const int clamped_count = sycl::min(count, num_max_dispatch_tokens_per_rank);
                        int min_token = num_max_dispatch_tokens_per_rank;
                        int max_token = -1;
                        for (int slot = 0; slot < clamped_count; ++slot) {
                            const int original_token =
                                src_info[static_cast<size_t>(le) * num_ranks * num_max_dispatch_tokens_per_rank + begin + slot];
                            if (original_token >= 0 && original_token < num_max_dispatch_tokens_per_rank) {
                                min_token = sycl::min(min_token, original_token);
                                max_token = sycl::max(max_token, original_token);
                            }
                        }
                        if (max_token >= min_token) {
                            const int span = max_token - min_token + 1;
                            uint8_t* src_base = send_data +
                                (static_cast<size_t>(dst_rank * num_local_experts + le) * num_max_dispatch_tokens_per_rank + min_token) *
                                hidden_bytes;
                            uint8_t* dst_base = combine_data +
                                (static_cast<size_t>(global_expert) * num_max_dispatch_tokens_per_rank + min_token) * hidden_bytes;
                            const size_t total = static_cast<size_t>(span) * hidden_bytes;
                            const size_t nchunks = (total + put_chunk - 1) / put_chunk;
                            for (size_t c = sg_in_wg; c < nchunks; c += sgs_per_wg) {
                                const size_t off = c * put_chunk;
                                const size_t this_bytes = sycl::min(put_chunk, total - off);
                                ishmemx_putmem_nbi_warp(dst_base + off, src_base + off, this_bytes,
                                                        dst_rank, static_cast<unsigned int>(le), true, sg, /*force_db=*/false);
                            }
                        }
                        // All sub-groups of this WG have posted their puts on QP=le.
                        sycl::group_barrier(group);
                        // Leader drains QP=le then posts the arrival flag atomic-add on
                        // the SAME QP. Empty-span channels still post the flag (+1) so
                        // the receiver observes an arrival.
                        if (local_id == 0) {
                            ishmemx_quiet_qp(dst_rank, static_cast<unsigned int>(le));
                            ll_sender_flush(flag_sender_fence);
                            ishmemx_long_atomic_add_qp(combine_flag_i + global_expert * 2 + combine_parity,
                                                      1L, dst_rank, static_cast<unsigned int>(le));
                        }
                        sycl::group_barrier(group);
                    }

                    // -------- Phase 4: flag poll (CombineWait logic, grid-strided) -
                    // Cross-PE arrival sync spread across ALL work-items of ALL WGs.
                    // Double-buffered atomic-add flag: the sender adds +1 onto this call's
                    // parity slot (zeroed by the previous opposite-parity call's send-start
                    // clean), so a nonzero value means THIS epoch's payload has arrived.
                    for (int ge = global_id; ge < num_experts; ge += global_size) {
                        const int owner = ge / num_local_experts;
                        if (owner == rank) {
                            continue;  // self-owned: flag set locally
                        }
                        if (ll_rank_masked(mask_buffer_ptr, owner)) {
                            continue;  // masked owner never sends
                        }
                        uint64_t spins = 0;
                        while (true) {
                            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                            if (ll_read_flag64(&combine_flag_i[ge * 2 + combine_parity], flag_lsc_mode) != 0) {
                                break;
                            }
                            if (++spins >= poll_cap) {
                                break;  // timeout -> proceed; Reduce reads zeros
                            }
                            visa_spin_hint();
                        }
                    }

                    // -------- Phase 5: THE SINGLE grid-sync (== CUDA this_grid().sync)
                    // Ensure every expert flag observed grid-wide before ANY reduce.
                    gb.arrive_and_wait(item);

                    // -------- Phase 6: reduce (CombineReduce logic) ---------------
                    auto* out = static_cast<sycl::ext::oneapi::bfloat16*>(combined_x);
                    if (flag_recv_acq != 0) {
                        ll_recv_acquire(flag_recv_acq);
                    }
                    for (size_t idx = global_id; idx < reduce_work; idx += global_size) {
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
                    if (combine_wait_recv_cost_stats != nullptr && global_id < num_ranks) {
                        combine_wait_recv_cost_stats[global_id] += 0;
                    }
                });
        });
    }
#endif
}

}  // namespace internode_ll
}  // namespace deep_ep
