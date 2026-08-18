#include <c10/util/Float8_e4m3fn.h>

#include "xpu_kernels.hpp"

#ifdef DEEP_EP_ENABLE_ISHMEM
#include <ishmem.h>
#include <ishmemx.h>
#endif

// ============================================================================
// Low-latency dispatch & combine: SINGLE fused kernels (CUDA-parity).
// NamedBarrier (warp-group-subset) + root_group grid barrier.
// No SLM (nd_launch limitation) — private vars with sub-group/WG barriers.
// IGC_SelectiveFunctionControl=1 REQUIRED at build time.
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
// One fused kernel per collective. NamedBarrier + root_group.
class LowLatencyDispatchFusedKernel;
class LowLatencyCombineFusedKernel;
// Obsolete phase-split names (not instantiated).
class LLDispatchSendKernel;
class LLDispatchCountKernel;
class LLDispatchRecvKernel;
class LLCombineSendKernel;
class LLCombineReduceKernel;

// CUDA-parity unified per-token message: [int4 header (src_idx + 3 reserved)]
// [payload (fp8 | bf16)] [fp8 scale_inv floats]. num_bytes_per_msg is sized at the
// bf16 max (which is >= the fp8 size) so a single allocation serves both dtypes.
inline size_t ll_num_bytes_per_msg(int hidden) {
    const int num_scales = (hidden % 128 == 0) ? hidden / 128 : 0;
    const size_t bf16 = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    const size_t fp8 = static_cast<size_t>(hidden) + static_cast<size_t>(num_scales) * sizeof(float);
    return sizeof(int) * 4 + std::max(bf16, fp8);
}

struct LowLatencyLayout {
    size_t combine_data_bytes;
    size_t combine_flag_bytes;
    size_t mask_bytes;

    // Dispatch unified-message regions (CUDA rdma_recv_x / rdma_x / rdma_recv_count).
    size_t dispatch_data_offset;   // rdma_recv_x: num_dispatch_slots * num_bytes_per_msg
    size_t rdma_x_offset;          // rdma_x    : num_max_dispatch_tokens * num_bytes_per_msg (send staging)
    size_t dispatch_count_offset;  // rdma_recv_count: num_local_experts*num_ranks*2 longs (double-buffered flag)
    // Combine staging/regions (unchanged; still per-hidden-row arrays).
    size_t send_data_offset;
    size_t send_count_offset;
    size_t combine_data_offset;
    size_t combine_flag_offset;
    size_t mask_offset;
    size_t sync_offset;
    size_t barrier_offset;  // 2 x uint32_t: GridBarrier {counter, sense} scratch (combine)
    size_t slot_counter_offset;    // num_experts ints: CUDA atomic_counter_per_expert.
    size_t finish_counter_offset;  // num_experts ints: CUDA atomic_finish_counter_per_expert
                                   // (barrier-free per-expert send completion; system-scope add).
    size_t finish_ready_offset;    // num_experts ints: uc_store'd "reached target" flag published
                                   // by the last incrementer, uc_load-polled by the count-sender
                                   // (BMG cross-work-group coherence, GridBarrier uc-publish pattern).
    size_t total_bytes;
};

inline LowLatencyLayout make_layout(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    const int num_local_experts = num_experts / num_ranks;
    LowLatencyLayout l{};
    const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    const size_t msg_bytes = ll_num_bytes_per_msg(hidden);
    const size_t num_dispatch_slots = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
    const size_t num_send_slots = static_cast<size_t>(num_ranks) * num_local_experts * num_max_dispatch_tokens_per_rank;
    const size_t num_combine_slots = static_cast<size_t>(num_experts) * num_max_dispatch_tokens_per_rank;
    l.combine_data_bytes = num_combine_slots * hidden_bytes;
    l.combine_flag_bytes = static_cast<size_t>(num_experts) * 2 * sizeof(long);
    l.mask_bytes = static_cast<size_t>(num_ranks) * sizeof(int);

    size_t offset = 0;
    auto add = [&](size_t bytes) {
        const size_t old = offset;
        offset = align_up<size_t>(offset + bytes, NUM_BUFFER_ALIGNMENT_BYTES);
        return old;
    };
    // --- Dispatch unified-message regions (CUDA rdma_recv_x / rdma_x / rdma_recv_count).
    // rdma_recv_x: per (local_expert, src_rank, slot) message received from peers.
    l.dispatch_data_offset = add(num_dispatch_slots * msg_bytes);
    // rdma_x: per LOCAL token message staging (cast+packed, then put per (token,expert)).
    l.rdma_x_offset = add(static_cast<size_t>(num_max_dispatch_tokens_per_rank) * msg_bytes);
    // rdma_recv_count: double-buffered count flag, 2 parity slots per (local_expert, src_rank).
    // Flag for call parity p lives at index (le*num_ranks+src)*2 + p; the opposite slot is
    // cleaned at the next call's send-start (CUDA next_clean parity) -> epoch-safe.
    l.dispatch_count_offset = add(static_cast<size_t>(num_local_experts) * num_ranks * 2 * sizeof(long));
    // --- Combine staging/regions (unchanged from the validated combine path).
    l.send_data_offset = add(num_send_slots * hidden_bytes);
    l.send_count_offset = add(static_cast<size_t>(num_ranks) * num_local_experts * sizeof(int));
    l.combine_data_offset = add(l.combine_data_bytes);
    l.combine_flag_offset = add(l.combine_flag_bytes);
    l.mask_offset = add(l.mask_bytes);
    l.sync_offset = add(static_cast<size_t>(num_ranks) * sizeof(int));
    // GridBarrier scratch: 2 zero-initialized uint32_t (counter + sense), used by the
    // fused combine kernel. Must mirror the identical add() in
    // deep_ep_xpu.cpp::get_low_latency_buffer_layout so total_bytes/offsets stay consistent.
    l.barrier_offset = add(2 * sizeof(uint32_t));
    // Per-expert global atomic slot counter (CUDA atomic_counter_per_expert), num_experts ints.
    l.slot_counter_offset = add(static_cast<size_t>(num_experts) * sizeof(int));
    // Per-expert finish-counter (CUDA atomic_finish_counter_per_expert), num_experts ints:
    // barrier-free per-expert send completion. Worker sub-groups system-scope-add +1 per
    // send; the responsible WG adds (FINISHED_SUM_TAG - count); WG0 adds FINISHED_SUM_TAG to
    // every expert. Target == 2*FINISHED_SUM_TAG.
    l.finish_counter_offset = add(static_cast<size_t>(num_experts) * sizeof(int));
    // Per-expert "reached target" ready flag, num_experts ints: uc_store'd by whichever
    // incrementer lands the 2*TAG-th unit, uc_load-polled by the count-sender (BMG
    // cross-work-group coherence -- a plain/atomic counter load does NOT observe cross-WG).
    l.finish_ready_offset = add(static_cast<size_t>(num_experts) * sizeof(int));
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

// CUDA FINISHED_SUM_TAG (configs.cuh): per-expert finish-counter target is 2x this.
// Accounting per expert e: WG0 seeds +TAG; the responsible WG's counter warp adds
// (TAG - count); each of the `count` sends adds +1 => final == 2*TAG.
constexpr int kFinishedSumTag = 1024;

// Tier-1 multi-WG tuning.
constexpr int kLLWGSize = 128;  // work-items per work-group (4 sub-groups of 32)
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
// commit gate in ishmemx_putmem_nbi_subgroup requires the producing sub-groups to be
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

// Grid size for the barrier-free 3-kernel dispatch SEND kernel. Because sends are
// grid-strided over tokens with NO cross-work-group spin/barrier (the kernel
// boundary provides the completion guarantee), the SEND grid has NO co-residency
// constraint and can be as large as useful for token parallelism. Default to the
// device compute-unit count capped by the token count; override with DEEP_EP_LL_SEND_WGS.
inline int ll_send_wgs(sycl::queue& q, int num_tokens) {
    const char* env = std::getenv("DEEP_EP_LL_SEND_WGS");
    if (env != nullptr && env[0] != '\0' && std::atoi(env) > 0) {
        return std::atoi(env);
    }
    int wgs = ll_put_wgs(q);
    if (num_tokens > 0 && wgs > num_tokens) wgs = num_tokens;
    if (wgs < 1) wgs = 1;
    return wgs;
}

// Work-group size and grid for the token-parallel CONSUME kernels (combine reduce,
// dispatch recv-copy). Unlike the send/put kernels these do PURE local memory
// ops (no IBGDA warp put -> no per-QP ordered-commit gate), so the grid has NO
// co-residency constraint and can be oversubscribed: extra work-groups simply
// serialize through the scheduler with no spin/barrier, so more blocks strictly
// reduce the per-thread grid-stride count. CUDA scales its combine grid with the
// token count for exactly this reason (num_sms = max(num_experts,
// num_combined_tokens / num_recv_per_sm)). Default to a generous multiple of the
// device compute units capped at kLLConsumeMaxWGs; override via env.
constexpr int kLLConsumeWGSize = 512;
constexpr int kLLConsumeMaxWGs = 512;
inline int ll_consume_wgs(sycl::queue& q, int units, int min_wgs) {
    const char* env = std::getenv("DEEP_EP_LL_REDUCE_WGS");
    int wgs;
    if (env != nullptr && env[0] != '\0' && std::atoi(env) > 0) {
        wgs = std::atoi(env);
    } else {
        int cu = static_cast<int>(q.get_device().get_info<sycl::info::device::max_compute_units>());
        if (cu < 1) cu = 1;
        wgs = std::min(cu * 4, kLLConsumeMaxWGs);
    }
    if (wgs > units && units > 0) wgs = units;   // no point launching more blocks than work items
    if (wgs < min_wgs) wgs = min_wgs;
    if (wgs < 1) wgs = 1;
    return wgs;
}

// Grid size (work-groups) for the fused dispatch/combine kernels. GridBarrier
// DEADLOCKS unless every launched work-group is concurrently resident. With the
// 128-work-item WG (kLLWGSize, 4 sub-groups of 32) each WG uses half the threads
// of the old 256-WI WG, so many more WGs co-reside: 48/64 WGs launch and complete
// the grid barrier fine on this BMG (160 EUs). The limiter is now PERFORMANCE, not
// co-residency: a token-count sweep (H7168, 8 experts) shows avg time bottoms out
// at 32 WGs for 128/256 tokens (128: 3969us, 256: 6744us) and at ~16 WGs for 32
// tokens; beyond 32 the GridBarrier cost dominates (48/64 regress). So default to
// 32 -- the best single value across the 32..256-token scaling range and stable
// (tight tail). Override with DEEP_EP_LL_FUSED_WGS. Must be >= num_local_experts*num_ranks.
constexpr int kLLFusedMaxCoresidentWGs = 32;
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

// CUDA calculate_fp8_scales parity (utils.cuh:484). Caller floors amax at
// kFP8Margin (1e-4). round_scale uses the bit-exact fast_log2_ceil/fast_pow2
// integer-exponent path (NOT ceil(log2)/exp2), so the scale is a power of two
// identical to the CUDA sender.
inline void ll_calc_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
    if (round_scale) {
        const float x = amax * (1.0f / 448.0f);  // kFinfoAmaxInvE4M3
        const uint32_t bx = sycl::bit_cast<uint32_t>(x);
        const int exp_x = static_cast<int>((bx >> 23) & 0xffu);
        const uint32_t man = bx & ((1u << 23) - 1u);
        const int e = exp_x - 127 + (man != 0u ? 1 : 0);  // fast_log2_ceil
        scale = sycl::bit_cast<float>(static_cast<uint32_t>((-e + 127)) << 23);      // fast_pow2(-e)
        scale_inv = sycl::bit_cast<float>(static_cast<uint32_t>((e + 127)) << 23);   // fast_pow2(e)
    } else {
        scale_inv = amax * (1.0f / 448.0f);
        scale = 448.0f / amax;
    }
}

// Cooperative bf16->fp8 (E4M3) cast of one token's `hidden` elements by a single
// sub-group, with per-128-channel amax scaling -- CUDA dispatch send-phase parity
// (internode_ll.cu:217). Writes fp8 bytes to `dst_fp8` (contiguous, stride hidden)
// and one scale_inv float per 128-block to `dst_scales` (num_scales floats). The
// amax is reduced across the whole sub-group per block (32 lanes x per_lane = 128).
// kStoreUC selects uc_store (self write-through into the symmetric recv region so a
// later uc_load reader observes it) vs a plain cached store (remote send staging,
// flushed by a system-release fence before the RDMA put).
template <bool kStoreUC>
inline void coop_cast_bf16_to_fp8(uint8_t* dst_fp8,
                                  float* dst_scales,
                                  const sycl::ext::oneapi::bfloat16* src,
                                  int num_scales,
                                  bool round_scale,
                                  const sycl::sub_group& sg,
                                  int lane,
                                  int sg_size) {
    const int per_lane = 128 / sg_size;  // 4 for the reqd sub-group size 32
    for (int blk = 0; blk < num_scales; ++blk) {
        const int base = blk * 128;
        float vals[8];
        float amax = 1.0e-4f;  // kFP8Margin
        for (int j = 0; j < per_lane; ++j) {
            const float fv = static_cast<float>(src[base + lane * per_lane + j]);
            vals[j] = fv;
            amax = sycl::fmax(amax, sycl::fabs(fv));
        }
        amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
        float scale, scale_inv;
        ll_calc_fp8_scales(amax, scale, scale_inv, round_scale);
        if (lane == 0) {
            if constexpr (kStoreUC) {
                uc_store<float>(&dst_scales[blk], scale_inv);
            } else {
                dst_scales[blk] = scale_inv;
            }
        }
        for (int j = 0; j < per_lane; ++j) {
            const int idx = base + lane * per_lane + j;
            const uint8_t b = c10::Float8_e4m3fn(vals[j] * scale).x;
            if constexpr (kStoreUC) {
                uc_store<uint8_t>(&dst_fp8[idx], b);
            } else {
                dst_fp8[idx] = b;
            }
        }
    }
}

#ifdef DEEP_EP_ENABLE_ISHMEM
// Multi-sub-group cooperative bf16->fp8 cast: like coop_cast_bf16_to_fp8<false>
// but each caster sub-group owns a STRIDE of the 128-channel blocks (blk = warp_id;
// blk += num_caster_warps) so several sub-groups cast ONE token's message together
// (CUDA send-phase: all num_warps-1 worker warps cooperate on one token). Plain
// (cached) stores; the caller issues a system-release fence before the RDMA put.
inline void cast_token_fp8_strided(uint8_t* dst_fp8,
                                   float* dst_scales,
                                   const sycl::ext::oneapi::bfloat16* src,
                                   int num_scales,
                                   bool round_scale,
                                   const sycl::sub_group& sg,
                                   int lane,
                                   int sg_size,
                                   int blk_start,
                                   int blk_stride) {
    const int per_lane = 128 / sg_size;  // 4 for the reqd sub-group size 32
    for (int blk = blk_start; blk < num_scales; blk += blk_stride) {
        const int base = blk * 128;
        float vals[8];
        float amax = 1.0e-4f;  // kFP8Margin
        for (int j = 0; j < per_lane; ++j) {
            const float fv = static_cast<float>(src[base + lane * per_lane + j]);
            vals[j] = fv;
            amax = sycl::fmax(amax, sycl::fabs(fv));
        }
        amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
        float scale, scale_inv;
        ll_calc_fp8_scales(amax, scale, scale_inv, round_scale);
        if (lane == 0)
            dst_scales[blk] = scale_inv;
        for (int j = 0; j < per_lane; ++j) {
            const int idx = base + lane * per_lane + j;
            dst_fp8[idx] = c10::Float8_e4m3fn(vals[j] * scale).x;
        }
    }
}

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
    const int num_scales = (hidden % 128 == 0) ? hidden / 128 : 0;
    const size_t msg_bytes = ll_num_bytes_per_msg(hidden);
    const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(sycl::ext::oneapi::bfloat16);
    // Wire message = [int4 header][payload (fp8 1B|bf16 2B)][fp8 scales]. `used_bytes`
    // (== CUDA num_bytes_per_msg) is the actually-transferred prefix; `msg_bytes` is the
    // fixed dtype-independent slot stride.
    const size_t payload_bytes = use_fp8 ? static_cast<size_t>(hidden) : hidden_bytes;
    const size_t scales_off = sizeof(int) * 4 + payload_bytes;
    const size_t used_bytes = sizeof(int) * 4 + payload_bytes + (use_fp8 ? static_cast<size_t>(num_scales) * sizeof(float) : 0);

    auto* base = static_cast<uint8_t*>(rdma_buffer);
    auto* rdma_recv_x = base + layout.dispatch_data_offset;   // recv region: peers put messages here
    auto* rdma_x = base + layout.rdma_x_offset;               // send staging: one message per local token
    auto* rdma_recv_count = reinterpret_cast<long*>(base + layout.dispatch_count_offset);
    auto* slot_counter = reinterpret_cast<int*>(base + layout.slot_counter_offset);          // CUDA atomic_counter_per_expert
    // finish_counter: barrier-free per-expert send completion (used by fused dispatch kernel).

    const size_t recv_src_elems = static_cast<size_t>(num_local_experts) * num_ranks * num_max_dispatch_tokens_per_rank;
    const uint64_t poll_cap = ll_poll_cap();
    const int flag_lsc_mode = ll_flag_lsc_mode();
    const int flag_sender_fence = ll_flag_sender_fence();
    const int flag_recv_acq = ll_flag_recv_acq();
    const bool recv_uncached = (flag_recv_acq == 0);
    const int dispatch_parity = cur_parity & 1;
    const int num_recv_channels = num_local_experts * num_ranks;
    const int dst_scale_stride = use_ue8m0 ? ((num_scales + 3) / 4) : num_scales;
    auto* dst_scale_float = static_cast<float*>(packed_recv_x_scales);
    auto* dst_scale_int = static_cast<int32_t*>(packed_recv_x_scales);
    auto* dst_data = static_cast<uint8_t*>(packed_recv_x);

    // ---- CUDA launch geometry (internode_ll.cu::dispatch host code). With num_experts
    // <= compute-units this yields num_warp_groups=1, num_warps_per_group=32 => WG=1024
    // (32 sub-groups), grid = num_experts BIG blocks. The single GridBarrier is over only
    // ~num_experts co-resident WGs (cheap), and the per-expert finish-counter completes
    // among those co-resident blocks -- so NO separate counting kernel is needed.
    const int num_device_sms = static_cast<int>(queue.get_device().get_info<sycl::info::device::max_compute_units>());
    int num_warp_groups = (num_experts + num_device_sms - 1) / std::max(num_device_sms, 1);
    if (num_warp_groups < 1) num_warp_groups = 1;
    int num_warps_per_group = 32 / num_warp_groups;
    if (num_warps_per_group < 1) num_warps_per_group = 1;
    const int num_warps = num_warp_groups * num_warps_per_group;           // sub-groups per WG
    const int num_sms = (num_experts + num_warp_groups - 1) / num_warp_groups;  // grid (work-groups)
    const int wg_size = num_warps * 32;
    TORCH_CHECK(num_topk + 1 <= num_warps, "LL dispatch requires num_warps > num_topk");
    TORCH_CHECK(num_warps_per_group > 1, "LL dispatch requires num_warps_per_group > 1 (recv overlap)");
    TORCH_CHECK(num_warp_groups >= 1, "num_warp_groups must be >= 1");

    // Zero caller outputs + workspace.
    queue.memset(packed_recv_count, 0, static_cast<size_t>(num_local_experts) * sizeof(int));
    queue.memset(packed_recv_src_info, 0xFF, recv_src_elems * sizeof(int));
    queue.memset(slot_counter, 0, static_cast<size_t>(num_experts) * sizeof(int));

    // ======================= FUSED DISPATCH =======================
    {
        auto* finish_counter = reinterpret_cast<int*>(base + layout.finish_counter_offset);
        queue.memset(finish_counter, 0, static_cast<size_t>(num_experts) * sizeof(int));

<<<<<<< HEAD
        nd_launch_root_sync<LowLatencyDispatchFusedKernel>(
            queue, static_cast<size_t>(num_sms), static_cast<size_t>(wg_size),
=======
    // ---- Kernel 1: cast + put every token to its top-k experts (no counting). ----
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<LLDispatchSendKernel>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(send_wgs) * wg_size), sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                auto group = item.get_group();
                auto sg = item.get_sub_group();
                const int sm_id = static_cast<int>(item.get_group_linear_id());
                const int warp_id = static_cast<int>(sg.get_group_linear_id());
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sg_size = static_cast<int>(sg.get_local_range()[0]);
                // No counter warp in the split: ALL warps cast cooperatively.
                const int caster_tid = warp_id * 32 + lane;
                const int num_caster_threads = num_warps * 32;

                // Block 0 cleans the opposite-parity rdma_recv_count (CUDA next_clean).
                if (sm_id == 0 && warp_id == 0) {
                    const int clean_parity = 1 - dispatch_parity;
                    for (int i = lane; i < num_recv_channels; i += sg_size)
                        uc_store<long>(&rdma_recv_count[i * 2 + clean_parity], 0L);
                }

                // Token loop striped across the send grid. Each iteration is a whole-WG
                // cooperative cast followed by a whole-WG barrier; warp `warp_id < num_topk`
                // then puts the finished message to topk_idx[token, warp_id]'s expert slot.
                for (int t = sm_id; t < num_tokens; t += send_wgs) {
                    uint8_t* msg = rdma_x + static_cast<size_t>(t) * msg_bytes;
                    int* hdr = reinterpret_cast<int*>(msg);
                    const auto* src_bf16 = reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(
                        static_cast<const uint8_t*>(x) + static_cast<size_t>(t) * hidden_bytes);
                    const int dst_expert = (warp_id < num_topk)
                        ? static_cast<int>(topk_idx[static_cast<size_t>(t) * num_topk + warp_id]) : -1;
                    if (caster_tid == 0) hdr[0] = t;  // CUDA rdma_x_src_idx
                    if (use_fp8) {
                        cast_token_fp8_strided(msg + sizeof(int) * 4,
                                               reinterpret_cast<float*>(msg + scales_off),
                                               src_bf16, num_scales, round_scale, sg, lane, sg_size,
                                               warp_id, num_warps);
                    } else {
                        coop_copy_bytes(msg + sizeof(int) * 4, reinterpret_cast<const uint8_t*>(src_bf16),
                                        payload_bytes, caster_tid, num_caster_threads);
                    }
                    // Device-scope release: make the cast bytes NIC-visible (HBM/L2 via PCIe
                    // P2P) before the doorbell. WG-scope group_barrier alone does not (F2).
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                    sycl::group_barrier(group);
                    if (dst_expert >= 0 && dst_expert < num_experts) {
                        const int dst_rank = dst_expert / num_local_experts;
                        const int le = dst_expert % num_local_experts;
                        int slot = 0;
                        if (lane == 0) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space> sc(slot_counter[dst_expert]);
                            slot = sc.fetch_add(1);
                        }
                        slot = sycl::group_broadcast(sg, slot, 0);
                        const size_t dst_slot =
                            (static_cast<size_t>(le) * num_ranks + rank) * num_max_dispatch_tokens_per_rank + slot;
                        uint8_t* dst = rdma_recv_x + dst_slot * msg_bytes;
                        if (!ll_rank_masked(mask_buffer_ptr, dst_rank)) {
                            if (dst_rank == rank)
                                coop_copy_bytes_store_uc(dst, msg, used_bytes, lane, sg_size);
                            else
                                ishmemx_putmem_nbi_subgroup(dst, msg, used_bytes, dst_rank,
                                                        static_cast<unsigned int>(le), true, sg, false);
                        }
                    }
                }
            });
    });

    // ---- Kernel 2: post count flags (phase A) then poll + copy messages (phase B). ----
    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> shared_recv_cnt(sycl::range<1>(std::max(num_warp_groups, 1)), cgh);
        sycl::local_accessor<int, 1> shared_recv_begin(sycl::range<1>(std::max(num_warp_groups, 1)), cgh);
        cgh.parallel_for<LLDispatchRecvKernel>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_sms) * wg_size), sycl::range<1>(wg_size)),
>>>>>>> 0bc30b5ce426df96303242f64d2664b2e06b839d
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                auto group = item.get_group();
                auto sg = item.get_sub_group();
                const int sm_id = static_cast<int>(item.get_group_linear_id());
                const int num_sms_total = static_cast<int>(item.get_group_range(0));
                const int warp_id = static_cast<int>(sg.get_group_linear_id());
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sg_size = static_cast<int>(sg.get_local_range()[0]);
                const int warp_group_id = warp_id / num_warps_per_group;
                const int sub_warp_id = warp_id % num_warps_per_group;
                const int responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

<<<<<<< HEAD
                // Private "shared" values (num_warp_groups==1 => same WG).
                int local_send_count = 0;
                int local_recv_count = 0;
                int local_recv_begin = 0;

                // SEND: caster warps + counter warp
                if (warp_id < num_warps - 1) {
                    const int caster_tid = warp_id * 32 + lane;
                    const int num_caster_threads = (num_warps - 1) * 32;
                    if (sm_id == 0 && warp_id == 0) {
                        const int clean_parity = 1 - dispatch_parity;
                        for (int i = lane; i < num_recv_channels; i += sg_size)
                            uc_store<long>(&rdma_recv_count[i * 2 + clean_parity], 0L);
                    }
                    NamedBarrier caster_barrier;
                    caster_barrier.init(num_warps - 1);
                    for (int t = sm_id; t < num_tokens; t += num_sms_total) {
                        uint8_t* msg = rdma_x + static_cast<size_t>(t) * msg_bytes;
                        int* hdr = reinterpret_cast<int*>(msg);
                        const auto* src_bf16 = reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(
                            static_cast<const uint8_t*>(x) + static_cast<size_t>(t) * hidden_bytes);
                        const int dst_expert = (warp_id < num_topk)
                            ? static_cast<int>(topk_idx[static_cast<size_t>(t) * num_topk + warp_id]) : -1;
                        if (caster_tid == 0) hdr[0] = t;
                        if (use_fp8) {
                            cast_token_fp8_strided(msg + sizeof(int) * 4,
                                                   reinterpret_cast<float*>(msg + scales_off),
                                                   src_bf16, num_scales, round_scale, sg, lane, sg_size,
                                                   warp_id, num_warps - 1);
                        } else {
                            coop_copy_bytes(msg + sizeof(int) * 4, reinterpret_cast<const uint8_t*>(src_bf16),
                                            payload_bytes, caster_tid, num_caster_threads);
                        }
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                        caster_barrier.sync(kNamedBarrierGlobalFence);
                        if (dst_expert >= 0 && dst_expert < num_experts) {
                            const int dst_rank = dst_expert / num_local_experts;
                            const int le = dst_expert % num_local_experts;
                            int slot = 0;
                            if (lane == 0) {
                                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space> sc(slot_counter[dst_expert]);
                                slot = sc.fetch_add(1);
                            }
                            slot = sycl::group_broadcast(sg, slot, 0);
                            const size_t dst_slot =
                                (static_cast<size_t>(le) * num_ranks + rank) * num_max_dispatch_tokens_per_rank + slot;
                            uint8_t* dst = rdma_recv_x + dst_slot * msg_bytes;
                            if (!ll_rank_masked(mask_buffer_ptr, dst_rank)) {
                                if (dst_rank == rank)
                                    coop_copy_bytes_store_uc(dst, msg, used_bytes, lane, sg_size);
                                else
                                    ishmemx_putmem_nbi_warp(dst, msg, used_bytes, dst_rank,
                                                            static_cast<unsigned int>(le), true, sg, false);
                            }
                            sycl::group_barrier(sg);
                            if (lane == 0) {
                                sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                                 sycl::access::address_space::global_space> fc(finish_counter[dst_expert]);
                                fc.fetch_add(1);
=======
                // ---- Phase A: count tokens this rank sent to responsible_expert_idx and
                // post the count flag (-count-1). The kernel boundary already guarantees all
                // payload puts are posted; quiet_qp drains QP=le (flushing deferred doorbells)
                // and the RC-ordered AMO lands after the payloads. Only warp 0 counts.
                if (responsible_expert_idx < num_experts && warp_id == 0) {
                    const int dst_rank = responsible_expert_idx / num_local_experts;
                    const int le = responsible_expert_idx % num_local_experts;
                    int cnt = 0;
                    for (int i = lane; i < num_tokens * num_topk; i += sg_size)
                        if (static_cast<int>(topk_idx[i]) == responsible_expert_idx) ++cnt;
                    cnt = sycl::reduce_over_group(sg, cnt, sycl::plus<int>());
                    if (lane == 0) {
                        const int slot = (le * num_ranks + rank) * 2 + dispatch_parity;
                        if (!ll_rank_masked(mask_buffer_ptr, dst_rank)) {
                            if (dst_rank == rank) {
                                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                                uc_store<long>(&rdma_recv_count[slot], static_cast<long>(-cnt - 1));
                            } else {
                                ishmemx_fence_qp(dst_rank, static_cast<unsigned int>(le));
                                ll_sender_flush(flag_sender_fence);
                                ishmemx_long_atomic_add_qp(&rdma_recv_count[slot], static_cast<long>(-cnt - 1),
                                                           dst_rank, static_cast<unsigned int>(le));
>>>>>>> 0bc30b5ce426df96303242f64d2664b2e06b839d
                            }
                        }
                    }
                } else {
                    if (sm_id == 0) {
                        for (int i = lane; i < num_experts; i += sg_size) {
                            sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                             sycl::access::address_space::global_space> fc(finish_counter[i]);
                            fc.fetch_add(kFinishedSumTag);
                        }
                    }
                    int expert_count[32] = {0};
                    const int expert_begin_idx = sm_id * num_warp_groups;
                    const int expert_end_idx = sycl::min(expert_begin_idx + num_warp_groups, num_experts);
                    for (int i = lane; i < num_tokens * num_topk; i += sg_size) {
                        int idx = static_cast<int>(topk_idx[i]);
                        if (idx >= expert_begin_idx && idx < expert_end_idx)
                            expert_count[idx - expert_begin_idx]++;
                    }
                    for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
                        const int sum = sycl::reduce_over_group(sg, expert_count[i - expert_begin_idx], sycl::plus<int>());
                        if (lane == 0) {
                            local_send_count = sum;
                            sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                             sycl::access::address_space::global_space> fc(finish_counter[i]);
                            fc.fetch_add(kFinishedSumTag - sum);
                        }
                    }
                }
                sycl::group_barrier(group);

                if (responsible_expert_idx < num_experts && sub_warp_id == 0 && lane == 0) {
                    const int dst_rank = responsible_expert_idx / num_local_experts;
                    const int le = responsible_expert_idx % num_local_experts;
                    uint64_t spins = 0;
                    while (true) {
                        const int val = uc_load<int>(&finish_counter[responsible_expert_idx]);
                        if (val == kFinishedSumTag * 2) break;
                        if (++spins >= poll_cap) break;
                        visa_spin_hint();
                    }
                    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                    const int slot = (le * num_ranks + rank) * 2 + dispatch_parity;
                    if (!ll_rank_masked(mask_buffer_ptr, dst_rank)) {
                        if (dst_rank == rank) {
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                            uc_store<long>(&rdma_recv_count[slot], static_cast<long>(-local_send_count - 1));
                        } else {
                            ishmemx_quiet_qp(dst_rank, static_cast<unsigned int>(le));
                            ll_sender_flush(flag_sender_fence);
                            ishmemx_long_atomic_add_qp(&rdma_recv_count[slot], static_cast<long>(-local_send_count - 1),
                                                       dst_rank, static_cast<unsigned int>(le));
                        }
                    }
                    slot_counter[responsible_expert_idx] = 0;
                    finish_counter[responsible_expert_idx] = 0;
                    if (dst_rank == 0) packed_recv_count[le] = 0;
                }

                {
                    auto root = item.ext_oneapi_get_root_group();
                    sycl::group_barrier(root);
                }

                if (responsible_expert_idx < num_experts) {
                    const int src_rank = responsible_expert_idx / num_local_experts;
                    const int local_expert = responsible_expert_idx % num_local_experts;
                    if (sub_warp_id == 1 && lane == 0) {
                        int count = 0;
                        if (!ll_rank_masked(mask_buffer_ptr, src_rank)) {
                            const int cslot = (local_expert * num_ranks + src_rank) * 2 + dispatch_parity;
                            uint64_t spins = 0;
                            long raw = 0;
                            while (true) {
                                raw = ll_read_flag64(&rdma_recv_count[cslot], flag_lsc_mode);
                                if (raw != 0) break;
                                if (++spins >= poll_cap) break;
                                visa_spin_hint();
                            }
                            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                            count = (raw == 0) ? 0 : static_cast<int>(-raw - 1);
                        }
                        int begin;
                        {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space> rc(packed_recv_count[local_expert]);
                            begin = (count > 0) ? rc.fetch_add(count) : rc.load();
                        }
                        packed_recv_layout_range[local_expert * num_ranks + src_rank] =
                            static_cast<int64_t>(pack_range(count, begin));
                        if (cumulative_local_expert_recv_stats != nullptr && count > 0) {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space> st(cumulative_local_expert_recv_stats[local_expert]);
                            st.fetch_add(count);
                        }
                        local_recv_count = count;
                        local_recv_begin = begin;
                    }
                    sycl::group_barrier(group);
                    const int count = local_recv_count;
                    const int begin = local_recv_begin;
                    if (count > 0) {
                        if (flag_recv_acq != 0 && lane == 0) ll_recv_acquire(flag_recv_acq);
                        const size_t src_base = (static_cast<size_t>(local_expert) * num_ranks + src_rank) * num_max_dispatch_tokens_per_rank;
                        const size_t dst_base = static_cast<size_t>(local_expert) * num_ranks * num_max_dispatch_tokens_per_rank + begin;
                        for (int i = sub_warp_id; i < count; i += num_warps_per_group) {
                            const uint8_t* rmsg = rdma_recv_x + (src_base + i) * msg_bytes;
                            const size_t dst_row = dst_base + i;
                            if (lane == 0) {
                                const int src_idx = recv_uncached ? uc_load(reinterpret_cast<const int*>(rmsg))
                                                                  : *reinterpret_cast<const int*>(rmsg);
                                packed_recv_src_info[dst_row] = src_idx;
                            }
                            const uint8_t* rpayload = rmsg + sizeof(int) * 4;
                            if (recv_uncached)
                                coop_copy_bytes_uc(dst_data + dst_row * payload_bytes, rpayload, payload_bytes, lane, sg_size);
                            else
                                coop_copy_bytes(dst_data + dst_row * payload_bytes, rpayload, payload_bytes, lane, sg_size);
                            if (use_fp8) {
                                const float* rscales = reinterpret_cast<const float*>(rmsg + scales_off);
                                for (int s = lane; s < num_scales; s += sg_size) {
                                    const float scale_inv = recv_uncached ? uc_load(&rscales[s]) : rscales[s];
                                    if (use_ue8m0) {
                                        const int pack_idx = s / 4;
                                        const int pack_shift = (s % 4) * 8;
                                        const int32_t sb = static_cast<int32_t>(ue8m0_from_float(scale_inv)) << pack_shift;
                                        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space> sp(dst_scale_int[dst_row * dst_scale_stride + pack_idx]);
                                        sp.fetch_or(sb);
                                    } else {
                                        dst_scale_float[dst_row * num_scales + s] = scale_inv;
                                    }
                                }
                            }
                        }
                    }
                }
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
    (void)send_count;
    (void)send_elems;
    (void)max_put;
    const uint64_t poll_cap = ll_poll_cap();
    const int flag_lsc_mode = ll_flag_lsc_mode();
    const int flag_sender_fence = ll_flag_sender_fence();
    const int flag_recv_acq = ll_flag_recv_acq();

    // --- FUSED COMBINE: NamedBarrier + root_group ---
    {
        const int num_device_sms = static_cast<int>(queue.get_device().get_info<sycl::info::device::max_compute_units>());
        int num_warp_groups = (num_experts + num_device_sms - 1) / std::max(num_device_sms, 1);
        if (num_warp_groups < 1) num_warp_groups = 1;
        int num_warps_per_group = 32 / num_warp_groups;
        if (num_warps_per_group < 1) num_warps_per_group = 1;
        const int num_warps = num_warp_groups * num_warps_per_group;
        const int num_sms = (num_experts + num_warp_groups - 1) / num_warp_groups;
        const int wg_size = num_warps * 32;
        TORCH_CHECK(num_warps_per_group > 1, "LL combine requires num_warps_per_group > 1");
        TORCH_CHECK(num_warp_groups >= 1, "num_warp_groups must be >= 1");

        const size_t reduce_work = static_cast<size_t>(num_combined_tokens) * hidden;
        auto* clean_flag = reinterpret_cast<int*>(base + layout.slot_counter_offset);
        queue.memset(clean_flag, 0, sizeof(int));

        nd_launch_root_sync<LowLatencyCombineFusedKernel>(
            queue, static_cast<size_t>(num_sms), static_cast<size_t>(wg_size),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                auto group = item.get_group();
                auto sg = item.get_sub_group();
                const int sm_id = static_cast<int>(item.get_group_linear_id());
                const int num_sms_total = static_cast<int>(item.get_group_range(0));
                const int warp_id = static_cast<int>(sg.get_group_linear_id());
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sg_size = static_cast<int>(sg.get_local_range()[0]);
                const int warp_group_id = warp_id / num_warps_per_group;
                const int sub_warp_id = warp_id % num_warps_per_group;
                const int responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

<<<<<<< HEAD
                if (sm_id == 0 && warp_group_id == 0 && sub_warp_id == 0) {
                    const int clean_parity = combine_parity ^ 1;
                    for (int i = lane; i < num_experts; i += sg_size)
                        uc_store<long>(&combine_flag_i[i * 2 + clean_parity], 0L);
                    sycl::group_barrier(sg);
                    if (lane == 0) {
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
=======
                    // ============================ SEND PHASE ============================
                    // Clean the OPPOSITE-parity combine_flag receive slots (CUDA next_clean),
                    // then release atomic_clean_flag (+num_experts) so flag posts wait for it.
                    if (sm_id == 0 && warp_group_id == 0 && sub_warp_id == 0) {
                        const int clean_parity = combine_parity ^ 1;
                        for (int i = lane; i < num_experts; i += sg_size)
                            uc_store<long>(&combine_flag_i[i * 2 + clean_parity], 0L);
                        sycl::group_barrier(sg);
                        if (lane == 0) {
                            // clean_flag is an intra-GPU cross-WG gate (CUDA atomic_clean_flag,
                            // device-scope .gpu). Release orders the zeroed opposite-parity
                            // combine_flag_i slots before the counter bump (F5).
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                            sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space> cf(clean_flag[0]);
                            cf.fetch_add(num_experts);
                        }
                    }

                    // Issue per-token IBGDA sends for this responsible expert. Each sub-warp
                    // owns a stride of this expert's tokens; it copies the token's hidden row
                    // into per-token symmetric staging (send_data at the token's own slot),
                    // then warp-puts it to the destination's ORIGINAL token slot src_idx.
                    if (responsible_expert_idx < num_experts) {
                        const int dst_rank = responsible_expert_idx / num_local_experts;
                        const int le = responsible_expert_idx % num_local_experts;
                        const int global_expert = rank * num_local_experts + le;
                        int count = 0, begin = 0;
                        unpack_range(layout_range[le * num_ranks + dst_rank], count, begin);
                        const bool masked = ll_rank_masked(mask_buffer_ptr, dst_rank);
                        const size_t local_stride =
                            static_cast<size_t>(num_ranks) * num_max_dispatch_tokens_per_rank;
                        const auto* local_x =
                            static_cast<const uint8_t*>(x) + static_cast<size_t>(le) * local_stride * hidden_bytes;
                        const int* local_src_info = src_info + static_cast<size_t>(le) * local_stride;

                        if (!masked) {
                            for (int token_idx = begin + sub_warp_id; token_idx < begin + count;
                                 token_idx += num_warps_per_group) {
                                int src_idx = (lane == 0) ? local_src_info[token_idx] : 0;
                                src_idx = sycl::group_broadcast(sg, src_idx, 0);
                                if (src_idx < 0 || src_idx >= num_max_dispatch_tokens_per_rank) continue;
                                const size_t dst_slot =
                                    static_cast<size_t>(global_expert) * num_max_dispatch_tokens_per_rank + src_idx;
                                uint8_t* dst = combine_data + dst_slot * hidden_bytes;
                                const uint8_t* srcrow = local_x + static_cast<size_t>(token_idx) * hidden_bytes;
                                if (dst_rank == rank) {
                                    // Self: write directly into local combine_data (== CUDA p2p copy).
                                    coop_copy_bytes_store_uc(dst, srcrow, hidden_bytes, lane, sg_size);
                                } else {
                                    // Remote: stage into symmetric send_data, flush, warp-put.
                                    uint8_t* stage = send_data +
                                        static_cast<size_t>(le * local_stride + token_idx) * hidden_bytes;
                                    coop_copy_bytes(stage, srcrow, hidden_bytes, lane, sg_size);
                                    sycl::group_barrier(sg);
                                    // NIC DMAs GPU HBM -> device-scope release suffices for the
                                    // put's source visibility; system scope unnecessary (F4).
                                    // (CUDA: __syncwarp + tma_store_wait, no system fence.)
                                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                                    ishmemx_putmem_nbi_subgroup(dst, stage, hidden_bytes, dst_rank,
                                                            static_cast<unsigned int>(le), true, sg, false);
                                }
                            }
                        }
                    }

                    // Warp-group barrier (== CUDA bar.sync warp_group+1). With
                    // num_warp_groups==1 this is the whole-WG barrier; reached
                    // unconditionally by every warp so no divergent-barrier deadlock.
                    sycl::group_barrier(group);

                    // Flag post: sub-warp 1 lane 0. Wait atomic_clean_flag>0 (next_clean done),
                    // then post the arrival flag (+1) on QP=le. Self => uc_store; remote =>
                    // quiet + atomic-add on the payload's QP (RC in-order: flag after payload).
                    if (responsible_expert_idx < num_experts && sub_warp_id == 1 && lane == 0) {
                        const int dst_rank = responsible_expert_idx / num_local_experts;
                        const int le = responsible_expert_idx % num_local_experts;
                        const int global_expert = rank * num_local_experts + le;
                        {
                            uint64_t spins = 0;
                            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space> cf(clean_flag[0]);
                            while (cf.load() == 0) {
                                if (++spins >= poll_cap) break;
                                visa_spin_hint();
                            }
                            // Device-scope acquire pairs with the clean_flag device release (F5).
                            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
                        }
                        if (!ll_rank_masked(mask_buffer_ptr, dst_rank)) {
                            const int slot = global_expert * 2 + combine_parity;
                            if (dst_rank == rank) {
                                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                                uc_store<long>(&combine_flag_i[slot], 1L);
                            } else {
                                ishmemx_fence_qp(dst_rank, static_cast<unsigned int>(le));
                                ll_sender_flush(flag_sender_fence);
                                ishmemx_long_atomic_add_qp(&combine_flag_i[slot], 1L, dst_rank,
                                                           static_cast<unsigned int>(le));
                            }
                        }
>>>>>>> 0bc30b5ce426df96303242f64d2664b2e06b839d
                        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> cf(clean_flag[0]);
                        cf.fetch_add(num_experts);
                    }
                }

                if (responsible_expert_idx < num_experts) {
                    const int dst_rank = responsible_expert_idx / num_local_experts;
                    const int le = responsible_expert_idx % num_local_experts;
                    const int global_expert = rank * num_local_experts + le;
                    int count = 0, begin = 0;
                    unpack_range(layout_range[le * num_ranks + dst_rank], count, begin);
                    const bool masked = ll_rank_masked(mask_buffer_ptr, dst_rank);
                    const size_t local_stride = static_cast<size_t>(num_ranks) * num_max_dispatch_tokens_per_rank;
                    const auto* local_x = static_cast<const uint8_t*>(x) + static_cast<size_t>(le) * local_stride * hidden_bytes;
                    const int* local_src_info = src_info + static_cast<size_t>(le) * local_stride;
                    if (!masked) {
                        for (int token_idx = begin + sub_warp_id; token_idx < begin + count;
                             token_idx += num_warps_per_group) {
                            int src_idx = (lane == 0) ? local_src_info[token_idx] : 0;
                            src_idx = sycl::group_broadcast(sg, src_idx, 0);
                            if (src_idx < 0 || src_idx >= num_max_dispatch_tokens_per_rank) continue;
                            const size_t dst_slot = static_cast<size_t>(global_expert) * num_max_dispatch_tokens_per_rank + src_idx;
                            uint8_t* dst = combine_data + dst_slot * hidden_bytes;
                            const uint8_t* srcrow = local_x + static_cast<size_t>(token_idx) * hidden_bytes;
                            if (dst_rank == rank) {
                                coop_copy_bytes_store_uc(dst, srcrow, hidden_bytes, lane, sg_size);
                            } else {
                                uint8_t* stage = send_data + static_cast<size_t>(le * local_stride + token_idx) * hidden_bytes;
                                coop_copy_bytes(stage, srcrow, hidden_bytes, lane, sg_size);
                                sycl::group_barrier(sg);
                                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                                ishmemx_putmem_nbi_warp(dst, stage, hidden_bytes, dst_rank,
                                                        static_cast<unsigned int>(le), true, sg, false);
                            }
                        }
                    }
                }

                {
                    NamedBarrier wg_barrier;
                    wg_barrier.init(num_warps_per_group);
                    wg_barrier.sync(kNamedBarrierGlobalFence);
                }

                if (responsible_expert_idx < num_experts && sub_warp_id == 1 && lane == 0) {
                    const int dst_rank = responsible_expert_idx / num_local_experts;
                    const int le = responsible_expert_idx % num_local_experts;
                    const int global_expert = rank * num_local_experts + le;
                    {
                        uint64_t spins = 0;
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space> cf(clean_flag[0]);
                        while (cf.load() == 0) { if (++spins >= poll_cap) break; visa_spin_hint(); }
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
                    }
                    if (!ll_rank_masked(mask_buffer_ptr, dst_rank)) {
                        const int slot = global_expert * 2 + combine_parity;
                        if (dst_rank == rank) {
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                            uc_store<long>(&combine_flag_i[slot], 1L);
                        } else {
                            ishmemx_quiet_qp(dst_rank, static_cast<unsigned int>(le));
                            ll_sender_flush(flag_sender_fence);
                            ishmemx_long_atomic_add_qp(&combine_flag_i[slot], 1L, dst_rank, static_cast<unsigned int>(le));
                        }
                    }
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> cf(clean_flag[0]);
                    cf.fetch_add(-1);
                }

                if (responsible_expert_idx < num_experts && sub_warp_id == 0 && lane == 0) {
                    const int src_rank = responsible_expert_idx / num_local_experts;
                    if (src_rank != rank && !ll_rank_masked(mask_buffer_ptr, src_rank)) {
                        const int slot = responsible_expert_idx * 2 + combine_parity;
                        uint64_t spins = 0;
                        while (ll_read_flag64(&combine_flag_i[slot], flag_lsc_mode) == 0) {
                            if (++spins >= poll_cap) break;
                            visa_spin_hint();
                        }
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                    }
                }

                { auto root = item.ext_oneapi_get_root_group(); sycl::group_barrier(root); }

                const bool recv_uncached = (flag_recv_acq == 0);
                if (!recv_uncached) ll_recv_acquire(flag_recv_acq);

                const size_t global_tid = static_cast<size_t>(sm_id) * wg_size + static_cast<size_t>(item.get_local_linear_id());
                const size_t global_size_total = static_cast<size_t>(num_sms_total) * wg_size;
                auto* out = static_cast<sycl::ext::oneapi::bfloat16*>(combined_x);

                for (size_t idx = global_tid; idx < reduce_work; idx += global_size_total) {
                    const int token_idx = static_cast<int>(idx / hidden);
                    const int h = static_cast<int>(idx % hidden);
                    float acc = 0.0f;
                    for (int k = 0; k < num_topk; ++k) {
                        const int expert = static_cast<int>(topk_idx[token_idx * num_topk + k]);
                        if (expert < 0 || expert >= num_experts) continue;
                        if (ll_rank_masked(mask_buffer_ptr, expert / num_local_experts)) continue;
                        const auto* value = reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(
                            combine_data + (static_cast<size_t>(expert) * num_max_dispatch_tokens_per_rank + token_idx) * hidden_bytes);
                        acc += (recv_uncached ? static_cast<float>(uc_load(value + h)) : static_cast<float>(value[h]))
                               * topk_weights[token_idx * num_topk + k];
                    }
                    out[static_cast<size_t>(token_idx) * hidden + h] = bf16_from_float(acc);
                }
            });
        (void)combine_wait_recv_cost_stats;
    }
#endif
}

}  // namespace internode_ll
}  // namespace deep_ep
