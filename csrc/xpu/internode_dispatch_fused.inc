// ============================================================================
// FUSED, WARP-SPECIALIZED INTERNODE **DISPATCH** (CUDA-faithful port)
// ----------------------------------------------------------------------------
// This is a 1:1 port of the single fused CUDA kernel
//     csrc/cuda_kernels/internode.cu:447  `dispatch<...>`
// It replaces the phase-split micro-kernel decomposition that the rest of
// csrc/xpu/internode.cpp still implements.
//
// Structure mirrored from CUDA (all line numbers refer to internode.cu):
//   * grid   = num_channels * 2 work-groups; even sm_id = "forwarder" SM,
//              odd sm_id = "sender/receiver" SM                       (:485,1305)
//   * block  = (DEEP_EP_FUSED_SENDER_WARPS + 1 + NUM_MAX_NVL_PEERS) * 32
//            = (7 + 1 + 8) * 32 = 512 work-items = 16 sub-groups      (:446,1305)
//   * roles  = kRDMASender / kRDMASenderCoordinator / kRDMAAndNVLForwarder /
//              kForwarderCoordinator / kNVLReceivers                  (:482-509)
//
// Warp-subset synchronization -- CUDA `barrier.sync <id>, <count>` becomes the
// SPIR-V NamedBarrier (`cl_khr_subgroup_named_barrier`). `count` is expressed in
// SUB-GROUPS, not work-items:
//
//   CUDA :563  sync_rdma_sender_smem : barrier.sync 0, (7 + 1) * 32 = 256 WI
//   SYCL       named_barrier_init(DEEP_EP_FUSED_SENDER_WARPS + 1) =  8 sub-groups
//              participants: warps 0..6 (kRDMASender) + warp 7 (kRDMASenderCoordinator)
//
//   CUDA :580  sync_forwarder_smem   : barrier.sync 1, (8 + 1) * 32 = 288 WI
//   SYCL       named_barrier_init(NUM_MAX_NVL_PEERS + 1)             =  9 sub-groups
//              participants: warps 0..7 (kRDMAAndNVLForwarder) + warp 8
//              (kForwarderCoordinator with target_rank == 0)
//
// IGC rule (see csrc/xpu/named_barrier_usage.md): the handle returned by
// named_barrier_init() must be a plain SSA local OF THE FUNCTION that calls
// work_group_named_barrier() -- no members, args, PHIs, GEPs or SLM round-trips.
// Both handles are therefore materialized unconditionally at the very top of the
// kernel lambda and used only through literal if/else. This exact shape (512 WI /
// 16 sub-groups / counts 8 and 9 / iSHMEM device calls inside participants) is
// validated by csrc/xpu/tools/test_nbarrier_dispatch_shape.cpp.
//
// Deliberate deviations from CUDA, and why:
//   D1. TMA (`tma_load_1d` / `tma_store_1d` / mbarrier) has no Xe equivalent.
//       Replaced by sub-group-cooperative 16B (int4) LSC copies, which is what
//       the CUDA code did before TMA was introduced.
//   D2. `NUM_MAX_NVL_PEERS` is a hard 8 in CUDA. On XPU num_nvl_ranks may be
//       smaller (2 in the 2-node x 2-GPU simulator). The work-group keeps the
//       full 16 sub-groups so the NamedBarrier arrive-counts stay compile-time
//       constants (8 / 9), and forwarder/receiver warps whose `target_rank`
//       lands >= num_nvl_ranks do no work. Because `(warp_id + channel_id) % 8`
//       is a bijection on [0,8), exactly num_nvl_ranks warps stay active.
//       Idle FORWARDER warps still enter sync_forwarder_smem() exactly once
//       (they must, or the 9-way barrier deadlocks); idle RECEIVER warps enter
//       no barrier at all and return.
//   D3. CUDA's per-lane `acquire_lock(rdma_send_channel_lock + lane_id)` lets
//       every lane of a warp hold a DIFFERENT SLM lock simultaneously. That is
//       safe under Volta+ independent thread scheduling but can deadlock on Xe,
//       where a sub-group is a single lock-stepped EU thread: a lane that has
//       taken lock[a] cannot reach its release while a sibling lane spins on
//       lock[b]. The critical section is therefore serialized over
//       dst_rdma_rank, so a sub-group holds at most ONE lock at a time and no
//       cycle can form. Semantics (window/tail release order) are unchanged.
//   D4. `translate_dst_rdma_rank<false>()` returns the bare rdma_rank in CUDA
//       because NVSHMEM is initialized on a same-gpu-index team. iSHMEM has a
//       single flat PE space, so dst_pe = dst_rdma_rank * num_nvl_ranks +
//       nvl_rank (identical to the rest of csrc/xpu/internode*.cpp).
//   D5. Cache/coherence: NIC- and P2P-delivered bytes are not coherent with the
//       Xe cache hierarchy. Queue counters (`rdma_channel_tail/head`,
//       `nvl_channel_tail/head`, `nvl_channel_prefix_*`, `rdma_channel_meta`)
//       are read with uc_load()/written with uc_store(), and payload reads are
//       preceded by lsc_fence_sysacq() and issued through ld_nc_global_v
//       (`lsc_load.ugm.uc.ca`).
//   D6. CUDA relies on `ibgda_get_state()->num_rc_per_pe` for its QP fan-out.
//       There is no iSHMEM equivalent, so the QP index is
//       `channel_id % qps_per_pe` (payload/tail) and
//       `(channel_id + num_channels) % qps_per_pe` (head credit), with
//       qps_per_pe read from ISHMEM_IBGDA_QPS_PER_PE.
// ============================================================================

// STATUS / REMAINING PREREQUISITES (measured, not assumed)
// ----------------------------------------------------------------------------
// Validated so far on Arc Pro B60 / 2 nodes x 2 GPUs:
//   * The kernel COMPILES and device-links against libishmem.a with no
//     "Stack call has been detected" warning -> NamedBarrier + iSHMEM coexist
//     inside a real fused DeepEP kernel (retracting the old blocker).
//   * With DEEP_EP_INTERNODE_FUSED=1 it LAUNCHES AND COMPLETES: no vISA
//     "NBarrierCnt" failure, no hang on either named barrier, no DEVICE_LOST.
//
// Two prerequisites remain before it can be the default:
//
//   P1. `notify_dispatch` is not ported. In CUDA (internode.cu:93) a SEPARATE
//       kernel performs the cross-rank count exchange and produces
//       rdma_channel_prefix_matrix / gbl_channel_prefix_matrix /
//       recv_rdma_rank_prefix_sum / recv_gbl_rank_prefix_sum, which
//       `dispatch` then consumes READ-ONLY (they are `const int*` there).
//       The XPU port never implemented it: those four tensors are allocated as
//       torch::zeros() by deep_ep_xpu.cpp:1537-1545 and are FILLED INSIDE the
//       phase-split dispatch (csrc/xpu/internode.cpp:1640/1671/1744/1762).
//       Running the fused kernel therefore feeds it all-zero routing metadata,
//       which surfaces as `recv_gbl_rank_prefix_sum[-1] == 0` in
//       tests/test_internode.py:325.
//
//   P2. `combine` must be fused in the same change. dispatch and combine are
//       coupled through send_rdma_head / send_nvl_head: this kernel writes them
//       with the CUDA meaning (send_nvl_head indexed by RDMA-queue slot, CUDA
//       internode.cu:983-988; send_rdma_head = the RDMA ring tail, :657-658)
//       while the phase-split combine at csrc/xpu/internode.cpp:2261/2863
//       consumes the legacy XPU meaning (destination-local receive position,
//       produced at csrc/xpu/internode.cpp:1890/1911). A source rank cannot
//       derive the legacy value locally -- it needs the DESTINATION's
//       recv_gbl_channel_prefix_matrix row -- which is exactly why the legacy
//       path carries the extra count push-back kernels (F-K9/F-K10).
//
// Neither prerequisite is a synchronization limitation: both named barriers
// behave exactly as the CUDA barrier.sync sites they replace.
// ----------------------------------------------------------------------------

// ---- csrc/cuda_kernels/buffer.cuh ports (device-side symmetric slicing) ----

template <typename dtype_t>
struct FusedAsymBuffer {
    uint8_t* ptr = nullptr;
    int64_t num_bytes = 0;
    int64_t total_bytes = 0;

    FusedAsymBuffer() = default;

    FusedAsymBuffer(void*& gbl_ptr, int num_elems, int num_ranks, int sm_id, int num_sms, int offset) {
        num_bytes = static_cast<int64_t>(num_elems) * static_cast<int64_t>(sizeof(dtype_t));
        const int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms;
        ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id + num_bytes * offset;
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    // Mirrors CUDA `advance_also`: keeps the *other* rank's cursor in lock-step.
    FusedAsymBuffer advance_also(void*& gbl_ptr) const {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    dtype_t* buffer(int idx = 0) const { return reinterpret_cast<dtype_t*>(ptr + num_bytes * idx); }
};

template <typename dtype_t, bool kDecoupled = true>
struct FusedSymBuffer {
    uint8_t* send_ptr = nullptr;
    uint8_t* recv_ptr = nullptr;
    int64_t num_bytes = 0;
    int64_t total_bytes = 0;

    FusedSymBuffer() = default;

    FusedSymBuffer(void*& gbl_ptr, int num_elems, int num_ranks, int sm_id, int num_sms) {
        num_bytes = static_cast<int64_t>(num_elems) * static_cast<int64_t>(sizeof(dtype_t));
        const int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms * (static_cast<int>(kDecoupled) + 1);
        send_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id;
        recv_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * (sm_id + num_sms);
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    dtype_t* send_buffer(int idx = 0) const { return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx); }
    dtype_t* recv_buffer(int idx = 0) const { return reinterpret_cast<dtype_t*>(recv_ptr + num_bytes * idx); }
    // Non-decoupled (head/tail counters): single region, `send_ptr` is the base.
    dtype_t* buffer(int idx = 0) const { return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx); }
};

// CUDA internode.cu:42
inline int fused_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk) {
    const size_t raw = static_cast<size_t>(hidden_int4) * sizeof(int4_t) + sizeof(SourceMeta) +
                       static_cast<size_t>(num_scales) * sizeof(float) + static_cast<size_t>(num_topk) * sizeof(int) +
                       static_cast<size_t>(num_topk) * sizeof(float);
    return static_cast<int>(align_offset(raw, sizeof(int4_t)));
}

// Number of QPs iSHMEM actually created per PE (ISHMEM_IBGDA_QPS_PER_PE, rounded
// down to a power of two and clamped to [1,16] by iSHMEM itself).
inline unsigned fused_qps_per_pe() {
    const char* e = std::getenv("ISHMEM_IBGDA_QPS_PER_PE");
    int v = e != nullptr ? std::atoi(e) : 1;
    if (v < 1) v = 1;
    if (v > 16) v = 16;
    // round down to power of two
    int p = 1;
    while (p * 2 <= v) p *= 2;
    return static_cast<unsigned>(p);
}

// Bounded spin so a lost NIC completion surfaces as a diagnostic instead of a hang.
constexpr uint64_t kFusedSpinCap = 200000000ull;
constexpr int kFusedIntMax = 2147483647;

// CUDA internode.cu:446 launch bounds: (7 + 1 + 8) * 32 = 512 work-items.
#define DEEP_EP_FUSED_SENDER_WARPS 7
#define DEEP_EP_FUSED_NUM_WARPS (DEEP_EP_FUSED_SENDER_WARPS + 1 + NUM_MAX_NVL_PEERS)
#define DEEP_EP_FUSED_NUM_THREADS (DEEP_EP_FUSED_NUM_WARPS * 32)

template <int kNumRDMARanks>
class FusedDispatchKernel;

template <int kNumRDMARanks>
class FusedDispatchCleanKernel;

// ---------------------------------------------------------------------------
// Zeroing of the control planes. CUDA does this inside `notify_dispatch`
// (internode.cu:150-166) / `cached_notify` (internode.cu:1311), which are
// SEPARATE kernels there too, so keeping it out of the fused kernel is faithful.
// ---------------------------------------------------------------------------
template <int kNumRDMARanks>
void launch_fused_dispatch_clean(void* rdma_buffer_ptr,
                                 void** buffer_ptrs_gpu,
                                 int nvl_rank,
                                 int num_channels,
                                 int num_nvl_ranks,
                                 int num_bytes_per_token,
                                 int num_max_rdma_chunked_recv_tokens,
                                 int num_max_nvl_chunked_recv_tokens,
                                 sycl::queue& queue) {
    // --- RDMA control plane (host-side memset: rdma_buffer_ptr is host-known) ---
    {
        void* p = rdma_buffer_ptr;
        FusedSymBuffer<uint8_t> data(p, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, 0, num_channels);
        (void)data;
        // meta (SymBuffer<int>, decoupled => x2) + head (uint64) + tail (uint64)
        const size_t n_meta_bytes =
            static_cast<size_t>(num_nvl_ranks * 2 + 2) * sizeof(int) * kNumRDMARanks * num_channels * 2;
        const size_t n_head_bytes = sizeof(uint64_t) * kNumRDMARanks * num_channels;
        const size_t n_tail_bytes = sizeof(uint64_t) * kNumRDMARanks * num_channels;
        queue.memset(p, 0, n_meta_bytes + n_head_bytes + n_tail_bytes);
    }
    // --- NVL control plane: buffer_ptrs_gpu is DEVICE memory, so the base pointer
    //     must be resolved on the device. One tiny kernel does the zeroing. ---
    {
        const size_t nvl_x_bytes = static_cast<size_t>(num_max_nvl_chunked_recv_tokens) * num_bytes_per_token *
                                   num_nvl_ranks * num_channels;
        const size_t n_ints = static_cast<size_t>(kNumRDMARanks) * num_nvl_ranks * num_channels * 2 +  // prefix_start/end
                              static_cast<size_t>(1) * num_nvl_ranks * num_channels * 2;              // head + tail
        constexpr size_t kWG = 256;
        const size_t global = ((n_ints + kWG - 1) / kWG) * kWG;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<FusedDispatchCleanKernel<kNumRDMARanks>>(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kWG)), [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_global_linear_id();
                    if (i >= n_ints) return;
                    auto* base = static_cast<uint8_t*>(buffer_ptrs_gpu[nvl_rank]) + nvl_x_bytes;
                    uc_store(reinterpret_cast<int*>(base) + i, 0);
                });
        });
    }
}

// ---------------------------------------------------------------------------
// The fused kernel.
// ---------------------------------------------------------------------------
template <int kNumRDMARanks>
void launch_fused_dispatch(void* recv_x,
                           float* recv_x_scales,
                           topk_idx_t* recv_topk_idx,
                           float* recv_topk_weights,
                           SourceMeta* recv_src_meta,
                           const void* x,
                           const float* x_scales,
                           const topk_idx_t* topk_idx,
                           const float* topk_weights,
                           int* send_rdma_head,
                           int* send_nvl_head,
                           int* recv_rdma_channel_prefix_matrix,
                           int* recv_gbl_channel_prefix_matrix,
                           const int* rdma_channel_prefix_matrix,
                           const int* recv_rdma_rank_prefix_sum,
                           const int* gbl_channel_prefix_matrix,
                           const int* recv_gbl_rank_prefix_sum,
                           const bool* is_token_in_rank,
                           int num_tokens,
                           int hidden_int4,
                           int num_scales,
                           int num_topk,
                           int num_experts,
                           void* rdma_buffer_ptr,
                           int num_max_rdma_chunked_send_tokens,
                           int num_max_rdma_chunked_recv_tokens,
                           void** buffer_ptrs,
                           int num_max_nvl_chunked_send_tokens,
                           int num_max_nvl_chunked_recv_tokens,
                           int rank,
                           int num_ranks,
                           int num_nvl_ranks,
                           int num_channels,
                           bool kCachedMode,
                           sycl::queue& queue) {
#ifdef DEEP_EP_ENABLE_ISHMEM
    const int kNumThreads = DEEP_EP_FUSED_NUM_THREADS;

    const int num_bytes_per_token = fused_num_bytes_per_token(hidden_int4, num_scales, num_topk);
    const int hidden_bytes = hidden_int4 * static_cast<int>(sizeof(int4_t));
    const int scale_bytes = num_scales * static_cast<int>(sizeof(float));
    const unsigned qps_per_pe = fused_qps_per_pe();

    const int num_sms = num_channels * 2;

    // ---- §24.2 discriminator: OCCUPANCY PADDING (diagnostic only, default off) ----
    // `num_sms` and `num_channels` are perfectly collinear (grid = 2 * channels), so a
    // DEEP_EP_NUM_SMS sweep can never tell "too many work-groups" apart from "too many
    // channels".  `DEEP_EP_FUSED_PAD_WGS=N` appends N work-groups that do NOTHING but
    // hold their slot for `DEEP_EP_FUSED_PAD_CYCLES` GPU cycles and exit.  They touch no
    // memory and depend on nothing, so they cannot themselves hang or corrupt - they only
    // add co-residency pressure at a FIXED channel count.
    static const int pad_wgs_env = [] {
        const char* e = std::getenv("DEEP_EP_FUSED_PAD_WGS");
        return (e != nullptr && e[0] != '\0') ? std::max(std::atoi(e), 0) : 0;
    }();
    static const uint64_t pad_cycles_env = [] {
        const char* e = std::getenv("DEEP_EP_FUSED_PAD_CYCLES");
        return (e != nullptr && e[0] != '\0') ? static_cast<uint64_t>(std::atoll(e)) : 30000000ull;
    }();
    const int pad_wgs = pad_wgs_env;
    const int total_wgs = num_sms + pad_wgs;
    const uint64_t pad_cycles = pad_cycles_env;

    queue.submit([&](sycl::handler& cgh) {
        // CUDA internode.cu:558-560 (RDMA sender warp synchronization state).
        sycl::local_accessor<int, 1> smem_lock(sycl::range<1>{static_cast<size_t>(kNumRDMARanks)}, cgh);
        sycl::local_accessor<int, 1> smem_tail(sycl::range<1>{static_cast<size_t>(kNumRDMARanks)}, cgh);
        sycl::local_accessor<unsigned int, 1> smem_window(sycl::range<1>{static_cast<size_t>(kNumRDMARanks)}, cgh);
        // CUDA internode.cu:577-579 (forward warp synchronization state).
        sycl::local_accessor<int, 1> smem_fwd_head(sycl::range<1>{static_cast<size_t>(NUM_MAX_NVL_PEERS * kNumRDMARanks)}, cgh);
        sycl::local_accessor<int, 1> smem_fwd_retired(sycl::range<1>{static_cast<size_t>(NUM_MAX_NVL_PEERS)}, cgh);

        cgh.parallel_for<FusedDispatchKernel<kNumRDMARanks>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(total_wgs) * kNumThreads), sycl::range<1>(kNumThreads)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                auto sg = item.get_sub_group();
                const int raw_sm_id = static_cast<int>(item.get_group(0));
                // §24.2 padding work-group: hold the slot, touch nothing, then retire.
                // Pads take the LOW work-group ids so the hardware gives them their slots
                // FIRST - that is what actually denies co-residency to the real ones.
                if (raw_sm_id < pad_wgs) {
                    const uint64_t t0 = dev_clock();
                    while (dev_clock() - t0 < pad_cycles) {
                    }
                    return;
                }
                const int sm_id = raw_sm_id - pad_wgs;
                const int thread_id = static_cast<int>(item.get_local_id(0));
                const int warp_id = static_cast<int>(sg.get_group_linear_id());
                const int lane_id = static_cast<int>(sg.get_local_linear_id());
                const int channel_id = sm_id / 2;
                const bool is_forwarder = (sm_id % 2 == 0);
                const int rdma_rank = rank / num_nvl_ranks;
                const int nvl_rank = rank % num_nvl_ranks;

                // IGC: both named-barrier handles are plain SSA locals of THIS
                // function, materialized before any control flow.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                auto* nb_rdma_sender = ::named_barrier_init(DEEP_EP_FUSED_SENDER_WARPS + 1);  // CUDA barrier id 0
                auto* nb_forwarder = ::named_barrier_init(NUM_MAX_NVL_PEERS + 1);              // CUDA barrier id 1
#endif

                // ---- Warp roles: CUDA internode.cu:491-508 ----
                enum Role { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };
                Role warp_role;
                int target_rank;
                if (is_forwarder) {
                    if (warp_id < NUM_MAX_NVL_PEERS) {
                        warp_role = kRDMAAndNVLForwarder;
                        target_rank = (warp_id + channel_id) % NUM_MAX_NVL_PEERS;
                    } else {
                        warp_role = kForwarderCoordinator;
                        target_rank = warp_id - NUM_MAX_NVL_PEERS;
                    }
                } else if (warp_id < DEEP_EP_FUSED_SENDER_WARPS) {
                    warp_role = kRDMASender;
                    target_rank = -1;
                } else if (warp_id == DEEP_EP_FUSED_SENDER_WARPS) {
                    warp_role = kRDMASenderCoordinator;
                    target_rank = -1;
                } else {
                    warp_role = kNVLReceivers;
                    target_rank = (warp_id + channel_id - DEEP_EP_FUSED_SENDER_WARPS) % NUM_MAX_NVL_PEERS;
                }

                // ---- RDMA symmetric layout: CUDA internode.cu:518-527 ----
                void* rdma_cursor = rdma_buffer_ptr;
                FusedSymBuffer<uint8_t> rdma_channel_data(
                    rdma_cursor, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
                FusedSymBuffer<int> rdma_channel_meta(rdma_cursor, num_nvl_ranks * 2 + 2, kNumRDMARanks, channel_id, num_channels);
                FusedSymBuffer<uint64_t, false> rdma_channel_head(rdma_cursor, 1, kNumRDMARanks, channel_id, num_channels);
                FusedSymBuffer<uint64_t, false> rdma_channel_tail(rdma_cursor, 1, kNumRDMARanks, channel_id, num_channels);

                // ---- NVL buffer layouts: CUDA internode.cu:529-556 ----
                // `rs_wr` = Read for Senders / Write for Receivers,
                // `ws_rr` = Write for Senders / Read for Receivers.
                // NOTE this protocol is already PUSH-ONLY across the IPC boundary:
                // the forwarder only WRITES peer memory (x / prefix / tail) and only
                // READS its own (head); the receiver mirrors it. No remote IPC READ
                // exists, so the BMG remote-read instability does not apply and the
                // CUDA data flow is kept verbatim.
                void* rs_wr_buffer_ptr = buffer_ptrs[nvl_rank];
                void* ws_rr_buffer_ptr = buffer_ptrs[nvl_rank];
                int rs_wr_rank = nvl_rank, ws_rr_rank = nvl_rank;
                const bool nvl_active = (target_rank >= 0 && target_rank < num_nvl_ranks);
                if (warp_role == kRDMAAndNVLForwarder && nvl_active) {
                    rs_wr_buffer_ptr = buffer_ptrs[nvl_rank];
                    ws_rr_buffer_ptr = buffer_ptrs[target_rank];
                    rs_wr_rank = nvl_rank;
                    ws_rr_rank = target_rank;
                }
                if (warp_role == kNVLReceivers && nvl_active) {
                    rs_wr_buffer_ptr = buffer_ptrs[target_rank];
                    ws_rr_buffer_ptr = buffer_ptrs[nvl_rank];
                    rs_wr_rank = target_rank;
                    ws_rr_rank = nvl_rank;
                }

                FusedAsymBuffer<uint8_t> nvl_channel_x(ws_rr_buffer_ptr,
                                                       num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                                       num_nvl_ranks,
                                                       channel_id,
                                                       num_channels,
                                                       rs_wr_rank);
                nvl_channel_x.advance_also(rs_wr_buffer_ptr);
                FusedAsymBuffer<int> nvl_channel_prefix_start(
                    ws_rr_buffer_ptr, kNumRDMARanks, num_nvl_ranks, channel_id, num_channels, rs_wr_rank);
                nvl_channel_prefix_start.advance_also(rs_wr_buffer_ptr);
                FusedAsymBuffer<int> nvl_channel_prefix_end(
                    ws_rr_buffer_ptr, kNumRDMARanks, num_nvl_ranks, channel_id, num_channels, rs_wr_rank);
                nvl_channel_prefix_end.advance_also(rs_wr_buffer_ptr);
                FusedAsymBuffer<int> nvl_channel_head(rs_wr_buffer_ptr, 1, num_nvl_ranks, channel_id, num_channels, ws_rr_rank);
                nvl_channel_head.advance_also(ws_rr_buffer_ptr);
                FusedAsymBuffer<int> nvl_channel_tail(ws_rr_buffer_ptr, 1, num_nvl_ranks, channel_id, num_channels, rs_wr_rank);
                nvl_channel_tail.advance_also(rs_wr_buffer_ptr);

                // =========================================================
                // Role: kRDMASender  (CUDA internode.cu:582-748)
                // =========================================================
                if (warp_role == kRDMASender) {
                    int token_start_idx, token_end_idx;
                    get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

                    // Send number of tokens in this channel by `-value - 1`.
                    for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += DEEP_EP_FUSED_SENDER_WARPS) {
                        int* dst_ptr = dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank)
                                                                  : rdma_channel_meta.send_buffer(dst_rdma_rank);
                        if (lane_id < num_nvl_ranks) {
                            const int v = (channel_id == 0)
                                              ? 0
                                              : gbl_channel_prefix_matrix[(dst_rdma_rank * num_nvl_ranks + lane_id) * num_channels +
                                                                          channel_id - 1];
                            uc_store(dst_ptr + lane_id, -v - 1);
                        } else if (lane_id < num_nvl_ranks * 2) {
                            const int v =
                                gbl_channel_prefix_matrix[(dst_rdma_rank * num_nvl_ranks + lane_id - num_nvl_ranks) * num_channels +
                                                          channel_id];
                            uc_store(dst_ptr + lane_id, -v - 1);
                        } else if (lane_id == num_nvl_ranks * 2) {
                            const int v =
                                (channel_id == 0) ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
                            uc_store(dst_ptr + lane_id, -v - 1);
                        } else if (lane_id == num_nvl_ranks * 2 + 1) {
                            const int v = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
                            uc_store(dst_ptr + lane_id, -v - 1);
                        }
                        sycl::group_barrier(sg);

                        if (dst_rdma_rank != rdma_rank) {
                            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                            lsc_fence_sysrel();
                            const int dst_pe = dst_rdma_rank * num_nvl_ranks + nvl_rank;
                            const unsigned qp = static_cast<unsigned>(channel_id) % qps_per_pe;
                            ishmemx_putmem_nbi_subgroup(rdma_channel_meta.recv_buffer(rdma_rank),
                                                        rdma_channel_meta.send_buffer(dst_rdma_rank),
                                                        sizeof(int) * (num_nvl_ranks * 2 + 2),
                                                        dst_pe,
                                                        qp,
                                                        true,
                                                        sg,
                                                        true);
                            sycl::group_barrier(sg);
                        }
                    }
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    ::work_group_named_barrier(nb_rdma_sender, 0x3);  // CUDA :621 sync_rdma_sender_smem()
#else
                    item.barrier(sycl::access::fence_space::global_and_local);
#endif

                    // ---- Iterate over tokens and copy into the symmetric send buffer ----
                    int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
                    uint8_t* send_buffer_lane = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id)
                                                                     : rdma_channel_data.send_buffer(lane_id);
                    for (int token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
                        // Read RDMA rank existence: NUM_MAX_NVL_PEERS bools packed as one u64
                        // in CUDA. num_nvl_ranks may be < 8 here, so read the bools directly.
                        uint32_t in_rank_mask = 0;
                        if (lane_id < kNumRDMARanks) {
                            for (int i = 0; i < num_nvl_ranks; ++i) {
                                if (is_token_in_rank[static_cast<int64_t>(token_idx) * num_ranks + lane_id * num_nvl_ranks + i])
                                    in_rank_mask |= (1u << i);
                            }
                            global_rdma_tail_idx += (in_rank_mask != 0);
                        }
                        sycl::group_barrier(sg);

                        if ((token_idx - token_start_idx) % DEEP_EP_FUSED_SENDER_WARPS != warp_id) continue;
                        const int rdma_tail_idx = (in_rank_mask == 0) ? -1 : global_rdma_tail_idx - 1;

                        // Wait for the remote buffer to be released.
                        uint64_t spins = 0;
                        while (in_rank_mask != 0 && rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                            cached_rdma_channel_head = static_cast<int>(poll_load(rdma_channel_head.buffer(lane_id)));
                            if (++spins > kFusedSpinCap) break;
                        }
                        sycl::group_barrier(sg);

                        if (lane_id < kNumRDMARanks && !kCachedMode && send_rdma_head != nullptr)
                            send_rdma_head[static_cast<int64_t>(token_idx) * kNumRDMARanks + lane_id] = rdma_tail_idx;

                        // Broadcast tails (CUDA :664-681).
                        SourceMeta src_meta{};
                        int num_topk_ranks = 0;
                        uint8_t* dst_send_buffers[kNumRDMARanks];
                        for (int i = 0; i < kNumRDMARanks; ++i) {
                            int slot_idx = sycl::select_from_group(sg, rdma_tail_idx, i);
                            if (slot_idx < 0) continue;
                            slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                            const uint32_t recv_mask = sycl::select_from_group(sg, in_rank_mask, i);
                            if (lane_id == num_topk_ranks) {
                                src_meta.src_rdma_rank = rdma_rank;
                                src_meta.is_token_in_nvl_rank_bits = static_cast<int>(recv_mask);
                                src_meta.src_nvl_rank = nvl_rank;
                            }
                            const uint64_t base = sycl::select_from_group(sg, reinterpret_cast<uint64_t>(send_buffer_lane), i);
                            dst_send_buffers[num_topk_ranks++] =
                                reinterpret_cast<uint8_t*>(base) + static_cast<size_t>(slot_idx) * num_bytes_per_token;
                        }

                        // Copy `x` (16B granularity) broadcast to every destination.
                        {
                            const int4_t* src4 = reinterpret_cast<const int4_t*>(x) + static_cast<int64_t>(token_idx) * hidden_int4;
                            for (int i = lane_id; i < hidden_int4; i += 32) {
                                const int4_t v = ld_nc_global_v(src4 + i);
                                for (int j = 0; j < num_topk_ranks; ++j)
                                    st_na_global_v(reinterpret_cast<int4_t*>(dst_send_buffers[j]) + i, v);
                            }
                        }
                        for (int i = 0; i < num_topk_ranks; ++i) dst_send_buffers[i] += hidden_bytes;

                        // Copy `x_scales`.
                        if (x_scales != nullptr) {
                            for (int i = lane_id; i < num_scales; i += 32) {
                                const float v = x_scales[static_cast<int64_t>(token_idx) * num_scales + i];
                                for (int j = 0; j < num_topk_ranks; ++j) reinterpret_cast<float*>(dst_send_buffers[j])[i] = v;
                            }
                        }
                        for (int i = 0; i < num_topk_ranks; ++i) dst_send_buffers[i] += scale_bytes;

                        // Copy source metadata.
                        if (lane_id < num_topk_ranks) *reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]) = src_meta;
                        for (int i = 0; i < num_topk_ranks; ++i) dst_send_buffers[i] += sizeof(SourceMeta);

                        // Copy `topk_idx` / `topk_weights`.
                        if (topk_idx != nullptr) {
                            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
                                const int rank_idx = i / num_topk, copy_idx = i % num_topk;
                                const int idx_value =
                                    static_cast<int>(topk_idx[static_cast<int64_t>(token_idx) * num_topk + copy_idx]);
                                const float w_value = topk_weights[static_cast<int64_t>(token_idx) * num_topk + copy_idx];
                                reinterpret_cast<int*>(dst_send_buffers[rank_idx])[copy_idx] = idx_value;
                                reinterpret_cast<float*>(dst_send_buffers[rank_idx] + sizeof(int) * num_topk)[copy_idx] = w_value;
                            }
                        }
                        sycl::group_barrier(sg);
                        // Payload must be globally visible before the coordinator's
                        // doorbell / local tail bump observes the released slot.
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        lsc_fence_sysrel();

                        // ---- Release the transaction in the window (CUDA :721-745) ----
                        // D3: serialized over dst_rdma_rank so a sub-group never holds
                        // more than one SLM lock at a time.
                        for (int r = 0; r < kNumRDMARanks; ++r) {
                            const uint32_t r_mask = sycl::select_from_group(sg, in_rank_mask, r);
                            if (r_mask == 0) continue;
                            if (lane_id == r) {
                                sycl::atomic_ref<int,
                                                 sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::work_group,
                                                 sycl::access::address_space::local_space>
                                    lock_ref(smem_lock[r]);
                                sycl::atomic_ref<int,
                                                 sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::work_group,
                                                 sycl::access::address_space::local_space>
                                    tail_ref(smem_tail[r]);
                                int latest_tail, offset;
                                uint64_t guard = 0;
                                while (true) {
                                    while (lock_ref.exchange(1) == 1) {
                                        if (++guard > kFusedSpinCap) break;
                                    }
                                    latest_tail = tail_ref.load();
                                    offset = rdma_tail_idx - latest_tail;
                                    if (offset < 32) break;
                                    lock_ref.store(0);
                                    if (++guard > kFusedSpinCap) break;
                                }
                                unsigned int window = smem_window[r] | (1u << offset);
                                if (offset == 0) {
                                    const unsigned int inv = ~window;
                                    const int num_empty_slots = (inv == 0) ? 32 : static_cast<int>(sycl::ctz(inv));
                                    tail_ref.store(latest_tail + num_empty_slots);
                                    window = (num_empty_slots >= 32) ? 0u : (window >> num_empty_slots);
                                }
                                smem_window[r] = window;
                                lock_ref.store(0);
                            }
                            sycl::group_barrier(sg);
                        }
                    }
                }
                // =========================================================
                // Role: kRDMASenderCoordinator (CUDA internode.cu:748-843)
                // =========================================================
                else if (warp_role == kRDMASenderCoordinator) {
                    if (lane_id < kNumRDMARanks) {
                        smem_lock[lane_id] = 0;
                        smem_tail[lane_id] = 0;
                        smem_window[lane_id] = 0;
                    }
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    ::work_group_named_barrier(nb_rdma_sender, 0x3);  // CUDA :918
#else
                    item.barrier(sycl::access::fence_space::global_and_local);
#endif

                    int num_tokens_to_send = 0;
                    if (lane_id < kNumRDMARanks) {
                        num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
                        if (channel_id > 0)
                            num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
                    }

                    int last_issued_tail = 0;
                    uint64_t spins = 0;
                    while (sycl::any_of_group(sg, num_tokens_to_send > 0)) {
                        if (++spins > kFusedSpinCap) break;
                        for (int i = 0; i < kNumRDMARanks; ++i) {
                            const int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                            const int synced_num_tokens_to_send = sycl::select_from_group(sg, num_tokens_to_send, dst_rdma_rank);
                            if (synced_num_tokens_to_send == 0) continue;

                            int processed_tail;
                            {
                                sycl::atomic_ref<int,
                                                 sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::work_group,
                                                 sycl::access::address_space::local_space>
                                    tail_ref(smem_tail[dst_rdma_rank]);
                                processed_tail = sycl::select_from_group(sg, tail_ref.load(), 0);
                            }
                            const int synced_last_issued_tail = sycl::select_from_group(sg, last_issued_tail, dst_rdma_rank);
                            const int num_tokens_processed = processed_tail - synced_last_issued_tail;
                            if (num_tokens_processed != synced_num_tokens_to_send &&
                                num_tokens_processed < num_max_rdma_chunked_send_tokens)
                                continue;

                            const int num_tokens_to_issue = sycl::min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
                            const unsigned qp = static_cast<unsigned>(channel_id) % qps_per_pe;
                            const int dst_pe = dst_rdma_rank * num_nvl_ranks + nvl_rank;
                            if (dst_rdma_rank != rdma_rank) {
                                const int dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                                const size_t num_bytes_per_msg =
                                    static_cast<size_t>(num_bytes_per_token) * num_tokens_to_issue;
                                uint8_t* dst_ptr = rdma_channel_data.recv_buffer(rdma_rank) +
                                                   static_cast<size_t>(dst_slot_idx) * num_bytes_per_token;
                                uint8_t* src_ptr = rdma_channel_data.send_buffer(dst_rdma_rank) +
                                                   static_cast<size_t>(dst_slot_idx) * num_bytes_per_token;
                                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                                lsc_fence_sysrel();
                                ishmemx_putmem_nbi_subgroup(dst_ptr, src_ptr, num_bytes_per_msg, dst_pe, qp, true, sg, true);
                                sycl::group_barrier(sg);
                            } else {
                                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                                lsc_fence_sysrel();
                            }
                            sycl::group_barrier(sg);

                            if (lane_id == dst_rdma_rank) {
                                last_issued_tail += num_tokens_to_issue;
                                num_tokens_to_send -= num_tokens_to_issue;
                                if (dst_rdma_rank == rdma_rank) {
                                    // Local RDMA rank: plain system-scope atomic bump.
                                    sycl::atomic_ref<uint64_t,
                                                     sycl::memory_order::acq_rel,
                                                     sycl::memory_scope::system,
                                                     sycl::access::address_space::global_space>
                                        t(*rdma_channel_tail.buffer(rdma_rank));
                                    t.fetch_add(static_cast<uint64_t>(num_tokens_to_issue));
                                } else {
                                    // RC-ordered behind the payload put on the SAME qp.
                                    ishmemx_fence_qp(dst_pe, qp);
                                    ishmemx_long_atomic_add_qp(reinterpret_cast<long*>(rdma_channel_tail.buffer(rdma_rank)),
                                                               static_cast<long>(num_tokens_to_issue),
                                                               dst_pe,
                                                               qp);
                                }
                            }
                            sycl::group_barrier(sg);
                        }
                    }
                }
                // =========================================================
                // Role: kRDMAAndNVLForwarder (CUDA internode.cu:843-1013)
                // =========================================================
                else if (warp_role == kRDMAAndNVLForwarder) {
                    const int dst_nvl_rank = target_rank;
                    if (!nvl_active) {
                        // D2: idle forwarder warp; it MUST still arrive at the 9-way
                        // barrier or the participating warps deadlock.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                        ::work_group_named_barrier(nb_forwarder, 0x3);
#else
                        item.barrier(sycl::access::fence_space::global_and_local);
#endif
                        return;
                    }

                    int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
                    if (lane_id < kNumRDMARanks) {
                        uint64_t spins = 0;
                        while (true) {
                            const int meta_0 = poll_load(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                            const int meta_1 = poll_load(rdma_channel_meta.recv_buffer(lane_id) + num_nvl_ranks + dst_nvl_rank);
                            const int meta_2 = poll_load(rdma_channel_meta.recv_buffer(lane_id) + num_nvl_ranks * 2);
                            const int meta_3 = poll_load(rdma_channel_meta.recv_buffer(lane_id) + num_nvl_ranks * 2 + 1);
                            if (meta_0 < 0 && meta_1 < 0 && meta_2 < 0 && meta_3 < 0) {
                                const int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                                uc_store(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
                                uc_store(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);
                                src_rdma_channel_prefix = -meta_2 - 1;
                                const int src_rdma_channel_prefix_1 = -meta_3 - 1;
                                num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                                if (!kCachedMode && recv_rdma_channel_prefix_matrix != nullptr)
                                    recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
                                src_rdma_channel_prefix += (lane_id == 0) ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
                                break;
                            }
                            if (++spins > kFusedSpinCap) break;
                        }
                    }
                    sycl::group_barrier(sg);
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    lsc_fence_sysrel();

                    int* my_send_nvl_head =
                        (send_nvl_head != nullptr)
                            ? (send_nvl_head + static_cast<int64_t>(src_rdma_channel_prefix) * NUM_MAX_NVL_PEERS + dst_nvl_rank)
                            : nullptr;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    ::work_group_named_barrier(nb_forwarder, 0x3);  // CUDA :959 sync_forwarder_smem()
#else
                    item.barrier(sycl::access::fence_space::global_and_local);
#endif

                    int src_rdma_rank = sm_id % kNumRDMARanks;
                    int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
                    int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
                    while (sycl::any_of_group(sg, num_tokens_to_recv_from_rdma > 0)) {
                        // Wait for destination NVL queue space.
                        {
                            uint64_t spins = 0;
                            while (true) {
                                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens) break;
                                cached_nvl_channel_head = sycl::select_from_group(sg, poll_load(nvl_channel_head.buffer()), 0);
                                if (++spins > kFusedSpinCap) break;
                            }
                        }

                        // Find next source RDMA rank (round-robin).
                        {
                            uint64_t spins = 0;
                            while (true) {
                                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                                if (sycl::select_from_group(sg, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                                    if (lane_id == src_rdma_rank && cached_rdma_channel_head == cached_rdma_channel_tail) {
                                        lsc_fence_sysacq();
                                        cached_rdma_channel_tail = static_cast<int>(poll_load(rdma_channel_tail.buffer(src_rdma_rank)));
                                    }
                                    if (sycl::select_from_group(
                                            sg, cached_rdma_channel_tail > cached_rdma_channel_head ? 1 : 0, src_rdma_rank))
                                        break;
                                }
                                if (++spins > kFusedSpinCap) break;
                            }
                        }
                        const int src_rdma_head = sycl::select_from_group(sg, cached_rdma_channel_head, src_rdma_rank);
                        int src_rdma_tail = sycl::select_from_group(sg, cached_rdma_channel_tail, src_rdma_rank);

                        // Ensure NIC-delivered payload is visible before it is read.
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();

                        for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
                            const int rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
                            uint8_t* shifted = rdma_channel_data.recv_buffer(src_rdma_rank) +
                                               static_cast<size_t>(rdma_slot_idx) * num_bytes_per_token;
                            const SourceMeta src_meta =
                                *reinterpret_cast<const SourceMeta*>(shifted + hidden_bytes + scale_bytes);
                            if (lane_id == src_rdma_rank) num_tokens_to_recv_from_rdma -= 1;
                            const bool is_in_dst_nvl_rank = ((src_meta.is_token_in_nvl_rank_bits >> dst_nvl_rank) & 1) != 0;
                            if (lane_id == src_rdma_rank) {
                                const int cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                                rdma_nvl_token_idx += is_in_dst_nvl_rank ? 1 : 0;
                                if (!kCachedMode && my_send_nvl_head != nullptr)
                                    my_send_nvl_head[static_cast<int64_t>(i) * NUM_MAX_NVL_PEERS] = cached_head;
                            }
                            if (!is_in_dst_nvl_rank) continue;

                            const int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;
                            uint8_t* dst_shifted =
                                nvl_channel_x.buffer() + static_cast<size_t>(dst_slot_idx) * num_bytes_per_token;

                            // D1: TMA replaced by sub-group cooperative 16B copy.
                            const int n16 = num_bytes_per_token / static_cast<int>(sizeof(int4_t));
                            for (int k = lane_id; k < n16; k += 32)
                                st_na_global_v(reinterpret_cast<int4_t*>(dst_shifted) + k,
                                               ld_nc_global_v(reinterpret_cast<const int4_t*>(shifted) + k));
                            sycl::group_barrier(sg);

                            if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens) src_rdma_tail = i + 1;
                        }

                        if (lane_id == src_rdma_rank) {
                            cached_rdma_channel_head = src_rdma_tail;
                            smem_fwd_head[dst_nvl_rank * kNumRDMARanks + src_rdma_rank] = src_rdma_tail;
                        }
                        sycl::group_barrier(sg);
                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        lsc_fence_sysrel();
                        if (lane_id == 0) uc_store(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
                        sycl::group_barrier(sg);
                    }

                    sycl::group_barrier(sg);
                    if (lane_id == 0) {
                        sycl::atomic_ref<int,
                                         sycl::memory_order::acq_rel,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            r(smem_fwd_retired[dst_nvl_rank]);
                        r.store(1);
                    }
                }
                // =========================================================
                // Role: kForwarderCoordinator (CUDA internode.cu:1013-1054)
                // =========================================================
                else if (warp_role == kForwarderCoordinator) {
                    if (target_rank > 0) return;  // CUDA :1015-1016 (extra warps exit)

                    for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
                        smem_fwd_head[(i % NUM_MAX_NVL_PEERS) * kNumRDMARanks + (i / NUM_MAX_NVL_PEERS)] = 0;
                    if (lane_id < NUM_MAX_NVL_PEERS) smem_fwd_retired[lane_id] = (lane_id < num_nvl_ranks) ? 0 : 1;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
                    ::work_group_named_barrier(nb_forwarder, 0x3);  // CUDA :1029
#else
                    item.barrier(sycl::access::fence_space::global_and_local);
#endif

                    int last_head = 0;
                    const int target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
                    uint64_t spins = 0;
                    while (true) {
                        int min_head = kFusedIntMax;
                        for (int i = 0; i < num_nvl_ranks; ++i) {
                            sycl::atomic_ref<int,
                                             sycl::memory_order::acq_rel,
                                             sycl::memory_scope::work_group,
                                             sycl::access::address_space::local_space>
                                r(smem_fwd_retired[i]);
                            if (r.load() == 0) {
                                sycl::atomic_ref<int,
                                                 sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::work_group,
                                                 sycl::access::address_space::local_space>
                                    h(smem_fwd_head[i * kNumRDMARanks + target_rdma]);
                                min_head = sycl::min(min_head, h.load());
                            }
                        }
                        if (sycl::all_of_group(sg, min_head == kFusedIntMax)) break;

                        if (min_head != kFusedIntMax && min_head >= last_head + num_max_rdma_chunked_send_tokens &&
                            lane_id < kNumRDMARanks) {
                            const int dst_pe = lane_id * num_nvl_ranks + nvl_rank;
                            const unsigned qp = static_cast<unsigned>(channel_id + num_channels) % qps_per_pe;
                            if (lane_id == rdma_rank) {
                                sycl::atomic_ref<uint64_t,
                                                 sycl::memory_order::acq_rel,
                                                 sycl::memory_scope::system,
                                                 sycl::access::address_space::global_space>
                                    h(*rdma_channel_head.buffer(rdma_rank));
                                h.fetch_add(static_cast<uint64_t>(min_head - last_head));
                            } else {
                                ishmemx_long_atomic_add_qp(reinterpret_cast<long*>(rdma_channel_head.buffer(rdma_rank)),
                                                           static_cast<long>(min_head - last_head),
                                                           dst_pe,
                                                           qp);
                            }
                            last_head = min_head;
                        }
                        if (++spins > kFusedSpinCap) break;
                    }
                }
                // =========================================================
                // Role: kNVLReceivers (CUDA internode.cu:1054-1191)
                // =========================================================
                else {
                    const int src_nvl_rank = target_rank;
                    if (!nvl_active) return;  // D2: no barrier participation on this path

                    int total_offset = 0;
                    const int local_expert_begin = (num_experts > 0) ? rank * (num_experts / num_ranks) : 0;
                    const int local_expert_end = (num_experts > 0) ? local_expert_begin + (num_experts / num_ranks) : 0;

                    if (lane_id < kNumRDMARanks && lane_id * num_nvl_ranks + src_nvl_rank > 0)
                        total_offset = recv_gbl_rank_prefix_sum[lane_id * num_nvl_ranks + src_nvl_rank - 1];

                    int start_offset = 0, end_offset = 0;
                    if (lane_id < kNumRDMARanks) {
                        uint64_t spins = 0;
                        while (true) {
                            start_offset = poll_load(nvl_channel_prefix_start.buffer() + lane_id);
                            end_offset = poll_load(nvl_channel_prefix_end.buffer() + lane_id);
                            if (start_offset < 0 && end_offset < 0) {
                                start_offset = -start_offset - 1;
                                end_offset = -end_offset - 1;
                                total_offset += start_offset;
                                break;
                            }
                            if (++spins > kFusedSpinCap) break;
                        }
                    }
                    int num_tokens_to_recv = sycl::reduce_over_group(sg, end_offset - start_offset, sycl::plus<int>());

                    if (lane_id < kNumRDMARanks && !kCachedMode && recv_gbl_channel_prefix_matrix != nullptr)
                        recv_gbl_channel_prefix_matrix[(lane_id * num_nvl_ranks + src_nvl_rank) * num_channels + channel_id] =
                            total_offset;
                    sycl::group_barrier(sg);

                    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
                    while (num_tokens_to_recv > 0) {
                        uint64_t spins = 0;
                        while (true) {
                            if (cached_channel_head_idx != cached_channel_tail_idx) break;
                            cached_channel_tail_idx = sycl::select_from_group(sg, poll_load(nvl_channel_tail.buffer()), 0);
                            if (++spins > kFusedSpinCap) break;
                        }
                        if (cached_channel_head_idx == cached_channel_tail_idx) break;  // spin cap hit

                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        lsc_fence_sysacq();

                        const int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
                        for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
                            const int token_idx_in_buffer = (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;
                            uint8_t* shifted =
                                nvl_channel_x.buffer() + static_cast<size_t>(token_idx_in_buffer) * num_bytes_per_token;
                            const SourceMeta meta = *reinterpret_cast<const SourceMeta*>(shifted + hidden_bytes + scale_bytes);
                            const int64_t recv_token_idx = sycl::select_from_group(sg, total_offset, meta.src_rdma_rank);
                            if (lane_id == meta.src_rdma_rank) total_offset += 1;

                            // Copy hidden.
                            {
                                int4_t* dst4 = reinterpret_cast<int4_t*>(recv_x) + recv_token_idx * hidden_int4;
                                const int4_t* src4 = reinterpret_cast<const int4_t*>(shifted);
                                for (int k = lane_id; k < hidden_int4; k += 32) st_na_global_v(dst4 + k, ld_nc_global_v(src4 + k));
                            }
                            shifted += hidden_bytes;

                            // Copy scales.
                            if (recv_x_scales != nullptr) {
                                const float* src_s = reinterpret_cast<const float*>(shifted);
                                float* dst_s = recv_x_scales + recv_token_idx * num_scales;
                                for (int k = lane_id; k < num_scales; k += 32) dst_s[k] = src_s[k];
                            }
                            shifted += scale_bytes;

                            if (!kCachedMode && recv_src_meta != nullptr && lane_id == 0) recv_src_meta[recv_token_idx] = meta;
                            shifted += sizeof(SourceMeta);

                            if (recv_topk_idx != nullptr && lane_id < num_topk) {
                                topk_idx_t idx_value = static_cast<topk_idx_t>(reinterpret_cast<const int*>(shifted)[lane_id]);
                                float weight_value =
                                    reinterpret_cast<const float*>(shifted + sizeof(int) * num_topk)[lane_id];
                                const int64_t recv_idx = recv_token_idx * num_topk + lane_id;
                                if (num_experts > 0) {
                                    idx_value = (idx_value >= local_expert_begin && idx_value < local_expert_end)
                                                    ? idx_value - local_expert_begin
                                                    : -1;
                                    weight_value = idx_value >= 0 ? weight_value : 0.0f;
                                }
                                recv_topk_idx[recv_idx] = idx_value;
                                recv_topk_weights[recv_idx] = weight_value;
                            }
                            sycl::group_barrier(sg);
                        }

                        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        lsc_fence_sysrel();
                        if (lane_id == 0) uc_store(nvl_channel_head.buffer(), cached_channel_head_idx);
                        sycl::group_barrier(sg);
                    }
                }
            });
    });
#endif
}
