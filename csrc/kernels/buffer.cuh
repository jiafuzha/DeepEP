#pragma once

#include "configs.cuh"
#include "exception.cuh"

namespace deep_ep {

template <typename dtype_t>
struct Buffer {
private:
    uint8_t* ptr;

public:
    int64_t total_bytes;

    EP_DEVICE EP_FORCEINLINE Buffer() : ptr(nullptr), total_bytes(0) {}

    EP_DEVICE EP_FORCEINLINE Buffer(void*& gbl_ptr, int num_elems, int offset = 0) {
        total_bytes = num_elems * sizeof(dtype_t);
        ptr = static_cast<uint8_t*>(gbl_ptr) + offset * sizeof(dtype_t);
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    EP_DEVICE EP_FORCEINLINE Buffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    EP_DEVICE EP_FORCEINLINE dtype_t* buffer() { return reinterpret_cast<dtype_t*>(ptr); }

    EP_DEVICE EP_FORCEINLINE dtype_t& operator[](int idx) { return buffer()[idx]; }
};

template <typename dtype_t, int kNumRanks = 1>
struct AsymBuffer {
private:
    uint8_t* ptrs[kNumRanks];
    int64_t num_bytes;

public:
    int64_t total_bytes;

    EP_DEVICE EP_FORCEINLINE AsymBuffer(void*& gbl_ptr, int num_elems, int num_ranks, int sm_id = 0, int num_sms = 1, int offset = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "");
        num_bytes = num_elems * sizeof(dtype_t);

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms;
        ptrs[0] = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id + num_bytes * offset;
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    EP_DEVICE EP_FORCEINLINE AsymBuffer(void** gbl_ptrs, int num_elems, int num_ranks, int sm_id = 0, int num_sms = 1, int offset = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "");
        num_bytes = num_elems * sizeof(dtype_t);

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms;
        for (int i = 0; i < kNumRanks; ++i) {
            ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + per_channel_bytes * sm_id + num_bytes * offset;
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        }
    }

    EP_DEVICE EP_FORCEINLINE void advance(int shift) {
        #pragma unroll
        for (int i = 0; i < kNumRanks; ++i)
            ptrs[i] = ptrs[i] + shift * sizeof(dtype_t);
    }

    EP_DEVICE EP_FORCEINLINE AsymBuffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    template <int kNumAlsoRanks>
    EP_DEVICE EP_FORCEINLINE AsymBuffer advance_also(void** gbl_ptrs) {
        for (int i = 0; i < kNumAlsoRanks; ++i)
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        return *this;
    }

    EP_DEVICE EP_FORCEINLINE dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t*>(ptrs[0] + num_bytes * idx);
    }

    EP_DEVICE EP_FORCEINLINE dtype_t* buffer_by(int rank_idx, int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t*>(ptrs[rank_idx] + num_bytes * idx);
    }
};

template <typename dtype_t, bool kDecoupled = true>
struct SymBuffer {
private:
    // NOTES: for non-decoupled case, `recv_ptr` is not used
    uint8_t* send_ptr;
    uint8_t* recv_ptr;
    int64_t num_bytes;

public:
    int64_t total_bytes;

    EP_DEVICE EP_FORCEINLINE SymBuffer(void*& gbl_ptr, int num_elems, int num_ranks, int sm_id = 0, int num_sms = 1) {
        num_bytes = num_elems * sizeof(dtype_t);

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms * (static_cast<int>(kDecoupled) + 1);
        send_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id;
        recv_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * (sm_id + num_sms);
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    EP_DEVICE EP_FORCEINLINE dtype_t* send_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`send_buffer` is only available for non-decoupled case");
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }

    EP_DEVICE EP_FORCEINLINE dtype_t* recv_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`recv_buffer` is only available for non-decoupled case");
        return reinterpret_cast<dtype_t*>(recv_ptr + num_bytes * idx);
    }

    EP_DEVICE EP_FORCEINLINE dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(not kDecoupled, "`buffer` is only available for decoupled case");
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }
};

}  // namespace deep_ep
