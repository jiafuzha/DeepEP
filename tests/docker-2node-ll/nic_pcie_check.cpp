// nic_pcie_check.cpp
//
// Minimal iSHMEM IBGDA initializer used by verify_nic_selection.sh to validate
// that iSHMEM's AUTOMATIC GPU->NIC affinity selection picks a NIC under the SAME
// PCIe switch as each rank's GPU.
//
// We deliberately do NOT pin ISHMEM_IBGDA_NIC here: the whole point is to
// exercise iSHMEM's auto-selection (hwloc/sysfs PCIe affinity scoring in
// ibgda.cpp). On init, iSHMEM logs a per-PE "GPU-NIC mapping summary" under
// ISHMEM_DEBUG=1 with columns:
//     pe, ze_mask, expected_nic, selected_nic, nic_bdf, gpu_bdf
// The wrapper script forces ISHMEM_DEBUG=1, captures that summary, and asserts
// (host sysfs) that selected_nic's BDF shares a PCIe switch with gpu_bdf.
//
// This program just bootstraps iSHMEM (MPI runtime, unique-id) on the GPU
// chosen by ZE_AFFINITY_MASK/LOCAL_RANK, then finalizes. It prints a single
// marker line so the harness can confirm every rank initialized.

#include <ishmem.h>
#include <ishmemx.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>

namespace {

int env_int_or(const char* name, int default_value) {
    const char* value = std::getenv(name);
    return value == nullptr ? default_value : std::atoi(value);
}

bool try_parse_env_int(const char* key, int* out) {
    const char* val = std::getenv(key);
    if (val == nullptr) return false;
    *out = std::atoi(val);
    return true;
}

// Mirrors resolve_device_index() in xpu_ishmem_mapped_api_ut.cpp so the GPU
// chosen here matches the GPU the real LL test binds (LOCAL_RANK within the
// ZE_AFFINITY_MASK-filtered device list).
int resolve_device_index() {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    const int num_devices = static_cast<int>(devices.size());
    if (num_devices == 0) {
        std::cerr << "nic_pcie_check: no GPU devices found\n";
        return 0;
    }
    int local_rank = -1;
    for (const char* key : {"LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK",
                            "SLURM_LOCALID", "PMI_LOCAL_RANK", "PALS_LOCAL_RANKID"}) {
        if (try_parse_env_int(key, &local_rank)) {
            if (local_rank < 0) continue;
            return local_rank % num_devices;
        }
    }
    return 0;
}

}  // namespace

int main() {
    const int device_idx = resolve_device_index();
    const int rank = env_int_or("PMI_RANK", env_int_or("RANK", 0));
    const int nranks = env_int_or("PMI_SIZE", env_int_or("WORLD_SIZE", 2));

    ishmemx_uniqueid_t unique_id{};
    if (ishmemx_get_uniqueid(&unique_id) != 0) {
        std::cerr << "nic_pcie_check: ishmemx_get_uniqueid failed\n";
        return 2;
    }

    ishmemx_attr_t attr{};
    attr.runtime = ISHMEMX_RUNTIME_MPI;
    attr.initialize_runtime = true;
    attr.gpu = true;
    attr.use_uid = true;
    attr.nranks = nranks;
    attr.rank = rank;
    attr.uid = &unique_id;
    attr.device_idx = device_idx;
    ishmemx_init_attr(&attr);

    const int my_pe = ishmem_my_pe();
    const char* ze_mask = std::getenv("ZE_AFFINITY_MASK");
    // Marker line: confirms this PE bootstrapped IBGDA. The authoritative
    // GPU<->NIC binding (selected_nic/nic_bdf/gpu_bdf) is emitted by iSHMEM's
    // own binding summary at PE 0 under ISHMEM_DEBUG=1, which the harness parses.
    std::cerr << "NIC_PCIE_CHECK_INIT pe=" << my_pe << " device_idx=" << device_idx
              << " ze_mask=" << (ze_mask ? ze_mask : "unset") << "\n";
    std::cerr.flush();

    ishmem_barrier_all();
    ishmem_finalize();
    return 0;
}
