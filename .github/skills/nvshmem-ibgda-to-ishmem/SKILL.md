# nvshmem(ibgda)-to-ishmem Migration Skill

## Purpose
Automate and guide the migration of NVSHMEM and IBGDA device/host API calls to iSHMEM (ibgda) equivalents for XPU/SYCL platforms. This skill inventories all NVSHMEM/IBGDA usage, maps each to its iSHMEM equivalent, and provides migration patterns and usage examples. It also summarizes iSHMEM API usage and advanced patterns from the ishmem_ibgda reference implementation.

## Workflow
1. **Inventory** all NVSHMEM and IBGDA API calls in the codebase (device and host).
2. **Map** each call to its iSHMEM (ibgda) equivalent using the provided mapping tables.
3. **Report** any unmapped or ambiguous APIs for manual review.
4. **Suggest** migration patterns and code snippets for each mapped API.
5. **Summarize** iSHMEM API usage and advanced patterns from the ishmem_ibgda repo, including device/host collectives, memory management, and synchronization.
6. **Validate** that all NVSHMEM/IBGDA calls are replaced and no CUDA/NVSHMEM/IBGDA code remains in migrated kernels.

## Mapping Table Example
| NVSHMEM/IBGDA API         | iSHMEM (ibgda) Equivalent         |
|--------------------------|-----------------------------------|
| nvshmem_init             | ishmem_init                        |
| nvshmem_finalize         | ishmem_finalize                    |
| nvshmem_barrier_all      | ishmem_barrier_all                 |
| nvshmem_malloc           | ishmem_malloc                      |
| nvshmem_free             | ishmem_free                        |
| nvshmem_put              | ishmem_put                         |
| nvshmem_get              | ishmem_get                         |
| nvshmem_atomic_add       | ishmem_atomic_add                  |
| ...                      | ...                                |

## iSHMEM API Usage Patterns (from ishmem_ibgda)
- **Initialization/Finalization:**
  - `ishmem_init`, `ishmem_finalize`
- **Memory Management:**
  - `ishmem_malloc`, `ishmem_free`, `ishmem_align`
- **Point-to-Point and Collective Operations:**
  - `ishmem_put`, `ishmem_get`, `ishmem_barrier_all`, `ishmem_team_split_strided`, `ishmem_team_destroy`
- **Atomic Operations:**
  - `ishmem_atomic_add`, `ishmem_atomic_fetch_add`, `ishmem_atomic_compare_swap`
- **Synchronization:**
  - `ishmem_quiet`, `ishmem_fence`, `ishmem_barrier_all`
- **Advanced Patterns:**
  - Device-side collectives, team-based operations, memory key/rkey management, and direct RDMA primitives for low-latency communication.
For more details, refer to the [ishmem_ibgda reference](https://github.com/NVIDIA/ishmem_ibgda).

## Completion Criteria
- All NVSHMEM/IBGDA calls are mapped and replaced with iSHMEM equivalents.
- No unmapped or ambiguous APIs remain (or are explicitly reported).
- iSHMEM usage follows best practices from ishmem_ibgda reference.

## Example Prompts
- "List all NVSHMEM/IBGDA calls and their iSHMEM equivalents."
- "Suggest iSHMEM migration for nvshmem_barrier_all and nvshmem_put."
- "Show advanced iSHMEM usage for device-side collectives."
- "Report any unmapped NVSHMEM/IBGDA APIs."

## Related Customizations
- PTX assembly to SYCL barrier skill
- CUDA-to-SYCL kernel migration agent
- iSHMEM collectives and memory management best practices

---
This skill enables systematic, lossless migration from NVSHMEM/IBGDA to iSHMEM (ibgda) for XPU/SYCL platforms, leveraging both mapping tables and real-world usage from the ishmem_ibgda reference.