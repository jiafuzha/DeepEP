# UT: SPIR-V NamedBarrier is incompatible with the iSHMEM IBGDA device library (Intel Xe / BMG)

Minimal, self-contained unit test that reproduces the compiler/runtime bug behind the
DeepEP low-latency F1 dead-end: a **SYCL SPIR-V named barrier** (`work_group_named_barrier`,
`cl_khr_subgroup_named_barrier`) **cannot coexist** with the **non-inlined iSHMEM IBGDA RDC
device subroutines** in one `-fsycl-rdc` module. IGC stamps `.kernel_attr NBarrierCnt=1` onto
the kernel *and* every non-inlined iSHMEM device subroutine, and vISA rejects the duplicate:

```
Error in CISA routine with name: _ZTS17ReproIshmemKernel
              Error Message: More than 1 kernel attribute defined NBarrierCnt
```

This is why the DeepEP LL kernel cannot use a named-barrier (warp-group-scoped `bar.sync`
parity) and is stuck on a whole-work-group barrier + counter-warp shadow loop
(see `csrc/xpu/internode_ll_design.md` and the issue writeup in the session `files/`).

## Files
- `repro_ishmem_real.cpp` — a named-barrier kernel that also references the real iSHMEM
  IBGDA RDMA device path (`ishmem_putmem`/`ishmem_quiet`, under a runtime-false guard so no
  iSHMEM bootstrap is needed). `-DUSE_ISHMEM=0` builds the control (named barrier only).
- `run_ishmem_real.sh` — builds & runs three configs and prints PASS/REPRODUCED for each.

## How it maps to the reported problems
| Config | What it builds | Result |
|---|---|---|
| **[A] control** | named barrier only (`-DUSE_ISHMEM=0`) | **PASS** — a named barrier alone is fine (single `NBarrierCnt`). |
| **[B] repro** | named barrier + real iSHMEM IBGDA device path | **FAIL** — `More than 1 kernel attribute defined NBarrierCnt` at JIT (**Problem 1**). |
| **[C] workaround** | [B] + `IGC_FunctionControl=1` (force-inline-all) | **PASS** — inlining removes the iSHMEM subroutines, but costs ~+32% latency in the real LL kernel (**Problem 2**). |

The `NBarrierCnt` error is a **JIT-compile** error: it surfaces at the first kernel launch,
*before* the kernel body runs, so iSHMEM does **not** need to be initialized (no MPI/NIC
bootstrap) to reproduce it. IGC writes the exact message to `kernel.errors.txt` in the
working directory; the script finds and prints it.

## Prerequisites
- Intel Xe / BMG GPU (validated on Arc Pro B60), Level Zero.
- oneAPI **2025.3** DPC++ (the toolchain DeepEP builds with; matched by the build container).
- The iSHMEM IBGDA install DeepEP links against, at
  `ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install` (override via `ISHMEM_DIR`).

> Note: on host oneAPI **2026.0** the small synthetic-function shape may inline away and not
> reproduce; run inside the DeepEP build container (2025.3) with the real iSHMEM archive, as
> below. That is the faithful condition — it is the same `-fsycl-rdc` + iSHMEM device link
> `setup.py` performs for `deep_ep_cpp`.

## Run
Inside the DeepEP build container (recommended — matches the production toolchain):

```bash
docker run --rm --device /dev/dri \
  -v "$PWD/tests/repro-ishmem-named-barrier:/repro" \
  -v /root/jiafuzha/ishmem_ibgda:/root/jiafuzha/ishmem_ibgda \
  deepep_jiafuzha bash -lc \
  'source /opt/intel/oneapi/setvars.sh --force; cd /repro; ./run_ishmem_real.sh'
```

Or directly, if the current shell already has oneAPI 2025.3 + the iSHMEM install:

```bash
source /opt/intel/oneapi/setvars.sh --force
cd tests/repro-ishmem-named-barrier
./run_ishmem_real.sh
```

## Expected output (abridged)
```
[A] control    : PASS (named barrier alone is fine)
[B] repro      : REPRODUCED (NBarrierCnt vISA failure)
      Error in CISA routine with name: _ZTS17ReproIshmemKernel
                    Error Message: More than 1 kernel attribute defined NBarrierCnt
[C] workaround : PASS (force-inline compiles + runs)
```
The script exits `0` iff A passes, B reproduces the failure, and C works around it.

## Ask for the iSHMEM/IGC team
Provide an inline-able / header-only device path (or an annotation) so the iSHMEM IBGDA
device routines don't end up as separate non-inlined vISA `.function` subroutines under
`-fsycl-rdc`, OR stop IGC propagating the module-level `NBarrierCnt` attribute onto functions
that contain no named barriers — so a named barrier can coexist without forcing whole-library
inlining. See the full write-up saved with this work.
