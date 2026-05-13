---
description: "Use when modifying DeepEP kernel files under csrc/kernels during CUDA-to-XPU migration, backend decoupling, or synchronization/IPC refactors. Enforces orchestration-vs-backend boundaries and migration-safe ordering rules."
name: "DeepEP Kernel Layer Boundary"
applyTo: "csrc/kernels/**"
---
# DeepEP Kernel Layer Boundary

Keep kernel migration changes aligned with the 4-layer contract in [copilot-instructions](../copilot-instructions.md).

## Boundary rules

- Do not introduce raw backend intrinsics in orchestration files:
  - Allowed orchestration files: `csrc/kernels/api.cuh`, `csrc/kernels/runtime.cu`, `csrc/kernels/layout.cu`, `csrc/kernels/launch.cuh`
  - Disallow new inline PTX (`asm`, `asm volatile`) and backend-specific memory-order primitives in these files.
- Keep backend-specific primitives and transport details in backend/device files:
  - `csrc/kernels/utils.cuh`, `csrc/kernels/intranode.cu`, `csrc/kernels/internode.cu`, `csrc/kernels/internode_ll.cu`
- Keep layout and metadata logic backend-agnostic:
  - Prefer algorithmic logic in `layout.cu` with no transport-specific assumptions.

## Migration requirements

- Primitive-first migration:
  - First introduce or reuse backend helper APIs for fences, atomics, barriers, and noncoherent/volatile loads.
  - Then migrate call sites in kernels and orchestration to those helpers.
- Preserve behavior switches and contracts:
  - Keep behavior compatible with `DISABLE_SM90_FEATURES`, `DISABLE_AGGRESSIVE_PTX_INSTRS`, and `TOPK_IDX_BITS`.
- Preserve dispatch/combine semantics and handle reuse expectations.

## Review checklist for kernel edits

- Layer classification is explicit for every touched file.
- No new backend-intrinsic leakage into orchestration files.
- Memory-order intent is documented where acquire/release/fence semantics are changed.
- Synchronization replacements preserve happens-before guarantees.
- If grid-wide sync is replaced, split-kernel sequencing and completion assumptions are explicitly validated.

## Validation expectations

- Run relevant script-level tests for touched paths:
  - `python tests/test_intranode.py --num-processes 2`
  - `python tests/test_internode.py --num-processes 2` (when internode paths are touched)
  - `python tests/test_low_latency.py --num-processes 2` (when low-latency paths are touched)
- In migration plans, include:
  - Layer-by-layer change summary
  - API mapping decisions (CUDA/PTX/NVSHMEM to SYCL/vISA/Level Zero/iSHMEM)
  - Residual risk and follow-up patch list
