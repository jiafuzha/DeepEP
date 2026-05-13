---
description: "Use when editing DeepEP kernel primitive files for CUDA-to-XPU migration, especially PTX/fence/atomic/launch behavior in csrc/kernels/utils.cuh and csrc/kernels/launch.cuh."
name: "DeepEP Kernel Primitive Migration"
applyTo: ["csrc/kernels/utils.cuh", "csrc/kernels/launch.cuh"]
---
# DeepEP Kernel Primitive Migration

Use this instruction for low-level primitive and launch-path edits during CUDA to Intel XPU migration.

## File intent boundaries

- [csrc/kernels/utils.cuh](../../csrc/kernels/utils.cuh): backend primitive wrappers only.
- [csrc/kernels/launch.cuh](../../csrc/kernels/launch.cuh): launch/config selection only.
- Do not add routing/layout logic here.
- Do not move orchestration logic from [csrc/kernels/runtime.cu](../../csrc/kernels/runtime.cu) into these files.

## Primitive migration rules

- Prefer helper abstraction over direct intrinsic usage.
- New raw inline PTX is disallowed unless confined to backend helper wrappers and justified by measured need.
- For acquire/release/fence behavior, preserve semantic intent first, then optimize.
- Prefer SYCL memory model mappings (`atomic_ref`, memory order, and scope) before architecture-specific lowering.

## Launch-path migration rules

- Keep launch decision logic backend-neutral at the interface level.
- If CUDA cooperative-grid behavior cannot be matched directly, use explicit split-kernel sequencing and document the completion guarantee.
- Any change to launch attributes must document expected ordering and synchronization effects.

## Required migration notes in PR-ready output

- Original primitive or launch behavior and the target mapping.
- Why the mapping preserves ordering/completion semantics.
- Any weaker/stronger semantics introduced and risk impact.
- Follow-up validation steps if semantics are conditionally equivalent.

## Validation expectations

- If changing fence/atomic/barrier wrappers, run:
  - `python tests/test_intranode.py --num-processes 2`
- If changing internode or low-latency launch/ordering behavior, also run:
  - `python tests/test_internode.py --num-processes 2`
  - `python tests/test_low_latency.py --num-processes 2`
