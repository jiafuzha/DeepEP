# DeepEP Copilot Instructions

## Build, test, and lint commands

### Build and install

- Main CUDA package:
  - `NVSHMEM_DIR=/path/to/nvshmem python setup.py build`
  - `NVSHMEM_DIR=/path/to/nvshmem python setup.py install`
  - `./install.sh` builds a wheel from the repo root and installs it.
- Metadata-only install for non-CUDA environments:
  - `DEEPEP_SKIP_CUDA_BUILD=1 python -m pip install -e . --no-build-isolation`
- Staged XPU package:
  - `DEEPEP_BACKEND=xpu ./install.sh`
  - `cd xpu && ./install.sh`
  - `DEEPEP_XPU_BUILD_MODE=native-staged` enables the CMake-backed staged native build; otherwise XPU defaults to the stub build and may fall back to it automatically.

### Lint and formatting

- `bash ./format.sh` formats and lints files changed relative to `main`/`origin/main`
- `bash ./format.sh --all` runs the full repo pass
- CI uses `bash ./format.sh`
- Tool versions are pinned in `requirements-lint.txt` (`yapf`, `ruff`, optional `clang-format`)

### Tests

- The main CUDA tests are executable scripts, not pytest suites. They spawn ranks themselves:
  - `python tests/test_intranode.py`
  - `python tests/test_internode.py`
  - `python tests/test_low_latency.py`
- Run one CUDA test suite with custom arguments by invoking the script directly, for example:
  - `python tests/test_intranode.py --num-processes 8 --num-tokens 4096 --hidden 7168`
- The README expectation is that `tests/utils.py:init_dist()` may need to be edited for local cluster settings before running the distributed CUDA tests.
- The staged XPU tests use pytest:
  - `python -m pytest xpu/tests -q`
  - Single test example: `python -m pytest xpu/tests/test_xpu_force_python_stub.py::test_force_python_stub_entrypoint_smoke -q`

## High-level architecture

- `deep_ep/` is a thin Python API over the native extension. `deep_ep/__init__.py` mainly re-exports `Buffer`, `EventOverlap`, `Config`, and `topk_idx_t`.
- `deep_ep/buffer.py` is the main control layer. It bootstraps communication state, synchronizes device IDs / IPC handles / NVSHMEM IDs, selects intranode vs internode paths, and exposes the public dispatch/combine/low-latency APIs.
- `csrc/deep_ep.cpp` and `csrc/deep_ep.hpp` are the pybind/runtime bridge. They own buffer allocation, shared-memory/IPC setup, host counters, workspace management, and the communication stream that the Python `Buffer` wraps.
- `csrc/kernels/layout.cu` computes the dispatch metadata first; `intranode.cu`, `internode.cu`, and `internode_ll.cu` implement the three kernel families; `runtime.cu` handles runtime services and buffer lifecycle.
- The repo is organized around three communication modes:
  - intranode high-throughput: NVLink
  - internode high-throughput: NVLink + RDMA/NVSHMEM
  - low-latency decoding: RDMA-focused path with hook-based overlap support
- `tests/utils.py` is part of the operational architecture, not just test glue: it owns distributed bootstrap, FP8 helpers, benchmarking helpers, and the cluster-specific `init_dist()` setup used by the standalone CUDA regression scripts.
- `xpu/` is a staged migration workspace that mirrors the CUDA layout (`xpu/deep_ep`, `xpu/csrc`) rather than introducing a separate architecture. Read it as a file-for-file migration track.
- `xpu/deep_ep/_extension.py` prefers the XPU extension, then falls back to `deep_ep_cpp_xpu` or `deep_ep_cpp`; `xpu/deep_ep/buffer.py` keeps the public API aligned while adding staged Python fallbacks for unsupported XPU paths.

## Key conventions

- `Buffer` is the center of the API surface. The intended flow is `get_dispatch_layout(...)` -> `dispatch(...)` -> `combine(...)`, and the handle returned from `dispatch` is the cache key for later cached dispatch/combine calls.
- Do not hard-code the expert-index dtype; use `deep_ep.topk_idx_t` (or the XPU package equivalent). The native module exports the supported dtype.
- High-throughput and low-latency paths have different buffer assumptions. If normal kernels have dirtied the low-latency buffers, call `clean_low_latency_buffer(...)` before using low-latency APIs.
- Performance tuning is encoded in `Buffer.get_dispatch_config()` and `Buffer.get_combine_config()` as rank-count-specific config maps. The standalone test scripts are the practical place where those configs are exercised and tuned.
- Async APIs use `EventOverlap`, and low-latency APIs may additionally return receive hooks. Preserve that event/hook contract instead of replacing it with ad hoc stream waits.
- `Buffer.set_num_sms()` is a real tuning control, not a cosmetic setter. CUDA code requires an even SM count; the staged XPU path also auto-detects SM/compute-unit count and allows env overrides.
- `explicitly_destroy=True` changes lifetime management: callers are then responsible for calling `destroy()`.
- XPU work is intentionally staged. Many XPU fallbacks in `xpu/deep_ep/buffer.py` only support single-rank/local degradation paths; do not assume feature parity with CUDA internode or low-latency transport just because the method exists.
- Keep CUDA and XPU changes structurally mirrored when touching shared surfaces. The XPU tree is meant to stay comparable to the original CUDA tree during migration.
- Environment variables are part of normal repo behavior, especially `NVSHMEM_DIR`, `DISABLE_SM90_FEATURES`, `DISABLE_AGGRESSIVE_PTX_INSTRS`, `DEEPEP_SKIP_CUDA_BUILD`, `DEEPEP_BACKEND`, `DEEPEP_XPU_BUILD_MODE`, `DEEPEP_XPU_NATIVE_STAGED_STRICT`, `DEEPEP_XPU_FORCE_PY_STUB`, `DEEPEP_NUM_SMS`, and `DEEPEP_XPU_CONFIG_STRATEGY`.
