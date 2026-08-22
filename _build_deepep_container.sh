#!/bin/bash
# Build DeepEP INSIDE the docker-2node-v2 container, which carries the same
# oneAPI 2025.3 the sims run (the host oneAPI is 2026.0 and has no torch).
set -e
echo "START $(date)"
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export PIP_BREAK_SYSTEM_PACKAGES=1
export DEEP_EP_TARGET=xpu
export DEEP_EP_COMBINE_TELEMETRY=${DEEP_EP_COMBINE_TELEMETRY:-0}
export ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install
unset TORCH_XPU_ARCH_LIST XPU_AOT_TARGETS
cd /root/jiafuzha/code-repo/zjf2012/DeepEP
# setuptools tracks only the .cpp mtimes, NOT the .inc files they include, so a
# pure-.inc edit is silently NOT recompiled.  Touch the translation units.
touch csrc/xpu/*.cpp
rm -rf build/ishmem-sycl-dlink
python3 setup.py build_ext --inplace
echo "END $(date)"
