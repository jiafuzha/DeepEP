#!/bin/bash
set -e
echo "START $(date)"
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export PIP_BREAK_SYSTEM_PACKAGES=1
export DEEP_EP_TARGET=xpu
export ISHMEM_DIR=/root/jiafuzha/ishmem_ibgda/build/_install
unset TORCH_XPU_ARCH_LIST XPU_AOT_TARGETS
cd /root/jiafuzha/code-repo/zjf2012/DeepEP
rm -rf build/ishmem-sycl-dlink build
python3 setup.py build_ext --inplace
echo "EXIT_CODE=$? END $(date)"
