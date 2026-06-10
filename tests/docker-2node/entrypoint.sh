#!/bin/bash
# Source oneAPI and activate conda env, then exec the command
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate jiafuzha_deepep 2>/dev/null || true
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
exec "$@"
