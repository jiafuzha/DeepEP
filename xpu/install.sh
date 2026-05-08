#!/usr/bin/env bash
set -euo pipefail

original_dir=$(pwd)
script_dir=$(dirname "$0")
cd "$script_dir"

rm -rf dist
export DEEPEP_XPU_BUILD_MODE="${DEEPEP_XPU_BUILD_MODE:-stub}"
export DEEPEP_XPU_NATIVE_STAGED_STRICT="${DEEPEP_XPU_NATIVE_STAGED_STRICT:-0}"
python setup.py bdist_wheel
pip install dist/*.whl

cd "$original_dir"
