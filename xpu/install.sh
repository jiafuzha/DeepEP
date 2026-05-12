#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./install.sh [--check] [--build] [--experimental] [--mode intranode|full]

Behavior:
  --check         Print the current XPU build status and exit (default).
  --build         Build/install a wheel, then run an import smoke test.
  --experimental  Required together with --build. Opts into the current
                  experimental extension build/install flow.
  --mode          Select the mirrored XPU source set. Default: intranode.

Examples:
  ./install.sh --check
  ./install.sh --build --experimental --mode intranode
  ./install.sh --build --experimental --mode full
EOF
}

action="check"
mode="${DEEP_EP_XPU_BUILD_MODE:-intranode}"
experimental="${DEEP_EP_XPU_ALLOW_EXPERIMENTAL_BUILD:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            action="check"
            ;;
        --build)
            action="build"
            ;;
        --experimental)
            experimental="1"
            ;;
        --mode)
            shift
            if [[ $# -eq 0 ]]; then
                echo "error: --mode requires a value" >&2
                usage
                exit 2
            fi
            mode="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
    shift
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
original_dir=$(pwd)
cd "$script_dir"

if [[ -z "${CMPLR_ROOT:-}" && -f /opt/intel/oneapi/setvars.sh ]]; then
    echo "[DeepEP XPU] Sourcing /opt/intel/oneapi/setvars.sh --force"
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force
fi

export CC="${CC:-icx}"
export CXX="${CXX:-icpx}"
export DEEP_EP_XPU_BUILD_MODE="$mode"
export DEEP_EP_XPU_ALLOW_EXPERIMENTAL_BUILD="$experimental"
export TORCH_XPU_ARCH_LIST="${TORCH_XPU_ARCH_LIST:-pvc}"

if [[ "$mode" == "full" ]]; then
    export ISHMEM_DIR="${ISHMEM_DIR:-/opt/intel/ishmem}"
fi

echo "[DeepEP XPU] Build status probe"
python setup.py xpu_build_info

if [[ "$action" == "check" ]]; then
    cd "$original_dir"
    exit 0
fi

if [[ "$experimental" != "1" ]]; then
    echo
    echo "[DeepEP XPU] Refusing to compile without --experimental."
    echo "The mirrored XPU tree is still experimental, so real builds stay behind an explicit opt-in gate."
    cd "$original_dir"
    exit 2
fi

rm -rf build dist
python setup.py bdist_wheel
pip install dist/*.whl
python - <<'PY'
import deep_ep
import deep_ep_xpu_cpp

print('[DeepEP XPU] Import smoke passed')
print(f'  deep_ep module: {deep_ep.__file__}')
print(f'  deep_ep_xpu_cpp module: {deep_ep_xpu_cpp.__file__}')
PY

cd "$original_dir"
