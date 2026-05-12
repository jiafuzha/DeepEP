import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import setuptools
from torch.utils.cpp_extension import BuildExtension, SyclExtension, library_paths

DEFAULT_XPU_ARCH = 'pvc'
DEFAULT_BUILD_MODE = 'intranode'
EXPERIMENTAL_BUILD_ENV = 'DEEP_EP_XPU_ALLOW_EXPERIMENTAL_BUILD'
BUILD_MODE_ENV = 'DEEP_EP_XPU_BUILD_MODE'
COMPILE_COMMANDS = {
    'bdist_wheel',
    'build',
    'build_ext',
    'develop',
    'editable_wheel',
    'install',
}
KNOWN_CAVEATS = (
    'XPU support in xpu/ is still experimental.',
    'The intranode runtime is the primary exercised execution path today.',
    'Intranode validation still depends on a peer-access-capable local XPU topology; not every shared XPU host is suitable.',
    'DEEP_EP_XPU_BUILD_MODE=full builds/imports the mirrored internode sources, but low-latency APIs remain explicit unsupported stubs.',
)


@dataclass(frozen=True)
class BuildPlan:
    mode: str
    description: str
    sources: tuple[str, ...]
    requires_ishmem: bool
    disable_ishmem: bool


def env_enabled(name: str, default: str = '0') -> bool:
    return os.getenv(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


def normalize_build_mode() -> str:
    mode = os.getenv(BUILD_MODE_ENV, DEFAULT_BUILD_MODE).strip().lower()
    if mode not in ('intranode', 'full'):
        raise SystemExit(f'Unsupported {BUILD_MODE_ENV}={mode!r}. Supported values: intranode, full.')
    return mode


def get_build_plan() -> BuildPlan:
    mode = normalize_build_mode()
    common_sources = (
        'csrc/deep_ep.cpp.dp.cpp',
        'csrc/kernels/runtime.dp.cpp',
        'csrc/kernels/layout.dp.cpp',
        'csrc/kernels/intranode.dp.cpp',
    )
    if mode == 'full':
        return BuildPlan(
            mode='full',
            description='Experimental full XPU tree (intranode + internode + explicit unsupported low-latency surface)',
            sources=common_sources + (
                'csrc/kernels/internode.dp.cpp',
                'csrc/kernels/internode_ll.dp.cpp',
            ),
            requires_ishmem=True,
            disable_ishmem=False,
        )
    return BuildPlan(
        mode='intranode',
        description='Experimental intranode-only XPU tree',
        sources=common_sources,
        requires_ishmem=False,
        disable_ishmem=True,
    )


def get_ishmem_dir(required: bool) -> str | None:
    ishmem_dir = os.getenv('ISHMEM_DIR', '/opt/intel/ishmem')
    if required and not os.path.exists(ishmem_dir):
        raise SystemExit(f'ISHMEM_DIR points to a missing path: {ishmem_dir}\n'
                         'Set ISHMEM_DIR to a valid iSHMEM installation before attempting a full XPU build.')
    return ishmem_dir if os.path.exists(ishmem_dir) else None


def detect_environment_issues(plan: BuildPlan) -> list[str]:
    issues = []

    if not os.getenv('CC'):
        issues.append('CC is not set (expected: icx).')
    elif os.getenv('CC') != 'icx':
        issues.append(f'CC={os.getenv("CC")} (expected: icx).')

    if not os.getenv('CXX'):
        issues.append('CXX is not set (expected: icpx).')
    elif os.getenv('CXX') != 'icpx':
        issues.append(f'CXX={os.getenv("CXX")} (expected: icpx).')

    if not (os.getenv('ONEAPI_ROOT') or os.getenv('CMPLR_ROOT')):
        issues.append('oneAPI environment is not loaded (source /opt/intel/oneapi/setvars.sh --force).')

    if shutil.which('icx') is None:
        issues.append('icx is not available on PATH.')

    if shutil.which('icpx') is None:
        issues.append('icpx is not available on PATH.')

    if plan.requires_ishmem:
        ishmem_dir = os.getenv('ISHMEM_DIR', '/opt/intel/ishmem')
        if not os.path.exists(ishmem_dir):
            issues.append(f'ISHMEM_DIR does not exist: {ishmem_dir}.')

    return issues


def format_status(plan: BuildPlan, issues: list[str]) -> str:
    ishmem_dir = os.getenv('ISHMEM_DIR', '/opt/intel/ishmem')
    lines = [
        'DeepEP XPU build status',
        f' - Mode: {plan.mode}',
        f' - Description: {plan.description}',
        f' - Sources: {", ".join(plan.sources)}',
        f' - TORCH_XPU_ARCH_LIST: {os.getenv("TORCH_XPU_ARCH_LIST", DEFAULT_XPU_ARCH)}',
        f' - CC: {os.getenv("CC", "<unset>")}',
        f' - CXX: {os.getenv("CXX", "<unset>")}',
        f' - oneAPI loaded: {"yes" if (os.getenv("ONEAPI_ROOT") or os.getenv("CMPLR_ROOT")) else "no"}',
        f' - ISHMEM_DIR: {ishmem_dir}{" (required)" if plan.requires_ishmem else " (optional in this mode)"}',
        f' - Experimental build gate ({EXPERIMENTAL_BUILD_ENV}): {os.getenv(EXPERIMENTAL_BUILD_ENV, "0")}',
        ' - Current caveats:',
    ]
    lines.extend(f'   * {caveat}' for caveat in KNOWN_CAVEATS)
    if issues:
        lines.append(' - Environment issues:')
        lines.extend(f'   * {issue}' for issue in issues)
    else:
        lines.append(' - Environment checks: OK for an experimental compile attempt.')
    lines.append(' - Notes:')
    lines.append(f'   * Build commands are blocked by default; set {EXPERIMENTAL_BUILD_ENV}=1 to attempt compilation.')
    lines.append(f'   * {BUILD_MODE_ENV}=intranode skips iSHMEM-linked internode/low-latency sources.')
    lines.append(
        f'   * {BUILD_MODE_ENV}=full enables the mirrored full tree, requires ISHMEM_DIR, and keeps low-latency entrypoints as explicit unsupported stubs.'
    )
    return '\n'.join(lines)


def print_status(plan: BuildPlan, issues: list[str]) -> None:
    print(format_status(plan, issues))
    print()


class XpuBuildInfoCommand(setuptools.Command):
    description = 'print DeepEP XPU build status, requirements, and current caveats'
    user_options: list[tuple[str, str | None, str]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        plan = get_build_plan()
        issues = detect_environment_issues(plan)
        print_status(plan, issues)


def should_compile(argv: list[str]) -> bool:
    return any(arg in COMPILE_COMMANDS for arg in argv[1:])


def guard_build_request(plan: BuildPlan, issues: list[str]) -> None:
    if not should_compile(sys.argv):
        return

    print_status(plan, issues)

    if not env_enabled(EXPERIMENTAL_BUILD_ENV):
        raise SystemExit('DeepEP XPU extension compilation is intentionally gated because the mirrored XPU tree '
                         'is still experimental.\n'
                         f'Set {EXPERIMENTAL_BUILD_ENV}=1 to attempt the current experimental build anyway.')

    if issues:
        raise SystemExit('DeepEP XPU experimental build requested, but required toolchain/environment checks failed.\n'
                         'Resolve the issues listed above and retry.')


def build_extension(plan: BuildPlan) -> SyclExtension:
    ishmem_dir = get_ishmem_dir(plan.requires_ishmem)
    torch_lib_dirs = tuple(library_paths())
    os.environ['TORCH_XPU_ARCH_LIST'] = os.getenv('TORCH_XPU_ARCH_LIST', DEFAULT_XPU_ARCH)

    cxx_flags = [
        '-O3',
        '-fsycl',
        '-fPIC',
        '-std=c++20',
        '-Wno-deprecated-declarations',
        '-Wno-unused-variable',
        '-Wno-sign-compare',
        '-Wno-reorder',
        '-Wno-attributes',
    ]
    include_dirs = ['csrc/']
    library_dirs = []
    extra_link_args = ['-lze_loader']

    for torch_lib_dir in torch_lib_dirs:
        if torch_lib_dir not in library_dirs:
            library_dirs.append(torch_lib_dir)
        extra_link_args.append(f'-Wl,-rpath,{torch_lib_dir}')

    if plan.disable_ishmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
    else:
        include_dirs.append(f'{ishmem_dir}/include')
        library_dirs.append(f'{ishmem_dir}/lib')
        extra_link_args.extend([f'-Wl,-rpath,{ishmem_dir}/lib', '-lishmem'])

    if 'TOPK_IDX_BITS' in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    print('Build summary:')
    print(f' > Mode: {plan.mode}')
    print(f' > Sources: {list(plan.sources)}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {{"cxx": {cxx_flags}}}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_XPU_ARCH_LIST"]}')
    print(f' > iSHMEM path: {ishmem_dir if ishmem_dir is not None else "<not used>"}')
    print()

    return SyclExtension(
        name='deep_ep_xpu_cpp',
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        sources=list(plan.sources),
        extra_compile_args={'cxx': cxx_flags},
        extra_link_args=extra_link_args,
    )


def get_revision_suffix() -> str:
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        return '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception:
        return ''


if __name__ == '__main__':
    plan = get_build_plan()
    issues = detect_environment_issues(plan)
    guard_build_request(plan, issues)
    ext_modules = [build_extension(plan)] if should_compile(sys.argv) else []

    setuptools.setup(
        name='deep_ep_xpu',
        version='1.2.1' + get_revision_suffix(),
        packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),
        ext_modules=ext_modules,
        cmdclass={
            'build_ext': BuildExtension,
            'xpu_build_info': XpuBuildInfoCommand,
        },
    )
