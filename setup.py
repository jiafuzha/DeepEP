import importlib
import os
import subprocess

import setuptools
import torch

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension

try:
    from torch.utils.cpp_extension import SyclExtension
except ImportError:
    SyclExtension = None


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


def detect_backend():
    forced_backend = os.getenv('DEEPEP_BACKEND')
    if forced_backend is not None:
        return forced_backend
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    return 'cpu'


def get_cuda_extension():
    from torch.utils.cpp_extension import CUDAExtension

    disable_nvshmem = False
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'
    if nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem  # noqa: F401
        except (ModuleNotFoundError, AttributeError, IndexError):
            print(
                'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n'
            )
            disable_nvshmem = True
    else:
        disable_nvshmem = False

    if not disable_nvshmem:
        assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
    include_dirs = ['csrc/']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    if disable_nvshmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
    else:
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        library_dirs.extend([f'{nvshmem_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
        extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')
        assert disable_nvshmem
    else:
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    print('Build summary:')
    print(' > Backend: cuda')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    return [CUDAExtension(name='deep_ep_cpp',
                          include_dirs=include_dirs,
                          library_dirs=library_dirs,
                          sources=sources,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args)]


def get_xpu_extension():
    assert SyclExtension is not None, 'PyTorch SyclExtension is unavailable in this environment'
    ishmem_dir = os.getenv('ISHMEM_DIR', '/opt/intel/ishmem')
    assert os.path.exists(ishmem_dir), f'The specified iSHMEM directory does not exist: {ishmem_dir}'

    cxx_flags = [
        '-O3',
        '-fsycl',
        '-Wno-deprecated-declarations',
        '-Wno-unused-variable',
        '-Wno-sign-compare',
        '-Wno-reorder',
        '-Wno-attributes',
        '-DDEEPEP_XPU_NATIVE',
        '-DDEEPEP_USE_ISHMEM',
        '-DDISABLE_NVSHMEM',
        '-DDISABLE_SM90_FEATURES',
        '-DDISABLE_AGGRESSIVE_PTX_INSTRS',
    ]
    original_sources = [
        'csrc/xpu_native_module.cpp',
        'csrc/xpu_native_runtime.cpp',
        'csrc/xpu_native_stubs.cpp',
        'csrc/kernels/layout.cu',
        'csrc/kernels/internode_ll_xpu.cpp',
    ]
    generated_source_dir = Path('build/xpu_native_sources')
    generated_source_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for source in original_sources:
        if source.endswith('.cu'):
            source_path = Path(source).resolve()
            generated_path = generated_source_dir / f'{source_path.stem}_sycl.cpp'
            generated_path.write_text(f'#include "{source_path}"\n')
            sources.append(str(generated_path))
        else:
            sources.append(source)
    include_dirs = ['csrc/', f'{ishmem_dir}/include']
    library_dirs = [f'{ishmem_dir}/lib']
    torch_lib_dir = str(Path(torch.__file__).resolve().parent / 'lib')
    extra_link_args = [f'-L{ishmem_dir}/lib', '-lishmem', '-lze_loader', f'-Wl,-rpath,{ishmem_dir}/lib', f'-Wl,-rpath,{torch_lib_dir}']

    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    print('Build summary:')
    print(' > Backend: xpu')
    print(' > Native XPU extension build: enabled')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {cxx_flags}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > iSHMEM path: {ishmem_dir}')
    print()

    return [SyclExtension(name='deep_ep_cpp',
                          include_dirs=include_dirs,
                          library_dirs=library_dirs,
                          sources=sources,
                          extra_compile_args={'cxx': cxx_flags},
                          extra_link_args=extra_link_args)]


if __name__ == '__main__':
    backend = detect_backend()
    ext_modules = []

    if backend == 'cuda':
        ext_modules = get_cuda_extension()
    elif backend == 'xpu':
        if int(os.getenv('DEEPEP_BUILD_XPU_NATIVE', '0')):
            ext_modules = get_xpu_extension()
        else:
            print('Build summary:')
            print(' > Backend: xpu')
            print(' > Using the Python intranode fallback backend and XPU capability shim for Intel XPU bring-up')
            print(' > Set DEEPEP_BUILD_XPU_NATIVE=1 to attempt a native XPU extension build')
            print()
    else:
        print('Build summary:')
        print(f' > Backend: {backend}')
        print(' > No supported accelerator backend detected, packaging Python sources only')
        print()

    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(name='deep_ep',
                     version='1.2.1' + revision,
                     packages=setuptools.find_packages(include=['deep_ep']),
                     ext_modules=ext_modules,
                     cmdclass={'build_ext': BuildExtension})
