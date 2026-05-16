import os
import subprocess
import setuptools
import importlib
import torch.utils.cpp_extension as torch_cpp_extension

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, SyclExtension


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


def extract_archive_objects_for_sycl_dlink(archive_path, output_dir):
    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = output_dir / '.archive-stamp'
    archive_state = f'{archive_path}:{archive_path.stat().st_mtime_ns}:{archive_path.stat().st_size}\n'
    objects = sorted(output_dir.glob('*.o'))
    if objects and stamp.exists() and stamp.read_text() == archive_state:
        return [str(path) for path in objects]

    for path in objects:
        path.unlink()
    subprocess.check_call(['ar', 'x', str(archive_path)], cwd=output_dir)
    objects = sorted(output_dir.glob('*.o'))
    if not objects:
        raise RuntimeError(f'No object files were extracted from {archive_path}')
    stamp.write_text(archive_state)
    return [str(path) for path in objects]


def append_sycl_dlink_objects(object_paths):
    original_get_sycl_device_flags = torch_cpp_extension._get_sycl_device_flags

    def patched_get_sycl_device_flags(cflags):
        return original_get_sycl_device_flags(cflags) + object_paths

    torch_cpp_extension._get_sycl_device_flags = patched_get_sycl_device_flags


if __name__ == '__main__':
    target = os.getenv('DEEP_EP_TARGET', 'xpu').lower()
    if target not in ('cuda', 'xpu'):
        raise ValueError(f'Unsupported DEEP_EP_TARGET={target!r}, expected "cuda" or "xpu"')

    if target == 'xpu':
        cxx_flags = [
            '-O3',
            '-Wno-deprecated-declarations',
            '-Wno-unused-variable',
            '-Wno-sign-compare',
            '-Wno-reorder',
            '-Wno-attributes',
            '-DDEEP_EP_XPU',
            '-DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING',
        ]
        sycl_flags = ['-O3', '-fsycl', '-fsycl-rdc', '-DDEEP_EP_XPU']
        sources = [
            'csrc/xpu/deep_ep_xpu.cpp',
            'csrc/xpu/layout.sycl',
            'csrc/xpu/intranode.sycl',
            'csrc/xpu/internode.sycl',
            'csrc/xpu/internode_ll.sycl',
        ]
        include_dirs = [str(Path('csrc').resolve())]
        library_dirs = []
        extra_link_args = ['-lze_loader']
        sycl_dlink_objects = []

        ishmem_dir = os.getenv('ISHMEM_DIR', '/opt/intel/ishmem')
        ishmem_pkg_config = Path(ishmem_dir) / 'lib' / 'pkgconfig'
        if ishmem_pkg_config.exists():
            env = os.environ.copy()
            env['PKG_CONFIG_PATH'] = f'{ishmem_pkg_config}:{env.get("PKG_CONFIG_PATH", "")}'
            try:
                ishmem_cflags = subprocess.check_output(['pkg-config', '--cflags', 'ishmem'], env=env, text=True).split()
                ishmem_libs = subprocess.check_output(['pkg-config', '--libs', 'ishmem'], env=env, text=True).split()
            except (subprocess.CalledProcessError, FileNotFoundError):
                ishmem_cflags, ishmem_libs = [], []
            if ishmem_cflags and ishmem_libs:
                cxx_flags.append('-DDEEP_EP_ENABLE_ISHMEM')
                sycl_flags.append('-DDEEP_EP_ENABLE_ISHMEM')
                for flag in ishmem_cflags:
                    if flag.startswith('-I'):
                        include_dirs.append(flag[2:])
                    else:
                        cxx_flags.append(flag)
                        sycl_flags.append(flag)
                for flag in ishmem_libs:
                    if flag.startswith('-L'):
                        library_dirs.append(flag[2:])
                    else:
                        extra_link_args.append(flag)
                ishmem_archive = Path(ishmem_dir) / 'lib' / 'libishmem.a'
                if ishmem_archive.exists():
                    sycl_dlink_objects = extract_archive_objects_for_sycl_dlink(ishmem_archive, Path('build') / 'ishmem-sycl-dlink')
                    append_sycl_dlink_objects(sycl_dlink_objects)
            else:
                print(f'Warning: iSHMEM pkg-config metadata was found at {ishmem_pkg_config}, but flags could not be resolved')
        else:
            print(f'Warning: iSHMEM was not found at {ishmem_dir}; XPU internode runtime will be disabled')

        if "TOPK_IDX_BITS" in os.environ:
            topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
            if topk_idx_bits not in (32, 64):
                raise ValueError(f'Unsupported TOPK_IDX_BITS={topk_idx_bits}, expected 32 or 64')
            cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
            sycl_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

        extra_compile_args = {
            'cxx': cxx_flags,
            'sycl': sycl_flags,
        }

        print('Build summary:')
        print(' > Target: xpu')
        print(f' > Sources: {sources}')
        print(f' > Includes: {include_dirs}')
        print(f' > Libraries: {library_dirs}')
        print(f' > Compilation flags: {extra_compile_args}')
        print(f' > Link flags: {extra_link_args}')
        print(f' > iSHMEM SYCL device-link objects: {len(sycl_dlink_objects)}')
        print()

        try:
            cmd = ['git', 'rev-parse', '--short', 'HEAD']
            revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
        except Exception as _:
            revision = ''

        setuptools.setup(name='deep_ep',
                         version='1.2.1' + revision,
                         packages=setuptools.find_packages(include=['deep_ep']),
                         ext_modules=[
                             SyclExtension(name='deep_ep_cpp',
                                           include_dirs=include_dirs,
                                           library_dirs=library_dirs,
                                           sources=sources,
                                           extra_compile_args=extra_compile_args,
                                           extra_link_args=extra_link_args)
                         ],
                         cmdclass={'build_ext': BuildExtension})
        raise SystemExit

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

    # NVSHMEM flags
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
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert disable_nvshmem
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(name='deep_ep',
                     version='1.2.1' + revision,
                     packages=setuptools.find_packages(include=['deep_ep']),
                     ext_modules=[
                         CUDAExtension(name='deep_ep_cpp',
                                       include_dirs=include_dirs,
                                       library_dirs=library_dirs,
                                       sources=sources,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                     ],
                     cmdclass={'build_ext': BuildExtension})
