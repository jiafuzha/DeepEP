import os
import subprocess
import setuptools
import torch.utils.cpp_extension as torch_cpp_extension

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, SyclExtension


def _purge_stale_sycl_dlink_objects():
    # Torch's SyclExtension device-link step produces build/temp*/**/sycl_dlink.o by
    # combining DeepEP's RDC device objects with the extra iSHMEM device objects we inject
    # via append_sycl_dlink_objects(). Those injected objects are NOT tracked as ninja
    # dependencies, so when only the iSHMEM archive changes ninja reports "no work to do"
    # and the .so keeps a STALE iSHMEM device image. Delete the device-link object(s) so the
    # next ninja build regenerates them against the freshly-extracted iSHMEM objects.
    for obj in Path('build').glob('temp*/**/sycl_dlink.o'):
        try:
            obj.unlink()
        except OSError:
            pass


def assert_ishmem_archive_fresh(archive_path, ishmem_dir):
    # Guard against linking a STALE libishmem.a: the extraction stamp only compares the
    # archive file to itself, so editing/pulling iSHMEM source without rebuilding+installing
    # would silently link the old archive. Fail loudly (with a rebuild hint) when the
    # installed archive is older than the iSHMEM source tree. Bypass: DEEP_EP_ALLOW_STALE_ISHMEM=1.
    if os.getenv('DEEP_EP_ALLOW_STALE_ISHMEM', '0') == '1':
        return
    src_env = os.getenv('ISHMEM_SRC_DIR')
    # ISHMEM_DIR is conventionally <repo>/build/_install, so the source lives at <repo>/src.
    candidates = [Path(src_env)] if src_env else [Path(ishmem_dir).parent.parent / 'src']
    src_root = next((c for c in candidates if c.is_dir()), None)
    if src_root is None:
        return  # cannot locate the source tree; skip the check
    archive_mtime = Path(archive_path).stat().st_mtime_ns
    newest_mtime, newest_file = 0, None
    for ext in ('*.cpp', '*.cc', '*.c', '*.h', '*.hpp', '*.cuh', '*.cu'):
        for f in src_root.rglob(ext):
            m = f.stat().st_mtime_ns
            if m > newest_mtime:
                newest_mtime, newest_file = m, f
    if newest_file is not None and newest_mtime > archive_mtime:
        ishmem_build_dir = Path(ishmem_dir).parent
        raise RuntimeError(
            f'STALE iSHMEM archive detected -- refusing to link an out-of-date libishmem.a.\n'
            f'  archive     : {archive_path} (mtime_ns={archive_mtime})\n'
            f'  newer source: {newest_file} (mtime_ns={newest_mtime})\n'
            f'The installed libishmem.a is older than the iSHMEM source under {src_root}.\n'
            f'Rebuild iSHMEM first:\n'
            f'  cmake --build {ishmem_build_dir} --target install\n'
            f'then rebuild DeepEP (or just run ./build_xpu.sh which does both).\n'
            f'To bypass this check set DEEP_EP_ALLOW_STALE_ISHMEM=1.')


def check_ishmem_bnxt_inlinable(ishmem_dir):
    # iSHMEM's bnxt IBGDA device helpers in src/ibgda_device_impl.h are marked
    # ISHMEMI_IBGDA_BNXT_NOINLINE, which the iSHMEM CMake defines as
    # __attribute__((noinline)) unless configured with -DISHMEMI_IBGDA_BNXT_NOINLINE=OFF.
    #
    # That macro is a PRIVATE target_compile_definitions on ishmem-objects, and
    # ibgda_device_impl.h is NOT part of the installed include tree, so DeepEP CANNOT
    # influence it from here -- it is baked into libishmem.a at iSHMEM build time.
    #
    # It matters because noinline forces IGC to outline those helpers as vISA stack
    # calls and stamp `.kernel_attr NBarrierCnt=N` on each one, so any DeepEP kernel
    # that uses SPIR-V named barriers alongside iSHMEM dies at module finalization with
    # "More than 1 kernel attribute defined NBarrierCnt" -> "parsing vISA inline
    # assembly failed" at runtime. Detect the bad configuration and say so loudly.
    if os.getenv('DEEP_EP_ALLOW_BNXT_NOINLINE', '0') == '1':
        return
    cache = Path(ishmem_dir).parent / 'CMakeCache.txt'
    if not cache.is_file():
        return  # not a cmake build tree we can introspect; skip
    try:
        for line in cache.read_text(errors='ignore').splitlines():
            if line.startswith('ISHMEMI_IBGDA_BNXT_NOINLINE:'):
                if line.strip().split('=', 1)[-1].strip().upper() in ('ON', 'TRUE', '1', 'YES'):
                    print(
                        '\n' + '=' * 78 + '\n'
                        'WARNING: libishmem.a was built with ISHMEMI_IBGDA_BNXT_NOINLINE=ON.\n'
                        f'  cmake cache: {cache}\n'
                        'The bnxt IBGDA device helpers therefore carry __attribute__((noinline)),\n'
                        'so IGC will outline them as vISA stack calls and stamp NBarrierCnt on each.\n'
                        'Any kernel combining SPIR-V named barriers with iSHMEM will then fail with\n'
                        '  "More than 1 kernel attribute defined NBarrierCnt"\n'
                        '  -> runtime: "error: parsing vISA inline assembly failed"\n'
                        'This CANNOT be fixed from DeepEP: the macro is PRIVATE to the iSHMEM build\n'
                        'and ibgda_device_impl.h is not installed. Rebuild iSHMEM with\n'
                        '  -DISHMEMI_IBGDA_BNXT_NOINLINE=OFF   (e.g. bash _build_ishmem.sh)\n'
                        'then remove build/ishmem-sycl-dlink and rebuild DeepEP.\n'
                        'Silence this check with DEEP_EP_ALLOW_BNXT_NOINLINE=1.\n' + '=' * 78 + '\n')
                return
    except OSError:
        return


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
    # The archive changed: force the device-link object to be rebuilt (see docstring above).
    _purge_stale_sycl_dlink_objects()
    return [str(path) for path in objects]


def append_sycl_dlink_objects(object_paths):
    original_get_sycl_device_flags = torch_cpp_extension._get_sycl_device_flags

    def patched_get_sycl_device_flags(cflags):
        return original_get_sycl_device_flags(cflags) + object_paths

    torch_cpp_extension._get_sycl_device_flags = patched_get_sycl_device_flags


def register_cpp_sycl_sources(source_paths):
    sycl_source_paths = {str(Path(path).resolve()) for path in source_paths}
    original_is_sycl_file = torch_cpp_extension._is_sycl_file

    def patched_is_sycl_file(path):
        return original_is_sycl_file(path) or str(Path(path).resolve()) in sycl_source_paths

    torch_cpp_extension._is_sycl_file = patched_is_sycl_file


if __name__ == '__main__':
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
    sycl_flags = ['-O3', '-fsycl', '-fsycl-rdc', '-fsycl-targets=spir64', '-DDEEP_EP_XPU']
    sycl_sources = [
        'csrc/xpu/layout.cpp',
        'csrc/xpu/intranode.cpp',
        'csrc/xpu/internode.cpp',
        'csrc/xpu/internode_ll.cpp',
    ]
    register_cpp_sycl_sources(sycl_sources)
    sources = [
        'csrc/xpu/deep_ep_xpu.cpp',
        *sycl_sources,
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
            extra_link_args.append('-lhwloc')
            ishmem_archive = Path(ishmem_dir) / 'lib' / 'libishmem.a'
            if ishmem_archive.exists():
                assert_ishmem_archive_fresh(ishmem_archive, ishmem_dir)
                check_ishmem_bnxt_inlinable(ishmem_dir)
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

    if os.getenv('DEEP_EP_COMBINE_TELEMETRY', '0') == '1':
        # Diagnostic-only build: per-warp-role cycle attribution inside the fused
        # internode combine kernel (stall vs copy). Never enable for production.
        cxx_flags.append('-DDEEP_EP_COMBINE_TELEMETRY')
        sycl_flags.append('-DDEEP_EP_COMBINE_TELEMETRY')

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
