import glob
import importlib
import os
import shutil
import subprocess
import tempfile
import sys
from typing import Tuple
from pathlib import Path

import setuptools


def _get_pybind11_cmake_dir() -> str:
    try:
        import pybind11

        return pybind11.get_cmake_dir()
    except Exception:
        return ''


def _get_torch_cmake_prefix_path() -> str:
    try:
        import torch.utils

        return torch.utils.cmake_prefix_path
    except Exception:
        return ''


if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception:
        revision = ''

    setup_kwargs = {
        'name': 'deep_ep_xpu',
        'version': '0.1.0' + revision,
        'packages': ['xpu', 'xpu.deep_ep'],
        'package_dir': {'xpu': '.'},
        'py_modules': ['deep_ep_cpp_xpu'],
        'description': 'Staged XPU migration package for DeepEP',
    }
    build_mode = os.environ.get('DEEPEP_XPU_BUILD_MODE', 'stub').strip().lower()
    if build_mode not in ('stub', 'native-staged'):
        raise RuntimeError(f'Unsupported DEEPEP_XPU_BUILD_MODE={build_mode}; expected stub or native-staged')
    strict_native = os.environ.get('DEEPEP_XPU_NATIVE_STAGED_STRICT', '0').strip().lower() in ('1', 'true', 'yes', 'on')

    def maybe_fallback_to_stub(current_mode: str, reason: str) -> str:
        if current_mode != 'native-staged':
            return current_mode
        if strict_native:
            raise RuntimeError(
                f'native-staged build was requested and strict mode is enabled: {reason}. '
                'Unset DEEPEP_XPU_NATIVE_STAGED_STRICT or use DEEPEP_XPU_BUILD_MODE=stub.'
            )
        print(f'[deep_ep_xpu] native-staged unavailable ({reason}); falling back to stub mode', file=sys.stderr)
        return 'stub'

    def native_staged_preflight() -> Tuple[bool, str]:
        project_root = Path(__file__).resolve().parent
        cmake_src = project_root / 'csrc'
        if not cmake_src.exists():
            return False, f'missing CMake source directory: {cmake_src}'

        pybind11_cmake_dir = _get_pybind11_cmake_dir()
        torch_cmake_prefix_path = _get_torch_cmake_prefix_path()

        try:
            with tempfile.TemporaryDirectory(prefix='deep_ep_xpu_cmake_preflight_') as tmpdir:
                build_dir = Path(tmpdir)
                configure_cmd = [
                    'cmake',
                    '-S', str(cmake_src),
                    '-B', str(build_dir),
                    '-DDEEPEP_USE_XPU=ON',
                    '-DDEEPEP_XPU_NATIVE_STAGED=ON',
                    f'-DPython_EXECUTABLE={sys.executable}',
                ]
                if pybind11_cmake_dir:
                    configure_cmd.append(f'-Dpybind11_DIR={pybind11_cmake_dir}')
                if torch_cmake_prefix_path:
                    configure_cmd.append(f'-DCMAKE_PREFIX_PATH={torch_cmake_prefix_path}')
                subprocess.check_call(configure_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            return False, str(exc)
        return True, ''

    torch_spec = importlib.util.find_spec('torch')
    if build_mode == 'native-staged' and shutil.which('cmake') is None:
        build_mode = maybe_fallback_to_stub(build_mode, 'cmake not found in PATH')

    if build_mode == 'native-staged' and torch_spec is None:
        build_mode = maybe_fallback_to_stub(build_mode, 'torch is not installed')

    if build_mode == 'native-staged':
        preflight_ok, preflight_reason = native_staged_preflight()
        if not preflight_ok:
            build_mode = maybe_fallback_to_stub(build_mode, f'cmake preflight failed: {preflight_reason}')

    if torch_spec is not None and build_mode == 'stub':
        from torch.utils.cpp_extension import CppExtension, BuildExtension

        setup_kwargs['ext_modules'] = [
            CppExtension(
                name='deep_ep_cpp_xpu',
                sources=['csrc/deep_ep_xpu_stub.cpp'],
                include_dirs=['csrc'],
                extra_compile_args=['-O2'],
            )
        ]
        setup_kwargs['cmdclass'] = {'build_ext': BuildExtension}
    elif torch_spec is not None and build_mode == 'native-staged':
        from setuptools import Extension
        from setuptools.command.build_ext import build_ext

        class CMakeStagedBuild(build_ext):
            def build_extension(self, ext):
                ext_path = Path(self.get_ext_fullpath(ext.name)).resolve()
                ext_path.parent.mkdir(parents=True, exist_ok=True)

                project_root = Path(__file__).resolve().parent
                cmake_src = project_root / 'csrc'
                cmake_build = Path(self.build_temp).resolve() / 'deep_ep_xpu_native'
                cmake_build.mkdir(parents=True, exist_ok=True)
                pybind11_cmake_dir = _get_pybind11_cmake_dir()
                torch_cmake_prefix_path = _get_torch_cmake_prefix_path()

                configure_cmd = [
                    'cmake',
                    '-S', str(cmake_src),
                    '-B', str(cmake_build),
                    '-DDEEPEP_USE_XPU=ON',
                    '-DDEEPEP_XPU_NATIVE_STAGED=ON',
                    f'-DPython_EXECUTABLE={sys.executable}',
                ]
                if pybind11_cmake_dir:
                    configure_cmd.append(f'-Dpybind11_DIR={pybind11_cmake_dir}')
                if torch_cmake_prefix_path:
                    configure_cmd.append(f'-DCMAKE_PREFIX_PATH={torch_cmake_prefix_path}')
                build_cmd = ['cmake', '--build', str(cmake_build), '--config', 'Release']

                subprocess.check_call(configure_cmd)
                subprocess.check_call(build_cmd)

                candidates = sorted(glob.glob(str(cmake_build / 'deep_ep_cpp_xpu*.so')))
                if not candidates:
                    raise RuntimeError('CMake native-staged build finished but deep_ep_cpp_xpu*.so was not produced')
                shutil.copy2(candidates[0], ext_path)

        setup_kwargs['ext_modules'] = [Extension('deep_ep_cpp_xpu', sources=[])]
        setup_kwargs['cmdclass'] = {'build_ext': CMakeStagedBuild}

    setuptools.setup(**setup_kwargs)
