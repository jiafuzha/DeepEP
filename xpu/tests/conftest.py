import importlib
import os
import sys
from types import ModuleType

import pytest
import torch


def _purge_xpu_python_stub_modules() -> None:
    targets = []
    for name in list(sys.modules):
        if name == 'xpu.deep_ep' or name.startswith('xpu.deep_ep.'):
            targets.append(name)
        elif name == 'xpu.deep_ep_cpp_xpu' or name.startswith('xpu.deep_ep_cpp_xpu'):
            targets.append(name)
    for name in targets:
        sys.modules.pop(name, None)


@pytest.fixture
def forced_stub_backend() -> tuple[ModuleType, ModuleType]:
    old_force_value = os.environ.get('DEEPEP_XPU_FORCE_PY_STUB')
    os.environ['DEEPEP_XPU_FORCE_PY_STUB'] = '1'

    _purge_xpu_python_stub_modules()
    buffer_mod = importlib.import_module('xpu.deep_ep.buffer')
    ext = buffer_mod._EXT
    assert buffer_mod._EXT_NAME == 'xpu.deep_ep_cpp_xpu.py-stub'

    try:
        yield buffer_mod, ext
    finally:
        if old_force_value is None:
            os.environ.pop('DEEPEP_XPU_FORCE_PY_STUB', None)
        else:
            os.environ['DEEPEP_XPU_FORCE_PY_STUB'] = old_force_value
        _purge_xpu_python_stub_modules()


@pytest.fixture(scope='session')
def warmed_xpu_allocator() -> None:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.set_device(0)
        _ = torch.empty((1,), device='xpu')


@pytest.fixture(scope='session')
def xpu_stub_backend_module(warmed_xpu_allocator) -> ModuleType:
    del warmed_xpu_allocator
    return importlib.import_module('xpu.deep_ep_cpp_xpu')


@pytest.fixture(scope='session')
def native_xpu_extension_info(warmed_xpu_allocator) -> tuple[ModuleType, str, bool, bool]:
    del warmed_xpu_allocator
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    is_native_backend = ext_path.endswith(('.so', '.pyd'))
    has_xpu_runtime = hasattr(torch, 'xpu') and torch.xpu.is_available()
    return ext, ext_path, is_native_backend, has_xpu_runtime


@pytest.fixture
def require_native_xpu_runtime(native_xpu_extension_info) -> ModuleType:
    ext, _, is_native_backend, has_xpu_runtime = native_xpu_extension_info
    if not is_native_backend:
        pytest.skip('native staged extension is not active')
    if not has_xpu_runtime:
        pytest.skip('xpu runtime is not available')
    return ext
