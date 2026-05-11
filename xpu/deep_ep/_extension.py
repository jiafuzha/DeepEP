import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Tuple


def _force_python_stub_enabled() -> bool:
    value = os.environ.get('DEEPEP_XPU_FORCE_PY_STUB', '')
    return value.lower() in ('1', 'true', 'yes', 'on')


def _load_forced_python_stub() -> Tuple[ModuleType, str]:
    package_root = Path(__file__).resolve().parent.parent
    stub_path = package_root / 'deep_ep_cpp_xpu.py'
    if not stub_path.is_file():
        raise ImportError(f'Forced Python stub requested but file is missing: {stub_path}')

    module_name = 'xpu.deep_ep_cpp_xpu_py_stub'
    spec = importlib.util.spec_from_file_location(module_name, stub_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to create module spec for forced Python stub: {stub_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, 'xpu.deep_ep_cpp_xpu.py-stub'


def load_extension() -> Tuple[ModuleType, str]:
    """Load XPU extension first, then fallback to CUDA extension during migration."""
    if _force_python_stub_enabled():
        return _load_forced_python_stub()

    for name in ("xpu.deep_ep_cpp_xpu", "deep_ep_cpp_xpu", "deep_ep_cpp"):
        try:
            return importlib.import_module(name), name
        except ImportError:
            continue
    raise ImportError("Cannot import deep_ep extension module. Expected deep_ep_cpp_xpu or deep_ep_cpp")
