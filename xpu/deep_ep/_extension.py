import importlib
from types import ModuleType
from typing import Tuple


def load_extension() -> Tuple[ModuleType, str]:
    """Load XPU extension first, then fallback to CUDA extension during migration."""
    for name in ("xpu.deep_ep_cpp_xpu", "deep_ep_cpp_xpu", "deep_ep_cpp"):
        try:
            return importlib.import_module(name), name
        except ImportError:
            continue
    raise ImportError("Cannot import deep_ep extension module. Expected deep_ep_cpp_xpu or deep_ep_cpp")
