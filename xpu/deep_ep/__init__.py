import torch
from typing import Any, Optional

from .utils import EventOverlap
from .buffer import Buffer

_BACKEND_IMPORT_ERROR: Optional[Exception] = None

try:
    # noinspection PyUnresolvedReferences
    from deep_ep_xpu_cpp import Config, topk_idx_t
except Exception as exc:  # pragma: no cover - exercised when the native extension is unavailable
    _BACKEND_IMPORT_ERROR = exc
    topk_idx_t = torch.int64

    def _raise_backend_unavailable() -> None:
        raise ModuleNotFoundError(
            "deep_ep_xpu_cpp is not available. The Python XPU wrapper can be imported, "
            "but Buffer/Config operations require the native extension to be built and importable.") from _BACKEND_IMPORT_ERROR

    class Config:  # type: ignore[no-redef]
        """Compatibility placeholder until `deep_ep_xpu_cpp.Config` is available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_backend_unavailable()


__all__ = ['Buffer', 'Config', 'EventOverlap', 'topk_idx_t']
