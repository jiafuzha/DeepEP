import torch

from .utils import EventOverlap
from .buffer import Buffer
from ._extension import load_extension

_EXT, _EXT_NAME = load_extension()
Config = _EXT.Config

# Use the extension-defined index type when available.
topk_idx_t = getattr(_EXT, 'topk_idx_t', torch.int64)

__all__ = ['Buffer', 'Config', 'EventOverlap', 'topk_idx_t']
