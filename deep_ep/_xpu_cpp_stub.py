import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    num_sms: int
    nvl_chunk_size: int
    nvl_buffer_size: int
    rdma_chunk_size: int = 0
    rdma_buffer_size: int = 0


class EventHandle:

    def current_stream_wait(self) -> None:
        return


topk_idx_t = torch.int32 if int(os.environ.get('TOPK_IDX_BITS', '64')) == 32 else torch.int64


def is_sm90_compiled() -> bool:
    return False


def get_low_latency_rdma_size_hint(*_args, **_kwargs) -> int:
    raise NotImplementedError('Low-latency kernels are not available in the Python XPU backend')

