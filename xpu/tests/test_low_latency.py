import argparse

import torch
import torch.distributed as dist

import deep_ep
import deep_ep_xpu_cpp
from utils import init_dist, require_xpu_devices


def _expect_not_supported(label, fn):
    try:
        fn()
    except (NotImplementedError, RuntimeError) as exc:
        return str(exc)
    raise AssertionError(f'{label} unexpectedly succeeded on XPU')


def assert_low_latency_unsupported(buffer: deep_ep.Buffer, group: dist.ProcessGroup, rank: int) -> None:
    num_ranks = dist.get_world_size(group)
    num_experts = num_ranks
    num_tokens = 1
    hidden = 128

    messages = {
        'raw_get_low_latency_rdma_size_hint': _expect_not_supported(
            'deep_ep_xpu_cpp.get_low_latency_rdma_size_hint',
            lambda: deep_ep_xpu_cpp.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts),
        ),
        'get_low_latency_rdma_size_hint': _expect_not_supported(
            'Buffer.get_low_latency_rdma_size_hint',
            lambda: deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts),
        ),
        'constructor': _expect_not_supported(
            'Buffer(..., low_latency_mode=True)',
            lambda: deep_ep.Buffer(group, 0, 0, low_latency_mode=True, explicitly_destroy=True),
        ),
        'clean_low_latency_buffer': _expect_not_supported(
            'Buffer.clean_low_latency_buffer',
            lambda: buffer.clean_low_latency_buffer(num_tokens, hidden, num_experts),
        ),
    }

    x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device='xpu')
    packed_x = torch.zeros((1, num_ranks * num_tokens, hidden), dtype=torch.bfloat16, device='xpu')
    topk_idx = torch.zeros((num_tokens, 1), dtype=deep_ep.topk_idx_t, device='xpu')
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32, device='xpu')
    src_info = torch.zeros((1, num_ranks * num_tokens), dtype=torch.int32, device='xpu')
    layout_range = torch.zeros((1, num_ranks), dtype=torch.int64, device='xpu')
    mask_status = torch.zeros((num_ranks,), dtype=torch.int32, device='xpu')
    handle = (src_info, layout_range, num_tokens, hidden, num_experts)

    messages['low_latency_dispatch'] = _expect_not_supported(
        'Buffer.low_latency_dispatch',
        lambda: buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts),
    )
    messages['low_latency_combine'] = _expect_not_supported(
        'Buffer.low_latency_combine',
        lambda: buffer.low_latency_combine(packed_x, topk_idx, topk_weights, handle),
    )
    messages['low_latency_update_mask_buffer'] = _expect_not_supported(
        'Buffer.low_latency_update_mask_buffer',
        lambda: buffer.low_latency_update_mask_buffer(0, True),
    )
    messages['low_latency_query_mask_buffer'] = _expect_not_supported(
        'Buffer.low_latency_query_mask_buffer',
        lambda: buffer.low_latency_query_mask_buffer(mask_status),
    )
    messages['low_latency_clean_mask_buffer'] = _expect_not_supported(
        'Buffer.low_latency_clean_mask_buffer',
        lambda: buffer.low_latency_clean_mask_buffer(),
    )
    messages['get_next_low_latency_combine_buffer'] = _expect_not_supported(
        'Buffer.get_next_low_latency_combine_buffer',
        lambda: buffer.get_next_low_latency_combine_buffer(handle),
    )

    expected_fragment = 'intentionally unsupported'
    for label, message in messages.items():
        assert expected_fragment in message, f'{label} returned an unexpected error: {message}'

    print(f'[rank {rank}] verified explicit XPU low-latency unsupported behavior', flush=True)


def test_loop(local_rank: int, num_local_ranks: int, _: argparse.Namespace):
    rank, _, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(group, num_nvl_bytes=0, num_rdma_bytes=0, explicitly_destroy=True)
    assert_low_latency_unsupported(buffer, group, rank)
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify mirrored XPU low-latency APIs fail explicitly as unsupported')
    parser.add_argument('--num-processes', type=int, default=2, help='Number of processes to spawn (default: 2)')
    args = parser.parse_args()
    require_xpu_devices(args.num_processes, 'tests/test_low_latency.py')
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)
