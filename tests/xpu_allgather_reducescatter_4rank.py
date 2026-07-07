"""PyTorch allgather and reducescatter validation across 4 ranks on XPU.

Default target layout:
  - b70-hq-1: ranks 0-1 on 2 XPU devices
  - b70-hq-2: ranks 2-3 on 2 XPU devices

Example with torchrun:

  On b70-hq-1:
    MASTER_ADDR=b70-hq-1 MASTER_PORT=29521 \
    python -m torch.distributed.run \
      --nnodes=2 --nproc-per-node=2 --node_rank=0 \
      tests/xpu_allgather_reducescatter_4rank.py

  On b70-hq-2:
    MASTER_ADDR=b70-hq-1 MASTER_PORT=29521 \
    python -m torch.distributed.run \
      --nnodes=2 --nproc-per-node=2 --node_rank=1 \
      tests/xpu_allgather_reducescatter_4rank.py

The script also accepts MPI-style environments by translating PMI/OMPI rank
variables into the standard torch.distributed env:// variables.
"""

import argparse
import os
import statistics
import time
from typing import Dict, List

import torch
import torch.distributed as dist


DTYPE_MAP: Dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}

# Row values are wrapped by this modulus so that synthetic data stays exactly
# representable (and its cross-rank reduction stays exact) in low-precision
# dtypes such as bf16, even when the token count grows large. bf16 represents
# integers exactly only up to 256; keeping per-row values small ensures the
# reduce_scatter sum across ranks does not lose precision, so the strict
# equality validation remains valid at any --num-tensors.
_ROW_MOD = 8


def _env_int(*names: str, default: int = 0) -> int:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return int(value)
    return default


def _populate_dist_env() -> None:
    if os.getenv("RANK") is None:
        os.environ["RANK"] = str(_env_int("PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK", default=0))
    if os.getenv("WORLD_SIZE") is None:
        os.environ["WORLD_SIZE"] = str(_env_int("PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE", default=1))
    if os.getenv("LOCAL_RANK") is None:
        os.environ["LOCAL_RANK"] = str(
            _env_int("MPI_LOCALRANKID", "PMI_LOCAL_RANK", "PMIX_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
                     default=int(os.environ["RANK"])))


def _resolve_backend(device_type: str, requested_backend: str | None) -> str:
    if requested_backend is not None:
        return requested_backend
    return "xccl" if device_type == "xpu" else "gloo"


def _resolve_device(device_type: str, local_rank: int) -> torch.device:
    if device_type == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Requested --device xpu, but torch.xpu is unavailable")
        num_devices = torch.xpu.device_count()
        if num_devices <= 0:
            raise RuntimeError("Requested --device xpu, but no XPU devices were found")
        device = torch.device(f"xpu:{local_rank % num_devices}")
        torch.xpu.set_device(device)
        return device
    if device_type == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device type: {device_type}")


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if actual.dtype.is_floating_point:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
    else:
        if not torch.equal(actual, expected):
            raise AssertionError(f"{name} mismatch")


def _make_allgather_input(rank: int, num_tensors: int, tensor_dimension: int, dtype: torch.dtype,
                          device: torch.device) -> torch.Tensor:
    rows = (torch.arange(num_tensors, device=device, dtype=torch.int32) % _ROW_MOD).view(num_tensors, 1)
    values = rows + (rank + 1) * 10
    return values.to(dtype=dtype).expand(num_tensors, tensor_dimension).contiguous()


def _make_reducescatter_chunk(src_rank: int, dst_rank: int, num_tensors: int, tensor_dimension: int,
                              dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    rows = (torch.arange(num_tensors, device=device, dtype=torch.int32) % _ROW_MOD).view(num_tensors, 1)
    values = rows + (src_rank + 1) * (dst_rank + 1)
    return values.to(dtype=dtype).expand(num_tensors, tensor_dimension).contiguous()


def _make_reducescatter_input(rank: int, world_size: int, num_tensors: int, tensor_dimension: int, dtype: torch.dtype,
                              device: torch.device) -> torch.Tensor:
    return torch.cat(
        [_make_reducescatter_chunk(rank, dst_rank, num_tensors, tensor_dimension, dtype, device)
         for dst_rank in range(world_size)],
        dim=0,
    )


def _run_allgather(rank: int, world_size: int, num_tensors: int, tensor_dimension: int, dtype: torch.dtype,
                   device: torch.device) -> None:
    local_tensor = _make_allgather_input(rank, num_tensors, tensor_dimension, dtype, device)
    expected = torch.cat(
        [_make_allgather_input(src_rank, num_tensors, tensor_dimension, dtype, device) for src_rank in range(world_size)],
        dim=0,
    )

    if hasattr(dist, "all_gather_into_tensor"):
        gathered = torch.empty((world_size * num_tensors, tensor_dimension), dtype=dtype, device=device)
        dist.all_gather_into_tensor(gathered, local_tensor)
    else:
        pieces = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(pieces, local_tensor)
        gathered = torch.cat(pieces, dim=0)

    _assert_equal(gathered, expected, "all_gather result")


def _run_reducescatter(rank: int, world_size: int, num_tensors: int, tensor_dimension: int, dtype: torch.dtype,
                       device: torch.device) -> None:
    scatter_input = _make_reducescatter_input(rank, world_size, num_tensors, tensor_dimension, dtype, device)
    output = torch.empty((num_tensors, tensor_dimension), dtype=dtype, device=device)
    expected = sum(
        _make_reducescatter_chunk(src_rank, rank, num_tensors, tensor_dimension, dtype, device)
        for src_rank in range(world_size))

    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(output, scatter_input, op=dist.ReduceOp.SUM)
    else:
        dist.reduce_scatter(output,
                            list(scatter_input.chunk(world_size, dim=0)),
                            op=dist.ReduceOp.SUM)

    _assert_equal(output, expected, "reduce_scatter result")


def _synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize(device)


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


def _time_op(op, device: torch.device, warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        op()
    _synchronize(device)
    dist.barrier()
    latencies: List[float] = []
    for _ in range(iters):
        _synchronize(device)
        start = time.perf_counter()
        op()
        _synchronize(device)
        latencies.append(time.perf_counter() - start)
    return latencies


def _reduce_stat(value: float, world_size: int, device: torch.device, op=dist.ReduceOp.MAX) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=op)
    result = t.item()
    if op == dist.ReduceOp.SUM:
        return result / world_size
    return result


def _report(name: str, latencies: List[float], total_bytes: float, world_size: int, rank: int,
            device: torch.device) -> None:
    local_avg = statistics.mean(latencies)
    local_min = min(latencies)
    local_p50 = statistics.median(latencies)
    # Align to the slowest rank (collective latency is bounded by the slowest participant).
    avg = _reduce_stat(local_avg, world_size, device, dist.ReduceOp.MAX)
    p50 = _reduce_stat(local_p50, world_size, device, dist.ReduceOp.MAX)
    fastest = _reduce_stat(local_min, world_size, device, dist.ReduceOp.MAX)
    algbw = total_bytes / avg  # bytes/s
    busbw = algbw * (world_size - 1) / world_size
    if rank == 0:
        print(
            f"[{name:14s}] "
            f"avg_lat={avg * 1e6:9.2f} us  "
            f"p50_lat={p50 * 1e6:9.2f} us  "
            f"min_lat={fastest * 1e6:9.2f} us  "
            f"msg={total_bytes / (1024 * 1024):8.2f} MiB  "
            f"algbw={algbw / 1e9:7.2f} GB/s  "
            f"busbw={busbw / 1e9:7.2f} GB/s",
            flush=True,
        )


def _benchmark(rank: int, world_size: int, num_tensors: int, tensor_dimension: int, dtype: torch.dtype,
               device: torch.device, warmup: int, iters: int) -> None:
    elem_bytes = _dtype_bytes(dtype)
    per_rank_bytes = num_tensors * tensor_dimension * elem_bytes

    # all_gather: each rank contributes per_rank_bytes; full gathered buffer is world_size * per_rank_bytes.
    ag_in = _make_allgather_input(rank, num_tensors, tensor_dimension, dtype, device)
    ag_out = torch.empty((world_size * num_tensors, tensor_dimension), dtype=dtype, device=device)

    def ag_op():
        dist.all_gather_into_tensor(ag_out, ag_in)

    ag_lat = _time_op(ag_op, device, warmup, iters)
    _report("all_gather", ag_lat, world_size * per_rank_bytes, world_size, rank, device)

    # reduce_scatter: full input is world_size * per_rank_bytes; each rank keeps per_rank_bytes.
    rs_in = _make_reducescatter_input(rank, world_size, num_tensors, tensor_dimension, dtype, device)
    rs_out = torch.empty((num_tensors, tensor_dimension), dtype=dtype, device=device)

    def rs_op():
        dist.reduce_scatter_tensor(rs_out, rs_in, op=dist.ReduceOp.SUM)

    rs_lat = _time_op(rs_op, device, warmup, iters)
    _report("reduce_scatter", rs_lat, world_size * per_rank_bytes, world_size, rank, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ranks", type=int, default=4)
    parser.add_argument("--num-tensors", type=int, default=32)
    parser.add_argument("--tensor-dtype", type=str, default="bf16", choices=sorted(DTYPE_MAP))
    parser.add_argument("--tensor-dimension", type=int, default=7168)
    parser.add_argument("--device", type=str, default="xpu", choices=("xpu", "cpu"))
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true", help="Run latency/throughput benchmark after validation")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=50)
    args = parser.parse_args()

    _populate_dist_env()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = _resolve_device(args.device, local_rank)
    backend = _resolve_backend(args.device, args.backend)
    dtype = DTYPE_MAP[args.tensor_dtype]

    dist.init_process_group(backend=backend, init_method="env://")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != args.num_ranks:
            raise AssertionError(f"Expected {args.num_ranks} ranks, but initialized {world_size}")

        _run_allgather(rank, world_size, args.num_tensors, args.tensor_dimension, dtype, device)
        _run_reducescatter(rank, world_size, args.num_tensors, args.tensor_dimension, dtype, device)

        if device.type == "xpu":
            torch.xpu.synchronize(device)
        dist.barrier()
        if rank == 0:
            print(
                f"PASS all_gather + reduce_scatter on {world_size} ranks "
                f"(num_tensors={args.num_tensors}, dtype={args.tensor_dtype}, tensor_dimension={args.tensor_dimension})",
                flush=True,
            )

        if args.benchmark:
            if rank == 0:
                print(
                    f"Benchmark: backend={backend} device={device.type} world_size={world_size} "
                    f"num_tensors={args.num_tensors} dtype={args.tensor_dtype} "
                    f"tensor_dimension={args.tensor_dimension} warmup={args.warmup_iters} iters={args.bench_iters}",
                    flush=True,
                )
            dist.barrier()
            _benchmark(rank, world_size, args.num_tensors, args.tensor_dimension, dtype, device, args.warmup_iters,
                       args.bench_iters)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
