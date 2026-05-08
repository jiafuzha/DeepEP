import importlib
import subprocess
import sys
import textwrap

import pytest
import torch


def test_xpu_python_package_importable():
    module = importlib.import_module('xpu.deep_ep')
    assert hasattr(module, 'Buffer')
    assert hasattr(module, 'Config')


def test_xpu_extension_loads_and_exposes_topk_dtype():
    buffer_mod = importlib.import_module('xpu.deep_ep.buffer')
    assert buffer_mod._EXT_NAME in ('xpu.deep_ep_cpp_xpu', 'deep_ep_cpp_xpu', 'deep_ep_cpp')
    assert hasattr(buffer_mod._EXT, 'topk_idx_t')


def _run_inline_python(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)


def test_native_xpu_stale_ipc_handle_rejected_after_exporter_destroy():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        exporter = ext.Buffer(0, 2, 256, 0, False, False, False, False)
        stale_handle = exporter.get_local_ipc_handle()
        exporter.destroy()

        importer = ext.Buffer(1, 2, 256, 0, False, False, False, False)
        local_handle = importer.get_local_ipc_handle()
        device_id = importer.get_local_device_id()
        importer.sync([device_id, device_id], [stale_handle, local_handle], None)
        """
    )

    completed = _run_inline_python(script)
    # In staged native mode, stale-handle rejection currently terminates the worker process.
    assert completed.returncode != 0


def test_native_xpu_malformed_ipc_handle_metadata_rejected():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        exporter = ext.Buffer(0, 2, 256, 0, False, False, False, False)
        bad_handle = bytearray(exporter.get_local_ipc_handle())
        bad_handle[0] ^= 0xFF

        importer = ext.Buffer(1, 2, 256, 0, False, False, False, False)
        local_handle = importer.get_local_ipc_handle()
        device_id = importer.get_local_device_id()
        importer.sync([device_id, device_id], [bytes(bad_handle), local_handle], None)
        """
    )

    completed = _run_inline_python(script)
    # Malformed-handle rejection currently terminates the worker process in staged mode.
    assert completed.returncode != 0


def test_native_xpu_corrupted_ipc_checksum_rejected():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        exporter = ext.Buffer(0, 2, 256, 0, False, False, False, False)
        bad_handle = bytearray(exporter.get_local_ipc_handle())
        assert len(bad_handle) > 48
        bad_handle[48] ^= 0x01

        importer = ext.Buffer(1, 2, 256, 0, False, False, False, False)
        local_handle = importer.get_local_ipc_handle()
        device_id = importer.get_local_device_id()
        importer.sync([device_id, device_id], [bytes(bad_handle), local_handle], None)
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode != 0


def test_native_xpu_duplicate_remote_ipc_handle_rejected():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        exporter = ext.Buffer(0, 3, 256, 0, False, False, False, False)
        duplicated_remote_handle = exporter.get_local_ipc_handle()

        importer = ext.Buffer(1, 3, 256, 0, False, False, False, False)
        local_handle = importer.get_local_ipc_handle()
        device_id = importer.get_local_device_id()
        importer.sync(
            [device_id, device_id, device_id],
            [duplicated_remote_handle, local_handle, duplicated_remote_handle],
            None,
        )
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode != 0


def test_native_xpu_ipc_handle_kind_mismatch_rejected():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        exporter = ext.Buffer(0, 2, 256, 0, False, False, False, True)
        exporter_handle = exporter.get_local_ipc_handle()

        importer = ext.Buffer(1, 2, 256, 0, False, False, False, False)
        local_handle = importer.get_local_ipc_handle()
        device_id = importer.get_local_device_id()
        importer.sync([device_id, device_id], [exporter_handle, local_handle], None)
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode != 0


def test_native_xpu_internode_requires_rdma_bytes():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        # Pre-sync semantics on a topology that would previously be misclassified.
        topo_buf = ext.Buffer(9, 16, 256, 0, False, False, False, False)
        assert not topo_buf.is_available()
        assert not topo_buf.is_internode_available()

        # Post-sync semantics in a single-rank local setup.
        sync_buf = ext.Buffer(0, 1, 256, 0, False, False, False, False)
        device_id = sync_buf.get_local_device_id()
        local_handle = sync_buf.get_local_ipc_handle()
        sync_buf.sync([device_id], [local_handle], None)

        assert sync_buf.is_available()
        assert not sync_buf.is_internode_available()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_use_fabric_hint_is_accepted_in_staged_mode():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        buf = ext.Buffer(0, 1, 256, 0, False, True, False, True)
        device_id = buf.get_local_device_id()
        local_handle = buf.get_local_ipc_handle()
        buf.sync([device_id], [local_handle], None)
        assert buf.is_available()
        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_rdma_bytes_are_ignored_in_staged_mode():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        # Staged XPU mode does not support RDMA internode path yet;
        # requested RDMA bytes should be downgraded to local-only behavior.
        buf = ext.Buffer(0, 1, 256, 4096, False, True, False, False)
        device_id = buf.get_local_device_id()
        local_handle = buf.get_local_ipc_handle()
        buf.sync([device_id], [local_handle], None)
        assert buf.is_available()
        assert not buf.is_internode_available()
        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_manual_destroy_is_safe_with_auto_destructor_mode():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import gc
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        # explicitly_destroy=False means destructor will auto-destroy.
        # Manual destroy should still be safe and not trigger a second-destroy crash.
        buf = ext.Buffer(0, 1, 256, 0, False, False, False, False)
        device_id = buf.get_local_device_id()
        local_handle = buf.get_local_ipc_handle()
        buf.sync([device_id], [local_handle], None)
        buf.destroy()
        del buf
        gc.collect()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_double_destroy_is_idempotent():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        buf = ext.Buffer(0, 1, 32 * 1024 * 1024, 0, False, True, False, False)
        device_id = buf.get_local_device_id()
        local_handle = buf.get_local_ipc_handle()
        buf.sync([device_id], [local_handle], None)
        buf.destroy()
        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_local_device_id_matches_torch_current_device():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        buf = ext.Buffer(0, 1, 256, 0, False, True, False, False)
        assert buf.get_local_device_id() == torch.xpu.current_device()
        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_device_synchronize_succeeds():
    """Verify that device synchronization via c10::xpu::syncStreamsOnDevice() works."""
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        buf = ext.Buffer(0, 1, 256, 0, False, True, False, False)
        device_id = buf.get_local_device_id()
        local_handle = buf.get_local_ipc_handle()
        buf.sync([device_id], [local_handle], None)
        # destroy() internally calls runtime_device_synchronize()
        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_layout_get_dispatch_layout_succeeds():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        # Single-rank layout case that exercises native layout::get_dispatch_layout.
        topk_idx = torch.tensor([[0, 1], [1, -1], [-1, -1]], dtype=torch.int64, device='xpu')
        buf = ext.Buffer(0, 1, 256, 0, False, True, False, False)
        try:
            per_rank, per_rdma, per_expert, in_rank, _ = buf.get_dispatch_layout(topk_idx, 2, None, False, False)
            assert per_rdma is None
            assert per_rank.cpu().tolist() == [2]
            assert per_expert.cpu().tolist() == [1, 2]
            assert in_rank.cpu().to(torch.int32).view(-1).tolist() == [1, 1, 0]
        finally:
            buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_intranode_dispatch_combine_single_rank_succeeds():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        x = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
                [17, 18, 19, 20, 21, 22, 23, 24],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk_idx = torch.tensor([[0], [0], [-1]], dtype=torch.int64, device='xpu')
        topk_weights = torch.ones((3, 1), dtype=torch.float32, device='xpu')

        buf = ext.Buffer(0, 1, 32 * 1024 * 1024, 0, False, True, False, False)
        cfg = ext.Config()
        try:
            per_rank, per_rdma, per_expert, in_rank, _ = buf.get_dispatch_layout(topk_idx, 1, None, False, False)
            assert per_rdma is None
            assert per_rank.cpu().tolist() == [2]
            assert per_expert.cpu().tolist() == [2]

            dispatch_out = buf.intranode_dispatch(
                x,
                None,
                topk_idx,
                topk_weights,
                per_rank,
                in_rank,
                per_expert,
                0,
                None,
                None,
                1,
                0,
                cfg,
                None,
                False,
                False,
            )
            recv_x, _, recv_topk_idx, recv_topk_weights, _, rank_prefix, _, channel_prefix, recv_src_idx, send_head, _ = dispatch_out
            assert recv_x.size(0) == 2
            assert recv_src_idx.cpu().tolist() == [0, 1]
            assert recv_topk_idx.cpu().tolist() == [[0], [0]]
            assert recv_topk_weights.cpu().tolist() == [[1.0], [1.0]]

            combined_x, combined_topk_weights, _ = buf.intranode_combine(
                recv_x,
                recv_topk_weights,
                None,
                None,
                recv_src_idx,
                rank_prefix,
                channel_prefix,
                send_head,
                cfg,
                None,
                False,
                False,
            )

            assert combined_x.cpu().to(torch.float32).tolist() == [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
            assert combined_topk_weights.cpu().tolist() == [[1.0], [1.0], [0.0]]
        finally:
            buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_intranode_combine_reduces_and_applies_bias_single_rank():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        x = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [10, 20, 30, 40, 50, 60, 70, 80],
                [100, 200, 300, 400, 500, 600, 700, 800],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk_weights = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32, device='xpu')
        bias = torch.ones((2, 8), dtype=torch.bfloat16, device='xpu')
        src_idx = torch.tensor([0, 0, 1], dtype=torch.int32, device='xpu')
        rank_prefix = torch.tensor([[2]], dtype=torch.int32, device='xpu')
        channel_prefix = torch.zeros((1, 10), dtype=torch.int32, device='xpu')
        send_head = torch.zeros((2, 1), dtype=torch.int32, device='xpu')

        buf = ext.Buffer(0, 1, 32 * 1024 * 1024, 0, False, True, False, False)
        cfg = ext.Config()
        try:
            combined_x, combined_topk_weights, _ = buf.intranode_combine(
                x,
                topk_weights,
                bias,
                None,
                src_idx,
                rank_prefix,
                channel_prefix,
                send_head,
                cfg,
                None,
                False,
                False,
            )

            expected_x = torch.tensor(
                [
                    [12.0, 23.0, 34.0, 45.0, 56.0, 67.0, 78.0, 89.0],
                    [101.0, 201.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
                ],
                dtype=torch.float32,
            )
            assert torch.allclose(combined_x.cpu().to(torch.float32), expected_x)
            assert combined_topk_weights.cpu().tolist() == [[3.0], [3.0]]
        finally:
            buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_intranode_dispatch_cached_handle_single_rank_succeeds():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        x1 = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
                [17, 18, 19, 20, 21, 22, 23, 24],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        x2 = torch.tensor(
            [
                [101, 102, 103, 104, 105, 106, 107, 108],
                [109, 110, 111, 112, 113, 114, 115, 116],
                [117, 118, 119, 120, 121, 122, 123, 124],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk_idx = torch.tensor([[0], [0], [-1]], dtype=torch.int64, device='xpu')
        topk_weights = torch.ones((3, 1), dtype=torch.float32, device='xpu')

        buf = ext.Buffer(0, 1, 32 * 1024 * 1024, 0, False, True, False, False)
        cfg = ext.Config()
        try:
            per_rank, per_rdma, per_expert, in_rank, _ = buf.get_dispatch_layout(topk_idx, 1, None, False, False)
            first = buf.intranode_dispatch(
                x1,
                None,
                topk_idx,
                topk_weights,
                per_rank,
                in_rank,
                per_expert,
                0,
                None,
                None,
                1,
                0,
                cfg,
                None,
                False,
                False,
            )
            _, _, _, _, _, rank_prefix, channel_prefix, recv_channel_prefix, recv_src_idx, _, _ = first

            second = buf.intranode_dispatch(
                x2,
                None,
                None,
                None,
                None,
                in_rank,
                None,
                int(recv_src_idx.size(0)),
                rank_prefix,
                channel_prefix,
                1,
                0,
                cfg,
                None,
                False,
                False,
            )
            recv_x, _, recv_topk_idx, recv_topk_weights, _, _, _, recv_channel_prefix_cached, recv_src_idx_cached, _, _ = second

            assert recv_topk_idx is None
            assert recv_topk_weights is None
            assert recv_src_idx_cached.cpu().tolist() == [0, 1]
            assert recv_channel_prefix_cached.shape == recv_channel_prefix.shape
            assert recv_x.cpu().to(torch.float32).tolist() == [
                [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
                [109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0],
            ]
        finally:
            buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_intranode_dispatch_two_rank_same_process_succeeds():
    ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
    ext_path = getattr(ext, '__file__', '')
    if not ext_path.endswith(('.so', '.pyd')):
        pytest.skip('native staged extension is not active')
    if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
        pytest.skip('xpu runtime is not available')

    script = textwrap.dedent(
        """
        import importlib
        import threading
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        _ = torch.empty((1,), device='xpu')

        buf0 = ext.Buffer(0, 2, 32 * 1024 * 1024, 0, False, True, False, False)
        buf1 = ext.Buffer(1, 2, 32 * 1024 * 1024, 0, False, True, False, False)
        cfg = ext.Config()

        handles = [buf0.get_local_ipc_handle(), buf1.get_local_ipc_handle()]
        device_id = buf0.get_local_device_id()
        buf0.sync([device_id, device_id], handles, None)
        buf1.sync([device_id, device_id], handles, None)

        x0 = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        x1 = torch.tensor(
            [
                [101, 102, 103, 104, 105, 106, 107, 108],
                [109, 110, 111, 112, 113, 114, 115, 116],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk0 = torch.tensor([[0], [1]], dtype=torch.int64, device='xpu')
        topk1 = torch.tensor([[0], [1]], dtype=torch.int64, device='xpu')
        weights0 = torch.ones((2, 1), dtype=torch.float32, device='xpu')
        weights1 = torch.ones((2, 1), dtype=torch.float32, device='xpu')

        layout0 = buf0.get_dispatch_layout(topk0, 2, None, False, False)
        layout1 = buf1.get_dispatch_layout(topk1, 2, None, False, False)

        results = [None, None]
        errors = []

        def run_dispatch(index, buf, x, topk, weights, layout):
            try:
                results[index] = buf.intranode_dispatch(
                    x,
                    None,
                    topk,
                    weights,
                    layout[0],
                    layout[3],
                    layout[2],
                    0,
                    None,
                    None,
                    1,
                    0,
                    cfg,
                    None,
                    False,
                    False,
                )
            except Exception as exc:
                errors.append(repr(exc))

        t0 = threading.Thread(target=run_dispatch, args=(0, buf0, x0, topk0, weights0, layout0))
        t1 = threading.Thread(target=run_dispatch, args=(1, buf1, x1, topk1, weights1, layout1))
        t0.start()
        t1.start()
        t0.join()
        t1.join()

        assert not errors, errors

        recv_x0, _, recv_topk0, recv_w0, _, _, _, recv_channel0, recv_src0, _, _ = results[0]
        recv_x1, _, recv_topk1, recv_w1, _, _, _, recv_channel1, recv_src1, _, _ = results[1]

        assert recv_src0.cpu().tolist() == [0, 0]
        assert recv_src1.cpu().tolist() == [1, 1]
        assert recv_topk0.cpu().tolist() == [[0], [0]]
        assert recv_topk1.cpu().tolist() == [[1], [1]]
        assert recv_w0.cpu().tolist() == [[1.0], [1.0]]
        assert recv_w1.cpu().tolist() == [[1.0], [1.0]]
        assert recv_channel0.shape == (2, cfg.num_sms // 2)
        assert recv_channel1.shape == (2, cfg.num_sms // 2)
        assert recv_x0.cpu().to(torch.float32).tolist() == [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        ]
        assert recv_x1.cpu().to(torch.float32).tolist() == [
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0],
        ]

        buf0.destroy()
        buf1.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr
