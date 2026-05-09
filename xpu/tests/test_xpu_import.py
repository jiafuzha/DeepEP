import importlib
import os
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


def _run_inline_python(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    return subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=merged_env)


def _run_two_process_xpu_scripts(script_template: str) -> None:
    shared_env = os.environ.copy()
    shared_env['ZE_AFFINITY_MASK'] = '0,1'

    probe = _run_inline_python(
        "import sys, torch; sys.exit(0 if hasattr(torch, 'xpu') and torch.xpu.is_available() and torch.xpu.device_count() >= 2 else 2)",
        env={'ZE_AFFINITY_MASK': '0,1'},
    )
    if probe.returncode == 2:
        pytest.skip('at least two xpu devices are required under ZE_AFFINITY_MASK=0,1')
    assert probe.returncode == 0, probe.stderr

    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for local_device in (0, 1):
        script = script_template.replace('__LOCAL_DEVICE__', str(local_device))
        proc = subprocess.Popen(
            [sys.executable, '-c', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=shared_env,
        )
        processes.append((local_device, proc))

    for local_device, proc in processes:
        try:
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail(
                f'process for xpu device {local_device} timed out\nstdout:\n{stdout}\nstderr:\n{stderr}'
            )

        assert proc.returncode == 0, (
            f'process for xpu device {local_device} failed with return code {proc.returncode}\n'
            f'stdout:\n{stdout}\nstderr:\n{stderr}'
        )


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
        import torch

        ext = importlib.import_module('xpu.deep_ep_cpp_xpu')
        local_device = __LOCAL_DEVICE__
        torch.xpu.set_device(local_device)
        _ = torch.empty((1,), device='xpu')

        buf = ext.Buffer(0, 1, 32 * 1024 * 1024, 0, False, True, False, False)
        cfg = ext.Config()
        device_id = buf.get_local_device_id()
        local_handle = buf.get_local_ipc_handle()
        buf.sync([device_id], [local_handle], None)

        base = 100 * local_device
        x = torch.tensor(
            [
                [1 + base, 2 + base, 3 + base, 4 + base, 5 + base, 6 + base, 7 + base, 8 + base],
                [9 + base, 10 + base, 11 + base, 12 + base, 13 + base, 14 + base, 15 + base, 16 + base],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk = torch.tensor([[0], [0]], dtype=torch.int64, device='xpu')
        weights = torch.ones((2, 1), dtype=torch.float32, device='xpu')

        layout = buf.get_dispatch_layout(topk, 1, None, False, False)
        recv_x, _, recv_topk, recv_w, _, _, _, recv_channel, recv_src, _, _ = buf.intranode_dispatch(
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

        assert recv_src.cpu().tolist() == [0, 1]
        assert recv_topk.cpu().tolist() == [[0], [0]]
        assert recv_w.cpu().tolist() == [[1.0], [1.0]]
        assert recv_channel.shape[0] == 1
        assert recv_channel.shape[1] > 0
        assert recv_x.cpu().to(torch.float32).tolist() == [
            [1.0 + base, 2.0 + base, 3.0 + base, 4.0 + base, 5.0 + base, 6.0 + base, 7.0 + base, 8.0 + base],
            [9.0 + base, 10.0 + base, 11.0 + base, 12.0 + base, 13.0 + base, 14.0 + base, 15.0 + base, 16.0 + base],
        ]

        buf.destroy()
        """
    )

    _run_two_process_xpu_scripts(script)


def test_native_xpu_intranode_dispatch_two_rank_cached_same_process_succeeds():
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
        local_device = __LOCAL_DEVICE__
        torch.xpu.set_device(local_device)
        _ = torch.empty((1,), device='xpu')

        buf = ext.Buffer(0, 1, 32 * 1024 * 1024, 0, False, True, False, False)
        cfg = ext.Config()

        device_id = buf.get_local_device_id()
        local_handle = buf.get_local_ipc_handle()
        buf.sync([device_id], [local_handle], None)

        base = 100 * local_device
        x_uncached = torch.tensor(
            [
                [1 + base, 2 + base, 3 + base, 4 + base, 5 + base, 6 + base, 7 + base, 8 + base],
                [9 + base, 10 + base, 11 + base, 12 + base, 13 + base, 14 + base, 15 + base, 16 + base],
                [17 + base, 18 + base, 19 + base, 20 + base, 21 + base, 22 + base, 23 + base, 24 + base],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        x_cached = torch.tensor(
            [
                [201 + base, 202 + base, 203 + base, 204 + base, 205 + base, 206 + base, 207 + base, 208 + base],
                [209 + base, 210 + base, 211 + base, 212 + base, 213 + base, 214 + base, 215 + base, 216 + base],
                [217 + base, 218 + base, 219 + base, 220 + base, 221 + base, 222 + base, 223 + base, 224 + base],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk = torch.tensor([[0], [0], [-1]], dtype=torch.int64, device='xpu')
        weights = torch.ones((3, 1), dtype=torch.float32, device='xpu')

        layout = buf.get_dispatch_layout(topk, 1, None, False, False)
        first = buf.intranode_dispatch(
            x_uncached,
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

        rank_prefix = first[5]
        channel_prefix = first[6]
        recv_channel_prefix = first[7]
        recv_src_idx = first[8]

        second = buf.intranode_dispatch(
            x_cached,
            None,
            None,
            None,
            None,
            layout[3],
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
        assert recv_x.cpu().to(torch.float32).tolist() == x_cached[:2].cpu().to(torch.float32).tolist()

        buf.destroy()
        """
    )

    _run_two_process_xpu_scripts(script)


def test_native_xpu_intranode_combine_two_rank_same_process_succeeds():
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
        local_device = __LOCAL_DEVICE__
        torch.xpu.set_device(local_device)
        _ = torch.empty((1,), device='xpu')

        buf = ext.Buffer(0, 1, 32 * 1024 * 1024, 0, False, True, False, False)
        cfg = ext.Config()
        device_id = buf.get_local_device_id()
        local_handle = buf.get_local_ipc_handle()
        buf.sync([device_id], [local_handle], None)

        base = 100 * local_device
        x = torch.tensor(
            [
                [1 + base, 2 + base, 3 + base, 4 + base, 5 + base, 6 + base, 7 + base, 8 + base],
                [9 + base, 10 + base, 11 + base, 12 + base, 13 + base, 14 + base, 15 + base, 16 + base],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk = torch.tensor([[0], [0]], dtype=torch.int64, device='xpu')
        weights = torch.ones((2, 1), dtype=torch.float32, device='xpu')

        layout = buf.get_dispatch_layout(topk, 1, None, False, False)
        dispatch_out = buf.intranode_dispatch(
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

        recv_x, _, _, recv_w, _, rank_prefix, channel_prefix, _, recv_src, send_head, _ = dispatch_out
        combined_x, combined_w, _ = buf.intranode_combine(
            recv_x,
            recv_w,
            None,
            None,
            recv_src,
            rank_prefix,
            channel_prefix,
            send_head,
            cfg,
            None,
            False,
            False,
        )

        assert combined_x.cpu().to(torch.float32).tolist() == [
            [1.0 + base, 2.0 + base, 3.0 + base, 4.0 + base, 5.0 + base, 6.0 + base, 7.0 + base, 8.0 + base],
            [9.0 + base, 10.0 + base, 11.0 + base, 12.0 + base, 13.0 + base, 14.0 + base, 15.0 + base, 16.0 + base],
        ]
        assert combined_w.cpu().tolist() == [[1.0], [1.0]]

        buf.destroy()
        """
    )

    _run_two_process_xpu_scripts(script)


def test_native_xpu_intranode_combine_two_rank_cached_same_process_succeeds():
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
        local_device = __LOCAL_DEVICE__
        torch.xpu.set_device(local_device)
        _ = torch.empty((1,), device='xpu')

        buf = ext.Buffer(0, 1, 32 * 1024 * 1024, 0, False, True, False, False)
        cfg = ext.Config()

        device_id = buf.get_local_device_id()
        local_handle = buf.get_local_ipc_handle()
        buf.sync([device_id], [local_handle], None)

        base = 100 * local_device
        x_uncached = torch.tensor(
            [
                [1 + base, 2 + base, 3 + base, 4 + base, 5 + base, 6 + base, 7 + base, 8 + base],
                [9 + base, 10 + base, 11 + base, 12 + base, 13 + base, 14 + base, 15 + base, 16 + base],
                [17 + base, 18 + base, 19 + base, 20 + base, 21 + base, 22 + base, 23 + base, 24 + base],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        x_cached = torch.tensor(
            [
                [201 + base, 202 + base, 203 + base, 204 + base, 205 + base, 206 + base, 207 + base, 208 + base],
                [209 + base, 210 + base, 211 + base, 212 + base, 213 + base, 214 + base, 215 + base, 216 + base],
                [217 + base, 218 + base, 219 + base, 220 + base, 221 + base, 222 + base, 223 + base, 224 + base],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk = torch.tensor([[0], [0], [-1]], dtype=torch.int64, device='xpu')
        weights = torch.ones((3, 1), dtype=torch.float32, device='xpu')

        layout = buf.get_dispatch_layout(topk, 1, None, False, False)
        first = buf.intranode_dispatch(
            x_uncached,
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

        second = buf.intranode_dispatch(
            x_cached,
            None,
            None,
            None,
            None,
            layout[3],
            None,
            int(first[8].size(0)),
            first[5],
            first[6],
            1,
            0,
            cfg,
            None,
            False,
            False,
        )

        recv_x, _, _, recv_topk_weights, _, rank_prefix, channel_prefix, _, recv_src_idx, send_head, _ = second
        combined_x, combined_w, _ = buf.intranode_combine(
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

        assert combined_w is None
        expected = torch.zeros((3, 8), dtype=torch.float32)
        expected[:2] = x_cached[:2].cpu().to(torch.float32)
        assert combined_x.cpu().to(torch.float32).tolist() == expected.tolist()

        buf.destroy()
        """
    )

    _run_two_process_xpu_scripts(script)


def test_native_xpu_internode_dispatch_combine_single_rank_succeeds():
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

        # Keep staged local RDMA buffer enabled via low_latency_mode=True.
        buf = ext.Buffer(0, 1, 0, 32 * 1024 * 1024, True, True, False, False)
        cfg = ext.Config()
        device_id = buf.get_local_device_id()
        buf.sync([device_id], [], None)

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

        per_rank, _, per_expert, in_rank, _ = buf.get_dispatch_layout(topk_idx, 1, None, False, False)
        per_rdma = torch.tensor([2], dtype=torch.int32, device='xpu')

        dispatch_out = buf.internode_dispatch(
            x,
            None,
            topk_idx,
            topk_weights,
            per_rank,
            per_rdma,
            in_rank,
            per_expert,
            0,
            0,
            None,
            None,
            None,
            None,
            1,
            0,
            cfg,
            None,
            False,
            False,
        )

        recv_x, _, recv_topk_idx, recv_topk_weights = dispatch_out[:4]
        rdma_channel_prefix = dispatch_out[5]
        gbl_channel_prefix = dispatch_out[6]
        recv_rdma_rank_prefix = dispatch_out[8]
        recv_src_meta = dispatch_out[11]
        send_rdma_head = dispatch_out[12]
        send_nvl_head = dispatch_out[13]

        assert recv_x.size(0) == 2
        assert recv_topk_idx.cpu().tolist() == [[0], [0]]
        assert recv_topk_weights.cpu().tolist() == [[1.0], [1.0]]

        combined_x, combined_w, _ = buf.internode_combine(
            recv_x,
            recv_topk_weights,
            None,
            None,
            recv_src_meta,
            in_rank,
            rdma_channel_prefix,
            recv_rdma_rank_prefix,
            gbl_channel_prefix,
            send_rdma_head,
            send_nvl_head,
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
        assert combined_w.cpu().tolist() == [[1.0], [1.0], [0.0]]

        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_internode_dispatch_cached_handle_single_rank_succeeds():
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

        buf = ext.Buffer(0, 1, 0, 32 * 1024 * 1024, True, True, False, False)
        cfg = ext.Config()
        device_id = buf.get_local_device_id()
        buf.sync([device_id], [], None)

        x_uncached = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
                [17, 18, 19, 20, 21, 22, 23, 24],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        x_cached = torch.tensor(
            [
                [201, 202, 203, 204, 205, 206, 207, 208],
                [209, 210, 211, 212, 213, 214, 215, 216],
                [217, 218, 219, 220, 221, 222, 223, 224],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk_idx = torch.tensor([[0], [0], [-1]], dtype=torch.int64, device='xpu')
        topk_weights = torch.ones((3, 1), dtype=torch.float32, device='xpu')

        per_rank, _, per_expert, in_rank, _ = buf.get_dispatch_layout(topk_idx, 1, None, False, False)
        per_rdma = torch.tensor([2], dtype=torch.int32, device='xpu')

        first = buf.internode_dispatch(
            x_uncached,
            None,
            topk_idx,
            topk_weights,
            per_rank,
            per_rdma,
            in_rank,
            per_expert,
            0,
            0,
            None,
            None,
            None,
            None,
            1,
            0,
            cfg,
            None,
            False,
            False,
        )

        second = buf.internode_dispatch(
            x_cached,
            None,
            None,
            None,
            None,
            None,
            in_rank,
            None,
            int(first[0].size(0)),
            int(first[0].size(0)),
            first[5],
            first[8],
            first[6],
            first[10],
            1,
            0,
            cfg,
            None,
            False,
            False,
        )

        recv_x, _, recv_topk_idx, recv_topk_weights = second[:4]
        assert recv_topk_idx is None
        assert recv_topk_weights is None
        assert recv_x.cpu().to(torch.float32).tolist() == [
            [201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0],
            [209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0],
        ]

        combined_x, combined_w, _ = buf.internode_combine(
            recv_x,
            first[3],
            None,
            None,
            first[11],
            in_rank,
            first[5],
            first[8],
            first[6],
            first[12],
            first[13],
            cfg,
            None,
            False,
            False,
        )

        expected = torch.zeros((3, 8), dtype=torch.float32)
        expected[:2] = x_cached[:2].cpu().to(torch.float32)
        assert combined_x.cpu().to(torch.float32).tolist() == expected.tolist()
        assert combined_w.cpu().tolist() == [[1.0], [1.0], [0.0]]

        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_internode_dispatch_combine_two_rank_same_process_succeeds():
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
        local_device = __LOCAL_DEVICE__
        torch.xpu.set_device(local_device)
        _ = torch.empty((1,), device='xpu')

        num_experts = 1
        rdma_bytes = ext.get_low_latency_rdma_size_hint(4, 128, 1, num_experts)

        buf = ext.Buffer(0, 1, 0, rdma_bytes, True, True, False, False)
        cfg = ext.Config()
        device_id = buf.get_local_device_id()
        buf.sync([device_id], [], None)

        base = 100 * local_device
        x = torch.tensor(
            [
                [1 + base, 2 + base, 3 + base, 4 + base, 5 + base, 6 + base, 7 + base, 8 + base],
                [9 + base, 10 + base, 11 + base, 12 + base, 13 + base, 14 + base, 15 + base, 16 + base],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk = torch.tensor([[0], [0]], dtype=torch.int64, device='xpu')
        w = torch.ones((2, 1), dtype=torch.float32, device='xpu')

        layout = buf.get_dispatch_layout(topk, num_experts, None, False, False)
        per_rdma = torch.tensor([2], dtype=torch.int32, device='xpu')

        dispatched = buf.internode_dispatch(
            x,
            None,
            topk,
            w,
            layout[0],
            per_rdma,
            layout[3],
            layout[2],
            0,
            0,
            None,
            None,
            None,
            None,
            1,
            0,
            cfg,
            None,
            False,
            False,
        )

        combined_x, combined_w, _ = buf.internode_combine(
            dispatched[0],
            dispatched[3],
            None,
            None,
            dispatched[11],
            layout[3],
            dispatched[5],
            dispatched[8],
            dispatched[6],
            dispatched[12],
            dispatched[13],
            cfg,
            None,
            False,
            False,
        )

        assert combined_x.cpu().to(torch.float32).tolist() == x.cpu().to(torch.float32).tolist()
        assert combined_w.cpu().tolist() == [[1.0], [1.0]]

        buf.destroy()
        """
    )

    _run_two_process_xpu_scripts(script)


def test_native_xpu_internode_dispatch_combine_two_rank_cached_same_process_succeeds():
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
        local_device = __LOCAL_DEVICE__
        torch.xpu.set_device(local_device)
        _ = torch.empty((1,), device='xpu')

        num_experts = 1
        rdma_bytes = ext.get_low_latency_rdma_size_hint(4, 128, 1, num_experts)

        buf = ext.Buffer(0, 1, 0, rdma_bytes, True, True, False, False)
        cfg = ext.Config()
        device_id = buf.get_local_device_id()
        buf.sync([device_id], [], None)

        base = 100 * local_device
        x_uncached = torch.tensor(
            [
                [1 + base, 2 + base, 3 + base, 4 + base, 5 + base, 6 + base, 7 + base, 8 + base],
                [9 + base, 10 + base, 11 + base, 12 + base, 13 + base, 14 + base, 15 + base, 16 + base],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        x_cached = torch.tensor(
            [
                [201 + base, 202 + base, 203 + base, 204 + base, 205 + base, 206 + base, 207 + base, 208 + base],
                [209 + base, 210 + base, 211 + base, 212 + base, 213 + base, 214 + base, 215 + base, 216 + base],
            ],
            dtype=torch.bfloat16,
            device='xpu',
        )
        topk = torch.tensor([[0], [0]], dtype=torch.int64, device='xpu')
        w = torch.ones((2, 1), dtype=torch.float32, device='xpu')

        layout = buf.get_dispatch_layout(topk, num_experts, None, False, False)
        per_rdma = torch.tensor([2], dtype=torch.int32, device='xpu')

        first = buf.internode_dispatch(
            x_uncached,
            None,
            topk,
            w,
            layout[0],
            per_rdma,
            layout[3],
            layout[2],
            0,
            0,
            None,
            None,
            None,
            None,
            1,
            0,
            cfg,
            None,
            False,
            False,
        )

        second = buf.internode_dispatch(
            x_cached,
            None,
            None,
            None,
            None,
            None,
            layout[3],
            None,
            int(first[0].size(0)),
            int(first[0].size(0)),
            first[5],
            first[8],
            first[6],
            first[10],
            1,
            0,
            cfg,
            None,
            False,
            False,
        )

        combined_x, combined_w, _ = buf.internode_combine(
            second[0],
            first[3],
            None,
            None,
            first[11],
            layout[3],
            first[5],
            first[8],
            first[6],
            first[12],
            first[13],
            cfg,
            None,
            False,
            False,
        )

        assert combined_x.cpu().to(torch.float32).tolist() == x_cached.cpu().to(torch.float32).tolist()
        assert combined_w.cpu().tolist() == [[1.0], [1.0]]

        buf.destroy()
        """
    )

    _run_two_process_xpu_scripts(script)


def test_native_xpu_low_latency_maintenance_single_rank_succeeds():
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

        num_max_dispatch_tokens_per_rank = 4
        hidden = 512
        num_experts = 1
        rdma_bytes = ext.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank,
            hidden,
            1,
            num_experts,
        )

        buf = ext.Buffer(0, 1, 0, rdma_bytes, True, True, True, False)
        device_id = buf.get_local_device_id()
        buf.sync([device_id], [], None)

        status = torch.empty((1,), dtype=torch.int32, device='xpu')
        buf.low_latency_query_mask_buffer(status)
        assert status.cpu().tolist() == [0]

        buf.low_latency_update_mask_buffer(0, True)
        buf.low_latency_query_mask_buffer(status)
        assert status.cpu().tolist() == [1]

        buf.low_latency_clean_mask_buffer()
        buf.low_latency_query_mask_buffer(status)
        assert status.cpu().tolist() == [0]

        raw_buffer = buf.get_local_buffer_tensor(torch.int32, 0, True)
        raw_buffer.fill_(1)
        before_nonzero = torch.count_nonzero(raw_buffer).item()
        buf.clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)
        after_nonzero = torch.count_nonzero(raw_buffer).item()
        assert after_nonzero < before_nonzero
        assert raw_buffer[0].item() == 0

        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_low_latency_dispatch_combine_single_rank_succeeds():
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

        num_max_dispatch_tokens_per_rank = 4
        hidden = 128
        num_experts = 2
        rdma_bytes = ext.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank,
            hidden,
            1,
            num_experts,
        )

        buf = ext.Buffer(0, 1, 0, rdma_bytes, True, True, False, False)
        device_id = buf.get_local_device_id()
        buf.sync([device_id], [], None)

        x = torch.arange(4 * hidden, dtype=torch.float32, device='xpu').reshape(4, hidden).to(torch.bfloat16)
        topk_idx = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64, device='xpu')
        topk_weights = torch.ones((4, 1), dtype=torch.float32, device='xpu')

        packed_x, packed_scales, recv_count, src_info, layout_range, _, _ = buf.low_latency_dispatch(
            x,
            topk_idx,
            None,
            None,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            False,
            False,
            False,
            False,
            False,
        )

        assert packed_scales is None
        assert tuple(packed_x.shape) == (num_experts, num_max_dispatch_tokens_per_rank, hidden)
        assert recv_count.cpu().tolist() == [2, 2]
        assert tuple(src_info.shape) == (num_experts, num_max_dispatch_tokens_per_rank)
        assert tuple(layout_range.shape) == (num_experts, 1)

        combined_x, _, _ = buf.low_latency_combine(
            packed_x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            None,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            False,
            False,
            False,
            False,
            None,
        )

        assert combined_x.cpu().to(torch.float32).tolist() == x.cpu().to(torch.float32).tolist()

        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_low_latency_dispatch_combine_two_rank_same_process_succeeds():
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
        local_device = __LOCAL_DEVICE__
        torch.xpu.set_device(local_device)
        _ = torch.empty((1,), device='xpu')

        num_max_dispatch_tokens_per_rank = 4
        hidden = 128
        num_experts = 2
        rdma_bytes = ext.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank,
            hidden,
            1,
            num_experts,
        )

        buf = ext.Buffer(0, 1, 0, rdma_bytes, True, True, False, False)
        device_id = buf.get_local_device_id()
        buf.sync([device_id], [], None)

        base = 1000 * local_device
        x = (torch.arange(2 * hidden, dtype=torch.float32, device='xpu').reshape(2, hidden) + base).to(torch.bfloat16)
        topk = torch.tensor([[0], [1]], dtype=torch.int64, device='xpu')
        w = torch.ones((2, 1), dtype=torch.float32, device='xpu')

        packed_x, _, _, src_info, layout_range, _, _ = buf.low_latency_dispatch(
            x,
            topk,
            None,
            None,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            False,
            False,
            False,
            False,
            False,
        )
        combined_x, _, _ = buf.low_latency_combine(
            packed_x,
            topk,
            w,
            src_info,
            layout_range,
            None,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            False,
            False,
            False,
            False,
            None,
        )

        assert combined_x.cpu().to(torch.float32).tolist() == x.cpu().to(torch.float32).tolist()

        buf.destroy()
        """
    )

    _run_two_process_xpu_scripts(script)


def test_native_xpu_low_latency_dispatch_fp8_mode_single_rank_succeeds():
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

        num_max_dispatch_tokens_per_rank = 4
        hidden = 512
        num_experts = 2
        rdma_bytes = ext.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank,
            hidden,
            1,
            num_experts,
        )

        buf = ext.Buffer(0, 1, 0, rdma_bytes, True, True, False, False)
        device_id = buf.get_local_device_id()
        buf.sync([device_id], [], None)

        x = torch.arange(4 * hidden, dtype=torch.float32, device='xpu').reshape(4, hidden).to(torch.bfloat16)
        topk_idx = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64, device='xpu')
        topk_weights = torch.ones((4, 1), dtype=torch.float32, device='xpu')

        packed_x, packed_scales, recv_count, src_info, layout_range, _, _ = buf.low_latency_dispatch(
            x,
            topk_idx,
            None,
            None,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            True,
            False,
            False,
            False,
            False,
        )

        assert packed_x.dtype == torch.bfloat16
        assert packed_scales is not None
        assert packed_scales.dtype == torch.float32
        assert tuple(packed_scales.shape) == (num_experts, num_max_dispatch_tokens_per_rank, hidden // 128)
        assert torch.all(packed_scales == 1).item()
        assert recv_count.cpu().tolist() == [2, 2]

        combined_x, _, _ = buf.low_latency_combine(
            packed_x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            None,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            False,
            False,
            False,
            False,
            None,
        )

        assert combined_x.cpu().to(torch.float32).tolist() == x.cpu().to(torch.float32).tolist()

        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


def test_native_xpu_low_latency_combine_logfmt_single_rank_succeeds():
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

        num_max_dispatch_tokens_per_rank = 4
        hidden = 128
        num_experts = 2
        rdma_bytes = ext.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank,
            hidden,
            1,
            num_experts,
        )

        buf = ext.Buffer(0, 1, 0, rdma_bytes, True, True, False, False)
        device_id = buf.get_local_device_id()
        buf.sync([device_id], [], None)

        x = torch.arange(4 * hidden, dtype=torch.float32, device='xpu').reshape(4, hidden).to(torch.bfloat16)
        topk_idx = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64, device='xpu')
        topk_weights = torch.ones((4, 1), dtype=torch.float32, device='xpu')

        packed_x, _, _, src_info, layout_range, _, _ = buf.low_latency_dispatch(
            x,
            topk_idx,
            None,
            None,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            False,
            False,
            False,
            False,
            False,
        )

        combined_x, _, _ = buf.low_latency_combine(
            packed_x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            None,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            True,
            False,
            False,
            False,
            None,
        )

        assert combined_x.cpu().to(torch.float32).tolist() == x.cpu().to(torch.float32).tolist()

        buf.destroy()
        """
    )

    completed = _run_inline_python(script)
    assert completed.returncode == 0, completed.stderr


