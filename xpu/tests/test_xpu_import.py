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
