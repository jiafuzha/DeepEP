def test_force_python_stub_entrypoint_smoke(forced_stub_backend) -> None:
    buffer_mod, ext = forced_stub_backend

    ext_file = getattr(ext, '__file__', '')
    assert ext_file.endswith('.py'), ext_file

    # Sanity-check a few APIs from the forced Python stub.
    assert hasattr(ext, 'Buffer')
    assert hasattr(ext, 'Config')
    assert hasattr(ext, 'get_low_latency_rdma_size_hint')
