from modiff.runtime_environment import configure_allocator


def test_windows_and_macos_use_default_allocator():
    for platform_name in ("win32", "darwin"):
        environment = {}
        result = configure_allocator(environment, platform_name=platform_name)
        assert environment == ({"MIMALLOC_PURGE_DELAY": "0"} if platform_name == "win32" else {})
        assert result["source"] == "torch_default"


def test_linux_default_and_operator_overrides():
    environment = {}
    configure_allocator(environment, platform_name="linux", torch_version="2.8.0+cu128")
    assert environment == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"):
        for value in ("", "backend:cudaMallocAsync", "expandable_segments:True"):
            environment = {key: value}
            result = configure_allocator(environment, platform_name="win32")
            assert environment == {key: value, "MIMALLOC_PURGE_DELAY": "0"}
            assert result["source"] == "operator"
            assert bool(result["warning"]) == (value == "expandable_segments:True")


def test_linux_preserves_rocm_operator_settings():
    environment = {"PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.8"}
    result = configure_allocator(environment, platform_name="linux", torch_version="2.9.1+rocm7.2")
    assert environment == {"PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.8"}
    assert result["source"] == "operator"


def test_new_torch_uses_modern_allocator_variable_without_importing_torch():
    environment = {}
    result = configure_allocator(environment, platform_name="linux", torch_version="2.9.1+rocm7.2")
    assert environment == {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    assert result["variable"] == "PYTORCH_ALLOC_CONF"


def test_windows_preserves_host_allocator_operator_settings():
    for value in ("", "0", "10", "100", "-1"):
        for gpu_setting in ({}, {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"}):
            environment = {"MIMALLOC_PURGE_DELAY": value, **gpu_setting}
            expected = environment.copy()
            configure_allocator(environment, platform_name="win32")
            assert environment == expected


def test_other_platforms_do_not_inject_windows_host_allocator_default():
    for platform_name in ("linux", "darwin", "freebsd"):
        for existing in ({}, {"MIMALLOC_PURGE_DELAY": "100"}):
            environment = existing.copy()
            configure_allocator(environment, platform_name=platform_name, torch_version="2.11.0")
            assert environment.get("MIMALLOC_PURGE_DELAY") == existing.get("MIMALLOC_PURGE_DELAY")
