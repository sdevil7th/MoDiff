import builtins

import pytest

from modiff import runtime_environment
from modiff.runtime_environment import configure_allocator


def test_windows_and_macos_use_default_allocator():
    for platform_name in ("win32", "darwin"):
        environment = {}
        result = configure_allocator(environment, platform_name=platform_name)
        assert environment == {}
        assert result["source"] == "torch_default"


def test_linux_default_and_operator_overrides():
    environment = {}
    configure_allocator(environment, platform_name="linux", torch_version="2.8.0+cu128")
    assert environment == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"):
        for value in ("", "backend:cudaMallocAsync", "expandable_segments:True"):
            environment = {key: value}
            result = configure_allocator(environment, platform_name="win32")
            assert environment == {key: value}
            assert result["source"] == "operator"
            assert bool(result["warning"]) == (value == "expandable_segments:True")


def test_linux_preserves_rocm_operator_settings():
    environment = {"PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.8"}
    result = configure_allocator(environment, platform_name="linux", torch_version="2.9.1+rocm7.2")
    assert environment == {"PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.8"}
    assert result["source"] == "operator"


def test_new_torch_uses_modern_allocator_variable_without_importing_torch():
    environment = {}
    result = configure_allocator(environment, platform_name="linux", torch_version="2.10.0+cu130")
    assert environment == {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    assert result["variable"] == "PYTORCH_ALLOC_CONF"


@pytest.mark.parametrize("torch_version", ["2.9.1+rocm7.2.0", "2.10.0+rocm7.14.0"])
def test_linux_rocm_uses_torch_default_allocator(torch_version):
    environment = {}
    result = configure_allocator(environment, platform_name="linux", torch_version=torch_version)
    assert environment == {}
    assert result == {"source": "torch_default", "variable": None, "setting": None, "warning": None}


def test_allocator_discovers_rocm_metadata_before_torch_import(monkeypatch):
    real_import = builtins.__import__

    def reject_torch_import(name, *args, **kwargs):
        assert name.split(".", 1)[0] != "torch", "Allocator configuration must precede Torch import"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch_import)
    monkeypatch.setattr(runtime_environment.importlib.metadata, "version", lambda name: "2.10.0+rocm7.14.0")
    environment = {}
    result = configure_allocator(environment, platform_name="linux")
    assert environment == {}
    assert result["source"] == "torch_default"


@pytest.mark.parametrize("key", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"])
@pytest.mark.parametrize("value", ["", "expandable_segments:True", "expandable_segments:False"])
def test_linux_rocm_preserves_explicit_allocator_settings(key, value):
    environment = {key: value}
    result = configure_allocator(environment, platform_name="linux", torch_version="2.10.0+rocm7.14.0")
    assert environment == {key: value}
    assert result == {"source": "operator", "variable": key, "setting": value, "warning": None}


def test_profile_environment_does_not_override_actual_torch_backend():
    environment = {"MODIFF_RUNTIME_PROFILE": "amd-instinct-rocm-linux"}
    result = configure_allocator(environment, platform_name="linux", torch_version="2.10.0+cu130")
    assert environment["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert result["source"] == "platform_default"
