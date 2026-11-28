import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest

from modiff import runtime_environment
from modiff.runtime_environment import configure_allocator, configure_model_loading


def test_rocm_runtime_injects_only_the_selected_sdk(tmp_path):
    system_root = tmp_path / "rocm"
    (system_root / "lib").mkdir(parents=True)
    (system_root / "core-7.13" / "lib").mkdir(parents=True)
    selected = tmp_path / "explicit 7.2"
    (selected / "lib").mkdir(parents=True)
    environment = {"ROCM_PATH": str(selected), "LD_LIBRARY_PATH": "/operator/lib"}

    runtime_environment.configure_rocm_library_path(environment, system_root=system_root)

    assert environment["LD_LIBRARY_PATH"].split(os.pathsep) == [
        str((selected / "lib").resolve()),
        "/operator/lib",
    ]
    assert environment["HIP_PATH"] == str(selected)
    automatic = {}
    runtime_environment.configure_rocm_library_path(automatic, system_root=system_root)
    assert automatic["LD_LIBRARY_PATH"] == str((system_root / "lib").resolve())
    hip_only = {"HIP_PATH": str(selected)}
    runtime_environment.configure_rocm_library_path(hip_only, system_root=system_root)
    assert hip_only["ROCM_PATH"] == str(selected)
    assert hip_only["LD_LIBRARY_PATH"] == str((selected / "lib").resolve())
    missing = {"ROCM_PATH": str(tmp_path / "missing"), "LD_LIBRARY_PATH": "/operator/lib"}
    runtime_environment.configure_rocm_library_path(missing, system_root=system_root)
    assert missing["LD_LIBRARY_PATH"] == "/operator/lib"


@pytest.mark.skipif(sys.platform != "linux", reason="The Ryzen ROCm launcher runs on Linux.")
def test_rocm_shell_launcher_honors_explicit_sdk(tmp_path):
    sdk_lib = tmp_path / "rocm explicit" / "lib"
    sdk_lib.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "MODIFF_RUNTIME_PROFILE": "amd-rocm-linux",
        "ROCM_PATH": str(sdk_lib.parent),
        "LD_LIBRARY_PATH": "/operator/runtime/lib",
    }
    result = subprocess.run(
        ["bash", str(root / "scripts" / "with-runtime-env.sh"), "printenv", "LD_LIBRARY_PATH"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15, check=True,
    )
    assert result.stdout.strip().split(os.pathsep) == [str(sdk_lib.resolve()), "/operator/runtime/lib"]


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
    result = configure_allocator(environment, platform_name="linux", torch_version="2.10.0+cu130")
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


@pytest.mark.parametrize("platform_name", ["win32", "linux", "darwin", "freebsd"])
def test_model_loading_platform_defaults_without_model_imports(monkeypatch, platform_name):
    real_import = builtins.__import__

    def reject_model_import(name, *args, **kwargs):
        assert name.split(".", 1)[0] not in {"torch", "diffusers", "transformers", "peft"}
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_model_import)
    environment = {"HF_PARALLEL_LOADING_WORKERS": "3", "unrelated": "preserved"}
    configure_model_loading(environment, platform_name=platform_name)
    assert environment == {
        "HF_PARALLEL_LOADING_WORKERS": "3", "unrelated": "preserved",
        "HF_ENABLE_PARALLEL_LOADING": "0" if platform_name == "win32" else "YES",
        **({"HF_DEACTIVATE_ASYNC_LOAD": "1"} if platform_name == "win32" else {}),
    }
    expected = environment.copy()
    configure_model_loading(environment, platform_name=platform_name)
    assert environment == expected


@pytest.mark.parametrize("platform_name", ["win32", "linux", "darwin"])
@pytest.mark.parametrize("key", ["HF_ENABLE_PARALLEL_LOADING", "HF_DEACTIVATE_ASYNC_LOAD"])
@pytest.mark.parametrize("value", ["", "0", "1", "false", "YES"])
def test_model_loading_preserves_each_explicit_operator_flag(platform_name, key, value):
    environment = {key: value}
    configure_model_loading(environment, platform_name=platform_name)
    expected = {"HF_ENABLE_PARALLEL_LOADING": "0" if platform_name == "win32" else "YES"}
    if platform_name == "win32":
        expected["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    expected[key] = value
    assert environment == expected


def test_model_loading_preserves_both_explicit_choices_and_worker_count():
    environment = {"HF_ENABLE_PARALLEL_LOADING": "YES", "HF_DEACTIVATE_ASYNC_LOAD": "",
                   "HF_PARALLEL_LOADING_WORKERS": "2"}
    expected = environment.copy()
    configure_model_loading(environment, platform_name="win32")
    assert environment == expected


@pytest.mark.parametrize("telemetry_override", [None, "0"])
def test_main_configures_hub_telemetry_before_local_only_lookup(tmp_path, telemetry_override):
    script = """
import sys

import httpx

calls = []

def intercept(_client, request, **kwargs):
    calls.append(request.url.path)
    return httpx.Response(503, request=request)

httpx.Client.send = intercept
import main
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

try:
    hf_hub_download("fixture/cache-only", "optional.json", revision="a" * 40,
                    local_files_only=True, token=False)
except LocalEntryNotFoundError:
    pass
else:
    raise AssertionError("The fresh temporary cache unexpectedly resolved a file")
if sys.argv[1] == "disabled":
    assert not calls, calls
from huggingface_hub import constants
assert constants.HF_HUB_DISABLE_TELEMETRY == (sys.argv[1] == "disabled")
"""
    environment = os.environ.copy()
    for name in (
        "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DISABLE_TELEMETRY", "DO_NOT_TRACK",
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    ):
        environment.pop(name, None)
    environment.update({
        "HF_HOME": str(tmp_path / "hf"),
        "HF_HUB_CACHE": str(tmp_path / "hf" / "hub"),
        "HF_TOKEN_PATH": str(tmp_path / "absent-token"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if telemetry_override is not None:
        environment["HF_HUB_DISABLE_TELEMETRY"] = telemetry_override
    result = subprocess.run(
        [
            sys.executable, "-B", "-c", script,
            "disabled" if telemetry_override is None else "enabled",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
