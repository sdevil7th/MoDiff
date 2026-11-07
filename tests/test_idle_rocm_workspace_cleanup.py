"""Idle ROCm workspace reclamation without touching live graph ownership."""

from types import SimpleNamespace

import pytest

from modiff.server import WebServer


@pytest.fixture
def cleanup(monkeypatch):
    calls = []

    def record(name):
        return lambda: calls.append(name)

    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        device_count=lambda: 1,
        synchronize=record("synchronize"),
        empty_cache=record("empty_cache"),
        ipc_collect=record("ipc_collect"),
    )
    torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(hip="7.2"),
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=record("workspace_clear")),
        mps=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr("modiff.server.import_module", lambda name: torch if name == "torch" else None)
    app = SimpleNamespace(current_task=None, node_cache={})
    return app, torch, calls


def test_idle_rocm_releases_workspace_before_allocator_cache(cleanup):
    app, _torch, calls = cleanup
    assert WebServer._best_effort_device_cache_clear(app) == []
    assert calls == ["synchronize", "workspace_clear", "empty_cache", "ipc_collect"]


@pytest.mark.parametrize("ownership", ["active_task", "cached_owner"])
def test_live_ownership_preserves_workspaces(cleanup, ownership):
    app, _torch, calls = cleanup
    if ownership == "active_task":
        app.current_task = {"task_id": "active"}
    else:
        # Eligibility must not traverse opaque cached node/model values.
        app.node_cache = {"unrelated": object()}
    assert WebServer._best_effort_device_cache_clear(app) == []
    assert calls == ["empty_cache", "ipc_collect"]


@pytest.mark.parametrize("unsupported", ["nvidia", "missing_version", "missing_binding", "missing_C", "uninitialized", "multiple_devices", "no_devices"])
def test_unsupported_or_uninitialized_runtime_retains_existing_cleanup(cleanup, unsupported):
    app, torch, calls = cleanup
    if unsupported == "nvidia":
        torch.version.hip = None
    elif unsupported == "missing_version":
        del torch.version
    elif unsupported == "missing_binding":
        del torch._C._cuda_clearCublasWorkspaces
    elif unsupported == "missing_C":
        del torch._C
    elif unsupported == "uninitialized":
        torch.cuda.is_initialized = lambda: False
    else:
        torch.cuda.device_count = lambda: 2 if unsupported == "multiple_devices" else 0
    assert WebServer._best_effort_device_cache_clear(app) == []
    assert calls == ["empty_cache", "ipc_collect"]


@pytest.mark.parametrize("failing", ["device_count", "synchronize", "workspace_clear"])
def test_workspace_cleanup_failure_preserves_allocator_fallback(cleanup, failing):
    app, torch, calls = cleanup

    def fail():
        calls.append(failing)
        raise RuntimeError("workspace unavailable")

    if failing == "device_count":
        torch.cuda.device_count = fail
        expected = ["device_count", "empty_cache", "ipc_collect"]
        label = "torch.cuda.device_count"
    elif failing == "synchronize":
        torch.cuda.synchronize = fail
        expected = ["synchronize", "empty_cache", "ipc_collect"]
        label = "torch.cuda.synchronize"
    else:
        torch._C._cuda_clearCublasWorkspaces = fail
        expected = ["synchronize", "workspace_clear", "empty_cache", "ipc_collect"]
        label = "torch._C._cuda_clearCublasWorkspaces"
    assert WebServer._best_effort_device_cache_clear(app) == [f"{label}: workspace unavailable"]
    assert calls == expected


def test_allocator_failure_does_not_skip_remaining_cleanup(cleanup):
    app, torch, calls = cleanup

    def fail():
        calls.append("empty_cache")
        raise RuntimeError("allocator unavailable")

    torch.cuda.empty_cache = fail
    assert WebServer._best_effort_device_cache_clear(app) == ["torch.cuda.empty_cache: allocator unavailable"]
    assert calls == ["synchronize", "workspace_clear", "empty_cache", "ipc_collect"]


@pytest.mark.parametrize("mps_available", [False, True])
def test_cpu_and_mps_never_enter_cuda_workspace_cleanup(cleanup, mps_available):
    app, torch, calls = cleanup
    torch.cuda.is_available = lambda: False
    torch.mps.is_available = lambda: mps_available
    torch.mps.empty_cache = lambda: calls.append("mps_empty_cache")
    assert WebServer._best_effort_device_cache_clear(app) == []
    assert calls == (["mps_empty_cache"] if mps_available else [])
