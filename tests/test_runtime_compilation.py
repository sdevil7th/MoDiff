import json
from types import SimpleNamespace
import subprocess

import pytest

from modiff import runtime_compilation as compilation


@pytest.fixture(autouse=True)
def clear_probe_cache():
    compilation._RESULTS.clear()
    yield
    compilation._RESULTS.clear()


def test_importable_api_is_not_compiler_readiness(monkeypatch):
    monkeypatch.setattr(compilation.subprocess, "run", lambda *args, **kw: pytest.fail("Inspection compiled a kernel"))
    assert compilation.cached_compilation_status(device="cuda:0")["probe_required"]
    assert not compilation.cached_compilation_status(device="cuda:0")["available"]


def test_probe_executes_kernel_and_reuses_only_matching_environment(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"supported": True, "executed": True, "device": "cuda:0", "backend": "inductor"}), stderr="")
    monkeypatch.setattr(compilation.subprocess, "run", run)
    assert compilation.require_compilation(device="cuda:0")["available"]
    assert compilation.require_compilation(device="cuda:0")["available"]
    assert len(calls) == 1
    assert "actual = compiled(value)" in calls[0][0][-1]
    assert "torch.cuda.synchronize(device)" in calls[0][0][-1]
    assert calls[0][1]["timeout"] == 120
    assert not compilation.cached_compilation_status(device="cuda:1")["available"]
    assert not compilation.cached_compilation_status(device="cuda:0", require_flex=True)["available"]
    monkeypatch.setenv("CC", "changed-toolchain")
    assert not compilation.cached_compilation_status(device="cuda:0")["available"]


@pytest.mark.parametrize("result", [
    SimpleNamespace(returncode=1, stdout="", stderr="No module named triton"),
    SimpleNamespace(returncode=0, stdout='{"supported":true}', stderr=""),
    subprocess.TimeoutExpired("probe", 120),
])
def test_missing_toolchain_incomplete_probe_and_timeout_are_actionable(monkeypatch, result):
    def run(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(compilation.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Windows uses triton-windows"):
        compilation.require_compilation(device="cuda:0", require_flex=True)
    assert not compilation.cached_compilation_status(device="cuda:0", require_flex=True)["available"]


def test_flex_probe_executes_compiled_attention():
    script = compilation.compilation_probe_script(device="cuda:0", require_flex=True)
    assert "output = attention(q, q, q)" in script
    compile(script, "compiler-probe", "exec")


@pytest.mark.parametrize("detail", [
    {"device": "cpu", "backend": "inductor", "flexExecuted": True},
    {"device": "cuda:0", "backend": "eager", "flexExecuted": True},
    {"device": "cuda:0", "backend": "inductor", "flexExecuted": False},
])
def test_probe_must_execute_requested_device_backend_and_flex(monkeypatch, detail):
    detail.update(supported=True, executed=True)
    monkeypatch.setattr(compilation.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(detail), stderr=""))
    with pytest.raises(RuntimeError, match="compiled kernel did not execute"):
        compilation.require_compilation(device="cuda:0", require_flex=True)


def test_required_compiled_model_checks_toolchain_before_repository_or_weights(monkeypatch):
    from modules.ModularDiffusers.loaders import ModelsLoader, REQUIRED_REGIONAL_COMPILE_MODEL_TYPES

    loader = ModelsLoader()
    monkeypatch.setattr(loader, "_preflight_reviewed_builtin_selection", lambda **kw: pytest.fail("Resolved model before compiler check"))
    def fail_probe(**kwargs):
        assert kwargs == {"device": "cuda:0", "require_flex": True}
        raise RuntimeError("Compiler toolchain unavailable")
    monkeypatch.setattr(compilation, "require_compilation", fail_probe)
    with pytest.raises(RuntimeError, match="Compiler toolchain unavailable"):
        loader.execute(model_type=next(iter(REQUIRED_REGIONAL_COMPILE_MODEL_TYPES)), repo_id="unused/model", device="cuda:0", dtype="bfloat16")
