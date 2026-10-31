from modiff import dev, install, runtime_environment
import pytest


def test_setup_preserves_existing_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dev, "ROOT", tmp_path)
    (tmp_path / ".venv").mkdir()
    called = []
    monkeypatch.setattr(install, "main", lambda args: called.append(args) or 0)
    assert dev.main(["setup", "--accelerator", "cpu", "--non-interactive"]) == 2
    assert not called
    assert "Existing .venv preserved" in capsys.readouterr().err
    assert dev.main(["plan", "--accelerator", "cpu"]) == 0
    assert called.pop() == ["--accelerator", "cpu", "--dry-run"]
    assert dev.main(["setup", "--repair", "--accelerator", "cpu"]) == 0
    assert called.pop() == ["--repair", "--accelerator", "cpu"]


def test_runtime_uses_managed_python_and_profile_environment(tmp_path, monkeypatch):
    import json
    import os

    monkeypatch.setattr(dev, "ROOT", tmp_path)
    venv = tmp_path / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    (venv / "modiff-profile.json").write_text(json.dumps({"profile": "amd-rocm-linux"}))
    monkeypatch.setattr(install, "_rocm_environment", lambda: {"ROCM_PATH": "/reviewed/rocm"})
    monkeypatch.setattr(runtime_environment.importlib.metadata, "version", lambda name: "2.9.1+rocm7.2.0")
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    monkeypatch.setattr(dev.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    assert dev.main(["check", "--json"]) == 0
    cmd, kw = calls.pop()
    assert cmd == [str(python), "-m", "modiff.preflight", "--json"]
    assert kw["cwd"] == tmp_path
    assert kw["env"]["ROCM_PATH"] == "/reviewed/rocm"
    allocator = kw["env"].get("PYTORCH_ALLOC_CONF", kw["env"].get("PYTORCH_CUDA_ALLOC_CONF"))
    assert allocator is None
    assert dev.main(["run"]) == 0
    assert calls.pop()[0] == [str(python), str(tmp_path / "main.py")]


def test_missing_runtime_and_unknown_command_do_not_install(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "ROOT", tmp_path)
    assert dev.main(["check"]) == 2
    assert dev.main(["sync"]) == 2


@pytest.mark.parametrize("platform_name", ["win32", "linux", "darwin"])
@pytest.mark.parametrize("command", ["check", "run"])
@pytest.mark.parametrize("operator_flags", [
    {}, {"HF_ENABLE_PARALLEL_LOADING": "YES"}, {"HF_DEACTIVATE_ASYNC_LOAD": ""},
    {"HF_ENABLE_PARALLEL_LOADING": "", "HF_DEACTIVATE_ASYNC_LOAD": "0"},
])
def test_managed_commands_forward_model_loading_policy(tmp_path, monkeypatch, platform_name, command, operator_flags):
    import os
    from modiff import runtime_profile

    monkeypatch.setattr(dev, "ROOT", tmp_path)
    monkeypatch.setattr(dev.sys, "platform", platform_name)
    monkeypatch.setattr(runtime_profile, "read_state", lambda venv: {})
    python = tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    for key in ("HF_ENABLE_PARALLEL_LOADING", "HF_DEACTIVATE_ASYNC_LOAD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_PARALLEL_LOADING_WORKERS", "3")
    for key, value in operator_flags.items():
        monkeypatch.setenv(key, value)
    cmd, environment = dev.runtime_command(command, [])
    assert cmd == [str(python), *(["-m", "modiff.preflight"] if command == "check" else [str(tmp_path / "main.py")])]
    expected = {"HF_ENABLE_PARALLEL_LOADING": "0" if platform_name == "win32" else "YES"}
    if platform_name == "win32":
        expected["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    expected.update(operator_flags)
    assert {key: environment[key] for key in expected} == expected
    if "HF_DEACTIVATE_ASYNC_LOAD" not in expected:
        assert "HF_DEACTIVATE_ASYNC_LOAD" not in environment
    assert environment["HF_PARALLEL_LOADING_WORKERS"] == "3"
