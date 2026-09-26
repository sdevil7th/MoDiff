from modiff import dev, install


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
    calls = []
    monkeypatch.setattr(dev.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    assert dev.main(["check", "--json"]) == 0
    cmd, kw = calls.pop()
    assert cmd == [str(python), "-m", "modiff.preflight", "--json"]
    assert kw["cwd"] == tmp_path
    assert kw["env"]["ROCM_PATH"] == "/reviewed/rocm"
    assert kw["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert dev.main(["run"]) == 0
    assert calls.pop()[0] == [str(python), str(tmp_path / "main.py")]


def test_missing_runtime_and_unknown_command_do_not_install(tmp_path, monkeypatch):
    monkeypatch.setattr(dev, "ROOT", tmp_path)
    assert dev.main(["check"]) == 2
    assert dev.main(["sync"]) == 2
