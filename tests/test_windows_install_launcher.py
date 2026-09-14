"""Exercise the public PowerShell launcher with native stderr and exit codes."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell launcher")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_install_redirected_native_stderr_preserves_exit_code(tmp_path, exit_code):
    powershell = shutil.which("powershell.exe")
    if not powershell:
        pytest.skip("Windows PowerShell is unavailable")
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "install.ps1", tmp_path / "install.ps1")
    (tmp_path / "python.cmd").write_text(
        f"@echo off\necho native progress 1>&2\necho native finished\nexit /b {exit_code}\n",
        encoding="ascii",
    )
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command",
         "& ./install.ps1 -BackendOnly -NonInteractive > result.log 2>&1; exit $LASTEXITCODE"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    log = (tmp_path / "result.log").read_text(encoding="utf-16")
    assert "native finished" in log, result.stderr + log
    assert result.returncode == exit_code, result.stderr + log


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell launcher")
@pytest.mark.parametrize("exit_code,preflight_code", [(0, 0), (7, 0), (0, 2)])
def test_run_redirected_native_stderr_preserves_failure_boundary(tmp_path, exit_code, preflight_code):
    powershell = shutil.which("powershell.exe")
    if not powershell:
        pytest.skip("Windows PowerShell is unavailable")
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "run.ps1", tmp_path / "run.ps1")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(tmp_path / ".venv")], check=True)
    (tmp_path / ".venv/modiff-profile.json").write_text('{"profile":"cpu"}', encoding="utf-8")
    module = tmp_path / "modiff"
    module.mkdir()
    (module / "__init__.py").touch()
    (module / "hardware.py").write_text(
        "import sys\nprint('device progress', file=sys.stderr)\n"
        "def get_hardware_snapshot(**kwargs): return {}\n", encoding="utf-8",
    )
    (module / "runtime_profile.py").write_text(
        f"def runtime_profile(snapshot): return {{'execution_ready': {preflight_code == 0}}}\n",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        "import sys\nprint('worker progress', file=sys.stderr)\n"
        f"print('worker finished')\nsys.exit({exit_code})\n", encoding="utf-8",
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command",
         "& ./run.ps1 > result.log 2>&1; exit $LASTEXITCODE"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    log = (tmp_path / "result.log").read_text(encoding="utf-16")
    if preflight_code:
        assert result.returncode != 0
        assert "worker finished" not in log
        assert "not execution-ready" in result.stderr + log
    else:
        assert "worker finished" in log, result.stderr + log
        assert result.returncode == exit_code, result.stderr + log
