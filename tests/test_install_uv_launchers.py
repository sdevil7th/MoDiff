"""Run public installer bootstraps without downloads or application execution."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _write_command(path, content):
    path.write_text(content, encoding="utf-8", newline="\n")
    if os.name != "nt":
        path.chmod(0o755)


def _launcher_fixture(tmp_path, *, python=True, path_uv=True, version="uv 0.12.23", run_code=0, probe_code=0):
    root = tmp_path / "backend checkout"
    root.mkdir()
    package = root / "modiff"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(ROOT / "modiff/tool_locks.py", package / "tool_locks.py")
    (package / "install.py").write_text(
        "from pathlib import Path\nfrom modiff.tool_locks import resolve_uv\n"
        "print('DIRECT_INSTALL')\nprint(resolve_uv(Path('.modiff')))\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(PATH=str(root), PATHEXT=".COM;.EXE;.BAT;.CMD",
                       PYTHONDONTWRITEBYTECODE="1", UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never")
    environment.pop("PYTHON", None)
    if os.name == "nt":
        shell = shutil.which("powershell.exe")
        if not shell:
            pytest.skip("Windows PowerShell is unavailable")
        shutil.copyfile(ROOT / "install.ps1", root / "install.ps1")
        command = [shell, "-NoProfile", "-Command",
                   f"& '{root / 'install.ps1'}' -BackendOnly -NonInteractive; exit $LASTEXITCODE"]
        if python:
            _write_command(root / "python.cmd", f'@echo off\n"{sys.executable}" %*\nexit /b %errorlevel%\n')
        if path_uv:
            batch_version = version.replace("(", "^(").replace(")", "^)")
            _write_command(root / "uv.cmd", f'@echo off\nif "%~1"=="--version" (\necho {batch_version}\nexit /b {probe_code}\n)\n'
                           f'echo UV_BOOTSTRAP %*\necho native progress 1>&2\nexit /b {run_code}\n')
    else:
        shell = shutil.which("bash")
        if not shell:
            pytest.skip("Bash is unavailable")
        dirname = shutil.which("dirname")
        assert dirname, "The public Bash launcher needs the ordinary dirname utility"
        shutil.copy2(dirname, root / "dirname")
        shutil.copyfile(ROOT / "install.sh", root / "install.sh")
        command = [shell, str(root / "install.sh"), "--backend-only", "--non-interactive"]
        if python:
            _write_command(root / "python3", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        if path_uv:
            _write_command(root / "uv", f'#!/bin/sh\nif [ "$1" = "--version" ]; then\nprintf "%s\\n" "{version}"\nexit {probe_code}\nfi\n'
                           f'printf "UV_BOOTSTRAP %s\\n" "$*"\nprintf "native progress\\n" >&2\nexit {run_code}\n')
    return root, command, environment


def _run_fixture(root, command, environment):
    return subprocess.run(command, cwd=root.parent, env=environment,
                          capture_output=True, text=True, timeout=30)


def _assert_diagnostic(result, expected):
    """Check diagnostic words separately from terminal line wrapping."""
    assert expected in " ".join(result.stderr.split()), result.stderr


@pytest.mark.parametrize("message,expected", [
    ("native progress\n", "native progress"),
    ("checkout/install.ps1 : Install uv from the documented source and add it to \nPATH.\n", "add it to PATH"),
    ("checkout/install.ps1 : Install\r\n  uv from the documented source.\n", "Install uv"),
    ("The command did not report a uv\nversion.\n", "did not report a uv version"),
    ("Cannot run\r\nuv.\n", "Cannot run uv"),
    ("native\nprogress\n", "native progress"),
])
def test_diagnostic_assertions_accept_line_wrapping_without_changing_words(message, expected):
    result = subprocess.CompletedProcess([], 2, stdout="", stderr=message)
    _assert_diagnostic(result, expected)


@pytest.mark.parametrize("message", ["add it to SYSTEM_PATH", "Install something else", ""])
def test_diagnostic_assertions_still_reject_different_or_missing_guidance(message):
    result = subprocess.CompletedProcess([], 2, stdout="", stderr=message)
    with pytest.raises(AssertionError):
        _assert_diagnostic(result, "add it to PATH")


@pytest.mark.parametrize("version", ["uv 0.12.23", "uv 9.0.1 (operator upgrade)"])
def test_installer_validates_operator_uv_then_preserves_isolated_bootstrap(tmp_path, version):
    root, command, environment = _launcher_fixture(tmp_path, version=version)
    result = _run_fixture(root, command, environment)
    assert result.returncode == 0, result.stderr
    assert "UV_BOOTSTRAP run --no-project --python 3.12 python -m modiff.install" in result.stdout
    assert "--backend-only" in result.stdout and "--non-interactive" in result.stdout
    _assert_diagnostic(result, "native progress")
    assert "DIRECT_INSTALL" not in result.stdout


@pytest.mark.parametrize("version", ["forged", "uv unavailable"])
def test_installer_rejects_malformed_path_uv_before_bootstrap_even_with_valid_legacy(tmp_path, version):
    installed_uv = shutil.which("uv")
    root, command, environment = _launcher_fixture(tmp_path, version=version)
    if installed_uv:
        legacy = root / ".modiff/tools/uv" / ("uv.exe" if os.name == "nt" else "uv")
        legacy.parent.mkdir(parents=True)
        shutil.copy2(installed_uv, legacy)
    result = _run_fixture(root, command, environment)
    assert result.returncode != 0
    _assert_diagnostic(result, "did not report a uv version")
    assert "UV_BOOTSTRAP" not in result.stdout
    assert "DIRECT_INSTALL" not in result.stdout


@pytest.mark.parametrize("run_code", [0, 7])
def test_installer_keeps_path_uv_bootstrap_without_python_and_its_exit_code(tmp_path, run_code):
    root, command, environment = _launcher_fixture(tmp_path, python=False, run_code=run_code)
    result = _run_fixture(root, command, environment)
    assert result.returncode == run_code, result.stderr
    assert "UV_BOOTSTRAP run --no-project --python 3.12 python" in result.stdout
    _assert_diagnostic(result, "native progress")


def test_installer_uses_canonical_legacy_resolution_with_python(tmp_path):
    installed_uv = shutil.which("uv")
    if not installed_uv:
        pytest.skip("An operator uv is required for a genuine legacy --version probe")
    root, command, environment = _launcher_fixture(tmp_path, path_uv=False)
    legacy = root / ".modiff/tools/uv" / ("uv.exe" if os.name == "nt" else "uv")
    legacy.parent.mkdir(parents=True)
    shutil.copy2(installed_uv, legacy)
    result = _run_fixture(root, command, environment)
    assert result.returncode == 0, result.stderr
    assert "DIRECT_INSTALL" in result.stdout
    assert str(legacy.resolve()) in result.stdout


def test_failed_path_uv_version_probe_does_not_start_bootstrap(tmp_path):
    root, command, environment = _launcher_fixture(tmp_path, probe_code=9)
    result = _run_fixture(root, command, environment)
    assert result.returncode != 0
    _assert_diagnostic(result, "Cannot run uv")
    assert "UV_BOOTSTRAP" not in result.stdout


@pytest.mark.parametrize("python", [False, True])
def test_installer_missing_uv_reports_operator_prerequisite(tmp_path, python):
    root, command, environment = _launcher_fixture(tmp_path, python=python, path_uv=False)
    result = _run_fixture(root, command, environment)
    assert result.returncode != 0
    _assert_diagnostic(result, "Install uv")
    assert "UV_BOOTSTRAP" not in result.stdout


def test_legacy_uv_without_bootstrap_python_remains_a_clear_path_prerequisite(tmp_path):
    root, command, environment = _launcher_fixture(tmp_path, python=False, path_uv=False)
    legacy = root / ".modiff/tools/uv" / ("uv.exe" if os.name == "nt" else "uv")
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"must not execute an unchecked legacy binary")
    result = _run_fixture(root, command, environment)
    assert result.returncode == 2
    _assert_diagnostic(result, "add it to PATH")
    assert "UV_BOOTSTRAP" not in result.stdout


def test_unusable_python_does_not_block_operator_uv_bootstrap(tmp_path):
    root, command, environment = _launcher_fixture(tmp_path)
    python_path = root / ("python.cmd" if os.name == "nt" else "python3")
    _write_command(python_path, "@echo off\nexit /b 9\n" if os.name == "nt" else "#!/bin/sh\nexit 9\n")
    result = _run_fixture(root, command, environment)
    assert result.returncode == 0, result.stderr
    assert "UV_BOOTSTRAP run --no-project --python 3.12 python" in result.stdout
