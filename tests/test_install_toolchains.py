"""Operator uv discovery and recovery from copied toolchains."""

import json
from pathlib import Path
import subprocess

import pytest

from modiff import install, tool_locks


@pytest.mark.parametrize("platform_name,foreign", [("windows", "uv-linux/uv"), ("linux", "uv.exe"), ("macos", "uv-linux/uv")])
def test_foreign_app_local_uv_is_never_executed(tmp_path, monkeypatch, platform_name, foreign):
    executable = tmp_path / "tools" / "uv" / foreign
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"foreign")
    monkeypatch.setattr(tool_locks.shutil, "which", lambda _: None)
    monkeypatch.setattr(tool_locks, "_uv_version", lambda _: pytest.fail("Do not execute foreign binaries"))
    with pytest.raises(RuntimeError, match="Install uv"):
        tool_locks.resolve_uv(tmp_path, platform_name=platform_name, machine="x86_64")


def test_operator_uv_upgrade_does_not_require_an_old_receipt(tmp_path, monkeypatch):
    executable = tmp_path / "uv"
    executable.write_bytes(b"operator uv")
    monkeypatch.setattr(tool_locks.shutil, "which", lambda _: str(executable))
    observed = []
    monkeypatch.setattr(tool_locks, "_uv_version", lambda path: observed.append(path) or "0.12.23")
    assert tool_locks.resolve_uv(tmp_path) == str(executable)
    assert observed == [executable]


def test_linked_app_local_uv_is_rejected_when_path_uv_is_absent(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    try:
        (tools / "uv").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires permission on this host")
    monkeypatch.setattr(tool_locks.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Install uv"):
        tool_locks.resolve_uv(tmp_path)


def test_uv_must_report_a_real_version(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_locks.subprocess, "run", lambda *args, **kw: subprocess.CompletedProcess(args, 0, stdout="forged", stderr=""))
    with pytest.raises(RuntimeError, match="did not report a uv version"):
        tool_locks._uv_version(tmp_path / "uv")


@pytest.mark.parametrize("stale_action", ["./run.sh", r".\run.ps1"])
def test_failed_install_reports_current_plan_resume_command(tmp_path, monkeypatch, capsys, stale_action):
    journal = tmp_path / "install-state.json"
    journal.write_text(json.dumps({"next_action": stale_action}), encoding="utf-8")
    monkeypatch.setattr(install, "MANAGED_ROOT", tmp_path)
    monkeypatch.setattr(install, "JOURNAL_PATH", journal)
    monkeypatch.setattr(install, "detect_host", lambda: {
        "os": "windows", "architecture": "x86_64", "nvidia_usable": True,
    })

    def fail_uv():
        raise RuntimeError("toolchain failed")

    monkeypatch.setattr(install, "_ensure_uv", fail_uv)
    assert install.main(["--json"]) == 2
    payload = json.loads(capsys.readouterr().err)
    expected = r".\install.ps1 -Accelerator auto -Resume"
    assert payload["resume_command"] == expected
    assert json.loads(journal.read_text(encoding="utf-8"))["next_action"] == expected


def test_windows_node_does_not_pair_linux_binary_with_windows_npm(tmp_path, monkeypatch):
    managed = tmp_path / "tools" / "node"
    old = managed / "node-linux" / "bin" / "node"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"linux node")
    (old.parent / "npm").write_bytes(b"linux npm")
    monkeypatch.setattr(install, "MANAGED_ROOT", tmp_path)
    monkeypatch.setattr(install, "normalized_os", lambda: "windows")
    downloads = []

    def download(name):
        downloads.append(name)
        (managed / "node.exe").write_bytes(b"windows node")
        (managed / "npm.cmd").write_bytes(b"windows npm")
        return managed

    monkeypatch.setattr(install, "_download_tool", download)

    def command(argv):
        assert Path(argv[0]).name == "node.exe"
        return {"returncode": 0, "stdout": "24.12.0", "stderr": ""}

    monkeypatch.setattr(install, "_command", command)
    result = install._ensure_node()
    assert downloads == ["node"]
    assert Path(result["npm"]).name == "npm.cmd"
