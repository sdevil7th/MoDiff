"""Installer recovery from toolchains copied from another platform."""

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from modiff import install


@pytest.mark.parametrize(
    "platform_name,expected,foreign",
    [
        ("windows", "uv.exe", "uv-x86_64-unknown-linux-gnu/uv"),
        ("linux", "uv-x86_64-unknown-linux-gnu/uv", "uv.exe"),
        ("macos", "uv-x86_64-apple-darwin/uv", "uv-x86_64-unknown-linux-gnu/uv"),
    ],
)
def test_uv_downloads_current_platform_instead_of_reusing_foreign_binary(
    tmp_path, monkeypatch, platform_name, expected, foreign
):
    managed = tmp_path / "tools" / "uv"
    old = managed / foreign
    old.parent.mkdir(parents=True)
    old.write_bytes(b"foreign binary")
    binary = b"reviewed current-platform binary"
    lock = {
        "executable": expected,
        "executableSha256": hashlib.sha256(binary).hexdigest(),
        "archiveSha256": "a" * 64,
    }
    monkeypatch.setattr(install, "MANAGED_ROOT", tmp_path)
    monkeypatch.setattr(install, "normalized_os", lambda: platform_name)
    monkeypatch.setattr(install, "normalized_arch", lambda: "x86_64")
    monkeypatch.setattr(install, "UV_TOOL_LOCKS", {(platform_name, "x86_64"): lock})
    downloads = []

    def download(name):
        downloads.append(name)
        executable = managed / expected
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(binary)
        return managed

    monkeypatch.setattr(install, "_download_tool", download)
    assert Path(install._ensure_uv()) == managed / expected
    assert downloads == ["uv"]
    receipt = json.loads((managed / "receipt.json").read_text(encoding="utf-8"))
    assert receipt == {"schemaVersion": 1, **lock}
    # A verified current-platform executable is reused without downloading.
    assert Path(install._ensure_uv()) == managed / expected
    assert downloads == ["uv"]


def test_uv_still_rejects_tampered_current_platform_binary(tmp_path, monkeypatch):
    managed = tmp_path / "tools" / "uv"
    managed.mkdir(parents=True)
    (managed / "uv.exe").write_bytes(b"tampered")
    monkeypatch.setattr(install, "MANAGED_ROOT", tmp_path)
    monkeypatch.setattr(install, "normalized_os", lambda: "windows")
    monkeypatch.setattr(install, "normalized_arch", lambda: "x86_64")
    monkeypatch.setattr(install, "_download_tool", lambda _: pytest.fail("Must reject tampering"))
    with pytest.raises(RuntimeError, match="integrity check"):
        install._ensure_uv()
    assert not (managed / "receipt.json").exists()


@pytest.mark.parametrize("tampered", [False, True])
def test_uv_verifies_archive_and_extracted_binary_during_platform_replacement(tmp_path, monkeypatch, tampered):
    managed = tmp_path / "tools" / "uv"
    foreign = managed / "linux" / "uv"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"linux")
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    archive = downloads / "uv.zip"
    reviewed = b"reviewed windows binary"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("uv.exe", b"tampered" if tampered else reviewed)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    url = "https://github.com/astral-sh/uv/releases/download/test/uv.zip"
    lock = {
        "url": url, "archiveSha256": archive_hash, "executable": "uv.exe",
        "executableSha256": hashlib.sha256(reviewed).hexdigest(),
    }
    monkeypatch.setattr(install, "MANAGED_ROOT", tmp_path)
    monkeypatch.setattr(install, "normalized_os", lambda: "windows")
    monkeypatch.setattr(install, "normalized_arch", lambda: "x86_64")
    monkeypatch.setattr(install, "UV_TOOL_LOCKS", {("windows", "x86_64"): lock})
    monkeypatch.setattr(install, "TOOL_ARCHIVES", {("windows", "x86_64", "uv"): (url, archive_hash)})
    monkeypatch.setattr(install.urllib.request, "urlopen", lambda _: pytest.fail("Use verified cached archive"))
    if tampered:
        with pytest.raises(RuntimeError, match="integrity check"):
            install._ensure_uv()
        assert not (managed / "receipt.json").exists()
    else:
        assert Path(install._ensure_uv()).read_bytes() == reviewed
    assert not foreign.exists()


def test_unsupported_uv_platform_fails_before_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "MANAGED_ROOT", tmp_path)
    monkeypatch.setattr(install, "normalized_arch", lambda: "unsupported")
    monkeypatch.setattr(install, "_download_tool", lambda _: pytest.fail("No reviewed lock"))
    with pytest.raises(RuntimeError, match="reviewed platform lock"):
        install._ensure_uv()


def test_linked_uv_directory_is_rejected_before_download(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    try:
        (tools / "uv").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires permission on this host")
    monkeypatch.setattr(install, "MANAGED_ROOT", tmp_path)
    monkeypatch.setattr(install, "_download_tool", lambda _: pytest.fail("Must reject linked directory"))
    with pytest.raises(RuntimeError, match="directory is unsafe"):
        install._ensure_uv()


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
