"""Locate an operator-installed uv without coupling it to a release pin."""

from __future__ import annotations

from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys


def _regular_managed_path(path: Path, root: Path) -> bool:
    """Reject links/reparse points in a legacy app-local tool path."""
    try:
        relative = path.relative_to(root)
        current = root
        for part in (None, *relative.parts):
            if part is not None:
                current /= part
            details = current.lstat()
            if stat.S_ISLNK(details.st_mode) or bool(
                getattr(details, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            ):
                return False
        return stat.S_ISREG(details.st_mode)
    except (OSError, ValueError):
        return False


def _uv_version(executable: Path) -> str:
    try:
        result = subprocess.run(
            [str(executable), "--version"], check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot run uv at {executable}. Reinstall uv using its official installer.") from exc
    match = re.fullmatch(r"uv (\d+\.\d+\.\d+)(?:\s+.*)?", result.stdout.strip())
    if not match:
        raise RuntimeError(f"The executable at {executable} did not report a uv version.")
    return match.group(1)


def resolve_uv(managed_root: Path, *, platform_name: str | None = None, machine: str | None = None) -> str:
    """Prefer PATH; permit an existing current-host app-local uv as a fallback.

    PATH is controlled by the operator. App-local compatibility paths are
    bounded to this host and do not follow filesystem links. No downloaded
    archive, stale receipt or fixed executable hash prevents a uv upgrade.
    """
    installed = shutil.which("uv")
    if installed:
        executable = Path(installed).resolve(strict=True)
        _uv_version(executable)
        return str(executable)
    platform_name = platform_name or ("windows" if sys.platform == "win32" else "macos" if sys.platform == "darwin" else "linux")
    machine = machine or platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    target = {
        ("linux", "x86_64"): "uv-x86_64-unknown-linux-gnu/uv",
        ("linux", "arm64"): "uv-aarch64-unknown-linux-gnu/uv",
        ("macos", "x86_64"): "uv-x86_64-apple-darwin/uv",
        ("macos", "arm64"): "uv-aarch64-apple-darwin/uv",
        ("windows", "x86_64"): "uv.exe",
        ("windows", "arm64"): "uv.exe",
    }.get((platform_name, machine))
    root = Path(managed_root)
    candidates = [root / "tools" / "uv" / target] if target else []
    if platform_name != "windows" and target:
        candidates.append(root / "tools" / "uv" / "uv")
    for executable in candidates:
        if _regular_managed_path(executable, root):
            _uv_version(executable)
            return str(executable.resolve(strict=True))
    raise RuntimeError("Install uv from https://docs.astral.sh/uv/getting-started/installation/ and add it to PATH.")
