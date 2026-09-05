"""Process-start identity for the backend source loaded by a worker.

The public attestation intentionally contains only relative-source hashes folded
into one fingerprint.  The Gallery harness retains the full before/after file
inventory separately and compares it with this worker-owned startup claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Any


_SOURCE_SUFFIX = re.compile(r"\.(?:py|toml|ini)$", re.IGNORECASE)
_PROCESS_IDENTITY_LOCK = threading.Lock()
_PROCESS_IDENTITY: dict[str, Any] | None = None


def _source_files_under(target: Path) -> list[Path]:
    if not target.exists():
        return []
    if target.is_file():
        return [target]

    files: list[Path] = []
    with os.scandir(target) as entries:
        for entry in entries:
            if entry.name == "__pycache__" or entry.name.startswith("."):
                continue
            child = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                files.extend(_source_files_under(child))
            elif _SOURCE_SUFFIX.search(entry.name):
                files.append(child)
    return files


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def backend_source_identity(backend_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return the same canonical source identity used by the client harness."""

    root = Path(backend_root).resolve() if backend_root is not None else Path(__file__).resolve().parents[1]
    targets = [root / "main.py", root / "pyproject.toml", root / "modiff", root / "modules", root / "utils"]
    source_files = sorted(
        {path.resolve() for target in targets for path in _source_files_under(target)},
        key=lambda path: path.as_posix(),
    )
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in source_files
    ]
    payload = {"gitCommit": _git_commit(root), "files": files}
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schemaVersion": 1,
        "claim": "process_start_backend_source_identity",
        "gitCommit": payload["gitCommit"],
        "fingerprint": f"sha256:backend-source-v1:{digest}",
        "fileCount": len(files),
        "capturedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def capture_process_backend_source_identity() -> dict[str, Any]:
    """Capture once in the worker before importing the executable backend."""

    global _PROCESS_IDENTITY
    with _PROCESS_IDENTITY_LOCK:
        if _PROCESS_IDENTITY is None:
            _PROCESS_IDENTITY = backend_source_identity()
        return dict(_PROCESS_IDENTITY)


def process_backend_source_identity() -> dict[str, Any]:
    """Return the immutable worker-start claim, capturing only for embeddings/tests."""

    return capture_process_backend_source_identity()
