"""Backend-authoritative saved workflow documents for every local frontend."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any


_LOCK = threading.RLock()
_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")


def _root(data_dir: str | Path) -> Path:
    return Path(data_dir) / "user-workflows"


def _workflow_id(value: Any) -> str:
    identifier = str(value or "")
    if not _ID.fullmatch(identifier):
        raise ValueError("Workflow id must contain only letters, numbers, underscores, and hyphens.")
    return identifier


def _path(data_dir: str | Path, workflow_id: Any) -> Path:
    return _root(data_dir) / f"{_workflow_id(workflow_id)}.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("snapshot"), dict):
        raise ValueError(f"Saved workflow {path.name} is invalid.")
    return value


def list_workflows(data_dir: str | Path) -> list[dict[str, Any]]:
    root = _root(data_dir)
    if not root.exists():
        return []
    with _LOCK:
        records = []
        for path in root.glob("*.json"):
            try:
                records.append(_read(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return sorted(records, key=lambda item: float(item.get("updatedAt") or 0), reverse=True)


def get_workflow(data_dir: str | Path, workflow_id: Any) -> dict[str, Any] | None:
    path = _path(data_dir, workflow_id)
    if not path.is_file():
        return None
    with _LOCK:
        return _read(path)


def save_workflow(data_dir: str | Path, workflow_id: Any, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("snapshot"), dict):
        raise ValueError("Saved workflow payload requires a snapshot object.")
    path = _path(data_dir, workflow_id)
    now = int(time.time() * 1000)
    with _LOCK:
        existing = _read(path) if path.is_file() else None
        record = {
            "id": _workflow_id(workflow_id),
            "title": str(payload.get("title") or "Workflow").strip()[:160] or "Workflow",
            "snapshot": payload["snapshot"],
            "source": payload.get("source") if isinstance(payload.get("source"), str) else "manual",
            "sourceLabel": payload.get("sourceLabel") if isinstance(payload.get("sourceLabel"), str) else None,
            "createdAt": int((existing or {}).get("createdAt") or payload.get("createdAt") or now),
            "updatedAt": now,
            "revision": int((existing or {}).get("revision") or 0) + 1,
            "clientId": str(payload.get("clientId") or ""),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)
        return record


def delete_workflow(data_dir: str | Path, workflow_id: Any) -> bool:
    path = _path(data_dir, workflow_id)
    with _LOCK:
        existed = path.is_file()
        path.unlink(missing_ok=True)
        return existed
