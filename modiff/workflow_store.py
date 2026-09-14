"""Backend-authoritative saved workflow documents for every local frontend."""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

from modiff.studio_persistence_lock import STUDIO_PERSISTENCE_LOCK


_LOCK = STUDIO_PERSISTENCE_LOCK
_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_SUMMARY_FIELDS = ("id", "title", "source", "sourceLabel", "createdAt", "updatedAt", "revision", "clientId")
# Metadata only, never graph snapshots. File identity detects writes made by
# migration/rollback or another local process without a separate invalidation API.
_SUMMARY_CACHE_LIMIT = 8192
_SUMMARY_CACHE: OrderedDict[Path, tuple[tuple[int, ...], dict[str, Any]]] = OrderedDict()


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


def list_workflow_summaries(data_dir: str | Path) -> list[dict[str, Any]]:
    """List workflow metadata without returning multi-megabyte graph snapshots."""

    root = _root(data_dir).resolve()
    if not root.exists():
        return []
    with _LOCK:
        summaries = []
        for path in root.glob("*.json"):
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                cached = _SUMMARY_CACHE.get(path)
                if cached is None or cached[0] != identity:
                    record = _read(path)
                    summary = {key: record.get(key) for key in _SUMMARY_FIELDS}
                    _SUMMARY_CACHE[path] = (identity, summary)
                    while len(_SUMMARY_CACHE) > _SUMMARY_CACHE_LIMIT:
                        _SUMMARY_CACHE.popitem(last=False)
                else:
                    summary = cached[1]
                _SUMMARY_CACHE.move_to_end(path)
                summaries.append(deepcopy(summary))
            except (OSError, ValueError, json.JSONDecodeError):
                _SUMMARY_CACHE.pop(path, None)
                continue
    return sorted(summaries, key=lambda item: float(item.get("updatedAt") or 0), reverse=True)


def get_workflow(data_dir: str | Path, workflow_id: Any) -> dict[str, Any] | None:
    path = _path(data_dir, workflow_id)
    with _LOCK:
        if not path.is_file():
            return None
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
