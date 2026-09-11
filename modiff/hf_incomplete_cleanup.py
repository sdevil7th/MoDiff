"""Fail-closed cleanup plans for stale Hugging Face ``.incomplete`` blobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable


STALE_AFTER_SECONDS = 60 * 60


class HfIncompleteCleanupError(ValueError):
    pass


def _cache_roots(models: Iterable[dict[str, Any]]) -> list[Path]:
    roots: dict[str, Path] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        values = [model.get("cache_dir"), *(model.get("cache_dirs") or [])]
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            root = Path(value).expanduser().resolve()
            roots.setdefault(str(root), root)
    return list(roots.values())


def _plan_hash(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "planHash"}
    payload["files"] = [
        {key: value for key, value in item.items() if key != "ageSeconds"}
        for item in payload.get("files") or []
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _candidate(root: Path, path: Path, *, now: float) -> dict[str, Any] | None:
    if path.is_symlink() or path.name.lower().endswith(".incomplete") is False or path.parent.name != "blobs":
        return None
    try:
        stat = path.stat()
        relative = path.resolve().relative_to(root)
    except (OSError, ValueError):
        return None
    if not path.is_file():
        return None
    age_seconds = max(0, int(now - stat.st_mtime))
    return {
        "cacheRoot": str(root),
        "relativePath": relative.as_posix(),
        "sizeBytes": stat.st_size,
        "modifiedTimeNs": stat.st_mtime_ns,
        "ageSeconds": age_seconds,
        "eligible": age_seconds >= STALE_AFTER_SECONDS,
    }


def build_hf_incomplete_cleanup_plan(
    *,
    models: list[dict[str, Any]],
    current_task: dict[str, Any] | None,
    queued_tasks: dict[str, Any] | list[Any],
    download_repos: dict[str, Any] | set[str] | list[str],
    template_gallery_active: bool,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    files = []
    for root in _cache_roots(models):
        if not root.is_dir():
            continue
        try:
            paths = root.glob("models--*/blobs/*.incomplete")
            files.extend(item for path in paths if (item := _candidate(root, path, now=now)) is not None)
        except OSError:
            continue
    files.sort(key=lambda item: (item["cacheRoot"], item["relativePath"]))
    blockers = []
    if current_task:
        blockers.append({"code": "graph_execution_active"})
    if queued_tasks:
        blockers.append({"code": "graph_execution_queued"})
    if download_repos:
        blockers.append({"code": "model_download_active"})
    if template_gallery_active:
        blockers.append({"code": "template_gallery_install_active"})
    recent_count = sum(not item["eligible"] for item in files)
    if recent_count:
        blockers.append({"code": "recent_incomplete_download", "fileCount": recent_count})
    eligible = [item for item in files if item["eligible"]]
    plan = {
        "schemaVersion": 1,
        "kind": "hf_incomplete_cleanup_plan",
        "files": files,
        "eligibleFileCount": len(eligible),
        "eligibleBytes": sum(item["sizeBytes"] for item in eligible),
        "blockers": blockers,
        "canCleanup": bool(eligible) and not blockers,
    }
    plan["planHash"] = _plan_hash(plan)
    return plan


def cleanup_hf_incomplete_files(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("canCleanup") is not True:
        raise HfIncompleteCleanupError("The cleanup plan is not eligible for execution.")
    removed = []
    removed_bytes = 0
    for item in plan.get("files") or []:
        if not isinstance(item, dict) or item.get("eligible") is not True:
            continue
        root = Path(str(item.get("cacheRoot") or "")).resolve()
        relative = Path(str(item.get("relativePath") or ""))
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise HfIncompleteCleanupError("A cleanup target escaped its cache root.") from error
        if path.is_symlink() or path.parent.name != "blobs" or not path.name.lower().endswith(".incomplete"):
            raise HfIncompleteCleanupError("A cleanup target is no longer a safe incomplete blob.")
        try:
            stat = path.stat()
        except OSError as error:
            raise HfIncompleteCleanupError("A cleanup target disappeared; request a fresh plan.") from error
        if (
            not path.is_file()
            or stat.st_size != item.get("sizeBytes")
            or stat.st_mtime_ns != item.get("modifiedTimeNs")
        ):
            raise HfIncompleteCleanupError("A cleanup target changed; request a fresh plan.")
        path.unlink()
        removed.append(str(path))
        removed_bytes += stat.st_size
    return {"removed": removed, "removedFileCount": len(removed), "removedBytes": removed_bytes}
