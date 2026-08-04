"""Small file-backed media manifest for retained, user-cleanable intermediates."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from modiff.config import CONFIG
from modiff.path_identifiers import resolve_runtime_input_path


_LOCK = threading.RLock()


def asset_root(root: str | os.PathLike[str] | None = None) -> Path:
    return Path(root or (Path(CONFIG.paths["temp"]) / "media_assets")).resolve()


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _read(root: Path) -> dict[str, Any]:
    path = _manifest_path(root)
    if not path.is_file():
        return {"schema_version": 1, "assets": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "assets": {}}
    return {
        "schema_version": 1,
        "assets": value.get("assets") if isinstance(value.get("assets"), dict) else {},
    }


def _write(root: Path, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(root)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def allocate_video_path(*, task_id: str | None = None, suffix: str = ".mp4", root=None) -> tuple[str, Path]:
    root_path = asset_root(root)
    asset_id = uuid.uuid4().hex
    task_part = str(task_id or "session").replace("/", "_").replace("\\", "_")
    destination = root_path / task_part / f"{asset_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    return asset_id, destination


def register_video_asset(
    path: str | os.PathLike[str],
    *,
    asset_id: str | None = None,
    task_id: str | None = None,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    temporary: bool = True,
    pinned: bool = False,
    source_asset_ids: list[str] | None = None,
    operation: str | None = None,
    root=None,
) -> dict[str, Any]:
    root_path = asset_root(root)
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Media asset file does not exist: {resolved}")
    now = time.time()
    record = {
        "schema_version": 1,
        "asset_id": str(asset_id or uuid.uuid4().hex),
        "storage": "file",
        "path": str(resolved),
        "media_type": "video",
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "duration_seconds": float(frame_count / fps) if fps else 0.0,
        "task_id": str(task_id) if task_id else None,
        "temporary": bool(temporary),
        "pinned": bool(pinned),
        "created_at": now,
        "updated_at": now,
    }
    if source_asset_ids:
        record["source_asset_ids"] = [str(value) for value in source_asset_ids]
    if operation:
        record["operation"] = str(operation)
    with _LOCK:
        manifest = _read(root_path)
        manifest["assets"][record["asset_id"]] = record
        _write(root_path, manifest)
    return record


def current_task_id() -> str | None:
    """Return the active task identity without making media helpers server-dependent."""
    try:
        from modiff.server import server

        value = (server.current_task or {}).get("task_id")
        return str(value) if value else None
    except Exception:
        return None


def coerce_video_asset(value: Any) -> dict[str, Any]:
    """Normalize a retained asset record or file path to file-backed video metadata."""
    if isinstance(value, dict):
        path_value = value.get("path") or value.get("file")
        if not path_value:
            raise ValueError("Video asset records must contain a path or file value.")
        path = resolve_runtime_input_path(str(path_value)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video asset file does not exist: {path}")
        record = dict(value)
        record.update({"storage": "file", "path": str(path), "media_type": "video"})
        required = ("width", "height", "fps", "frame_count", "duration_seconds")
        if any(record.get(key) is None for key in required):
            record.update(probe_video_file(path))
        return record
    if isinstance(value, (str, os.PathLike)) and str(value):
        path = resolve_runtime_input_path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")
        return {
            "schema_version": 1,
            "asset_id": None,
            "storage": "file",
            "path": str(path),
            "media_type": "video",
            **probe_video_file(path),
        }
    raise TypeError("Expected a retained video asset record or an existing video file path.")


def probe_video_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read video dimensions and timing without materializing the frame sequence."""
    import imageio.v2 as imageio

    resolved = Path(path).expanduser().resolve()
    reader = imageio.get_reader(str(resolved), "ffmpeg")
    try:
        metadata = reader.get_meta_data()
        width, height = metadata.get("size", (0, 0))
        fps = float(metadata.get("fps") or 0)
        duration = float(metadata.get("duration") or 0)
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            frame_count = int(round(duration * fps)) if duration and fps else 0
    finally:
        reader.close()
    if not duration and fps and frame_count:
        duration = frame_count / fps
    return {
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def run_ffmpeg(arguments: list[str], destination: str | os.PathLike[str]) -> Path:
    """Run the bundled FFmpeg executable and never leave a partial destination."""
    from imageio_ffmpeg import get_ffmpeg_exe

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [get_ffmpeg_exe(), "-y", "-v", "error", *map(str, arguments), str(output)]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode:
            detail = (result.stderr or result.stdout or "unknown FFmpeg error").strip()
            raise RuntimeError(f"FFmpeg failed while creating {output.name}: {detail}")
        if not output.is_file():
            raise RuntimeError(f"FFmpeg reported success but did not create {output}.")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output


def register_derived_video_asset(
    destination: str | os.PathLike[str],
    *,
    asset_id: str | None = None,
    task_id: str | None = None,
    source_assets: list[dict[str, Any]] | None = None,
    operation: str,
    pinned: bool = False,
    root=None,
) -> dict[str, Any]:
    """Probe and register the output of a file-native media operation."""
    metadata = probe_video_file(destination)
    source_ids = [str(item["asset_id"]) for item in (source_assets or []) if item.get("asset_id")]
    return register_video_asset(
        destination,
        asset_id=asset_id,
        task_id=task_id,
        width=metadata["width"],
        height=metadata["height"],
        fps=metadata["fps"],
        frame_count=metadata["frame_count"],
        temporary=True,
        pinned=pinned,
        source_asset_ids=source_ids,
        operation=operation,
        root=root,
    )


def list_media_assets(*, root=None) -> list[dict[str, Any]]:
    root_path = asset_root(root)
    with _LOCK:
        records = list(_read(root_path)["assets"].values())
    return sorted(records, key=lambda item: float(item.get("created_at") or 0), reverse=True)


def cleanup_media_assets(
    *,
    task_id: str | None = None,
    older_than_seconds: float | None = None,
    include_pinned: bool = False,
    root=None,
) -> dict[str, Any]:
    root_path = asset_root(root)
    cutoff = time.time() - max(0.0, float(older_than_seconds)) if older_than_seconds is not None else None
    removed = []
    errors = []
    with _LOCK:
        manifest = _read(root_path)
        retained = {}
        for asset_id, record in manifest["assets"].items():
            selected = bool(record.get("temporary", False))
            if task_id is not None:
                selected = selected and record.get("task_id") == str(task_id)
            if cutoff is not None:
                selected = selected and float(record.get("created_at") or 0) < cutoff
            if record.get("pinned") and not include_pinned:
                selected = False
            if not selected:
                retained[asset_id] = record
                continue
            path = Path(str(record.get("path") or ""))
            try:
                resolved = path.resolve()
                resolved.relative_to(root_path)
                resolved.unlink(missing_ok=True)
                removed.append({"asset_id": asset_id, "path": str(resolved)})
            except Exception as exc:
                errors.append({"asset_id": asset_id, "path": str(path), "error": str(exc)})
                retained[asset_id] = record
        manifest["assets"] = retained
        _write(root_path, manifest)
    return {"removed": removed, "errors": errors, "remaining": len(retained)}
