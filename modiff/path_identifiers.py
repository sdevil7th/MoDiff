"""Portable identifiers for files managed by MoDiff's configured roots.

The HTTP API cannot safely use ``Path.relative_to(work_dir)`` for uploads
stored under a separately configured data directory.  Besides failing for
separate roots (and for separate drives on Windows), falling back to an
absolute path would disclose a host path and make saved workflows
machine-specific.  ``@data/<relative>`` is therefore the public identifier for
files rooted at ``data_dir``.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


DATA_PATH_PREFIX = "@data"


def _resolved_root(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def is_data_path_identifier(value: object) -> bool:
    try:
        raw = os.fspath(value)
    except TypeError:
        return False
    return isinstance(raw, str) and (raw == DATA_PATH_PREFIX or raw.startswith(f"{DATA_PATH_PREFIX}/"))


def _data_identifier_parts(value: str | os.PathLike[str]) -> tuple[str, ...]:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not is_data_path_identifier(raw):
        raise ValueError("Not a MoDiff data-path identifier.")
    if "\x00" in raw or "\\" in raw:
        raise ValueError("The data-path identifier is invalid.")
    relative = raw.removeprefix(f"{DATA_PATH_PREFIX}/") if raw != DATA_PATH_PREFIX else ""
    if not relative:
        return ()
    raw_parts = relative.split("/")
    if any(part in {"", ".", ".."} or ":" in part for part in raw_parts):
        raise ValueError("The data-path identifier contains an invalid segment.")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure.parts != tuple(raw_parts):
        raise ValueError("The data-path identifier is invalid.")
    return tuple(raw_parts)


def data_path_identifier(
    path: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
) -> str:
    """Return a host-independent identifier for a path contained by data_root."""

    root = _resolved_root(data_root)
    candidate = Path(path).expanduser().resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("The path is outside the configured MoDiff data directory.") from exc
    if relative == Path("."):
        return DATA_PATH_PREFIX
    identifier = f"{DATA_PATH_PREFIX}/{PurePosixPath(*relative.parts).as_posix()}"
    _data_identifier_parts(identifier)
    return identifier


def resolve_data_path_identifier(
    value: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
) -> Path:
    """Resolve an ``@data`` identifier and reject traversal or symlink escape."""

    parts = _data_identifier_parts(value)
    root = _resolved_root(data_root)
    candidate = root.joinpath(*parts).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("The data-path identifier escapes the configured data directory.") from exc
    return candidate


def resolve_managed_path_identifier(
    value: str | os.PathLike[str],
    *,
    work_root: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
) -> Path | None:
    """Resolve an HTTP path value inside the configured work or data root.

    Existing work-root-relative values (including the historical ``data/...``
    form when data lives under the workspace) remain valid.  Absolute values
    are accepted only when already contained by a configured root, which keeps
    old local records readable without issuing new absolute identifiers.
    """

    try:
        raw = os.fspath(value)
    except TypeError:
        return None
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        return None
    if raw == DATA_PATH_PREFIX or raw.startswith(f"{DATA_PATH_PREFIX}/"):
        try:
            return resolve_data_path_identifier(raw, data_root)
        except ValueError:
            return None
    if raw.startswith(DATA_PATH_PREFIX):
        return None

    work = _resolved_root(work_root)
    data = _resolved_root(data_root)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = work / candidate
    candidate = candidate.resolve(strict=False)
    for root in (work, data):
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    return None


def resolve_runtime_input_path(
    value: str | os.PathLike[str],
    *,
    work_root: str | os.PathLike[str] | None = None,
    data_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a node path while preserving legacy local-path behavior.

    Only the server-issued ``@data`` namespace is containment constrained here.
    Existing explicit absolute paths and work-root-relative graph values retain
    their historical trusted-local semantics.
    """

    if work_root is None or data_root is None:
        from modiff.config import CONFIG

        work_root = work_root or CONFIG.paths["work_dir"]
        data_root = data_root or CONFIG.paths["data"]
    raw = os.fspath(value)
    if is_data_path_identifier(raw):
        return resolve_data_path_identifier(raw, data_root)
    if isinstance(raw, str) and raw.startswith(DATA_PATH_PREFIX):
        raise ValueError("The MoDiff data-path identifier is invalid.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(work_root).expanduser() / path
    return path
