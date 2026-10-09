"""Exact local artifacts for the mandatory Cosmos safety runtime.

This module imports no model runtime and grants no execution qualification.
The optional runtime and gated artifact access remain separate prerequisites.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from modiff.hf_cache_layout import resolve_snapshot_cache_file


COSMOS_SAFETY_RUNTIME_PROFILE_ID = "cosmos-guardrail-0.3.1"
COSMOS_SAFETY_MODEL_TYPES = frozenset({"Cosmos3OmniModularPipeline", "Cosmos3DistilledModularPipeline"})
COSMOS_SAFETY_BUNDLE_MEMBER = "cosmos_safety_checker"
COSMOS_GUARDRAIL_REPOSITORY = "nvidia/Cosmos-Guardrail1"
COSMOS_GUARDRAIL_REVISION = "d6d4bfa899a71454a700907664f3e88f503950cf"
COSMOS_TEXT_GUARD_REPOSITORY = "Qwen/Qwen3Guard-Gen-0.6B"
COSMOS_TEXT_GUARD_REVISION = "fada3b2f655b89601929198343c94cd2f64d93cc"
_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "data" / "cosmos-safety-artifacts.v1.json"


def cosmos_safety_artifacts():
    value = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
    artifacts = value.get("artifacts") if value.get("schemaVersion") == 1 else None
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("The reviewed Cosmos safety artifact contract is invalid.")
    identities = [(item.get("repository"), item.get("revision")) for item in artifacts]
    if identities != [
        (COSMOS_GUARDRAIL_REPOSITORY, COSMOS_GUARDRAIL_REVISION),
        (COSMOS_TEXT_GUARD_REPOSITORY, COSMOS_TEXT_GUARD_REVISION),
    ]:
        raise ValueError("The reviewed Cosmos safety artifact identities changed.")
    for artifact in artifacts:
        files = artifact.get("files")
        if not isinstance(files, list) or not files or len(files) > 256:
            raise ValueError("The reviewed Cosmos safety file selection is invalid.")
        names = []
        for entry in files:
            name = entry.get("path")
            parts = PurePosixPath(name).parts if isinstance(name, str) else ()
            if (
                not parts or name != PurePosixPath(name).as_posix() or name.startswith("/")
                or any(part in {"", ".", ".."} or ":" in part for part in parts)
                or type(entry.get("byteSize")) is not int or entry["byteSize"] < 0
                or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", str(entry.get("blobHash"))) is None
            ):
                raise ValueError("The reviewed Cosmos safety file metadata is invalid.")
            names.append(name)
        if names != sorted(set(names)) or artifact.get("marker") not in names:
            raise ValueError("The reviewed Cosmos safety file selection is ambiguous.")
    return artifacts


def _alias_identity(value):
    # Reading a valid artifact may update its access time. Keep the alias's
    # identity and mutation metadata without treating that read as a change.
    # Compare values from the same stat API: Windows path stat reports birth
    # time as ctime, while descriptor fstat can report actual change time.
    return (
        value.st_mode, value.st_ino, value.st_dev, value.st_nlink, value.st_uid,
        value.st_gid, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    )


def verify_cosmos_safety_snapshot(snapshot, artifact):
    """Verify exact selected bytes within the supported managed Hub blob layouts.

    The caller resolves this directory with exact_cached_snapshot_path first.
    Every alias and target is rechecked while hashing; a changed or incomplete
    snapshot cannot silently fall back to an online Hub lookup.
    """
    snapshot = Path(snapshot).absolute()
    if snapshot.name != artifact["revision"] or snapshot.parent.name != "snapshots":
        raise ValueError("Cosmos safety requires its exact installed snapshot revision.")
    repository = snapshot.parent.parent
    for directory in (repository, snapshot.parent, snapshot):
        if directory.is_symlink() or getattr(directory, "is_junction", lambda: False)():
            raise ValueError("Cosmos safety snapshot directories must not be linked.")
    for entry in artifact["files"]:
        alias = snapshot.joinpath(*PurePosixPath(entry["path"]).parts)
        resolved = resolve_snapshot_cache_file(
            alias, snapshot=snapshot, cache_root=repository.parent, repository=artifact["repository"],
        )
        if not resolved.is_file():
            raise ValueError("A Cosmos safety artifact is not a regular file.")
        alias_before = alias.lstat()
        before = resolved.stat()
        if before.st_size != entry["byteSize"]:
            raise ValueError(f"Cosmos safety artifact size differs: {entry['path']}")
        algorithm = "sha256" if len(entry["blobHash"]) == 64 else "sha1"
        digest = hashlib.new(algorithm)
        if algorithm == "sha1":
            digest.update(f"blob {before.st_size}\0".encode("ascii"))
        with resolved.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError("A Cosmos safety file changed before validation.")
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        if (
            _alias_identity(opened) != _alias_identity(after)
            or _alias_identity(before) != _alias_identity(resolved.stat())
            or _alias_identity(alias.lstat()) != _alias_identity(alias_before)
            or alias.resolve(strict=True) != resolved
            or digest.hexdigest() != entry["blobHash"]
        ):
            raise ValueError(f"Cosmos safety artifact failed its reviewed digest: {entry['path']}")
    return snapshot
