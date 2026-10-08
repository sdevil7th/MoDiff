"""Finite snapshot/blob containment for the supported Hugging Face cache layouts."""
from pathlib import Path
import os
import re


def _require_unlinked_directory(path):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError("Installed Hugging Face cache directories must not be linked.")
    if not path.is_dir():
        raise ValueError("Installed Hugging Face cache directory is missing.")


def resolve_snapshot_cache_file(alias, *, snapshot, cache_root, repository):
    """Resolve a snapshot file into its own blobs or Hub's shared sharded blobs.

    Shared blob names are Hub storage identities, not content SHA-256 values.
    Consumers with reviewed artifact digests must validate the actual bytes.
    """
    alias, snapshot = Path(alias).absolute(), Path(snapshot).absolute()
    cache_root = Path(cache_root).resolve(strict=True)
    repo_cache = snapshot.parent.parent
    if snapshot.parent.name != "snapshots" or repo_cache.name != "models--" + repository.replace("/", "--"):
        raise ValueError("Installed Hugging Face snapshot does not preserve its repository cache.")
    if repo_cache.parent.resolve(strict=True) != cache_root:
        raise ValueError("Installed Hugging Face snapshot is outside its managed cache root.")
    for directory in (repo_cache, snapshot.parent, snapshot):
        _require_unlinked_directory(directory)
    try:
        relative = alias.relative_to(snapshot)
    except ValueError as error:
        raise ValueError("Installed Hugging Face file is outside its exact snapshot.") from error
    for directory in reversed(alias.parent.parents):
        if directory == snapshot or directory.is_relative_to(snapshot):
            _require_unlinked_directory(directory)
    _require_unlinked_directory(alias.parent)
    resolved = alias.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Installed Hugging Face snapshot entry is not a regular file.")
    if not alias.is_symlink() and not getattr(alias, "is_junction", lambda: False)():
        if resolved != snapshot.resolve(strict=True) / relative:
            raise ValueError("Installed Hugging Face snapshot file escaped its own directory.")
        return resolved

    # Official snapshot links point directly at a blob. Do not let a foreign
    # repository alias (or another link) borrow the shared blob's containment.
    if alias.is_symlink():
        direct_target = Path(os.path.abspath(alias.parent / os.readlink(alias)))
        if direct_target != resolved:
            raise ValueError("Installed Hugging Face snapshot links must point directly at an allowed blob.")

    own_blobs = repo_cache.resolve(strict=True) / "blobs"
    if own_blobs.is_symlink() or getattr(own_blobs, "is_junction", lambda: False)():
        raise ValueError("Installed Hugging Face repository blobs directory must not be linked.")
    if resolved.is_relative_to(own_blobs):
        _require_unlinked_directory(own_blobs)
        directory = own_blobs
        for part in resolved.parent.relative_to(own_blobs).parts:
            directory /= part
            _require_unlinked_directory(directory)
        return resolved

    try:
        shared = resolved.relative_to(cache_root / "blobs")
    except ValueError as error:
        raise ValueError("Installed Hugging Face snapshot target is outside its allowed blob cache.") from error
    if (
        len(shared.parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", shared.name) is None
        or shared.parts[0] != shared.name[:2]
    ):
        raise ValueError("Installed Hugging Face shared blob path is not the reviewed sharded layout.")
    _require_unlinked_directory(cache_root / "blobs")
    _require_unlinked_directory(cache_root / "blobs" / shared.parts[0])
    return resolved
