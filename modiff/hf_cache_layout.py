"""Finite snapshot/blob containment for the supported Hugging Face cache layouts."""
from pathlib import Path, PureWindowsPath
import os
import re


class HuggingFaceCacheLayoutError(ValueError):
    """An installed snapshot violates the finite, owned cache layout contract."""


def _require_unlinked_directory(path):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise HuggingFaceCacheLayoutError("Installed Hugging Face cache directories must not be linked.")
    if not path.is_dir():
        raise HuggingFaceCacheLayoutError("Installed Hugging Face cache directory is missing.")


def _lexical_cache_path(path):
    """Compare Windows substitute-path spelling without following another link."""
    if isinstance(path, PureWindowsPath):
        value = str(path)
        if value[:8].casefold() == "\\\\?\\unc\\":
            return PureWindowsPath("\\\\" + value[8:])
        if value.startswith("\\\\?\\") and re.match(r"[A-Za-z]:\\", value[4:]):
            return PureWindowsPath(value[4:])
    return path


def _flat_repository_blob_path(target, own_blobs):
    """Check lexical bridge ownership without replacing its filesystem path."""
    try:
        bridge = _lexical_cache_path(target).relative_to(_lexical_cache_path(own_blobs))
    except ValueError as error:
        raise HuggingFaceCacheLayoutError("Installed Hugging Face snapshot links must target their own blob or an allowed shared blob.") from error
    if len(bridge.parts) != 1 or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", bridge.name) is None:
        raise HuggingFaceCacheLayoutError("Installed Hugging Face repository blob bridge is not the reviewed flat layout.")
    return bridge


def resolve_snapshot_cache_file(alias, *, snapshot, cache_root, repository):
    """Resolve a snapshot file into its own blobs or Hub's shared sharded blobs.

    Shared blob names are Hub storage identities, not content SHA-256 values.
    Consumers with reviewed artifact digests must validate the actual bytes.
    """
    alias, snapshot = Path(alias).absolute(), Path(snapshot).absolute()
    cache_root = Path(cache_root).resolve(strict=True)
    repo_cache = snapshot.parent.parent
    if snapshot.parent.name != "snapshots" or repo_cache.name != "models--" + repository.replace("/", "--"):
        raise HuggingFaceCacheLayoutError("Installed Hugging Face snapshot does not preserve its repository cache.")
    if repo_cache.parent.resolve(strict=True) != cache_root:
        raise HuggingFaceCacheLayoutError("Installed Hugging Face snapshot is outside its managed cache root.")
    for directory in (repo_cache, snapshot.parent, snapshot):
        _require_unlinked_directory(directory)
    try:
        relative = alias.relative_to(snapshot)
    except ValueError as error:
        raise HuggingFaceCacheLayoutError("Installed Hugging Face file is outside its exact snapshot.") from error
    for directory in reversed(alias.parent.parents):
        if directory == snapshot or directory.is_relative_to(snapshot):
            _require_unlinked_directory(directory)
    _require_unlinked_directory(alias.parent)
    resolved = alias.resolve(strict=True)
    if not resolved.is_file():
        raise HuggingFaceCacheLayoutError("Installed Hugging Face snapshot entry is not a regular file.")
    if not alias.is_symlink() and not getattr(alias, "is_junction", lambda: False)():
        if resolved != snapshot.resolve(strict=True) / relative:
            raise HuggingFaceCacheLayoutError("Installed Hugging Face snapshot file escaped its own directory.")
        return resolved

    own_blobs = repo_cache.resolve(strict=True) / "blobs"
    if own_blobs.is_symlink() or getattr(own_blobs, "is_junction", lambda: False)():
        raise HuggingFaceCacheLayoutError("Installed Hugging Face repository blobs directory must not be linked.")

    # Hub may migrate an own-repository flat blob into its shared store while
    # preserving the snapshot link. Permit precisely that one bridge; foreign
    # aliases, extra hops, and linked directories cannot borrow containment.
    if alias.is_symlink():
        direct_target = Path(os.path.abspath(alias.parent / os.readlink(alias)))
        if _lexical_cache_path(direct_target) != _lexical_cache_path(resolved):
            _flat_repository_blob_path(direct_target, own_blobs)
            _require_unlinked_directory(own_blobs)
            if not direct_target.is_symlink():
                raise HuggingFaceCacheLayoutError("Installed Hugging Face repository blob bridge is not the reviewed flat layout.")
            shared_target = Path(os.path.abspath(direct_target.parent / os.readlink(direct_target)))
            if (
                _lexical_cache_path(shared_target) != _lexical_cache_path(resolved)
                or not resolved.is_relative_to(cache_root / "blobs")
            ):
                raise HuggingFaceCacheLayoutError("Installed Hugging Face repository blobs must bridge directly to an allowed shared blob.")
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
        raise HuggingFaceCacheLayoutError("Installed Hugging Face snapshot target is outside its allowed blob cache.") from error
    if (
        len(shared.parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", shared.name) is None
        or shared.parts[0] != shared.name[:2]
    ):
        raise HuggingFaceCacheLayoutError("Installed Hugging Face shared blob path is not the reviewed sharded layout.")
    _require_unlinked_directory(cache_root / "blobs")
    _require_unlinked_directory(cache_root / "blobs" / shared.parts[0])
    return resolved
