"""Immutable, offline source builds for app-owned optional-runtime wheels.

This module extends the reviewed-wheel overlay boundary without changing its
ordinary PyPI policy.  A source assembly is admitted only when the official
commit archive, the no-code deterministic recipe, and the expected output
wheel bytes are all part of a source-controlled optional-runtime
specification.  Upstream build hooks are never imported or executed.  The
derived wheel is then handled by the existing METADATA/WHEEL/RECORD and
extracted-file-seal validators.
"""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import unicodedata
import uuid
from typing import Any
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from modiff.runtime_overlays import (
    InstallLease,
    MAX_LOCKED_ARCHIVE_BYTES,
    OverlayCancelled,
    _wheel_target_path,
    locked_artifact_file_seal,
    locked_artifact_path,
)
import zipfile


SOURCE_BUILD_SCHEMA_VERSION = 1
SOURCE_BUILD_RECIPE = "modiff_pure_python_wheel_v1"
MAX_SOURCE_ARCHIVE_ENTRIES = 100_000
MAX_SOURCE_MEMBER_BYTES = 512 * 1024**2
MAX_SOURCE_TOTAL_BYTES = 2 * 1024**3
MAX_SOURCE_PATH_BYTES = 1024
MAX_SOURCE_DEPTH = 64
MAX_BUILD_OUTPUT_BYTES = 512 * 1024**2
MAX_WHEEL_MEMBERS = 100_000
FIXED_SOURCE_DATE_EPOCH = 315_532_800
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
}


@dataclass(frozen=True)
class SourceBuildResult:
    """One authenticated wheel derived from an immutable source contract."""

    artifact: dict[str, Any]
    path: Path
    receipt: dict[str, Any]


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("A source archive redirect is forbidden.")


def _is_reparse_point(details: os.stat_result) -> bool:
    """Recognize Windows junctions/symlinks without platform branching."""

    return bool(
        getattr(details, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _safe_directory_details(details: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(details.st_mode)
        and not stat.S_ISLNK(details.st_mode)
        and not _is_reparse_point(details)
    )


def _safe_regular_details(details: os.stat_result) -> bool:
    return (
        stat.S_ISREG(details.st_mode)
        and not stat.S_ISLNK(details.st_mode)
        and not _is_reparse_point(details)
        and details.st_nlink == 1
    )


def _canonical_digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


def _normalized_distribution(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(".", "-")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARACTERS)
    )


def _bounded_positive_integer(value: Any, *, maximum: int) -> bool:
    return type(value) is int and 0 < value <= maximum


def _safe_relative_path(value: Any) -> PurePosixPath:
    raw = str(value or "")
    if (
        not raw
        or len(raw.encode("utf-8", errors="ignore")) > MAX_SOURCE_PATH_BYTES
        or "\\" in raw
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise RuntimeError("A source-build contract contains an invalid path.")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or len(path.parts) > MAX_SOURCE_DEPTH
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError("A source-build contract contains a traversal path.")
    for part in path.parts:
        if (
            ":" in part
            or any(character in '<>"|?*' for character in part)
            or part.rstrip(" .") != part
            or part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
        ):
            raise RuntimeError("A source-build contract contains a platform-unsafe path.")
    return path


def _safe_header_value(value: Any, *, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise RuntimeError("A source-build wheel metadata value is invalid.")
    text = value
    if (
        not text
        or len(text.encode("utf-8", errors="ignore")) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise RuntimeError("A source-build wheel metadata value is invalid.")
    return text


def _validate_wheel_metadata(value: Any) -> dict[str, Any]:
    fields = {
        "summary",
        "license",
        "requiresPython",
        "requiresDist",
        "consoleScripts",
        "packageRoot",
        "licenseFile",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeError("The source-build wheel metadata lock is malformed.")
    requirements = value.get("requiresDist")
    scripts = value.get("consoleScripts")
    if (
        not isinstance(requirements, list)
        or len(requirements) > 128
        or not all(isinstance(requirement, str) for requirement in requirements)
        or not isinstance(scripts, dict)
        or len(scripts) > 64
        or not all(
            isinstance(name, str) and isinstance(target, str)
            for name, target in scripts.items()
        )
    ):
        raise RuntimeError("The source-build wheel metadata lock is invalid.")
    normalized_requirements = [
        _safe_header_value(requirement) for requirement in requirements
    ]
    if len(normalized_requirements) != len(set(normalized_requirements)):
        raise RuntimeError("The source-build wheel requirements are not unique.")
    normalized_scripts: dict[str, str] = {}
    for raw_name, raw_target in scripts.items():
        name = raw_name
        target = raw_target
        if (
            not name
            or len(name) > 128
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in name)
            or not target
            or len(target) > 256
            or target.count(":") != 1
            or target.startswith(":")
            or target.endswith(":")
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:" for character in target)
        ):
            raise RuntimeError("A source-build console script lock is invalid.")
        normalized_scripts[name] = target
    return {
        "summary": _safe_header_value(value.get("summary")),
        "license": _safe_header_value(value.get("license"), maximum=128),
        "requiresPython": _safe_header_value(
            value.get("requiresPython"), maximum=128
        ),
        "requiresDist": normalized_requirements,
        "consoleScripts": dict(sorted(normalized_scripts.items())),
        "packageRoot": _safe_relative_path(value.get("packageRoot")).as_posix(),
        "licenseFile": _safe_relative_path(value.get("licenseFile")).as_posix(),
    }


def validate_source_build_contract(contract: Any) -> dict[str, Any]:
    """Return a normalized exact contract or fail before network/filesystem use."""

    if not isinstance(contract, dict):
        raise RuntimeError("The optional-runtime source-build contract is malformed.")
    required = {
        "schemaVersion",
        "distribution",
        "version",
        "sourceArtifact",
        "recipe",
        "pythonTag",
        "sourceDateEpoch",
        "sourceFiles",
        "sourceTrees",
        "buildDependencies",
        "wheelMetadata",
        "outputWheel",
    }
    if set(contract) != required or contract.get("schemaVersion") != SOURCE_BUILD_SCHEMA_VERSION:
        raise RuntimeError("The optional-runtime source-build contract has an invalid schema.")
    distribution = _normalized_distribution(contract.get("distribution"))
    version = str(contract.get("version") or "")
    if (
        not distribution
        or contract.get("distribution") != distribution
        or distribution.strip("-") != distribution
        or "--" in distribution
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in distribution
        )
        or not version
        or len(version) > 128
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.!+_"
            for character in version
        )
        or contract.get("recipe") != SOURCE_BUILD_RECIPE
        or contract.get("pythonTag") != "py3"
        or contract.get("sourceDateEpoch") != FIXED_SOURCE_DATE_EPOCH
    ):
        raise RuntimeError("The optional-runtime source-build identity is invalid.")

    source = contract.get("sourceArtifact")
    source_fields = {
        "kind",
        "repository",
        "commit",
        "archiveRoot",
        "filename",
        "url",
        "sha256",
        "byteSize",
    }
    if not isinstance(source, dict) or set(source) != source_fields:
        raise RuntimeError("The optional-runtime source archive lock is malformed.")
    repository = str(source.get("repository") or "")
    repository_parts = repository.split("/")
    commit = str(source.get("commit") or "").lower()
    archive_root = str(source.get("archiveRoot") or "")
    filename = str(source.get("filename") or "")
    expected_url = f"https://codeload.github.com/{repository}/tar.gz/{commit}"
    expected_root = f"{repository_parts[-1]}-{commit}" if len(repository_parts) == 2 else ""
    if (
        source.get("kind") != "github_commit_tarball"
        or len(repository_parts) != 2
        or any(
            not part
            or part in {".", ".."}
            or len(part) > 100
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in part)
            for part in repository_parts
        )
        or len(commit) != 40
        or not set(commit).issubset(_SHA256_CHARACTERS)
        or source.get("commit") != commit
        or archive_root != expected_root
        or _safe_relative_path(archive_root).as_posix() != archive_root
        or filename != f"{expected_root}.tar.gz"
        or source.get("url") != expected_url
        or not _valid_sha256(source.get("sha256"))
        or not _bounded_positive_integer(
            source.get("byteSize"), maximum=MAX_LOCKED_ARCHIVE_BYTES
        )
    ):
        raise RuntimeError("The optional-runtime source archive lock is invalid.")

    raw_source_files = contract.get("sourceFiles")
    raw_source_trees = contract.get("sourceTrees")
    if (
        not isinstance(raw_source_files, list)
        or not all(isinstance(item, str) for item in raw_source_files)
        or not isinstance(raw_source_trees, list)
        or not all(isinstance(item, str) for item in raw_source_trees)
    ):
        raise RuntimeError("The optional-runtime source selection is malformed.")
    source_files = tuple(_safe_relative_path(item).as_posix() for item in raw_source_files)
    source_trees = tuple(_safe_relative_path(item).as_posix() for item in raw_source_trees)
    if (
        not source_files
        or not source_trees
        or len(source_files) > 64
        or len(source_trees) > 16
        or len(set(source_files)) != len(source_files)
        or len(set(source_trees)) != len(source_trees)
        or "setup.py" not in source_files
        or any(
            file == tree or file.startswith(f"{tree}/")
            for file in source_files
            for tree in source_trees
        )
        or any(
            first == second or first.startswith(f"{second}/") or second.startswith(f"{first}/")
            for index, first in enumerate(source_trees)
            for second in source_trees[index + 1 :]
        )
    ):
        raise RuntimeError("The optional-runtime source selection is invalid.")

    dependencies = contract.get("buildDependencies")
    if not isinstance(dependencies, list) or dependencies:
        raise RuntimeError("The no-code source assembler has a zero-dependency closure.")

    wheel_metadata = _validate_wheel_metadata(contract.get("wheelMetadata"))
    package_root = PurePosixPath(wheel_metadata["packageRoot"])
    if (
        wheel_metadata["licenseFile"] not in source_files
        or any(
            len(PurePosixPath(tree).parts) <= len(package_root.parts)
            or PurePosixPath(tree).parts[: len(package_root.parts)]
            != package_root.parts
            for tree in source_trees
        )
    ):
        raise RuntimeError("The source-build package selection is invalid.")

    output = contract.get("outputWheel")
    output_fields = {
        "distribution",
        "version",
        "filename",
        "sha256",
        "byteSize",
        "platform",
        "pythonTag",
        "machine",
    }
    if not isinstance(output, dict) or set(output) != output_fields:
        raise RuntimeError("The source-build output wheel lock is malformed.")
    if (
        output.get("distribution") != distribution
        or output.get("version") != version
        or Path(str(output.get("filename") or "")).name != output.get("filename")
        or not str(output.get("filename") or "").endswith("-py3-none-any.whl")
        or not _valid_sha256(output.get("sha256"))
        or not _bounded_positive_integer(
            output.get("byteSize"), maximum=MAX_BUILD_OUTPUT_BYTES
        )
        or output.get("platform") != "any"
        or output.get("pythonTag") != "py3"
        or output.get("machine") != "any"
        or output.get("filename")
        != f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"
    ):
        raise RuntimeError("The source-build output wheel lock is invalid.")

    normalized = dict(contract)
    normalized["distribution"] = distribution
    normalized["version"] = version
    normalized["sourceArtifact"] = dict(source)
    normalized["sourceFiles"] = list(source_files)
    normalized["sourceTrees"] = list(source_trees)
    normalized["buildDependencies"] = []
    normalized["wheelMetadata"] = wheel_metadata
    normalized["outputWheel"] = dict(output)
    return normalized


def source_build_output_artifact(contract: Any) -> dict[str, Any]:
    """Project the derived wheel lock used by the ordinary wheel validator."""

    return dict(validate_source_build_contract(contract)["outputWheel"])


def _source_artifact_path(cache_root: Path, source: dict[str, Any]) -> Path:
    root = cache_root.resolve(strict=True)
    candidate = (root / source["sha256"] / source["filename"]).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("The source archive cache path escapes its root.") from exc
    return candidate


def _stream_sha256(path: Path, *, maximum_bytes: int, lease: InstallLease) -> tuple[str, int]:
    details = path.lstat()
    if (
        not _safe_regular_details(details)
        or details.st_size > maximum_bytes
    ):
        raise RuntimeError("A source-build artifact is unsafe or oversized.")
    hasher = hashlib.sha256()
    observed = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            if lease.cancel_event.is_set():
                raise OverlayCancelled("Optional-runtime source acquisition was cancelled.")
            observed += len(chunk)
            if observed > maximum_bytes:
                raise RuntimeError("A source-build artifact exceeded its reviewed size.")
            hasher.update(chunk)
    if observed != details.st_size:
        raise RuntimeError("A source-build artifact changed while it was read.")
    return hasher.hexdigest(), observed


def cache_locked_source_archive(
    contract: Any,
    cache_root: Path,
    *,
    lease: InstallLease,
) -> Path:
    """Acquire one exact official commit archive with redirects/proxies disabled."""

    normalized = validate_source_build_contract(contract)
    source = normalized["sourceArtifact"]
    cache_root.mkdir(parents=True, exist_ok=True)
    root_details = cache_root.lstat()
    if not _safe_directory_details(root_details):
        raise RuntimeError("The source archive cache root is unsafe.")
    destination = _source_artifact_path(cache_root, source)
    destination.parent.mkdir(parents=False, exist_ok=True)
    parent_details = destination.parent.lstat()
    if not _safe_directory_details(parent_details):
        raise RuntimeError("The source archive cache directory is unsafe.")
    if destination.exists():
        digest, byte_size = _stream_sha256(
            destination,
            maximum_bytes=MAX_LOCKED_ARCHIVE_BYTES,
            lease=lease,
        )
        if digest != source["sha256"] or byte_size != source["byteSize"]:
            raise RuntimeError("A cached source archive failed its catalog identity.")
        return destination
    if lease.cancel_event.is_set():
        raise OverlayCancelled("Optional-runtime source acquisition was cancelled.")

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    opener = build_opener(ProxyHandler({}), HTTPSHandler(), _RejectRedirects())
    try:
        request = Request(
            source["url"],
            headers={"User-Agent": "MoDiff optional-runtime source acquisition"},
        )
        with opener.open(request, timeout=15) as response, temporary.open("xb") as output:
            if response.geturl() != source["url"]:
                raise RuntimeError("A source archive URL redirected outside its reviewed source.")
            length = response.headers.get("Content-Length")
            try:
                if length is not None and int(length) != source["byteSize"]:
                    raise RuntimeError("A source archive has an unexpected size.")
            except ValueError as exc:
                raise RuntimeError("A source archive has an invalid length header.") from exc
            hasher = hashlib.sha256()
            observed = 0
            while chunk := response.read(1024 * 1024):
                if lease.cancel_event.is_set():
                    raise OverlayCancelled("Optional-runtime source acquisition was cancelled.")
                observed += len(chunk)
                if observed > MAX_LOCKED_ARCHIVE_BYTES:
                    raise RuntimeError("A source archive exceeded its safe size.")
                output.write(chunk)
                hasher.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if hasher.hexdigest() != source["sha256"] or observed != source["byteSize"]:
            raise RuntimeError("A downloaded source archive failed its catalog identity.")
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


def _resolved_link_target(path: PurePosixPath, target: str) -> PurePosixPath:
    if (
        not target
        or "\\" in target
        or PurePosixPath(target).is_absolute()
        or any(ord(character) < 32 or ord(character) == 127 for character in target)
    ):
        raise RuntimeError("A source archive contains an unsafe link target.")
    stack = list(path.parent.parts)
    for part in PurePosixPath(target).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not stack:
                raise RuntimeError("A source archive link escapes its root.")
            stack.pop()
        else:
            stack.append(part)
    if not stack:
        raise RuntimeError("A source archive link has no contained target.")
    return PurePosixPath(*stack)


def _selected_source_path(relative: str, *, files: set[str], trees: tuple[str, ...]) -> bool:
    return relative in files or any(relative.startswith(f"{tree}/") for tree in trees)


def extract_locked_source_tree(
    contract: Any,
    archive: Path,
    destination: Path,
    *,
    lease: InstallLease,
) -> dict[str, Any]:
    """Inspect the complete tarball and extract only the reviewed build inputs."""

    normalized = validate_source_build_contract(contract)
    source = normalized["sourceArtifact"]
    digest, byte_size = _stream_sha256(
        archive,
        maximum_bytes=MAX_LOCKED_ARCHIVE_BYTES,
        lease=lease,
    )
    if digest != source["sha256"] or byte_size != source["byteSize"]:
        raise RuntimeError("The source archive failed its reviewed identity.")
    destination.mkdir(parents=False, exist_ok=False)
    if not _safe_directory_details(destination.lstat()):
        raise RuntimeError("The extracted source destination is unsafe.")
    destination_root = destination.resolve(strict=True)
    source_files = set(normalized["sourceFiles"])
    source_trees = tuple(normalized["sourceTrees"])
    selected_files: set[str] = set()
    populated_trees: set[str] = set()
    casefolded_paths: set[str] = set()
    source_seal: dict[str, str] = {}
    total_declared = 0
    extracted_bytes = 0
    entries = 0
    try:
        opened = tarfile.open(archive, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("The reviewed source archive is not a valid gzip tarball.") from exc
    with opened:
        for member in opened:
            if lease.cancel_event.is_set():
                raise OverlayCancelled("Optional-runtime source inspection was cancelled.")
            entries += 1
            if entries > MAX_SOURCE_ARCHIVE_ENTRIES:
                raise RuntimeError("The source archive contains too many members.")
            raw_name = member.name.rstrip("/")
            path = _safe_relative_path(raw_name)
            folded = unicodedata.normalize("NFC", path.as_posix()).casefold()
            if folded in casefolded_paths:
                raise RuntimeError("The source archive contains duplicate or colliding paths.")
            casefolded_paths.add(folded)
            if path.parts[0] != source["archiveRoot"]:
                raise RuntimeError("The source archive has an unexpected root directory.")
            relative_path = PurePosixPath(*path.parts[1:])
            relative = relative_path.as_posix() if relative_path.parts else ""
            if (
                member.size < 0
                or member.size > MAX_SOURCE_MEMBER_BYTES
                or total_declared + member.size > MAX_SOURCE_TOTAL_BYTES
            ):
                raise RuntimeError("The source archive exceeds its reviewed expansion bounds.")
            total_declared += member.size
            if member.isdir():
                if member.size != 0:
                    raise RuntimeError("A source archive directory declares file data.")
                continue
            if member.issym():
                if member.size != 0:
                    raise RuntimeError("A source archive link declares file data.")
                target = _resolved_link_target(path, member.linkname)
                if target.parts[0] != source["archiveRoot"]:
                    raise RuntimeError("A source archive link escapes its reviewed root.")
                if relative and _selected_source_path(
                    relative,
                    files=source_files,
                    trees=source_trees,
                ):
                    raise RuntimeError("A reviewed source input may not be a symbolic link.")
                continue
            if member.islnk() or member.isdev() or member.isfifo() or not member.isfile():
                raise RuntimeError("The source archive contains a non-regular member.")
            if not relative or not _selected_source_path(
                relative,
                files=source_files,
                trees=source_trees,
            ):
                continue
            output_path = destination_root.joinpath(*relative_path.parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not _safe_directory_details(output_path.parent.lstat()):
                raise RuntimeError("A source extraction directory is unsafe.")
            resolved_parent = output_path.parent.resolve(strict=True)
            resolved_parent.relative_to(destination_root)
            archive_file = opened.extractfile(member)
            if archive_file is None:
                raise RuntimeError("A reviewed source member could not be read.")
            hasher = hashlib.sha256()
            observed = 0
            with archive_file, output_path.open("xb") as output:
                while chunk := archive_file.read(1024 * 1024):
                    if lease.cancel_event.is_set():
                        raise OverlayCancelled("Optional-runtime source extraction was cancelled.")
                    observed += len(chunk)
                    if observed > member.size:
                        raise RuntimeError("A source member expanded beyond its declared size.")
                    output.write(chunk)
                    hasher.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            if not _safe_regular_details(output_path.lstat()):
                raise RuntimeError("An extracted source member is unsafe.")
            if observed != member.size:
                raise RuntimeError("A reviewed source member was truncated.")
            output_path.chmod(0o644)
            os.utime(
                output_path,
                (normalized["sourceDateEpoch"], normalized["sourceDateEpoch"]),
                follow_symlinks=False,
            )
            selected_files.add(relative)
            for tree in source_trees:
                if relative.startswith(f"{tree}/"):
                    populated_trees.add(tree)
            source_seal[relative] = hasher.hexdigest()
            extracted_bytes += observed
    if not source_files.issubset(selected_files) or populated_trees != set(source_trees):
        raise RuntimeError("The source archive does not contain the complete reviewed selection.")
    directories = sorted(
        (path for path in destination_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        if not _safe_directory_details(directory.lstat()):
            raise RuntimeError("The extracted source tree contains a link.")
        directory.chmod(0o755)
        os.utime(
            directory,
            (normalized["sourceDateEpoch"], normalized["sourceDateEpoch"]),
            follow_symlinks=False,
        )
    destination_root.chmod(0o755)
    os.utime(
        destination_root,
        (normalized["sourceDateEpoch"], normalized["sourceDateEpoch"]),
        follow_symlinks=False,
    )
    return {
        "schemaVersion": 1,
        "archiveSha256": source["sha256"],
        "archiveBytes": source["byteSize"],
        "memberCount": entries,
        "selectedFileCount": len(source_seal),
        "selectedBytes": extracted_bytes,
        "sourceSealDigest": _canonical_digest(dict(sorted(source_seal.items()))),
    }


def _wheel_info(name: str) -> zipfile.ZipInfo:
    """Return one completely normalized regular-file ZIP member."""

    _wheel_target_path(name)
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _record_row(name: str, digest: bytes, byte_size: int) -> tuple[str, str, str]:
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return name, f"sha256={encoded}", str(byte_size)


def _metadata_documents(contract: dict[str, Any], source_root: Path) -> dict[str, bytes]:
    distribution = contract["distribution"]
    version = contract["version"]
    metadata = contract["wheelMetadata"]
    dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    headers = [
        "Metadata-Version: 2.4",
        f"Name: {distribution}",
        f"Version: {version}",
        f"Summary: {metadata['summary']}",
        f"License: {metadata['license']}",
        f"Requires-Python: {metadata['requiresPython']}",
        f"License-File: {PurePosixPath(metadata['licenseFile']).name}",
    ]
    headers.extend(f"Requires-Dist: {item}" for item in metadata["requiresDist"])
    top_levels = sorted(
        {
            PurePosixPath(tree).parts[len(PurePosixPath(metadata["packageRoot"]).parts)]
            for tree in contract["sourceTrees"]
        }
    )
    documents = {
        f"{dist_info}/METADATA": ("\n".join(headers) + "\n\n").encode("utf-8"),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            f"Generator: {SOURCE_BUILD_RECIPE}\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n\n"
        ).encode("utf-8"),
        f"{dist_info}/top_level.txt": ("\n".join(top_levels) + "\n").encode("utf-8"),
    }
    if metadata["consoleScripts"]:
        script_lines = ["[console_scripts]"]
        script_lines.extend(
            f"{name} = {target}"
            for name, target in metadata["consoleScripts"].items()
        )
        documents[f"{dist_info}/entry_points.txt"] = (
            "\n".join(script_lines) + "\n"
        ).encode("utf-8")
    license_path = source_root.joinpath(
        *PurePosixPath(metadata["licenseFile"]).parts
    )
    details = license_path.lstat()
    if not _safe_regular_details(details):
        raise RuntimeError("The source-build license input is not a regular file.")
    if details.st_size > MAX_SOURCE_MEMBER_BYTES:
        raise RuntimeError("The source-build license input is oversized.")
    documents[f"{dist_info}/licenses/{license_path.name}"] = license_path.read_bytes()
    return documents


def _package_members(
    contract: dict[str, Any],
    source_root: Path,
) -> list[tuple[str, Path]]:
    """Map reviewed package trees to wheel members without importing them."""

    root = source_root.resolve(strict=True)
    if not _safe_directory_details(source_root.lstat()):
        raise RuntimeError("The source-build package root is unsafe.")
    package_root = PurePosixPath(contract["wheelMetadata"]["packageRoot"])
    members: list[tuple[str, Path]] = []
    seen: set[str] = set()
    casefolded: set[str] = set()
    for raw_tree in contract["sourceTrees"]:
        tree = PurePosixPath(raw_tree)
        relative_tree = PurePosixPath(*tree.parts[len(package_root.parts) :])
        tree_path = root.joinpath(*tree.parts)
        tree_details = tree_path.lstat()
        if not _safe_directory_details(tree_details):
            raise RuntimeError("A source-build package tree is not a regular directory.")
        tree_path.resolve(strict=True).relative_to(root)
        for directory, directory_names, filenames in os.walk(
            tree_path,
            topdown=True,
            followlinks=False,
        ):
            current = Path(directory)
            if not _safe_directory_details(current.lstat()):
                raise RuntimeError("A source-build package directory is unsafe.")
            current.resolve(strict=True).relative_to(root)
            safe_directories: list[str] = []
            for name in sorted(directory_names):
                child = current / name
                child_details = child.lstat()
                if not _safe_directory_details(child_details):
                    raise RuntimeError("A source-build package tree contains a link.")
                safe_directories.append(name)
            directory_names[:] = safe_directories
            for name in sorted(filenames):
                path = current / name
                details = path.lstat()
                if (
                    not _safe_regular_details(details)
                    or details.st_size > MAX_SOURCE_MEMBER_BYTES
                ):
                    raise RuntimeError("A source-build package input is unsafe or oversized.")
                path.resolve(strict=True).relative_to(root)
                relative = path.relative_to(tree_path)
                member = (relative_tree / PurePosixPath(relative.as_posix())).as_posix()
                member = _wheel_target_path(member)
                folded = unicodedata.normalize("NFC", member).casefold()
                if member in seen or folded in casefolded:
                    raise RuntimeError("Source-build package paths collide in the wheel.")
                seen.add(member)
                casefolded.add(folded)
                members.append((member, path))
                if len(members) > MAX_WHEEL_MEMBERS:
                    raise RuntimeError("The source-build wheel contains too many members.")
    if not members:
        raise RuntimeError("The source-build package selection is empty.")
    return sorted(members, key=lambda item: item[0])


def assemble_locked_pure_python_wheel(
    contract: Any,
    source_root: Path,
    output_path: Path,
    *,
    lease: InstallLease,
) -> dict[str, Any]:
    """Assemble a byte-stable wheel without executing any upstream code."""

    normalized = validate_source_build_contract(contract)
    source_root = source_root.resolve(strict=True)
    members = _package_members(normalized, source_root)
    documents = _metadata_documents(normalized, source_root)
    existing = {
        unicodedata.normalize("NFC", name).casefold() for name, _path in members
    }
    if any(
        unicodedata.normalize("NFC", name).casefold() in existing
        for name in documents
    ):
        raise RuntimeError("Source and generated wheel members collide.")
    if len(members) + len(documents) + 1 > MAX_WHEEL_MEMBERS:
        raise RuntimeError("The source-build wheel contains too many members.")

    record_rows: list[tuple[str, str, str]] = []
    total_bytes = 0
    output_parent = output_path.parent.resolve(strict=True)
    if not _safe_directory_details(output_parent.lstat()):
        raise RuntimeError("The source-build output directory is unsafe.")
    with output_path.open("xb") as raw_output:
        with zipfile.ZipFile(
            raw_output,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=False,
            strict_timestamps=True,
        ) as wheel:
            ordered_members = sorted(
                [
                    (member_name, path, None)
                    for member_name, path in members
                ]
                + [
                    (member_name, None, body)
                    for member_name, body in documents.items()
                ],
                key=lambda item: item[0],
            )
            for member_name, path, generated_body in ordered_members:
                if lease.cancel_event.is_set():
                    raise OverlayCancelled("Optional-runtime source assembly was cancelled.")
                if generated_body is not None:
                    total_bytes += len(generated_body)
                    if total_bytes > MAX_BUILD_OUTPUT_BYTES:
                        raise RuntimeError("The source-build wheel exceeds its safe size.")
                    wheel.writestr(_wheel_info(member_name), generated_body)
                    record_rows.append(
                        _record_row(
                            member_name,
                            hashlib.sha256(generated_body).digest(),
                            len(generated_body),
                        )
                    )
                    continue
                if path is None:
                    raise RuntimeError("The source-build member plan is incomplete.")
                hasher = hashlib.sha256()
                observed = 0
                with path.open("rb") as source, wheel.open(
                    _wheel_info(member_name), "w", force_zip64=False
                ) as output:
                    while chunk := source.read(1024 * 1024):
                        if lease.cancel_event.is_set():
                            raise OverlayCancelled(
                                "Optional-runtime source assembly was cancelled."
                            )
                        observed += len(chunk)
                        total_bytes += len(chunk)
                        if (
                            observed > MAX_SOURCE_MEMBER_BYTES
                            or total_bytes > MAX_BUILD_OUTPUT_BYTES
                        ):
                            raise RuntimeError("The source-build wheel exceeds its safe size.")
                        output.write(chunk)
                        hasher.update(chunk)
                record_rows.append(
                    _record_row(member_name, hasher.digest(), observed)
                )
            dist_info = (
                f"{normalized['distribution'].replace('-', '_')}-"
                f"{normalized['version']}.dist-info"
            )
            record_name = f"{dist_info}/RECORD"
            record_rows.append((record_name, "", ""))
            record_stream = io.StringIO(newline="")
            csv.writer(record_stream, lineterminator="\n").writerows(record_rows)
            wheel.writestr(_wheel_info(record_name), record_stream.getvalue().encode("utf-8"))
        raw_output.flush()
        os.fsync(raw_output.fileno())
    return {
        "schemaVersion": 1,
        "recipe": normalized["recipe"],
        "memberCount": len(record_rows),
        "sourceMemberCount": len(members),
        "uncompressedWheelBytes": total_bytes,
    }


def _store_built_wheel(
    source: Path,
    artifact: dict[str, Any],
    cache_root: Path,
    *,
    lease: InstallLease,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    if not _safe_directory_details(cache_root.lstat()):
        raise RuntimeError("The source-build wheel cache root is unsafe.")
    destination = locked_artifact_path(cache_root, artifact)
    destination.parent.mkdir(parents=False, exist_ok=True)
    if not _safe_directory_details(destination.parent.lstat()):
        raise RuntimeError("The source-build wheel cache directory is unsafe.")
    if destination.exists():
        digest, byte_size = _stream_sha256(
            destination,
            maximum_bytes=MAX_BUILD_OUTPUT_BYTES,
            lease=lease,
        )
        if digest != artifact["sha256"] or byte_size != artifact["byteSize"]:
            raise RuntimeError("A cached source-built wheel failed its reviewed identity.")
        return destination
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        hasher = hashlib.sha256()
        observed = 0
        with source.open("rb") as input_file, temporary.open("xb") as output_file:
            while chunk := input_file.read(1024 * 1024):
                if lease.cancel_event.is_set():
                    raise OverlayCancelled("Optional-runtime source build was cancelled.")
                observed += len(chunk)
                if observed > MAX_BUILD_OUTPUT_BYTES:
                    raise RuntimeError("A source-built wheel exceeds its reviewed size.")
                output_file.write(chunk)
                hasher.update(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
        if hasher.hexdigest() != artifact["sha256"] or observed != artifact["byteSize"]:
            raise RuntimeError("The source-built wheel differs from its reviewed output lock.")
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return destination


def build_locked_source_wheel(
    contract: Any,
    *,
    cache_root: Path,
    work_root: Path,
    lease: InstallLease,
) -> SourceBuildResult:
    """Assemble and authenticate one exact wheel without running source code."""

    normalized = validate_source_build_contract(contract)
    work_root.mkdir(parents=False, exist_ok=False)
    work_details = work_root.lstat()
    if not _safe_directory_details(work_details):
        raise RuntimeError("The source-build workspace is unsafe.")
    source_archive = cache_locked_source_archive(normalized, cache_root, lease=lease)
    source_root = work_root / "source"
    source_receipt = extract_locked_source_tree(
        normalized,
        source_archive,
        source_root,
        lease=lease,
    )
    output_root = work_root / "output"
    output_root.mkdir()
    if not _safe_directory_details(output_root.lstat()):
        raise RuntimeError("The source-build output workspace is unsafe.")
    output_artifact = dict(normalized["outputWheel"])
    built_wheel = output_root / output_artifact["filename"]
    assembly_receipt = assemble_locked_pure_python_wheel(
        normalized,
        source_root,
        built_wheel,
        lease=lease,
    )
    built_details = built_wheel.lstat()
    if not _safe_regular_details(built_details):
        raise RuntimeError("The source build output is not a safe regular wheel.")
    digest, output_bytes = _stream_sha256(
        built_wheel,
        maximum_bytes=MAX_BUILD_OUTPUT_BYTES,
        lease=lease,
    )
    if digest != output_artifact["sha256"] or output_bytes != output_artifact["byteSize"]:
        raise RuntimeError("The source-built wheel differs from its reviewed output lock.")
    cached_output = _store_built_wheel(
        built_wheel,
        output_artifact,
        cache_root,
        lease=lease,
    )
    output_seal = locked_artifact_file_seal(
        [output_artifact],
        cache_root,
        lease=lease,
    )
    receipt = {
        "schemaVersion": 1,
        "contractDigest": _canonical_digest(normalized),
        "source": source_receipt,
        "assembly": assembly_receipt,
        "buildDependencyCount": 0,
        "buildRecipe": SOURCE_BUILD_RECIPE,
        "pythonTag": normalized["pythonTag"],
        "outputSha256": output_artifact["sha256"],
        "outputBytes": output_artifact["byteSize"],
        "outputFileSealDigest": _canonical_digest(output_seal),
    }
    return SourceBuildResult(
        artifact=output_artifact,
        path=cached_output,
        receipt=receipt,
    )
