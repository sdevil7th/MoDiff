"""Security boundary for app-managed Python package overlays.

The functions in this module are deliberately standard-library only.  They
stage packages without mutating the running interpreter, serialize every
installer through one process- and OS-backed lease, and validate a completed overlay in
an isolated child interpreter before it can be promoted or activated.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import configparser
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import importlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
from pathlib import PurePosixPath
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
from typing import Any, Iterable
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener
import zipfile


PYPI_SIMPLE_INDEX = "https://pypi.org/simple"
PINNED_DIFFUSERS_SOURCE_URL = "https://github.com/huggingface/diffusers.git"
PINNED_DIFFUSERS_COMMIT = "13a7bee4878d62fccc8d25f97e480e68de96fa03"
PINNED_DIFFUSERS_VERSION = "0.40.0.dev0"
_DIGEST_PREFIX = "sha256:"
MANAGED_ROOT = Path(
    os.environ.get("MODIFF_MANAGED_ROOT") or Path(__file__).resolve().parents[1] / ".modiff"
)
INSTALL_LEASE_PATH = MANAGED_ROOT / "optimizations" / "install.lock"
_INSTALL_LOCK = threading.Lock()
_ACTIVE_INSTALL: "InstallLease | None" = None
MAX_LOCKED_ARCHIVE_BYTES = 512 * 1024**2
MAX_LOCKED_ARCHIVE_TOTAL_BYTES = 2 * 1024**3
MAX_LOCKED_WHEEL_ENTRIES = 100_000
MAX_LOCKED_WHEEL_MEMBER_BYTES = 512 * 1024**2
MAX_LOCKED_WHEEL_TOTAL_BYTES = 2 * 1024**3


class OverlayCancelled(RuntimeError):
    """Raised after the caller cancels an in-progress staging operation."""


class OverlayInstallBusy(RuntimeError):
    """Raised when another optional-runtime install owns the global lease."""


class OverlayStorageUnsafe(RuntimeError):
    """Raised when the managed lease/staging root fails containment checks."""


def ensure_managed_directory(path: Path, *, managed_root: Path = MANAGED_ROOT) -> Path:
    """Create/validate a directory chain without accepting links or reparse points."""

    root = Path(managed_root)
    root.mkdir(parents=True, exist_ok=True)
    root_info = root.lstat()
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or stat.S_ISLNK(root_info.st_mode)
        or bool(
            getattr(root_info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    ):
        raise RuntimeError("The managed runtime root is a link or reparse point.")
    trusted_root = root.resolve(strict=True)
    raw_path = Path(path)
    try:
        relative = raw_path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise RuntimeError("The managed runtime directory escapes its root.") from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        info = current.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or bool(
                getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ):
            raise RuntimeError("A managed runtime directory is a link or reparse point.")
        current.resolve(strict=True).relative_to(trusted_root)
    return current.resolve(strict=True)


@dataclass
class InstallLease:
    """One authoritative process-local install and its running subprocess."""

    token: str
    owner_kind: str
    owner_id: str
    cancel_event: threading.Event
    process: subprocess.Popen | None = None
    lock_file: Any = None
    committed: bool = False


def _acquire_os_lock() -> Any:
    lock_file = None
    try:
        lock_parent = ensure_managed_directory(
            INSTALL_LEASE_PATH.parent,
            managed_root=MANAGED_ROOT,
        )
        lock_path = lock_parent / INSTALL_LEASE_PATH.name
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        lock_file = os.fdopen(descriptor, "r+b", closefd=True)
        path_info = lock_path.lstat()
        file_info = os.fstat(lock_file.fileno())
        if (
            not stat.S_ISREG(path_info.st_mode)
            or stat.S_ISLNK(path_info.st_mode)
            or bool(
                getattr(path_info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            or getattr(path_info, "st_nlink", 1) != 1
            or (path_info.st_dev, path_info.st_ino) != (file_info.st_dev, file_info.st_ino)
        ):
            raise OSError("The optional-runtime lease file is unsafe.")
        lock_path.resolve(strict=True).relative_to(lock_parent)
    except (OSError, RuntimeError, ValueError) as exc:
        if lock_file is not None:
            lock_file.close()
        raise OverlayStorageUnsafe("The optional-runtime lease path is unsafe.") from exc
    try:
        if os.name == "nt":
            import msvcrt

            if lock_file.seek(0, os.SEEK_END) == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError) as exc:
        lock_file.close()
        raise OverlayInstallBusy("Another optional-runtime installation is already active.") from exc
    return lock_file


def _release_os_lock(lock_file: Any) -> None:
    if lock_file is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except (OSError, IOError):
        pass
    finally:
        lock_file.close()


def reserve_install(owner_kind: str, owner_id: str) -> InstallLease:
    """Reserve the one install slot before an HTTP handler returns ``202``."""

    global _ACTIVE_INSTALL
    kind = str(owner_kind or "").strip()
    identifier = str(owner_id or "").strip()
    if not kind or not identifier or len(kind) > 64 or len(identifier) > 256:
        raise ValueError("A bounded install owner kind and identifier are required.")
    with _INSTALL_LOCK:
        if _ACTIVE_INSTALL is not None:
            raise OverlayInstallBusy("Another optional-runtime installation is already active.")
        lock_file = _acquire_os_lock()
        lease = InstallLease(
            token=f"overlay-install-{uuid.uuid4().hex}",
            owner_kind=kind,
            owner_id=identifier,
            cancel_event=threading.Event(),
            lock_file=lock_file,
        )
        _ACTIVE_INSTALL = lease
        return lease


def active_install() -> dict[str, str] | None:
    with _INSTALL_LOCK:
        if _ACTIVE_INSTALL is None:
            return None
        return {
            "token": _ACTIVE_INSTALL.token,
            "ownerKind": _ACTIVE_INSTALL.owner_kind,
            "ownerId": _ACTIVE_INSTALL.owner_id,
        }


def cancel_install(token: str) -> bool:
    """Signal cancellation and terminate the exact child process, if running."""

    with _INSTALL_LOCK:
        lease = _ACTIVE_INSTALL
        if lease is None or lease.token != token or lease.committed:
            return False
        lease.cancel_event.set()
        process = lease.process
    if process is not None:
        _terminate_process(process)
    return True


def release_install(lease: InstallLease) -> None:
    global _ACTIVE_INSTALL
    with _INSTALL_LOCK:
        if _ACTIVE_INSTALL is lease:
            _ACTIVE_INSTALL = None
        lease.process = None
        lock_file = lease.lock_file
        lease.lock_file = None
    _release_os_lock(lock_file)


def promote_staged_environment(lease: InstallLease, staged: Path, destination: Path) -> None:
    """Atomically commit only while the exact uncancelled lease is authoritative."""

    with _INSTALL_LOCK:
        if _ACTIVE_INSTALL is not lease or lease.cancel_event.is_set() or lease.committed:
            raise OverlayCancelled("Optional-runtime installation was cancelled before promotion.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(destination)
        lease.committed = True


def _set_lease_process(lease: InstallLease, process: subprocess.Popen | None) -> None:
    with _INSTALL_LOCK:
        if _ACTIVE_INSTALL is not lease:
            if process is not None:
                _terminate_process(process)
            raise OverlayCancelled("The optional-runtime install lease is no longer active.")
        lease.process = process


def _terminate_process(process: subprocess.Popen) -> None:
    if os.name == "nt" and process.poll() is not None:
        return
    process_group = process.pid
    try:
        if os.name != "nt":
            os.killpg(process_group, signal.SIGTERM)
        else:
            system_root = Path(os.environ.get("SystemRoot") or r"C:\Windows")
            taskkill = (system_root / "System32" / "taskkill.exe").resolve()
            subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=5,
            )
        process.wait(timeout=2)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name != "nt":
            # The leader can exit while a child ignores SIGTERM. Always issue
            # the terminal signal to the original process group as well.
            os.killpg(process_group, signal.SIGKILL)
        else:
            if process.poll() is None:
                process.kill()
        if process.poll() is None:
            process.wait(timeout=2)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        pass


def sanitized_install_environment(site_packages: Path) -> dict[str, str]:
    """Return an install environment that cannot redirect the reviewed index."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("PIP_", "UV_"))
        and key.upper() not in {"PYTHONPATH", "PYTHONHOME"}
    }
    environment.update(
        {
            "PIP_NO_INPUT": "1",
            "UV_NO_PROGRESS": "1",
            "UV_NO_CONFIG": "1",
            "MODIFF_STAGED_SITE_PACKAGES": str(site_packages),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _stream_sha256(
    path: Path,
    *,
    maximum_bytes: int,
    cancel_event: threading.Event | None = None,
) -> tuple[str, int]:
    details = path.lstat()
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or details.st_nlink != 1
        or bool(
            getattr(details, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
        or details.st_size > maximum_bytes
    ):
        raise RuntimeError("A locked optional-runtime artifact is unsafe or oversized.")
    hasher = hashlib.sha256()
    observed = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            if cancel_event is not None and cancel_event.is_set():
                raise OverlayCancelled("Optional-runtime artifact verification was cancelled.")
            observed += len(chunk)
            if observed > maximum_bytes:
                raise RuntimeError("A locked optional-runtime artifact exceeds its safe limit.")
            hasher.update(chunk)
    if observed != details.st_size:
        raise RuntimeError("A locked optional-runtime artifact changed while it was read.")
    return hasher.hexdigest(), observed


def locked_artifact_path(archive_root: Path, artifact: dict[str, Any]) -> Path:
    """Derive a contained cache path from a reviewed artifact lock."""

    digest = str(artifact.get("sha256") or "").lower()
    filename = str(artifact.get("filename") or "")
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not filename
        or len(filename) > 256
        or Path(filename).name != filename
        or not filename.endswith(".whl")
    ):
        raise RuntimeError("The optional-runtime artifact lock is malformed.")
    root = archive_root.resolve(strict=True)
    candidate = (root / digest / filename).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("The optional-runtime artifact cache path escapes its root.") from exc
    return candidate


def cache_locked_artifacts(
    artifacts: Iterable[dict[str, Any]],
    archive_root: Path,
    *,
    lease: InstallLease,
) -> list[Path]:
    """Acquire exact catalog URLs into a hash-addressed cache under the lease."""

    archive_root.mkdir(parents=True, exist_ok=True)
    root_info = archive_root.lstat()
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or stat.S_ISLNK(root_info.st_mode)
        or bool(
            getattr(root_info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    ):
        raise RuntimeError("The optional-runtime artifact cache root is unsafe.")
    opener = build_opener(ProxyHandler({}), HTTPSHandler())
    cached: list[Path] = []
    total = 0
    for artifact in artifacts:
        if lease.cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime artifact acquisition was cancelled.")
        destination = locked_artifact_path(archive_root, artifact)
        destination.parent.mkdir(parents=False, exist_ok=True)
        parent_info = destination.parent.lstat()
        if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
            raise RuntimeError("The optional-runtime artifact cache directory is unsafe.")
        if destination.exists():
            observed_digest, observed_size = _stream_sha256(
                destination,
                maximum_bytes=MAX_LOCKED_ARCHIVE_BYTES,
                cancel_event=lease.cancel_event,
            )
            if observed_digest != artifact["sha256"]:
                raise RuntimeError("A cached optional-runtime artifact failed its catalog digest.")
            total += observed_size
            if total > MAX_LOCKED_ARCHIVE_TOTAL_BYTES:
                raise RuntimeError("The locked artifact set exceeds its reviewed size bound.")
            cached.append(destination)
            continue
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            request = Request(
                str(artifact["url"]),
                headers={"User-Agent": "MoDiff optional-runtime artifact acquisition"},
            )
            with opener.open(request, timeout=15) as response, temporary.open("xb") as output:
                if response.geturl() != artifact["url"]:
                    raise RuntimeError("A locked artifact URL redirected outside its reviewed source.")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > MAX_LOCKED_ARCHIVE_BYTES:
                    raise RuntimeError("A locked optional-runtime artifact is oversized.")
                hasher = hashlib.sha256()
                observed_size = 0
                while chunk := response.read(1024 * 1024):
                    if lease.cancel_event.is_set():
                        raise OverlayCancelled("Optional-runtime artifact acquisition was cancelled.")
                    observed_size += len(chunk)
                    if observed_size > MAX_LOCKED_ARCHIVE_BYTES:
                        raise RuntimeError("A locked optional-runtime artifact is oversized.")
                    output.write(chunk)
                    hasher.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            if hasher.hexdigest() != artifact["sha256"]:
                raise RuntimeError("A downloaded optional-runtime artifact failed its catalog digest.")
            temporary.replace(destination)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        total += observed_size
        if total > MAX_LOCKED_ARCHIVE_TOTAL_BYTES:
            raise RuntimeError("The locked artifact set exceeds its reviewed size bound.")
        cached.append(destination)
    return cached


def _wheel_target_path(member_name: str) -> str:
    if not member_name or len(member_name) > 1024 or "\\" in member_name or "\0" in member_name:
        raise RuntimeError("A locked wheel contains an invalid member path.")
    path = PurePosixPath(member_name)
    if (
        path.is_absolute()
        or len(path.parts) > 64
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError("A locked wheel contains a traversal path.")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
    }
    for part in path.parts:
        if (
            ":" in part
            or any(character in '<>"|?*' for character in part)
            or part.rstrip(" .") != part
            or part.split(".", 1)[0].upper() in reserved
        ):
            raise RuntimeError("A locked wheel contains a Windows-unsafe member path.")
    parts = list(path.parts)
    if parts[0].endswith(".data"):
        if len(parts) < 3 or parts[1] not in {"purelib", "platlib", "scripts"}:
            raise RuntimeError("A locked wheel uses an unsupported data installation scheme.")
        parts = (["bin"] if parts[1] == "scripts" else []) + parts[2:]
    if not parts:
        raise RuntimeError("A locked wheel member has no install target.")
    target = PurePosixPath(*parts).as_posix()
    if PurePosixPath(target).suffix.lower() == ".pth" or PurePosixPath(target).name.lower() in {
        "sitecustomize.py",
        "usercustomize.py",
    }:
        raise RuntimeError("A locked wheel contains a forbidden startup hook.")
    return target


def locked_artifact_file_seal(
    artifacts: Iterable[dict[str, Any]],
    archive_root: Path,
    *,
    lease: InstallLease | None = None,
) -> dict[str, str]:
    """Authenticate wheel archives and derive their exact extracted file seal."""

    root = archive_root.resolve(strict=True)
    expected_files: dict[str, str] = {}
    casefolded: set[str] = set()
    total_archive_bytes = 0
    total_member_bytes = 0
    entries = 0
    seen_distributions: set[str] = set()
    for artifact in artifacts:
        if lease is not None and lease.cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime artifact verification was cancelled.")
        archive = locked_artifact_path(root, artifact)
        archive.resolve(strict=True).relative_to(root)
        observed_digest, observed_size = _stream_sha256(
            archive,
            maximum_bytes=MAX_LOCKED_ARCHIVE_BYTES,
            cancel_event=lease.cancel_event if lease is not None else None,
        )
        total_archive_bytes += observed_size
        if (
            total_archive_bytes > MAX_LOCKED_ARCHIVE_TOTAL_BYTES
            or observed_digest != artifact.get("sha256")
        ):
            raise RuntimeError("A locked wheel archive failed its catalog identity.")
        filename_parts = str(artifact.get("filename") or "").removesuffix(".whl").split("-")
        distribution = _normalized_distribution(artifact.get("distribution"))
        version = str(artifact.get("version") or "")
        if (
            len(filename_parts) < 5
            or _normalized_distribution(filename_parts[0]) != distribution
            or filename_parts[1] != version
            or distribution in seen_distributions
        ):
            raise RuntimeError("A locked wheel filename does not match its reviewed project/version.")
        seen_distributions.add(distribution)
        metadata_documents: list[bytes] = []
        wheel_documents: list[bytes] = []
        record_targets: list[str] = []
        metadata_parents: set[str] = set()
        try:
            wheel = zipfile.ZipFile(archive)
        except (OSError, zipfile.BadZipFile) as exc:
            raise RuntimeError("A locked optional-runtime artifact is not a valid wheel ZIP.") from exc
        with wheel:
            seen_archive_members: set[str] = set()
            for info in wheel.infolist():
                if lease is not None and lease.cancel_event.is_set():
                    raise OverlayCancelled("Optional-runtime artifact verification was cancelled.")
                entries += 1
                if entries > MAX_LOCKED_WHEEL_ENTRIES:
                    raise RuntimeError("The locked wheel set contains too many members.")
                target = _wheel_target_path(info.filename.rstrip("/"))
                folded = unicodedata.normalize("NFC", target).casefold()
                if target in seen_archive_members or folded in casefolded:
                    raise RuntimeError("The locked wheel set contains duplicate install targets.")
                seen_archive_members.add(target)
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode) or (mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise RuntimeError("A locked wheel contains a non-regular member.")
                if info.is_dir():
                    continue
                casefolded.add(folded)
                if info.file_size > MAX_LOCKED_WHEEL_MEMBER_BYTES:
                    raise RuntimeError("A locked wheel member exceeds its safe size.")
                total_member_bytes += info.file_size
                if total_member_bytes > MAX_LOCKED_WHEEL_TOTAL_BYTES:
                    raise RuntimeError("The locked wheel set exceeds its extracted size bound.")
                hasher = hashlib.sha256()
                observed_member_size = 0
                bounded_document = (
                    bytearray()
                    if target.endswith(
                        (".dist-info/METADATA", ".dist-info/WHEEL", ".dist-info/RECORD")
                    )
                    else None
                )
                with wheel.open(info, "r") as source:
                    while chunk := source.read(1024 * 1024):
                        if lease is not None and lease.cancel_event.is_set():
                            raise OverlayCancelled(
                                "Optional-runtime artifact verification was cancelled."
                            )
                        observed_member_size += len(chunk)
                        if observed_member_size > info.file_size:
                            raise RuntimeError("A locked wheel member expanded beyond its declared size.")
                        hasher.update(chunk)
                        if bounded_document is not None:
                            document_limit = (
                                4 * 1024 * 1024
                                if target.endswith(".dist-info/RECORD")
                                else 2 * 1024 * 1024
                            )
                            if len(bounded_document) + len(chunk) > document_limit:
                                raise RuntimeError("A locked wheel metadata document is oversized.")
                            bounded_document.extend(chunk)
                if observed_member_size != info.file_size:
                    raise RuntimeError("A locked wheel member was truncated.")
                expected_files[target] = hasher.hexdigest()
                if target.endswith(".dist-info/METADATA"):
                    metadata_documents.append(bytes(bounded_document or b""))
                    metadata_parents.add(PurePosixPath(target).parent.as_posix())
                elif target.endswith(".dist-info/WHEEL"):
                    wheel_documents.append(bytes(bounded_document or b""))
                    metadata_parents.add(PurePosixPath(target).parent.as_posix())
                elif target.endswith(".dist-info/RECORD"):
                    record_targets.append(target)
                    metadata_parents.add(PurePosixPath(target).parent.as_posix())
        if (
            len(metadata_documents) != 1
            or len(wheel_documents) != 1
            or len(record_targets) != 1
            or len(metadata_parents) != 1
        ):
            raise RuntimeError("A locked wheel must contain one matching METADATA, WHEEL, and RECORD.")
        dist_info_name = next(iter(metadata_parents)).removesuffix(".dist-info")
        if "-" not in dist_info_name:
            raise RuntimeError("A locked wheel has an invalid dist-info directory name.")
        dist_info_project, dist_info_version = dist_info_name.rsplit("-", 1)
        if (
            _normalized_distribution(dist_info_project) != distribution
            or dist_info_version != version
        ):
            raise RuntimeError("A locked wheel dist-info path does not match its catalog lock.")
        parsed_metadata = BytesParser(policy=email_policy).parsebytes(metadata_documents[0])
        if (
            _normalized_distribution(parsed_metadata.get("Name")) != distribution
            or str(parsed_metadata.get("Version") or "") != version
        ):
            raise RuntimeError("A locked wheel METADATA identity does not match its catalog lock.")
        parsed_wheel = BytesParser(policy=email_policy).parsebytes(wheel_documents[0])
        expected_wheel_tag = "-".join(filename_parts[-3:])
        wheel_tags = [str(value) for value in (parsed_wheel.get_all("Tag") or [])]
        if (
            not parsed_wheel.get("Wheel-Version")
            or str(parsed_wheel.get("Root-Is-Purelib") or "").lower()
            not in {"true", "false"}
            or expected_wheel_tag not in wheel_tags
        ):
            raise RuntimeError("A locked wheel has an invalid WHEEL identity document.")
    return dict(sorted(expected_files.items()))


def normalize_locked_wheel_install(
    site_packages: Path,
    artifacts: Iterable[dict[str, Any]],
    archive_root: Path,
    *,
    lease: InstallLease | None = None,
) -> None:
    """Remove three known installer receipts and restore authenticated RECORDs."""

    artifact_list = [dict(item) for item in artifacts]
    expected = locked_artifact_file_seal(artifact_list, archive_root, lease=lease)
    root = site_packages.resolve(strict=True)
    metadata_roots = {
        PurePosixPath(path).parent.as_posix()
        for path in expected
        if path.endswith(".dist-info/METADATA")
    }
    for metadata_root in metadata_roots:
        if lease is not None and lease.cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime normalization was cancelled.")
        for filename in ("INSTALLER", "direct_url.json", "REQUESTED"):
            relative = f"{metadata_root}/{filename}"
            if relative in expected:
                continue
            candidate = root.joinpath(*PurePosixPath(relative).parts)
            try:
                details = candidate.lstat()
            except FileNotFoundError:
                continue
            if (
                not stat.S_ISREG(details.st_mode)
                or stat.S_ISLNK(details.st_mode)
                or bool(
                    getattr(details, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                )
            ):
                raise RuntimeError("An installer-generated receipt is not a safe regular file.")
            candidate.resolve(strict=True).relative_to(root)
            candidate.unlink()
    generated_scripts: set[str] = set()
    for artifact in artifact_list:
        if lease is not None and lease.cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime normalization was cancelled.")
        archive = locked_artifact_path(archive_root, artifact)
        with zipfile.ZipFile(archive) as wheel:
            for info in wheel.infolist():
                if lease is not None and lease.cancel_event.is_set():
                    raise OverlayCancelled("Optional-runtime normalization was cancelled.")
                if info.is_dir():
                    continue
                target = _wheel_target_path(info.filename)
                if target.endswith(".dist-info/entry_points.txt"):
                    if info.file_size > 2 * 1024 * 1024:
                        raise RuntimeError("A locked wheel entry-point document is oversized.")
                    parser = configparser.ConfigParser(interpolation=None, strict=True)
                    parser.optionxform = str
                    try:
                        parser.read_string(wheel.read(info).decode("utf-8"))
                    except (UnicodeDecodeError, configparser.Error) as exc:
                        raise RuntimeError("A locked wheel has invalid entry-point metadata.") from exc
                    for section in ("console_scripts", "gui_scripts"):
                        if not parser.has_section(section):
                            continue
                        for script_name in parser.options(section):
                            if (
                                not script_name
                                or len(script_name) > 128
                                or any(
                                    character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                                    for character in script_name
                                )
                            ):
                                raise RuntimeError("A locked wheel has an unsafe entry-point name.")
                            generated_scripts.add(script_name)
                if not target.endswith(".dist-info/RECORD"):
                    continue
                if info.file_size > 4 * 1024 * 1024:
                    raise RuntimeError("A locked wheel RECORD is oversized.")
                body = wheel.read(info)
                if hashlib.sha256(body).hexdigest() != expected[target]:
                    raise RuntimeError("A locked wheel RECORD failed its archive identity.")
                destination = root.joinpath(*PurePosixPath(target).parts)
                destination.parent.resolve(strict=True).relative_to(root)
                try:
                    details = destination.lstat()
                    if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
                        raise RuntimeError("An installed wheel RECORD is not a regular file.")
                except FileNotFoundError:
                    pass
                temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("xb") as output:
                        output.write(body)
                        output.flush()
                        os.fsync(output.fileno())
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
    for script_name in generated_scripts:
        if lease is not None and lease.cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime normalization was cancelled.")
        for relative in tuple(
            f"{directory}/{filename}"
            for directory in ("bin", "Scripts")
            for filename in (
                script_name,
                f"{script_name}.exe",
                f"{script_name}-script.py",
                f"{script_name}.exe.manifest",
            )
        ):
            if relative in expected:
                continue
            candidate = root.joinpath(*PurePosixPath(relative).parts)
            try:
                details = candidate.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
                raise RuntimeError("An installer-generated console script is not a regular file.")
            candidate.resolve(strict=True).relative_to(root)
            candidate.unlink()
    for directory in (root / "bin", root / "Scripts"):
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
        except FileNotFoundError:
            pass


def _overlay_directory_inventory(
    site_packages: Path, *, cancel_event: threading.Event | None = None
) -> set[str]:
    root = site_packages.resolve(strict=True)
    directories: set[str] = set()
    pending = [root]
    entries = 1
    while pending:
        if cancel_event is not None and cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime overlay verification was cancelled.")
        parent = pending.pop()
        with os.scandir(parent) as children:
            for child in children:
                entries += 1
                if entries > 100_000:
                    raise RuntimeError("The overlay directory inventory exceeds its safe bound.")
                path = Path(child.path)
                details = path.lstat()
                if stat.S_ISDIR(details.st_mode):
                    resolved = path.resolve(strict=True)
                    resolved.relative_to(root)
                    directories.add(resolved.relative_to(root).as_posix())
                    pending.append(resolved)
    return directories


def _expected_parent_directories(relative_files: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for filename in relative_files:
        parent = PurePosixPath(filename).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def verify_artifact_anchored_overlay(
    site_packages: Path,
    artifacts: Iterable[dict[str, Any]],
    archive_root: Path,
    *,
    lease: InstallLease | None = None,
) -> dict[str, Any]:
    """Match every extracted byte to rehashed catalog-locked wheel archives."""

    artifact_list = [dict(item) for item in artifacts]
    if not artifact_list:
        raise RuntimeError("An artifact-anchored overlay requires a complete wheel lock.")
    expected = locked_artifact_file_seal(artifact_list, archive_root, lease=lease)
    root = site_packages.resolve(strict=True)
    cancel_event = lease.cancel_event if lease is not None else None
    actual_paths = scan_overlay_tree(site_packages, cancel_event=cancel_event)
    actual_names = {path.resolve(strict=True).relative_to(root).as_posix() for path in actual_paths}
    if (
        actual_names != set(expected)
        or _overlay_directory_inventory(site_packages, cancel_event=cancel_event)
        != _expected_parent_directories(expected)
    ):
        raise RuntimeError("The extracted overlay does not exactly match its locked wheels.")
    actual: dict[str, str] = {}
    for path in actual_paths:
        if lease is not None and lease.cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime overlay verification was cancelled.")
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
        digest, _size = _stream_sha256(
            resolved,
            maximum_bytes=MAX_LOCKED_WHEEL_MEMBER_BYTES,
            cancel_event=lease.cancel_event if lease is not None else None,
        )
        if digest != expected[relative]:
            raise RuntimeError("An extracted overlay file differs from its locked wheel.")
        actual[relative] = digest
    anchor_body = {
        "schemaVersion": 1,
        "artifacts": artifact_list,
        "fileSeal": dict(sorted(actual.items())),
    }
    encoded = json.dumps(anchor_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    anchor_body["digest"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    return anchor_body


def observed_overlay_file_seal(site_packages: Path) -> dict[str, str]:
    root = site_packages.resolve(strict=True)
    seal: dict[str, str] = {}
    for path in scan_overlay_tree(site_packages):
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
        digest, _size = _stream_sha256(resolved, maximum_bytes=MAX_LOCKED_WHEEL_MEMBER_BYTES)
        seal[relative] = digest
    return dict(sorted(seal.items()))


_WATCHDOG_SCRIPT = r'''
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

parent_pid = int(sys.argv[1])
command = json.loads(sys.argv[2])

def parent_alive():
    if os.name != "nt":
        return os.getppid() == parent_pid
    SYNCHRONIZE = 0x00100000
    WAIT_TIMEOUT = 0x00000102
    handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
    if not handle:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)

child = subprocess.Popen(command)
while child.poll() is None:
    if not parent_alive():
        if os.name == "nt":
            system_root = Path(os.environ.get("SystemRoot") or r"C:\Windows")
            taskkill = (system_root / "System32" / "taskkill.exe").resolve()
            subprocess.run(
                [str(taskkill), "/PID", str(child.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
            raise SystemExit(137)
        os.killpg(os.getpgrp(), signal.SIGKILL)
    time.sleep(0.05)
raise SystemExit(child.returncode)
'''


def run_cancellable_command(
    command: list[str],
    *,
    environment: dict[str, str],
    lease: InstallLease,
    timeout: int,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Run one owned child process and make cancellation stop that process."""

    if lease.cancel_event.is_set():
        raise OverlayCancelled("Optional-runtime installation was cancelled.")
    creation: dict[str, Any] = {"start_new_session": True}
    lock_fd = lease.lock_file.fileno()
    os.set_inheritable(lock_fd, True)
    if os.name == "nt":
        creation = {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | getattr(subprocess, "CREATE_NO_WINDOW", 0),
            "close_fds": False,
        }
    else:
        creation["pass_fds"] = (lock_fd,)
    started = time.monotonic()
    working_directory = Path(cwd or tempfile.gettempdir()).resolve(strict=True)
    if not working_directory.is_dir():
        raise RuntimeError("Optional-runtime subprocess working directory is unavailable.")
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stdout_file, tempfile.TemporaryFile(
        mode="w+", encoding="utf-8", errors="replace"
    ) as stderr_file:
        watchdog_command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _WATCHDOG_SCRIPT,
            str(os.getpid()),
            json.dumps(command, separators=(",", ":")),
        ]
        try:
            process = subprocess.Popen(
                watchdog_command,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                cwd=str(working_directory),
                **creation,
            )
        finally:
            os.set_inheritable(lock_fd, False)
        _set_lease_process(lease, process)
        try:
            while process.poll() is None:
                if lease.cancel_event.wait(0.05):
                    _terminate_process(process)
                    raise OverlayCancelled("Optional-runtime installation was cancelled.")
                if time.monotonic() - started > timeout:
                    _terminate_process(process)
                    raise RuntimeError("Optional-runtime subprocess exceeded its time limit.")
                output_bytes = os.fstat(stdout_file.fileno()).st_size + os.fstat(stderr_file.fileno()).st_size
                if output_bytes > 8 * 1024**2:
                    _terminate_process(process)
                    raise RuntimeError("Optional-runtime subprocess output exceeded its safe limit.")
            if lease.cancel_event.is_set():
                raise OverlayCancelled("Optional-runtime installation was cancelled.")
            output_bytes = os.fstat(stdout_file.fileno()).st_size + os.fstat(stderr_file.fileno()).st_size
            if output_bytes > 8 * 1024**2:
                raise RuntimeError("Optional-runtime subprocess output exceeded its safe limit.")
            stdout_file.seek(0, os.SEEK_END)
            stdout_size = stdout_file.tell()
            stdout_file.seek(max(0, stdout_size - 8000))
            stderr_file.seek(0, os.SEEK_END)
            stderr_size = stderr_file.tell()
            stderr_file.seek(max(0, stderr_size - 8000))
            return {
                "returnCode": int(process.returncode or 0),
                "stdout": stdout_file.read(),
                "stderr": stderr_file.read(),
                "elapsedSeconds": time.monotonic() - started,
            }
        finally:
            _set_lease_process(lease, None)


def _normalized_distribution(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(".", "-")


def _resolved_distribution_origin(distribution: metadata.Distribution) -> str:
    raw_path = getattr(distribution, "_path", None)
    if raw_path is None:
        raise RuntimeError("Installed distribution metadata has no resolvable origin.")
    return str(Path(raw_path).resolve(strict=True))


def _import_identity(import_name: str, metadata_origin: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        raise RuntimeError(f"Required base module {import_name!r} is unavailable.")
    metadata_root = Path(metadata_origin).resolve(strict=True).parent
    origin = spec.origin
    resolved_origin = None
    if origin not in {None, "namespace", "built-in", "frozen"}:
        resolved = Path(origin).resolve(strict=True)
        try:
            resolved.relative_to(metadata_root)
        except ValueError as exc:
            raise RuntimeError(f"Base module {import_name!r} does not match its distribution origin.") from exc
        resolved_origin = str(resolved)
    locations = []
    for location in list(spec.submodule_search_locations or []):
        resolved = Path(location).resolve(strict=True)
        try:
            resolved.relative_to(metadata_root)
        except ValueError as exc:
            raise RuntimeError(f"Base package {import_name!r} has an unexpected search path.") from exc
        locations.append(str(resolved))
    if resolved_origin is None and not locations:
        raise RuntimeError(f"Base module {import_name!r} has no filesystem import identity.")
    return {"origin": resolved_origin, "searchLocations": sorted(locations)}


def _diffusers_identity() -> dict[str, str]:
    try:
        distribution = metadata.distribution("diffusers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("The pinned Diffusers distribution is unavailable.") from exc
    try:
        direct = json.loads(distribution.read_text("direct_url.json") or "")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Diffusers has no readable immutable source receipt.") from exc
    vcs = direct.get("vcs_info") if isinstance(direct, dict) else None
    source = str(direct.get("url") or "") if isinstance(direct, dict) else ""
    commit = str(vcs.get("commit_id") or "") if isinstance(vcs, dict) else ""
    requested = str(vcs.get("requested_revision") or "") if isinstance(vcs, dict) else ""
    if (
        str(distribution.version) != PINNED_DIFFUSERS_VERSION
        or source != PINNED_DIFFUSERS_SOURCE_URL
        or not isinstance(vcs, dict)
        or vcs.get("vcs") != "git"
        or commit != PINNED_DIFFUSERS_COMMIT
        or requested != PINNED_DIFFUSERS_COMMIT
        or "dir_info" in direct
    ):
        raise RuntimeError("Installed Diffusers does not match MoDiff's reviewed source and commit.")
    metadata_origin = _resolved_distribution_origin(distribution)
    return {
        "distribution": "diffusers",
        "version": str(distribution.version),
        "sourceUrl": source,
        "vcs": "git",
        "dirInfoPresent": False,
        "commitId": commit,
        "requestedRevision": requested,
        "metadataOrigin": metadata_origin,
        "importIdentity": _import_identity("diffusers", metadata_origin),
    }


def _accelerator_identity() -> dict[str, str]:
    # runtime_profile is itself standard-library only and does not import Torch.
    from modiff.runtime_profile import (
        MANIFEST_PATH,
        PROJECT_ROOT,
        RUNTIME_CONTRACT_SCHEMA,
        load_manifest,
        lock_digest,
        read_state,
        runtime_contract_paths,
    )

    saved = read_state(Path(sys.prefix))
    if (
        not isinstance(saved, dict)
        or saved.get("runtime_contract_schema") != RUNTIME_CONTRACT_SCHEMA
    ):
        raise RuntimeError("The accelerator profile has no verified installation receipt.")
    profile_id = str(saved.get("profile") or "")
    installed_digest = str(saved.get("lock_digest") or "")
    manifest = load_manifest()
    profile = manifest.get("profiles", {}).get(profile_id)
    requirement_name = profile.get("requirements") if isinstance(profile, dict) else None
    if not profile_id or not isinstance(requirement_name, str) or not requirement_name:
        raise RuntimeError("The accelerator profile contract is unavailable.")
    requirement = PROJECT_ROOT / requirement_name
    current_digest = lock_digest(
        requirement,
        contract_paths=runtime_contract_paths(requirement),
        profile=profile_id,
    )
    if installed_digest != current_digest:
        raise RuntimeError("The accelerator profile lock has drifted and requires repair.")
    contract_files = []
    for path in (*runtime_contract_paths(requirement), MANIFEST_PATH):
        resolved = Path(path).resolve(strict=True)
        body = resolved.read_bytes()
        contract_files.append(
            {
                "path": str(resolved),
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    return {
        "profileId": profile_id,
        "lockDigest": current_digest,
        "manifestRevision": str(manifest.get("revision") or ""),
        "contractFiles": contract_files,
    }


def current_base_binding(base_distributions: Iterable[str | dict[str, Any]]) -> dict[str, Any]:
    """Freeze every host identity an overlay is allowed to depend on."""

    base_packages = []
    seen: set[str] = set()
    for raw_contract in base_distributions:
        if isinstance(raw_contract, dict):
            raw_name = raw_contract.get("distribution")
            import_name = str(raw_contract.get("importName") or "").strip()
            specifier = str(raw_contract.get("specifier") or "").strip()
        else:
            raw_name = raw_contract
            import_name = str(raw_contract).replace("-", "_")
            specifier = ""
        name = _normalized_distribution(raw_name)
        if not name or not import_name or name in seen:
            continue
        seen.add(name)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required base distribution {name!r} is unavailable.") from exc
        base_packages.append(
            {
                "distribution": name,
                "importName": import_name,
                "specifier": specifier,
                "version": str(distribution.version),
                "metadataOrigin": _resolved_distribution_origin(distribution),
            }
        )
        base_packages[-1]["importIdentity"] = _import_identity(
            import_name,
            base_packages[-1]["metadataOrigin"],
        )
    packaging_contract = next(
        (item for item in base_packages if item["distribution"] == "packaging"),
        None,
    )
    if packaging_contract is None:
        raise RuntimeError("The optional-runtime base contract must include packaging.")
    packaging_module = importlib.import_module("packaging")
    if str(Path(packaging_module.__file__).resolve(strict=True)) != packaging_contract["importIdentity"]["origin"]:
        raise RuntimeError("The trusted packaging module does not match its frozen import origin.")
    from packaging.specifiers import SpecifierSet

    for package in base_packages:
        if package["specifier"] and package["version"] not in SpecifierSet(package["specifier"]):
            raise RuntimeError(
                f"Base distribution {package['distribution']!r} violates its reviewed constraint."
            )
    return {
        "schemaVersion": 1,
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
            "cacheTag": str(sys.implementation.cache_tag or ""),
            "hexVersion": sys.hexversion,
        },
        "accelerator": _accelerator_identity(),
        "diffusers": _diffusers_identity(),
        "basePackages": sorted(base_packages, key=lambda item: item["distribution"]),
    }


def binding_digest(binding: dict[str, Any]) -> str:
    body = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{_DIGEST_PREFIX}{hashlib.sha256(body).hexdigest()}"


def binding_matches(
    expected: dict[str, Any],
    base_distributions: Iterable[str | dict[str, Any]],
) -> bool:
    try:
        return expected == current_base_binding(base_distributions)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def scan_overlay_tree(
    site_packages: Path, *, cancel_event: threading.Event | None = None
) -> list[Path]:
    """Reject links, reparse points, and .pth startup directives."""

    root = site_packages.resolve(strict=True)
    maximum_entries = 100_000
    maximum_bytes = 2 * 1024**3
    maximum_file_bytes = 512 * 1024**2
    maximum_depth = 64
    entries = 1
    total_bytes = 0
    regular_files: list[Path] = []
    pending = [site_packages]
    while pending:
        if cancel_event is not None and cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime overlay scan was cancelled.")
        path = pending.pop()
        try:
            depth = len(path.relative_to(site_packages).parts)
        except ValueError as exc:
            raise RuntimeError("The staged overlay escapes its managed root.") from exc
        if depth > maximum_depth:
            raise RuntimeError("The staged overlay exceeds the maximum directory depth.")
        try:
            info = path.lstat()
        except OSError as exc:
            raise RuntimeError("The staged overlay contains an unreadable path.") from exc
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise RuntimeError("The staged overlay contains a link or reparse-point escape.")
        try:
            path.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise RuntimeError("The staged overlay escapes its managed site-packages root.") from exc
        if path.is_file() and path.suffix.lower() == ".pth":
            raise RuntimeError("The staged overlay contains a forbidden .pth startup directive.")
        if path.name.lower() in {"sitecustomize.py", "usercustomize.py"}:
            raise RuntimeError("The staged overlay contains a forbidden interpreter startup module.")
        if stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1:
                raise RuntimeError("The staged overlay contains a multiply-linked file.")
            regular_files.append(path)
            total_bytes += int(info.st_size)
            if info.st_size > maximum_file_bytes:
                raise RuntimeError("The staged overlay contains an oversized file.")
            if total_bytes > maximum_bytes:
                raise RuntimeError("The staged overlay exceeds the maximum validated size.")
        elif stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(path) as children:
                    for child in children:
                        entries += 1
                        if entries > maximum_entries:
                            raise RuntimeError("The staged overlay contains too many filesystem entries.")
                        pending.append(Path(child.path))
            except OSError as exc:
                raise RuntimeError("The staged overlay contains an unreadable directory.") from exc
        else:
            raise RuntimeError("The staged overlay contains a non-regular filesystem entry.")
    return regular_files


def overlay_file_seal(site_packages: Path, expected_distributions: Iterable[str]) -> dict[str, str]:
    """Require every staged file to be owned by a reviewed wheel RECORD."""

    root = site_packages.resolve(strict=True)
    regular_files = scan_overlay_tree(site_packages)
    expected = {_normalized_distribution(name) for name in expected_distributions}
    # Bound every metadata document before importlib.metadata is allowed to
    # parse it. Its ``Distribution.files`` and ``metadata`` properties may
    # otherwise materialize an attacker-sized RECORD/METADATA document.
    metadata_roots: set[Path] = set()
    for path in regular_files:
        relative = path.resolve(strict=True).relative_to(root)
        if len(relative.parts) != 2 or not relative.parts[0].endswith(".dist-info"):
            continue
        if relative.name not in {"METADATA", "RECORD"}:
            continue
        metadata_root = (root / relative.parts[0]).resolve(strict=True)
        metadata_roots.add(metadata_root)
    if len(metadata_roots) != len(expected):
        raise RuntimeError("The staged wheel metadata inventory is incomplete or unexpected.")
    for metadata_root in metadata_roots:
        metadata_path = metadata_root / "METADATA"
        record_path = metadata_root / "RECORD"
        try:
            metadata_info = metadata_path.lstat()
            record_info = record_path.lstat()
        except OSError as exc:
            raise RuntimeError("A staged wheel has missing metadata inventory.") from exc
        if (
            not stat.S_ISREG(metadata_info.st_mode)
            or not stat.S_ISREG(record_info.st_mode)
            or metadata_info.st_size > 2 * 1024 * 1024
            or record_info.st_size > 4 * 1024 * 1024
        ):
            raise RuntimeError("A staged wheel has oversized or invalid metadata inventory.")
        with record_path.open("r", encoding="utf-8", errors="strict", newline="") as record:
            for record_rows, row in enumerate(record, start=1):
                if record_rows > 100_000 or len(row) > 16_384:
                    raise RuntimeError("A staged wheel RECORD exceeds its safe parsing bounds.")
    discovered: dict[str, list[metadata.Distribution]] = {}
    for distribution in metadata.distributions(path=[str(root)]):
        raw_metadata_root = getattr(distribution, "_path", None)
        try:
            metadata_root = Path(raw_metadata_root).resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise RuntimeError("A staged distribution metadata origin is invalid.") from exc
        if metadata_root not in metadata_roots:
            raise RuntimeError("The staged overlay contains unexpected distribution metadata.")
        name = _normalized_distribution(distribution.metadata.get("Name") or "")
        discovered.setdefault(name, []).append(distribution)
    if set(discovered) != expected or any(len(values) != 1 for values in discovered.values()):
        raise RuntimeError("The staged distribution set does not match the reviewed closure.")
    owned: set[Path] = set()
    for distributions in discovered.values():
        distribution = distributions[0]
        files = distribution.files
        if files is None:
            raise RuntimeError("A staged distribution has no RECORD file inventory.")
        for relative in files:
            try:
                lexical = Path(distribution.locate_file(relative)).resolve(strict=False)
                lexical.relative_to(root)
            except ValueError:
                # Wheel console scripts may be recorded outside target
                # site-packages; they are not importable overlay content.
                continue
            try:
                located = lexical.resolve(strict=True)
            except OSError as exc:
                raise RuntimeError("A staged wheel RECORD entry is missing.") from exc
            if not located.is_file():
                raise RuntimeError("A staged wheel RECORD entry is not a regular file.")
            file_hash = getattr(relative, "hash", None)
            if file_hash is not None:
                hasher = hashlib.new(file_hash.mode)
                with located.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        hasher.update(chunk)
                observed = base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode("ascii")
                if observed != file_hash.value:
                    raise RuntimeError("A staged wheel file does not match its RECORD digest.")
            owned.add(located)
    actual = {path.resolve(strict=True) for path in regular_files}
    owned_relative = {path.relative_to(root).as_posix() for path in owned}
    if (
        actual != owned
        or _overlay_directory_inventory(site_packages)
        != _expected_parent_directories(owned_relative)
    ):
        raise RuntimeError("The staged overlay contains unowned or missing wheel files.")
    seal: dict[str, str] = {}
    for path in sorted(actual, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
        seal[relative] = hasher.hexdigest()
    return seal


def overlay_file_seal_matches(
    site_packages: Path,
    expected_distributions: Iterable[str],
    expected_seal: dict[str, str],
) -> bool:
    try:
        return overlay_file_seal(site_packages, expected_distributions) == expected_seal
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


_VALIDATION_SCRIPT = r'''
import importlib
import importlib.util
from importlib import metadata
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import stat
import sys

payload_body = Path(sys.argv[1]).read_bytes()
if len(payload_body) > 4 * 1024 * 1024 or hashlib.sha256(payload_body).hexdigest() != sys.argv[2]:
    raise RuntimeError("the isolated validation payload failed its bounded digest")
payload = json.loads(payload_body.decode("utf-8"))
site = Path(payload["sitePackages"]).resolve(strict=True)

def verify_trusted_file_seal():
    expected = payload.get("trustedFileSeal")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("the isolated validator has no trusted file seal")
    actual = {}
    pending = [(site, 0)]
    entries = 1
    total_bytes = 0
    while pending:
        path, depth = pending.pop()
        if depth > 64:
            raise RuntimeError("the overlay exceeds the validator depth bound")
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode) or bool(
            getattr(details, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise RuntimeError("the overlay contains a link or reparse point")
        path.resolve(strict=True).relative_to(site)
        if stat.S_ISDIR(details.st_mode):
            with os.scandir(path) as children:
                for child in children:
                    entries += 1
                    if entries > 100_000:
                        raise RuntimeError("the overlay exceeds the validator entry bound")
                    pending.append((Path(child.path), depth + 1))
            continue
        if not stat.S_ISREG(details.st_mode):
            raise RuntimeError("the overlay contains a non-regular entry")
        if details.st_nlink != 1:
            raise RuntimeError("the overlay contains a multiply-linked file")
        relative = path.resolve(strict=True).relative_to(site).as_posix()
        if path.suffix.lower() == ".pth" or path.name.lower() in {"sitecustomize.py", "usercustomize.py"}:
            raise RuntimeError("the overlay contains an interpreter startup hook")
        if details.st_size > 512 * 1024**2:
            raise RuntimeError("an overlay file exceeds the validator size bound")
        total_bytes += details.st_size
        if total_bytes > 2 * 1024**3:
            raise RuntimeError("the overlay exceeds the validator total size bound")
        hasher = hashlib.sha256()
        observed = 0
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                observed += len(chunk)
                if observed > details.st_size:
                    raise RuntimeError("an overlay file changed while validation read it")
                hasher.update(chunk)
        if observed != details.st_size:
            raise RuntimeError("an overlay file changed while validation read it")
        actual[relative] = hasher.hexdigest()
    if actual != expected:
        raise RuntimeError("the overlay no longer matches its trusted file seal")

verify_trusted_file_seal()
sys.path.insert(0, str(site))

def deny_network(event, _args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname"}:
        raise RuntimeError("network access is forbidden during optional-runtime validation")

sys.addaudithook(deny_network)

def inside(path):
    try:
        Path(path).resolve(strict=True).relative_to(site)
        return True
    except (OSError, TypeError, ValueError):
        return False

def inside_root(path, root):
    try:
        Path(path).resolve(strict=True).relative_to(Path(root).resolve(strict=True))
        return True
    except (OSError, TypeError, ValueError):
        return False

def assert_import_origin(import_name, metadata_origin, *, must_be_overlay, expected_identity=None):
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        raise RuntimeError(f"module {import_name!r} has no import specification")
    roots = list(spec.submodule_search_locations or [])
    origin = spec.origin
    candidates = [value for value in [origin, *roots] if value not in {None, "namespace", "built-in", "frozen"}]
    if not candidates:
        raise RuntimeError(f"module {import_name!r} has no filesystem origin")
    metadata_root = str(Path(metadata_origin).resolve(strict=True).parent)
    observed_origin = None
    observed_locations = []
    for candidate in candidates:
        if must_be_overlay:
            if not inside(candidate):
                raise RuntimeError(f"module {import_name!r} escapes the overlay")
        elif inside(candidate) or not inside_root(candidate, metadata_root):
            raise RuntimeError(f"base module {import_name!r} has an unexpected import origin")
    if origin not in {None, "namespace", "built-in", "frozen"}:
        observed_origin = str(Path(origin).resolve(strict=True))
    observed_locations = sorted(str(Path(value).resolve(strict=True)) for value in roots)
    if expected_identity is not None and {
        "origin": observed_origin,
        "searchLocations": observed_locations,
    } != expected_identity:
        raise RuntimeError(f"base module {import_name!r} import identity drifted")

for spec_record in payload["specs"]:
    canonical = json.dumps(spec_record["spec"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    observed_digest = "sha256:" + hashlib.sha256(canonical).hexdigest()
    if observed_digest != spec_record["specDigest"]:
        raise RuntimeError("the executable optional-runtime spec digest does not match")

canonical_binding = json.dumps(payload["binding"], sort_keys=True, separators=(",", ":")).encode("utf-8")
if "sha256:" + hashlib.sha256(canonical_binding).hexdigest() != payload["bindingDigest"]:
    raise RuntimeError("the host binding digest does not match")
for contract_file in payload["binding"]["accelerator"]["contractFiles"]:
    body = Path(contract_file["path"]).resolve(strict=True).read_bytes()
    if len(body) != contract_file["size"] or hashlib.sha256(body).hexdigest() != contract_file["sha256"]:
        raise RuntimeError("the accelerator profile contract changed during validation")

overlay_distributions = {}
for distribution in metadata.distributions(path=[str(site)]):
    name = str(distribution.metadata.get("Name") or "").lower().replace("_", "-").replace(".", "-")
    overlay_distributions.setdefault(name, []).append(distribution)

expected_distributions = {contract["distribution"] for contract in payload["packages"]}
if set(overlay_distributions) != expected_distributions:
    raise RuntimeError("the overlay distribution set does not match the reviewed staged closure")

observed = []
for contract in payload["packages"]:
    name = contract["distribution"]
    matches = overlay_distributions.get(name, [])
    if len(matches) != 1:
        raise RuntimeError(f"overlay distribution {name!r} has {len(matches)} metadata records")
    distribution = matches[0]
    if str(distribution.version) != contract["requiredVersion"]:
        raise RuntimeError(f"overlay distribution {name!r} has the wrong version")
    if not inside(getattr(distribution, "_path", None)):
        raise RuntimeError(f"overlay distribution {name!r} metadata escapes the overlay")
    assert_import_origin(contract["importName"], getattr(distribution, "_path", None), must_be_overlay=True)
    module = importlib.import_module(contract["importName"])
    if not inside(getattr(module, "__file__", None)):
        raise RuntimeError(f"overlay module {contract['importName']!r} did not load from the overlay")
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if origin not in {None, "namespace"} and not inside(origin):
        raise RuntimeError(f"overlay module {contract['importName']!r} spec escapes the overlay")
    for package_path in list(getattr(module, "__path__", []) or []):
        if not inside(package_path):
            raise RuntimeError(f"overlay module {contract['importName']!r} package path escapes the overlay")
    required_classes = set(contract.get("requiredClassSymbols", []))
    for symbol in contract.get("requiredSymbols", []):
        if ":" in symbol:
            module_name, attribute_path = symbol.split(":", 1)
            value = importlib.import_module(module_name)
            if not inside(getattr(value, "__file__", None)):
                raise RuntimeError("a reviewed overlay submodule escaped the overlay")
        else:
            attribute_path = symbol
            value = module
        for part in attribute_path.split("."):
            if not hasattr(value, part):
                raise RuntimeError(f"overlay module {contract['importName']!r} lacks a reviewed symbol")
            value = getattr(value, part)
        if ".dummy_" in str(getattr(value, "__module__", "")):
            raise RuntimeError(f"overlay module {contract['importName']!r} resolved a dummy symbol")
        if symbol in required_classes:
            if not inspect.isclass(value):
                raise RuntimeError(f"overlay module {contract['importName']!r} resolved a non-class symbol")
        elif not callable(value):
            raise RuntimeError(f"overlay module {contract['importName']!r} resolved a non-callable symbol")
    observed.append({"distribution": name, "version": str(distribution.version)})

for base in payload["binding"]["basePackages"]:
    name = base["distribution"]
    if name in overlay_distributions:
        raise RuntimeError(f"base-owned distribution {name!r} was shadowed into the overlay")
    distribution = metadata.distribution(name)
    origin = str(Path(getattr(distribution, "_path", "")).resolve(strict=True))
    if str(distribution.version) != base["version"] or origin != base["metadataOrigin"]:
        raise RuntimeError(f"base-owned distribution {name!r} drifted during validation")
    assert_import_origin(
        base["importName"],
        origin,
        must_be_overlay=False,
        expected_identity=base["importIdentity"],
    )
    base_module = importlib.import_module(base["importName"])
    actual_origin = getattr(base_module, "__file__", None)
    if base["importIdentity"]["origin"] is not None and str(Path(actual_origin).resolve(strict=True)) != base["importIdentity"]["origin"]:
        raise RuntimeError(f"base module {base['importName']!r} loaded from an unexpected origin")
    actual_locations = sorted(str(Path(value).resolve(strict=True)) for value in list(getattr(base_module, "__path__", []) or []))
    if actual_locations != base["importIdentity"]["searchLocations"]:
        raise RuntimeError(f"base package {base['importName']!r} loaded with an unexpected search path")

from packaging.specifiers import SpecifierSet
for base in payload["binding"]["basePackages"]:
    if base.get("specifier") and base["version"] not in SpecifierSet(base["specifier"]):
        raise RuntimeError(f"base-owned distribution {base['distribution']!r} violates its reviewed constraint")

diffusers = metadata.distribution("diffusers")
direct = json.loads(diffusers.read_text("direct_url.json") or "")
vcs = direct.get("vcs_info") if isinstance(direct, dict) else None
diffusers_identity = payload["binding"]["diffusers"]
if (
    str(diffusers.version) != diffusers_identity["version"]
    or direct.get("url") != diffusers_identity["sourceUrl"]
    or not isinstance(vcs, dict)
    or vcs.get("vcs") != diffusers_identity["vcs"]
    or vcs.get("commit_id") != diffusers_identity["commitId"]
    or vcs.get("requested_revision") != diffusers_identity["requestedRevision"]
    or ("dir_info" in direct) != diffusers_identity["dirInfoPresent"]
    or str(Path(getattr(diffusers, "_path", "")).resolve(strict=True)) != diffusers_identity["metadataOrigin"]
):
    raise RuntimeError("Diffusers identity drifted during validation")
assert_import_origin(
    "diffusers",
    diffusers_identity["metadataOrigin"],
    must_be_overlay=False,
    expected_identity=diffusers_identity["importIdentity"],
)
diffusers_module = importlib.import_module("diffusers")
if str(Path(diffusers_module.__file__).resolve(strict=True)) != diffusers_identity["importIdentity"]["origin"]:
    raise RuntimeError("Diffusers loaded from an unexpected origin")
if sorted(str(Path(value).resolve(strict=True)) for value in list(diffusers_module.__path__)) != diffusers_identity["importIdentity"]["searchLocations"]:
    raise RuntimeError("Diffusers loaded with an unexpected package path")
dummy_type = importlib.import_module("diffusers.utils.import_utils").DummyObject
diffusers_root = str(Path(diffusers_identity["metadataOrigin"]).resolve(strict=True).parent)
for spec_record in payload["specs"]:
    for symbol in spec_record["spec"].get("requiredDiffusersSymbols", []):
        value = diffusers_module
        for part in symbol.split("."):
            if not hasattr(value, part):
                raise RuntimeError("Diffusers lacks a reviewed optional-runtime symbol")
            value = getattr(value, part)
        if (
            not inspect.isclass(value)
            or getattr(value, "__name__", None) != symbol.split(".")[-1]
            or not str(getattr(value, "__module__", "")).startswith("diffusers")
            or not inside_root(inspect.getfile(value), diffusers_root)
            or isinstance(value, dummy_type)
            or ".utils.dummy_" in str(getattr(value, "__module__", ""))
        ):
            raise RuntimeError("Diffusers resolved a dummy optional-runtime symbol")
    required_method_parameters = {
        item["method"]: set(item.get("requiredParameters", []))
        for item in spec_record["spec"].get("pipelineAdapterMethods", [])
    }
    for pipeline_symbol in spec_record["spec"].get("pipelineAdapterSymbols", []):
        pipeline_class = getattr(diffusers_module, pipeline_symbol, None)
        if not inspect.isclass(pipeline_class) or isinstance(pipeline_class, dummy_type):
            raise RuntimeError("Diffusers lacks a real reviewed pipeline adapter class")
        for method_name, parameter_names in required_method_parameters.items():
            method = getattr(pipeline_class, method_name, None)
            if not callable(method) or not parameter_names.issubset(inspect.signature(method).parameters):
                raise RuntimeError("A reviewed Diffusers pipeline adapter signature drifted")
    if (
        spec_record.get("kind") == "optional_runtime"
        and "add_weighted_adapter"
        not in spec_record["spec"].get("excludedQualificationSymbols", [])
    ):
        raise RuntimeError("The unproven weighted-adapter merge API must remain excluded")
if any(spec_record["spec"].get("requirePeftBackend") for spec_record in payload["specs"]):
    if importlib.import_module("diffusers.utils").USE_PEFT_BACKEND is not True:
        raise RuntimeError("Diffusers did not enable its reviewed PEFT backend")

python_identity = payload["binding"]["python"]
if (
    sys.implementation.name != python_identity["implementation"]
    or platform.python_version() != python_identity["version"]
    or str(sys.implementation.cache_tag or "") != python_identity["cacheTag"]
    or sys.hexversion != python_identity["hexVersion"]
):
    raise RuntimeError("Python identity drifted during validation")

print(json.dumps({"status": "passed", "packages": observed}, sort_keys=True))
'''


def run_fresh_validation(
    site_packages: Path,
    *,
    packages: Iterable[dict[str, Any]],
    binding: dict[str, Any],
    specs: Iterable[dict[str, Any]],
    lease: InstallLease,
    timeout: int = 180,
    trusted_file_seal: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate exact versions, origins, symbols, and host binding in ``-I``."""

    package_list = list(packages)
    expected_names = [str(package["distribution"]) for package in package_list]
    before_seal = (
        observed_overlay_file_seal(site_packages)
        if trusted_file_seal is not None
        else overlay_file_seal(site_packages, expected_names)
    )
    if trusted_file_seal is not None and before_seal != trusted_file_seal:
        raise RuntimeError("The staged overlay does not match its catalog-derived file seal.")
    payload = {
        "sitePackages": str(site_packages.resolve(strict=True)),
        "packages": package_list,
        "binding": binding,
        "bindingDigest": binding_digest(binding),
        "specs": list(specs),
        "trustedFileSeal": before_seal,
    }
    validation_environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in {"PYTHONPATH", "PYTHONHOME"}
    }
    validation_environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
    )
    with tempfile.TemporaryDirectory(prefix="modiff-overlay-validation-") as temporary:
        validation_root = Path(temporary).resolve(strict=True)
        payload_path = validation_root / "payload.json"
        payload_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(payload_body) > 4 * 1024 * 1024:
            raise RuntimeError("The isolated validation payload exceeds its safe limit.")
        payload_path.write_bytes(payload_body)
        result = run_cancellable_command(
            [
                sys.executable,
                "-I",
                "-s",
                "-B",
                "-c",
                _VALIDATION_SCRIPT,
                str(payload_path),
                hashlib.sha256(payload_body).hexdigest(),
            ],
            environment=validation_environment,
            lease=lease,
            timeout=timeout,
            cwd=validation_root,
        )
    detail = None
    if result["stdout"].strip():
        try:
            detail = json.loads(result["stdout"].strip().splitlines()[-1])
        except ValueError:
            detail = None
    expected = sorted(str(package["distribution"]) for package in payload["packages"])
    observed = (
        sorted(str(package.get("distribution")) for package in detail.get("packages", []))
        if isinstance(detail, dict) and isinstance(detail.get("packages"), list)
        else []
    )
    passed = (
        result["returnCode"] == 0
        and isinstance(detail, dict)
        and detail.get("status") == "passed"
        and observed == expected
        and not lease.cancel_event.is_set()
        and (
            observed_overlay_file_seal(site_packages)
            if trusted_file_seal is not None
            else overlay_file_seal(site_packages, expected_names)
        )
        == before_seal
    )
    return {
        "status": "passed" if passed else "failed",
        "returnCode": result["returnCode"],
        "detail": detail if passed else None,
        "error": None if passed else "The isolated optional-runtime validation failed.",
        "elapsedSeconds": result["elapsedSeconds"],
        "binding": binding if passed else None,
        "bindingDigest": binding_digest(binding) if passed else None,
        "fileSeal": before_seal if passed else None,
    }
