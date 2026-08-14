from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import validate_repo_id

from modiff.config import CONFIG
from utils.huggingface import HF_DOWNLOAD_FREE_SPACE_RESERVE_BYTES, app_hub_download_mode


TEMPLATE_GALLERY_SOURCE_PATH = Path("web/assets/template-asset-source.v1.json")
TEMPLATE_GALLERY_ROOT = Path("web/template-gallery")
TEMPLATE_GALLERY_MANIFEST_MAX_BYTES = 8 * 1024**2
TEMPLATE_GALLERY_SOURCE_MAX_BYTES = 64 * 1024
TEMPLATE_GALLERY_MAX_ASSETS = 10_000
TEMPLATE_GALLERY_MAX_FILE_BYTES = 2 * 1024**3
TEMPLATE_GALLERY_MAX_TOTAL_BYTES = 32 * 1024**3
TEMPLATE_GALLERY_DOWNLOAD_OVERHEAD_BYTES = 8 * 1024**2
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
ASSET_SET_ID = re.compile(r"^sha256:canonical-json:[0-9a-f]{64}$")
BYTE_SHA256 = re.compile(r"^sha256:bytes:[0-9a-f]{64}$")


class TemplateGalleryError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_bounded_json(path: Path, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    try:
        details = path.stat()
    except OSError as error:
        raise TemplateGalleryError("template_gallery_contract_missing", f"{label} is unavailable.") from error
    if not path.is_file() or details.st_size < 2 or details.st_size > maximum_bytes:
        raise TemplateGalleryError("template_gallery_contract_invalid", f"{label} is not a bounded regular file.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TemplateGalleryError("template_gallery_contract_invalid", f"{label} is not valid JSON.") from error
    if not isinstance(value, dict):
        raise TemplateGalleryError("template_gallery_contract_invalid", f"{label} must be a JSON object.")
    return value


def _safe_gallery_path(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("template-gallery/") or len(value) > 512:
        raise TemplateGalleryError("template_gallery_manifest_invalid", "Gallery asset paths must use the canonical prefix.")
    if "\\" in value or "//" in value:
        raise TemplateGalleryError("template_gallery_manifest_invalid", f"Gallery asset path is unsafe: {value!r}.")
    parts = PurePosixPath(value).parts
    if len(parts) < 2 or parts[0] != "template-gallery" or any(part in {"", ".", ".."} for part in parts):
        raise TemplateGalleryError("template_gallery_manifest_invalid", f"Gallery asset path is unsafe: {value!r}.")
    return value


def load_template_gallery_source(path: Path = TEMPLATE_GALLERY_SOURCE_PATH) -> dict[str, Any]:
    source = _read_bounded_json(path, maximum_bytes=TEMPLATE_GALLERY_SOURCE_MAX_BYTES, label="Template Gallery source")
    unavailable = source.get("unavailableAssets")
    try:
        validate_repo_id(source.get("repoId"))
    except (TypeError, ValueError) as error:
        raise TemplateGalleryError("template_gallery_source_invalid", "Template Gallery source repository is invalid.") from error
    if (
        source.get("schemaVersion") != 1
        or source.get("mode") != "huggingface"
        or source.get("repoType") != "dataset"
        or source.get("localBasePath") != "/template-gallery"
        or source.get("pathPrefix") != "template-gallery"
        or source.get("assetManifestPath") != "_modiff/template-assets.v1.json"
        or not IMMUTABLE_REVISION.fullmatch(str(source.get("revision") or ""))
        or not ASSET_SET_ID.fullmatch(str(source.get("assetSetId") or ""))
        or not ASSET_SET_ID.fullmatch(str(source.get("completeAssetSetId") or ""))
        or not isinstance(unavailable, list)
        or len(unavailable) > TEMPLATE_GALLERY_MAX_ASSETS
        or unavailable != sorted(set(unavailable))
    ):
        raise TemplateGalleryError("template_gallery_source_invalid", "Template Gallery source contract is invalid.")
    for asset_path in unavailable:
        _safe_gallery_path(asset_path)
    return source


def validate_template_gallery_manifest(source: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    records = manifest.get("assets")
    missing = manifest.get("missingAssets")
    if (
        manifest.get("schemaVersion") != 1
        or not isinstance(manifest.get("assetVersion"), str)
        or not manifest["assetVersion"]
        or not isinstance(records, list)
        or len(records) < 1
        or len(records) > TEMPLATE_GALLERY_MAX_ASSETS
        or not isinstance(missing, list)
        or missing != sorted(set(missing))
    ):
        raise TemplateGalleryError("template_gallery_manifest_invalid", "Pinned Template Gallery manifest is malformed.")

    unavailable = source["unavailableAssets"]
    if unavailable:
        if (
            manifest.get("status") != "blocked"
            or manifest.get("publicationKind") != "provisional-rights-approved-subset"
            or missing != unavailable
            or manifest.get("completeAssetSetId") != source["completeAssetSetId"]
        ):
            raise TemplateGalleryError(
                "template_gallery_manifest_invalid",
                "Pinned Template Gallery subset does not match the reviewed unavailable-asset contract.",
            )
    elif manifest.get("status") != "ready" or missing:
        raise TemplateGalleryError("template_gallery_manifest_invalid", "Pinned Template Gallery is not release-ready.")

    paths: list[str] = []
    total_bytes = 0
    expected_record_keys = {"path", "size", "sha256", "contentType", "purposes"}
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise TemplateGalleryError("template_gallery_manifest_invalid", "Gallery asset record shape is invalid.")
        path = _safe_gallery_path(record.get("path"))
        size = record.get("size")
        purposes = record.get("purposes")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > TEMPLATE_GALLERY_MAX_FILE_BYTES
            or not BYTE_SHA256.fullmatch(str(record.get("sha256") or ""))
            or not isinstance(record.get("contentType"), str)
            or not record["contentType"]
            or len(record["contentType"]) > 128
            or not isinstance(purposes, list)
            or not purposes
            or len(purposes) > 16
            or any(not isinstance(purpose, str) or not purpose or len(purpose) > 64 for purpose in purposes)
        ):
            raise TemplateGalleryError("template_gallery_manifest_invalid", f"Gallery asset metadata is invalid: {path}.")
        paths.append(path)
        total_bytes += size
    if (
        paths != sorted(set(paths))
        or set(paths).intersection(missing)
        or total_bytes > TEMPLATE_GALLERY_MAX_TOTAL_BYTES
        or manifest.get("assetCount") != len(records)
        or manifest.get("totalBytes") != total_bytes
    ):
        raise TemplateGalleryError("template_gallery_manifest_invalid", "Gallery asset inventory totals are invalid.")

    identity_payload = {
        "schemaVersion": 1,
        "assetVersion": manifest["assetVersion"],
        "assets": records,
    }
    calculated_identity = "sha256:canonical-json:" + hashlib.sha256(
        _canonical_json_bytes(identity_payload)
    ).hexdigest()
    if manifest.get("assetSetId") != source["assetSetId"] or calculated_identity != source["assetSetId"]:
        raise TemplateGalleryError(
            "template_gallery_manifest_mismatch",
            "Pinned Template Gallery manifest does not match the app's immutable source identity.",
        )
    return manifest


def configured_template_gallery_cache_root() -> Path:
    configured = CONFIG.hf.get("cache_dir") if isinstance(CONFIG.hf, dict) else None
    return Path(configured or HUGGINGFACE_HUB_CACHE).expanduser()


def fetch_template_gallery_contract(
    source_path: Path = TEMPLATE_GALLERY_SOURCE_PATH,
    *,
    cache_root: Path | None = None,
    download_file: Callable[..., str] = hf_hub_download,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = load_template_gallery_source(source_path)
    try:
        manifest_path = Path(
            download_file(
                repo_id=source["repoId"],
                repo_type="dataset",
                revision=source["revision"],
                filename=source["assetManifestPath"],
                token=False,
                cache_dir=str(cache_root or configured_template_gallery_cache_root()),
            )
        )
    except Exception as error:
        raise TemplateGalleryError(
            "template_gallery_download_unavailable",
            "The app could not fetch the pinned Template Gallery manifest.",
        ) from error
    manifest = _read_bounded_json(
        manifest_path,
        maximum_bytes=TEMPLATE_GALLERY_MANIFEST_MAX_BYTES,
        label="Pinned Template Gallery manifest",
    )
    return source, validate_template_gallery_manifest(source, manifest)


def _asset_destination(root: Path, asset_path: str) -> Path:
    parts = PurePosixPath(_safe_gallery_path(asset_path)).parts[1:]
    return root.joinpath(*parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:bytes:" + digest.hexdigest()


def _path_is_link_or_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(details.st_mode) or bool(
        getattr(details, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def template_gallery_destination_present(root: Path = TEMPLATE_GALLERY_ROOT) -> bool:
    try:
        return root.exists() or _path_is_link_or_reparse(root)
    except OSError:
        return True


def verify_template_gallery_tree(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if _path_is_link_or_reparse(root) or not root.is_dir():
        raise TemplateGalleryError("template_gallery_not_installed", "Template Gallery install directory is absent or unsafe.")
    expected = {record["path"]: record for record in manifest["assets"]}
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directories:
            if _path_is_link_or_reparse(directory_path / name):
                raise TemplateGalleryError("template_gallery_install_unsafe", "Template Gallery contains a linked directory.")
        for name in filenames:
            candidate = directory_path / name
            if _path_is_link_or_reparse(candidate):
                raise TemplateGalleryError("template_gallery_install_unsafe", "Template Gallery contains a linked file.")
            relative = candidate.relative_to(root).as_posix()
            canonical = f"template-gallery/{relative}"
            if canonical not in expected:
                raise TemplateGalleryError("template_gallery_install_mismatch", "Template Gallery contains an unexpected file.")
    for asset_path, record in expected.items():
        candidate = _asset_destination(root, asset_path)
        try:
            details = candidate.lstat()
        except OSError as error:
            raise TemplateGalleryError("template_gallery_install_incomplete", f"Gallery asset is missing: {asset_path}.") from error
        if _path_is_link_or_reparse(candidate) or not candidate.is_file() or details.st_size != record["size"]:
            raise TemplateGalleryError("template_gallery_install_mismatch", f"Gallery asset metadata differs: {asset_path}.")
        if _sha256_file(candidate) != record["sha256"]:
            raise TemplateGalleryError("template_gallery_install_mismatch", f"Gallery asset hash differs: {asset_path}.")
    return {
        "installed": True,
        "complete": True,
        "repairRequired": False,
        "assetCount": len(expected),
        "totalBytes": manifest["totalBytes"],
        "assetSetId": manifest["assetSetId"],
    }


def _existing_storage_anchor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise TemplateGalleryError(
                "template_gallery_storage_unavailable",
                "The Template Gallery storage volume is unavailable.",
            )
        candidate = parent
    return candidate


def plan_template_gallery_install(
    source_path: Path = TEMPLATE_GALLERY_SOURCE_PATH,
    gallery_root: Path = TEMPLATE_GALLERY_ROOT,
    *,
    cache_root: Path | None = None,
    queued_reservation_bytes: int = 0,
    reserve_bytes: int = HF_DOWNLOAD_FREE_SPACE_RESERVE_BYTES,
    download_file: Callable[..., str] = hf_hub_download,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cache = Path(cache_root or configured_template_gallery_cache_root())
    source, manifest = fetch_template_gallery_contract(source_path, cache_root=cache, download_file=download_file)
    installed = False
    repair_reason = None
    try:
        verify_template_gallery_tree(gallery_root, manifest)
        installed = True
    except TemplateGalleryError as error:
        repair_reason = error.code

    cache_anchor = _existing_storage_anchor(cache)
    destination_anchor = _existing_storage_anchor(gallery_root.parent)
    cache_disk = shutil.disk_usage(cache_anchor)
    destination_disk = shutil.disk_usage(destination_anchor)
    download_bytes = 0 if installed else manifest["totalBytes"] + TEMPLATE_GALLERY_DOWNLOAD_OVERHEAD_BYTES
    staging_bytes = 0 if installed else manifest["totalBytes"]
    reservation_bytes = download_bytes + staging_bytes
    same_filesystem = cache_anchor.stat().st_dev == destination_anchor.stat().st_dev
    if same_filesystem:
        fits_cache = fits_destination = (
            reservation_bytes + queued_reservation_bytes + reserve_bytes <= cache_disk.free
        )
    else:
        fits_cache = download_bytes + queued_reservation_bytes + reserve_bytes <= cache_disk.free
        fits_destination = staging_bytes + reserve_bytes <= destination_disk.free
    plan = {
        "schemaVersion": 1,
        "repoId": source["repoId"],
        "repoType": "dataset",
        "revision": source["revision"],
        "assetSetId": source["assetSetId"],
        "assetCount": manifest["assetCount"],
        "totalBytes": manifest["totalBytes"],
        "downloadBytes": download_bytes,
        "stagingBytes": staging_bytes,
        "reservationBytes": reservation_bytes,
        "queuedReservationBytes": queued_reservation_bytes,
        "reserveBytes": reserve_bytes,
        "cacheFreeBytes": cache_disk.free,
        "destinationFreeBytes": destination_disk.free,
        "sameFilesystem": same_filesystem,
        "sizeKnown": True,
        "fitsWithQueue": bool(fits_cache and fits_destination),
        "installed": installed,
        "repairRequired": bool(template_gallery_destination_present(gallery_root) and not installed),
        "repairReason": repair_reason,
    }
    return source, manifest, plan


def _copy_verified_asset(source: Path, destination: Path, record: dict[str, Any]) -> None:
    if not source.is_file():
        raise TemplateGalleryError("template_gallery_download_incomplete", f"Downloaded asset is missing: {record['path']}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        while chunk := source_handle.read(1024 * 1024):
            destination_handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    os.chmod(destination, 0o644)
    if size != record["size"] or "sha256:bytes:" + digest.hexdigest() != record["sha256"]:
        raise TemplateGalleryError("template_gallery_download_mismatch", f"Downloaded asset differs: {record['path']}.")


def _write_receipt(path: Path, source: dict[str, Any], manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schemaVersion": 1,
        "repoId": source["repoId"],
        "revision": source["revision"],
        "assetSetId": manifest["assetSetId"],
        "assetCount": manifest["assetCount"],
        "totalBytes": manifest["totalBytes"],
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_template_gallery(
    source: dict[str, Any],
    manifest: dict[str, Any],
    gallery_root: Path = TEMPLATE_GALLERY_ROOT,
    *,
    cache_root: Path | None = None,
    receipt_path: Path | None = None,
    download_snapshot: Callable[..., str] = snapshot_download,
) -> dict[str, Any]:
    cache = Path(cache_root or configured_template_gallery_cache_root())
    try:
        with app_hub_download_mode():
            snapshot = Path(
                download_snapshot(
                    repo_id=source["repoId"],
                    repo_type="dataset",
                    revision=source["revision"],
                    token=False,
                    cache_dir=str(cache),
                    allow_patterns=[record["path"] for record in manifest["assets"]],
                )
            )
    except Exception as error:
        raise TemplateGalleryError(
            "template_gallery_download_unavailable",
            "The app could not download the pinned Template Gallery snapshot.",
        ) from error
    parent = gallery_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    if _path_is_link_or_reparse(parent) or _path_is_link_or_reparse(gallery_root):
        raise TemplateGalleryError("template_gallery_install_unsafe", "Template Gallery destination is linked.")
    stage = Path(tempfile.mkdtemp(prefix=".template-gallery-stage-", dir=parent))
    backup: Path | None = None
    try:
        for record in manifest["assets"]:
            source_asset = snapshot.joinpath(*PurePosixPath(record["path"]).parts)
            destination_asset = _asset_destination(stage, record["path"])
            _copy_verified_asset(source_asset, destination_asset, record)
        verify_template_gallery_tree(stage, manifest)

        if gallery_root.exists():
            backup = parent / f".template-gallery-backup-{uuid.uuid4().hex}"
            os.replace(gallery_root, backup)
        try:
            os.replace(stage, gallery_root)
        except Exception:
            if backup is not None and backup.exists() and not gallery_root.exists():
                os.replace(backup, gallery_root)
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        if receipt_path is not None:
            _write_receipt(receipt_path, source, manifest)
        return {
            **verify_template_gallery_tree(gallery_root, manifest),
            "repoId": source["repoId"],
            "revision": source["revision"],
            "restartRequired": True,
        }
    except TemplateGalleryError:
        raise
    except Exception as error:
        raise TemplateGalleryError(
            "template_gallery_install_failed",
            "The app could not atomically install the verified Template Gallery snapshot.",
        ) from error
    finally:
        if stage.exists():
            shutil.rmtree(stage)
