"""Managed optional runtime optimizations.

Optional accelerator packages are deliberately kept outside MoDiff's active
environment.  A package is installed into a staged overlay, validated in a
fresh interpreter against the active Torch/Diffusers ABI, and only becomes
visible after an explicit activation and worker restart.  The previous overlay
remains addressable for rollback.

This module must remain importable with the Python standard library only:
``main.py`` activates the selected overlay before importing Torch or MoDiff's
runtime configuration.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlparse
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    optional_runtime_base_contracts,
    public_optional_runtime_profiles,
)
from modiff.runtime_overlays import (
    InstallLease,
    OverlayCancelled,
    OverlayInstallBusy,
    active_install,
    binding_digest,
    binding_matches,
    cache_locked_artifacts,
    current_base_binding,
    ensure_managed_directory,
    flush_managed_directory,
    normalize_locked_wheel_install,
    release_install,
    reserve_install,
    run_cancellable_command,
    run_fresh_validation,
    sanitized_install_environment,
    overlay_file_seal_matches,
    promote_staged_environment,
    remove_managed_directory,
    remove_managed_file,
    verify_artifact_anchored_overlay,
    locked_artifact_file_seal,
    managed_directory_identity,
)
from modiff.tool_locks import UV_TOOL_LOCKS


ROOT = Path(__file__).resolve().parents[1]
MANAGED_ROOT = Path(os.environ.get("MODIFF_MANAGED_ROOT") or ROOT / ".modiff")
OPTIMIZATION_ROOT = MANAGED_ROOT / "optimizations"
ENVIRONMENTS_DIR = OPTIMIZATION_ROOT / "environments"
STAGING_DIR = OPTIMIZATION_ROOT / "staging"
ARTIFACTS_DIR = OPTIMIZATION_ROOT / "artifacts"
STATE_PATH = OPTIMIZATION_ROOT / "state.json"
RECEIPTS_PATH = OPTIMIZATION_ROOT / "qualification-receipts.json"
PROMOTION_PATH = OPTIMIZATION_ROOT / "promotion.json"
CATALOG_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 2
RECEIPT_SCHEMA_VERSION = 1
_CATALOG_ENVIRONMENT_LIMIT = 32
_CATALOG_ENVIRONMENT_SCAN_LIMIT = 4096
_CATALOG_INACTIVE_DOCUMENT_LIMIT = 2 * 1024 * 1024
_CATALOG_PRIORITY_DOCUMENT_LIMIT = 8 * 1024 * 1024

_STATE_LOCK = threading.RLock()


class _ManagedJsonInvalid(RuntimeError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, value: dict[str, Any], *, root: Path) -> None:
    trusted_root_path = ensure_managed_directory(Path(root), managed_root=MANAGED_ROOT)
    trusted_root = trusted_root_path.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    parent.relative_to(trusted_root)
    parent_info = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or bool(
            getattr(parent_info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    ):
        raise OSError("The managed JSON parent is unsafe.")
    try:
        target_info = path.lstat()
        if (
            not stat.S_ISREG(target_info.st_mode)
            or stat.S_ISLNK(target_info.st_mode)
            or target_info.st_nlink != 1
            or bool(
                getattr(target_info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ):
            raise OSError("The managed JSON target is unsafe.")
    except FileNotFoundError:
        pass
    body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(body) > 32 * 1024 * 1024:
        raise OSError("The managed JSON document exceeds its safe size.")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        path.parent.resolve(strict=True).relative_to(trusted_root)
        temporary.replace(path)
        flush_managed_directory(path.parent, managed_root=MANAGED_ROOT)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(
    path: Path,
    fallback: dict[str, Any],
    *,
    root: Path | None = None,
    max_bytes: int = 32 * 1024 * 1024,
    reject_invalid_existing: bool = False,
) -> dict[str, Any]:
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(_value):
        raise ValueError("Non-finite JSON number")

    def invalid():
        if reject_invalid_existing:
            raise _ManagedJsonInvalid("The managed JSON document is invalid.")
        return deepcopy(fallback)

    try:
        raw_root = Path(root or path.parent).absolute()
        root_details = raw_root.lstat()
        if (
            not stat.S_ISDIR(root_details.st_mode)
            or stat.S_ISLNK(root_details.st_mode)
            or bool(
                getattr(root_details, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ):
            return invalid()
        try:
            raw_root.relative_to(Path(MANAGED_ROOT).absolute())
        except ValueError:
            trusted_root = raw_root.resolve(strict=True)
        else:
            trusted_root = _verified_existing_managed_directory(
                raw_root,
                managed_root=MANAGED_ROOT,
            )
        parent_details = path.parent.lstat()
        if (
            not stat.S_ISDIR(parent_details.st_mode)
            or stat.S_ISLNK(parent_details.st_mode)
            or bool(
                getattr(parent_details, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ):
            return invalid()
        parent = path.parent.resolve(strict=True)
        parent.relative_to(trusted_root)
        details = path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or details.st_nlink != 1
            or bool(
                getattr(details, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ):
            return invalid()
        resolved = path.resolve(strict=True)
        resolved.relative_to(trusted_root)
        if details.st_size > max_bytes:
            return invalid()
        raw = path.read_bytes()
        if len(raw) != details.st_size:
            return invalid()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
        pending = [(value, 0)]
        nodes = 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            if nodes > 200_000 or depth > 64:
                return invalid()
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
        return value if isinstance(value, dict) else invalid()
    except _ManagedJsonInvalid:
        raise
    except (OSError, TypeError, ValueError, UnicodeDecodeError, RecursionError) as exc:
        if reject_invalid_existing:
            try:
                path.lstat()
            except FileNotFoundError:
                return deepcopy(fallback)
            except OSError as probe_error:
                raise _ManagedJsonInvalid("The managed JSON document is inaccessible.") from probe_error
            raise _ManagedJsonInvalid("The managed JSON document is invalid.") from exc
        return deepcopy(fallback)


def _default_state() -> dict[str, Any]:
    return {
        "schemaVersion": STATE_SCHEMA_VERSION,
        "activeEnvironmentId": None,
        "previousEnvironmentId": None,
        "activeTrustClass": None,
        "previousTrustClass": None,
        "enabledCapabilities": [],
        "updatedAt": _now(),
    }


def read_state() -> dict[str, Any]:
    with _STATE_LOCK:
        try:
            raw = _read_json(
                STATE_PATH,
                _default_state(),
                root=OPTIMIZATION_ROOT,
                reject_invalid_existing=True,
            )
            storage_status = "ok"
        except _ManagedJsonInvalid:
            raw = _default_state()
            storage_status = "repair_required"

        def environment_id(value):
            return (
                value
                if isinstance(value, str)
                and re.fullmatch(r"runtime-[0-9]{1,16}-[0-9a-f]{8}", value)
                else None
            )

        def trust_class(value):
            return value if value in {"artifact_locked_optional", "legacy_optimization"} else None

        enabled = raw.get("enabledCapabilities")
        if not isinstance(enabled, list):
            storage_status = "repair_required"
            enabled = []
        normalized_enabled = [
            item
            for item in enabled
            if isinstance(item, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", item)
            and item in _catalog()
        ]
        active_environment_id = environment_id(raw.get("activeEnvironmentId"))
        previous_environment_id = environment_id(raw.get("previousEnvironmentId"))
        active_trust_class = trust_class(raw.get("activeTrustClass"))
        previous_trust_class = trust_class(raw.get("previousTrustClass"))
        updated_at = raw.get("updatedAt")
        expected_keys = {
            "schemaVersion",
            "activeEnvironmentId",
            "previousEnvironmentId",
            "activeTrustClass",
            "previousTrustClass",
            "enabledCapabilities",
            "updatedAt",
        }
        if (
            set(raw) != expected_keys
            or raw.get("schemaVersion") != STATE_SCHEMA_VERSION
            or (
                raw.get("activeEnvironmentId") is not None
                and active_environment_id is None
            )
            or (
                raw.get("previousEnvironmentId") is not None
                and previous_environment_id is None
            )
            or (raw.get("activeTrustClass") is not None and active_trust_class is None)
            or (
                raw.get("previousTrustClass") is not None
                and previous_trust_class is None
            )
            or (active_environment_id is None) != (active_trust_class is None)
            or (previous_environment_id is None) != (previous_trust_class is None)
            or (
                active_environment_id is not None
                and active_environment_id == previous_environment_id
            )
            or len(enabled) > 64
            or len(normalized_enabled) != len(enabled)
            or len(set(normalized_enabled)) != len(normalized_enabled)
            or _public_utc_timestamp(updated_at) is None
        ):
            storage_status = "repair_required"
        return {
            "schemaVersion": STATE_SCHEMA_VERSION,
            "activeEnvironmentId": active_environment_id,
            "previousEnvironmentId": previous_environment_id,
            "activeTrustClass": active_trust_class,
            "previousTrustClass": previous_trust_class,
            "enabledCapabilities": normalized_enabled[:64],
            "updatedAt": updated_at if isinstance(updated_at, str) else _now(),
            "_storageStatus": storage_status,
        }


def _write_state(state: dict[str, Any]) -> dict[str, Any]:
    with _STATE_LOCK:
        state = deepcopy(state)
        state.pop("_storageStatus", None)
        state["schemaVersion"] = STATE_SCHEMA_VERSION
        state["updatedAt"] = _now()
        _atomic_json(STATE_PATH, state, root=OPTIMIZATION_ROOT)
        return state


def _reset_state_to_base() -> dict[str, Any]:
    """Repair a corrupt state file without ever following an unsafe entry."""

    with _STATE_LOCK:
        trusted_root = _verified_existing_managed_directory(
            OPTIMIZATION_ROOT,
            managed_root=MANAGED_ROOT,
        )
        if STATE_PATH.absolute().parent != OPTIMIZATION_ROOT.absolute():
            raise OSError("The runtime state path is outside its managed root.")
        state_path = trusted_root / STATE_PATH.name
        try:
            details = state_path.lstat()
        except FileNotFoundError:
            details = None
        if details is not None:
            reparse = bool(
                getattr(details, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            if stat.S_ISDIR(details.st_mode):
                raise OSError("The runtime state path is an unsafe directory.")
            if stat.S_ISLNK(details.st_mode) or reparse or (
                stat.S_ISREG(details.st_mode) and details.st_nlink != 1
            ):
                state_path.unlink()
            elif not stat.S_ISREG(details.st_mode):
                raise OSError("The runtime state path is not a regular file.")
        return _write_state(_default_state())


def _canonical_digest(value: dict[str, Any]) -> str:
    body = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


def _promotion_anchor(inspection: dict[str, Any]) -> dict[str, str]:
    if inspection.get("status") != "ready":
        raise RuntimeError("An optional-runtime promotion candidate is not fully validated.")
    manifest = inspection.get("manifest")
    validation = inspection.get("validation")
    if not isinstance(manifest, dict) or not isinstance(validation, dict):
        raise RuntimeError("An optional-runtime promotion candidate lacks its validation records.")
    return {
        "manifestDigest": _canonical_digest(manifest),
        "validationDigest": _canonical_digest(validation),
    }


def _read_promotion_record() -> dict[str, Any] | None:
    _verified_existing_managed_directory(
        OPTIMIZATION_ROOT,
        managed_root=MANAGED_ROOT,
    )
    if PROMOTION_PATH.absolute().parent != OPTIMIZATION_ROOT.absolute():
        raise OSError("The optional-runtime promotion journal is outside its managed root.")
    try:
        record = _read_json(
            PROMOTION_PATH,
            {},
            root=OPTIMIZATION_ROOT,
            max_bytes=16 * 1024,
            reject_invalid_existing=True,
        )
    except _ManagedJsonInvalid as exc:
        raise RuntimeError("The optional-runtime promotion journal requires repair.") from exc
    if not record:
        return None
    expected_keys = {
        "schemaVersion",
        "environmentId",
        "phase",
        "manifestDigest",
        "validationDigest",
        "updatedAt",
    }
    digest = r"sha256:[0-9a-f]{64}"
    if (
        set(record) != expected_keys
        or record.get("schemaVersion") != 1
        or not isinstance(record.get("environmentId"), str)
        or not re.fullmatch(
            r"runtime-[0-9]{1,16}-[0-9a-f]{8}",
            record["environmentId"],
        )
        or record.get("phase") not in {"prepared", "promoted"}
        or not isinstance(record.get("manifestDigest"), str)
        or not re.fullmatch(digest, record["manifestDigest"])
        or not isinstance(record.get("validationDigest"), str)
        or not re.fullmatch(digest, record["validationDigest"])
        or _public_utc_timestamp(record.get("updatedAt")) is None
    ):
        raise RuntimeError("The optional-runtime promotion journal requires repair.")
    return record


def _write_promotion_record(
    environment_id: str,
    *,
    phase: str,
    anchor: dict[str, str],
) -> dict[str, Any]:
    record = {
        "schemaVersion": 1,
        "environmentId": environment_id,
        "phase": phase,
        "manifestDigest": anchor["manifestDigest"],
        "validationDigest": anchor["validationDigest"],
        "updatedAt": _now(),
    }
    _atomic_json(PROMOTION_PATH, record, root=OPTIMIZATION_ROOT)
    return record


def _promotion_matches(record: dict[str, Any], inspection: dict[str, Any]) -> bool:
    try:
        anchor = _promotion_anchor(inspection)
    except RuntimeError:
        return False
    return all(anchor[key] == record[key] for key in ("manifestDigest", "validationDigest"))


def _clear_promotion_record() -> None:
    remove_managed_file(
        PROMOTION_PATH,
        parent=OPTIMIZATION_ROOT,
        managed_root=MANAGED_ROOT,
    )


def _reconcile_promotion(lease: InstallLease) -> str | None:
    """Complete or acknowledge one exact interrupted promotion under the install lease."""

    record = _read_promotion_record()
    if record is None:
        return None
    environment_id = record["environmentId"]
    staged = STAGING_DIR / environment_id
    destination = ENVIRONMENTS_DIR / environment_id

    def entry_exists(path: Path) -> bool:
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        return True

    staged_exists = entry_exists(staged)
    destination_exists = entry_exists(destination)
    if staged_exists == destination_exists:
        raise RuntimeError("The optional-runtime promotion journal has an ambiguous filesystem state.")
    if destination_exists:
        inspection = _environment_inspection(
            environment_id,
            environment_root=_verified_existing_managed_directory(
                ENVIRONMENTS_DIR,
                managed_root=MANAGED_ROOT,
            ),
        )
        if not _promotion_matches(record, inspection):
            raise RuntimeError("The promoted optional runtime no longer matches its durable journal.")
        if record["phase"] == "prepared":
            _write_promotion_record(environment_id, phase="promoted", anchor=record)
        _clear_promotion_record()
        return environment_id
    if record["phase"] != "prepared":
        raise RuntimeError("The optional-runtime promotion journal is missing its promoted environment.")
    inspection = _environment_inspection(
        environment_id,
        environment_root=_verified_existing_managed_directory(
            STAGING_DIR,
            managed_root=MANAGED_ROOT,
        ),
    )
    if not _promotion_matches(record, inspection):
        raise RuntimeError("The staged optional runtime no longer matches its durable journal.")
    lease.staged_identity = managed_directory_identity(staged)
    promote_staged_environment(lease, staged, destination)
    promoted = _environment_inspection(
        environment_id,
        environment_root=_verified_existing_managed_directory(
            ENVIRONMENTS_DIR,
            managed_root=MANAGED_ROOT,
        ),
    )
    if not _promotion_matches(record, promoted):
        raise RuntimeError("The recovered optional runtime failed its post-promotion identity check.")
    _write_promotion_record(environment_id, phase="promoted", anchor=record)
    _clear_promotion_record()
    return environment_id


def _optimization_spec(capability_id: str) -> dict[str, Any]:
    capability = _catalog().get(capability_id)
    if capability is None:
        raise ValueError(f"Unknown optimization capability {capability_id!r}.")
    base_packages = [
        {"distribution": "packaging", "importName": "packaging", "specifier": ">=20.0"},
        {"distribution": "numpy", "importName": "numpy", "specifier": ">=1.17"},
        {"distribution": "torch", "importName": "torch", "specifier": ">=2.6.0"},
        {
            "distribution": "huggingface-hub",
            "importName": "huggingface_hub",
            "specifier": ">=1.23.0,<2.0",
        },
    ]
    spec = {
        "schemaVersion": 1,
        "kind": "optimization",
        "id": capability_id,
        "basePackages": base_packages,
        **deepcopy(capability),
    }
    return {"kind": "optimization", "id": capability_id, "spec": spec, "specDigest": _canonical_digest(spec)}


def _optional_runtime_spec(profile_id: str) -> dict[str, Any]:
    try:
        profile = OPTIONAL_RUNTIME_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown optional runtime profile {profile_id!r}.") from exc
    spec = profile.to_spec_dict()
    return {
        "kind": "optional_runtime",
        "id": profile_id,
        "spec": spec,
        "specDigest": profile.spec_digest,
    }


def _spec_is_current(record: dict[str, Any]) -> bool:
    try:
        kind = str(record.get("kind") or "")
        identifier = str(record.get("id") or "")
        current = (
            _optional_runtime_spec(identifier)
            if kind == "optional_runtime"
            else _optimization_spec(identifier)
            if kind == "optimization"
            else None
        )
        return bool(
            current
            and record.get("specDigest") == current["specDigest"]
            and record.get("spec") == current["spec"]
            and (
                kind != "optional_runtime"
                or OPTIONAL_RUNTIME_PROFILES[identifier].contract_for_target().activation_available
                is True
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _optimization_package_contracts(capability_id: str) -> list[dict[str, Any]]:
    capability = _catalog().get(capability_id) or {}
    requirements = [
        str(value)
        for value in [*(capability.get("buildPackages") or []), *(capability.get("packages") or [])]
    ]
    contracts = []
    seen = set()
    primary = str(capability.get("distribution") or "").lower().replace("_", "-")
    for requirement in requirements:
        if "==" not in requirement or requirement.count("==") != 1:
            raise RuntimeError("Optimization package requirements must be exact reviewed pins.")
        distribution, version = requirement.split("==", 1)
        normalized = distribution.lower().replace("_", "-")
        if not normalized or not version or normalized in seen:
            continue
        seen.add(normalized)
        contracts.append(
            {
                "distribution": normalized,
                "importName": (
                    str(capability.get("importName"))
                    if normalized == primary and capability.get("importName")
                    else normalized.replace("-", "_")
                ),
                "requiredVersion": version,
                "requirement": f"{distribution}=={version}",
                "requiredSymbols": [],
                "role": "optimization_dependency" if normalized != primary else "optimization_root",
            }
        )
    return contracts


def _expected_contracts_for_specs(
    specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    package_contracts: list[dict[str, Any]] = []
    base_contracts: list[dict[str, Any]] = []
    for record in specs:
        if not _spec_is_current(record):
            raise RuntimeError("An overlay executable spec is stale or unavailable.")
        if record["kind"] == "optional_runtime":
            profile = OPTIONAL_RUNTIME_PROFILES[record["id"]]
            package_contracts = _merge_contracts(
                package_contracts,
                [package.to_spec_dict() for package in profile.packages],
            )
            base_contracts = _merge_contracts(
                base_contracts,
                list(optional_runtime_base_contracts([profile.id])),
            )
        else:
            package_contracts = _merge_contracts(
                package_contracts,
                _optimization_package_contracts(record["id"]),
            )
            base_contracts = _merge_contracts(
                base_contracts,
                list(record["spec"].get("basePackages") or []),
            )
    return package_contracts, base_contracts


def _verified_existing_managed_directory(path: Path, *, managed_root: Path) -> Path:
    """Resolve an existing managed directory without following reparse components."""

    raw_root = Path(managed_root).absolute()
    root_details = raw_root.lstat()
    if (
        not stat.S_ISDIR(root_details.st_mode)
        or stat.S_ISLNK(root_details.st_mode)
        or bool(
            getattr(root_details, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    ):
        raise OSError("The managed runtime root is unsafe.")
    trusted_root = raw_root.resolve(strict=True)
    raw_path = Path(path).absolute()
    relative = raw_path.relative_to(raw_root)
    current = raw_root
    for component in relative.parts:
        current = current / component
        details = current.lstat()
        if (
            not stat.S_ISDIR(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or bool(
                getattr(details, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ):
            raise OSError("A managed runtime directory component is unsafe.")
        current.resolve(strict=True).relative_to(trusted_root)
    return current.resolve(strict=True)


def _environment_inspection(
    environment_id: str | None,
    *,
    verify_integrity: bool = True,
    document_max_bytes: int = 32 * 1024 * 1024,
    environment_root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(environment_id, str) or not environment_id or len(environment_id) > 128:
        return {"status": "repair_required", "reason": "invalid_environment_id"}
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in environment_id):
        return {"status": "repair_required", "reason": "invalid_environment_id"}
    try:
        trusted_environments = environment_root or _verified_existing_managed_directory(
            ENVIRONMENTS_DIR,
            managed_root=MANAGED_ROOT,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return {"status": "repair_required", "reason": "environment_root_unsafe"}
    raw_candidate = trusted_environments / environment_id
    try:
        raw_info = raw_candidate.lstat()
    except OSError:
        return {"status": "repair_required", "reason": "environment_missing"}
    if not stat.S_ISDIR(raw_info.st_mode) or stat.S_ISLNK(raw_info.st_mode) or (
        getattr(raw_info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        return {"status": "repair_required", "reason": "environment_link_or_type"}
    candidate = raw_candidate.resolve()
    try:
        candidate.relative_to(trusted_environments)
    except ValueError:
        return {"status": "repair_required", "reason": "environment_escape"}
    manifest = _read_json(
        candidate / "manifest.json",
        {},
        root=candidate,
        max_bytes=document_max_bytes,
    )
    validation = _read_json(
        candidate / "validation.json",
        {},
        root=candidate,
        max_bytes=document_max_bytes,
    )
    site_packages = candidate / "site-packages"
    manifest_specs = manifest.get("specs")
    if (
        not isinstance(manifest_specs, list)
        or not manifest_specs
        or len(manifest_specs) > 32
        or not all(isinstance(record, dict) for record in manifest_specs)
    ):
        return {"status": "repair_required", "reason": "stale_executable_spec"}
    try:
        expected_packages, expected_base = _expected_contracts_for_specs(manifest_specs)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return {"status": "repair_required", "reason": "stale_executable_spec"}
    expected_spec_ids = [
        {"kind": item["kind"], "id": item["id"], "specDigest": item["specDigest"]}
        for item in manifest_specs
    ]
    spec_kinds = {str(item.get("kind") or "") for item in manifest_specs}
    expected_trust_class = (
        "artifact_locked_optional"
        if spec_kinds == {"optional_runtime"}
        else "legacy_optimization"
        if spec_kinds == {"optimization"}
        else None
    )
    detail = validation.get("detail") if isinstance(validation.get("detail"), dict) else {}
    observed_packages = sorted(
        (str(item.get("distribution")), str(item.get("version")))
        for item in detail.get("packages") or []
        if isinstance(item, dict)
    )
    exact_packages = sorted(
        (str(item.get("distribution")), str(item.get("requiredVersion")))
        for item in expected_packages
    )
    if (
        manifest.get("schemaVersion") != 2
        or manifest.get("kind") != "runtime_overlay"
        or manifest.get("id") != environment_id
        or expected_trust_class is None
        or manifest.get("trustClass") != expected_trust_class
        or validation.get("status") != "passed"
        or validation.get("schemaVersion") != 2
        or validation.get("environmentId") != environment_id
        or validation.get("specs") != expected_spec_ids
        or detail.get("status") != "passed"
        or observed_packages != exact_packages
        or not all(_spec_is_current(record) for record in manifest_specs)
        or manifest.get("packageContracts") != expected_packages
        or manifest.get("basePackageContracts") != expected_base
        or not isinstance(validation.get("binding"), dict)
        or validation.get("bindingDigest") != binding_digest(validation.get("binding"))
        or not isinstance(validation.get("fileSeal"), dict)
        or not validation.get("fileSeal")
    ):
        return {"status": "repair_required", "reason": "missing_or_stale_validation"}
    expected_distributions = [
        str(package.get("distribution") or "") for package in manifest["packageContracts"]
    ]
    expected_artifacts: list[dict[str, Any]] = []
    if expected_trust_class == "artifact_locked_optional":
        try:
            for record in manifest_specs:
                expected_artifacts.extend(
                    _artifact_install_plan(OPTIONAL_RUNTIME_PROFILES[record["id"]])
                )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return {"status": "repair_required", "reason": "artifact_lock_unavailable"}
        if manifest.get("artifactLocks") != expected_artifacts:
            return {"status": "repair_required", "reason": "artifact_lock_drift"}
        if (
            validation.get("trustClass") != expected_trust_class
            or not isinstance(manifest.get("artifactAnchorDigest"), str)
            or validation.get("artifactAnchorDigest") != manifest.get("artifactAnchorDigest")
        ):
            return {"status": "repair_required", "reason": "artifact_anchor_missing"}
    elif manifest.get("artifactLocks") not in ([], None) or validation.get("artifactAnchorDigest") is not None:
        return {"status": "repair_required", "reason": "legacy_artifact_claim"}
    if not verify_integrity:
        return {
            "status": "recorded",
            "sitePackages": site_packages,
            "manifest": manifest,
            "validation": validation,
        }
    if not binding_matches(validation["binding"], manifest["basePackageContracts"]):
        return {"status": "repair_required", "reason": "host_binding_drift"}
    if expected_trust_class == "artifact_locked_optional":
        try:
            artifact_anchor = verify_artifact_anchored_overlay(
                site_packages,
                expected_artifacts,
                ARTIFACTS_DIR,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"status": "repair_required", "reason": "artifact_anchor_drift"}
        if (
            artifact_anchor.get("digest") != manifest.get("artifactAnchorDigest")
            or artifact_anchor.get("fileSeal") != validation.get("fileSeal")
        ):
            return {"status": "repair_required", "reason": "artifact_anchor_drift"}
    elif not overlay_file_seal_matches(
        site_packages, expected_distributions, validation["fileSeal"]
    ):
        return {"status": "repair_required", "reason": "overlay_file_drift"}
    return {
        "status": "ready",
        "sitePackages": site_packages,
        "manifest": manifest,
        "validation": validation,
    }


def _fresh_validation_matches(inspection: dict[str, Any], lease: InstallLease) -> bool:
    manifest = inspection["manifest"]
    binding = current_base_binding(manifest["basePackageContracts"])
    trusted_file_seal = None
    if manifest.get("trustClass") == "artifact_locked_optional":
        anchor = verify_artifact_anchored_overlay(
            inspection["sitePackages"],
            manifest["artifactLocks"],
            ARTIFACTS_DIR,
            lease=lease,
        )
        if anchor.get("digest") != manifest.get("artifactAnchorDigest"):
            return False
        trusted_file_seal = anchor["fileSeal"]
    validation = run_fresh_validation(
        inspection["sitePackages"],
        packages=manifest["packageContracts"],
        binding=binding,
        specs=manifest["specs"],
        lease=lease,
        trusted_file_seal=trusted_file_seal,
    )
    persisted = inspection["validation"]
    return bool(
        validation.get("status") == "passed"
        and validation.get("bindingDigest") == persisted.get("bindingDigest")
        and validation.get("fileSeal") == persisted.get("fileSeal")
        and validation.get("detail") == persisted.get("detail")
    )


def _safe_environment_path(
    environment_id: str | None,
    lease: InstallLease,
    *,
    expected_trust_class: str,
) -> Path | None:
    inspection = _environment_inspection(environment_id)
    if (
        inspection.get("status") != "ready"
        or inspection.get("manifest", {}).get("trustClass") != expected_trust_class
    ):
        return None
    return inspection["sitePackages"] if _fresh_validation_matches(inspection, lease) else None


def activate_runtime_overlay() -> str | None:
    """Add the validated active overlay before importing heavyweight modules."""
    deadline = time.monotonic() + 5.0
    while True:
        try:
            lease = reserve_install("startup_activation", "active_environment")
            break
        except OverlayInstallBusy:
            if time.monotonic() >= deadline:
                os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "busy_recovery_only"
                return None
            time.sleep(0.05)
        except (OSError, RuntimeError, ValueError):
            os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "repair_required"
            return None
    try:
        _reconcile_promotion(lease)
        state = read_state()
        if state.get("_storageStatus") != "ok":
            os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "repair_required"
            return None
        environment_id = state.get("activeEnvironmentId")
        if environment_id is None:
            os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "base"
            return None
        try:
            active_trust_class = state.get("activeTrustClass")
            if active_trust_class != "artifact_locked_optional":
                site_packages = None
            else:
                site_packages = _safe_environment_path(
                    environment_id,
                    lease,
                    expected_trust_class=active_trust_class,
                )
        except (OSError, RuntimeError, TypeError, ValueError):
            site_packages = None
        if site_packages is None:
            os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "repair_required"
            return None
        # Never use site.addsitedir: wheel-supplied .pth files can execute
        # code. Hold the cross-process lease through the final plain-path
        # insertion so state cannot race validation.
        sys.dont_write_bytecode = True
        normalized = str(site_packages)
        if normalized in sys.path:
            sys.path.remove(normalized)
        sys.path.insert(0, normalized)
        os.environ["MODIFF_OPTIMIZATION_ENVIRONMENT"] = str(environment_id)
        os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] = "active"
        return str(environment_id)
    finally:
        release_install(lease)


def _catalog() -> dict[str, dict[str, Any]]:
    # Versions are reviewed pins, not an online "latest" resolver. Upgrading a
    # pin requires updating the official-source review and qualification tests.
    return {
        "hub_attention_kernels": {
            "label": "Hub attention kernels",
            "kind": "package",
            "distribution": "kernels",
            "importName": "kernels",
            "packages": ["kernels==0.16.0"],
            "installMode": "binary",
            # The regular dependency set contains no Torch package. It is safe
            # to resolve the signed-kernel metadata helpers into the overlay.
            "includeDependencies": True,
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux"],
            "automaticEligible": True,
            "summary": "Prebuilt attention kernels fetched through the Hugging Face kernels runtime.",
            "documentation": "https://huggingface.co/docs/kernels/main/installation",
        },
        "flash_attention_2": {
            "label": "FlashAttention 2",
            "kind": "package",
            "distribution": "flash-attn",
            "importName": "flash_attn",
            "buildPackages": ["ninja==1.13.0"],
            "packages": ["flash-attn==2.8.3.post1"],
            "installMode": "source",
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux"],
            "automaticEligible": True,
            "summary": "Source-built FlashAttention 2 for qualified CUDA or ROCm hardware.",
            "documentation": "https://github.com/Dao-AILab/flash-attention",
        },
        "torchao": {
            "label": "TorchAO quantization",
            "kind": "package",
            "distribution": "torchao",
            "importName": "torchao",
            "packages": ["torchao==0.17.0"],
            "installMode": "binary",
            "profiles": ["nvidia-cuda", "amd-rocm-linux", "cpu"],
            "platforms": ["linux", "windows", "macos"],
            "automaticEligible": True,
            "summary": "Torch-native weight-only, integer, and floating-point quantization recipes.",
            "documentation": "https://docs.pytorch.org/ao/stable/workflows/inference.html",
        },
        "optimum_quanto": {
            "label": "Optimum Quanto",
            "kind": "package",
            "distribution": "optimum-quanto",
            "importName": "optimum.quanto",
            "packages": ["ninja==1.13.0", "optimum-quanto==0.2.7"],
            "installMode": "binary",
            "profiles": ["nvidia-cuda", "amd-rocm-linux", "cpu"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Quanto float8 and integer weight quantization.",
            "documentation": "https://huggingface.co/docs/diffusers/main/api/quantization",
        },
        "bitsandbytes": {
            "label": "bitsandbytes quantization",
            "kind": "package",
            "distribution": "bitsandbytes",
            "importName": "bitsandbytes",
            "packages": ["bitsandbytes==0.50.0"],
            "installMode": "binary",
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Qualified 4-bit and 8-bit linear-layer quantization.",
            "documentation": "https://huggingface.co/docs/bitsandbytes/stable/en/installation",
        },
        "sage_attention": {
            "label": "SageAttention",
            "kind": "package",
            "distribution": "sageattention",
            "importName": "sageattention",
            "packages": ["sageattention==1.0.6"],
            "installMode": "source",
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux"],
            "automaticEligible": False,
            "summary": "Experimental quantized attention kernels; manual opt-in and workload qualification required.",
            "documentation": "https://github.com/thu-ml/SageAttention",
        },
        "xformers": {
            "label": "xFormers",
            "kind": "profile",
            "distribution": "xformers",
            "importName": "xformers",
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Profile-managed xFormers build matched to the installed PyTorch release.",
            "documentation": "https://github.com/facebookresearch/xformers",
        },
        "aiter": {
            "label": "AMD AITER",
            "kind": "external",
            "distribution": "amd-aiter",
            "importName": "aiter",
            "profiles": ["amd-rocm-linux"],
            "platforms": ["linux"],
            "automaticEligible": False,
            "summary": "AMD datacenter-kernel package; only qualified Instinct/ABI combinations are supported.",
            "documentation": "https://github.com/ROCm/aiter",
        },
        "regional_compile": {
            "label": "Regional torch.compile",
            "kind": "runtime",
            "profiles": ["nvidia-cuda", "amd-rocm-linux", "cpu"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Compile repeated model blocks instead of recompiling a whole pipeline.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/fp16",
        },
        "denoiser_cache": {
            "label": "Denoiser cache",
            "kind": "runtime",
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Model-specific block, residual, or timestep caching with explicit output review.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/cache",
        },
        "layerwise_casting": {
            "label": "Layerwise casting",
            "kind": "runtime",
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Store compatible weights at lower precision and cast them only for computation.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/memory",
        },
        "channels_last": {
            "label": "Channels-last memory format",
            "kind": "runtime",
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Opt-in convolution layout optimization for qualified UNet/VAE workloads.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/fp16",
        },
        "quantization_offload": {
            "label": "Quantization with offload",
            "kind": "runtime",
            "implemented": False,
            "profiles": ["nvidia-cuda", "amd-rocm-linux"],
            "platforms": ["linux", "windows"],
            "automaticEligible": True,
            "summary": "Researched, but not selectable until each quantizer/offload ordering has a qualified runtime contract.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/memory",
        },
        "context_parallel": {
            "label": "Multi-GPU context parallelism",
            "kind": "runtime",
            "implemented": False,
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux"],
            "automaticEligible": False,
            "summary": "Experimental multi-GPU sharding for large transformer/video workloads.",
            "documentation": "https://huggingface.co/docs/diffusers/main/training/distributed_inference",
        },
        "fused_qkv": {
            "label": "Fused QKV projections",
            "kind": "runtime",
            "implemented": False,
            "profiles": ["nvidia-cuda"],
            "platforms": ["linux"],
            "automaticEligible": False,
            "summary": "Experimental model-specific projection fusion.",
            "documentation": "https://huggingface.co/docs/diffusers/main/optimization/fp16",
        },
    }


def _normalized_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _profile_id(runtime_profile: dict[str, Any] | None) -> str:
    profile = runtime_profile if isinstance(runtime_profile, dict) else {}
    return str(
        profile.get("installed") or profile.get("installed_profile") or profile.get("installedProfile") or "unverified"
    )


def _catalog_environment_ids(
    state: dict[str, Any], environment_root: Path
) -> tuple[list[str], dict[str, Any]]:
    """Return a bounded view that never drops the active or rollback target."""

    priority: list[str] = []
    for key in ("activeEnvironmentId", "previousEnvironmentId"):
        value = state.get(key)
        if isinstance(value, str) and value not in priority:
            priority.append(value)

    discovered: list[str] = []
    observed = 0
    scan_truncated = False
    try:
        with os.scandir(environment_root) as entries:
            for entry in entries:
                observed += 1
                if observed > _CATALOG_ENVIRONMENT_SCAN_LIMIT:
                    scan_truncated = True
                    break
                name = str(entry.name)
                if (
                    name in priority
                    or len(name) > 128
                    or not name
                    or any(
                        character
                        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                        for character in name
                    )
                ):
                    continue
                discovered.append(name)
    except OSError:
        discovered = []

    remaining = max(0, _CATALOG_ENVIRONMENT_LIMIT - len(priority))
    selected = [*priority, *sorted(set(discovered))[:remaining]]
    omitted_by_limit = len(set(discovered)) > remaining
    return selected, {
        "returned": len(selected),
        "observed": min(observed, _CATALOG_ENVIRONMENT_SCAN_LIMIT),
        "limit": _CATALOG_ENVIRONMENT_LIMIT,
        "scanLimit": _CATALOG_ENVIRONMENT_SCAN_LIMIT,
        "truncated": bool(scan_truncated or omitted_by_limit),
    }


def _public_environment_id(value: Any) -> str | None:
    return (
        value
        if isinstance(value, str)
        and re.fullmatch(r"runtime-[0-9]{1,16}-[0-9a-f]{8}", value)
        else None
    )


def _public_utc_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value
    ):
        return None
    try:
        parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return value if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) == value else None


def _public_sha256_digest(value: Any) -> str | None:
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
        else None
    )


def public_catalog(
    *,
    runtime_profile: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = read_state()
    profile_id = _profile_id(runtime_profile)
    os_name = _normalized_platform()
    hardware = hardware if isinstance(hardware, dict) else {}
    torch_state = hardware.get("torch") if isinstance(hardware.get("torch"), dict) else {}
    torch_version = str(torch_state.get("version") or "")
    devices = hardware.get("devices") if isinstance(hardware.get("devices"), list) else []
    accelerators = hardware.get("accelerators") if isinstance(hardware.get("accelerators"), list) else []
    device_count = len(devices or accelerators)
    enabled = {str(item) for item in state.get("enabledCapabilities") or []}
    capabilities = []
    for capability_id, raw in _catalog().items():
        item = {"id": capability_id, **deepcopy(raw)}
        implemented = item.get("implemented", True) is True
        compatible = profile_id in item.get("profiles", []) and os_name in item.get("platforms", [])
        reason = None
        if profile_id not in item.get("profiles", []):
            reason = f"Requires one of: {', '.join(item.get('profiles') or [])}."
        elif os_name not in item.get("platforms", []):
            reason = f"Not packaged for {os_name}."
        if capability_id == "flash_attention_2":
            if profile_id == "nvidia-cuda" and torch_version and not torch_version.startswith("2.8"):
                compatible = False
                reason = "The reviewed FlashAttention build is qualified only with MoDiff's PyTorch 2.8 CUDA profile."
            elif profile_id == "amd-rocm-linux":
                architectures = {
                    str(value)
                    for value in (hardware.get("amd_architectures") or hardware.get("amdArchitectures") or [])
                }
                supported = {
                    "gfx90a",
                    "gfx940",
                    "gfx941",
                    "gfx942",
                    "gfx950",
                    "gfx1100",
                    "gfx1101",
                    "gfx1200",
                    "gfx1201",
                    "gfx1151",
                }
                if architectures and not architectures.intersection(supported):
                    compatible = False
                    reason = "The detected AMD architecture is outside FlashAttention's reviewed ROCm set."
        if capability_id == "context_parallel" and device_count < 2:
            compatible = False
            reason = "Requires at least two compatible accelerators."
        if not implemented:
            compatible = False
            reason = "The upstream feature is documented, but MoDiff has not qualified a safe execution contract yet."
        installed_version = _package_version(str(item.get("distribution"))) if item.get("distribution") else None
        installed = bool(installed_version) if item.get("distribution") else implemented
        can_enable = compatible and (
            item.get("kind") == "runtime" or (item.get("kind") in {"package", "profile", "external"} and installed)
        )
        item.update(
            {
                "implemented": implemented,
                "compatible": compatible,
                "disabledReason": reason,
                "enabled": capability_id in enabled,
                "installed": installed,
                "installedVersion": installed_version,
                "canInstall": item.get("kind") == "package" and compatible,
                "canEnable": can_enable,
                "requiresRestart": item.get("kind") == "package",
            }
        )
        if item.get("kind") == "package":
            item["canInstall"] = False
            item["canEnable"] = False
            item["disabledReason"] = (
                "This legacy package profile has no reviewed immutable artifact lock and remains unqualified."
            )
        if item.get("kind") == "external":
            item["disabledReason"] = (
                reason or "No generally safe app-managed wheel matches every supported ROCm device and Torch ABI."
            )
        capabilities.append(item)
    environments = []
    environment_scan = {
        "returned": 0,
        "observed": 0,
        "limit": _CATALOG_ENVIRONMENT_LIMIT,
        "scanLimit": _CATALOG_ENVIRONMENT_SCAN_LIMIT,
        "truncated": False,
    }
    try:
        catalog_environment_root = _verified_existing_managed_directory(
            ENVIRONMENTS_DIR,
            managed_root=MANAGED_ROOT,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        catalog_environment_root = None
    if catalog_environment_root is not None:
        environment_ids, environment_scan = _catalog_environment_ids(
            state, catalog_environment_root
        )
        priority_ids = {
            state.get("activeEnvironmentId"),
            state.get("previousEnvironmentId"),
        }
        for environment_id in environment_ids:
            inspection = _environment_inspection(
                environment_id,
                verify_integrity=False,
                document_max_bytes=(
                    _CATALOG_PRIORITY_DOCUMENT_LIMIT
                    if environment_id in priority_ids
                    else _CATALOG_INACTIVE_DOCUMENT_LIMIT
                ),
                environment_root=catalog_environment_root,
            )
            manifest = inspection.get("manifest") if inspection.get("status") in {"ready", "recorded"} else {}
            validation = inspection.get("validation") if inspection.get("status") in {"ready", "recorded"} else {}
            public_environment_status = (
                "legacy_unqualified"
                if (manifest or {}).get("trustClass") == "legacy_optimization"
                else inspection.get("status")
            )
            specs = (manifest or {}).get("specs")
            capabilities_from_specs = [
                str(item.get("id"))
                for item in specs or []
                if isinstance(item, dict)
                and item.get("kind") == "optimization"
                and str(item.get("id")) in _catalog()
            ]
            environments.append(
                {
                    "id": _public_environment_id(environment_id),
                    "createdAt": _public_utc_timestamp((manifest or {}).get("createdAt")),
                    "capabilities": [
                        str(item)[:128]
                        for item in capabilities_from_specs
                        if str(item) in _catalog()
                    ][:32],
                    "validation": {
                        "status": (
                            validation.get("status")
                            if validation.get("status") in {"passed", "failed"}
                            else None
                        ),
                        "validatedAt": _public_utc_timestamp(validation.get("validatedAt")),
                        "bindingDigest": _public_sha256_digest(validation.get("bindingDigest")),
                    },
                    "status": public_environment_status,
                    "active": environment_id == state.get("activeEnvironmentId"),
                    "activationAvailable": False,
                }
            )
    receipts = read_receipts().get("receipts") or []
    public_state = {
        "schemaVersion": STATE_SCHEMA_VERSION,
        "activeEnvironmentId": (
            _public_environment_id(state.get("activeEnvironmentId"))
        ),
        "previousEnvironmentId": (
            _public_environment_id(state.get("previousEnvironmentId"))
        ),
        "enabledCapabilities": sorted(
            item for item in enabled if item in _catalog()
        ),
    }
    process_load_status = os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS")
    if state.get("_storageStatus") != "ok":
        process_load_status = "repair_required"
    elif process_load_status not in {
        "base",
        "active",
        "busy_recovery_only",
        "repair_required",
        "restart_required",
    }:
        process_load_status = "base"
    return {
        "schemaVersion": CATALOG_SCHEMA_VERSION,
        "processLoadStatus": process_load_status,
        "state": public_state,
        "profile": profile_id,
        "platform": os_name,
        "capabilities": capabilities,
        "environments": environments,
        "environmentScan": environment_scan,
        "qualification": {
            "receiptCount": len(receipts),
            "qualifiedCount": sum(1 for item in receipts if item.get("status") == "qualified"),
            "observedCount": sum(1 for item in receipts if item.get("status") == "observed"),
        },
    }


def _verified_uv_executable() -> str:
    tool_root = _verified_existing_managed_directory(
        MANAGED_ROOT / "tools" / "uv",
        managed_root=MANAGED_ROOT,
    )
    receipt = _read_json(tool_root / "receipt.json", {}, root=tool_root)
    machine = _machine_name()
    reviewed = UV_TOOL_LOCKS.get((_platform_name(), machine))
    if not reviewed:
        raise RuntimeError(
            "MoDiff has no reviewed immutable uv executable lock for this platform."
        )
    expected_archive = reviewed.get("archiveSha256")
    expected_executable = reviewed.get("executableSha256")
    relative = receipt.get("executable")
    if (
        receipt.get("schemaVersion") != 1
        or receipt.get("archiveSha256") != expected_archive
        or not isinstance(relative, str)
        or not relative
        or len(relative) > 256
    ):
        raise RuntimeError("MoDiff's managed uv executable has no verified receipt.")
    executable = (tool_root / relative).resolve(strict=True)
    try:
        executable.relative_to(tool_root)
    except ValueError as exc:
        raise RuntimeError("The managed uv receipt escapes its tool directory.") from exc
    if not executable.is_file() or executable.is_symlink():
        raise RuntimeError("The managed uv executable is unavailable or linked.")
    hasher = hashlib.sha256()
    with executable.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    if (
        not isinstance(expected_executable, str)
        or len(expected_executable) != 64
        or hasher.hexdigest() != expected_executable
        or receipt.get("executableSha256") != expected_executable
    ):
        raise RuntimeError("The managed uv executable failed its integrity check.")
    return str(executable)


def _platform_name() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _machine_name() -> str:
    machine = platform.machine().strip().lower().replace("-", "_")
    return {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)


def _artifact_install_plan(profile) -> list[dict[str, Any]]:
    """Resolve one complete, immutable wheel set for this Python/platform."""

    platform_name = _platform_name()
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = _machine_name()
    expected = {package.distribution: package.version for package in profile.packages}
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename

    supported_tags = set(sys_tags())
    selected: dict[str, dict[str, Any]] = {}
    for artifact in profile.artifact_locks:
        if not isinstance(artifact, dict):
            continue
        artifact_machine = str(artifact.get("machine") or "").strip().lower().replace("-", "_")
        if (
            artifact.get("platform") != platform_name
            or artifact.get("pythonTag") != python_tag
            or artifact_machine not in {machine, "any"}
        ):
            continue
        distribution = str(artifact.get("distribution") or "").lower().replace("_", "-")
        version = str(artifact.get("version") or "")
        filename = str(artifact.get("filename") or "")
        url = str(artifact.get("url") or "")
        digest = str(artifact.get("sha256") or "").lower()
        byte_size = artifact.get("byteSize")
        parsed = urlparse(url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise RuntimeError("The optional-runtime artifact URL has an invalid port.") from exc
        try:
            wheel_name, wheel_version, _build, wheel_tags = parse_wheel_filename(filename)
        except ValueError as exc:
            raise RuntimeError("The optional-runtime artifact filename is not a valid wheel.") from exc
        if not set(wheel_tags).intersection(supported_tags):
            continue
        if (
            distribution not in expected
            or version != expected[distribution]
            or distribution in selected
            or not filename.endswith(".whl")
            or parsed.scheme != "https"
            or parsed.hostname != "files.pythonhosted.org"
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
            or Path(parsed.path).name != filename
            or canonicalize_name(wheel_name) != canonicalize_name(distribution)
            or str(wheel_version) != version
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
            or byte_size <= 0
            or byte_size > 512 * 1024**2
        ):
            raise RuntimeError("The optional-runtime artifact lock is invalid for this platform.")
        selected[distribution] = {
            "distribution": distribution,
            "version": version,
            "filename": filename,
            "url": url,
            "sha256": digest,
            "byteSize": byte_size,
            "platform": platform_name,
            "pythonTag": python_tag,
            "machine": artifact_machine,
        }
    if set(selected) != set(expected):
        raise RuntimeError("The optional-runtime artifact lock is incomplete for this Python and platform.")
    return [selected[package.distribution] for package in profile.packages]


def _artifact_install_urls(profile) -> list[str]:
    """Compatibility projection used by contract tests and diagnostics."""

    return [f"{item['url']}#sha256={item['sha256']}" for item in _artifact_install_plan(profile)]


def validate_optional_runtime_install_request(
    profile_id: str,
    spec_digest: str,
    *,
    consent: Any,
) -> dict[str, Any]:
    """Fail closed before a lease, staging directory, or subprocess exists."""

    if not isinstance(profile_id, str) or not profile_id or len(profile_id) > 256:
        raise ValueError("A bounded optional runtime profileId is required.")
    if not isinstance(spec_digest, str) or len(spec_digest) != 71:
        raise ValueError("An exact optional runtime specDigest is required.")
    if consent is not True:
        raise ValueError("Explicit consent=true is required to install an optional runtime.")
    spec = _optional_runtime_spec(profile_id)
    profile = OPTIONAL_RUNTIME_PROFILES[profile_id]
    if spec_digest != spec["specDigest"]:
        raise ValueError("The optional runtime specDigest does not match the reviewed catalog.")
    if profile.contract_for_target().install_action_available is not True:
        raise RuntimeError("This optional runtime is not qualified for installation.")
    base_contracts = list(optional_runtime_base_contracts([profile_id]))
    # Establish constraints, exact observed versions/origins, accelerator lock,
    # and Diffusers identity before a lease or staging directory can exist.
    current_base_binding(base_contracts)
    artifacts = _artifact_install_plan(profile)
    installer = _verified_uv_executable()
    return {
        "profile": profile,
        "spec": spec,
        "artifactLocks": artifacts,
        "baseContracts": base_contracts,
        "installerExecutable": installer,
    }


def validate_optional_runtime_activation_request(
    profile_id: str,
    spec_digest: str,
    *,
    consent: Any,
) -> dict[str, Any]:
    if consent is not True:
        raise ValueError("Explicit consent=true is required to activate an optional runtime.")
    spec = _optional_runtime_spec(profile_id)
    profile = OPTIONAL_RUNTIME_PROFILES[profile_id]
    if spec_digest != spec["specDigest"]:
        raise ValueError("The optional runtime specDigest does not match the reviewed catalog.")
    if profile.contract_for_target().activation_available is not True:
        raise RuntimeError("This optional runtime is not qualified for activation.")
    return spec


def _merge_contracts(existing: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {str(item.get("distribution") or ""): deepcopy(item) for item in existing}
    for contract in additions:
        distribution = str(contract.get("distribution") or "")
        previous = merged.get(distribution)
        if previous is None:
            merged[distribution] = deepcopy(contract)
            continue
        if previous == contract:
            continue
        # Base-owned requirements from two reviewed specs may legitimately
        # narrow the same distribution. Preserve one import identity and form
        # the deterministic intersection of both specifier sets. Staged wheel
        # contracts remain exact and may never be merged this way.
        if (
            "requirement" not in previous
            and "requirement" not in contract
            and previous.get("importName") == contract.get("importName")
            and set(previous) <= {"distribution", "importName", "specifier", "platforms"}
            and set(contract) <= {"distribution", "importName", "specifier", "platforms"}
            and previous.get("platforms") == contract.get("platforms")
        ):
            constraints = {
                item.strip()
                for value in (previous.get("specifier"), contract.get("specifier"))
                for item in str(value or "").split(",")
                if item.strip()
            }
            combined = deepcopy(previous)
            combined["specifier"] = ",".join(sorted(constraints))
            merged[distribution] = combined
            continue
        raise RuntimeError(f"Overlay package contract conflict for {distribution!r}.")
    return [merged[name] for name in sorted(merged)]


def _locked_requirements_body(
    artifacts: list[dict[str, Any]],
    cached_artifacts: list[Path],
) -> bytes:
    if len(artifacts) != len(cached_artifacts) or not artifacts:
        raise RuntimeError("A complete locked requirements set is required.")
    lines = [
        f"{artifact['distribution']} @ {path.resolve(strict=True).as_uri()} --hash=sha256:{artifact['sha256']}"
        for artifact, path in zip(artifacts, cached_artifacts, strict=True)
    ]
    body = ("\n".join(lines) + "\n").encode("utf-8")
    if len(body) > 64 * 1024:
        raise RuntimeError("The locked optional-runtime requirements document is oversized.")
    return body


def _install_reviewed_overlay(
    *,
    spec: dict[str, Any],
    package_contracts: list[dict[str, Any]],
    base_contracts: list[dict[str, Any]],
    install_urls: list[str],
    lease: InstallLease,
    progress: Callable[[dict[str, Any]], None] | None,
    hash_locked: bool = True,
    installer_executable: str | None = None,
    artifact_locks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def report(phase: str, message: str) -> None:
        if progress:
            progress({"phase": phase, "message": message, "updatedAt": _now()})

    recovered_environment = _reconcile_promotion(lease)
    if recovered_environment is not None:
        raise RuntimeError(
            "An interrupted optional-runtime promotion was recovered. Retry the requested operation."
        )
    state = read_state()
    if state.get("_storageStatus") != "ok":
        raise RuntimeError("Repair or reset the corrupt runtime state before installing an overlay.")
    if not installer_executable:
        raise RuntimeError("A source-controlled verified installer executable is required.")
    environment_id = f"runtime-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    staged = STAGING_DIR / environment_id
    destination = ENVIRONMENTS_DIR / environment_id
    site_packages = staged / "site-packages"
    try:
        ensure_managed_directory(OPTIMIZATION_ROOT, managed_root=MANAGED_ROOT)
        ensure_managed_directory(STAGING_DIR, managed_root=MANAGED_ROOT)
        ensure_managed_directory(ENVIRONMENTS_DIR, managed_root=MANAGED_ROOT)
        staged.mkdir(parents=False, exist_ok=False)
        staged_info = staged.lstat()
        if not stat.S_ISDIR(staged_info.st_mode) or stat.S_ISLNK(staged_info.st_mode):
            raise RuntimeError("The optional-runtime staging directory is unsafe.")
        staged.resolve(strict=True).relative_to(STAGING_DIR.resolve(strict=True))
        lease.staged_identity = managed_directory_identity(staged)
        specs: list[dict[str, Any]] = []
        existing_packages: list[dict[str, Any]] = []
        existing_base: list[dict[str, Any]] = []
        active = _environment_inspection(state.get("activeEnvironmentId"))
        requested_trust_class = "artifact_locked_optional" if hash_locked else "legacy_optimization"
        if (
            active.get("status") == "ready"
            and not hash_locked
            and active.get("manifest", {}).get("trustClass") == requested_trust_class
            and _fresh_validation_matches(active, lease)
        ):
            report("copying", "Copying the last validated optional environment.")
            shutil.copytree(active["sitePackages"], site_packages)
            active_manifest = active["manifest"]
            specs = deepcopy(active_manifest.get("specs") or [])
            existing_packages = deepcopy(active_manifest.get("packageContracts") or [])
            existing_base = deepcopy(active_manifest.get("basePackageContracts") or [])
        else:
            site_packages.mkdir()
        specs = [
            item
            for item in specs
            if not (item.get("kind") == spec["kind"] and item.get("id") == spec["id"])
        ]
        specs.append(deepcopy(spec))
        packages = _merge_contracts(existing_packages, package_contracts)
        base_packages = _merge_contracts(existing_base, base_contracts)
        binding = current_base_binding(base_packages)
        selected_artifacts = [dict(item) for item in (artifact_locks or [])]
        if hash_locked:
            if spec.get("kind") != "optional_runtime" or not selected_artifacts:
                raise RuntimeError("A locked optional runtime requires a complete artifact plan.")
            # Optional and hashless legacy overlays are deliberately separate
            # trust classes. Never carry legacy bytes into an authenticated
            # optional environment.
            if any(item.get("kind") != "optional_runtime" for item in specs):
                raise RuntimeError("Hashless legacy packages cannot be mixed into an optional runtime.")
            ensure_managed_directory(ARTIFACTS_DIR, managed_root=MANAGED_ROOT)
            cached_artifacts = cache_locked_artifacts(
                selected_artifacts,
                ARTIFACTS_DIR,
                lease=lease,
            )
            # Authenticate and structurally inspect every wheel before giving
            # any archive to the installer/extractor.
            locked_artifact_file_seal(
                selected_artifacts,
                ARTIFACTS_DIR,
                lease=lease,
            )
            requirements_body = _locked_requirements_body(
                selected_artifacts,
                cached_artifacts,
            )
            requirements_path = staged / "locked-requirements.txt"
            with requirements_path.open("xb") as output:
                output.write(requirements_body)
                output.flush()
                os.fsync(output.fileno())
            flush_managed_directory(staged, managed_root=MANAGED_ROOT)
            effective_install_urls = []
        else:
            if any(item.get("kind") != "optimization" for item in specs):
                raise RuntimeError("Optional-runtime packages cannot be mixed into a legacy overlay.")
            effective_install_urls = list(install_urls)
        manifest = {
            "schemaVersion": 2,
            "kind": "runtime_overlay",
            "id": environment_id,
            "trustClass": requested_trust_class,
            "createdAt": _now(),
            "specs": specs,
            "packageContracts": packages,
            "basePackageContracts": base_packages,
            "requestedPackages": [package["requirement"] for package in package_contracts],
            "artifactLocks": selected_artifacts,
        }
        _atomic_json(staged / "manifest.json", manifest, root=staged)
        report("installing", "Installing the reviewed wheel set into a staged environment.")
        command = [
            installer_executable,
            "--no-config",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(site_packages),
            "--upgrade",
            "--no-deps",
            "--link-mode",
            "copy",
        ]
        if hash_locked:
            command.extend(
                [
                    "--no-index",
                    "--require-hashes",
                    "--only-binary",
                    ":all:",
                    "--requirement",
                    str(requirements_path),
                ]
            )
        command.extend(effective_install_urls)
        try:
            install_result = run_cancellable_command(
                command,
                environment=sanitized_install_environment(site_packages),
                lease=lease,
                timeout=1800,
                cwd=staged,
            )
        finally:
            if hash_locked:
                remove_managed_file(
                    requirements_path,
                    parent=staged,
                    managed_root=MANAGED_ROOT,
                )
        if lease.cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime installation was cancelled.")
        _atomic_json(
            staged / "install.json",
            {
                "returnCode": install_result["returnCode"],
                "elapsedSeconds": install_result["elapsedSeconds"],
                "commandDigest": _canonical_digest({"command": command}),
            },
            root=staged,
        )
        if install_result["returnCode"] != 0:
            raise RuntimeError("The reviewed optional-runtime wheel set could not be staged.")
        artifact_anchor = None
        if hash_locked:
            normalize_locked_wheel_install(
                site_packages,
                selected_artifacts,
                ARTIFACTS_DIR,
                lease=lease,
            )
            artifact_anchor = verify_artifact_anchored_overlay(
                site_packages,
                selected_artifacts,
                ARTIFACTS_DIR,
                lease=lease,
            )
            manifest["artifactAnchorDigest"] = artifact_anchor["digest"]
            _atomic_json(staged / "manifest.json", manifest, root=staged)
        report("validating", "Validating exact versions, origins, symbols, and host binding.")
        validation = run_fresh_validation(
            site_packages,
            packages=packages,
            binding=binding,
            specs=specs,
            lease=lease,
            trusted_file_seal=(artifact_anchor or {}).get("fileSeal"),
        )
        validation["schemaVersion"] = 2
        validation["environmentId"] = environment_id
        validation["specs"] = [
            {"kind": item["kind"], "id": item["id"], "specDigest": item["specDigest"]}
            for item in specs
        ]
        validation["validatedAt"] = _now()
        validation["trustClass"] = requested_trust_class
        validation["artifactAnchorDigest"] = (artifact_anchor or {}).get("digest")
        _atomic_json(staged / "validation.json", validation, root=staged)
        if lease.cancel_event.is_set():
            raise OverlayCancelled("Optional-runtime installation was cancelled.")
        if validation.get("status") != "passed":
            raise RuntimeError("The staged optional runtime failed isolated validation.")
        if binding != current_base_binding(base_packages) or not all(_spec_is_current(item) for item in specs):
            raise RuntimeError("The host or executable runtime spec changed before promotion.")
        staged_inspection = _environment_inspection(
            environment_id,
            environment_root=_verified_existing_managed_directory(
                STAGING_DIR,
                managed_root=MANAGED_ROOT,
            ),
        )
        promotion_anchor = _promotion_anchor(staged_inspection)
        _write_promotion_record(
            environment_id,
            phase="prepared",
            anchor=promotion_anchor,
        )
        promote_staged_environment(lease, staged, destination)
        promoted_inspection = _environment_inspection(
            environment_id,
            environment_root=_verified_existing_managed_directory(
                ENVIRONMENTS_DIR,
                managed_root=MANAGED_ROOT,
            ),
        )
        if _promotion_anchor(promoted_inspection) != promotion_anchor:
            raise RuntimeError("The optional runtime changed identity during promotion.")
        _write_promotion_record(
            environment_id,
            phase="promoted",
            anchor=promotion_anchor,
        )
        _clear_promotion_record()
        report("ready", "Validation passed. Explicit activation and restart are still required.")
        return {
            "environmentId": environment_id,
            "specs": [{"kind": item["kind"], "id": item["id"], "specDigest": item["specDigest"]} for item in specs],
            "validation": {
                "status": "passed",
                "bindingDigest": validation["bindingDigest"],
                "validatedAt": validation["validatedAt"],
            },
            "requiresActivation": True,
            "activeRuntimeChanged": False,
        }
    finally:
        try:
            if not lease.committed:
                remove_managed_directory(
                    staged,
                    parent=STAGING_DIR,
                    expected_identity=lease.staged_identity,
                )
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            pass
        finally:
            release_install(lease)


def install_optional_runtime(
    profile_id: str,
    spec_digest: str,
    *,
    consent: Any,
    lease: InstallLease | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    request = validate_optional_runtime_install_request(profile_id, spec_digest, consent=consent)
    owned_lease = lease or reserve_install("optional_runtime", profile_id)
    profile = request["profile"]
    return _install_reviewed_overlay(
        spec=request["spec"],
        package_contracts=[package.to_spec_dict() for package in profile.packages],
        base_contracts=request["baseContracts"],
        install_urls=[],
        lease=owned_lease,
        progress=progress,
        installer_executable=request["installerExecutable"],
        artifact_locks=request["artifactLocks"],
    )


def public_optional_runtime_catalog() -> dict[str, Any]:
    state = read_state()
    process_load_status = os.environ.get("MODIFF_RUNTIME_OVERLAY_STATUS", "base")
    if state.get("_storageStatus") != "ok":
        process_load_status = "repair_required"
    if process_load_status not in {
        "base",
        "active",
        "busy_recovery_only",
        "repair_required",
        "restart_required",
    }:
        process_load_status = "repair_required"
    active_id = state.get("activeEnvironmentId")
    public_active_id = _public_environment_id(active_id)
    environments = []
    inspections: dict[str, dict[str, Any]] = {}
    environment_scan = {
        "returned": 0,
        "observed": 0,
        "limit": _CATALOG_ENVIRONMENT_LIMIT,
        "scanLimit": _CATALOG_ENVIRONMENT_SCAN_LIMIT,
        "truncated": False,
    }
    try:
        catalog_environment_root = _verified_existing_managed_directory(
            ENVIRONMENTS_DIR,
            managed_root=MANAGED_ROOT,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        catalog_environment_root = None
    if catalog_environment_root is not None:
        environment_ids, environment_scan = _catalog_environment_ids(
            state, catalog_environment_root
        )
        priority_ids = {active_id, state.get("previousEnvironmentId")}
        for environment_id in environment_ids:
            public_environment_id = _public_environment_id(environment_id)
            inspection = _environment_inspection(
                environment_id,
                verify_integrity=False,
                document_max_bytes=(
                    _CATALOG_PRIORITY_DOCUMENT_LIMIT
                    if environment_id in priority_ids
                    else _CATALOG_INACTIVE_DOCUMENT_LIMIT
                ),
                environment_root=catalog_environment_root,
            )
            inspections[environment_id] = inspection
            manifest = inspection.get("manifest") if inspection.get("status") in {"ready", "recorded"} else {}
            trust_class = (manifest or {}).get("trustClass")
            environments.append(
                {
                    "id": public_environment_id,
                    "createdAt": _public_utc_timestamp((manifest or {}).get("createdAt")),
                    "specs": [
                        {
                            "kind": item.get("kind"),
                            "id": item.get("id"),
                            "specDigest": item.get("specDigest"),
                        }
                        for item in (manifest or {}).get("specs") or []
                        if isinstance(item, dict)
                    ][:32],
                    "status": (
                        "legacy_unqualified"
                        if trust_class == "legacy_optimization"
                        else "staged_unchecked"
                        if inspection.get("status") == "recorded"
                        else inspection.get("status")
                    ),
                    "active": environment_id == active_id,
                }
            )
    profiles = public_optional_runtime_profiles()
    for profile in profiles:
        matching = [
            environment
            for environment in environments
            if any(
                spec.get("kind") == "optional_runtime"
                and spec.get("id") == profile["id"]
                and spec.get("specDigest") == profile["specDigest"]
                for spec in environment["specs"]
            )
        ]
        profile["overlayStatus"] = (
            "active"
            if process_load_status == "active"
            and any(
                environment["active"]
                and environment["status"] in {"ready", "staged_unchecked"}
                for environment in matching
            )
            else "repair_required"
            if process_load_status in {"busy_recovery_only", "repair_required", "restart_required"}
            and any(environment["active"] for environment in matching)
            else "staged"
            if any(environment["status"] == "ready" for environment in matching)
            else "staged_unchecked"
            if any(environment["status"] == "staged_unchecked" for environment in matching)
            else "repair_required"
            if matching
            else "missing"
        )
    active_inspection = inspections.get(active_id) if active_id else None
    if active_id and active_inspection is None:
        active_inspection = _environment_inspection(active_id, verify_integrity=False)
    recorded_active_status = active_inspection.get("status") if active_inspection else "missing"
    active_status = (
        "active"
        if process_load_status == "active" and recorded_active_status in {"ready", "recorded"}
        else "repair_required"
        if process_load_status in {"busy_recovery_only", "repair_required"}
        else "restart_required"
        if process_load_status == "restart_required"
        else "staged_unchecked"
        if recorded_active_status == "recorded"
        else recorded_active_status
    )
    active_job = active_install()
    return {
        "schemaVersion": 1,
        "profiles": profiles,
        "overlay": {
            "processLoadStatus": process_load_status,
            "state": {
                "activeEnvironmentId": public_active_id,
                "previousEnvironmentId": (
                    _public_environment_id(state.get("previousEnvironmentId"))
                ),
                "activeStatus": active_status,
            },
            "environments": environments,
            "environmentScan": environment_scan,
        },
        "activeInstallJob": (
            {
                "ownerKind": (
                    active_job.get("ownerKind")
                    if active_job.get("ownerKind") in {"optional_runtime", "optimization"}
                    else None
                ),
                "ownerId": (
                    active_job.get("ownerId")
                    if isinstance(active_job.get("ownerId"), str)
                    and re.fullmatch(
                        r"[a-z0-9][a-z0-9_.-]{0,127}", active_job.get("ownerId")
                    )
                    else None
                ),
            }
            if active_job
            else None
        ),
    }


def activate_optional_runtime_environment(
    environment_id: str,
    profile_id: str,
    spec_digest: str,
    *,
    consent: Any,
) -> dict[str, Any]:
    required = validate_optional_runtime_activation_request(profile_id, spec_digest, consent=consent)
    return _activate_environment_transaction(
        environment_id,
        expected_trust_class="artifact_locked_optional",
        required_spec={
            "kind": "optional_runtime",
            "id": profile_id,
            "specDigest": required["specDigest"],
        },
    )


def rollback_optional_runtime_environment(*, consent: Any) -> dict[str, Any]:
    if consent is not True:
        raise ValueError("Explicit consent=true is required to roll back an optional runtime.")
    return _rollback_environment_transaction(expected_trust_class="artifact_locked_optional")


def install_capability(
    capability_id: str,
    *,
    runtime_profile: dict[str, Any] | None,
    hardware: dict[str, Any] | None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    lease: InstallLease | None = None,
) -> dict[str, Any]:
    del runtime_profile, hardware, progress, lease
    catalog = _catalog()
    capability = catalog.get(capability_id)
    if capability is None or capability.get("kind") != "package":
        raise ValueError(f"{capability_id!r} is not an app-installable optimization package.")
    raise RuntimeError(
        "Legacy optimization package installation remains unavailable until its complete immutable artifact lock is reviewed."
    )


def _activate_environment_transaction(
    environment_id: str,
    *,
    expected_trust_class: str,
    required_spec: dict[str, str] | None = None,
) -> dict[str, Any]:
    if expected_trust_class == "legacy_optimization":
        raise RuntimeError(
            "Legacy optimization overlays remain unqualified and cannot be activated without immutable artifact locks."
        )
    lease = reserve_install("activation", str(environment_id))
    try:
        _reconcile_promotion(lease)
        state = read_state()
        if state.get("_storageStatus") != "ok":
            raise RuntimeError("Repair or reset the corrupt runtime state before activation.")
        inspection = _environment_inspection(environment_id)
        if (
            inspection.get("status") != "ready"
            or inspection.get("manifest", {}).get("trustClass") != expected_trust_class
        ):
            raise ValueError("Only a currently validated matching environment can be activated.")
        if required_spec is not None and not any(
            item.get("kind") == required_spec["kind"]
            and item.get("id") == required_spec["id"]
            and item.get("specDigest") == required_spec["specDigest"]
            for item in inspection["manifest"].get("specs") or []
        ):
            raise ValueError("The environment does not contain the requested optional runtime spec.")
        if not _fresh_validation_matches(inspection, lease):
            raise ValueError("Only a currently validated staged environment can be activated.")
        if state.get("activeEnvironmentId") == environment_id:
            if state.get("activeTrustClass") != expected_trust_class:
                raise RuntimeError("The active environment trust-class receipt is stale.")
            return {"state": state, "restartRequired": False}
        current = state.get("activeEnvironmentId")
        if current is not None:
            current_inspection = _environment_inspection(current)
            if (
                state.get("activeTrustClass") != expected_trust_class
                or current_inspection.get("manifest", {}).get("trustClass")
                != expected_trust_class
            ):
                raise RuntimeError(
                    "Roll back the current runtime trust class before activating another class."
                )
        state["previousEnvironmentId"] = state.get("activeEnvironmentId")
        state["previousTrustClass"] = state.get("activeTrustClass")
        state["activeEnvironmentId"] = environment_id
        state["activeTrustClass"] = expected_trust_class
        state = _write_state(state)
        return {"state": state, "restartRequired": True}
    finally:
        release_install(lease)


def activate_environment(environment_id: str) -> dict[str, Any]:
    """Compatibility activation for hashless legacy optimization overlays."""

    return _activate_environment_transaction(
        environment_id,
        expected_trust_class="legacy_optimization",
    )


def _rollback_environment_transaction(*, expected_trust_class: str) -> dict[str, Any]:
    lease = reserve_install("rollback", "previous_environment")
    try:
        _reconcile_promotion(lease)
        state = read_state()
        if state.get("_storageStatus") != "ok":
            state = _reset_state_to_base()
            return {"state": state, "restartRequired": True}
        current = state.get("activeEnvironmentId")
        if expected_trust_class == "legacy_optimization":
            if current is not None and state.get("activeTrustClass") != expected_trust_class:
                raise RuntimeError(
                    "This rollback route does not own the active environment trust class."
                )
            state = _write_state(_default_state())
            return {"state": state, "restartRequired": current is not None}
        if current is not None and state.get("activeTrustClass") != expected_trust_class:
            raise RuntimeError("This rollback route does not own the active environment trust class.")
        previous = state.get("previousEnvironmentId")
        if previous is not None:
            if state.get("previousTrustClass") != expected_trust_class:
                raise RuntimeError("The previous environment belongs to another trust class.")
            inspection = _environment_inspection(previous)
            if (
                inspection.get("status") != "ready"
                or inspection.get("manifest", {}).get("trustClass") != expected_trust_class
                or not _fresh_validation_matches(inspection, lease)
            ):
                raise RuntimeError("The previous optional environment requires repair before rollback.")
        state["activeEnvironmentId"] = previous
        state["activeTrustClass"] = state.get("previousTrustClass")
        state["previousEnvironmentId"] = current
        state["previousTrustClass"] = expected_trust_class if current is not None else None
        state = _write_state(state)
        return {"state": state, "restartRequired": current != previous}
    finally:
        release_install(lease)


def rollback_environment() -> dict[str, Any]:
    """Compatibility rollback limited to hashless legacy optimization overlays."""

    return _rollback_environment_transaction(expected_trust_class="legacy_optimization")


def set_capability_enabled(capability_id: str, enabled: bool) -> dict[str, Any]:
    if capability_id not in _catalog():
        raise ValueError(f"Unknown optimization capability {capability_id!r}.")
    state = read_state()
    if state.get("_storageStatus") != "ok":
        raise RuntimeError("Repair or reset the corrupt runtime state before changing capabilities.")
    values = {str(item) for item in state.get("enabledCapabilities") or []}
    if enabled:
        values.add(capability_id)
    else:
        values.discard(capability_id)
    state["enabledCapabilities"] = sorted(values)
    return _write_state(state)


def read_receipts() -> dict[str, Any]:
    value = _read_json(
        RECEIPTS_PATH,
        {"schemaVersion": RECEIPT_SCHEMA_VERSION, "receipts": [], "updatedAt": _now()},
        root=OPTIMIZATION_ROOT,
    )
    receipts = value.get("receipts")
    if value.get("schemaVersion") != RECEIPT_SCHEMA_VERSION or not isinstance(receipts, list):
        return {
            "schemaVersion": RECEIPT_SCHEMA_VERSION,
            "receipts": [],
            "updatedAt": _now(),
        }
    return {
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "receipts": [item for item in receipts if isinstance(item, dict)][:500],
        "updatedAt": _public_utc_timestamp(value.get("updatedAt")) or _now(),
    }


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def workload_key_for_form(form: dict[str, Any] | None) -> str:
    """Hash result-affecting workload fields while excluding runtime tuning."""
    value = form if isinstance(form, dict) else {}
    excluded = {
        "resourceMode",
        "resourcePreference",
        "dtype",
        "quantizationMode",
        "quantizedComponents",
        "device",
        "deviceMap",
        "autoOffload",
        "offloadMode",
        "attentionBackend",
        "attention_backend",
        "regionalCompile",
        "regional_compile",
        "denoiserCache",
        "denoiser_cache",
        "channelsLast",
        "channels_last",
        "layerwiseCasting",
        "layerwise_casting",
    }
    return f"workload-{_stable_hash({key: value[key] for key in sorted(value) if key not in excluded})}"


def optimization_selections_from_graph(graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract explicit non-default runtime selections from an API graph."""
    found: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            attention = value.get("attention_backend")
            if isinstance(attention, str) and attention not in {"", "auto", "native"}:
                capability = {
                    "flash": "flash_attention_2",
                    "flash_hub": "hub_attention_kernels",
                    "flash_varlen": "flash_attention_2",
                    "flash_varlen_hub": "hub_attention_kernels",
                    "flash_4_hub": "hub_attention_kernels",
                    "_flash_3": "flash_attention_2",
                    "_flash_varlen_3": "flash_attention_2",
                    "_flash_3_hub": "hub_attention_kernels",
                    "_flash_3_varlen_hub": "hub_attention_kernels",
                    "sage": "sage_attention",
                    "sage_hub": "hub_attention_kernels",
                    "xformers": "xformers",
                    "aiter": "aiter",
                }.get(attention)
                if capability:
                    found[capability] = {
                        "capabilityId": capability,
                        "attentionBackend": attention,
                    }
            if value.get("regional_compile") is True:
                found["regional_compile"] = {
                    "capabilityId": "regional_compile",
                    "regionalCompile": True,
                }
            if value.get("layerwise_casting") is True:
                found["layerwise_casting"] = {
                    "capabilityId": "layerwise_casting",
                    "layerwiseCasting": True,
                }
            if value.get("channels_last") is True:
                found["channels_last"] = {
                    "capabilityId": "channels_last",
                    "channelsLast": True,
                }
            cache = value.get("denoiser_cache")
            if isinstance(cache, str) and cache not in {"", "none"}:
                found[f"denoiser_cache:{cache}"] = {
                    "capabilityId": "denoiser_cache",
                    "denoiserCache": cache,
                }
            quantization = value.get("quantization_mode") or value.get("backend")
            if isinstance(quantization, str) and quantization not in {"", "none"}:
                capability = (
                    "torchao"
                    if quantization.startswith("torchao")
                    else "optimum_quanto"
                    if quantization.startswith("quanto")
                    else "bitsandbytes"
                    if quantization.startswith("bnb")
                    else None
                )
                if capability:
                    found[f"{capability}:{quantization}"] = {
                        "capabilityId": capability,
                        "quantizationMode": quantization,
                    }
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(graph)
    return list(found.values())


def record_probe_receipt(
    *,
    capability_id: str,
    runtime_fingerprint: Any,
    result: dict[str, Any],
    environment_id: str | None = None,
) -> dict[str, Any]:
    if capability_id not in _catalog():
        raise ValueError(f"Unknown optimization capability {capability_id!r}.")
    detail = result.get("detail") if isinstance(result.get("detail"), dict) else {}
    sanitized_result: dict[str, Any] = {
        "status": "passed" if result.get("status") == "passed" else "failed",
        "diagnosticDigest": _stable_hash(result),
    }
    for key in ("supported", "compileAvailable", "cudaAvailable"):
        if isinstance(detail.get(key), bool):
            sanitized_result[key] = detail[key]
    device_count = detail.get("deviceCount")
    if isinstance(device_count, int) and not isinstance(device_count, bool) and 0 <= device_count <= 1024:
        sanitized_result["deviceCount"] = device_count
    return_code = result.get("returnCode")
    if isinstance(return_code, int) and not isinstance(return_code, bool) and -255 <= return_code <= 255:
        sanitized_result["returnCode"] = return_code
    elapsed = result.get("elapsedSeconds")
    if (
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(elapsed)
        and 0 <= elapsed <= 3600
    ):
        sanitized_result["elapsedSeconds"] = float(elapsed)
    receipt = {
        "id": f"probe-{uuid.uuid4().hex}",
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "kind": "compatibility_probe",
        "status": "probe_passed" if result.get("status") == "passed" else "probe_failed",
        "capabilityId": capability_id,
        "environmentId": environment_id or read_state().get("activeEnvironmentId"),
        "runtimeFingerprintHash": _stable_hash(runtime_fingerprint),
        "result": sanitized_result,
        "createdAt": _now(),
        # Import/synthetic probes never authorize Auto for a model workload.
        "autoEligible": False,
    }
    with _STATE_LOCK:
        document = read_receipts()
        document["receipts"] = [receipt, *document.get("receipts", [])][:500]
        document["updatedAt"] = _now()
        _atomic_json(RECEIPTS_PATH, document, root=OPTIMIZATION_ROOT)
    return receipt


def probe_capability(capability_id: str, *, runtime_fingerprint: Any) -> dict[str, Any]:
    capability = _catalog().get(capability_id)
    if capability is None:
        raise ValueError(f"Unknown optimization capability {capability_id!r}.")
    if capability.get("importName"):
        raise RuntimeError(
            "Legacy optimization package probes remain unavailable until immutable artifact locks are reviewed."
        )
    else:
        script = """
import json
import torch
from diffusers.hooks import FirstBlockCacheConfig, apply_layerwise_casting
capability = __CAPABILITY__
checks = {
    "regional_compile": callable(getattr(torch, "compile", None)),
    "denoiser_cache": FirstBlockCacheConfig is not None,
    "layerwise_casting": callable(apply_layerwise_casting) and hasattr(torch, "float8_e4m3fn"),
    "channels_last": hasattr(torch, "channels_last"),
}
supported = checks.get(capability, False)
print(json.dumps({
    "torch": str(torch.__version__),
    "compileAvailable": callable(getattr(torch, "compile", None)),
    "cudaAvailable": bool(torch.cuda.is_available()),
    "deviceCount": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "capability": capability,
    "supported": supported,
}))
if not supported:
    raise SystemExit(2)
""".replace("__CAPABILITY__", repr(capability_id))
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=os.environ.copy(),
        )
        detail = None
        try:
            detail = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else None
        except ValueError:
            detail = {"stdout": result.stdout.strip()[-4000:]}
        validation = {
            "status": "passed" if result.returncode == 0 else "failed",
            "returnCode": result.returncode,
            "detail": detail,
            "stderr": result.stderr.strip()[-4000:],
            "elapsedSeconds": time.monotonic() - started,
            "validatedAt": _now(),
        }
    return record_probe_receipt(
        capability_id=capability_id,
        runtime_fingerprint=runtime_fingerprint,
        result=validation,
        environment_id=read_state().get("activeEnvironmentId"),
    )


def record_workload_observation(
    *,
    capability_id: str,
    runtime_fingerprint: Any,
    model_type: str,
    mode: str,
    artifact: str,
    workload_key: str,
    selection: dict[str, Any],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    if capability_id not in _catalog():
        raise ValueError(f"Unknown optimization capability {capability_id!r}.")
    receipt = {
        "id": f"workload-{uuid.uuid4().hex}",
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "kind": "workload",
        "status": "observed",
        "capabilityId": capability_id,
        "environmentId": read_state().get("activeEnvironmentId"),
        "runtimeFingerprintHash": _stable_hash(runtime_fingerprint),
        "modelType": str(model_type),
        "mode": str(mode),
        "artifact": str(artifact),
        "workloadKey": str(workload_key),
        "selection": deepcopy(selection),
        "measurement": deepcopy(measurement),
        "createdAt": _now(),
        "autoEligible": False,
    }
    with _STATE_LOCK:
        document = read_receipts()
        document["receipts"] = [receipt, *document.get("receipts", [])][:500]
        document["updatedAt"] = _now()
        _atomic_json(RECEIPTS_PATH, document, root=OPTIMIZATION_ROOT)
    return receipt


def record_workload_baseline(
    *,
    runtime_fingerprint: Any,
    model_type: str,
    mode: str,
    artifact: str,
    workload_key: str,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    receipt = {
        "id": f"baseline-{uuid.uuid4().hex}",
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "kind": "workload_baseline",
        "status": "observed",
        "runtimeFingerprintHash": _stable_hash(runtime_fingerprint),
        "modelType": str(model_type),
        "mode": str(mode),
        "artifact": str(artifact),
        "workloadKey": str(workload_key),
        "measurement": deepcopy(measurement),
        "createdAt": _now(),
        "autoEligible": False,
    }
    with _STATE_LOCK:
        document = read_receipts()
        document["receipts"] = [receipt, *document.get("receipts", [])][:500]
        document["updatedAt"] = _now()
        _atomic_json(RECEIPTS_PATH, document, root=OPTIMIZATION_ROOT)
    return receipt


def _improvement_evidence(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    baseline_seconds = baseline.get("elapsedSeconds")
    optimized_seconds = optimized.get("elapsedSeconds")
    if (
        isinstance(baseline_seconds, (int, float))
        and baseline_seconds > 0
        and isinstance(optimized_seconds, (int, float))
    ):
        evidence["elapsedRatio"] = float(optimized_seconds) / float(baseline_seconds)
    baseline_memory = baseline.get("peakAllocatedBytes")
    optimized_memory = optimized.get("peakAllocatedBytes")
    if isinstance(baseline_memory, int) and baseline_memory > 0 and isinstance(optimized_memory, int):
        evidence["peakMemoryRatio"] = float(optimized_memory) / float(baseline_memory)
    evidence["improved"] = bool(
        evidence.get("elapsedRatio", 1.0) <= 0.98 or evidence.get("peakMemoryRatio", 1.0) <= 0.98
    )
    return evidence


def qualify_receipt(receipt_id: str, *, output_reviewed: bool) -> dict[str, Any]:
    if not output_reviewed:
        raise ValueError("A workload receipt requires explicit output review before Auto qualification.")
    with _STATE_LOCK:
        document = read_receipts()
        receipt = next((item for item in document.get("receipts", []) if item.get("id") == receipt_id), None)
        if not receipt or receipt.get("kind") != "workload" or receipt.get("status") != "observed":
            raise ValueError("Only an observed workload receipt can be qualified.")
        capability = _catalog().get(str(receipt.get("capabilityId"))) or {}
        if not capability.get("automaticEligible"):
            raise ValueError("This experimental capability is not eligible for automatic selection.")
        baseline = next(
            (
                item
                for item in document.get("receipts", [])
                if item.get("kind") == "workload_baseline"
                and item.get("status") == "observed"
                and item.get("runtimeFingerprintHash") == receipt.get("runtimeFingerprintHash")
                and item.get("modelType") == receipt.get("modelType")
                and item.get("mode") == receipt.get("mode")
                and item.get("artifact") == receipt.get("artifact")
                and item.get("workloadKey") == receipt.get("workloadKey")
            ),
            None,
        )
        if baseline is None:
            raise ValueError(
                "Run this unchanged workload once without optional optimizations to record a baseline first."
            )
        evidence = _improvement_evidence(
            baseline.get("measurement") if isinstance(baseline.get("measurement"), dict) else {},
            receipt.get("measurement") if isinstance(receipt.get("measurement"), dict) else {},
        )
        if not evidence.get("improved"):
            raise ValueError(
                "This optimization did not improve measured runtime or peak accelerator memory by at least 2%."
            )
        receipt["status"] = "qualified"
        receipt["autoEligible"] = True
        receipt["qualifiedAt"] = _now()
        receipt["baselineReceiptId"] = baseline.get("id")
        receipt["benchmarkEvidence"] = evidence
        document["updatedAt"] = _now()
        _atomic_json(RECEIPTS_PATH, document, root=OPTIMIZATION_ROOT)
        return deepcopy(receipt)


def qualified_auto_overrides(
    *,
    runtime_fingerprint: Any,
    model_type: str,
    mode: str,
    artifact: str,
    workload_key: str | None = None,
) -> dict[str, Any]:
    """Return only exact, enabled, workload-qualified optimization selections."""
    state = read_state()
    enabled = {str(item) for item in state.get("enabledCapabilities") or []}
    fingerprint_hash = _stable_hash(runtime_fingerprint)
    combined: dict[str, Any] = {}
    for receipt in read_receipts().get("receipts", []):
        if (
            receipt.get("status") != "qualified"
            or receipt.get("autoEligible") is not True
            or receipt.get("capabilityId") not in enabled
            or receipt.get("environmentId") != state.get("activeEnvironmentId")
            or receipt.get("runtimeFingerprintHash") != fingerprint_hash
            or receipt.get("modelType") != str(model_type)
            or receipt.get("mode") != str(mode)
            or receipt.get("artifact") != str(artifact)
            or (workload_key and receipt.get("workloadKey") != workload_key)
        ):
            continue
        selection = receipt.get("selection")
        if not isinstance(selection, dict):
            continue
        # Receipts are newest-first. Keep the newest qualified value when
        # several capabilities expose independent runtime fields.
        for key, value in selection.items():
            combined.setdefault(str(key), deepcopy(value))
    return combined


def delete_environment(environment_id: str) -> None:
    state = read_state()
    if environment_id in {state.get("activeEnvironmentId"), state.get("previousEnvironmentId")}:
        raise RuntimeError("Active and rollback environments cannot be deleted.")
    if not re.fullmatch(r"runtime-[0-9]{1,16}-[0-9a-f]{8}", str(environment_id or "")):
        raise ValueError("A valid managed environment identifier is required.")
    remove_managed_directory(ENVIRONMENTS_DIR / environment_id, parent=ENVIRONMENTS_DIR)
