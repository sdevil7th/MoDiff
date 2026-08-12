"""Reviewed custom Modular metadata, snapshots, and runtime bindings."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub.utils import validate_repo_id

from modiff.modular_workflow_discovery import (
    ModularWorkflowContractError,
    reviewed_modular_workflow_contract,
)

from .pipeline_schema import MoDiffPipelineConfig as PipelineConfig
from .pipeline_schema import MAX_CUSTOM_PIPELINE_REPOSITORY_CHARS
from .pipeline_schema import VerifiedMoDiffPipelineConfig


CUSTOM_PIPELINE_MODEL_TYPE = "DummyCustomPipeline"
CUSTOM_PIPELINE_EXECUTION_STATUS = "reviewed_official_components"
CUSTOM_PIPELINE_IDENTITY_SCHEMA = "modiff.custom-pipeline-identity.v2"
CUSTOM_PIPELINE_CONFIG_FILENAME = PipelineConfig.config_name
CUSTOM_PIPELINE_IDENTITY_FIELD = "modiff_pipeline_identity"

_COMMIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXECUTION_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY_KEYS = {
    "schema",
    "execution_id",
    "model_type",
    "source",
    "repo_id",
    "revision",
    "trust_remote_code",
    "config_filename",
    "config_sha256",
    "executable_manifest_sha256",
}

# Exact sidecars are individually capped at 1 MiB. Keep worst-case retained
# sidecar bytes near 32 MiB; correctness never depends on cache residency.
_BINDING_CACHE_LIMIT = 32
_UPSTREAM_INDEX_FILENAME = "modular_model_index.json"
_MAX_UPSTREAM_INDEX_BYTES = 1024 * 1024
_COMPONENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_OFFICIAL_COMPONENT_LIBRARIES = frozenset({"diffusers", "transformers"})
_COMPONENT_SPEC_KEYS = frozenset(
    {
        "type_hint",
        "pretrained_model_name_or_path",
        "subfolder",
        "variant",
        "revision",
    }
)


class CustomPipelineContractError(ValueError):
    """Actionable custom-pipeline admission failure exposed by the API."""

    def __init__(self, error_code: str, message: str, recovery_hint: str):
        super().__init__(message)
        self.modiff_category = "custom_pipeline"
        self.modiff_error_code = error_code
        self.modiff_recovery_hint = recovery_hint


@dataclass(frozen=True, slots=True)
class ReviewedComponentReference:
    """One official component and its immutable weight/config source."""

    name: str
    library: str
    class_name: str
    repository: str
    revision: str | None
    subfolder: str
    variant: str | None


@dataclass(frozen=True, slots=True)
class ReviewedCustomPipelineContract:
    """Bounded canonical upstream metadata, still free of imported model code."""

    pipeline_class_name: str
    blocks_class_name: str
    raw_index_bytes: bytes = field(repr=False, compare=False)
    index_sha256: str
    component_references: tuple[ReviewedComponentReference, ...]

    def index_document(self) -> dict[str, Any]:
        return _decode_upstream_index(
            self.raw_index_bytes,
            source_label=f"sha256:{self.index_sha256}:{_UPSTREAM_INDEX_FILENAME}",
        )


class PrivateExecutionSnapshot:
    """Task-private, content-addressed copy of the validated metadata bytes."""

    def __init__(self, *, identity: "CustomPipelineExecutionIdentity", sidecar: bytes, index: bytes):
        digest = hashlib.sha256(
            b"modiff-custom-execution-snapshot-v1\0"
            + bytes.fromhex(identity.config_sha256)
            + bytes.fromhex(hashlib.sha256(index).hexdigest())
        ).hexdigest()
        root = Path(tempfile.mkdtemp(prefix="modiff-custom-pipeline-"))
        os.chmod(root, 0o700)
        path = root / digest
        path.mkdir(mode=0o700)
        self.root = root
        self.path = path
        for filename, raw_bytes in (
            (CUSTOM_PIPELINE_CONFIG_FILENAME, sidecar),
            (_UPSTREAM_INDEX_FILENAME, index),
        ):
            target = path / filename
            with target.open("xb") as writer:
                writer.write(raw_bytes)
                writer.flush()
                os.fsync(writer.fileno())
            os.chmod(target, 0o400)
        os.chmod(path, 0o500)

    def cleanup(self) -> None:
        if not getattr(self, "root", None):
            return
        try:
            for path in sorted(self.root.rglob("*"), reverse=True):
                try:
                    os.chmod(path, 0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(self.root)
        finally:
            self.root = None

    def __del__(self):
        try:
            self.cleanup()
        except OSError:
            pass


def _reject_duplicate_upstream_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key {key!r} is not allowed")
        value[key] = item
    return value


def _reject_nonfinite_upstream_number(value):
    raise ValueError(f"Non-finite JSON number {value!r} is not allowed")


def _decode_upstream_index(raw_bytes: bytes, *, source_label: str) -> dict[str, Any]:
    if len(raw_bytes) > _MAX_UPSTREAM_INDEX_BYTES:
        raise CustomPipelineContractError(
            "custom_pipeline_snapshot_invalid",
            f"The canonical {_UPSTREAM_INDEX_FILENAME} exceeds the {_MAX_UPSTREAM_INDEX_BYTES}-byte limit.",
            "Repair the exact repository revision, then review the custom pipeline again.",
        )
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_upstream_keys,
            parse_constant=_reject_nonfinite_upstream_number,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise CustomPipelineContractError(
            "custom_pipeline_snapshot_invalid",
            f"The canonical {_UPSTREAM_INDEX_FILENAME} at {source_label!r} is not unambiguous UTF-8 JSON: {error}",
            "Repair the exact repository revision, then review the custom pipeline again.",
        ) from error
    if not isinstance(document, dict):
        raise CustomPipelineContractError(
            "custom_pipeline_snapshot_invalid",
            f"The canonical {_UPSTREAM_INDEX_FILENAME} must contain a JSON object.",
            "Repair the exact repository revision, then review the custom pipeline again.",
        )
    pending = [(document, 0)]
    values = 0
    while pending:
        item, depth = pending.pop()
        values += 1
        if values > 100_000 or depth > 64:
            raise CustomPipelineContractError(
                "custom_pipeline_snapshot_invalid",
                f"The canonical {_UPSTREAM_INDEX_FILENAME} exceeds the bounded JSON structure limits.",
                "Repair the exact repository revision, then review the custom pipeline again.",
            )
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    return document


def _read_bounded_file(path: Path, *, description: str) -> bytes:
    try:
        size = path.stat().st_size
        if size > _MAX_UPSTREAM_INDEX_BYTES:
            raise OSError(f"file exceeds the {_MAX_UPSTREAM_INDEX_BYTES}-byte limit")
        with path.open("rb") as reader:
            raw_bytes = reader.read(_MAX_UPSTREAM_INDEX_BYTES + 1)
    except OSError as error:
        raise CustomPipelineContractError(
            "custom_pipeline_snapshot_invalid",
            f"Could not read {description}: {error}",
            "Repair the exact repository revision, then review the custom pipeline again.",
        ) from error
    if len(raw_bytes) > _MAX_UPSTREAM_INDEX_BYTES:
        raise CustomPipelineContractError(
            "custom_pipeline_snapshot_invalid",
            f"{description} exceeds the {_MAX_UPSTREAM_INDEX_BYTES}-byte limit.",
            "Repair the exact repository revision, then review the custom pipeline again.",
        )
    return raw_bytes


def _index_path_from_verified(verified: VerifiedMoDiffPipelineConfig) -> Path:
    repository_path = Path(verified.repository_path).absolute()
    if verified.source == "local":
        path = repository_path / _UPSTREAM_INDEX_FILENAME
        if not path.exists() or not path.is_file() or path.is_symlink():
            raise CustomPipelineContractError(
                "custom_pipeline_snapshot_missing",
                f"The selected directory has no non-linked {_UPSTREAM_INDEX_FILENAME}.",
                f"Add the canonical {_UPSTREAM_INDEX_FILENAME}; no alternate filename is accepted.",
            )
        try:
            path.resolve(strict=True).relative_to(repository_path.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as error:
            raise CustomPipelineContractError(
                "custom_pipeline_snapshot_invalid",
                f"The local {_UPSTREAM_INDEX_FILENAME} escapes the selected repository.",
                "Use a regular metadata file contained by the selected directory.",
            ) from error
        return path

    path = repository_path / _UPSTREAM_INDEX_FILENAME
    if not path.exists() or not path.is_file():
        raise CustomPipelineContractError(
            "custom_pipeline_snapshot_missing",
            f"No cached {_UPSTREAM_INDEX_FILENAME} exists for {verified.repo_id}@{verified.revision}.",
            "Install that exact repository revision through Model Manager, then review the contract again.",
        )
    if path.parent != repository_path or not path.is_file():
        raise CustomPipelineContractError(
            "custom_pipeline_snapshot_invalid",
            f"The cached {_UPSTREAM_INDEX_FILENAME} is not a regular root metadata file.",
            "Repair the exact cached revision through Model Manager.",
        )
    if path.is_symlink():
        try:
            target = path.resolve(strict=True)
            blob_root = repository_path.parent.parent / "blobs"
            target.relative_to(blob_root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as error:
            raise CustomPipelineContractError(
                "custom_pipeline_snapshot_invalid",
                f"The cached {_UPSTREAM_INDEX_FILENAME} does not resolve inside its repository cache.",
                "Repair the exact cached revision through Model Manager.",
            ) from error
    return path


def _normalize_subfolder(value: Any, *, component_name: str) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or len(value) > 512 or "\\" in value or "\x00" in value:
        raise CustomPipelineContractError(
            "custom_pipeline_snapshot_invalid",
            f"Component {component_name!r} has an invalid subfolder.",
            "Use a short traversal-free relative POSIX subfolder.",
        )
    parts = value.split("/")
    if value.startswith("/") or any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise CustomPipelineContractError(
            "custom_pipeline_snapshot_invalid",
            f"Component {component_name!r} has an invalid subfolder.",
            "Use a short traversal-free relative POSIX subfolder.",
        )
    return value


def _review_upstream_contract(
    verified: VerifiedMoDiffPipelineConfig,
    raw_index_bytes: bytes,
) -> ReviewedCustomPipelineContract:
    document = _decode_upstream_index(raw_index_bytes, source_label=verified.repo_id)
    if "auto_map" in document:
        raise CustomPipelineContractError(
            "custom_pipeline_authorization_required",
            "The canonical pipeline metadata selects repository Python through auto_map and needs task-scoped operator authorization.",
            "Repository Python requires a fresh task-scoped operator authorization; imported workflow data is not consent.",
        )
    if document.get("requirements") not in (None, {}, []):
        raise CustomPipelineContractError(
            "custom_pipeline_component_unapproved",
            "Repository package requirements are not an approved custom pipeline dependency source.",
            "Use only MoDiff's reviewed base dependencies and P0.5 optional-runtime profiles.",
        )
    pipeline_class_name = document.get("_class_name")
    blocks_class_name = document.get("_blocks_class_name")
    try:
        workflow_contract = reviewed_modular_workflow_contract(pipeline_class_name)
    except ModularWorkflowContractError as error:
        raise CustomPipelineContractError(
            "custom_pipeline_component_unapproved",
            "The canonical pipeline class has no reviewed pinned workflow contract.",
            "Select a pipeline using an installed reviewed Modular Diffusers pipeline and blocks pair.",
        ) from error
    if not isinstance(blocks_class_name, str) or blocks_class_name != workflow_contract["blocksClass"]:
        raise CustomPipelineContractError(
            "custom_pipeline_component_unapproved",
            "The canonical pipeline/block class pair is not in MoDiff's pinned Diffusers contract.",
            "Select a pipeline using an installed reviewed Modular Diffusers class and its exact official blocks class.",
        )

    references = []
    for name, raw_component in document.items():
        if isinstance(name, str) and name.startswith("_"):
            continue
        if not isinstance(raw_component, list):
            continue
        if not _COMPONENT_NAME.fullmatch(str(name)) or len(raw_component) != 3 or not isinstance(raw_component[2], dict):
            raise CustomPipelineContractError(
                "custom_pipeline_snapshot_invalid",
                f"The canonical pipeline component {name!r} has malformed Modular metadata.",
                "Repair the exact modular_model_index.json and review it again.",
            )
        unknown_keys = set(raw_component[2]) - _COMPONENT_SPEC_KEYS
        if unknown_keys:
            raise CustomPipelineContractError(
                "custom_pipeline_snapshot_invalid",
                f"Component {name!r} has unsupported loading fields: {', '.join(sorted(unknown_keys))}.",
                "Use only the pinned Modular ComponentSpec loading fields.",
            )
        type_hint = raw_component[2].get("type_hint")
        if (
            not isinstance(type_hint, list)
            or len(type_hint) != 2
            or type_hint[0] not in _OFFICIAL_COMPONENT_LIBRARIES
            or not isinstance(type_hint[1], str)
            or _COMPONENT_NAME.fullmatch(type_hint[1]) is None
        ):
            raise CustomPipelineContractError(
                "custom_pipeline_component_unapproved",
                f"Component {name!r} does not declare an approved official Hugging Face library/class pair.",
                "Use only reviewed top-level Diffusers or Transformers component classes.",
            )
        outer_type_hint = raw_component[:2]
        if outer_type_hint not in ([None, None], type_hint):
            raise CustomPipelineContractError(
                "custom_pipeline_component_unapproved",
                f"Component {name!r} has conflicting outer library/class metadata.",
                "Use null outer metadata or the same reviewed official pair as type_hint.",
            )
        repository = raw_component[2].get("pretrained_model_name_or_path")
        revision = raw_component[2].get("revision")
        inherits_main_repository = repository in (None, "")
        if inherits_main_repository:
            repository = verified.repo_id
        if revision in (None, "") and repository == verified.repo_id and verified.source == "hub":
            revision = verified.revision
        if not isinstance(repository, str) or not repository or len(repository) > 4096:
            raise CustomPipelineContractError(
                "custom_pipeline_snapshot_invalid",
                f"Component {name!r} has an invalid repository reference.",
                "Use an explicit Hugging Face repository ID for every pretrained component.",
            )
        local_preview_reference = verified.source == "local" and inherits_main_repository
        if not local_preview_reference:
            try:
                validate_repo_id(repository)
            except ValueError as error:
                raise CustomPipelineContractError(
                    "custom_pipeline_unpinned_auxiliary",
                    f"Component {name!r} repository {repository!r} is not a Hugging Face repository ID.",
                    "Use a Hub repository ID and pin it to an exact 40-character commit; local component paths are not executable.",
                ) from error
        if not local_preview_reference and (
            not isinstance(revision, str) or _COMMIT_REVISION.fullmatch(revision) is None
        ):
            error_code = (
                "custom_pipeline_unpinned_auxiliary"
                if repository != verified.repo_id
                else "custom_pipeline_snapshot_invalid"
            )
            raise CustomPipelineContractError(
                error_code,
                f"Component {name!r} repository {repository!r} lacks an exact lowercase commit revision.",
                "Pin every main and auxiliary component repository to a 40-character commit.",
            )
        variant = raw_component[2].get("variant")
        if variant is not None and (not isinstance(variant, str) or len(variant) > 128):
            raise CustomPipelineContractError(
                "custom_pipeline_snapshot_invalid",
                f"Component {name!r} has an invalid variant.",
                "Use a bounded string variant or null.",
            )
        references.append(
            ReviewedComponentReference(
                name=name,
                library=type_hint[0],
                class_name=type_hint[1],
                repository=repository,
                revision=revision,
                subfolder=_normalize_subfolder(raw_component[2].get("subfolder"), component_name=name),
                variant=variant,
            )
        )
    return ReviewedCustomPipelineContract(
        pipeline_class_name=pipeline_class_name,
        blocks_class_name=blocks_class_name,
        raw_index_bytes=raw_index_bytes,
        index_sha256=hashlib.sha256(raw_index_bytes).hexdigest(),
        component_references=tuple(references),
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _identity_body(
    *,
    source: str,
    repo_id: str,
    revision: str | None,
    trust_remote_code: bool,
    config_sha256: str,
    executable_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": CUSTOM_PIPELINE_IDENTITY_SCHEMA,
        "model_type": CUSTOM_PIPELINE_MODEL_TYPE,
        "source": source,
        "repo_id": repo_id,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "config_filename": CUSTOM_PIPELINE_CONFIG_FILENAME,
        "config_sha256": config_sha256,
        "executable_manifest_sha256": executable_manifest_sha256,
    }


def _execution_id_for_body(body: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(body)).hexdigest()


@dataclass(frozen=True, slots=True)
class CustomPipelineExecutionIdentity:
    """A versioned, source-bound checksum; never an execution authorization."""

    source: str
    repo_id: str
    revision: str | None
    trust_remote_code: bool
    config_sha256: str
    executable_manifest_sha256: str
    execution_id: str

    @classmethod
    def create(
        cls,
        *,
        source: str,
        repo_id: str,
        revision: str | None,
        trust_remote_code: bool,
        config_sha256: str,
        executable_manifest_sha256: str,
    ) -> "CustomPipelineExecutionIdentity":
        if not isinstance(source, str) or source not in {"hub", "local"}:
            raise ValueError("Custom Modular Diffusers identity source must be exactly 'hub' or 'local'.")
        if not isinstance(repo_id, str) or not repo_id or repo_id != repo_id.strip():
            raise ValueError("Custom Modular Diffusers identity requires a normalized non-empty repository.")
        if len(repo_id) > MAX_CUSTOM_PIPELINE_REPOSITORY_CHARS:
            raise ValueError("Custom Modular Diffusers identity repository exceeds the 4096-character boundary.")
        if type(trust_remote_code) is not bool:
            raise TypeError("Custom Modular Diffusers trust_remote_code must be a JSON boolean.")
        if trust_remote_code:
            raise ValueError(
                "Custom Modular Diffusers repository code is disabled until MoDiff provides a reviewed, "
                "task-scoped authorization and isolated content-addressed execution path."
            )
        if source == "hub":
            if not isinstance(revision, str) or _COMMIT_REVISION.fullmatch(revision) is None:
                raise ValueError("Custom Hub pipeline identity requires a lowercase 40-character commit revision.")
        else:
            if revision is not None:
                raise ValueError("Local custom pipeline identities must use a null revision.")
        if not isinstance(config_sha256, str) or _SHA256.fullmatch(config_sha256) is None:
            raise ValueError("Custom Modular Diffusers identity requires a lowercase SHA-256 sidecar digest.")
        if (
            not isinstance(executable_manifest_sha256, str)
            or _SHA256.fullmatch(executable_manifest_sha256) is None
        ):
            raise ValueError(
                "Custom Modular Diffusers identity requires a lowercase SHA-256 executable-manifest digest."
            )

        body = _identity_body(
            source=source,
            repo_id=repo_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            config_sha256=config_sha256,
            executable_manifest_sha256=executable_manifest_sha256,
        )
        execution_id = _execution_id_for_body(body)
        return cls(
            source=source,
            repo_id=repo_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            config_sha256=config_sha256,
            executable_manifest_sha256=executable_manifest_sha256,
            execution_id=execution_id,
        )

    @classmethod
    def from_value(cls, value: Any) -> "CustomPipelineExecutionIdentity":
        if not isinstance(value, Mapping):
            raise ValueError("Custom Modular Diffusers contract identity must be a JSON object.")
        raw_keys = list(value.keys())
        if any(not isinstance(key, str) for key in raw_keys):
            raise ValueError("Custom Modular Diffusers contract identity keys must be JSON strings.")
        keys = set(raw_keys)
        if keys != _IDENTITY_KEYS:
            missing = sorted(_IDENTITY_KEYS - keys)
            unknown = sorted(keys - _IDENTITY_KEYS)
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unknown:
                detail.append("unknown " + ", ".join(unknown))
            raise ValueError("Malformed custom Modular Diffusers contract identity: " + "; ".join(detail))
        if value.get("schema") != CUSTOM_PIPELINE_IDENTITY_SCHEMA:
            raise ValueError(
                f"Unsupported custom Modular Diffusers identity schema {value.get('schema')!r}; "
                f"expected {CUSTOM_PIPELINE_IDENTITY_SCHEMA!r}."
            )
        if value.get("model_type") != CUSTOM_PIPELINE_MODEL_TYPE:
            raise ValueError("Custom Modular Diffusers identity has an incompatible model_type.")
        if value.get("config_filename") != CUSTOM_PIPELINE_CONFIG_FILENAME:
            raise ValueError(f"Custom Modular Diffusers identity must bind {CUSTOM_PIPELINE_CONFIG_FILENAME!r}.")
        execution_id = value.get("execution_id")
        if not isinstance(execution_id, str) or _EXECUTION_ID.fullmatch(execution_id) is None:
            raise ValueError("Custom Modular Diffusers identity has an invalid execution_id.")
        identity = cls.create(
            source=value.get("source"),
            repo_id=value.get("repo_id"),
            revision=value.get("revision"),
            trust_remote_code=value.get("trust_remote_code"),
            config_sha256=value.get("config_sha256"),
            executable_manifest_sha256=value.get("executable_manifest_sha256"),
        )
        if not hmac.compare_digest(identity.execution_id, execution_id):
            raise ValueError("Custom Modular Diffusers contract checksum does not match its identity fields.")
        return identity

    def to_dict(self) -> dict[str, Any]:
        return {
            **_identity_body(
                source=self.source,
                repo_id=self.repo_id,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                config_sha256=self.config_sha256,
                executable_manifest_sha256=self.executable_manifest_sha256,
            ),
            "execution_id": self.execution_id,
        }

    def selector_tuple(self) -> tuple[str, str, str | None, bool]:
        return self.source, self.repo_id, self.revision, self.trust_remote_code


@dataclass(frozen=True, slots=True)
class CustomPipelineBinding:
    """Pipeline-class substitute bound to reviewed metadata and drift checksums."""

    identity: CustomPipelineExecutionIdentity
    _config_bytes: bytes = field(repr=False, compare=False)
    _repository_path: str = field(repr=False, compare=False)
    _execution_contract: ReviewedCustomPipelineContract = field(repr=False, compare=False)

    @property
    def __name__(self) -> str:
        return CUSTOM_PIPELINE_MODEL_TYPE

    @property
    def execution_status(self) -> str:
        return CUSTOM_PIPELINE_EXECUTION_STATUS

    @property
    def repo_id(self) -> str:
        return self.identity.repo_id

    @property
    def revision(self) -> str | None:
        return self.identity.revision

    @property
    def trust_remote_code(self) -> bool:
        return self.identity.trust_remote_code

    @property
    def repository_path(self) -> str:
        return self._repository_path

    def pipeline_config(self) -> PipelineConfig:
        return PipelineConfig.from_json_bytes(
            self._config_bytes,
            source_label=f"{self.identity.execution_id}:{CUSTOM_PIPELINE_CONFIG_FILENAME}",
        )

    @property
    def execution_contract(self) -> ReviewedCustomPipelineContract:
        return self._execution_contract

    def instantiate(self, *, components_manager=None, collection=None):
        """Reconcile source bytes, snapshot them, then construct only installed official blocks."""

        if self.identity.source != "hub":
            resolve_custom_pipeline_identity(self.identity.to_dict())
            raise CustomPipelineContractError(
                "custom_pipeline_immutable_snapshot_required",
                "Local custom pipeline directories are mutable and are contract-preview only.",
                "Publish the reviewed metadata and components at immutable Hub commits before execution.",
            )
        current = resolve_custom_pipeline_identity(self.identity.to_dict())
        contract = current.execution_contract
        snapshot = PrivateExecutionSnapshot(
            identity=current.identity,
            sidecar=current._config_bytes,
            index=contract.raw_index_bytes,
        )
        try:
            import diffusers as diffusers_module
            from diffusers import ModularPipeline, ModularPipelineBlocks
            from diffusers.pipelines.pipeline_loading_utils import _fetch_class_library_tuple

            pipeline_class = getattr(diffusers_module, contract.pipeline_class_name, None)
            blocks_class = getattr(diffusers_module, contract.blocks_class_name, None)
            if (
                not isinstance(pipeline_class, type)
                or not issubclass(pipeline_class, ModularPipeline)
                or not isinstance(blocks_class, type)
                or not issubclass(blocks_class, ModularPipelineBlocks)
                or not pipeline_class.__module__.startswith("diffusers.")
                or not blocks_class.__module__.startswith("diffusers.")
            ):
                raise CustomPipelineContractError(
                    "custom_pipeline_component_unapproved",
                    "The installed Diffusers runtime does not provide the reviewed pipeline/block class pair.",
                    "Install and activate the exact reviewed optional runtime, then retry.",
                )
            blocks = blocks_class()
            expected = {
                component.name: tuple(_fetch_class_library_tuple(component.type_hint))
                for component in blocks.expected_components
                if component.default_creation_method == "from_pretrained"
            }
            observed = {
                component.name: (component.library, component.class_name)
                for component in contract.component_references
            }
            missing = sorted(set(expected) - set(observed))
            unexpected = sorted(set(observed) - set(expected))
            mismatched = sorted(
                name for name in set(expected) & set(observed) if observed[name] != expected[name]
            )
            if missing or unexpected or mismatched:
                details = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if unexpected:
                    details.append("unexpected " + ", ".join(unexpected))
                if mismatched:
                    details.append("unapproved types for " + ", ".join(mismatched))
                raise CustomPipelineContractError(
                    "custom_pipeline_component_unapproved",
                    "The canonical component contract does not match the installed official blocks: "
                    + "; ".join(details)
                    + ".",
                    "Use the exact component names and official library/classes declared by the pinned blocks.",
                )

            document = contract.index_document()
            for reference in contract.component_references:
                spec = document[reference.name][2]
                spec["pretrained_model_name_or_path"] = reference.repository
                spec["revision"] = reference.revision
                spec["subfolder"] = reference.subfolder
                spec["variant"] = reference.variant
            pipeline = pipeline_class(
                blocks=blocks,
                pretrained_model_name_or_path=self.identity.repo_id,
                components_manager=components_manager,
                collection=collection,
                modular_config_dict=document,
            )
            pipeline._modiff_custom_execution_snapshot = snapshot
            pipeline._modiff_custom_execution_identity = self.identity.to_dict()
            return pipeline
        except Exception:
            snapshot.cleanup()
            raise

    def __call__(self):
        return self.instantiate()


_binding_cache: "OrderedDict[tuple[str, str, str | None, bool, str, str], CustomPipelineBinding]" = OrderedDict()
_binding_cache_lock = threading.RLock()


def _binding_from_verified(
    verified: VerifiedMoDiffPipelineConfig,
    *,
    execution_contract: ReviewedCustomPipelineContract,
    trust_remote_code: bool,
    expected_identity: Mapping[str, Any] | CustomPipelineExecutionIdentity | None = None,
    allow_selector_change: bool = False,
) -> CustomPipelineBinding:
    identity = CustomPipelineExecutionIdentity.create(
        source=verified.source,
        repo_id=verified.repo_id,
        revision=verified.revision,
        trust_remote_code=trust_remote_code,
        config_sha256=verified.sha256,
        executable_manifest_sha256=verified.executable_manifest_sha256,
    )
    if expected_identity is not None:
        expected = (
            expected_identity
            if isinstance(expected_identity, CustomPipelineExecutionIdentity)
            else CustomPipelineExecutionIdentity.from_value(expected_identity)
        )
        selector_changed = expected.selector_tuple() != identity.selector_tuple()
        if expected != identity and not (allow_selector_change and selector_changed):
            if expected.config_sha256 != identity.config_sha256 and not selector_changed:
                raise ValueError(
                    f"Cached {CUSTOM_PIPELINE_CONFIG_FILENAME} no longer matches the persisted custom pipeline "
                    "identity. Review the exact sidecar and explicitly refresh the custom contract before running."
                )
            if (
                expected.executable_manifest_sha256 != identity.executable_manifest_sha256
                and not selector_changed
            ):
                raise ValueError(
                    "Cached custom pipeline executable metadata no longer matches the persisted contract identity. "
                    "Review the exact cached files and explicitly refresh the custom contract before running."
                )
            raise ValueError(
                "The selected custom Modular Diffusers source does not match its persisted contract identity. "
                "Refresh the loader contract after changing repository, source, revision, or trust."
            )

    # Keep the exact verified bytes.  Parsing a fresh object for every caller
    # isolates mutable ``node_params`` without making the binding depend on a
    # second serialization (or on cache residency).
    config_bytes = verified.raw_bytes
    key = (
        identity.source,
        identity.repo_id,
        identity.revision,
        identity.trust_remote_code,
        identity.config_sha256,
        identity.executable_manifest_sha256,
    )
    with _binding_cache_lock:
        existing = _binding_cache.get(key)
        if (
            existing is not None
            and existing._config_bytes == config_bytes
            and existing.repository_path == verified.repository_path
            and existing.execution_contract.raw_index_bytes == execution_contract.raw_index_bytes
        ):
            _binding_cache.move_to_end(key)
            return existing
        binding = CustomPipelineBinding(
            identity=identity,
            _config_bytes=config_bytes,
            _repository_path=verified.repository_path,
            _execution_contract=execution_contract,
        )
        _binding_cache[key] = binding
        _binding_cache.move_to_end(key)
        while len(_binding_cache) > _BINDING_CACHE_LIMIT:
            _binding_cache.popitem(last=False)
        return binding


def resolve_custom_pipeline_binding(
    *,
    source: str,
    repo_id: str,
    revision: str | None,
    trust_remote_code: bool,
    expected_identity: Mapping[str, Any] | CustomPipelineExecutionIdentity | None = None,
    allow_selector_change: bool = False,
) -> CustomPipelineBinding:
    """Verify one exact sidecar and canonical upstream metadata without importing model code."""

    if type(trust_remote_code) is not bool:
        raise TypeError("Custom Modular Diffusers trust_remote_code must be a JSON boolean.")
    if trust_remote_code:
        raise ValueError(
            "Custom Modular Diffusers repository code is disabled until MoDiff provides a reviewed, task-scoped "
            "authorization and isolated content-addressed execution path."
        )
    try:
        verified = PipelineConfig.load_verified(
            repo_id,
            source=source,
            revision=revision,
        )
    except (EnvironmentError, TypeError, ValueError) as error:
        if isinstance(error, CustomPipelineContractError):
            raise
        raise CustomPipelineContractError(
            "custom_pipeline_sidecar_missing",
            str(error),
            f"Provide the exact declarative {CUSTOM_PIPELINE_CONFIG_FILENAME}; alternate filenames are not accepted.",
        ) from error
    index_path = _index_path_from_verified(verified)
    raw_index_bytes = _read_bounded_file(
        index_path.resolve(strict=True),
        description=f"canonical {_UPSTREAM_INDEX_FILENAME}",
    )
    execution_contract = _review_upstream_contract(verified, raw_index_bytes)
    return _binding_from_verified(
        verified,
        execution_contract=execution_contract,
        trust_remote_code=trust_remote_code,
        expected_identity=expected_identity,
        allow_selector_change=allow_selector_change,
    )


def resolve_custom_pipeline_identity(value: Any) -> CustomPipelineBinding:
    """Recover and re-verify a persisted/runtime identity without network access."""

    identity = CustomPipelineExecutionIdentity.from_value(value)
    return resolve_custom_pipeline_binding(
        source=identity.source,
        repo_id=identity.repo_id,
        revision=identity.revision,
        trust_remote_code=identity.trust_remote_code,
        expected_identity=identity,
    )


def _clear_custom_pipeline_binding_cache_for_tests() -> None:
    with _binding_cache_lock:
        _binding_cache.clear()
