"""Versioned drift checksums and contract-only custom Modular bindings."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .pipeline_schema import MoDiffPipelineConfig as PipelineConfig
from .pipeline_schema import MAX_CUSTOM_PIPELINE_REPOSITORY_CHARS
from .pipeline_schema import VerifiedMoDiffPipelineConfig


CUSTOM_PIPELINE_MODEL_TYPE = "DummyCustomPipeline"
CUSTOM_PIPELINE_EXECUTION_STATUS = "contract_only"
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
    """Contract-only pipeline-class substitute bound to verified drift checksums."""

    identity: CustomPipelineExecutionIdentity
    _config_bytes: bytes = field(repr=False, compare=False)
    _repository_path: str = field(repr=False, compare=False)

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

    def __call__(self):
        # Upstream currently imports the repository-controlled library named in
        # each component type_hint even with trust_remote_code=False. Reconcile
        # drift, then fail closed until MoDiff has a reviewed component contract.
        resolve_custom_pipeline_identity(self.identity.to_dict())
        raise RuntimeError(
            "Custom Modular Diffusers execution is disabled until MoDiff validates every component type_hint "
            "against its reviewed executable dependency contract. Contract preview remains available."
        )


_binding_cache: "OrderedDict[tuple[str, str, str | None, bool, str, str], CustomPipelineBinding]" = OrderedDict()
_binding_cache_lock = threading.RLock()


def _binding_from_verified(
    verified: VerifiedMoDiffPipelineConfig,
    *,
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
        ):
            _binding_cache.move_to_end(key)
            return existing
        binding = CustomPipelineBinding(
            identity=identity,
            _config_bytes=config_bytes,
            _repository_path=verified.repository_path,
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
    """Verify a locally available sidecar/manifest and return a contract-only binding."""

    if type(trust_remote_code) is not bool:
        raise TypeError("Custom Modular Diffusers trust_remote_code must be a JSON boolean.")
    if trust_remote_code:
        raise ValueError(
            "Custom Modular Diffusers repository code is disabled until MoDiff provides a reviewed, task-scoped "
            "authorization and isolated content-addressed execution path."
        )
    verified = PipelineConfig.load_verified(
        repo_id,
        source=source,
        revision=revision,
    )
    return _binding_from_verified(
        verified,
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
