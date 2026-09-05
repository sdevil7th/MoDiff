"""Read-only partial evidence for historical Cluster Studio specifications.

The legacy Studio specification hash is a 32-bit FNV receipt.  It is useful
for checking that a recovered body is internally consistent, but it is not a
collision-resistant authority and it does not authenticate a historical node
library manifest.  This module therefore wraps recovered bodies in a checked-
in SHA-256 ledger whose boundary explicitly cannot authorize execution,
semantic equivalence, compiler output, or workflow mutation.

Raw frontend state captures are intentionally not copied into the repository.
The bounded extractor accepts exact pinned gzip files, reads only the saved
capability-specification collection, and intersects it with exact historical
Studio-spec identities retained by the workflow inventory.  It never extracts
workflow paths, instance IDs, prompts, parameters, or generated media.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


STUDIO_SPEC_EVIDENCE_SCHEMA_VERSION = 1
STUDIO_SPEC_REVIEW_SCHEMA_VERSION = 1
STUDIO_SPEC_EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "legacy-cluster-studio-spec-evidence.v1.json"
)
STUDIO_SPEC_REVIEW_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "legacy-cluster-studio-spec-partial-reviews.v1.json"
)

_MAX_LEDGER_BYTES = 4 * 1024 * 1024
_MAX_CAPTURE_COUNT = 8
_MAX_COMPRESSED_BYTES = 16 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_MAX_CAPTURE_SPECIFICATIONS = 2_048
_MAX_SPECIFICATION_BYTES = 256 * 1024
_MAX_SPECIFICATIONS = 512
_MAX_ROLES = 128
_MAX_EDGES = 512
_MAX_BINDINGS = 512
_MAX_AUTO_FIELDS = 128
_MAX_ACTIONS = 128

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_STUDIO_HASH = re.compile(r"^studio-spec-v1-[0-9a-f]{8}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_REVIEW_ID = re.compile(r"^legacy-studio-spec-partial-review:[A-Za-z0-9._:-]{1,180}$")

_SPEC_REQUIRED_FIELDS = {
    "schemaVersion",
    "canonicalizationVersion",
    "id",
    "modelType",
    "mode",
    "executionProfileId",
    "loaderModule",
    "loaderAction",
    "executionPath",
    "pipelineClass",
    "defaultRepo",
    "roles",
    "edges",
    "bindings",
    "autoFields",
    "actions",
    "contentHash",
}
_SPEC_OPTIONAL_FIELDS = {"auxiliaryTerminalRoles"}
_EVIDENCE_BOUNDARY = {
    "checkedInEvidenceOnly": True,
    "authorizesConversion": False,
    "authorizesExecution": False,
    "authorizesCompilerSupplement": False,
    "authorizesSemanticEquivalence": False,
    "containsHistoricalManifestBody": False,
    "containsBlockDefinitionV2": False,
    "containsWorkflowPathsOrInstanceIds": False,
    "containsPromptOrParameterValues": False,
}
_REVIEW_BOUNDARY = {
    "checkedInPartialReviewOnly": True,
    "authorizesConversion": False,
    "authorizesExecution": False,
    "authorizesCompilerSupplement": False,
    "authorizesSemanticEquivalence": False,
}
_CAPABILITY_SELECTOR = "/nodes/studioModelCapabilities/*/studioExecutionSpecs/*"


# These are local qualification captures, not publisher or execution
# authority.  Their collision-resistant byte receipts merely make extraction
# reproducible on a host where the ignored captures are still retained.
REVIEWED_CAPTURE_SOURCES: dict[str, dict[str, Any]] = {
    "frontend-capture-2026-08-24-sdxl": {
        "artifactLabel": (
            "frontend-pipeline-e2e-2026-08-24/sdxl-cluster-qualification/"
            "diagnostic-loader-variant-mismatch/inserted-state.json.gz"
        ),
        "compressedSha256": (
            "sha256:49a568bb1ebcd2a177fd08902d2ccf9addcdd4830a52057bf494ffa72443e1c8"
        ),
        "compressedBytes": 4_363_263,
        "uncompressedSha256": (
            "sha256:0e4f950acc64061642fa9e08fab2515132943040fcdd2e31cc1491ed7923337e"
        ),
        "uncompressedBytes": 52_654_748,
        "selector": _CAPABILITY_SELECTOR,
    },
    "frontend-capture-2026-08-25-smollm2": {
        "artifactLabel": (
            "frontend-pipeline-e2e-2026-08-25/"
            "transformers-cluster-smollm2-text-generation-v1/configured-state.json.gz"
        ),
        "compressedSha256": (
            "sha256:f16df92186ab1b1efcbc3547abf120acd5bb971ed106ff9f387ae7299d3b266c"
        ),
        "compressedBytes": 3_988_232,
        "uncompressedSha256": (
            "sha256:c1c1edb88d37bc18788a83109dc72f9dbbc77ac55641a31a6bca193f8c96852c"
        ),
        "uncompressedBytes": 48_121_577,
        "selector": _CAPABILITY_SELECTOR,
    },
}


class LegacyStudioSpecEvidenceError(ValueError):
    """Raised when partial historical Studio-spec evidence is malformed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _content_hash(value: Mapping[str, Any]) -> str:
    body = deepcopy(dict(value))
    body.pop("contentHash", None)
    return _canonical_sha256(body)


def _hash_omitting(value: Mapping[str, Any], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _ordered_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _ordered_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_ordered_value(item) for item in value]
    return value


def _legacy_stable_json(value: Any) -> str:
    return json.dumps(_ordered_value(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _legacy_hash_string(value: str) -> str:
    digest = 0x811C9DC5
    encoded = value.encode("utf-16-le")
    for index in range(0, len(encoded), 2):
        digest ^= encoded[index] | encoded[index + 1] << 8
        digest = digest * 0x01000193 & 0xFFFFFFFF
    return f"{digest:08x}"


def studio_spec_content_hash_v1(value: Mapping[str, Any]) -> str:
    """Return the historical 32-bit self-hash for one complete public body."""

    body = {key: deepcopy(item) for key, item in value.items() if key != "contentHash"}
    return f"studio-spec-v1-{_legacy_hash_string(_legacy_stable_json(body))}"


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LegacyStudioSpecEvidenceError(f"{label} has missing or unknown fields.")
    return value


def _object_with_optional(
    value: Any,
    required: set[str],
    optional: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LegacyStudioSpecEvidenceError(f"{label} must be an object.")
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise LegacyStudioSpecEvidenceError(f"{label} has missing or unknown fields.")
    return value


def _text(
    value: Any,
    label: str,
    *,
    maximum: int = 4096,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or (pattern is not None and pattern.fullmatch(value) is None)
    ):
        raise LegacyStudioSpecEvidenceError(f"{label} is invalid.")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LegacyStudioSpecEvidenceError(f"{label} must be a nonnegative integer.")
    return value


def _json_value(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > 9_007_199_254_740_991:
            raise LegacyStudioSpecEvidenceError(f"{label} contains an unsafe integer.")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LegacyStudioSpecEvidenceError(f"{label} contains a non-finite number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LegacyStudioSpecEvidenceError(f"{label} contains a non-string key.")
            _json_value(item, f"{label}.{key}")
        return
    raise LegacyStudioSpecEvidenceError(f"{label} contains a non-JSON value.")


def _string_tuple(value: Any, length: int, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise LegacyStudioSpecEvidenceError(f"{label} is invalid.")
    return tuple(_text(item, f"{label}[{index}]", maximum=512) for index, item in enumerate(value))


def validate_historical_studio_spec(value: Any) -> dict[str, Any]:
    """Validate one historical public Studio spec without current-registry inference."""

    spec = _object_with_optional(
        value,
        _SPEC_REQUIRED_FIELDS,
        _SPEC_OPTIONAL_FIELDS,
        "Historical Studio specification",
    )
    _json_value(spec, "Historical Studio specification")
    if spec.get("schemaVersion") != 1 or spec.get("canonicalizationVersion") != 1:
        raise LegacyStudioSpecEvidenceError("Historical Studio specification version is unsupported.")
    for field in (
        "id",
        "modelType",
        "mode",
        "executionProfileId",
        "loaderModule",
        "loaderAction",
        "executionPath",
        "pipelineClass",
        "defaultRepo",
    ):
        _text(spec.get(field), f"Historical Studio specification.{field}", maximum=512)
    _text(
        spec.get("contentHash"),
        "Historical Studio specification.contentHash",
        pattern=_STUDIO_HASH,
    )

    roles_value = spec.get("roles")
    if not isinstance(roles_value, list) or not roles_value or len(roles_value) > _MAX_ROLES:
        raise LegacyStudioSpecEvidenceError("Historical Studio specification.roles is invalid.")
    roles: set[str] = set()
    for index, raw in enumerate(roles_value):
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise LegacyStudioSpecEvidenceError(
                f"Historical Studio specification.roles[{index}] is invalid."
            )
        role = _text(raw[0], f"Historical Studio specification.roles[{index}][0]", maximum=256)
        node_key = _text(raw[1], f"Historical Studio specification.roles[{index}][1]", maximum=512)
        if role in roles or "." not in node_key:
            raise LegacyStudioSpecEvidenceError("Historical Studio specification roles are ambiguous.")
        for coordinate in raw[2:]:
            if not isinstance(coordinate, int) or isinstance(coordinate, bool) or abs(coordinate) > 1_000_000:
                raise LegacyStudioSpecEvidenceError("Historical Studio specification role position is invalid.")
        roles.add(role)

    edges_value = spec.get("edges")
    if not isinstance(edges_value, list) or len(edges_value) > _MAX_EDGES:
        raise LegacyStudioSpecEvidenceError("Historical Studio specification.edges is invalid.")
    edges: set[tuple[str, ...]] = set()
    adjacency = {role: set() for role in roles}
    outgoing: set[str] = set()
    for index, raw in enumerate(edges_value):
        edge = _string_tuple(raw, 4, f"Historical Studio specification.edges[{index}]")
        if edge in edges or edge[0] not in roles or edge[2] not in roles:
            raise LegacyStudioSpecEvidenceError("Historical Studio specification edges are ambiguous.")
        edges.add(edge)
        outgoing.add(edge[0])
        adjacency[edge[0]].add(edge[2])
        adjacency[edge[2]].add(edge[0])
    visited: set[str] = set()
    pending = [next(iter(roles))]
    while pending:
        role = pending.pop()
        if role in visited:
            continue
        visited.add(role)
        pending.extend(adjacency[role] - visited)
    if visited != roles:
        raise LegacyStudioSpecEvidenceError("Historical Studio specification graph is disconnected.")

    bindings_value = spec.get("bindings")
    if not isinstance(bindings_value, list) or len(bindings_value) > _MAX_BINDINGS:
        raise LegacyStudioSpecEvidenceError("Historical Studio specification.bindings is invalid.")
    binding_targets: set[tuple[str, str]] = set()
    for index, raw in enumerate(bindings_value):
        binding = _string_tuple(raw, 3, f"Historical Studio specification.bindings[{index}]")
        target = (binding[0], binding[1])
        if binding[0] not in roles or target in binding_targets:
            raise LegacyStudioSpecEvidenceError("Historical Studio specification bindings are ambiguous.")
        binding_targets.add(target)

    for field, maximum in (("autoFields", _MAX_AUTO_FIELDS), ("actions", _MAX_ACTIONS)):
        values = spec.get(field)
        if not isinstance(values, list) or len(values) > maximum:
            raise LegacyStudioSpecEvidenceError(f"Historical Studio specification.{field} is invalid.")
    auto_fields = spec["autoFields"]
    if any(not isinstance(item, str) or not item for item in auto_fields) or len(auto_fields) != len(
        set(auto_fields)
    ):
        raise LegacyStudioSpecEvidenceError("Historical Studio specification.autoFields is invalid.")
    if spec["actions"]:
        raise LegacyStudioSpecEvidenceError(
            "Historical Studio specification.actions must be empty."
        )

    auxiliary = spec.get("auxiliaryTerminalRoles", [])
    if (
        not isinstance(auxiliary, list)
        or len(auxiliary) != len(set(auxiliary))
        or any(not isinstance(role, str) or role not in roles or role in outgoing for role in auxiliary)
    ):
        raise LegacyStudioSpecEvidenceError(
            "Historical Studio specification.auxiliaryTerminalRoles is invalid."
        )
    expected_hash = studio_spec_content_hash_v1(spec)
    if spec["contentHash"] != expected_hash:
        raise LegacyStudioSpecEvidenceError(
            "Historical Studio specification self-hash does not match its complete body."
        )
    if len(_canonical_json(spec).encode("utf-8")) > _MAX_SPECIFICATION_BYTES:
        raise LegacyStudioSpecEvidenceError("Historical Studio specification exceeds the size limit.")
    return deepcopy(spec)


def _spec_identity(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(value["id"]), str(value["contentHash"]), str(value["executionProfileId"]))


def validate_studio_spec_evidence_ledger(value: Any) -> dict[str, Any]:
    ledger = _exact_object(
        value,
        {"schemaVersion", "kind", "boundary", "sources", "specifications", "contentHash"},
        "Historical Studio-spec evidence ledger",
    )
    if (
        ledger.get("schemaVersion") != STUDIO_SPEC_EVIDENCE_SCHEMA_VERSION
        or ledger.get("kind") != "legacy_cluster_studio_spec_partial_evidence"
        or ledger.get("boundary") != _EVIDENCE_BOUNDARY
    ):
        raise LegacyStudioSpecEvidenceError("Historical Studio-spec evidence boundary or version is invalid.")

    sources_value = ledger.get("sources")
    if not isinstance(sources_value, list) or len(sources_value) > _MAX_CAPTURE_COUNT:
        raise LegacyStudioSpecEvidenceError("Historical Studio-spec evidence sources are invalid.")
    sources = []
    previous_source_id: str | None = None
    for index, raw in enumerate(sources_value):
        source = _exact_object(
            raw,
            {
                "sourceId",
                "artifactLabel",
                "compressedSha256",
                "compressedBytes",
                "uncompressedSha256",
                "uncompressedBytes",
                "selector",
                "candidateSpecificationCount",
            },
            f"Historical Studio-spec evidence sources[{index}]",
        )
        source_id = _text(source.get("sourceId"), f"sources[{index}].sourceId", pattern=_SOURCE_ID)
        if previous_source_id is not None and source_id <= previous_source_id:
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec sources are not uniquely ordered.")
        previous_source_id = source_id
        _text(source.get("artifactLabel"), f"sources[{index}].artifactLabel", maximum=512)
        _text(source.get("compressedSha256"), f"sources[{index}].compressedSha256", pattern=_SHA256)
        _text(source.get("uncompressedSha256"), f"sources[{index}].uncompressedSha256", pattern=_SHA256)
        _nonnegative_integer(source.get("compressedBytes"), f"sources[{index}].compressedBytes")
        _nonnegative_integer(source.get("uncompressedBytes"), f"sources[{index}].uncompressedBytes")
        _nonnegative_integer(
            source.get("candidateSpecificationCount"),
            f"sources[{index}].candidateSpecificationCount",
        )
        if source.get("selector") != _CAPABILITY_SELECTOR:
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec evidence selector is unsupported.")
        sources.append(deepcopy(source))
    source_ids = {source["sourceId"] for source in sources}

    specifications_value = ledger.get("specifications")
    if not isinstance(specifications_value, list) or len(specifications_value) > _MAX_SPECIFICATIONS:
        raise LegacyStudioSpecEvidenceError("Historical Studio-spec evidence specifications are invalid.")
    specifications = []
    previous_identity: tuple[str, str, str] | None = None
    for index, raw in enumerate(specifications_value):
        entry = _exact_object(
            raw,
            {"identity", "canonicalBodySha256", "specification", "sourceIds"},
            f"Historical Studio-spec evidence specifications[{index}]",
        )
        identity = _exact_object(
            entry.get("identity"),
            {"id", "contentHash", "executionProfileId"},
            f"specifications[{index}].identity",
        )
        normalized_identity = (
            _text(identity.get("id"), f"specifications[{index}].identity.id", maximum=512),
            _text(
                identity.get("contentHash"),
                f"specifications[{index}].identity.contentHash",
                pattern=_STUDIO_HASH,
            ),
            _text(
                identity.get("executionProfileId"),
                f"specifications[{index}].identity.executionProfileId",
                maximum=512,
            ),
        )
        if previous_identity is not None and normalized_identity <= previous_identity:
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec bodies are not uniquely ordered.")
        previous_identity = normalized_identity
        specification = validate_historical_studio_spec(entry.get("specification"))
        if normalized_identity != _spec_identity(specification):
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec identity does not match its body.")
        canonical_sha = _text(
            entry.get("canonicalBodySha256"),
            f"specifications[{index}].canonicalBodySha256",
            pattern=_SHA256,
        )
        if canonical_sha != _canonical_sha256(specification):
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec canonical SHA-256 is invalid.")
        refs = entry.get("sourceIds")
        if (
            not isinstance(refs, list)
            or not refs
            or any(not isinstance(item, str) or item not in source_ids for item in refs)
            or refs != sorted(set(refs))
        ):
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec source references are invalid.")
        specifications.append(
            {
                "identity": dict(zip(("id", "contentHash", "executionProfileId"), normalized_identity, strict=True)),
                "canonicalBodySha256": canonical_sha,
                "specification": specification,
                "sourceIds": list(refs),
            }
        )
    normalized = {
        "schemaVersion": STUDIO_SPEC_EVIDENCE_SCHEMA_VERSION,
        "kind": "legacy_cluster_studio_spec_partial_evidence",
        "boundary": deepcopy(_EVIDENCE_BOUNDARY),
        "sources": sources,
        "specifications": specifications,
        "contentHash": _text(
            ledger.get("contentHash"),
            "Historical Studio-spec evidence ledger.contentHash",
            pattern=_SHA256,
        ),
    }
    if normalized["contentHash"] != _content_hash(normalized):
        raise LegacyStudioSpecEvidenceError("Historical Studio-spec evidence ledger hash is invalid.")
    if len(_canonical_json(normalized).encode("utf-8")) > _MAX_LEDGER_BYTES:
        raise LegacyStudioSpecEvidenceError("Historical Studio-spec evidence ledger exceeds the size limit.")
    return normalized


def _studio_identity(value: Any, label: str) -> dict[str, str]:
    item = _exact_object(value, {"id", "contentHash", "executionProfileId"}, label)
    return {
        "id": _text(item.get("id"), f"{label}.id", maximum=512),
        "contentHash": _text(item.get("contentHash"), f"{label}.contentHash", pattern=_STUDIO_HASH),
        "executionProfileId": _text(
            item.get("executionProfileId"), f"{label}.executionProfileId", maximum=512
        ),
    }


def validate_studio_spec_partial_review_ledger(
    value: Any,
    *,
    evidence_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = _exact_object(
        value,
        {"schemaVersion", "kind", "boundary", "reviews", "contentHash"},
        "Historical Studio-spec partial-review ledger",
    )
    if (
        ledger.get("schemaVersion") != STUDIO_SPEC_REVIEW_SCHEMA_VERSION
        or ledger.get("kind") != "legacy_cluster_studio_spec_partial_reviews"
        or ledger.get("boundary") != _REVIEW_BOUNDARY
    ):
        raise LegacyStudioSpecEvidenceError("Historical Studio-spec review boundary or version is invalid.")
    known_evidence = None
    if evidence_ledger is not None:
        known_evidence = {
            (*_spec_identity(entry["specification"]), entry["canonicalBodySha256"])
            for entry in validate_studio_spec_evidence_ledger(evidence_ledger)["specifications"]
        }
    reviews_value = ledger.get("reviews")
    if not isinstance(reviews_value, list) or len(reviews_value) > 1_024:
        raise LegacyStudioSpecEvidenceError("Historical Studio-spec partial reviews are invalid.")
    reviews = []
    previous_id: str | None = None
    historical_keys: set[tuple[str, ...]] = set()
    hashes: set[str] = set()
    for index, raw in enumerate(reviews_value):
        review = _exact_object(
            raw,
            {
                "id",
                "historical",
                "evidence",
                "decision",
                "reviewer",
                "reviewedAt",
                "notes",
                "reviewHash",
            },
            f"Historical Studio-spec partial reviews[{index}]",
        )
        review_id = _text(review.get("id"), f"reviews[{index}].id", pattern=_REVIEW_ID)
        if previous_id is not None and review_id <= previous_id:
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec partial reviews are not uniquely ordered.")
        previous_id = review_id
        historical = _exact_object(
            review.get("historical"),
            {
                "manifestDefinitionId",
                "libraryRevision",
                "manifestContentHash",
                "executionAdmissionId",
                "studioExecutionSpec",
            },
            f"reviews[{index}].historical",
        )
        normalized_historical = {
            "manifestDefinitionId": _text(
                historical.get("manifestDefinitionId"),
                f"reviews[{index}].historical.manifestDefinitionId",
                maximum=512,
            ),
            "libraryRevision": _text(
                historical.get("libraryRevision"),
                f"reviews[{index}].historical.libraryRevision",
                pattern=_COMMIT,
            ),
            "manifestContentHash": _text(
                historical.get("manifestContentHash"),
                f"reviews[{index}].historical.manifestContentHash",
                pattern=_SHA256,
            ),
            "executionAdmissionId": _text(
                historical.get("executionAdmissionId"),
                f"reviews[{index}].historical.executionAdmissionId",
                maximum=1024,
            ),
            "studioExecutionSpec": _studio_identity(
                historical.get("studioExecutionSpec"),
                f"reviews[{index}].historical.studioExecutionSpec",
            ),
        }
        evidence = _exact_object(
            review.get("evidence"),
            {"canonicalBodySha256"},
            f"reviews[{index}].evidence",
        )
        canonical_sha = _text(
            evidence.get("canonicalBodySha256"),
            f"reviews[{index}].evidence.canonicalBodySha256",
            pattern=_SHA256,
        )
        studio = normalized_historical["studioExecutionSpec"]
        evidence_key = (studio["id"], studio["contentHash"], studio["executionProfileId"], canonical_sha)
        if known_evidence is not None and evidence_key not in known_evidence:
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec review references unknown evidence.")
        decision = review.get("decision")
        if decision not in {"verified_partial_evidence", "rejected_evidence"}:
            raise LegacyStudioSpecEvidenceError(
                "Historical Studio-spec partial review decision cannot authorize equivalence."
            )
        reviewed_at = _text(review.get("reviewedAt"), f"reviews[{index}].reviewedAt", maximum=64)
        try:
            parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec review timestamp is invalid.") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec review timestamp requires an offset.")
        normalized = {
            "id": review_id,
            "historical": normalized_historical,
            "evidence": {"canonicalBodySha256": canonical_sha},
            "decision": decision,
            "reviewer": _text(review.get("reviewer"), f"reviews[{index}].reviewer", maximum=256),
            "reviewedAt": reviewed_at,
            "notes": _text(review.get("notes"), f"reviews[{index}].notes", maximum=4096),
            "reviewHash": _text(review.get("reviewHash"), f"reviews[{index}].reviewHash", pattern=_SHA256),
        }
        expected_hash = _hash_omitting(normalized, "reviewHash")
        if normalized["reviewHash"] != expected_hash:
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec partial review hash is invalid.")
        historical_key = (
            normalized_historical["manifestDefinitionId"],
            normalized_historical["libraryRevision"],
            normalized_historical["manifestContentHash"],
            normalized_historical["executionAdmissionId"],
            studio["id"],
            studio["contentHash"],
            studio["executionProfileId"],
        )
        if historical_key in historical_keys or normalized["reviewHash"] in hashes:
            raise LegacyStudioSpecEvidenceError("Historical Studio-spec partial reviews contain duplicate authority.")
        historical_keys.add(historical_key)
        hashes.add(normalized["reviewHash"])
        reviews.append(normalized)
    normalized_ledger = {
        "schemaVersion": STUDIO_SPEC_REVIEW_SCHEMA_VERSION,
        "kind": "legacy_cluster_studio_spec_partial_reviews",
        "boundary": deepcopy(_REVIEW_BOUNDARY),
        "reviews": reviews,
        "contentHash": _text(
            ledger.get("contentHash"),
            "Historical Studio-spec partial-review ledger.contentHash",
            pattern=_SHA256,
        ),
    }
    if normalized_ledger["contentHash"] != _content_hash(normalized_ledger):
        raise LegacyStudioSpecEvidenceError("Historical Studio-spec partial-review ledger hash is invalid.")
    if len(_canonical_json(normalized_ledger).encode("utf-8")) > _MAX_LEDGER_BYTES:
        raise LegacyStudioSpecEvidenceError(
            "Historical Studio-spec partial-review ledger exceeds the size limit."
        )
    return normalized_ledger


def _strict_json_loads(raw: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LegacyStudioSpecEvidenceError(f"{label} contains a duplicate object key.")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                LegacyStudioSpecEvidenceError(f"{label} contains non-finite JSON value {value}.")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LegacyStudioSpecEvidenceError(f"{label} is not strict UTF-8 JSON.") from exc


def _capture_specifications(
    source_id: str,
    path: Path,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    if path.is_symlink() or not path.is_file():
        raise LegacyStudioSpecEvidenceError(f"Capture {source_id} is not a safe regular file.")
    size = path.stat().st_size
    if size > _MAX_COMPRESSED_BYTES or size != expected.get("compressedBytes"):
        raise LegacyStudioSpecEvidenceError(f"Capture {source_id} compressed size is invalid.")
    compressed = path.read_bytes()
    compressed_sha = "sha256:" + hashlib.sha256(compressed).hexdigest()
    if compressed_sha != expected.get("compressedSha256"):
        raise LegacyStudioSpecEvidenceError(f"Capture {source_id} compressed SHA-256 is invalid.")
    try:
        with gzip.open(path, "rb") as handle:
            raw = handle.read(_MAX_UNCOMPRESSED_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise LegacyStudioSpecEvidenceError(f"Capture {source_id} gzip stream is invalid.") from exc
    if len(raw) > _MAX_UNCOMPRESSED_BYTES or len(raw) != expected.get("uncompressedBytes"):
        raise LegacyStudioSpecEvidenceError(f"Capture {source_id} uncompressed size is invalid.")
    uncompressed_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    if uncompressed_sha != expected.get("uncompressedSha256"):
        raise LegacyStudioSpecEvidenceError(f"Capture {source_id} uncompressed SHA-256 is invalid.")
    document = _strict_json_loads(raw, f"Capture {source_id}")
    nodes = document.get("nodes") if isinstance(document, dict) else None
    capabilities = nodes.get("studioModelCapabilities") if isinstance(nodes, dict) else None
    if not isinstance(capabilities, list):
        raise LegacyStudioSpecEvidenceError(f"Capture {source_id} has no capability specification collection.")
    specifications: dict[tuple[str, str, str], dict[str, Any]] = {}
    candidate_count = 0
    for capability_index, capability in enumerate(capabilities):
        values = capability.get("studioExecutionSpecs") if isinstance(capability, dict) else None
        if not isinstance(values, list):
            raise LegacyStudioSpecEvidenceError(
                f"Capture {source_id} capability {capability_index} has no specification list."
            )
        for value in values:
            candidate_count += 1
            if candidate_count > _MAX_CAPTURE_SPECIFICATIONS:
                raise LegacyStudioSpecEvidenceError(f"Capture {source_id} has too many specifications.")
            specification = validate_historical_studio_spec(value)
            identity = _spec_identity(specification)
            previous = specifications.get(identity)
            if previous is not None and previous != specification:
                raise LegacyStudioSpecEvidenceError(
                    f"Capture {source_id} contains conflicting bodies for one Studio-spec identity."
                )
            specifications[identity] = specification
    metadata = {
        "sourceId": source_id,
        "artifactLabel": _text(expected.get("artifactLabel"), f"Capture {source_id} artifactLabel", maximum=512),
        "compressedSha256": compressed_sha,
        "compressedBytes": size,
        "uncompressedSha256": uncompressed_sha,
        "uncompressedBytes": len(raw),
        "selector": _CAPABILITY_SELECTOR,
        "candidateSpecificationCount": len(specifications),
    }
    return metadata, specifications


def _historical_studio_identities(
    inventory: Mapping[str, Any],
    current_library: Mapping[str, Any],
) -> set[tuple[str, str, str]]:
    current_tuples: set[tuple[str, ...]] = set()
    definitions = current_library.get("definitions")
    if not isinstance(definitions, list):
        raise LegacyStudioSpecEvidenceError("Current node library definitions are malformed.")
    for definition in definitions:
        if not isinstance(definition, Mapping):
            continue
        manifest = (
            definition.get("id"),
            definition.get("libraryRevision"),
            definition.get("contentHash"),
        )
        for admission in definition.get("executionAdmissions", []):
            if not isinstance(admission, Mapping):
                continue
            studio = admission.get("studioExecutionSpec")
            if not isinstance(studio, Mapping):
                continue
            complete = (*manifest, admission.get("id"), *_spec_identity(studio))
            if all(isinstance(item, str) and item for item in complete):
                current_tuples.add(complete)
    identities: set[tuple[str, str, str]] = set()
    workflows = inventory.get("workflows")
    if not isinstance(workflows, list):
        raise LegacyStudioSpecEvidenceError("Composite migration inventory workflows are malformed.")
    for workflow in workflows:
        if not isinstance(workflow, Mapping):
            continue
        for composite in workflow.get("composites", []):
            if not isinstance(composite, Mapping) or composite.get("classification") != "legacy_cluster_root":
                continue
            refs = composite.get("sourceRefs")
            cluster = refs.get("legacyCluster") if isinstance(refs, Mapping) else None
            definition = cluster.get("definition") if isinstance(cluster, Mapping) else None
            studio = cluster.get("studioExecutionSpec") if isinstance(cluster, Mapping) else None
            if not isinstance(definition, Mapping) or not isinstance(studio, Mapping):
                continue
            try:
                studio_identity = _spec_identity(studio)
            except KeyError:
                continue
            complete = (
                definition.get("id"),
                definition.get("libraryRevision"),
                definition.get("contentHash"),
                cluster.get("admissionId"),
                *studio_identity,
            )
            if any(not isinstance(item, str) or not item for item in complete):
                continue
            if tuple(complete) not in current_tuples:
                identities.add(studio_identity)
    return identities


def extract_studio_spec_evidence(
    capture_paths: Mapping[str, str | Path],
    inventory: Mapping[str, Any],
    current_library: Mapping[str, Any],
    *,
    source_expectations: Mapping[str, Mapping[str, Any]] = REVIEWED_CAPTURE_SOURCES,
) -> dict[str, Any]:
    """Extract a canonical evidence-only ledger from exact pinned captures."""

    if not capture_paths or len(capture_paths) > _MAX_CAPTURE_COUNT:
        raise LegacyStudioSpecEvidenceError("Capture source collection is invalid.")
    unknown = set(capture_paths) - set(source_expectations)
    if unknown:
        raise LegacyStudioSpecEvidenceError("Capture source collection contains an unreviewed source ID.")
    metadata = []
    bodies: dict[tuple[str, str, str], dict[str, Any]] = {}
    body_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for source_id in sorted(capture_paths):
        source_metadata, source_bodies = _capture_specifications(
            source_id,
            Path(capture_paths[source_id]),
            source_expectations[source_id],
        )
        metadata.append(source_metadata)
        for identity, body in source_bodies.items():
            previous = bodies.get(identity)
            if previous is not None and previous != body:
                raise LegacyStudioSpecEvidenceError(
                    "Reviewed captures contain conflicting bodies for one Studio-spec identity."
                )
            bodies[identity] = body
            body_sources[identity].add(source_id)
    selected = _historical_studio_identities(inventory, current_library)
    specifications = [
        {
            "identity": {
                "id": identity[0],
                "contentHash": identity[1],
                "executionProfileId": identity[2],
            },
            "canonicalBodySha256": _canonical_sha256(bodies[identity]),
            "specification": deepcopy(bodies[identity]),
            "sourceIds": sorted(body_sources[identity]),
        }
        for identity in sorted(selected.intersection(bodies))
    ]
    body = {
        "schemaVersion": STUDIO_SPEC_EVIDENCE_SCHEMA_VERSION,
        "kind": "legacy_cluster_studio_spec_partial_evidence",
        "boundary": deepcopy(_EVIDENCE_BOUNDARY),
        "sources": metadata,
        "specifications": specifications,
    }
    return validate_studio_spec_evidence_ledger({**body, "contentHash": _canonical_sha256(body)})


def empty_partial_review_ledger() -> dict[str, Any]:
    body = {
        "schemaVersion": STUDIO_SPEC_REVIEW_SCHEMA_VERSION,
        "kind": "legacy_cluster_studio_spec_partial_reviews",
        "boundary": deepcopy(_REVIEW_BOUNDARY),
        "reviews": [],
    }
    return {**body, "contentHash": _canonical_sha256(body)}


def _load_checked_in(path: Path, validator: Any) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_LEDGER_BYTES:
            raise LegacyStudioSpecEvidenceError(
                "Checked-in historical Studio-spec ledger is not a safe bounded file."
            )
        raw = path.read_bytes()
        if len(raw) > _MAX_LEDGER_BYTES:
            raise LegacyStudioSpecEvidenceError("Checked-in historical Studio-spec ledger is too large.")
        document = _strict_json_loads(raw, "Checked-in historical Studio-spec ledger")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LegacyStudioSpecEvidenceError("Checked-in historical Studio-spec ledger cannot be loaded.") from exc
    return validator(document)


@lru_cache(maxsize=1)
def _checked_in_evidence() -> dict[str, Any]:
    return _load_checked_in(STUDIO_SPEC_EVIDENCE_PATH, validate_studio_spec_evidence_ledger)


@lru_cache(maxsize=1)
def _checked_in_reviews() -> dict[str, Any]:
    evidence = _checked_in_evidence()
    return _load_checked_in(
        STUDIO_SPEC_REVIEW_PATH,
        lambda value: validate_studio_spec_partial_review_ledger(value, evidence_ledger=evidence),
    )


def reviewed_studio_spec_evidence() -> dict[str, Any]:
    """Return a detached copy of checked-in evidence-only Studio bodies."""

    return deepcopy(_checked_in_evidence())


def reviewed_studio_spec_partial_reviews() -> dict[str, Any]:
    """Return checked-in partial review decisions; never conversion authority."""

    return deepcopy(_checked_in_reviews())
