"""Strict evidence ledger for exact historical Cluster definition bodies.

The checked-in ledger contains only public node-library definition bodies and
collision-resistant receipts for the retained local transcript records from
which they were recovered.  It does not contain the transcript records
themselves, workflow paths, instance identifiers, or workflow values.

Registration here affects the read-only recovery audit only.  It is not a
compiler mapping, semantic-equivalence receipt, execution admission, or
workflow-mutation authority.  Preview, explicit apply, byte backup, and
rollback remain the only migration path.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


ARCHIVED_CLUSTER_DEFINITIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "legacy-cluster-archived-definitions.v1.json"
)
ARCHIVED_CLUSTER_DEFINITIONS_SCHEMA_VERSION = 1

_MAX_LEDGER_BYTES = 4 * 1024 * 1024
_MAX_DEFINITION_BYTES = 512 * 1024
_MAX_RECORDS = 256
_MAX_COLLECTION_ITEMS = 16_384
_MAX_DEPTH = 64

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,1023}$")
_STUDIO_HASH = re.compile(r"^studio-spec-v1-[0-9a-f]{8}$")

_BOUNDARY = {
    "checkedInHistoricalDefinitionEvidence": True,
    "doesNotInferFromLabelsOrModels": True,
    "requiresExactCompilerMapping": True,
    "doesNotAuthorizeConversion": True,
    "doesNotAuthorizeWorkflowMutation": True,
    "doesNotAuthorizeExecution": True,
    "containsTranscriptContents": False,
    "containsWorkflowPathsOrInstanceIds": False,
    "containsWorkflowPromptOrParameterValues": False,
}
_LEDGER_FIELDS = {
    "schemaVersion",
    "kind",
    "boundary",
    "records",
    "contentHash",
}
_RECORD_FIELDS = {"definition", "registration", "source", "recordHash"}
_REGISTRATION_FIELDS = {"status", "reason"}
_SOURCE_FIELDS = {
    "transcriptPath",
    "fetchRequestLine",
    "fetchRequestRecordBytes",
    "fetchRequestRecordSha256",
    "definitionResponseLine",
    "definitionResponseRecordBytes",
    "definitionResponseRecordSha256",
    "capturedAt",
}
_DEFINITION_FIELDS = {
    "blockContractHash",
    "blockPlacements",
    "blocksClass",
    "components",
    "contentHash",
    "definitionKind",
    "description",
    "executionAdmissions",
    "executionClaim",
    "graphAdapterContracts",
    "id",
    "inputs",
    "integrationStatus",
    "label",
    "libraryRevision",
    "mutable",
    "outputs",
    "ownership",
    "pipelineClass",
    "pipelineKind",
    "provider",
    "publisher",
    "requiredInputAlternatives",
    "requiredInputs",
    "rootBlockDefinitionId",
    "schemaVersion",
    "stateKeys",
    "steps",
    "surface",
    "taskContractId",
    "taskId",
    "workflowId",
    "workflowKind",
}
_REGISTERED_STATUS = "recovery_audit_registered"
_BODY_ONLY_STATUS = "archived_body_only"
_REGISTERED_REASON = "exact_definition_and_execution_admission_recovered"
_BODY_ONLY_REASON = "historical_execution_admission_absent"


class LegacyClusterArchivedDefinitionError(ValueError):
    """Raised when historical definition evidence is malformed or drifts."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _hash_omitting(value: Mapping[str, Any], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LegacyClusterArchivedDefinitionError(f"{label} has missing or unknown fields.")
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
        raise LegacyClusterArchivedDefinitionError(f"{label} is invalid.")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LegacyClusterArchivedDefinitionError(f"{label} must be a positive integer.")
    return value


def _json_value(value: Any, label: str, *, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise LegacyClusterArchivedDefinitionError(f"{label} is nested too deeply.")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > 9_007_199_254_740_991:
            raise LegacyClusterArchivedDefinitionError(f"{label} contains an unsafe integer.")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LegacyClusterArchivedDefinitionError(f"{label} contains a non-finite number.")
        return
    if isinstance(value, list):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise LegacyClusterArchivedDefinitionError(f"{label} is too large.")
        for index, item in enumerate(value):
            _json_value(item, f"{label}[{index}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise LegacyClusterArchivedDefinitionError(f"{label} is too large.")
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 1024:
                raise LegacyClusterArchivedDefinitionError(f"{label} contains an invalid key.")
            _json_value(item, f"{label}.{key}", depth=depth + 1)
        return
    raise LegacyClusterArchivedDefinitionError(f"{label} contains a non-JSON value.")


def _definition(value: Any, label: str) -> dict[str, Any]:
    definition = _exact_object(value, _DEFINITION_FIELDS, label)
    _json_value(definition, label)
    encoded = json.dumps(
        definition,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_DEFINITION_BYTES:
        raise LegacyClusterArchivedDefinitionError(f"{label} is too large.")
    _text(definition["id"], f"{label}.id", maximum=1024, pattern=_PUBLIC_ID)
    _text(
        definition["libraryRevision"],
        f"{label}.libraryRevision",
        maximum=40,
        pattern=_COMMIT,
    )
    _text(definition["contentHash"], f"{label}.contentHash", pattern=_SHA256)
    if definition["contentHash"] != _hash_omitting(definition, "contentHash"):
        raise LegacyClusterArchivedDefinitionError(
            f"{label}.contentHash does not match the complete historical body."
        )
    if (
        definition["provider"] not in {"diffusers", "transformers"}
        or definition["publisher"] != "huggingface"
        or definition["ownership"] != "library"
        or definition["mutable"] is not False
        or not isinstance(definition["schemaVersion"], int)
        or isinstance(definition["schemaVersion"], bool)
        or definition["schemaVersion"] not in {4, 5}
    ):
        raise LegacyClusterArchivedDefinitionError(f"{label} has an invalid immutable boundary.")
    admissions = definition["executionAdmissions"]
    if not isinstance(admissions, list) or len(admissions) > 64:
        raise LegacyClusterArchivedDefinitionError(f"{label}.executionAdmissions is invalid.")
    admission_ids: list[str] = []
    for index, admission in enumerate(admissions):
        if not isinstance(admission, dict):
            raise LegacyClusterArchivedDefinitionError(
                f"{label}.executionAdmissions[{index}] must be an object."
            )
        admission_id = _text(
            admission.get("id"),
            f"{label}.executionAdmissions[{index}].id",
            maximum=1024,
            pattern=_PUBLIC_ID,
        )
        studio = _exact_object(
            admission.get("studioExecutionSpec"),
            {"id", "contentHash", "executionProfileId"},
            f"{label}.executionAdmissions[{index}].studioExecutionSpec",
        )
        _text(studio["id"], f"{label}.executionAdmissions[{index}].studioExecutionSpec.id", pattern=_PUBLIC_ID)
        _text(
            studio["contentHash"],
            f"{label}.executionAdmissions[{index}].studioExecutionSpec.contentHash",
            pattern=_STUDIO_HASH,
        )
        _text(
            studio["executionProfileId"],
            f"{label}.executionAdmissions[{index}].studioExecutionSpec.executionProfileId",
            pattern=_PUBLIC_ID,
        )
        admission_ids.append(admission_id)
    if len(admission_ids) != len(set(admission_ids)):
        raise LegacyClusterArchivedDefinitionError(f"{label} repeats an execution admission.")
    return deepcopy(definition)


def _source(value: Any, label: str) -> dict[str, Any]:
    source = _exact_object(value, _SOURCE_FIELDS, label)
    transcript_path = _text(source["transcriptPath"], f"{label}.transcriptPath", maximum=512)
    path = PurePosixPath(transcript_path)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".jsonl":
        raise LegacyClusterArchivedDefinitionError(f"{label}.transcriptPath is unsafe.")
    captured_at = _text(source["capturedAt"], f"{label}.capturedAt", maximum=64)
    try:
        parsed_at = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LegacyClusterArchivedDefinitionError(
            f"{label}.capturedAt must be ISO-8601."
        ) from exc
    if parsed_at.tzinfo is None or parsed_at.utcoffset() is None:
        raise LegacyClusterArchivedDefinitionError(f"{label}.capturedAt must include an offset.")
    request_line = _positive_integer(source["fetchRequestLine"], f"{label}.fetchRequestLine")
    response_line = _positive_integer(
        source["definitionResponseLine"], f"{label}.definitionResponseLine"
    )
    if response_line <= request_line:
        raise LegacyClusterArchivedDefinitionError(
            f"{label}.definitionResponseLine must follow its fetch request line."
        )
    request_bytes = _positive_integer(
        source["fetchRequestRecordBytes"], f"{label}.fetchRequestRecordBytes"
    )
    response_bytes = _positive_integer(
        source["definitionResponseRecordBytes"],
        f"{label}.definitionResponseRecordBytes",
    )
    if request_bytes > 1024 * 1024 or response_bytes > 4 * 1024 * 1024:
        raise LegacyClusterArchivedDefinitionError(f"{label} record size is out of bounds.")
    request_hash = _text(
        source["fetchRequestRecordSha256"],
        f"{label}.fetchRequestRecordSha256",
        pattern=_SHA256,
    )
    response_hash = _text(
        source["definitionResponseRecordSha256"],
        f"{label}.definitionResponseRecordSha256",
        pattern=_SHA256,
    )
    return {
        "transcriptPath": transcript_path,
        "fetchRequestLine": request_line,
        "fetchRequestRecordBytes": request_bytes,
        "fetchRequestRecordSha256": request_hash,
        "definitionResponseLine": response_line,
        "definitionResponseRecordBytes": response_bytes,
        "definitionResponseRecordSha256": response_hash,
        "capturedAt": captured_at,
    }


def _registration(value: Any, label: str, *, admission_count: int) -> dict[str, str]:
    registration = _exact_object(value, _REGISTRATION_FIELDS, label)
    status = registration.get("status")
    reason = registration.get("reason")
    expected = {
        _REGISTERED_STATUS: _REGISTERED_REASON,
        _BODY_ONLY_STATUS: _BODY_ONLY_REASON,
    }
    if status not in expected or reason != expected[status]:
        raise LegacyClusterArchivedDefinitionError(f"{label} is invalid.")
    if (status == _REGISTERED_STATUS and admission_count == 0) or (
        status == _BODY_ONLY_STATUS and admission_count != 0
    ):
        raise LegacyClusterArchivedDefinitionError(
            f"{label} conflicts with the recovered execution admissions."
        )
    return {"status": status, "reason": reason}


def validate_archived_cluster_definition_ledger(value: Any) -> dict[str, Any]:
    """Validate exact bodies, source receipts, registration status, and hashes."""

    ledger = _exact_object(value, _LEDGER_FIELDS, "Archived Cluster definition ledger")
    if (
        ledger["schemaVersion"] != ARCHIVED_CLUSTER_DEFINITIONS_SCHEMA_VERSION
        or ledger["kind"] != "legacy_cluster_archived_definitions"
        or ledger["boundary"] != _BOUNDARY
    ):
        raise LegacyClusterArchivedDefinitionError(
            "Archived Cluster definition ledger boundary or version is invalid."
        )
    records_value = ledger["records"]
    if not isinstance(records_value, list) or len(records_value) > _MAX_RECORDS:
        raise LegacyClusterArchivedDefinitionError(
            "Archived Cluster definition record collection is invalid."
        )
    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(records_value):
        label = f"Archived Cluster definition record[{index}]"
        record = _exact_object(raw_record, _RECORD_FIELDS, label)
        definition = _definition(record["definition"], f"{label}.definition")
        source = _source(record["source"], f"{label}.source")
        registration = _registration(
            record["registration"],
            f"{label}.registration",
            admission_count=len(definition["executionAdmissions"]),
        )
        normalized = {
            "definition": definition,
            "registration": registration,
            "source": source,
            "recordHash": _text(
                record["recordHash"], f"{label}.recordHash", pattern=_SHA256
            ),
        }
        if normalized["recordHash"] != _hash_omitting(normalized, "recordHash"):
            raise LegacyClusterArchivedDefinitionError(
                f"{label}.recordHash does not match its complete record."
            )
        records.append(normalized)
    order = lambda item: (
        item["definition"]["id"],
        item["definition"]["libraryRevision"],
        item["definition"]["contentHash"],
    )
    if records != sorted(records, key=order):
        raise LegacyClusterArchivedDefinitionError(
            "Archived Cluster definition records must use canonical identity order."
        )
    identities = [order(record) for record in records]
    record_hashes = [record["recordHash"] for record in records]
    source_records = [
        (
            record["source"]["transcriptPath"],
            record["source"]["definitionResponseLine"],
            record["source"]["definitionResponseRecordSha256"],
        )
        for record in records
    ]
    if (
        len(identities) != len(set(identities))
        or len(record_hashes) != len(set(record_hashes))
        or len(source_records) != len(set(source_records))
    ):
        raise LegacyClusterArchivedDefinitionError(
            "Archived Cluster definition ledger contains duplicate evidence."
        )
    normalized_ledger = {
        "schemaVersion": ARCHIVED_CLUSTER_DEFINITIONS_SCHEMA_VERSION,
        "kind": "legacy_cluster_archived_definitions",
        "boundary": deepcopy(_BOUNDARY),
        "records": records,
        "contentHash": _text(
            ledger["contentHash"],
            "Archived Cluster definition ledger.contentHash",
            pattern=_SHA256,
        ),
    }
    if normalized_ledger["contentHash"] != _hash_omitting(
        normalized_ledger, "contentHash"
    ):
        raise LegacyClusterArchivedDefinitionError(
            "Archived Cluster definition ledger content hash does not match its complete body."
        )
    return normalized_ledger


@lru_cache(maxsize=1)
def _checked_in_archived_cluster_definition_ledger() -> dict[str, Any]:
    try:
        if ARCHIVED_CLUSTER_DEFINITIONS_PATH.is_symlink():
            raise LegacyClusterArchivedDefinitionError(
                "Archived Cluster definition ledger must not be a symlink."
            )
        raw = ARCHIVED_CLUSTER_DEFINITIONS_PATH.read_bytes()
        if len(raw) > _MAX_LEDGER_BYTES:
            raise LegacyClusterArchivedDefinitionError(
                "Archived Cluster definition ledger is too large."
            )
        document = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LegacyClusterArchivedDefinitionError(
            "Archived Cluster definition ledger cannot be loaded."
        ) from exc
    return validate_archived_cluster_definition_ledger(document)


def reviewed_archived_cluster_definition_evidence() -> dict[str, Any]:
    """Return every detached exact body, including non-convertible evidence."""

    return deepcopy(_checked_in_archived_cluster_definition_ledger())


def reviewed_archived_cluster_definitions() -> list[dict[str, Any]]:
    """Return only bodies registered for the read-only recovery audit."""

    return [
        deepcopy(record["definition"])
        for record in _checked_in_archived_cluster_definition_ledger()["records"]
        if record["registration"]["status"] == _REGISTERED_STATUS
    ]
