"""Evidence-bound disposition for Hugging Face Cluster catalog-only routes.

The node library intentionally keeps reviewed upstream structures visible even
when MoDiff cannot execute them yet.  This module joins those definitions to
their checked-in artifact/legal/runtime review without treating structural
discovery as an execution admission.  A definition disappears from this
ledger only in the same change that gives it an exact execution admission.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


CATALOG_ONLY_GATES_SCHEMA_VERSION = 1
CATALOG_ONLY_GATES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "huggingface-cluster-catalog-gates.v1.json"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MAX_LEDGER_BYTES = 64 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,127}$")


class HuggingFaceCatalogOnlyGateError(ValueError):
    """The checked-in catalog-only gate ledger is malformed or stale."""


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise HuggingFaceCatalogOnlyGateError(f"{label} has missing or unknown fields.")
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evidence_document(path: str, expected_sha256: str) -> dict[str, Any]:
    candidate = (_PROJECT_ROOT / path).resolve()
    data_root = (_PROJECT_ROOT / "data").resolve()
    if candidate.parent != data_root or candidate.suffix != ".json":
        raise HuggingFaceCatalogOnlyGateError("Catalog-only evidence must name one top-level data JSON file.")
    try:
        payload = candidate.read_bytes()
    except OSError as exc:
        raise HuggingFaceCatalogOnlyGateError(f"Catalog-only evidence {path} is unavailable.") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise HuggingFaceCatalogOnlyGateError(f"Catalog-only evidence {path} no longer matches its review hash.")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuggingFaceCatalogOnlyGateError(f"Catalog-only evidence {path} is not valid JSON.") from exc
    if not isinstance(document, dict):
        raise HuggingFaceCatalogOnlyGateError(f"Catalog-only evidence {path} must be an object.")
    return document


def validate_huggingface_catalog_only_gates(value: Any) -> dict[str, Any]:
    document = _exact_object(
        value,
        {"schemaVersion", "kind", "families", "contentHash"},
        "Hugging Face catalog-only gate ledger",
    )
    if (
        document["schemaVersion"] != CATALOG_ONLY_GATES_SCHEMA_VERSION
        or document["kind"] != "huggingface_cluster_catalog_only_execution_gates"
        or document["contentHash"]
        != _canonical_hash({key: item for key, item in document.items() if key != "contentHash"})
    ):
        raise HuggingFaceCatalogOnlyGateError("Hugging Face catalog-only gate ledger version or hash is invalid.")
    families = document["families"]
    if not isinstance(families, list) or not families or len(families) > 64:
        raise HuggingFaceCatalogOnlyGateError("Hugging Face catalog-only gate families are invalid.")
    if families != sorted(families, key=lambda item: str(item.get("id", ""))):
        raise HuggingFaceCatalogOnlyGateError("Hugging Face catalog-only gate families are not canonically ordered.")

    seen_ids: set[str] = set()
    seen_classes: set[str] = set()
    normalized = []
    for index, raw_family in enumerate(families):
        family = _exact_object(
            raw_family,
            {"id", "pipelineClasses", "evidencePath", "evidenceSha256", "blockers"},
            f"Hugging Face catalog-only gate family[{index}]",
        )
        family_id = family["id"]
        if not isinstance(family_id, str) or _TOKEN.fullmatch(family_id) is None or family_id in seen_ids:
            raise HuggingFaceCatalogOnlyGateError("Hugging Face catalog-only gate family identity is invalid.")
        seen_ids.add(family_id)
        classes = family["pipelineClasses"]
        blockers = family["blockers"]
        if (
            not isinstance(classes, list)
            or not classes
            or classes != sorted(classes)
            or len(classes) != len(set(classes))
            or any(not isinstance(item, str) or not item.endswith("Pipeline") for item in classes)
            or seen_classes.intersection(classes)
        ):
            raise HuggingFaceCatalogOnlyGateError("Catalog-only pipeline classes are invalid or duplicated.")
        seen_classes.update(classes)
        if (
            not isinstance(blockers, list)
            or not blockers
            or len(blockers) != len(set(blockers))
            or any(not isinstance(item, str) or _TOKEN.fullmatch(item) is None for item in blockers)
        ):
            raise HuggingFaceCatalogOnlyGateError("Catalog-only blocker codes are invalid or duplicated.")
        evidence_path = family["evidencePath"]
        evidence_sha256 = family["evidenceSha256"]
        if not isinstance(evidence_path, str) or not _SHA256.fullmatch(str(evidence_sha256)):
            raise HuggingFaceCatalogOnlyGateError("Catalog-only evidence binding is invalid.")
        evidence = _evidence_document(evidence_path, evidence_sha256)
        admission = evidence.get("admission")
        if not isinstance(admission, dict):
            raise HuggingFaceCatalogOnlyGateError(f"Catalog-only evidence {evidence_path} has no admission review.")
        unresolved = admission.get("unresolvedGates")
        if isinstance(unresolved, list) and unresolved != blockers:
            raise HuggingFaceCatalogOnlyGateError(
                f"Catalog-only blockers for {family_id} disagree with the evidence admission review."
            )
        if admission.get("runtimeCatalogExposed") is not False or admission.get("downloadCatalogExposed") is not False:
            raise HuggingFaceCatalogOnlyGateError(
                f"Catalog-only evidence for {family_id} unexpectedly exposes runtime or downloads."
            )
        normalized.append(deepcopy(family))
    return {
        "schemaVersion": CATALOG_ONLY_GATES_SCHEMA_VERSION,
        "kind": "huggingface_cluster_catalog_only_execution_gates",
        "families": normalized,
        "contentHash": document["contentHash"],
    }


@lru_cache(maxsize=1)
def _reviewed_huggingface_catalog_only_gates() -> dict[str, Any]:
    payload = CATALOG_ONLY_GATES_PATH.read_bytes()
    if len(payload) > _MAX_LEDGER_BYTES:
        raise HuggingFaceCatalogOnlyGateError("Hugging Face catalog-only gate ledger exceeds 64 KiB.")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HuggingFaceCatalogOnlyGateError("Hugging Face catalog-only gate ledger is not valid JSON.") from exc
    return validate_huggingface_catalog_only_gates(document)


def reviewed_huggingface_catalog_only_gates() -> dict[str, Any]:
    """Return a detached, hash-bound copy of the reviewed family gates."""

    return deepcopy(_reviewed_huggingface_catalog_only_gates())


def audit_huggingface_catalog_only_definitions(
    library: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join every non-admitted library definition to one reviewed gate family.

    The audit is deliberately binary: it reports zero executable candidates.
    A route with complete evidence belongs in the ordinary exact admission
    compiler, not in this ledger.
    """

    if library is None:
        from modiff.huggingface_node_library import build_huggingface_node_library

        library = build_huggingface_node_library()
    gates = reviewed_huggingface_catalog_only_gates()
    family_by_class = {
        pipeline_class: family
        for family in gates["families"]
        for pipeline_class in family["pipelineClasses"]
    }
    definitions = []
    for definition in library.get("definitions", []):
        if not isinstance(definition, Mapping) or definition.get("executionAdmissions"):
            continue
        pipeline_class = definition.get("pipelineClass")
        family = family_by_class.get(pipeline_class)
        if family is None:
            raise HuggingFaceCatalogOnlyGateError(
                f"Catalog-only definition {definition.get('id')} has no reviewed execution-gate family."
            )
        definitions.append(
            {
                "definitionId": definition.get("id"),
                "pipelineClass": pipeline_class,
                "workflowId": definition.get("workflowId"),
                "familyId": family["id"],
                "evidencePath": family["evidencePath"],
                "evidenceSha256": family["evidenceSha256"],
                "blockers": deepcopy(family["blockers"]),
                "status": "catalog_only_evidence_blocked",
            }
        )
    definitions.sort(key=lambda item: (str(item["pipelineClass"]), str(item["workflowId"])))
    used_classes = {str(item["pipelineClass"]) for item in definitions}
    stale_classes = sorted(set(family_by_class) - used_classes)
    if stale_classes:
        raise HuggingFaceCatalogOnlyGateError(
            "Catalog-only gate ledger contains pipeline classes that are no longer catalog-only: "
            + ", ".join(stale_classes)
            + "."
        )
    body = {
        "schemaVersion": CATALOG_ONLY_GATES_SCHEMA_VERSION,
        "kind": "huggingface_cluster_catalog_only_audit",
        "familyCount": len(gates["families"]),
        "definitionCount": len(definitions),
        "routeEligibleCount": 0,
        "definitions": definitions,
    }
    return {**body, "contentHash": _canonical_hash(body)}
