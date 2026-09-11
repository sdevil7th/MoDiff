"""Fail-closed publication evidence audit for exact Hugging Face Cluster routes.

Static graph admission, a volatile per-run runtime qualification, a reviewed
output, a measured resource recipe, and Gallery publication are independent
authorities.  This module makes that separation inspectable without turning
local model state or family-level release measurements into public flags.

The checked-in resource coverage report keeps its historical model-family
recipe summary, but exact Cluster authority lives in a separate
``routeQualifications`` lane.  A route qualification binds two independently
captured measurements of one canonical workload to the current registered
``BlockDefinitionV2`` content hash and canonical SHA-256, admission, Studio
execution specification, immutable artifact, and complete dependency set.
Legacy family evidence remains useful release history but can never authorize
``autoEligible`` for an exact route.  Even an exact binding still requires an
explicit Auto publication authority before the immutable catalog flag may
change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from modiff.huggingface_cluster_promotions import (
    historical_promotion_receipt_for_admission,
    promotion_receipt_for_admission,
)
from modiff.huggingface_cluster_runtime import REGISTERED_BLOCK_V2_DEFINITION_PINS


RESOURCE_RECIPE_COVERAGE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "qualification"
    / "release"
    / "resource-recipe-coverage.v1.json"
)
RELEASE_CANDIDATE_REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "qualification"
    / "release"
    / "release-candidate-report.v1.json"
)

_RESOURCE_REPORT_HASH_PREFIX = "sha256:resource-recipe-coverage-v1:"
_RESOURCE_ROUTE_BINDING_HASH_PREFIX = "sha256:resource-route-binding-v1:"
_RESOURCE_RECIPE_HASH_PREFIX = "sha256:resource-recipe-v2:"
_CURRENT_RESOURCE_ROUTES_HASH_PREFIX = "sha256:current-resource-routes-v1:"
_RELEASE_REPORT_HASH_PREFIX = "sha256:release-candidate-report-v1:"
_BLOCK_DEFINITION_HASH = re.compile(r"^block-definition-v2-[a-f0-9]{8}$")
_SHA256 = re.compile(r"^sha256:[a-f0-9]{64}$")
_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_RESOURCE_RECIPE_FIELDS = {
    "modelType",
    "dtype",
    "offloadMode",
    "quantizationMode",
    "quantizedComponents",
    "autoOffload",
    "device",
    "deviceMap",
    "pipelineClass",
    "executionPath",
    "attentionBackend",
    "regionalCompile",
    "denoiserCache",
    "channelsLast",
    "layerwiseCasting",
    "recipeHash",
}
_EXACT_ROUTE_RUNTIME_CONTRACT_FIELDS = {
    "runtimeFingerprint",
    "runtimeLockFingerprint",
    "backendSourceFingerprint",
    "backendContractFingerprint",
    "deterministicFingerprint",
}
_EXACT_ROUTE_PROOF_FIELDS = {
    "taskId",
    "capturedAt",
    "graphHash",
    "outputHash",
    "outputCollectionHash",
    "executionDurationSeconds",
    "peakMemoryBytes",
    "peakReservedBytes",
    "processRssBytes",
    "backend",
    "device",
    "proofPath",
    "proofSha256",
}
_EXACT_ROUTE_EVIDENCE_FIELDS = {
    "bindingStatus",
    "evidenceSource",
    "proofCount",
    "taskIds",
    "lastSuccessAt",
    "proofs",
}


class ClusterPublicationAuditError(ValueError):
    """Raised when publication evidence is missing, malformed, or stale."""


def _canonical_hash(document: Mapping[str, Any], *, field: str, prefix: str) -> str:
    body = {key: value for key, value in document.items() if key != field}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return prefix + hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ClusterPublicationAuditError(f"{label} cannot be loaded.") from error
    if not isinstance(value, dict):
        raise ClusterPublicationAuditError(f"{label} is malformed.")
    return value


def _valid_resource_route_binding(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schemaVersion",
        "admissionId",
        "blockDefinition",
        "studioExecutionSpec",
        "artifact",
        "modelDependencies",
    }:
        return False
    block_definition = value.get("blockDefinition")
    studio_spec = value.get("studioExecutionSpec")
    artifact = value.get("artifact")
    dependencies = value.get("modelDependencies")
    return bool(
        value.get("schemaVersion") == 1
        and isinstance(value.get("admissionId"), str)
        and value["admissionId"]
        and isinstance(block_definition, dict)
        and set(block_definition) == {"definitionId", "contentHash", "canonicalSha256"}
        and block_definition.get("definitionId") == value["admissionId"]
        and isinstance(block_definition.get("contentHash"), str)
        and _BLOCK_DEFINITION_HASH.fullmatch(block_definition["contentHash"])
        and isinstance(block_definition.get("canonicalSha256"), str)
        and _SHA256.fullmatch(block_definition["canonicalSha256"])
        and isinstance(studio_spec, dict)
        and set(studio_spec) == {"id", "contentHash", "executionProfileId"}
        and all(isinstance(studio_spec.get(key), str) and studio_spec[key] for key in studio_spec)
        and isinstance(artifact, dict)
        and set(artifact) == {"repository", "revision"}
        and isinstance(artifact.get("repository"), str)
        and "/" in artifact["repository"]
        and isinstance(artifact.get("revision"), str)
        and _COMMIT.fullmatch(artifact["revision"])
        and isinstance(dependencies, list)
        and len(dependencies) <= 128
        and len({dependency.get("id") for dependency in dependencies if isinstance(dependency, dict)})
        == len(dependencies)
        and all(
            isinstance(dependency, dict)
            and set(dependency) == {"id", "kind", "repository", "revision"}
            and isinstance(dependency.get("id"), str)
            and dependency["id"]
            and isinstance(dependency.get("kind"), str)
            and dependency["kind"]
            and isinstance(dependency.get("repository"), str)
            and "/" in dependency["repository"]
            and isinstance(dependency.get("revision"), str)
            and _COMMIT.fullmatch(dependency["revision"])
            for dependency in dependencies
        )
        and dependencies
        == sorted(
            dependencies,
            key=lambda dependency: (
                dependency["id"],
                dependency["repository"],
                dependency["revision"],
            ),
        )
    )


def _valid_exact_route_recipe(recipe: Any) -> bool:
    if not isinstance(recipe, dict) or set(recipe) != _RESOURCE_RECIPE_FIELDS:
        return False
    recipe_hash = recipe.get("recipeHash")
    body = {key: value for key, value in recipe.items() if key != "recipeHash"}
    return bool(
        isinstance(recipe.get("modelType"), str)
        and recipe["modelType"]
        and isinstance(recipe.get("dtype"), str)
        and recipe["dtype"]
        and isinstance(recipe.get("offloadMode"), str)
        and recipe["offloadMode"]
        and isinstance(recipe.get("quantizationMode"), str)
        and recipe["quantizationMode"]
        and isinstance(recipe.get("quantizedComponents"), list)
        and all(isinstance(component, str) for component in recipe["quantizedComponents"])
        and all(
            isinstance(recipe.get(field), bool)
            for field in ("autoOffload", "regionalCompile", "channelsLast", "layerwiseCasting")
        )
        and isinstance(recipe_hash, str)
        and recipe_hash
        == _RESOURCE_RECIPE_HASH_PREFIX
        + hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _valid_nullable_positive_integer(value: Any) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value > 0)


def _valid_canonical_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc


def _valid_exact_route_runtime_contract(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == _EXACT_ROUTE_RUNTIME_CONTRACT_FIELDS
        and all(isinstance(value.get(field), str) and value[field] for field in value)
        and value["runtimeFingerprint"] == value["runtimeLockFingerprint"]
    )


def _valid_exact_route_proof(value: Any) -> bool:
    proof_path = value.get("proofPath") if isinstance(value, dict) else None
    return bool(
        isinstance(value, dict)
        and set(value) == _EXACT_ROUTE_PROOF_FIELDS
        and isinstance(value.get("taskId"), str)
        and value["taskId"]
        and _valid_canonical_utc_timestamp(value.get("capturedAt"))
        and isinstance(value.get("graphHash"), str)
        and value["graphHash"]
        and isinstance(value.get("outputHash"), str)
        and value["outputHash"]
        and isinstance(value.get("outputCollectionHash"), str)
        and value["outputCollectionHash"]
        and isinstance(value.get("executionDurationSeconds"), (int, float))
        and not isinstance(value["executionDurationSeconds"], bool)
        and value["executionDurationSeconds"] > 0
        and isinstance(value.get("peakMemoryBytes"), int)
        and not isinstance(value["peakMemoryBytes"], bool)
        and value["peakMemoryBytes"] > 0
        and _valid_nullable_positive_integer(value.get("peakReservedBytes"))
        and _valid_nullable_positive_integer(value.get("processRssBytes"))
        and isinstance(value.get("backend"), str)
        and value["backend"]
        and isinstance(value.get("device"), str)
        and value["device"]
        and isinstance(proof_path, str)
        and proof_path.startswith("data/qualification/release/route-resource-provenance/")
        and ".." not in Path(proof_path).parts
        and isinstance(value.get("proofSha256"), str)
        and _SHA256.fullmatch(value["proofSha256"])
    )


def _valid_exact_route_evidence(evidence: Any) -> bool:
    if not isinstance(evidence, dict) or set(evidence) != _EXACT_ROUTE_EVIDENCE_FIELDS:
        return False
    proofs = evidence.get("proofs")
    task_ids = evidence.get("taskIds")
    if (
        not isinstance(proofs, list)
        or len(proofs) != 2
        or not all(_valid_exact_route_proof(proof) for proof in proofs)
        or not isinstance(task_ids, list)
        or len(task_ids) != 2
        or task_ids != [proof["taskId"] for proof in proofs]
        or len(set(task_ids)) != 2
        or len({proof["capturedAt"] for proof in proofs}) != 2
        or len({proof["proofPath"] for proof in proofs}) != 2
        or len({proof["proofSha256"] for proof in proofs}) != 2
    ):
        return False
    return bool(
        evidence.get("bindingStatus") == "exact_route_bound_two_proof"
        and evidence.get("evidenceSource") == "route_resource_two_live_proofs"
        and evidence.get("proofCount") == 2
        and _valid_canonical_utc_timestamp(evidence.get("lastSuccessAt"))
        and evidence["lastSuccessAt"] == max(proof["capturedAt"] for proof in proofs)
    )


def validate_resource_recipe_coverage(document: Any) -> dict[str, Any]:
    """Validate the checked-in resource summary before using it as evidence."""

    if not isinstance(document, dict):
        raise ClusterPublicationAuditError("Resource recipe coverage is malformed.")
    legacy_fields = {
        "schemaVersion",
        "format",
        "generatedAt",
        "releaseContractHash",
        "status",
        "coverage",
        "recipes",
        "reportHash",
    }
    if set(document) not in (legacy_fields, legacy_fields | {"routeQualifications"}):
        raise ClusterPublicationAuditError("Resource recipe coverage has missing or unknown fields.")
    if (
        document.get("schemaVersion") != 1
        or document.get("format") != "modiff.resource-recipe-coverage.v1"
        or document.get("status") not in {"complete", "incomplete"}
        or document.get("reportHash")
        != _canonical_hash(document, field="reportHash", prefix=_RESOURCE_REPORT_HASH_PREFIX)
    ):
        raise ClusterPublicationAuditError("Resource recipe coverage identity or hash is invalid.")
    recipes = document.get("recipes")
    if not isinstance(recipes, list):
        raise ClusterPublicationAuditError("Resource recipe coverage recipes are malformed.")
    for recipe in recipes:
        if not isinstance(recipe, dict):
            raise ClusterPublicationAuditError("A resource recipe record is malformed.")
        if (
            not isinstance(recipe.get("modelType"), str)
            or not recipe.get("modelType")
            or recipe.get("status") not in {"qualified", "missing"}
            or not isinstance(recipe.get("templates"), list)
        ):
            raise ClusterPublicationAuditError("A resource recipe identity is malformed.")
        evidence = recipe.get("evidence")
        if recipe["status"] == "qualified" and not isinstance(evidence, dict):
            raise ClusterPublicationAuditError("A qualified resource recipe has no evidence.")
        if recipe["status"] == "missing" and evidence is not None:
            raise ClusterPublicationAuditError("A missing resource recipe cannot carry evidence.")
    route_qualifications = document.get("routeQualifications", [])
    if not isinstance(route_qualifications, list):
        raise ClusterPublicationAuditError("Resource route qualifications are malformed.")
    seen_route_recipe_keys: set[tuple[str, str]] = set()
    for qualification in route_qualifications:
        if not isinstance(qualification, dict) or set(qualification) != {
            "routeBinding",
            "routeBindingHash",
            "workloadHash",
            "modelSetHash",
            "recipe",
            "runtimeContract",
            "evidence",
        }:
            raise ClusterPublicationAuditError("A resource route qualification is malformed.")
        binding = qualification.get("routeBinding")
        binding_hash = qualification.get("routeBindingHash")
        workload_hash = qualification.get("workloadHash")
        model_set_hash = qualification.get("modelSetHash")
        recipe = qualification.get("recipe")
        runtime_contract = qualification.get("runtimeContract")
        evidence = qualification.get("evidence")
        if (
            not _valid_resource_route_binding(binding)
            or binding_hash
            != _RESOURCE_ROUTE_BINDING_HASH_PREFIX
            + hashlib.sha256(
                json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            or not isinstance(workload_hash, str)
            or not re.fullmatch(r"sha256:resource-workload-v1:[a-f0-9]{64}", workload_hash)
            or not isinstance(model_set_hash, str)
            or not re.fullmatch(r"sha256:model-set-v1:[a-f0-9]{64}", model_set_hash)
            or not _valid_exact_route_recipe(recipe)
            or not _valid_exact_route_runtime_contract(runtime_contract)
            or not _valid_exact_route_evidence(evidence)
        ):
            raise ClusterPublicationAuditError("A resource route qualification identity is invalid.")
        key = (binding_hash, recipe["recipeHash"])
        if key in seen_route_recipe_keys:
            raise ClusterPublicationAuditError("A resource route qualification is duplicated.")
        seen_route_recipe_keys.add(key)
    normalized = deepcopy(document)
    if "routeQualifications" not in normalized:
        normalized["routeQualifications"] = []
        normalized["reportHash"] = _canonical_hash(
            normalized,
            field="reportHash",
            prefix=_RESOURCE_REPORT_HASH_PREFIX,
        )
    return normalized


def validate_release_candidate_report(document: Any, *, release_contract_hash: str) -> dict[str, Any]:
    """Validate the global release result and bind it to resource coverage."""

    if not isinstance(document, dict):
        raise ClusterPublicationAuditError("Release candidate report is malformed.")
    if (
        document.get("schemaVersion") != 1
        or document.get("format") != "modiff.release-candidate-report.v1"
        or document.get("status") not in {"ready", "blocked"}
        or document.get("releaseContractHash") != release_contract_hash
        or not isinstance(document.get("blockers"), list)
        or document.get("reportHash")
        != _canonical_hash(document, field="reportHash", prefix=_RELEASE_REPORT_HASH_PREFIX)
    ):
        raise ClusterPublicationAuditError("Release candidate report identity, binding, or hash is invalid.")
    return deepcopy(document)


def checked_in_publication_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    resource = validate_resource_recipe_coverage(
        _read_json(RESOURCE_RECIPE_COVERAGE_PATH, label="Resource recipe coverage")
    )
    release = validate_release_candidate_report(
        _read_json(RELEASE_CANDIDATE_REPORT_PATH, label="Release candidate report"),
        release_contract_hash=str(resource["releaseContractHash"]),
    )
    return resource, release


def _reason(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _expected_resource_route_binding(admission: Mapping[str, Any]) -> dict[str, Any] | None:
    admission_id = str(admission.get("id") or "")
    definition_pin = REGISTERED_BLOCK_V2_DEFINITION_PINS.get(admission_id)
    if definition_pin is None:
        return None
    return {
        "schemaVersion": 1,
        "admissionId": admission_id,
        "blockDefinition": {
            # Registered BlockDefinitionV2 IDs are the exact admission IDs;
            # the catalog definition ID is retained independently in the
            # definition snapshot source.
            "definitionId": admission_id,
            "contentHash": definition_pin[0],
            "canonicalSha256": definition_pin[1],
        },
        "studioExecutionSpec": admission.get("studioExecutionSpec"),
        "artifact": {
            "repository": (admission.get("artifact") or {}).get("repo"),
            "revision": (admission.get("artifact") or {}).get("revision"),
        },
        "modelDependencies": sorted(
            [
                {
                    "id": dependency.get("id"),
                    "kind": dependency.get("kind"),
                    "repository": dependency.get("repo"),
                    "revision": dependency.get("revision"),
                }
                for dependency in admission.get("modelDependencies", ())
                if isinstance(dependency, Mapping)
            ],
            key=lambda dependency: (
                str(dependency["id"]),
                str(dependency["repository"]),
                str(dependency["revision"]),
            ),
        ),
    }


def current_resource_route_manifest(
    library: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic current V2 route identities for offline qualification.

    The Node qualification CLI consumes this projection instead of parsing the
    TypeScript route registry or inferring a route from model-family metadata.
    Only one current reviewed schema-v6 definition/admission may own an ID.
    """

    if library is None:
        from modiff.huggingface_node_library import reviewed_huggingface_node_library

        selected_library = reviewed_huggingface_node_library()
    else:
        selected_library = deepcopy(dict(library))
    routes: list[dict[str, Any]] = []
    seen_admissions: set[str] = set()
    for definition in selected_library.get("definitions", ()):
        if (
            not isinstance(definition, Mapping)
            or definition.get("schemaVersion") != 6
            or definition.get("ownership") != "library"
            or definition.get("mutable") is not False
            or not isinstance(definition.get("id"), str)
            or not isinstance(definition.get("contentHash"), str)
            or not isinstance(definition.get("libraryRevision"), str)
            or not isinstance(definition.get("pipelineClass"), str)
            or not isinstance(definition.get("workflowId"), str)
        ):
            continue
        for admission in definition.get("executionAdmissions", ()):
            if not isinstance(admission, Mapping):
                continue
            admission_id = admission.get("id")
            binding = _expected_resource_route_binding(admission)
            publication = admission.get("publication")
            if isinstance(admission_id, str) and admission_id and admission_id in seen_admissions:
                raise ClusterPublicationAuditError(
                    "Current resource-route manifest contains an ambiguous admission ID."
                )
            if (
                not isinstance(admission_id, str)
                or not admission_id
                or admission.get("definitionId") != definition["id"]
                or admission.get("status") != "admitted"
                or admission.get("claim") != "static_graph_contract_compatible"
                or admission.get("executable") is not False
                or not isinstance(publication, Mapping)
                or publication.get("readiness") != "graph_qualified"
                or publication.get("insertable") is not True
                or publication.get("executable") is not False
                or binding is None
                or not _valid_resource_route_binding(binding)
            ):
                continue
            seen_admissions.add(admission_id)
            binding_hash = _RESOURCE_ROUTE_BINDING_HASH_PREFIX + hashlib.sha256(
                json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            routes.append(
                {
                    "admissionId": admission_id,
                    "definition": {
                        "id": definition["id"],
                        "contentHash": definition["contentHash"],
                        "libraryRevision": definition["libraryRevision"],
                        "pipelineClass": definition["pipelineClass"],
                        "workflowId": definition["workflowId"],
                    },
                    "studioMode": admission.get("studioMode"),
                    "routeBinding": binding,
                    "routeBindingHash": binding_hash,
                }
            )
    routes.sort(key=lambda route: route["admissionId"])
    body = {
        "schemaVersion": 1,
        "format": "modiff.current-resource-routes.v1",
        "routes": routes,
    }
    body["manifestHash"] = _canonical_hash(
        body,
        field="manifestHash",
        prefix=_CURRENT_RESOURCE_ROUTES_HASH_PREFIX,
    )
    return body


def _require_current_resource_route_qualifications(
    document: Mapping[str, Any],
    *,
    library: Mapping[str, Any],
) -> None:
    admissions_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for definition in library.get("definitions", ()):
        if not isinstance(definition, Mapping):
            continue
        for admission in definition.get("executionAdmissions", ()):
            if not isinstance(admission, Mapping) or not isinstance(admission.get("id"), str):
                continue
            admissions_by_id.setdefault(admission["id"], []).append(admission)
    for qualification in document.get("routeQualifications", ()):
        binding = qualification.get("routeBinding") if isinstance(qualification, Mapping) else None
        admission_id = binding.get("admissionId") if isinstance(binding, Mapping) else None
        admissions = admissions_by_id.get(str(admission_id or ""), [])
        expected = _expected_resource_route_binding(admissions[0]) if len(admissions) == 1 else None
        if expected is None or binding != expected:
            raise ClusterPublicationAuditError(
                "A resource route qualification does not match one current pinned BlockDefinitionV2 admission."
            )


def _exact_route_resource_recipes(
    route_qualifications: list[dict[str, Any]],
    *,
    admission: Mapping[str, Any],
    model_type: str,
) -> list[dict[str, Any]]:
    expected_binding = _expected_resource_route_binding(admission)
    if expected_binding is None:
        return []
    matches = []
    for qualification in route_qualifications:
        if not isinstance(qualification, Mapping):
            continue
        # Never infer route identity from a candidate-id string, template, or
        # model family: modes within a family frequently share one resource
        # tuple. The complete registered V2 binding must match byte-for-byte.
        recipe = qualification.get("recipe")
        if (
            qualification.get("routeBinding") == expected_binding
            and isinstance(recipe, Mapping)
            and recipe.get("modelType") == model_type
        ):
            matches.append(deepcopy(dict(qualification)))
    return matches


def audit_cluster_publication_route(
    admission_id: str,
    *,
    library: Mapping[str, Any] | None = None,
    resource_coverage: Mapping[str, Any] | None = None,
    release_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit one exact route without granting any missing authority."""

    if library is None:
        from modiff.huggingface_node_library import reviewed_huggingface_node_library

        selected_library = reviewed_huggingface_node_library()
    else:
        selected_library = deepcopy(dict(library))
    if resource_coverage is None or release_report is None:
        checked_resource, checked_release = checked_in_publication_evidence()
        resource_coverage = resource_coverage or checked_resource
        release_report = release_report or checked_release
    resource = validate_resource_recipe_coverage(resource_coverage)
    _require_current_resource_route_qualifications(resource, library=selected_library)
    release = validate_release_candidate_report(
        release_report,
        release_contract_hash=str(resource["releaseContractHash"]),
    )

    matches = [
        (definition, admission)
        for definition in selected_library.get("definitions", ())
        if isinstance(definition, Mapping)
        for admission in definition.get("executionAdmissions", ())
        if isinstance(admission, Mapping) and admission.get("id") == admission_id
    ]
    if len(matches) != 1:
        raise ClusterPublicationAuditError("The exact Cluster admission is unavailable or ambiguous.")
    definition, admission = matches[0]
    publication = admission.get("publication")
    if not isinstance(publication, Mapping):
        raise ClusterPublicationAuditError("The exact Cluster publication contract is unavailable.")

    artifact = admission.get("artifact")
    spec = admission.get("studioExecutionSpec")
    promotion = (
        promotion_receipt_for_admission(
            admission_id,
            definition_id=str(definition.get("id") or ""),
            artifact=artifact,
            studio_execution_spec=spec,
        )
        if isinstance(artifact, Mapping) and isinstance(spec, Mapping)
        else None
    )
    historical_promotion = historical_promotion_receipt_for_admission(admission_id)
    live_proof = promotion is not None
    # Detect accidental bypasses in either direction.  The immutable catalog
    # must be a projection of the exact promotion ledger, not a parallel flag.
    if publication.get("liveProof") is not live_proof:
        raise ClusterPublicationAuditError(
            "The Cluster liveProof flag does not match its exact promotion receipt."
        )

    model_type = str(definition.get("pipelineClass") or "")
    family_recipes = [
        deepcopy(recipe)
        for recipe in resource.get("recipes", ())
        if isinstance(recipe, dict) and recipe.get("modelType") == model_type
    ]
    qualified_family_recipes = [recipe for recipe in family_recipes if recipe.get("status") == "qualified"]
    exact_route_recipes = _exact_route_resource_recipes(
        list(resource.get("routeQualifications", ())),
        admission=admission,
        model_type=model_type,
    )

    reasons = [
        _reason(
            "runtime_resource_admission_required",
            "Public executability remains false; each run still requires a current runtime, artifact, "
            "dependency, and resource authority.",
        )
    ]
    if historical_promotion is not None and not live_proof:
        reasons.append(
            _reason(
                "live_output_review_reapproval_required",
                "A legacy output approval is preserved for review history, but it did not bind the current "
                "BlockDefinitionV2 content hash and canonical SHA-256. A fresh exact-route approval is required.",
            )
        )
    elif not live_proof:
        reasons.append(
            _reason(
                "live_output_review_pending",
                "No approved visible-frontend output receipt matches this exact definition, artifact, "
                "and Studio execution specification.",
            )
        )
    if exact_route_recipes:
        reasons.append(
            _reason(
                "auto_publication_authority_missing",
                "An exact resource measurement exists, but no checked-in authority enables Auto for "
                "this immutable Cluster publication.",
            )
        )
    elif not family_recipes:
        reasons.append(
            _reason(
                "resource_recipe_declaration_missing",
                "The release contract declares no measured resource recipe for this model family.",
            )
        )
    elif not qualified_family_recipes:
        reasons.append(
            _reason(
                "resource_recipe_qualification_missing",
                "The declared model-family resource recipes have no accepted real-weight measurement.",
            )
        )
    else:
        reasons.append(
            _reason(
                "exact_route_resource_binding_missing",
                "Model-family resource evidence exists, but it is not bound to this admission, Studio "
                "execution-spec hash, and artifact revision.",
            )
        )
    reasons.extend(
        [
            _reason(
                "gallery_asset_publication_pending",
                "The approved output evidence remains local and has no immutable public asset receipt.",
            ),
            _reason(
                "gallery_rights_review_pending",
                "The route has no separate Gallery rights and publication approval receipt.",
            ),
        ]
    )

    flags = {
        "insertable": publication.get("insertable") is True,
        "executable": False,
        "liveProof": live_proof,
        "autoEligible": False,
        "galleryEligible": False,
        "releaseEligible": False,
    }
    if (
        publication.get("executable") is not False
        or publication.get("autoEligible") is not False
        or admission.get("executable") is not False
    ):
        raise ClusterPublicationAuditError("The Cluster publication illegally bypasses runtime or Auto gates.")

    return {
        "schemaVersion": 1,
        "kind": "huggingface_cluster_route_publication_audit",
        "definitionId": definition.get("id"),
        "admissionId": admission_id,
        "modelType": model_type,
        "studioMode": admission.get("studioMode"),
        "artifact": deepcopy(artifact),
        "studioExecutionSpec": deepcopy(spec),
        "flags": flags,
        "evidence": {
            "outputReview": {
                "status": (
                    "approved"
                    if promotion is not None
                    else "historical_non_authorizing"
                    if historical_promotion is not None
                    else "missing"
                ),
                "receiptId": (
                    promotion.get("id")
                    if promotion is not None
                    else historical_promotion.get("id")
                    if historical_promotion is not None
                    else None
                ),
            },
            "resourceRecipes": {
                "status": (
                    "exact_route_qualified"
                    if exact_route_recipes
                    else "family_only"
                    if qualified_family_recipes
                    else "unqualified"
                    if family_recipes
                    else "undeclared"
                ),
                "declaredFamilyCount": len(family_recipes),
                "qualifiedFamilyCount": len(qualified_family_recipes),
                "exactRouteQualifiedCount": len(exact_route_recipes),
                "reportHash": resource.get("reportHash"),
            },
            "globalReleaseCandidate": {
                "status": release.get("status"),
                "reportHash": release.get("reportHash"),
                "blockerIds": [
                    blocker.get("id")
                    for blocker in release.get("blockers", ())
                    if isinstance(blocker, Mapping)
                ],
            },
        },
        "blockers": reasons,
    }


def audit_current_cluster_publication_routes() -> list[dict[str, Any]]:
    """Return the current approved image/audio routes and unapproved Wan route."""

    admission_ids = (
        "diffusers.cluster-admission:QwenImageModularPipeline:text2image:mode:text_to_image",
        "diffusers.cluster-admission:MiniMaxMusic3ModularPipeline:default:workflow:official_top_level_blocks",
        "diffusers.cluster-admission:WanTI2VPipeline:text_to_video:mode:text_to_video",
    )
    resource, release = checked_in_publication_evidence()
    from modiff.huggingface_node_library import reviewed_huggingface_node_library

    library = reviewed_huggingface_node_library()
    return [
        audit_cluster_publication_route(
            admission_id,
            library=library,
            resource_coverage=resource,
            release_report=release,
        )
        for admission_id in admission_ids
    ]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact Hugging Face Cluster publication evidence.")
    parser.add_argument(
        "--print-current-resource-routes",
        action="store_true",
        help="Print the deterministic current BlockDefinitionV2 resource-route manifest.",
    )
    arguments = parser.parse_args()
    if arguments.print_current_resource_routes:
        print(json.dumps(current_resource_route_manifest(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return
    parser.error("Choose an audit operation.")


if __name__ == "__main__":
    _main()
