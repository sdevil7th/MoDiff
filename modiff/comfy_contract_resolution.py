"""Resolve Comfy research candidates against current MoDiff task boundaries.

The source research ledger deliberately does not infer runtime support.  This
second, still non-executable layer answers a narrower question for every
``new_contract_not_authored`` record: does MoDiff already have a same-family
task candidate, only a same-task boundary with another model, or no matching
task/output boundary at all?
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from modiff.comfy_research_contracts import validate_comfy_research_contract_ledger
from modiff.template_authoring_specs import load_template_authoring_spec_ledger


COMFY_CONTRACT_RESOLUTION_SCHEMA_VERSION = 1
COMFY_CONTRACT_RESOLUTION_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "research" / "comfy-contract-resolution.v1.json"
)

_COMFY_CONTRACT_PATH = "data/research/comfy-research-contracts.v1.json"
_COMFY_CATALOG_PATH = "data/research/comfy-workflow-catalog.v1.json"
_WORKFLOW_MANIFEST_PATH = "data/workflow-library-manifest.json"
_UPSTREAM_COVERAGE_PATH = "data/upstream-coverage.v1.json"
_AUTHORING_SPEC_PATH = "data/template-authoring-specs.v1.json"
_LABEL_TO_FAMILIES = {
    "ace-step": {"ACE Audio"},
    "chroma": {"Chroma"},
    "ernie-image": {"ERNIE Image"},
    "flux": {"FLUX Image"},
    "flux.2 dev": {"FLUX Image"},
    "flux.2 klein": {"FLUX Image"},
    "kandinsky": {"Kandinsky"},
    "lightricks": {"LTX Video"},
    "ltx-0.9.5": {"LTX Video"},
    "ltx-2": {"LTX Video"},
    "ltx-2.3": {"LTX Video"},
    "ltx-2.5": {"LTX Video"},
    "omnigen": {"OmniGen"},
    "qwen image edit 2511": {"Qwen Image"},
    "qwen image 2512": {"Qwen Image"},
    "qwen-image": {"Qwen Image"},
    "qwen-image-edit": {"Qwen Image"},
    "qwen-image-layered": {"Qwen Image"},
    "sd1.5": {"Stable Diffusion 1.x"},
    "sdxl": {"Stable Diffusion XL"},
    "stable audio": {"Stable Audio"},
    "wan": {"Wan Video"},
    "wan2.1": {"Wan Video"},
    "wan2.1 infinitetalk": {"Wan Video"},
    "wan2.2": {"Wan Video"},
    "z-image": {"Z-Image"},
}
_MEDIA_KIND_ALIASES = {"three_d": "video"}
_BOUNDARY = {
    "researchOnly": True,
    "importsComfyGraphs": False,
    "executesComfyNodes": False,
    "copiesComfyPrompts": False,
    "downloadsModelsOrMedia": False,
    "claimsExactCatalogCheckpointCompatibility": False,
    "claimsMoDiffWorkflowSupportFromCatalogMetadata": False,
    "publishesTemplates": False,
    "generatesAssets": False,
    "maximumClaim": "semantic_task_boundary_resolution",
}


class ComfyContractResolutionError(RuntimeError):
    """Raised when resolution sources or the checked-in ledger drift."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_hash(value: Mapping[str, Any]) -> str:
    semantic = dict(value)
    semantic.pop("contentHash", None)
    return "sha256:" + _sha256_bytes(_stable_json(semantic).encode("utf-8"))


def _read_json(root: Path, relative_path: str, *, label: str) -> tuple[dict[str, Any], str]:
    path = root / relative_path
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ComfyContractResolutionError(f"{label} is unavailable or invalid: {relative_path}") from error
    if not isinstance(value, dict):
        raise ComfyContractResolutionError(f"{label} must be a JSON object.")
    return value, "sha256:" + _sha256_bytes(raw)


def _records(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ComfyContractResolutionError(f"{label} must be a list of objects.")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComfyContractResolutionError(f"{label} must be a non-empty string.")
    return value


def _base_workflow_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    workflows = _records(manifest.get("workflows"), label="canonical workflows")
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in workflows:
        model_type = _string(row.get("modelType"), label="canonical model type")
        mode = _string(row.get("mode"), label="canonical mode")
        workflow_id = _string(row.get("id"), label="canonical workflow id")
        for field in ("modelFamily", "mediaKind", "graphHash", "qualificationStatus"):
            _string(row.get(field), label=f"canonical workflow {workflow_id} {field}")
        key = (model_type, mode)
        current = by_pair.get(key)
        if current is None or workflow_id == f"{model_type}:{mode}":
            by_pair[key] = row
    return sorted(by_pair.values(), key=lambda item: item["id"])


def _public_templates_by_workflow(coverage: Mapping[str, Any]) -> dict[str, list[str]]:
    rows = _records(coverage.get("canonicalWorkflows"), label="coverage canonical workflows")
    result = {}
    for row in rows:
        workflow_id = _string(row.get("id"), label="coverage workflow id")
        values = row.get("publicTemplateIds")
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ComfyContractResolutionError(f"Coverage public-template ids are invalid for {workflow_id}.")
        result[workflow_id] = sorted(values)
    return result


def _catalog_families(labels: list[str]) -> set[str]:
    families = set()
    for label in labels:
        families.update(_LABEL_TO_FAMILIES.get(label.casefold(), set()))
    return families


def _representative_options(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        family = row["modelFamily"]
        current = by_family.get(family)
        if current is None or row["id"] < current["id"]:
            by_family[family] = row
    return [
        {
            "canonicalWorkflowId": row["id"],
            "modelType": row["modelType"],
            "modelFamily": row["modelFamily"],
            "qualificationStatus": row["qualificationStatus"],
        }
        for _, row in sorted(by_family.items())
    ]


def build_comfy_contract_resolution_ledger(root: Path) -> dict[str, Any]:
    """Build an exhaustive resolution for all unresolved Comfy proposals."""

    root = root.resolve(strict=True)
    comfy, comfy_sha = _read_json(root, _COMFY_CONTRACT_PATH, label="Comfy research contracts")
    comfy_catalog, comfy_catalog_sha = _read_json(root, _COMFY_CATALOG_PATH, label="Comfy research catalog")
    manifest, manifest_sha = _read_json(root, _WORKFLOW_MANIFEST_PATH, label="workflow manifest")
    coverage, coverage_sha = _read_json(root, _UPSTREAM_COVERAGE_PATH, label="upstream coverage")
    authoring, authoring_sha = _read_json(root, _AUTHORING_SPEC_PATH, label="template authoring specs")
    if authoring != load_template_authoring_spec_ledger(root / _AUTHORING_SPEC_PATH, root=root):
        raise ComfyContractResolutionError("The template authoring specification ledger is invalid.")

    manifest_workflows = _records(manifest.get("workflows"), label="canonical workflows")
    supported_workflow_ids = {_string(row.get("id"), label="canonical workflow id") for row in manifest_workflows}
    validate_comfy_research_contract_ledger(
        comfy,
        catalog_ledger=comfy_catalog,
        catalog_ledger_sha256=comfy_catalog_sha.removeprefix("sha256:"),
        supported_workflow_ids=supported_workflow_ids,
    )
    workflows = _base_workflow_records(manifest)
    public_templates = _public_templates_by_workflow(coverage)
    authoring_by_workflow = {
        row["canonicalWorkflowId"]: row["id"]
        for row in _records(authoring.get("specifications"), label="template authoring specifications")
    }
    resolutions = []
    source_rows = [
        row
        for row in _records(comfy.get("contracts"), label="Comfy research contracts")
        if row.get("contractState") == "new_contract_not_authored"
    ]
    for row in source_rows:
        contract_id = _string(row.get("contractId"), label="Comfy research contract id")
        task_evidence = row.get("taskEvidence")
        media_evidence = row.get("mediaEvidence")
        model_evidence = row.get("modelEvidence")
        source_review = row.get("sourceReview")
        if any(
            not isinstance(value, Mapping) for value in (task_evidence, media_evidence, model_evidence, source_review)
        ):
            raise ComfyContractResolutionError(f"Comfy research evidence is incomplete for {contract_id}.")
        mode = _string(task_evidence.get("selectedCandidateMode"), label=f"{contract_id} selected mode")
        output_kinds = media_evidence.get("candidateOutputMediaKinds")
        labels = model_evidence.get("catalogModelLabels")
        if not isinstance(output_kinds, list) or any(
            not isinstance(value, str) or not value for value in output_kinds
        ):
            raise ComfyContractResolutionError(f"Comfy output media evidence is invalid for {contract_id}.")
        if not isinstance(labels, list) or any(not isinstance(value, str) or not value for value in labels):
            raise ComfyContractResolutionError(f"Comfy model evidence is invalid for {contract_id}.")
        normalized_output_kinds = {_MEDIA_KIND_ALIASES.get(value, value) for value in output_kinds}
        task_options = [
            workflow
            for workflow in workflows
            if workflow["mode"] == mode and workflow["mediaKind"] in normalized_output_kinds
        ]
        catalog_families = _catalog_families(labels)
        family_options = [workflow for workflow in task_options if workflow["modelFamily"] in catalog_families]
        if family_options:
            resolution_state = "existing_family_workflow_candidate"
            mapping_meaning = "task_and_family_semantic_candidate_only_exact_checkpoint_not_proven"
            options = family_options
        elif task_options:
            resolution_state = "existing_task_boundary_model_admission_required"
            mapping_meaning = "task_boundary_reuse_candidate_only_catalog_model_unmatched"
            options = task_options
        else:
            resolution_state = "new_task_boundary_required"
            mapping_meaning = "no_current_mode_and_output_boundary"
            options = []
        representatives = _representative_options(options)
        recommended = representatives[0] if representatives else None
        recommended_id = recommended["canonicalWorkflowId"] if recommended else None
        blockers = [
            "catalog_metadata_is_semantic_evidence_only",
            "exact_catalog_checkpoint_and_component_compatibility_not_proven",
            "model_artifact_and_input_output_rights_review_required",
            "execution_and_asset_quality_not_proven",
        ]
        if source_review.get("state") == "required":
            blockers.insert(0, "catalog_entry_source_review_required")
        if resolution_state == "existing_task_boundary_model_admission_required":
            blockers.append("exact_catalog_model_or_variant_admission_required")
        elif resolution_state == "new_task_boundary_required":
            blockers.extend(
                [
                    "new_bounded_modiff_task_contract_required",
                    "new_original_modiff_workflow_required",
                ]
            )
        else:
            blockers.append("family_match_requires_exact_checkpoint_and_variant_review")
        resolutions.append(
            {
                "contractId": contract_id,
                "sourceKind": row.get("sourceKind"),
                "catalogId": row.get("catalogId"),
                "title": row.get("title"),
                "selectedCandidateMode": mode,
                "candidateOutputMediaKinds": deepcopy(output_kinds),
                "catalogModelLabels": deepcopy(labels),
                "recognizedMoDiffFamilies": sorted(catalog_families),
                "resolutionState": resolution_state,
                "mappingMeaning": mapping_meaning,
                "currentTaskBoundaryOptionCount": len(task_options),
                "currentFamilyOptionCount": len(family_options),
                "representativeWorkflowOptions": representatives,
                "recommendedWorkflow": (
                    {
                        **deepcopy(recommended),
                        "publicTemplateIds": public_templates.get(recommended_id, []),
                        "authoringSpecId": authoring_by_workflow.get(recommended_id),
                    }
                    if recommended is not None
                    else None
                ),
                "exactCatalogModelReproductionRequiresAdmission": True,
                "blockers": blockers,
                "claims": {
                    "comfyGraphImported": False,
                    "comfyPromptCopied": False,
                    "exactCatalogCheckpointSupported": False,
                    "recommendedWorkflowEquivalent": False,
                    "newMoDiffWorkflowAuthored": False,
                    "publicTemplateCreated": False,
                    "assetGenerated": False,
                    "assetApproved": False,
                },
            }
        )

    resolutions.sort(key=lambda item: item["contractId"])
    state_counts = Counter(item["resolutionState"] for item in resolutions)
    mode_counts = Counter(item["selectedCandidateMode"] for item in resolutions)
    ledger: dict[str, Any] = {
        "schemaVersion": COMFY_CONTRACT_RESOLUTION_SCHEMA_VERSION,
        "contractKind": "non_executable_comfy_contract_resolution_ledger",
        "boundary": deepcopy(_BOUNDARY),
        "sources": {
            "comfyResearchContracts": {
                "path": _COMFY_CONTRACT_PATH,
                "sha256": comfy_sha,
                "contentHash": comfy.get("contentHash"),
            },
            "comfyResearchCatalog": {
                "path": _COMFY_CATALOG_PATH,
                "sha256": comfy_catalog_sha,
                "contentHash": comfy_catalog.get("contentHash"),
            },
            "workflowManifest": {"path": _WORKFLOW_MANIFEST_PATH, "sha256": manifest_sha},
            "upstreamCoverage": {
                "path": _UPSTREAM_COVERAGE_PATH,
                "sha256": coverage_sha,
                "contentHash": coverage.get("contentHash"),
            },
            "templateAuthoringSpecs": {
                "path": _AUTHORING_SPEC_PATH,
                "sha256": authoring_sha,
                "contentHash": authoring.get("contentHash"),
            },
        },
        "summary": {
            "sourceProposalCount": len(source_rows),
            "resolutionCount": len(resolutions),
            "resolutionStateCounts": {key: state_counts[key] for key in sorted(state_counts)},
            "modeCounts": {key: mode_counts[key] for key in sorted(mode_counts)},
            "recordsWithRecommendedWorkflow": sum(item["recommendedWorkflow"] is not None for item in resolutions),
            "recordsWithPublicTemplateOption": sum(
                bool(item["recommendedWorkflow"] and item["recommendedWorkflow"]["publicTemplateIds"])
                for item in resolutions
            ),
            "recordsWithHiddenAuthoringSpecOption": sum(
                bool(item["recommendedWorkflow"] and item["recommendedWorkflow"]["authoringSpecId"])
                for item in resolutions
            ),
            "exactCatalogCheckpointSupportClaims": 0,
            "newWorkflowClaims": 0,
            "assetClaims": 0,
        },
        "resolutions": resolutions,
    }
    ledger["contentHash"] = _content_hash(ledger)
    return ledger


def validate_comfy_contract_resolution_ledger(
    ledger: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    if ledger.get("schemaVersion") != COMFY_CONTRACT_RESOLUTION_SCHEMA_VERSION:
        raise ComfyContractResolutionError("The Comfy contract resolution schema is unsupported.")
    if ledger.get("contractKind") != "non_executable_comfy_contract_resolution_ledger":
        raise ComfyContractResolutionError("Comfy contract resolution must remain non-executable.")
    if ledger.get("contentHash") != _content_hash(ledger):
        raise ComfyContractResolutionError("The Comfy contract resolution content hash is invalid.")
    expected = build_comfy_contract_resolution_ledger(root)
    if dict(ledger) != expected:
        raise ComfyContractResolutionError(
            "The Comfy contract resolution ledger is stale, incomplete, or internally inconsistent."
        )
    return expected


def render_comfy_contract_resolution_ledger(ledger: Mapping[str, Any]) -> str:
    return json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_comfy_contract_resolution_ledger(
    path: Path = COMFY_CONTRACT_RESOLUTION_PATH,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComfyContractResolutionError(
            "The checked-in Comfy contract resolution ledger is unavailable or invalid."
        ) from error
    if not isinstance(ledger, dict):
        raise ComfyContractResolutionError("The Comfy contract resolution ledger must be an object.")
    return validate_comfy_contract_resolution_ledger(
        ledger,
        root=root if root is not None else Path(__file__).resolve().parents[1],
    )
