"""Fail-closed research contracts derived from the pinned official Comfy catalog.

The records in this module are authoring decisions, not executable workflow or
asset contracts.  They intentionally retain only catalog metadata evidence and
never import Comfy graphs, nodes, packages, models, or media.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from modiff.comfy_template_research import (
    PINNED_COMFY_WORKFLOW_TEMPLATES_COMMITTED_AT,
    PINNED_COMFY_WORKFLOW_TEMPLATES_REPOSITORY,
    PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION,
    ComfyTemplateResearchError,
    comfy_task_modes_for_tags,
    validate_comfy_template_research_ledger,
)


COMFY_RESEARCH_CONTRACT_SCHEMA_VERSION = 1
COMFY_RESEARCH_CATALOG_LEDGER_PATH = "data/research/comfy-workflow-catalog.v1.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXACT_MAPPING_STATUS = "existing_contract_candidate"
_RESEARCH_MAPPING_STATUS = "research_candidate"
_AMBIGUOUS_MAPPING_STATUSES = {
    "ambiguous_task_evidence",
    "ambiguous_task_or_model_evidence",
    "no_model_evidence",
}

_TASK_MODE_MEDIA_KINDS = {
    "audio_to_video": ("video",),
    "character_animate": ("video",),
    "character_replace": ("video",),
    "control_image": ("image",),
    "depth_estimation": ("image",),
    "edit_audio": ("audio",),
    "edit_image": ("image",),
    "edit_video": ("video",),
    "first_last_frame_to_video": ("video",),
    "frame_interpolation": ("video",),
    "image_to_3d": ("three_d",),
    "image_to_video": ("video",),
    "image_upscale": ("image",),
    "inpaint": ("image",),
    "layer_decomposition": ("image", "json"),
    "outpaint": ("image",),
    "reference_to_video": ("video",),
    "remove_background": ("image",),
    "speech_to_text": ("text",),
    "text_generation": ("text",),
    "text_to_3d": ("three_d",),
    "text_to_audio": ("audio",),
    "text_to_image": ("image",),
    "text_to_speech": ("audio",),
    "text_to_video": ("video",),
    "video_extension": ("video",),
    "video_inpaint": ("video",),
    "video_to_video": ("video",),
    "video_upscale": ("video",),
}

_BLUEPRINT_CATEGORY_MEDIA_HINTS = {
    "3D": ("three_d",),
    "Audio": ("audio",),
    "Conditioning & Preprocessors": ("image",),
    "Image Editing": ("image",),
    "Image generation and editing": ("image",),
    "Image Tools": ("image",),
    "Text Tools": ("text",),
    "Video generation and editing": ("video",),
    "Video Tools": ("video",),
}

_BOUNDARY = {
    "catalogMetadataOnly": True,
    "importsComfyGraphs": False,
    "copiesComfyNodesOrPackages": False,
    "downloadsModelsOrMedia": False,
    "executesComfyWorkflows": False,
    "publishesMoDiffTemplates": False,
    "generatesOrApprovesAssets": False,
    "executionClaim": "none",
    "assetClaim": "none",
}

_RECORD_CLAIMS = {
    "comfyGraphImported": False,
    "comfyWorkflowExecutable": False,
    "moDiffNodeSupportClaimedByRecord": False,
    "moDiffWorkflowSupportClaimedByRecord": False,
    "publicTemplateClaimedByRecord": False,
    "assetGenerated": False,
    "assetApproved": False,
}


class ComfyResearchContractError(ValueError):
    """The checked-in Comfy research-contract ledger is invalid."""


def _records(value: Any, *, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ComfyResearchContractError(f"{label} must be a list of objects.")
    return list(value)


def _strings(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ComfyResearchContractError(f"{label} must be a list of non-empty strings.")
    return list(dict.fromkeys(value))


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComfyResearchContractError(f"{label} must be a non-empty string.")
    return value


def _count(records: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def _category_maps(catalog: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    template_categories = {
        _string(item.get("id"), label="template category id"): item
        for item in _records(catalog.get("templateCategories"), label="template categories")
    }
    blueprint_categories = {
        _string(item.get("id"), label="blueprint category id"): item
        for item in _records(catalog.get("blueprintCategories"), label="blueprint categories")
    }
    return template_categories, blueprint_categories


def _task_evidence(record: Mapping[str, Any], *, source_kind: str) -> dict[str, Any]:
    tags = _strings(record.get("tags"), label=f"{source_kind} task tags")
    catalog_modes = list(comfy_task_modes_for_tags(tags))
    mapping = record.get("candidateMapping")
    mapped_mode = mapping.get("taskMode") if isinstance(mapping, Mapping) else None
    if mapped_mode is not None and (not isinstance(mapped_mode, str) or not mapped_mode):
        raise ComfyResearchContractError("Candidate mapping task mode must be a non-empty string.")

    if record.get("mappingStatus") == _EXACT_MAPPING_STATUS:
        state = "curated_exact_mapping"
    elif len(catalog_modes) == 1:
        state = "single_catalog_task"
    else:
        state = "ambiguous_or_missing"
    return {
        "state": state,
        "catalogTags": tags,
        "catalogTaskModes": catalog_modes,
        "selectedCandidateMode": mapped_mode,
    }


def _media_evidence(
    task_evidence: Mapping[str, Any],
    *,
    source_kind: str,
    category: Mapping[str, Any],
) -> dict[str, Any]:
    task_modes = list(task_evidence["catalogTaskModes"])
    selected_mode = task_evidence.get("selectedCandidateMode")
    if isinstance(selected_mode, str) and selected_mode not in task_modes:
        task_modes.append(selected_mode)
    output_kinds = sorted(
        {
            media_kind
            for task_mode in task_modes
            for media_kind in _TASK_MODE_MEDIA_KINDS.get(task_mode, ())
        }
    )
    category_title = _string(category.get("title"), label=f"{source_kind} category title")
    if source_kind == "template":
        presentation_type = category.get("type")
        if presentation_type is not None and (not isinstance(presentation_type, str) or not presentation_type):
            raise ComfyResearchContractError("Template catalog presentation media type is invalid.")
        category_hints = [presentation_type] if isinstance(presentation_type, str) else []
    else:
        presentation_type = None
        category_hints = list(_BLUEPRINT_CATEGORY_MEDIA_HINTS.get(category_title, ()))
    if output_kinds:
        state = "task_derived_candidate"
    elif category_hints:
        state = "catalog_category_only"
    else:
        state = "unresolved"
    return {
        "state": state,
        "catalogCategoryTitle": category_title,
        "catalogPresentationMediaType": presentation_type,
        "catalogCategoryMediaHints": category_hints,
        "candidateOutputMediaKinds": output_kinds,
    }


def _model_evidence(record: Mapping[str, Any], *, source_kind: str) -> dict[str, Any]:
    labels = _strings(record.get("models"), label=f"{source_kind} model labels")
    mapping = record.get("candidateMapping")
    if isinstance(mapping, Mapping):
        for label in _strings(mapping.get("sourceModelLabels"), label=f"{source_kind} mapped model labels"):
            if label not in labels:
                labels.append(label)
    return {
        "state": "catalog_labels_present" if labels else "missing",
        "catalogModelLabels": labels,
    }


def _exact_mapping(
    record: Mapping[str, Any],
    *,
    supported_workflow_ids: set[str],
) -> dict[str, Any] | None:
    if record.get("mappingStatus") != _EXACT_MAPPING_STATUS:
        return None
    mapping = record.get("candidateMapping")
    if not isinstance(mapping, Mapping):
        raise ComfyResearchContractError("Exact Comfy mapping status requires mapping evidence.")
    workflow_id = _string(mapping.get("moDiffWorkflowId"), label="mapped MoDiff workflow id")
    if workflow_id not in supported_workflow_ids:
        raise ComfyResearchContractError(f"Mapped MoDiff workflow {workflow_id!r} does not exist.")
    return {
        "state": "curated_semantic_comparison_candidate",
        "taskMode": _string(mapping.get("taskMode"), label="mapped task mode"),
        "sourceModelLabels": _strings(mapping.get("sourceModelLabels"), label="mapped model labels"),
        "moDiffWorkflowId": workflow_id,
    }


def _states(record: Mapping[str, Any], *, exact_mapping: Mapping[str, Any] | None) -> tuple[str, str]:
    source_review_required = record.get("disposition") == "source_review_required"
    mapping_status = record.get("mappingStatus")
    if exact_mapping is not None:
        return (
            "existing_modiff_contract_candidate",
            "source_review_before_semantic_comparison"
            if source_review_required
            else "semantic_comparison_pending",
        )
    if mapping_status == _RESEARCH_MAPPING_STATUS:
        return (
            "new_contract_not_authored",
            "source_review_before_contract_authoring"
            if source_review_required
            else "contract_authoring_required",
        )
    if mapping_status in _AMBIGUOUS_MAPPING_STATUSES:
        return "contract_undetermined", "evidence_resolution_required"
    raise ComfyResearchContractError(f"Unsupported non-ineligible mapping status {mapping_status!r}.")


def _blockers(
    record: Mapping[str, Any],
    *,
    exact_mapping: Mapping[str, Any] | None,
    task_evidence: Mapping[str, Any],
    media_evidence: Mapping[str, Any],
    model_evidence: Mapping[str, Any],
) -> list[str]:
    blockers = [
        "catalog_metadata_only",
        "model_artifact_rights_not_reviewed",
        "input_output_media_rights_not_reviewed",
        "execution_not_qualified_from_this_record",
        "asset_not_generated_or_reviewed_from_this_record",
    ]
    if record.get("disposition") == "source_review_required":
        blockers.append("catalog_entry_source_review_required")
    if record.get("requiresCustomNodes"):
        blockers.append("custom_node_dependencies_not_reviewed")
    if task_evidence["state"] == "ambiguous_or_missing":
        blockers.append("task_evidence_ambiguous_or_missing")
    if media_evidence["state"] != "task_derived_candidate":
        blockers.append("output_media_contract_not_resolved")
    if model_evidence["state"] == "missing":
        blockers.append("model_evidence_missing")
    if exact_mapping is None:
        blockers.extend(("bounded_modiff_contract_not_authored", "original_modiff_workflow_not_authored"))
    else:
        blockers.append("semantic_equivalence_not_reviewed")
    return blockers


def _authoring_requirements(
    record: Mapping[str, Any],
    *,
    exact_mapping: Mapping[str, Any] | None,
    task_evidence: Mapping[str, Any],
    model_evidence: Mapping[str, Any],
) -> list[str]:
    requirements = ["do_not_copy_or_execute_comfy_graph"]
    if record.get("disposition") == "source_review_required":
        requirements.append("complete_source_and_dependency_review")
    if record.get("requiresCustomNodes"):
        requirements.append("replace_or_independently_review_custom_node_dependencies")
    if task_evidence["state"] == "ambiguous_or_missing" or model_evidence["state"] == "missing":
        requirements.append("resolve_task_media_and_model_evidence")
    if exact_mapping is None:
        requirements.extend(
            (
                "select_supported_adapter_or_record_implementation_gap",
                "author_bounded_modiff_contract",
                "author_original_modiff_workflow",
            )
        )
    else:
        requirements.append("compare_catalog_semantics_to_existing_modiff_contract")
    requirements.extend(
        (
            "review_model_and_input_output_media_rights",
            "qualify_original_modiff_execution",
            "create_and_review_original_modiff_assets",
        )
    )
    return requirements


def _research_contract(
    record: Mapping[str, Any],
    *,
    source_kind: str,
    category: Mapping[str, Any],
    supported_workflow_ids: set[str],
) -> dict[str, Any]:
    catalog_id = _string(record.get("id"), label=f"Comfy {source_kind} id")
    disposition = _string(record.get("disposition"), label=f"Comfy {source_kind} disposition")
    if disposition == "ineligible":
        raise ComfyResearchContractError("Ineligible Comfy records cannot enter the research-contract ledger.")
    source_class = _string(record.get("sourceClass"), label=f"Comfy {source_kind} source class")
    task_evidence = _task_evidence(record, source_kind=source_kind)
    media_evidence = _media_evidence(
        task_evidence,
        source_kind=source_kind,
        category=category,
    )
    model_evidence = _model_evidence(record, source_kind=source_kind)
    exact_mapping = _exact_mapping(record, supported_workflow_ids=supported_workflow_ids)
    contract_state, decision_state = _states(record, exact_mapping=exact_mapping)
    source_reasons = _strings(record.get("admissionReasons"), label=f"Comfy {source_kind} source reasons")
    source_review_state = (
        "required" if disposition == "source_review_required" else "catalog_open_source_declaration_only"
    )
    entry: dict[str, Any] = {
        "contractId": f"comfy-research:{source_kind}:{catalog_id}",
        "sourceKind": source_kind,
        "catalogId": catalog_id,
        "categoryId": _string(record.get("categoryId"), label=f"Comfy {source_kind} category id"),
        "title": record.get("title") if isinstance(record.get("title"), str) else catalog_id,
        "catalogDisposition": disposition,
        "catalogMappingStatus": _string(
            record.get("mappingStatus"),
            label=f"Comfy {source_kind} mapping status",
        ),
        "contractState": contract_state,
        "decisionState": decision_state,
        "taskEvidence": task_evidence,
        "mediaEvidence": media_evidence,
        "modelEvidence": model_evidence,
        "sourceReview": {
            "state": source_review_state,
            "catalogSourceClass": source_class,
            "catalogOpenSourceDeclaration": record.get("openSource"),
            "reasons": source_reasons,
        },
        "rightsReview": {
            "state": "independent_review_required",
            "catalogMetadataLicense": "MIT",
            "comfyGraphReuseApproved": False,
            "runtimeDependencyRightsReviewed": False,
            "modelArtifactRightsReviewed": False,
            "inputMediaRightsReviewed": False,
            "outputAssetRightsReviewed": False,
        },
        "blockers": _blockers(
            record,
            exact_mapping=exact_mapping,
            task_evidence=task_evidence,
            media_evidence=media_evidence,
            model_evidence=model_evidence,
        ),
        "authoringRequirements": _authoring_requirements(
            record,
            exact_mapping=exact_mapping,
            task_evidence=task_evidence,
            model_evidence=model_evidence,
        ),
        "claims": dict(_RECORD_CLAIMS),
    }
    custom_nodes = _strings(record.get("requiresCustomNodes"), label=f"Comfy {source_kind} custom nodes")
    if custom_nodes:
        entry["catalogCustomNodeDependencies"] = custom_nodes
    if exact_mapping is not None:
        entry["exactMapping"] = exact_mapping
    return entry


def build_comfy_research_contract_ledger(
    *,
    catalog_ledger: Mapping[str, Any],
    catalog_ledger_sha256: str,
    supported_workflow_ids: set[str],
) -> dict[str, Any]:
    """Build contracts for every non-ineligible pinned catalog record."""

    if _SHA256_RE.fullmatch(catalog_ledger_sha256) is None:
        raise ComfyResearchContractError("Pinned Comfy catalog ledger hash is invalid.")
    try:
        validate_comfy_template_research_ledger(
            catalog_ledger,
            supported_workflow_ids=supported_workflow_ids,
        )
    except ComfyTemplateResearchError as exc:
        raise ComfyResearchContractError("Pinned Comfy catalog ledger is invalid.") from exc

    template_categories, blueprint_categories = _category_maps(catalog_ledger)
    contracts = []
    for source_kind, records, categories in (
        ("template", catalog_ledger["templates"], template_categories),
        ("blueprint", catalog_ledger["blueprints"], blueprint_categories),
    ):
        for record in records:
            if record.get("disposition") == "ineligible":
                continue
            category_id = _string(record.get("categoryId"), label=f"Comfy {source_kind} category id")
            category = categories.get(category_id)
            if category is None:
                raise ComfyResearchContractError(
                    f"Comfy {source_kind} {record.get('id')!r} references unknown category {category_id!r}."
                )
            contracts.append(
                _research_contract(
                    record,
                    source_kind=source_kind,
                    category=category,
                    supported_workflow_ids=supported_workflow_ids,
                )
            )
    contracts.sort(key=lambda item: item["contractId"])
    contract_ids = [item["contractId"] for item in contracts]
    if len(contract_ids) != len(set(contract_ids)):
        raise ComfyResearchContractError("Comfy research contract ids must be unique.")

    source = catalog_ledger["source"]
    template_contracts = [item for item in contracts if item["sourceKind"] == "template"]
    blueprint_contracts = [item for item in contracts if item["sourceKind"] == "blueprint"]
    exact_mappings = [item for item in contracts if "exactMapping" in item]
    return {
        "schemaVersion": COMFY_RESEARCH_CONTRACT_SCHEMA_VERSION,
        "contractKind": "non_executable_research_contract_ledger",
        "source": {
            "repository": PINNED_COMFY_WORKFLOW_TEMPLATES_REPOSITORY,
            "revision": PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION,
            "committedAt": PINNED_COMFY_WORKFLOW_TEMPLATES_COMMITTED_AT,
            "catalogMetadataLicense": source["license"],
            "catalogLedgerPath": COMFY_RESEARCH_CATALOG_LEDGER_PATH,
            "catalogLedgerSha256": catalog_ledger_sha256,
        },
        "boundary": dict(_BOUNDARY),
        "summary": {
            "contractCount": len(contracts),
            "templateContractCount": len(template_contracts),
            "blueprintContractCount": len(blueprint_contracts),
            "exactMappingCount": len(exact_mappings),
            "sourceKindCounts": _count(contracts, "sourceKind"),
            "catalogDispositionCounts": _count(contracts, "catalogDisposition"),
            "catalogMappingStatusCounts": _count(contracts, "catalogMappingStatus"),
            "contractStateCounts": _count(contracts, "contractState"),
            "decisionStateCounts": _count(contracts, "decisionState"),
        },
        "contracts": contracts,
    }


def validate_comfy_research_contract_ledger(
    ledger: Mapping[str, Any],
    *,
    catalog_ledger: Mapping[str, Any],
    catalog_ledger_sha256: str,
    supported_workflow_ids: set[str],
) -> dict[str, Any]:
    """Validate provenance, exhaustive coverage, boundaries, and determinism."""

    if ledger.get("schemaVersion") != COMFY_RESEARCH_CONTRACT_SCHEMA_VERSION:
        raise ComfyResearchContractError("Unsupported Comfy research-contract schema version.")
    if ledger.get("contractKind") != "non_executable_research_contract_ledger":
        raise ComfyResearchContractError("Comfy research contracts must remain non-executable.")
    if ledger.get("boundary") != _BOUNDARY:
        raise ComfyResearchContractError("Comfy research-contract execution or asset boundary changed.")
    source = ledger.get("source")
    if not isinstance(source, Mapping):
        raise ComfyResearchContractError("Comfy research-contract provenance is missing.")
    expected_source = {
        "repository": PINNED_COMFY_WORKFLOW_TEMPLATES_REPOSITORY,
        "revision": PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION,
        "committedAt": PINNED_COMFY_WORKFLOW_TEMPLATES_COMMITTED_AT,
        "catalogMetadataLicense": catalog_ledger.get("source", {}).get("license"),
        "catalogLedgerPath": COMFY_RESEARCH_CATALOG_LEDGER_PATH,
        "catalogLedgerSha256": catalog_ledger_sha256,
    }
    if dict(source) != expected_source:
        raise ComfyResearchContractError("Comfy research-contract provenance is stale or invalid.")
    expected = build_comfy_research_contract_ledger(
        catalog_ledger=catalog_ledger,
        catalog_ledger_sha256=catalog_ledger_sha256,
        supported_workflow_ids=supported_workflow_ids,
    )
    if dict(ledger) != expected:
        raise ComfyResearchContractError("Comfy research-contract ledger is stale or internally inconsistent.")
    return expected
