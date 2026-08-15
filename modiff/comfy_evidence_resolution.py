"""Resolve explicit task evidence in previously ambiguous Comfy records.

Only pinned catalog titles, ids, categories, and presentation-media hints are
used.  No Comfy graph, prompt, package, model, or media payload is opened.  The
result is semantic research evidence, never an execution or equivalence claim.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from modiff.comfy_research_contracts import validate_comfy_research_contract_ledger
from modiff.template_authoring_specs import load_template_authoring_spec_ledger


COMFY_EVIDENCE_RESOLUTION_SCHEMA_VERSION = 1
COMFY_EVIDENCE_RESOLUTION_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "research" / "comfy-evidence-resolution.v1.json"
)

_COMFY_CONTRACT_PATH = "data/research/comfy-research-contracts.v1.json"
_COMFY_CATALOG_PATH = "data/research/comfy-workflow-catalog.v1.json"
_WORKFLOW_MANIFEST_PATH = "data/workflow-library-manifest.json"
_UPSTREAM_COVERAGE_PATH = "data/upstream-coverage.v1.json"
_AUTHORING_SPEC_PATH = "data/template-authoring-specs.v1.json"
_MEDIA_KIND_ALIASES = {"text": "json", "three_d": "video"}
_FAMILY_MARKERS = {
    "ace-step": "ACE Audio",
    "chroma": "Chroma",
    "ernie image": "ERNIE Image",
    "flux": "FLUX Image",
    "kandinsky": "Kandinsky",
    "ltx": "LTX Video",
    "omnigen": "OmniGen",
    "qwen": "Qwen Image",
    "sdxl": "Stable Diffusion XL",
    "stable audio": "Stable Audio",
    "wan": "Wan Video",
    "z-image": "Z-Image",
    "z image": "Z-Image",
}
_BOUNDARY = {
    "researchOnly": True,
    "sourceFields": ["catalogId", "title", "categoryId", "presentationMediaHints"],
    "opensComfyGraphs": False,
    "importsComfyGraphs": False,
    "copiesComfyPrompts": False,
    "downloadsModelsOrMedia": False,
    "claimsExactModelOrWorkflowCompatibility": False,
    "publishesTemplates": False,
    "generatesAssets": False,
    "maximumClaim": "explicit_catalog_metadata_task_candidate",
}


class ComfyEvidenceResolutionError(RuntimeError):
    """Raised when evidence sources or the checked-in resolution drift."""


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
        raise ComfyEvidenceResolutionError(f"{label} is unavailable or invalid: {relative_path}") from error
    if not isinstance(value, dict):
        raise ComfyEvidenceResolutionError(f"{label} must be a JSON object.")
    return value, "sha256:" + _sha256_bytes(raw)


def _records(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ComfyEvidenceResolutionError(f"{label} must be a list of objects.")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComfyEvidenceResolutionError(f"{label} must be a non-empty string.")
    return value


def _contains(text: str, *patterns: str) -> bool:
    return any(pattern in text for pattern in patterns)


def _infer_task(title: str, catalog_id: str, category_id: str) -> tuple[str | None, str]:
    """Return a conservative task candidate and its metadata-only basis."""

    text = f"{title} {catalog_id}".casefold().replace("_", " ").replace("-", " ")
    category = category_id.casefold().replace("_", " ").replace("-", " ")

    if _contains(text, "audio generation"):
        return "text_to_audio", "explicit_title_audio_generation"
    if _contains(text, "image to gaussian splat", "image to 3d model", "image to model"):
        return "image_to_3d", "explicit_title_image_to_3d"
    if "geometry estimation" in text and "3d" in category:
        return "image_to_3d", "explicit_title_geometry_estimation_in_3d_category"
    if _contains(text, "video depth estimation"):
        return "video_depth_estimation", "explicit_title_video_depth_estimation"
    if _contains(text, "image depth estimation", "image to depth map", "depth estimation"):
        return "depth_estimation", "explicit_title_image_depth_estimation"
    if _contains(text, "video face detection"):
        return "video_face_detection", "explicit_title_video_face_detection"
    if _contains(text, "image face detection", "mediapipe: image face detection"):
        return "face_detection", "explicit_title_image_face_detection"
    if _contains(text, "video segmentation"):
        return "video_segmentation", "explicit_title_video_segmentation"
    if _contains(text, "image segmentation"):
        return "image_segmentation", "explicit_title_image_segmentation"
    if _contains(text, "video to pose map", "video multi person detection"):
        return "video_to_pose", "explicit_title_video_to_pose"
    if _contains(text, "image to pose map", "image multi person detection"):
        return "image_to_pose", "explicit_title_image_to_pose"
    if _contains(text, "remove background"):
        return "remove_background", "explicit_title_remove_background"
    if _contains(text, "image inpaint", "image inpainting"):
        return "inpaint", "explicit_title_image_inpaint"
    if _contains(text, "image outpaint", "image outpainting"):
        return "outpaint", "explicit_title_image_outpaint"
    if _contains(text, "image upscale", "upscale image", "latent upscale"):
        return "image_upscale", "explicit_title_image_upscale"
    if _contains(text, "canny to image", "depth to image", "pose to image", "controlnet") and "video" not in text:
        return "control_image", "explicit_title_control_to_image"
    if _contains(text, "any control to image", "depth control to image"):
        return "control_image", "explicit_title_control_to_image"
    if _contains(text, "reference image generation", "style reference", "redux model"):
        return "multi_image_reference_edit", "explicit_title_image_reference"
    if _contains(text, "image edit", "image editing", "illustration to realism", "light migration"):
        return "edit_image", "explicit_title_image_edit"
    if _contains(text, "text to image", "text-to-image", "image generation"):
        return "text_to_image", "explicit_title_text_to_image"
    if _contains(text, "flux.1 depth lora", "sd3.5 large depth"):
        return "control_image", "explicit_title_depth_conditioning"
    if "flux.1 dev onereward" in text:
        return "text_to_image", "image_category_model_generation_candidate"
    if "product mockup" in text or re.fullmatch(r".*\b(flux\.2 dev|hidream o1(?: dev)?)\b.*", text):
        return "text_to_image", "image_category_model_generation_candidate"

    if _contains(text, "canny to video", "depth to video", "pose to video"):
        return "control_to_video", "explicit_title_control_to_video"
    if _contains(text, "character replacement"):
        return "character_replace", "explicit_title_character_replace"
    if _contains(text, "motion transfer"):
        return "character_animate", "explicit_title_character_animation"
    if _contains(text, "first last frame to video", "first-last-frame to video"):
        return "first_last_frame_to_video", "explicit_title_first_last_frame_video"
    if _contains(text, "image to video"):
        return "image_to_video", "explicit_title_image_to_video"
    if _contains(text, "text to video"):
        return "text_to_video", "explicit_title_text_to_video"
    if _contains(text, "video inpaint"):
        return "video_inpaint", "explicit_title_video_inpaint"
    if _contains(text, "video upscale"):
        return "video_upscale", "explicit_title_video_upscale"
    if _contains(text, "video edit", "video editing"):
        return "edit_video", "explicit_title_video_edit"
    if _contains(text, "frame interpolation"):
        return "frame_interpolation", "explicit_title_frame_interpolation"
    if _contains(text, "get any video frame"):
        return "video_frame_extract", "explicit_title_video_frame_extract"
    if _contains(text, "merge videos", "video stitch", "multi keyframe video stitching"):
        return "video_stitch", "explicit_title_video_stitch"
    if _contains(text, "ic lora", "id lora") and "ltx" in text:
        return "reference_to_video", "explicit_title_reference_conditioned_video"

    if _contains(text, "prompt enhance"):
        return "text_generation", "explicit_title_prompt_enhancement"
    if _contains(text, "select per line text"):
        return "text_select", "explicit_title_text_selection"
    if _contains(text, "data type conversion"):
        return "data_conversion", "explicit_title_data_conversion"
    if _contains(text, "mask operations", "compositing"):
        return "mask_composite", "explicit_title_mask_composite"
    if _contains(text, "switch node"):
        return "graph_utility", "explicit_title_graph_utility"

    if "image tools" in category or "color adjustment" in text:
        if _contains(text, "crop images"):
            return "image_crop", "explicit_title_image_crop"
        if _contains(text, "split image grid"):
            return "image_tile", "explicit_title_image_tile"
        if _contains(text, "image channels"):
            return "image_channels", "explicit_title_image_channels"
        if _contains(
            text,
            "brightness and contrast",
            "color adjustment",
            "color balance",
            "color curves",
            "hue and saturation",
            "image levels",
        ):
            return "image_adjustment", "explicit_title_image_adjustment"
        if _contains(
            text,
            "chromatic aberration",
            "blur",
            "film grain",
            "glow",
            "sharpen",
            "unsharp mask",
        ):
            return "image_filter", "explicit_title_image_filter"

    return None, "insufficient_catalog_metadata"


def _base_workflows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _records(manifest.get("workflows"), label="canonical workflows")
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        model_type = _string(row.get("modelType"), label="canonical model type")
        mode = _string(row.get("mode"), label="canonical mode")
        workflow_id = _string(row.get("id"), label="canonical workflow id")
        for field in ("modelFamily", "mediaKind", "qualificationStatus"):
            _string(row.get(field), label=f"canonical workflow {workflow_id} {field}")
        key = (model_type, mode)
        current = by_pair.get(key)
        if current is None or workflow_id == f"{model_type}:{mode}":
            by_pair[key] = row
    return sorted(by_pair.values(), key=lambda item: item["id"])


def _public_templates(coverage: Mapping[str, Any]) -> dict[str, list[str]]:
    result = {}
    for row in _records(coverage.get("canonicalWorkflows"), label="coverage workflows"):
        workflow_id = _string(row.get("id"), label="coverage workflow id")
        values = row.get("publicTemplateIds")
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ComfyEvidenceResolutionError(f"Coverage template ids are invalid for {workflow_id}.")
        result[workflow_id] = sorted(values)
    return result


def _recognized_families(title: str, labels: list[str]) -> set[str]:
    evidence = " ".join([title, *labels]).casefold().replace("_", " ")
    return {family for marker, family in _FAMILY_MARKERS.items() if marker in evidence}


def _representative_options(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        current = by_family.get(row["modelFamily"])
        if current is None or row["id"] < current["id"]:
            by_family[row["modelFamily"]] = row
    return [
        {
            "canonicalWorkflowId": row["id"],
            "modelType": row["modelType"],
            "modelFamily": row["modelFamily"],
            "qualificationStatus": row["qualificationStatus"],
        }
        for _, row in sorted(by_family.items())
    ]


def build_comfy_evidence_resolution_ledger(root: Path) -> dict[str, Any]:
    """Resolve explicit task evidence for every undetermined research record."""

    root = root.resolve(strict=True)
    comfy, comfy_sha = _read_json(root, _COMFY_CONTRACT_PATH, label="Comfy research contracts")
    catalog, catalog_sha = _read_json(root, _COMFY_CATALOG_PATH, label="Comfy research catalog")
    manifest, manifest_sha = _read_json(root, _WORKFLOW_MANIFEST_PATH, label="workflow manifest")
    coverage, coverage_sha = _read_json(root, _UPSTREAM_COVERAGE_PATH, label="upstream coverage")
    authoring, authoring_sha = _read_json(root, _AUTHORING_SPEC_PATH, label="template authoring specs")
    if authoring != load_template_authoring_spec_ledger(root / _AUTHORING_SPEC_PATH, root=root):
        raise ComfyEvidenceResolutionError("The template authoring specification ledger is invalid.")
    manifest_rows = _records(manifest.get("workflows"), label="canonical workflows")
    workflow_ids = {_string(row.get("id"), label="canonical workflow id") for row in manifest_rows}
    validate_comfy_research_contract_ledger(
        comfy,
        catalog_ledger=catalog,
        catalog_ledger_sha256=catalog_sha.removeprefix("sha256:"),
        supported_workflow_ids=workflow_ids,
    )
    workflows = _base_workflows(manifest)
    public_by_workflow = _public_templates(coverage)
    authoring_by_workflow = {
        row["canonicalWorkflowId"]: row["id"]
        for row in _records(authoring.get("specifications"), label="template authoring specifications")
    }

    resolutions = []
    source_rows = [
        row
        for row in _records(comfy.get("contracts"), label="Comfy research contracts")
        if row.get("contractState") == "contract_undetermined"
    ]
    for row in source_rows:
        contract_id = _string(row.get("contractId"), label="Comfy research contract id")
        title = _string(row.get("title"), label=f"{contract_id} title")
        catalog_id = _string(row.get("catalogId"), label=f"{contract_id} catalog id")
        category_id = _string(row.get("categoryId"), label=f"{contract_id} category id")
        media_evidence = row.get("mediaEvidence")
        model_evidence = row.get("modelEvidence")
        source_review = row.get("sourceReview")
        if any(not isinstance(value, Mapping) for value in (media_evidence, model_evidence, source_review)):
            raise ComfyEvidenceResolutionError(f"Comfy evidence is incomplete for {contract_id}.")
        media_hints = media_evidence.get("catalogCategoryMediaHints")
        labels = model_evidence.get("catalogModelLabels")
        if not isinstance(media_hints, list) or any(not isinstance(value, str) or not value for value in media_hints):
            raise ComfyEvidenceResolutionError(f"Comfy media hints are invalid for {contract_id}.")
        if not isinstance(labels, list) or any(not isinstance(value, str) or not value for value in labels):
            raise ComfyEvidenceResolutionError(f"Comfy model labels are invalid for {contract_id}.")
        inferred_mode, basis = _infer_task(title, catalog_id, category_id)
        normalized_kinds = {_MEDIA_KIND_ALIASES.get(value, value) for value in media_hints}
        task_options = (
            [
                workflow
                for workflow in workflows
                if workflow["mode"] == inferred_mode and workflow["mediaKind"] in normalized_kinds
            ]
            if inferred_mode is not None
            else []
        )
        families = _recognized_families(title, labels)
        family_options = [workflow for workflow in task_options if workflow["modelFamily"] in families]
        if inferred_mode is None:
            resolution_state = "source_review_required"
            boundary_state = "task_undetermined"
            options = []
        elif family_options:
            resolution_state = "explicit_task_and_family_candidate"
            boundary_state = "existing_family_workflow_candidate"
            options = family_options
        elif task_options:
            resolution_state = "explicit_task_candidate"
            boundary_state = "existing_task_boundary_model_admission_required"
            options = task_options
        else:
            resolution_state = "explicit_new_task_candidate"
            boundary_state = "new_task_boundary_required"
            options = []
        representatives = _representative_options(options)
        recommendation = representatives[0] if representatives else None
        recommendation_id = recommendation["canonicalWorkflowId"] if recommendation else None
        blockers = [
            "catalog_metadata_only_no_graph_or_prompt_reviewed",
            "exact_model_component_and_rights_review_required",
            "execution_and_asset_quality_not_proven",
        ]
        if source_review.get("state") == "required":
            blockers.insert(0, "catalog_entry_source_review_required")
        if inferred_mode is None:
            blockers.append("task_and_model_evidence_still_insufficient")
        elif boundary_state == "new_task_boundary_required":
            blockers.extend(
                [
                    "new_bounded_modiff_task_contract_required",
                    "new_original_modiff_workflow_required",
                ]
            )
        else:
            blockers.append("metadata_inference_requires_independent_semantic_review")
        resolutions.append(
            {
                "contractId": contract_id,
                "sourceKind": row.get("sourceKind"),
                "catalogId": catalog_id,
                "title": title,
                "categoryId": category_id,
                "presentationMediaHints": deepcopy(media_hints),
                "catalogModelLabels": deepcopy(labels),
                "inferredMode": inferred_mode,
                "inferenceBasis": basis,
                "resolutionState": resolution_state,
                "taskBoundaryState": boundary_state,
                "recognizedMoDiffFamilies": sorted(families),
                "currentTaskBoundaryOptionCount": len(task_options),
                "currentFamilyOptionCount": len(family_options),
                "representativeWorkflowOptions": representatives,
                "recommendedWorkflow": (
                    {
                        **deepcopy(recommendation),
                        "publicTemplateIds": public_by_workflow.get(recommendation_id, []),
                        "authoringSpecId": authoring_by_workflow.get(recommendation_id),
                    }
                    if recommendation is not None
                    else None
                ),
                "blockers": blockers,
                "claims": {
                    "taskEvidenceConfirmedFromGraph": False,
                    "comfyGraphOpenedOrImported": False,
                    "comfyPromptCopied": False,
                    "exactModelSupported": False,
                    "recommendedWorkflowEquivalent": False,
                    "newMoDiffWorkflowAuthored": False,
                    "publicTemplateCreated": False,
                    "assetGenerated": False,
                    "assetApproved": False,
                },
            }
        )

    resolutions.sort(key=lambda item: item["contractId"])
    resolution_counts = Counter(item["resolutionState"] for item in resolutions)
    boundary_counts = Counter(item["taskBoundaryState"] for item in resolutions)
    mode_counts = Counter(
        item["inferredMode"] if item["inferredMode"] is not None else "unresolved" for item in resolutions
    )
    ledger: dict[str, Any] = {
        "schemaVersion": COMFY_EVIDENCE_RESOLUTION_SCHEMA_VERSION,
        "contractKind": "non_executable_comfy_evidence_resolution_ledger",
        "boundary": deepcopy(_BOUNDARY),
        "sources": {
            "comfyResearchContracts": {
                "path": _COMFY_CONTRACT_PATH,
                "sha256": comfy_sha,
                "contentHash": comfy.get("contentHash"),
            },
            "comfyResearchCatalog": {
                "path": _COMFY_CATALOG_PATH,
                "sha256": catalog_sha,
                "contentHash": catalog.get("contentHash"),
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
            "sourceUndeterminedCount": len(source_rows),
            "resolutionCount": len(resolutions),
            "resolutionStateCounts": {key: resolution_counts[key] for key in sorted(resolution_counts)},
            "taskBoundaryStateCounts": {key: boundary_counts[key] for key in sorted(boundary_counts)},
            "inferredModeCounts": {key: mode_counts[key] for key in sorted(mode_counts)},
            "recordsWithInferredTask": sum(item["inferredMode"] is not None for item in resolutions),
            "recordsStillUndetermined": sum(item["inferredMode"] is None for item in resolutions),
            "recordsWithRecommendedWorkflow": sum(item["recommendedWorkflow"] is not None for item in resolutions),
            "recordsWithPublicTemplateOption": sum(
                bool(item["recommendedWorkflow"] and item["recommendedWorkflow"]["publicTemplateIds"])
                for item in resolutions
            ),
            "recordsWithHiddenAuthoringSpecOption": sum(
                bool(item["recommendedWorkflow"] and item["recommendedWorkflow"]["authoringSpecId"])
                for item in resolutions
            ),
            "exactModelSupportClaims": 0,
            "newWorkflowClaims": 0,
            "assetClaims": 0,
        },
        "resolutions": resolutions,
    }
    ledger["contentHash"] = _content_hash(ledger)
    return ledger


def validate_comfy_evidence_resolution_ledger(
    ledger: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    if ledger.get("schemaVersion") != COMFY_EVIDENCE_RESOLUTION_SCHEMA_VERSION:
        raise ComfyEvidenceResolutionError("The Comfy evidence-resolution schema is unsupported.")
    if ledger.get("contractKind") != "non_executable_comfy_evidence_resolution_ledger":
        raise ComfyEvidenceResolutionError("Comfy evidence resolution must remain non-executable.")
    if ledger.get("contentHash") != _content_hash(ledger):
        raise ComfyEvidenceResolutionError("The Comfy evidence-resolution content hash is invalid.")
    expected = build_comfy_evidence_resolution_ledger(root)
    if dict(ledger) != expected:
        raise ComfyEvidenceResolutionError(
            "The Comfy evidence-resolution ledger is stale, incomplete, or internally inconsistent."
        )
    return expected


def render_comfy_evidence_resolution_ledger(ledger: Mapping[str, Any]) -> str:
    return json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_comfy_evidence_resolution_ledger(
    path: Path = COMFY_EVIDENCE_RESOLUTION_PATH,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComfyEvidenceResolutionError(
            "The checked-in Comfy evidence-resolution ledger is unavailable or invalid."
        ) from error
    if not isinstance(ledger, dict):
        raise ComfyEvidenceResolutionError("The Comfy evidence-resolution ledger must be an object.")
    return validate_comfy_evidence_resolution_ledger(
        ledger,
        root=root if root is not None else Path(__file__).resolve().parents[1],
    )
