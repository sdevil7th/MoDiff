"""Deterministic authoring specifications for hidden template candidates.

This layer advances the canonical candidate ledger with original prompt drafts,
exact graph-default snapshots, input-selection briefs, and review criteria.  It
does not select third-party media, execute a workflow, generate an asset, grant
rights, or publish a template.
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

from modiff.template_candidate_contracts import load_template_candidate_contract_ledger


TEMPLATE_AUTHORING_SPEC_SCHEMA_VERSION = 1
TEMPLATE_AUTHORING_SPEC_PATH = Path(__file__).resolve().parents[1] / "data" / "template-authoring-specs.v1.json"

_WORKFLOW_MANIFEST_PATH = "data/workflow-library-manifest.json"
_CANDIDATE_PATH = "data/template-candidate-contracts.v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MEDIA_FIELDS = {
    "requiredImages": "image",
    "requiredVideos": "video",
    "requiredAudio": "audio",
}
_NO_PROMPT_MODES = {
    "depth_estimation",
    "image_adjustment",
    "image_channels",
    "image_crop",
    "image_filter",
    "image_upscale",
    "image_tile",
    "mask_composite",
    "speech_to_text",
    "speech_translation",
    "unconditional_image",
}
_AUTHORING_DEFAULT_PARAMS = {
    "adain_factor",
    "alpha_channel",
    "audio_duration",
    "batch_size",
    "bottom",
    "cfg_normalize",
    "chunk_length_seconds",
    "class_label",
    "conditioning_scale",
    "control_guidance_end",
    "control_guidance_start",
    "denoise_strength",
    "downscale",
    "eta",
    "feather",
    "fill_color",
    "fps",
    "frame_rate",
    "frame_size",
    "framepack_sampling",
    "generation_mode",
    "guidance_scale",
    "guidance_scale_2",
    "height",
    "high_threshold",
    "image_guidance_scale",
    "language",
    "latent_window_size",
    "layers",
    "left",
    "low_threshold",
    "match_input_resolution",
    "max_sequence_length",
    "motion_encode_batch_size",
    "num_frames",
    "num_images_per_prompt",
    "num_inference_steps",
    "num_waveforms",
    "overlap",
    "pag_adaptive_scale",
    "pag_scale",
    "prediction_kind",
    "previous_conditioning_frames",
    "processing_resolution",
    "prompt_segments_json",
    "reference_strength",
    "resolution",
    "right",
    "sample_rate",
    "scheduler_flow_shift",
    "seed",
    "segment_frame_length",
    "stable_audio_guidance",
    "stable_audio_steps",
    "strength",
    "stride_length_seconds",
    "task",
    "task_type",
    "temporal_overlap",
    "temporal_overlap_condition_strength",
    "temporal_tile_size",
    "timestamps",
    "top",
    "true_cfg_scale",
    "use_en_prompt",
    "use_guidance_scale_2",
    "width",
}
_BOUNDARY = {
    "studioVisible": False,
    "publicTemplate": False,
    "selectsInputAssets": False,
    "approvesModelOrMediaRights": False,
    "downloadsModelsOrMedia": False,
    "executesWorkflows": False,
    "generatesAssets": False,
    "approvesQuality": False,
    "importsComfyGraphs": False,
    "copiesComfyPrompts": False,
    "capturesCanonicalDefaults": True,
    "draftsOriginalPromptText": True,
    "maximumClaim": "authoring_spec_drafted",
}

_TEXT_TO_IMAGE_PROMPTS = (
    "Editorial product photograph of a translucent cobalt glass radio on pale limestone, soft window light, precise reflections, restrained blue and amber palette, clean negative space, realistic materials",
    "A quiet rain-soaked tram stop at blue hour, one red umbrella, wet pavement reflections, cinematic natural light, layered depth, believable urban details, no visible brand marks",
    "Botanical field-study illustration of six imaginary alpine flowers arranged on warm archival paper, delicate ink contours, subtle watercolor washes, clear spacing, museum catalog composition",
    "Sunlit modern reading room carved into warm sandstone, linen seating, mature olive tree, long geometric shadows, tactile architectural photography, calm human scale",
    "Handcrafted miniature harbor town inside an open walnut music box, tiny boats and lighthouse, macro photography, shallow depth of field, warm practical lights, intricate physical detail",
    "A documentary portrait of an elderly bicycle mechanic in a compact workshop, honest expression, weathered tools, soft side light, natural skin texture, unobtrusive composition",
)
_TEXT_TO_VIDEO_PROMPTS = (
    "A slow cinematic dolly through a quiet greenhouse just after rain; droplets slide from broad leaves, morning mist catches the light, and the camera movement remains smooth and physically coherent",
    "Wide coastal grassland at dusk as wind moves in visible waves; a lone cyclist crosses the frame, clouds drift naturally, exposure stays stable, and motion remains continuous",
    "Close-up of a ceramic artist shaping a bowl on a spinning wheel; hands move deliberately, wet clay retains its form, the camera makes a subtle arc, and temporal details remain consistent",
    "A small research boat moving through calm arctic water beneath low fog; ice fragments drift slowly, soft light changes gradually, and the horizon remains stable",
)
_TEXT_TO_AUDIO_PROMPTS = (
    "Instrumental downtempo electronic piece with brushed drums, warm analog bass, soft granular textures, and a restrained melodic arc; clean mix, gradual development, resolved ending",
    "Intimate chamber-folk cue led by fingerpicked acoustic guitar, muted cello, light hand percussion, and room ambience; natural dynamics, memorable motif, gentle ending",
    "Atmospheric science-documentary score with glassy mallets, low strings, subtle pulses, and spacious reverberation; evolving structure, no abrupt cuts, controlled loudness",
)
_MODE_PROMPTS = {
    "character_animate": "Animate the supplied character using the pose and face performances while preserving identity, clothing, proportions, background continuity, and natural temporal motion.",
    "character_replace": "Replace the performer with the supplied character while preserving the source staging, camera movement, background, timing, face performance, and clean mask boundaries.",
    "control_edit_image": "Restyle the source as a refined editorial photograph while preserving subject identity and composition and following the supplied control geometry exactly.",
    "control_image": "Create a detailed cinematic scene that follows the supplied control structure precisely, with coherent lighting, realistic materials, clean edges, and no unwanted text.",
    "control_inpaint": "Replace only the masked region, follow the supplied control geometry, and match the source perspective, lighting, texture, scale, and edge transitions.",
    "control_to_video": "Generate a temporally coherent cinematic shot that follows the supplied control motion and structure, with stable subjects, smooth movement, and consistent lighting.",
    "control_video_to_video": "Transform the source video while following the control video frame by frame; preserve timing and camera motion, keep subjects stable, and avoid flicker or geometry drift.",
    "edit_image": "Transform the supplied image into a polished editorial scene with warmer natural light and refined materials while preserving the main subject, pose, perspective, and recognizable composition.",
    "image_to_text": "Describe the supplied image accurately and concisely. Identify the main subject, setting, visible actions, composition, lighting, and any clearly legible text without guessing hidden details.",
    "image_to_video": "Animate the supplied image into a short cinematic shot with subtle camera movement, physically plausible subject motion, stable identity, coherent depth, and no sudden scene changes.",
    "inpaint": "Replace only the masked area with a believable matching element; preserve all unmasked pixels conceptually and match perspective, illumination, material texture, scale, and depth of field.",
    "layer_decomposition": "Separate the supplied image into an ordered, useful layer stack: foreground subject, distinct occluding elements, midground, and background, with clean alpha boundaries and complete visual reconstruction.",
    "multi_image_reference_edit": "Combine the primary subject and composition from the first reference with the material palette and supporting details from the remaining references; keep identities distinct and spatial relationships coherent.",
    "outpaint": "Extend the supplied image beyond its original frame with a seamless continuation of perspective, lighting, architecture or landscape, texture, and depth; do not duplicate the central subject.",
    "reference_to_video": "Create a coherent short cinematic shot guided by the supplied reference, preserving its subject identity, visual language, palette, and spatial relationships while adding natural motion.",
    "text_generation": "Write a concise production brief for a 20-second documentary shot about a community repairing a storm-damaged footbridge. Include setting, subject action, camera plan, sound cues, and a clear ending in five short bullet points.",
    "text_to_3d": "A compact mid-century table radio with rounded corners, a large tuning dial, woven speaker grille, four rubber feet, and clean watertight geometry suitable for a neutral turntable preview.",
    "video_to_video": "Restyle the supplied video as a restrained cinematic documentary while preserving timing, camera motion, subject identity, scene layout, and temporal continuity.",
}

_NEGATIVE_IMAGE = (
    "low resolution, distorted geometry, duplicate subjects, malformed hands, inconsistent reflections, "
    "watermark, logo, illegible text, oversharpening, compression artifacts"
)
_NEGATIVE_VIDEO = (
    "flicker, frame jitter, temporal inconsistency, warped motion, duplicate subjects, identity drift, "
    "camera jumps, watermark, illegible text, compression artifacts"
)


class TemplateAuthoringSpecError(RuntimeError):
    """Raised when authoring sources or the checked-in ledger drift."""


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
        raise TemplateAuthoringSpecError(f"{label} is unavailable or invalid: {relative_path}") from error
    if not isinstance(value, dict):
        raise TemplateAuthoringSpecError(f"{label} must be a JSON object.")
    return value, "sha256:" + _sha256_bytes(raw)


def _records(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TemplateAuthoringSpecError(f"{label} must be a list of objects.")
    return value


def _humanize(value: str) -> str:
    return " ".join(word.capitalize() for word in value.split("_") if word)


def _prompt_for(workflow_id: str, mode: str) -> str | None:
    if mode in _NO_PROMPT_MODES:
        return None
    if mode == "text_to_image":
        prompts = _TEXT_TO_IMAGE_PROMPTS
    elif mode == "text_to_video":
        prompts = _TEXT_TO_VIDEO_PROMPTS
    elif mode == "text_to_audio":
        prompts = _TEXT_TO_AUDIO_PROMPTS
    else:
        prompt = _MODE_PROMPTS.get(mode)
        if prompt is None:
            raise TemplateAuthoringSpecError(f"No original prompt draft is defined for mode {mode!r}.")
        return prompt
    index = int(_sha256_bytes(workflow_id.encode("utf-8"))[:8], 16) % len(prompts)
    return prompts[index]


def _canonical_graph_hash(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def _graph_default_snapshot(graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise TemplateAuthoringSpecError("Canonical graph nodes must be a list.")
    defaults: list[dict[str, Any]] = []
    has_negative_prompt = False
    for node in nodes:
        if not isinstance(node, Mapping):
            raise TemplateAuthoringSpecError("Canonical graph nodes must be objects.")
        data = node.get("data")
        if not isinstance(data, Mapping):
            continue
        role = data.get("studioRole")
        params = data.get("params")
        if not isinstance(role, str) or not isinstance(params, Mapping):
            continue
        if "negative_prompt" in params:
            has_negative_prompt = True
        for parameter in sorted(_AUTHORING_DEFAULT_PARAMS & set(params)):
            field = params[parameter]
            if not isinstance(field, Mapping) or "value" not in field:
                continue
            defaults.append(
                {
                    "role": role,
                    "parameter": parameter,
                    "value": deepcopy(field["value"]),
                }
            )
    defaults.sort(key=lambda item: (item["role"], item["parameter"]))
    return defaults, has_negative_prompt


def _input_items(required_inputs: Any, *, mode: str) -> list[dict[str, Any]]:
    if isinstance(required_inputs, list):
        if required_inputs:
            raise TemplateAuthoringSpecError("Legacy candidate inputs must be empty or use classified media fields.")
        return []
    if not isinstance(required_inputs, Mapping):
        raise TemplateAuthoringSpecError("Candidate required inputs must be a list or object.")
    items = []
    for field_name, media_kind in _MEDIA_FIELDS.items():
        fields = required_inputs.get(field_name, [])
        if not isinstance(fields, list) or any(not isinstance(field, str) or not field for field in fields):
            raise TemplateAuthoringSpecError(f"Candidate {field_name} must contain non-empty strings.")
        for field in fields:
            minimum_count = 2 if mode == "multi_image_reference_edit" and field == "referenceImages" else 1
            items.append(
                {
                    "field": field,
                    "mediaKind": media_kind,
                    "minimumCount": minimum_count,
                    "selectionState": "pending_original_or_licensed_asset",
                    "rightsState": "review_required",
                    "technicalRequirements": [
                        f"decodable_{media_kind}",
                        "matches_exact_workflow_input_contract",
                        "contains_no_unreviewed_brand_or_personal_data",
                    ],
                }
            )
    return items


def _component_requirements(required_inputs: Any) -> list[dict[str, Any]]:
    if not isinstance(required_inputs, Mapping):
        return []
    requirements = required_inputs.get("modelRequirements", [])
    if not isinstance(requirements, list) or any(not isinstance(item, Mapping) for item in requirements):
        raise TemplateAuthoringSpecError("Candidate model requirements must be a list of objects.")
    selected = []
    for item in requirements:
        row = {}
        for field in ("id", "kind", "label", "repo", "revision"):
            value = item.get(field)
            if not isinstance(value, str) or not value:
                raise TemplateAuthoringSpecError(f"Candidate model requirement field {field!r} is invalid.")
            row[field] = value
        selected.append(row)
    return selected


def _review_criteria(media_kind: str, mode: str) -> list[str]:
    criteria = {
        "image": [
            "expected_dimensions_and_decodable_output",
            "prompt_or_task_alignment",
            "coherent_geometry_lighting_and_materials",
            "no_unrequested_text_watermark_or_duplicate_subject",
        ],
        "video": [
            "expected_dimensions_duration_frame_rate_and_decodable_output",
            "prompt_or_task_alignment",
            "temporal_coherence_and_stable_subject_identity",
            "no_flicker_camera_jumps_or_unrequested_text",
        ],
        "audio": [
            "expected_duration_sample_rate_and_decodable_output",
            "prompt_alignment_and_coherent_structure",
            "clean_loudness_without_clipping_or_abrupt_cut",
            "no_unrequested_voice_or_copyrighted_source_imitation",
        ],
        "json": [
            "bounded_decodable_result",
            "instruction_or_source_alignment",
            "no_unsupported_claims_or_hidden_detail_invention",
            "receipt_matches_exact_input_and_runtime",
        ],
    }.get(media_kind)
    if criteria is None:
        raise TemplateAuthoringSpecError(f"Unsupported candidate media kind {media_kind!r}.")
    if mode in {
        "character_animate",
        "character_replace",
        "control_edit_image",
        "control_inpaint",
        "control_video_to_video",
        "edit_image",
        "image_to_video",
        "inpaint",
        "multi_image_reference_edit",
        "outpaint",
        "reference_to_video",
        "video_to_video",
    }:
        criteria.append("source_identity_structure_and_unmasked_region_preservation")
    if mode == "depth_estimation":
        criteria.append("relative_depth_order_and_edge_alignment")
    if mode == "layer_decomposition":
        criteria.append("ordered_layers_alpha_edges_and_reconstruction_completeness")
    return criteria


def _negative_prompt(media_kind: str, *, supported: bool) -> dict[str, Any]:
    if not supported or media_kind not in {"image", "video"}:
        return {"status": "not_supported_by_canonical_graph", "value": None}
    return {
        "status": "drafted",
        "value": _NEGATIVE_IMAGE if media_kind == "image" else _NEGATIVE_VIDEO,
    }


def build_template_authoring_spec_ledger(root: Path) -> dict[str, Any]:
    """Build original authoring drafts from exact hidden workflow contracts."""

    root = root.resolve(strict=True)
    candidate, candidate_sha = _read_json(root, _CANDIDATE_PATH, label="template candidate ledger")
    validated_candidate = load_template_candidate_contract_ledger(
        root / _CANDIDATE_PATH,
        root=root,
    )
    if candidate != validated_candidate:
        raise TemplateAuthoringSpecError("The template candidate ledger failed exact validation.")
    manifest, manifest_sha = _read_json(root, _WORKFLOW_MANIFEST_PATH, label="workflow manifest")
    manifest_rows = _records(manifest.get("workflows"), label="workflow manifest records")
    manifest_by_id = {row.get("id"): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows) or any(not isinstance(key, str) for key in manifest_by_id):
        raise TemplateAuthoringSpecError("Workflow manifest ids must be unique strings.")

    specifications = []
    for contract in _records(candidate.get("contracts"), label="template candidate contracts"):
        workflow_id = contract.get("canonicalWorkflowId")
        if not isinstance(workflow_id, str) or workflow_id not in manifest_by_id:
            raise TemplateAuthoringSpecError("Template candidate has no exact workflow manifest record.")
        workflow = manifest_by_id[workflow_id]
        graph_path = contract.get("graphPath")
        graph_hash = contract.get("graphHash")
        if not isinstance(graph_path, str) or not isinstance(graph_hash, str) or _SHA256.fullmatch(graph_hash) is None:
            raise TemplateAuthoringSpecError(f"Template candidate {workflow_id} has an invalid graph binding.")
        try:
            graph = json.loads((root / "data" / "graphs" / graph_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TemplateAuthoringSpecError(f"Canonical graph is unavailable for {workflow_id}.") from error
        if not isinstance(graph, dict) or _canonical_graph_hash(graph) != graph_hash:
            raise TemplateAuthoringSpecError(f"Canonical graph hash drifted for {workflow_id}.")

        model_type = contract.get("modelType")
        mode = contract.get("mode")
        media_kind = contract.get("mediaKind")
        model_family = workflow.get("modelFamily")
        if any(not isinstance(value, str) or not value for value in (model_type, mode, media_kind, model_family)):
            raise TemplateAuthoringSpecError(f"Template candidate identity is invalid for {workflow_id}.")
        prompt = _prompt_for(workflow_id, mode)
        defaults, has_negative_prompt = _graph_default_snapshot(graph)
        inputs = _input_items(contract.get("requiredInputs"), mode=mode)
        required_artifacts = contract.get("requiredArtifacts")
        if not isinstance(required_artifacts, list) or any(
            not isinstance(artifact, str) or not artifact for artifact in required_artifacts
        ):
            raise TemplateAuthoringSpecError(f"Template candidate artifacts are invalid for {workflow_id}.")
        built_in_artifacts = bool(required_artifacts) and all(
            artifact.startswith("builtin://") for artifact in required_artifacts
        )
        generation_blockers = [
            (
                "builtin_contract_identity_must_match_active_app"
                if built_in_artifacts
                else "exact_model_cache_state_must_be_rechecked"
            ),
            "exact_runtime_execution_receipt_required",
            "resource_recipe_receipt_required",
            "model_and_output_rights_review_required",
            "generated_asset_and_human_quality_review_required",
        ]
        if inputs:
            generation_blockers.insert(0, "reviewed_input_selection_and_rights_required")
        specifications.append(
            {
                "id": f"template-authoring:{workflow_id}",
                "canonicalWorkflowId": workflow_id,
                "modelType": model_type,
                "modelFamily": model_family,
                "mode": mode,
                "mediaKind": media_kind,
                "title": f"{model_family} — {_humanize(mode)} [{model_type}]",
                "purpose": (
                    f"Author and review one {media_kind} result for the exact {model_family} "
                    f"{_humanize(mode).lower()} workflow without changing its canonical graph contract."
                ),
                "canonicalGraph": {"path": graph_path, "sha256": graph_hash},
                "promptPlan": {
                    "status": "not_applicable" if prompt is None else "drafted",
                    "prompt": prompt,
                    "negativePrompt": _negative_prompt(media_kind, supported=has_negative_prompt),
                    "provenance": "original_modiff_mode_draft_v1",
                    "editableBeforeQualification": True,
                    "mustBeLockedInGenerationReceipt": prompt is not None,
                },
                "defaultsPlan": {
                    "status": "canonical_defaults_captured_review_pending",
                    "sourceGraphSha256": graph_hash,
                    "fields": defaults,
                    "requiresLockedSeedForGeneration": any(item["parameter"] == "seed" for item in defaults),
                },
                "inputPlan": {
                    "status": "selection_required" if inputs else "not_applicable",
                    "items": inputs,
                },
                "artifactPlan": {
                    "requiredArtifacts": deepcopy(required_artifacts),
                    "componentRequirements": _component_requirements(contract.get("requiredInputs")),
                    "cacheState": (
                        "not_applicable_builtin_contract"
                        if built_in_artifacts
                        else "must_be_rechecked_through_app_before_generation"
                    ),
                },
                "rightsPlan": {
                    "status": "review_required",
                    "modelArtifactRights": "review_or_existing_catalog_decision_required",
                    "inputMediaRights": "required" if inputs else "not_applicable",
                    "outputPublicationRights": "required",
                    "thirdPartyBrandAndPersonalDataReview": "required",
                },
                "outputPlan": {
                    "mediaKinds": [media_kind],
                    "minimumAssetCount": 1,
                    "reviewCriteria": _review_criteria(media_kind, mode),
                },
                "generationPlan": {
                    "status": "blocked_pending_external_evidence",
                    "requiresQualifiedAccelerator": media_kind in {"image", "video", "audio"},
                    "blockers": generation_blockers,
                },
                "authoringState": (
                    "draft_complete_input_selection_pending" if inputs else "draft_complete_execution_pending"
                ),
                "assetState": "not_generated",
                "assets": [],
                "publicationState": "hidden_candidate",
                "claims": {
                    "inputSelected": False,
                    "rightsApproved": False,
                    "workflowExecuted": False,
                    "assetGenerated": False,
                    "qualityApproved": False,
                    "publicTemplate": False,
                },
            }
        )

    specifications.sort(key=lambda item: item["canonicalWorkflowId"])
    state_counts = Counter(item["authoringState"] for item in specifications)
    ledger: dict[str, Any] = {
        "schemaVersion": TEMPLATE_AUTHORING_SPEC_SCHEMA_VERSION,
        "contractKind": "non_public_template_authoring_spec_ledger",
        "boundary": deepcopy(_BOUNDARY),
        "sources": {
            "candidateContracts": {
                "path": _CANDIDATE_PATH,
                "sha256": candidate_sha,
                "contentHash": candidate.get("contentHash"),
            },
            "workflowManifest": {
                "path": _WORKFLOW_MANIFEST_PATH,
                "sha256": manifest_sha,
            },
        },
        "summary": {
            "authoringSpecCount": len(specifications),
            "promptDraftedCount": sum(item["promptPlan"]["status"] == "drafted" for item in specifications),
            "promptNotApplicableCount": sum(
                item["promptPlan"]["status"] == "not_applicable" for item in specifications
            ),
            "canonicalDefaultsCapturedCount": len(specifications),
            "inputSelectionPendingCount": sum(bool(item["inputPlan"]["items"]) for item in specifications),
            "rightsReviewPendingCount": len(specifications),
            "generationPendingCount": len(specifications),
            "assetCount": sum(len(item["assets"]) for item in specifications),
            "authoringStateCounts": {key: state_counts[key] for key in sorted(state_counts)},
        },
        "specifications": specifications,
    }
    ledger["contentHash"] = _content_hash(ledger)
    return ledger


def validate_template_authoring_spec_ledger(
    ledger: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Validate the ledger and rebuild it against every current source."""

    if ledger.get("schemaVersion") != TEMPLATE_AUTHORING_SPEC_SCHEMA_VERSION:
        raise TemplateAuthoringSpecError("The template authoring specification schema is unsupported.")
    if ledger.get("contractKind") != "non_public_template_authoring_spec_ledger":
        raise TemplateAuthoringSpecError("Template authoring specifications must remain non-public.")
    if ledger.get("contentHash") != _content_hash(ledger):
        raise TemplateAuthoringSpecError("The template authoring specification content hash is invalid.")
    expected = build_template_authoring_spec_ledger(root)
    if dict(ledger) != expected:
        raise TemplateAuthoringSpecError(
            "The template authoring specification ledger is stale, incomplete, or internally inconsistent."
        )
    return expected


def render_template_authoring_spec_ledger(ledger: Mapping[str, Any]) -> str:
    return json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_template_authoring_spec_ledger(
    path: Path = TEMPLATE_AUTHORING_SPEC_PATH,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TemplateAuthoringSpecError(
            "The checked-in template authoring specification ledger is unavailable or invalid."
        ) from error
    if not isinstance(ledger, dict):
        raise TemplateAuthoringSpecError("The template authoring specification ledger must be an object.")
    return validate_template_authoring_spec_ledger(
        ledger,
        root=root if root is not None else Path(__file__).resolve().parents[1],
    )
