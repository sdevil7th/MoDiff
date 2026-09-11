"""Deterministic, non-executable research contracts for the official Comfy catalog.

This module deliberately inventories catalog metadata only.  It neither imports
Comfy workflow graphs nor turns Comfy nodes into MoDiff execution support.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


COMFY_TEMPLATE_RESEARCH_SCHEMA_VERSION = 1
PINNED_COMFY_WORKFLOW_TEMPLATES_REPOSITORY = "https://github.com/Comfy-Org/workflow_templates"
PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION = "d9e66019b85da231b7c936ad9cb7ff08cec16557"
PINNED_COMFY_WORKFLOW_TEMPLATES_COMMITTED_AT = "2026-08-14T13:03:04+05:00"

COMFY_SOURCE_FILES = (
    "LICENSE",
    "templates/index.json",
    "templates/index.schema.json",
    "blueprints/index.json",
    "blueprints/index.schema.json",
    "packages/core/src/comfyui_workflow_templates_core/manifest.json",
    "packages/core/src/comfyui_workflow_templates_core/blueprints_manifest.json",
)

_TASK_TAG_TO_MODE = {
    "Audio Editing": "edit_audio",
    "Audio to Video": "audio_to_video",
    "Character Replacement": "character_replace",
    "FLF2V": "first_last_frame_to_video",
    "Frame Interpolation": "frame_interpolation",
    "Image Edit": "edit_image",
    "Image to 3D": "image_to_3d",
    "Image to Model": "image_to_3d",
    "Image to Video": "image_to_video",
    "Image Upscale": "image_upscale",
    "Inpainting": "inpaint",
    "Layer Decompose": "layer_decomposition",
    "Outpainting": "outpaint",
    "Reference to Video": "reference_to_video",
    "Remove Background": "remove_background",
    "Speech to Text": "speech_to_text",
    "Text Generation": "text_generation",
    "Text to Audio": "text_to_audio",
    "Text to Image": "text_to_image",
    "Text to Model": "text_to_3d",
    "Text to Music": "text_to_audio",
    "Text to Speech": "text_to_speech",
    "Text to Video": "text_to_video",
    "Video Edit": "edit_video",
    "Video Extension": "video_extension",
    "Video to Video": "video_to_video",
    "Video Upscale": "video_upscale",
}

# These are semantic comparison candidates, not assertions that a Comfy graph,
# checkpoint packaging, or recipe is equivalent to the named MoDiff workflow.
# Each target is required to exist in MoDiff's current canonical workflow
# manifest when the ledger is generated or validated.
_CURATED_TEMPLATE_TARGETS = {
    "audio_ace_step1_5_xl_base": ("text_to_audio", "AceStepAudioPipeline:text_to_audio"),
    "audio_ace_step1_5_xl_sft": ("text_to_audio", "AceStepAudioPipeline:text_to_audio"),
    "audio_ace_step1_5_xl_turbo": ("text_to_audio", "AceStepAudioPipeline:text_to_audio"),
    "audio_ace_step_1_5_checkpoint": ("text_to_audio", "AceStepAudioPipeline:text_to_audio"),
    "audio_ace_step_1_5_split": ("text_to_audio", "AceStepAudioPipeline:text_to_audio"),
    "audio_ace_step_1_5_split_4b": ("text_to_audio", "AceStepAudioPipeline:text_to_audio"),
    "audio_stable_audio_example": ("text_to_audio", "StableAudioPipeline:text_to_audio"),
    "flux1_krea_dev": ("text_to_image", "FluxKreaPipeline:text_to_image"),
    "flux_canny_model_example": ("control_image", "FluxCannyPipeline:control_image"),
    "flux_dev_checkpoint_example": ("text_to_image", "FluxDevPipeline:text_to_image"),
    "flux_dev_full_text_to_image": ("text_to_image", "FluxDevPipeline:text_to_image"),
    "flux_fill_inpaint_example": ("inpaint", "FluxFillPipeline:inpaint"),
    "flux_fill_outpaint_example": ("outpaint", "FluxFillPipeline:outpaint"),
    "flux_kontext_dev_basic": ("edit_image", "FluxKontextPipeline:edit_image"),
    "flux_schnell": ("text_to_image", "FluxSchnellPipeline:text_to_image"),
    "flux_schnell_full_text_to_image": ("text_to_image", "FluxSchnellPipeline:text_to_image"),
    "image_ernie_image_turbo": ("text_to_image", "ErnieImagePipeline:text_to_image"),
    "image_flux2_klein_image_edit_4b_base": ("edit_image", "Flux2KleinPipeline:edit_image"),
    "image_flux2_klein_image_edit_4b_distilled": ("edit_image", "Flux2KleinPipeline:edit_image"),
    "image_flux2_klein_text_to_image": ("text_to_image", "Flux2KleinPipeline:text_to_image"),
    "image_joyai_image_edit": ("edit_image", "JoyImageEditPipeline:edit_image"),
    "image_longcat_image_edit": ("edit_image", "LongCatImageEditPipeline:edit_image"),
    "image_longcat_text_to_image": ("text_to_image", "LongCatImagePipeline:text_to_image"),
    "image_ovis_text_to_image": ("text_to_image", "OvisImagePipeline:text_to_image"),
    "image_qwen_Image_2512": ("text_to_image", "QwenImageModularPipeline:text_to_image"),
    "image_qwen_image_edit": ("edit_image", "QwenImageEditModularPipeline:edit_image"),
    "image_qwen_image_edit_2511": ("edit_image", "QwenImageEditPlusModularPipeline:edit_image"),
    "image_qwen_image_edit_2511_int8": ("edit_image", "QwenImageEditPlusModularPipeline:edit_image"),
    "image_qwen_image_layered": (
        "layer_decomposition",
        "QwenImageLayeredModularPipeline:layer_decomposition",
    ),
    "image_z_image_turbo": ("text_to_image", "ZImageModularPipeline:text_to_image"),
    "image_z_image_turbo_int8": ("text_to_image", "ZImageModularPipeline:text_to_image"),
    "sdxl_simple_example": ("text_to_image", "StableDiffusionXLPipeline:text_to_image"),
    "sdxlturbo_example": ("text_to_image", "StableDiffusionXLTurboPipeline:text_to_image"),
    "text_to_video_wan": ("text_to_video", "WanVideoPipeline:text_to_video"),
    "video_ltx2_i2v": ("image_to_video", "LTX2ConditionPipeline:image_to_video"),
    "video_ltx2_t2v": ("text_to_video", "LTX2ConditionPipeline:text_to_video"),
    "video_wan2_2_14B_i2v": ("image_to_video", "WanImageToVideoPipeline:image_to_video"),
    "video_wan2_2_14B_t2v": ("text_to_video", "Wan22Pipeline:text_to_video"),
    "video_wan2_2_5B_ti2v": ("text_to_video", "WanTI2VPipeline:text_to_video"),
}

_CURATED_BLUEPRINT_TARGETS = {
    "image_edit_flux_2_klein_4b": (
        "edit_image",
        ("Flux.2 Klein 4B",),
        "Flux2KleinPipeline:edit_image",
    ),
    "image_edit_longcat_image_edit": (
        "edit_image",
        ("LongCat Image Edit",),
        "LongCatImageEditPipeline:edit_image",
    ),
    "image_edit_qwen_2511": (
        "edit_image",
        ("Qwen Image Edit 2511",),
        "QwenImageEditPlusModularPipeline:edit_image",
    ),
    "image_inpainting_flux_1_fill_dev": (
        "inpaint",
        ("Flux.1 Fill Dev",),
        "FluxFillPipeline:inpaint",
    ),
    "image_to_layers_qwen_image_layered": (
        "layer_decomposition",
        ("Qwen-Image-Layered",),
        "QwenImageLayeredModularPipeline:layer_decomposition",
    ),
    "image_to_video_wan_2_2": (
        "image_to_video",
        ("Wan 2.2",),
        "WanImageToVideoPipeline:image_to_video",
    ),
    "text_to_audio_ace_step_1_5": (
        "text_to_audio",
        ("ACE-Step 1.5",),
        "AceStepAudioPipeline:text_to_audio",
    ),
    "text_to_image_ernie_image_turbo": (
        "text_to_image",
        ("Ernie Image Turbo",),
        "ErnieImagePipeline:text_to_image",
    ),
    "text_to_image_flux_1_dev": (
        "text_to_image",
        ("Flux.1 Dev",),
        "FluxDevPipeline:text_to_image",
    ),
    "text_to_image_flux_1_krea_dev": (
        "text_to_image",
        ("Flux.1 Krea Dev",),
        "FluxKreaPipeline:text_to_image",
    ),
    "text_to_image_qwen_image_2512": (
        "text_to_image",
        ("Qwen-Image 2512",),
        "QwenImageModularPipeline:text_to_image",
    ),
    "text_to_image_z_image_turbo": (
        "text_to_image",
        ("Z-Image-Turbo",),
        "ZImageModularPipeline:text_to_image",
    ),
    "text_to_video_wan_2_2": (
        "text_to_video",
        ("Wan 2.2",),
        "Wan22Pipeline:text_to_video",
    ),
    "video_inpaint_wan2_1_vace": (
        "video_inpaint",
        ("Wan2.1 VACE",),
        "WanVACEPipeline:video_inpaint",
    ),
    "video_inpainting_wan2_1_vace": (
        "video_inpaint",
        ("Wan2.1 VACE",),
        "WanVACEPipeline:video_inpaint",
    ),
}

_EXPLICIT_HOSTED_BLUEPRINT_IDS = {
    "image_captioning_gemini",
    "video_captioning_gemini",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ComfyTemplateResearchError(ValueError):
    """The checked-in research ledger or its source metadata is invalid."""


def comfy_task_modes_for_tags(tags: Iterable[str]) -> tuple[str, ...]:
    """Return deterministic task-mode evidence from official catalog tags."""

    return tuple(sorted({_TASK_TAG_TO_MODE[tag] for tag in tags if tag in _TASK_TAG_TO_MODE}))


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComfyTemplateResearchError(f"{label} must be a non-empty string.")
    return value


def _string_list(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ComfyTemplateResearchError(f"{label} must be a list of non-empty strings.")
    return list(dict.fromkeys(value))


def _records(value: Any, *, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ComfyTemplateResearchError(f"{label} must be a list of objects.")
    return list(value)


def _category_id(kind: str, module_name: str, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{kind}:{module_name}:{slug}"


def _source_files(files: Mapping[str, str]) -> list[dict[str, str]]:
    if set(files) != set(COMFY_SOURCE_FILES):
        raise ComfyTemplateResearchError("Comfy research source-file coverage is incomplete or unexpected.")
    records = []
    for path in COMFY_SOURCE_FILES:
        digest = files[path]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ComfyTemplateResearchError(f"Comfy source hash for {path!r} is invalid.")
        records.append({"path": path, "sha256": digest})
    return records


def _template_source_class(template: Mapping[str, Any], tags: Sequence[str]) -> tuple[str, list[str]]:
    reasons = []
    if template.get("openSource") is False:
        reasons.append("catalog_open_source_false")
    if template.get("name", "").startswith("api_"):
        reasons.append("catalog_api_id_prefix")
    if "API" in tags:
        reasons.append("catalog_api_tag")
    if reasons:
        return "hosted_api", reasons
    if template.get("openSource") is True:
        return "local_open_source", []
    return "ambiguous", ["catalog_open_source_unspecified"]


def _candidate_mapping(
    *,
    template_id: str,
    tags: Sequence[str],
    models: Sequence[str],
    disposition: str,
    supported_workflow_ids: set[str],
) -> tuple[str, dict[str, Any] | None]:
    if disposition == "ineligible":
        return "ineligible", None

    tagged_modes = sorted({_TASK_TAG_TO_MODE[tag] for tag in tags if tag in _TASK_TAG_TO_MODE})
    curated = _CURATED_TEMPLATE_TARGETS.get(template_id)
    if curated is not None:
        task_mode, workflow_id = curated
        if tagged_modes and task_mode not in tagged_modes:
            raise ComfyTemplateResearchError(
                f"Curated Comfy task {task_mode!r} conflicts with catalog tags for {template_id!r}."
            )
        if workflow_id not in supported_workflow_ids:
            raise ComfyTemplateResearchError(f"Curated MoDiff workflow {workflow_id!r} does not exist.")
        if not models:
            raise ComfyTemplateResearchError(f"Curated Comfy template {template_id!r} has no model evidence.")
        return "existing_contract_candidate", {
            "taskMode": task_mode,
            "sourceModelLabels": list(models),
            "moDiffWorkflowId": workflow_id,
        }

    if not models:
        return "no_model_evidence", None
    if len(tagged_modes) != 1:
        return "ambiguous_task_evidence", None
    return "research_candidate", {
        "taskMode": tagged_modes[0],
        "sourceModelLabels": list(models),
    }


def _template_entry(
    template: Mapping[str, Any],
    *,
    category_id: str,
    supported_workflow_ids: set[str],
) -> dict[str, Any]:
    template_id = _string(template.get("name"), label="Comfy template name")
    tags = _string_list(template.get("tags"), label=f"Comfy template {template_id} tags")
    models = _string_list(template.get("models"), label=f"Comfy template {template_id} models")
    custom_nodes = _string_list(
        template.get("requiresCustomNodes"),
        label=f"Comfy template {template_id} custom nodes",
    )
    app_mode = template.get("isApp") is True
    source_class, source_reasons = _template_source_class(template, tags)

    admission_reasons = list(source_reasons)
    if custom_nodes:
        admission_reasons.append("comfy_custom_nodes_require_independent_review")
    if app_mode:
        admission_reasons.append("comfy_app_mode_is_not_a_modiff_contract")
    if source_class == "hosted_api":
        disposition = "ineligible"
    elif custom_nodes or app_mode:
        disposition = "ineligible"
    elif source_class == "ambiguous":
        disposition = "source_review_required"
    else:
        disposition = "research_candidate"

    mapping_status, mapping = _candidate_mapping(
        template_id=template_id,
        tags=tags,
        models=models,
        disposition=disposition,
        supported_workflow_ids=supported_workflow_ids,
    )
    _string(template.get("mediaType"), label=f"Comfy template {template_id} media type")
    entry: dict[str, Any] = {
        "id": template_id,
        "categoryId": category_id,
        "sourceClass": source_class,
        "disposition": disposition,
        "mappingStatus": mapping_status,
    }
    if isinstance(template.get("title"), str) and template["title"]:
        entry["title"] = template["title"]
    if isinstance(template.get("openSource"), bool):
        entry["openSource"] = template["openSource"]
    if app_mode:
        entry["appMode"] = True
    if admission_reasons:
        entry["admissionReasons"] = admission_reasons
    if tags:
        entry["tags"] = tags
    if models:
        entry["models"] = models
    if custom_nodes:
        entry["requiresCustomNodes"] = custom_nodes
    if mapping is not None:
        entry["candidateMapping"] = mapping
    return entry


def _blueprint_entry(
    blueprint: Mapping[str, Any],
    *,
    category_id: str,
    supported_workflow_ids: set[str],
) -> dict[str, Any]:
    blueprint_id = _string(blueprint.get("name"), label="Comfy blueprint name")
    custom_nodes = _string_list(
        blueprint.get("requiresCustomNodes"),
        label=f"Comfy blueprint {blueprint_id} custom nodes",
    )
    if blueprint_id in _EXPLICIT_HOSTED_BLUEPRINT_IDS:
        source_class = "hosted_api"
        disposition = "ineligible"
        reasons = ["catalog_description_identifies_hosted_gemini"]
    else:
        # The blueprint index has no openSource field.  The repository's MIT
        # license covers the checked-in JSON, not every referenced runtime,
        # model, or service, so no stronger execution-source claim is made.
        source_class = "ambiguous"
        disposition = "source_review_required"
        reasons = ["blueprint_index_has_no_open_source_metadata"]
        if custom_nodes:
            reasons.append("comfy_custom_nodes_require_independent_review")

    curated = _CURATED_BLUEPRINT_TARGETS.get(blueprint_id)
    mapping = None
    if disposition == "ineligible":
        mapping_status = "ineligible"
    elif curated is None:
        mapping_status = "ambiguous_task_or_model_evidence"
    else:
        task_mode, source_models, workflow_id = curated
        if workflow_id not in supported_workflow_ids:
            raise ComfyTemplateResearchError(f"Curated MoDiff workflow {workflow_id!r} does not exist.")
        mapping_status = "existing_contract_candidate"
        mapping = {
            "taskMode": task_mode,
            "sourceModelLabels": list(source_models),
            "moDiffWorkflowId": workflow_id,
        }

    _string(blueprint.get("mediaType"), label=f"Comfy blueprint {blueprint_id} media type")
    entry: dict[str, Any] = {
        "id": blueprint_id,
        "title": _string(blueprint.get("title"), label=f"Comfy blueprint {blueprint_id} title"),
        "categoryId": category_id,
        "sourceClass": source_class,
        "disposition": disposition,
        "admissionReasons": reasons,
        "mappingStatus": mapping_status,
    }
    if custom_nodes:
        entry["requiresCustomNodes"] = custom_nodes
    if mapping is not None:
        entry["candidateMapping"] = mapping
    return entry


def _count_values(records: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[key]) for record in records).items()))


def build_comfy_template_research_ledger(
    *,
    template_index: Sequence[Mapping[str, Any]],
    blueprint_index: Sequence[Mapping[str, Any]],
    template_manifest_ids: set[str],
    blueprint_manifest_ids: set[str],
    supported_workflow_ids: set[str],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build the pinned catalog ledger from already-loaded official metadata."""

    template_categories = []
    templates = []
    seen_template_categories = set()
    for category in template_index:
        module_name = _string(category.get("moduleName"), label="Comfy template category module")
        title = _string(category.get("title"), label="Comfy template category title")
        category_id = _category_id("templates", module_name, title)
        if category_id in seen_template_categories:
            raise ComfyTemplateResearchError(f"Duplicate Comfy template category {category_id!r}.")
        seen_template_categories.add(category_id)
        category_type = category.get("type") if isinstance(category.get("type"), str) else None
        category_templates = _records(category.get("templates"), label=f"Comfy category {title} templates")
        ids = [_string(item.get("name"), label=f"Comfy category {title} template name") for item in category_templates]
        template_categories.append(
            {
                "id": category_id,
                "moduleName": module_name,
                "title": title,
                **({"catalogCategory": category["category"]} if isinstance(category.get("category"), str) else {}),
                **({"type": category_type} if category_type is not None else {}),
                "templateIds": sorted(ids),
            }
        )
        templates.extend(
            _template_entry(
                item,
                category_id=category_id,
                supported_workflow_ids=supported_workflow_ids,
            )
            for item in category_templates
        )

    blueprint_categories = []
    blueprints = []
    seen_blueprint_categories = set()
    for category in blueprint_index:
        module_name = _string(category.get("moduleName"), label="Comfy blueprint category module")
        title = _string(category.get("title"), label="Comfy blueprint category title")
        category_id = _category_id("blueprints", module_name, title)
        if category_id in seen_blueprint_categories:
            raise ComfyTemplateResearchError(f"Duplicate Comfy blueprint category {category_id!r}.")
        seen_blueprint_categories.add(category_id)
        category_blueprints = _records(category.get("blueprints"), label=f"Comfy category {title} blueprints")
        ids = [
            _string(item.get("name"), label=f"Comfy category {title} blueprint name") for item in category_blueprints
        ]
        blueprint_categories.append(
            {
                "id": category_id,
                "moduleName": module_name,
                "title": title,
                "blueprintIds": sorted(ids),
            }
        )
        blueprints.extend(
            _blueprint_entry(
                item,
                category_id=category_id,
                supported_workflow_ids=supported_workflow_ids,
            )
            for item in category_blueprints
        )

    templates.sort(key=lambda item: item["id"])
    blueprints.sort(key=lambda item: item["id"])
    template_categories.sort(key=lambda item: item["id"])
    blueprint_categories.sort(key=lambda item: item["id"])
    template_ids = [item["id"] for item in templates]
    blueprint_ids = [item["id"] for item in blueprints]
    if len(template_ids) != len(set(template_ids)):
        raise ComfyTemplateResearchError("Comfy template ids must be globally unique.")
    if len(blueprint_ids) != len(set(blueprint_ids)):
        raise ComfyTemplateResearchError("Comfy blueprint ids must be globally unique.")

    template_id_set = set(template_ids)
    blueprint_id_set = set(blueprint_ids)
    template_manifest_missing = sorted(template_id_set - template_manifest_ids)
    template_manifest_extra = sorted(template_manifest_ids - template_id_set)
    blueprint_manifest_missing = sorted(blueprint_id_set - blueprint_manifest_ids)
    blueprint_manifest_extra = sorted(blueprint_manifest_ids - blueprint_id_set)

    return {
        "schemaVersion": COMFY_TEMPLATE_RESEARCH_SCHEMA_VERSION,
        "contractKind": "non_executable_research_ledger",
        "source": {
            "repository": PINNED_COMFY_WORKFLOW_TEMPLATES_REPOSITORY,
            "revision": PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION,
            "committedAt": PINNED_COMFY_WORKFLOW_TEMPLATES_COMMITTED_AT,
            "license": "MIT",
            "copyright": "Copyright (c) 2023-present Comfy Org",
            "files": _source_files(source_hashes),
        },
        "boundary": {
            "importsComfyGraphs": False,
            "executesComfyNodes": False,
            "downloadsModelsOrMedia": False,
            "mappingMeaning": "semantic_comparison_candidate_only",
            "supportClaim": "none",
        },
        "summary": {
            "templateCategoryCount": len(template_categories),
            "templateCount": len(templates),
            "templateSourceClassCounts": _count_values(templates, "sourceClass"),
            "templateDispositionCounts": _count_values(templates, "disposition"),
            "templateMappingStatusCounts": _count_values(templates, "mappingStatus"),
            "blueprintCategoryCount": len(blueprint_categories),
            "blueprintCount": len(blueprints),
            "blueprintSourceClassCounts": _count_values(blueprints, "sourceClass"),
            "blueprintDispositionCounts": _count_values(blueprints, "disposition"),
            "blueprintMappingStatusCounts": _count_values(blueprints, "mappingStatus"),
        },
        "distributionCoverage": {
            "templateCatalogIdsInCoreManifest": len(template_id_set & template_manifest_ids),
            "templateCatalogIdsMissingFromCoreManifest": template_manifest_missing,
            "coreManifestIdsOutsideTemplateCatalog": template_manifest_extra,
            "blueprintCatalogIdsInCoreManifest": len(blueprint_id_set & blueprint_manifest_ids),
            "blueprintCatalogIdsMissingFromCoreManifest": blueprint_manifest_missing,
            "coreManifestIdsOutsideBlueprintCatalog": blueprint_manifest_extra,
        },
        "templateCategories": template_categories,
        "templates": templates,
        "blueprintCategories": blueprint_categories,
        "blueprints": blueprints,
    }


def _reconstruct_template_index(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    templates = {item["id"]: item for item in _records(ledger.get("templates"), label="ledger templates")}
    categories = []
    for category in _records(ledger.get("templateCategories"), label="ledger template categories"):
        rows = []
        for template_id in _string_list(category.get("templateIds"), label="ledger template category ids"):
            entry = templates.get(template_id)
            if entry is None:
                raise ComfyTemplateResearchError(f"Template category references unknown id {template_id!r}.")
            row: dict[str, Any] = {
                "name": entry["id"],
                # Media type is validated while reading the official index but
                # intentionally omitted from the compact ledger: upstream uses
                # it for thumbnail media, not a MoDiff output contract.
                "mediaType": category.get("type") or "image",
                "tags": entry.get("tags", []),
                "models": entry.get("models", []),
            }
            if "title" in entry:
                row["title"] = entry["title"]
            if "openSource" in entry:
                row["openSource"] = entry["openSource"]
            if entry.get("appMode") is True:
                row["isApp"] = True
            if entry.get("requiresCustomNodes"):
                row["requiresCustomNodes"] = entry["requiresCustomNodes"]
            rows.append(row)
        reconstructed: dict[str, Any] = {
            "moduleName": category["moduleName"],
            "title": category["title"],
            "templates": rows,
        }
        if "catalogCategory" in category:
            reconstructed["category"] = category["catalogCategory"]
        if "type" in category:
            reconstructed["type"] = category["type"]
        categories.append(reconstructed)
    return categories


def _reconstruct_blueprint_index(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    blueprints = {item["id"]: item for item in _records(ledger.get("blueprints"), label="ledger blueprints")}
    categories = []
    for category in _records(ledger.get("blueprintCategories"), label="ledger blueprint categories"):
        rows = []
        for blueprint_id in _string_list(category.get("blueprintIds"), label="ledger blueprint category ids"):
            entry = blueprints.get(blueprint_id)
            if entry is None:
                raise ComfyTemplateResearchError(f"Blueprint category references unknown id {blueprint_id!r}.")
            row: dict[str, Any] = {
                "name": entry["id"],
                "title": entry["title"],
                "mediaType": "image",
            }
            if entry.get("requiresCustomNodes"):
                row["requiresCustomNodes"] = entry["requiresCustomNodes"]
            rows.append(row)
        categories.append(
            {
                "moduleName": category["moduleName"],
                "title": category["title"],
                "blueprints": rows,
            }
        )
    return categories


def validate_comfy_template_research_ledger(
    ledger: Mapping[str, Any],
    *,
    supported_workflow_ids: set[str],
) -> dict[str, Any]:
    """Validate provenance, classifications, mappings, coverage, and determinism."""

    if ledger.get("schemaVersion") != COMFY_TEMPLATE_RESEARCH_SCHEMA_VERSION:
        raise ComfyTemplateResearchError("Unsupported Comfy template research schema version.")
    if ledger.get("contractKind") != "non_executable_research_ledger":
        raise ComfyTemplateResearchError("Comfy research data must remain explicitly non-executable.")
    source = ledger.get("source")
    if not isinstance(source, Mapping):
        raise ComfyTemplateResearchError("Comfy research source provenance is missing.")
    expected_source = {
        "repository": PINNED_COMFY_WORKFLOW_TEMPLATES_REPOSITORY,
        "revision": PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION,
        "committedAt": PINNED_COMFY_WORKFLOW_TEMPLATES_COMMITTED_AT,
        "license": "MIT",
        "copyright": "Copyright (c) 2023-present Comfy Org",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ComfyTemplateResearchError(f"Comfy research source {key!r} is not pinned to the reviewed value.")
    source_files = {
        _string(item.get("path"), label="Comfy source path"): _string(
            item.get("sha256"),
            label="Comfy source hash",
        )
        for item in _records(source.get("files"), label="Comfy source files")
    }
    _source_files(source_files)

    boundary = ledger.get("boundary")
    expected_boundary = {
        "importsComfyGraphs": False,
        "executesComfyNodes": False,
        "downloadsModelsOrMedia": False,
        "mappingMeaning": "semantic_comparison_candidate_only",
        "supportClaim": "none",
    }
    if boundary != expected_boundary:
        raise ComfyTemplateResearchError("Comfy research execution boundary changed unexpectedly.")

    template_records = _records(ledger.get("templates"), label="ledger templates")
    blueprint_records = _records(ledger.get("blueprints"), label="ledger blueprints")
    distribution = ledger.get("distributionCoverage")
    if not isinstance(distribution, Mapping):
        raise ComfyTemplateResearchError("Comfy distribution coverage is missing.")
    template_manifest_ids = (
        {item["id"] for item in template_records}
        - set(
            _string_list(
                distribution.get("templateCatalogIdsMissingFromCoreManifest"),
                label="template ids missing from the core manifest",
            )
        )
    ) | set(
        _string_list(
            distribution.get("coreManifestIdsOutsideTemplateCatalog"),
            label="non-catalog template manifest ids",
        )
    )
    blueprint_manifest_ids = (
        {item["id"] for item in blueprint_records}
        - set(
            _string_list(
                distribution.get("blueprintCatalogIdsMissingFromCoreManifest"),
                label="blueprint ids missing from the core manifest",
            )
        )
    ) | set(
        _string_list(
            distribution.get("coreManifestIdsOutsideBlueprintCatalog"),
            label="non-catalog blueprint manifest ids",
        )
    )
    expected = build_comfy_template_research_ledger(
        template_index=_reconstruct_template_index(ledger),
        blueprint_index=_reconstruct_blueprint_index(ledger),
        template_manifest_ids=template_manifest_ids,
        blueprint_manifest_ids=blueprint_manifest_ids,
        supported_workflow_ids=supported_workflow_ids,
        source_hashes=source_files,
    )
    if dict(ledger) != expected:
        raise ComfyTemplateResearchError("Comfy template research ledger is stale or internally inconsistent.")
    return expected
