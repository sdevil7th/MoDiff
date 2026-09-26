"""Reviewed standard-Diffusers Cluster Node contracts.

These definitions describe exact Studio composites backed by ordinary
``DiffusionPipeline`` subclasses.  They are intentionally separate from the
official Modular Diffusers block snapshot: a standard pipeline may still be a
useful one-node Cluster, but its loader/generate/export graph must never be
presented as an upstream ``ModularPipeline.blocks`` hierarchy.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

from modiff.huggingface_cluster_promotions import promotion_receipt_for_admission
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.studio_execution_specs import (
    STUDIO_EXECUTION_SPEC_DEFINITIONS,
    studio_execution_spec_for_pair,
    studio_model_dependencies_for_pair,
    studio_capability_definitions,
)


_REVIEWED_SPECS = (
    "wan-22-ti2v-5b:text-to-video:v1",
    "ace-step-v1.5-xl-turbo:text-to-audio:v1",
    "ace-step-v1.5-xl-turbo:audio-variation:v1",
    "ace-step-v1.5-xl-turbo:audio-continuation:v1",
    "ace-step-v1.5-xl-turbo:audio-repaint:v1",
    "longcat-audio-dit-1b:text-to-audio:v1",
    "audioldm2-base:text-to-audio:v1",
    "flux-controlnet:control-image:v1",
    "flux-controlnet:control-edit-image:v1",
    "flux-controlnet:control-inpaint:v1",
    "flux2-klein-kv:text-to-image:v1",
    "flux2-klein-kv:edit-image:v1",
    "flux2-klein-kv:multi-image-reference-edit:v1",
)

_NEW_ORDINARY_FLUX_SPECS = (
    "flux-schnell:text-to-image:v1",
    "flux-krea:text-to-image:v1",
    "flux-depth:control-image:v1",
    "flux-depth:control-edit-image:v1",
    "flux-depth:control-inpaint:v1",
    "flux-canny:control-image:v1",
    "flux-canny:control-edit-image:v1",
    "flux-canny:control-inpaint:v1",
    "flux-redux:edit-image:v1",
    "flux-redux:multi-image-reference-edit:v1",
    "flux-fill:inpaint:v1",
    "flux-fill:outpaint:v1",
    "flux-dev:inpaint:v1",
    "flux-kontext:multi-image-reference-edit:v1",
    "flux-kontext-inpaint-direct:inpaint:v1",
    "flux-kontext-inpaint-direct:outpaint:v1",
    "flux2-klein:multi-image-reference-edit:v1",
    "flux2-klein-inpaint-direct:inpaint:v1",
    "flux2-klein-inpaint-direct:outpaint:v1",
    "flux2-dev:multi-image-reference-edit:v1",
)
_REVIEWED_SPECS += _NEW_ORDINARY_FLUX_SPECS

# Existing creator-facing recipe seeds for newly admitted ordinary operations.
# These affect new definitions only, never saved instance values or legacy forms.
_NEW_IMAGE_RECIPE_SEEDS = {
    'FluxFillPipeline': ('FLUX.1 Fill dev', 50, 30),
    'FluxDepthPipeline': ('FLUX.1 Depth dev', 50, 30),
    'FluxCannyPipeline': ('FLUX.1 Canny dev', 50, 30),
    'FluxKontextPipeline': ('FLUX.1 Kontext dev', 28, 3.5),
}

_IMAGE_INPUT_FIELDS = {
    "prompt": ("prompt", "str", True, "", "Describe the intended image or edit."),
    "negativePrompt": ("negative_prompt", "str", False, "", "Unwanted content."),
    "width": ("width", "int", False, 1024, "Output width in pixels."),
    "height": ("height", "int", False, 1024, "Output height in pixels."),
    "steps": ("num_inference_steps", "int", False, 28, "Denoising step count."),
    "guidanceScale": ("guidance_scale", "float", False, 3.5, "Guidance scale."),
    "seed": ("seed", "int", False, 42, "Random generator seed."),
    "maxSequenceLength": ("max_sequence_length", "int", False, 512, "Maximum prompt token count."),
    "strength": ("strength", "float", False, 0.8, "Source image denoising strength."),
    "conditioningScale": ("conditioning_scale", "float", False, 1.0, "Control image influence."),
    "controlGuidanceStart": ("control_guidance_start", "float", False, 0.0, "Start fraction of ControlNet guidance."),
    "controlGuidanceEnd": ("control_guidance_end", "float", False, 1.0, "End fraction of ControlNet guidance."),
    "referenceImages": ("image", "image", True, None, "Source or reference images for this operation."),
    "maskImage": ("mask_image", "image", True, None, "Mask identifying the edited region."),
    "controlImage": ("control_image", "image", True, None, "Prepared control image."),
    "outputType": ("output_type", "str", False, "pil", "Decoded image output type."),
}

_AUDIO_INPUT_FIELDS = {
    "prompt": ("prompt", "str", True, "", "Music description: genre, mood, instrumentation and arrangement."),
    "negativePrompt": ("negative_prompt", "str", False, "", "Unwanted audio content or artifacts."),
    "lyrics": ("lyrics", "str", False, "", "Original lyrics with structure tags on separate lines."),
    "audioDuration": ("audio_duration", "float", False, 30.0, "Generated audio duration in seconds."),
    "steps": ("num_inference_steps", "int", False, 8, "Native turbo denoising step count."),
    "guidanceScale": ("guidance_scale", "float", False, 1.0, "Guidance scale; turbo uses distilled guidance."),
    "sourceAudio": ("source_audio", "audio", True, None, "Source audio for the selected edit operation."),
    "extensionDuration": ("extension_duration", "float", False, 15.0, "Continuation length in seconds."),
    "repaintingStart": ("repainting_start", "float", False, 0.0, "Start of replaced section, in seconds."),
    "repaintingEnd": ("repainting_end", "float", False, 10.0, "End of replaced section, in seconds."),
    "audioCoverStrength": ("audio_cover_strength", "float", False, 0.5, "Source arrangement strength for variation."),
}

_AUDIO_MODE_INPUTS = {
    "text_to_audio": ("audioDuration",),
    "audio_variation": ("sourceAudio", "audioDuration", "audioCoverStrength"),
    "audio_continuation": ("sourceAudio", "extensionDuration"),
    "audio_repaint": ("sourceAudio", "repaintingStart", "repaintingEnd"),
}

_AUDIO_SEALED_VALUES = {
    "false": False,
    "text2music": "text2music", "cover": "cover", "continuation": "continuation", "repaint": "repaint",
    "sampleRate48000": 48000, "referenceWindow15": 15, "targetPeakMinus1": -1,
    "maxAdjustment12": 12, "boundaryFade001": 0.01,
    "text2audio": "text2audio", "sampleRate24000": 24000, "sampleRate16000": 16000,
    "numWaveforms3": 3,
}

_INPUT_FIELDS = {
    "prompt": ("prompt", "str", True, "", "Text prompt for the generated video."),
    "negativePrompt": (
        "negative_prompt",
        "str",
        False,
        "",
        "Content and artifacts to avoid in the generated video.",
    ),
    "width": ("width", "int", False, 1280, "Output width in pixels."),
    "height": ("height", "int", False, 704, "Output height in pixels."),
    "numFrames": ("num_frames", "int", False, 121, "Number of generated video frames."),
    "steps": ("num_inference_steps", "int", False, 50, "Denoising step count."),
    "maxSequenceLength": (
        "max_sequence_length",
        "int",
        False,
        512,
        "Maximum prompt sequence length.",
    ),
    "outputType": ("output_type", "str", False, "pil", "Decoded Diffusers output type."),
}

_INSTANCE_INPUT_SOURCES = (
    "prompt",
    "negativePrompt",
    "width",
    "height",
    "numFrames",
    "steps",
    "maxSequenceLength",
    "outputType",
)

_SEALED_VALUES = {
    "empty": "",
    "true": True,
}


def _hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _field(source: str, fields=None) -> dict[str, Any]:
    name, field_type, required, default, description = (fields or _INPUT_FIELDS)[source]
    return {
        "name": name,
        "type": field_type,
        "required": required,
        "default": default,
        "description": description,
    }


def _block_definition(class_name: str, kind: str, description: str) -> dict[str, Any]:
    body = {
        "schemaVersion": 1,
        "provider": "diffusers",
        "className": class_name,
        "kind": kind,
        "description": description,
        "inputs": [],
        "variadicInputs": [],
        "requiredInputs": [],
        "outputs": [],
        "components": [],
        "configs": [],
    }
    content_hash = _hash(body)
    return {
        **body,
        "id": f"diffusers.composite-block:{class_name}:{content_hash}",
        "contentHash": content_hash,
    }


def _definition_and_blocks(spec_id: str, diffusers_revision: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    studio_definition = STUDIO_EXECUTION_SPEC_DEFINITIONS[spec_id]
    model_type = str(studio_definition["modelType"])
    mode = str(studio_definition["mode"])
    profile = studio_definition["profile"]
    public_spec = studio_execution_spec_for_pair(model_type, mode)
    if public_spec is None or public_spec["id"] != spec_id:
        raise ValueError(f"Standard Diffusers Cluster spec {spec_id!r} is not uniquely resolvable.")
    if public_spec["executionPath"] not in {"direct-diffusers-image", "direct-diffusers-video", "direct-diffusers-audio"}:
        raise ValueError(f"Standard Diffusers Cluster spec {spec_id!r} is not a direct Diffusers route.")

    repository = str(profile["default_repo"])
    revision = require_catalog_revision(repository, model_type=model_type)
    binding_sources = sorted({str(source) for _role, _field_name, source in public_spec["bindings"]})
    audio = public_spec["executionPath"] == "direct-diffusers-audio"
    image = public_spec["executionPath"] == "direct-diffusers-image"
    field_sources = (tuple(_IMAGE_INPUT_FIELDS) if image else
                    ("prompt", "negativePrompt", "lyrics", "steps", "guidanceScale", *_AUDIO_MODE_INPUTS[mode]) if audio else _INSTANCE_INPUT_SOURCES)
    input_sources = [source for source in field_sources if source in binding_sources]
    input_fields = _AUDIO_INPUT_FIELDS if audio else _INPUT_FIELDS
    capability = deepcopy(studio_definition.get("capability") or studio_capability_definitions().get(model_type, {}))
    if image and model_type in _NEW_IMAGE_RECIPE_SEEDS:
        label, steps, guidance = _NEW_IMAGE_RECIPE_SEEDS[model_type]
        capability.update(label=label, recommendedSteps=steps, recommendedGuidance=guidance,
                          defaultSize={'width': 1024, 'height': 1024})
    if image:
        input_fields = dict(_IMAGE_INPUT_FIELDS)
        for source, value in (
            ("steps", capability["recommendedSteps"]),
            ("guidanceScale", capability["recommendedGuidance"]),
            ("width", capability["defaultSize"]["width"]),
            ("height", capability["defaultSize"]["height"]),
        ):
            name, kind, required, _default, description = input_fields[source]
            input_fields[source] = (name, kind, required, value, description)
        if not capability.get("supportsNegativePrompt", False):
            input_sources = [source for source in input_sources if source != "negativePrompt"]
        if not capability.get("supportsGuidance", True):
            input_sources = [source for source in input_sources if source != "guidanceScale"]
        if not capability.get("supportsControlImage", False):
            input_sources = [source for source in input_sources if source != "conditioningScale"]
        if model_type == 'FluxReduxPipeline' and mode == 'multi_image_reference_edit' and 'conditioningScale' in binding_sources:
            input_fields['conditioningScale'] = ('reference_strength', 'float', False, 1.0,
                                                'Reference embedding influence; independently overridable in Generate.')
            input_sources.append('conditioningScale')
        if model_type in {'FluxSchnellPipeline', 'FluxKreaPipeline', 'FluxReduxPipeline'} or mode == 'multi_image_reference_edit':
            input_sources = [source for source in input_sources if source != 'strength']
    if audio and model_type != "AceStepAudioPipeline":
        # Preserve existing ACE identities. New composites seed only their own
        # reviewed recipe; music-turbo defaults do not apply to sound generators.
        input_fields = dict(input_fields)
        for source, key, description in (
            ("audioDuration", "recommendedDuration", "Generated audio duration in seconds."),
            ("steps", "recommendedSteps", "Native denoising step count."),
            ("guidanceScale", "recommendedGuidance", "Prompt guidance scale."),
        ):
            name, kind, required, _default, _description = input_fields[source]
            input_fields[source] = (name, kind, required, capability[key], description)
        input_fields["prompt"] = ("prompt", "str", True, "", "Describe the sound, environment and temporal sequence.")
    inputs = [_field(source, input_fields) for source in input_sources]
    input_by_source = {source: field["name"] for source, field in zip(input_sources, inputs, strict=True)}
    model_dependencies = studio_model_dependencies_for_pair(model_type, mode) if image else []
    dependency_values = {}
    if model_dependencies:
        if len(model_dependencies) != 1:
            raise ValueError("Image composite auxiliary bindings require one exact dependency.")
        dependency_values = {key: model_dependencies[0][key] for key in ("kind", "repo", "revision")}
    sealed_values = {
        source: value
        for source, value in {
            "artifact": repository,
            "pipelineClass": public_spec["pipelineClass"],
            "defaultRevision": revision,
            "executionProfileId": public_spec["executionProfileId"],
            "mode": mode,
            **_SEALED_VALUES,
            **(_AUDIO_SEALED_VALUES if audio else {}),
            **({"removeAlpha": "remove alpha"} if image else {}),
            **dependency_values,
        }.items()
        if source in binding_sources
    }
    instance_input_bindings = [
        {"bindingSource": source, "input": input_by_source[source]} for source in input_sources
    ]
    execution_parameter_sources = sorted(
        set(binding_sources) - set(input_sources) - set(sealed_values)
    )

    role_blocks: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    placements: list[dict[str, Any]] = []
    for order, (role, node_key, _x, _y) in enumerate(public_spec["roles"]):
        action = str(node_key).rsplit(".", 1)[-1]
        block = _block_definition(
            action,
            "block",
            f"Existing MoDiff node {node_key} used by this reviewed standard Diffusers composite.",
        )
        role_blocks.append(block)
        steps.append(
            {
                "path": str(role),
                "className": action,
                "kind": "block",
                "description": block["description"],
            }
        )
        placements.append(
            {
                "path": [str(role)],
                "legacyPath": str(role),
                "order": order,
                "blockDefinitionId": block["id"],
            }
        )

    root_class = f"{model_type}{''.join(part.title() for part in mode.split('_'))}Composite"
    root_block = _block_definition(
        root_class,
        "sequential",
        "Standard Diffusers loader, execution recipe, generation, export, and preview container.",
    )
    definition_id = f"diffusers.composite:{model_type}:{mode}"
    adapter = {
        "schemaVersion": 1,
        "id": f"diffusers.composite-adapter:{model_type}:{mode}",
        "source": "mode",
        "adapterId": mode,
        "upstreamWorkflowId": mode,
        "requiredInputs": [field["name"] for field in inputs if field["required"]],
        "actionSequence": [str(role) for role, _node_key, _x, _y in public_spec["roles"]],
        "stateEdges": [],
        "upstreamBlockSequence": [str(node_key) for _role, node_key, _x, _y in public_spec["roles"]],
    }
    admission_id = f"diffusers.cluster-admission:{model_type}:{mode}:mode:{mode}"
    studio_execution_spec = {
        "id": public_spec["id"],
        "contentHash": public_spec["contentHash"],
        "executionProfileId": public_spec["executionProfileId"],
    }
    artifact = {"repo": repository, "revision": revision}
    promotion_receipt = promotion_receipt_for_admission(
        admission_id,
        definition_id=definition_id,
        artifact=artifact,
        studio_execution_spec=studio_execution_spec,
    )
    live_proof = (bool(profile.get("live_proof")) and spec_id not in _NEW_ORDINARY_FLUX_SPECS) or promotion_receipt is not None
    publication_reasons = [
        {
            "code": "runtime_resource_admission_required",
            "message": "The materialized Cluster graph must pass exact artifact and resource checks before Run is enabled.",
        }
    ]
    if not live_proof:
        publication_reasons.append(
            {
                "code": "live_output_review_pending",
                "message": "The exact Cluster execution route has not received approved visible-frontend live-output evidence.",
            }
        )
    admission = {
        "schemaVersion": 4,
        "id": admission_id,
        "definitionId": definition_id,
        "studioMode": mode,
        "bindingSources": binding_sources,
        "instanceInputBindings": instance_input_bindings,
        "executionParameterSources": execution_parameter_sources,
        "sealedBindingValues": sealed_values,
        "modelDependencies": model_dependencies,
        # Image role registries already carry the exact structured task signal
        # from _execution_spec_role_params; pipelineClass alone is not that
        # signal and must never be replayed as an onSignal value.
        "dynamicFieldActions": ([] if image else [{
            "role": "audioGenerate", "field": "pipeline", "event": "onSignal", "valueSource": "pipelineClass",
        }] if audio and model_type != "AceStepAudioPipeline" else []),
        "adapterContractId": adapter["id"],
        "studioExecutionSpec": studio_execution_spec,
        "artifact": artifact,
        "status": "admitted",
        "claim": "static_graph_contract_compatible",
        "executable": False,
        "publication": {
            "schemaVersion": 1,
            "readiness": "graph_qualified",
            "insertable": True,
            "executable": False,
            "autoEligible": False,
            "liveProof": live_proof,
            "reasons": publication_reasons,
        },
        "reasons": [],
    }
    body = {
        "schemaVersion": 5,
        "id": definition_id,
        "provider": "diffusers",
        "publisher": "huggingface",
        "surface": "diffusers_cluster_nodes",
        "definitionKind": "studio_execution_composite",
        "ownership": "library",
        "mutable": False,
        "libraryRevision": diffusers_revision,
        "pipelineClass": model_type,
        "blocksClass": str(profile["pipeline_class"]),
        "pipelineKind": "sequential",
        "workflowId": mode,
        "workflowKind": "sequential",
        "taskId": mode,
        "taskContractId": f"diffusers.task.{mode}.v1",
        "label": (
            f"{capability['label'].split(' — ')[0]} — {mode.replace('_', ' ').title()}" if image else
            f"{'ACE-Step' if model_type == 'AceStepAudioPipeline' else capability['family']} — {mode.replace('_', ' ').title()}"
            if audio else "Wan 2.2 TI2V 5B — Text to Video"
        ),
        "description": (
            "Standard Diffusers image loader, execution recipe, generation and preview nodes; not an upstream Modular hierarchy."
            if image else
            "Standard Diffusers audio loader, generation, processing and preview nodes; not an upstream Modular hierarchy."
            if audio else "Cluster Node containing the exact standard Diffusers Wan loader, execution recipe, "
            "text-to-video generation, export, and preview graph."
        ),
        "integrationStatus": "reviewed_diffusers_composite",
        "executionClaim": "discovery_only",
        "executionAdmissions": [admission],
        "graphAdapterContracts": [adapter],
        "inputs": inputs,
        "outputs": [
            {
                "name": "images" if image else "audio" if audio else "video",
                "type": "image" if image else "audio" if audio else "video",
                "required": True,
                "default": None,
                "description": "Generated images from the reviewed standard Diffusers graph." if image else "Exported audio from the reviewed standard Diffusers graph." if audio else "Exported video from the reviewed standard Diffusers graph.",
            }
        ],
        "requiredInputs": [field["name"] for field in inputs if field["required"]],
        "requiredInputAlternatives": [],
        "stateKeys": [],
        "components": [
            {
                "name": "pipeline",
                "type": str(profile["pipeline_class"]),
                "creationMethod": "from_pretrained",
                "reuseKey": ["model", "revision", "dtype", "device"],
            }
        ],
        "steps": steps,
        "blockContractHash": root_block["contentHash"],
        "rootBlockDefinitionId": root_block["id"],
        "blockPlacements": placements,
    }
    return {**body, "contentHash": _hash(body)}, [root_block, *role_blocks]


def reviewed_diffusers_cluster_catalog(diffusers_revision: str) -> dict[str, Any]:
    """Return exact standard-Diffusers definitions and composite blocks."""

    definitions: list[dict[str, Any]] = []
    blocks_by_id: dict[str, dict[str, Any]] = {}
    for spec_id in _REVIEWED_SPECS:
        definition, blocks = _definition_and_blocks(spec_id, diffusers_revision)
        definitions.append(definition)
        for block in blocks:
            existing = blocks_by_id.setdefault(block["id"], block)
            if existing != block:
                raise ValueError("Standard Diffusers composite block identity collision.")
    definitions.sort(key=lambda item: item["id"])
    by_task: dict[str, list[str]] = defaultdict(list)
    for definition in definitions:
        by_task[definition["taskId"]].append(definition["id"])
    return deepcopy(
        {
            "definitions": definitions,
            "blockDefinitions": sorted(blocks_by_id.values(), key=lambda item: item["id"]),
            "taskMemberships": dict(by_task),
        }
    )
