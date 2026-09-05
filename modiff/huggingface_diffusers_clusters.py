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
)


_REVIEWED_SPECS = ("wan-22-ti2v-5b:text-to-video:v1",)

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


def _field(source: str) -> dict[str, Any]:
    name, field_type, required, default, description = _INPUT_FIELDS[source]
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
    if public_spec["executionPath"] not in {"direct-diffusers-image", "direct-diffusers-video"}:
        raise ValueError(f"Standard Diffusers Cluster spec {spec_id!r} is not a direct Diffusers route.")

    repository = str(profile["default_repo"])
    revision = require_catalog_revision(repository, model_type=model_type)
    binding_sources = sorted({str(source) for _role, _field_name, source in public_spec["bindings"]})
    input_sources = [source for source in _INSTANCE_INPUT_SOURCES if source in binding_sources]
    inputs = [_field(source) for source in input_sources]
    input_by_source = {source: field["name"] for source, field in zip(input_sources, inputs, strict=True)}
    sealed_values = {
        source: value
        for source, value in {
            "artifact": repository,
            "pipelineClass": public_spec["pipelineClass"],
            "defaultRevision": revision,
            "executionProfileId": public_spec["executionProfileId"],
            "mode": mode,
            **_SEALED_VALUES,
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
    live_proof = bool(profile.get("live_proof")) or promotion_receipt is not None
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
        "modelDependencies": [],
        "dynamicFieldActions": [],
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
        "label": "Wan 2.2 TI2V 5B — Text to Video",
        "description": (
            "Cluster Node containing the exact standard Diffusers Wan loader, execution recipe, "
            "text-to-video generation, export, and preview graph."
        ),
        "integrationStatus": "reviewed_diffusers_composite",
        "executionClaim": "discovery_only",
        "executionAdmissions": [admission],
        "graphAdapterContracts": [adapter],
        "inputs": inputs,
        "outputs": [
            {
                "name": "video",
                "type": "video",
                "required": True,
                "default": None,
                "description": "Exported video from the reviewed standard Diffusers graph.",
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
