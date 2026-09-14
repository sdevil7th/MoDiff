"""Immutable first-party Transformers Cluster Node contracts.

These definitions wrap MoDiff's existing task-generic loader/action/preview
graphs.  They are deliberately derived from reviewed Studio execution specs,
exact artifact revisions, and the app-managed Transformers optional-runtime
contract.  Building the catalog is read-only and does not import Transformers,
Torch, node modules, or model weights.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any

from modiff.diffusers_profiles import execution_profiles_for_execution
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS, studio_execution_spec_for_pair


TRANSFORMERS_CLUSTER_CONTRACT_SCHEMA_VERSION = 1

_REVIEWED_SPECS = (
    "smollm2-135m-instruct:text-generation:v1",
    "smolvlm-256m-instruct:image-to-text:v1",
    "janus-pro-1b:text-generation:v1",
    "janus-pro-1b:image-to-text:v1",
    "janus-pro-1b:text-to-image:v1",
    "whisper-tiny:speech-to-text:v1",
    "whisper-tiny:speech-translation:v1",
    "wav2vec2-base-960h:speech-to-text:v1",
)

_SEALED_VALUES = {
    "anyToAnyImage": "image",
    "anyToAnyText": "text",
    "transcribe": "transcribe",
    "translate": "translate",
    "true": True,
}

_INPUT_FIELDS = {
    "dtype": ("dtype", "str", True, "float32", "Model weight and compute dtype."),
    "prompt": ("prompt", "str", True, "", "Text prompt passed to the reviewed task action."),
    "referenceImages": ("image", "image", True, None, "Local image input for the task."),
    "alphaMode": ("alpha_mode", "str", False, "ignore", "Alpha-channel handling for the local image."),
    "sourceAudio": ("audio", "audio", True, None, "Local audio input for speech recognition."),
    "speechLanguage": ("language", "str", False, "", "Optional source language code."),
    "speechTimestamps": ("timestamps", "str", False, "segment", "Timestamp detail: none, segment, or word."),
    "speechChunkSeconds": ("chunk_length_seconds", "float", False, 30.0, "Bounded audio chunk length."),
    "speechStrideSeconds": ("stride_length_seconds", "float", False, 5.0, "Bounded overlap between chunks."),
    "maxNewTokens": ("max_new_tokens", "int", False, 256, "Maximum number of generated tokens."),
    "minNewTokens": ("min_new_tokens", "int", False, 0, "Minimum number of generated tokens."),
    "doSample": ("do_sample", "bool", False, False, "Use sampling instead of deterministic decoding."),
    "temperature": ("temperature", "float", False, 1.0, "Sampling temperature."),
    "topP": ("top_p", "float", False, 1.0, "Nucleus sampling probability."),
    "topK": ("top_k", "int", False, 50, "Top-k sampling cutoff."),
    "numBeams": ("num_beams", "int", False, 1, "Beam-search width."),
    "repetitionPenalty": ("repetition_penalty", "float", False, 1.0, "Repetition penalty."),
    "useChatTemplate": ("use_chat_template", "bool", False, True, "Apply the model's reviewed chat template."),
}

_LABELS = {
    "smollm2-135m-instruct:text-generation:v1": "SmolLM2 135M Instruct — Text Generation",
    "smolvlm-256m-instruct:image-to-text:v1": "SmolVLM 256M Instruct — Image to Text",
    "janus-pro-1b:text-generation:v1": "Janus Pro 1B — Text Generation",
    "janus-pro-1b:image-to-text:v1": "Janus Pro 1B — Image to Text",
    "janus-pro-1b:text-to-image:v1": "Janus Pro 1B — Text to Image",
    "whisper-tiny:speech-to-text:v1": "Whisper Tiny — Speech to Text",
    "whisper-tiny:speech-translation:v1": "Whisper Tiny — Speech Translation",
    "wav2vec2-base-960h:speech-to-text:v1": "Wav2Vec2 Base 960h — Speech to Text",
}

_DESCRIPTIONS = {
    "text_generation": "Cluster Node containing an exact Transformers causal text loader, bounded generation action, and preview.",
    "image_to_text": "Cluster Node containing an exact Transformers multimodal loader, local image input, bounded text generation action, and preview.",
    "text_to_image": "Cluster Node containing the reviewed Janus loader, bounded native image generation action, and image preview.",
    "speech_to_text": "Cluster Node containing an exact reviewed ASR loader, local audio input, bounded transcription action, and preview.",
    "speech_translation": "Cluster Node containing an exact Whisper loader, local audio input, bounded speech translation action, and preview.",
}

def _hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _words(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split()).title()


def _field(source: str, *, default_dtype: str) -> dict[str, Any]:
    name, field_type, required, default, description = _INPUT_FIELDS[source]
    return {
        "name": name,
        "type": field_type,
        "required": required,
        "default": default_dtype if source == "dtype" else default,
        "description": description,
    }


def _block_definition(provider: str, class_name: str, kind: str, description: str) -> dict[str, Any]:
    body = {
        "schemaVersion": 1,
        "provider": provider,
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
        "id": f"transformers.composite-block:{class_name}:{content_hash}",
        "contentHash": content_hash,
    }


def _definition_and_blocks(spec_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    studio_definition = STUDIO_EXECUTION_SPEC_DEFINITIONS[spec_id]
    model_type = str(studio_definition["modelType"])
    mode = str(studio_definition["mode"])
    profile = studio_definition["profile"]
    public_spec = studio_execution_spec_for_pair(model_type, mode)
    if public_spec is None or public_spec["id"] != spec_id:
        raise ValueError(f"Transformers Cluster spec {spec_id!r} is not uniquely resolvable.")
    execution_profiles = tuple(execution_profiles_for_execution(model_type, mode))
    if len(execution_profiles) != 1 or execution_profiles[0].id != public_spec["executionProfileId"]:
        raise ValueError(f"Transformers Cluster spec {spec_id!r} has no unique execution profile.")
    execution_profile = execution_profiles[0]
    runtime_profile_ids = execution_profile.optional_runtime_profile_ids_for_target()
    if len(runtime_profile_ids) != 1 or runtime_profile_ids[0] not in OPTIONAL_RUNTIME_PROFILES:
        raise ValueError(f"Transformers Cluster spec {spec_id!r} has no exact target runtime profile.")
    runtime_profile_id = runtime_profile_ids[0]
    runtime_profile = OPTIONAL_RUNTIME_PROFILES[runtime_profile_id]
    repository = str(profile["default_repo"])
    revision = require_catalog_revision(repository, model_type=model_type)
    binding_sources = sorted({str(source) for _role, _field_name, source in public_spec["bindings"]})

    input_sources = [source for source in binding_sources if source in _INPUT_FIELDS]
    inputs = [_field(source, default_dtype=str(studio_definition["capability"]["defaultDtype"])) for source in input_sources]
    if profile["pipeline_class"] == "AutoModelForCTC":
        timestamp_input = next((field for field in inputs if field["name"] == "timestamps"), None)
        if timestamp_input is not None:
            timestamp_input["default"] = "word"
            timestamp_input["description"] = "CTC timestamp detail: none or word."
    input_by_source = {source: field["name"] for source, field in zip(input_sources, inputs, strict=True)}

    sealed_values = {
        source: value
        for source, value in {
            "artifact": repository,
            "defaultRevision": revision,
            "pipelineClass": public_spec["pipelineClass"],
            "executionProfileId": public_spec["executionProfileId"],
            **_SEALED_VALUES,
        }.items()
        if source in binding_sources
    }
    instance_input_bindings = [
        {"bindingSource": source, "input": input_by_source[source]} for source in input_sources
    ]
    instance_sources = set(input_sources)
    execution_parameter_sources = sorted(
        set(binding_sources) - instance_sources - set(sealed_values)
    )

    role_blocks: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    placements: list[dict[str, Any]] = []
    for order, (role, node_key, _x, _y) in enumerate(public_spec["roles"]):
        action = str(node_key).rsplit(".", 1)[-1]
        block = _block_definition(
            "transformers",
            action,
            "block",
            f"Existing MoDiff node {node_key} used by this reviewed Transformers composite.",
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
        "transformers",
        root_class,
        "sequential",
        "Provider-neutral container for the exact loader, task action, local input, and preview graph.",
    )
    adapter_id = mode
    definition_id = f"transformers.composite:{model_type}:{mode}"
    adapter = {
        "schemaVersion": 1,
        "id": f"transformers.composite-adapter:{model_type}:{mode}",
        "source": "mode",
        "adapterId": adapter_id,
        "upstreamWorkflowId": mode,
        "requiredInputs": [field["name"] for field in inputs if field["required"]],
        "actionSequence": [str(role) for role, _node_key, _x, _y in public_spec["roles"]],
        "stateEdges": [],
        "upstreamBlockSequence": [str(node_key) for _role, node_key, _x, _y in public_spec["roles"]],
    }
    live_proof = bool(profile.get("live_proof"))
    admission = {
        "schemaVersion": 4,
        "id": f"transformers.cluster-admission:{model_type}:{mode}",
        "definitionId": definition_id,
        "studioMode": mode,
        "bindingSources": binding_sources,
        "instanceInputBindings": instance_input_bindings,
        "executionParameterSources": execution_parameter_sources,
        "sealedBindingValues": sealed_values,
        "modelDependencies": [],
        "dynamicFieldActions": [],
        "adapterContractId": adapter["id"],
        "studioExecutionSpec": {
            "id": public_spec["id"],
            "contentHash": public_spec["contentHash"],
            "executionProfileId": public_spec["executionProfileId"],
        },
        "artifact": {"repo": repository, "revision": revision},
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
            "reasons": [
                {
                    "code": "runtime_resource_admission_required",
                    "message": "The materialized Cluster graph must pass exact optional-runtime, artifact, and resource checks before Run is enabled.",
                },
                *(
                    []
                    if live_proof
                    else [
                        {
                            "code": "live_output_review_pending",
                            "message": "The exact Cluster execution route has not received approved visible-frontend live-output evidence.",
                        }
                    ]
                ),
            ],
        },
        "reasons": [],
    }
    output_type = "image" if mode == "text_to_image" else "object"
    body = {
        "schemaVersion": 5,
        "id": definition_id,
        "provider": "transformers",
        "publisher": "huggingface",
        "surface": "transformers_cluster_nodes",
        "definitionKind": "studio_execution_composite",
        "ownership": "library",
        "mutable": False,
        "libraryRevision": revision,
        "pipelineClass": model_type,
        # This field retains the exact underlying execution class for the
        # common Cluster materializer; it is not presented as a pipeline node.
        "blocksClass": str(profile["pipeline_class"]),
        "pipelineKind": "sequential",
        "workflowId": mode,
        "workflowKind": "sequential",
        "taskId": mode,
        "taskContractId": f"transformers.task.{mode}.v1",
        "label": _LABELS[spec_id],
        "description": _DESCRIPTIONS[mode],
        "integrationStatus": "reviewed_transformers_contract",
        "executionClaim": "discovery_only",
        "executionAdmissions": [admission],
        "graphAdapterContracts": [adapter],
        "inputs": inputs,
        "outputs": [
            {
                "name": "image" if output_type == "image" else "result",
                "type": output_type,
                "required": True,
                "default": None,
                "description": "Normalized output from the reviewed Transformers task action.",
            }
        ],
        "requiredInputs": [field["name"] for field in inputs if field["required"]],
        "requiredInputAlternatives": [],
        "stateKeys": [],
        "components": [
            {
                "name": "optional_runtime",
                "type": f"{runtime_profile_id}@{runtime_profile.spec_digest}",
                "creationMethod": "explicit_app_setup",
                "reuseKey": ["optional_runtime"],
            },
            {
                "name": "model",
                "type": str(profile["pipeline_class"]),
                "creationMethod": "from_pretrained",
                "reuseKey": ["model", "revision", "dtype", "device"],
            },
        ],
        "steps": steps,
        "blockContractHash": root_block["contentHash"],
        "rootBlockDefinitionId": root_block["id"],
        "blockPlacements": placements,
    }
    definition = {**body, "contentHash": _hash(body)}
    return definition, [root_block, *role_blocks]


def reviewed_transformers_cluster_catalog() -> dict[str, Any]:
    """Return detached definitions, blocks, and task memberships."""

    definitions: list[dict[str, Any]] = []
    blocks_by_id: dict[str, dict[str, Any]] = {}
    for spec_id in _REVIEWED_SPECS:
        definition, blocks = _definition_and_blocks(spec_id)
        definitions.append(definition)
        for block in blocks:
            existing = blocks_by_id.setdefault(block["id"], block)
            if existing != block:
                raise ValueError("Transformers composite block identity collision.")
    definitions.sort(key=lambda item: item["id"])
    by_task: dict[str, list[str]] = defaultdict(list)
    for definition in definitions:
        by_task[definition["taskId"]].append(definition["id"])
    task_contracts = [
        {
            "schemaVersion": 5,
            "id": f"transformers.task.{task_id}.v1",
            "provider": "transformers",
            "taskId": task_id,
            "label": _words(task_id),
            "definitionIds": definition_ids,
        }
        for task_id, definition_ids in sorted(by_task.items())
    ]
    return deepcopy(
        {
            "definitions": definitions,
            "blockDefinitions": sorted(blocks_by_id.values(), key=lambda item: item["id"]),
            "taskContracts": task_contracts,
        }
    )


def expected_transformers_cluster_admissions() -> dict[str, list[dict[str, Any]]]:
    return {
        definition["id"]: deepcopy(definition["executionAdmissions"])
        for definition in reviewed_transformers_cluster_catalog()["definitions"]
    }
