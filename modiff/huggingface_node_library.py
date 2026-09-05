"""Read-only first-party Hugging Face node-library contracts.

The library is derived from MoDiff's reviewed, no-weight Modular Diffusers
snapshot.  It deliberately separates immutable library definitions from the
user-owned ``/studio/blocks`` store and groups model-specific workflow
definitions under shared task contracts.

Importing this module must not import Diffusers, Transformers, Torch, or any
model stack.  Execution and Auto eligibility remain owned by their existing
reviewed registries; appearing here is a discovery claim only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from modiff.modular_contract_only_registry import (
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME,
    equivalent_modular_targets,
)
from modiff.modular_block_contracts import (
    MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION,
    ModularBlockContractError,
    reviewed_modular_block_snapshot,
    validate_modular_block_snapshot,
)
from modiff.modular_block_role_adapters import build_reviewed_modular_block_role_adapters
from modiff.modular_container_state_adapters import build_reviewed_modular_container_state_adapters
from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.modular_workflow_discovery import (
    load_reviewed_modular_workflow_snapshot,
    validate_modular_workflow_snapshot,
)
from modiff.huggingface_transformers_clusters import reviewed_transformers_cluster_catalog
from modiff.huggingface_diffusers_clusters import reviewed_diffusers_cluster_catalog


HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION = 6
HUGGING_FACE_NODE_LIBRARY_PROVIDER = "diffusers"
HUGGING_FACE_NODE_LIBRARY_PROVIDERS = ("diffusers", "transformers")
HUGGING_FACE_NODE_LIBRARY_PUBLISHER = "huggingface"

_MAX_LIBRARY_BYTES = 8 * 1024 * 1024
_MAX_TASK_CONTRACTS = 256
_MAX_DEFINITIONS = 512
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,255}$")
_DEFINITION_ID = re.compile(
    r"^(?:diffusers\.(?:modular|composite)|transformers\.composite):[A-Za-z_][A-Za-z0-9_]{0,127}:[A-Za-z_][A-Za-z0-9_.-]{0,255}$"
)
_TASK_CONTRACT_ID = re.compile(r"^(?:diffusers|transformers)\.task\.[A-Za-z_][A-Za-z0-9_.-]{0,255}\.v1$")
_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRANSFORMERS_BLOCK_ID = re.compile(
    r"^transformers\.composite-block:[A-Za-z_][A-Za-z0-9_]{0,255}:sha256:[0-9a-f]{64}$"
)
_DIFFUSERS_COMPOSITE_BLOCK_ID = re.compile(
    r"^diffusers\.composite-block:[A-Za-z_][A-Za-z0-9_]{0,255}:sha256:[0-9a-f]{64}$"
)


class HuggingFaceNodeLibraryError(ValueError):
    """The first-party Hugging Face node-library contract is invalid."""


_PUBLISHER_PROMPT_EXAMPLES = {
    "black-forest-labs/FLUX.1-dev": "A cat holding a sign that says hello world",
    "stabilityai/stable-diffusion-xl-base-1.0": "An astronaut riding a green horse",
    "Qwen/Qwen-Image-2512": (
        "A 20-year-old East Asian woman with expressive brown eyes and naturally wavy long hair, wearing a "
        "modern bright outfit at an anime convention. Natural indoor illumination and a casual phone-photo "
        "composition with vivid, fresh detail."
    ),
    "Qwen/Qwen-Image-Edit": "Change the rabbit's color to purple, with a flash light background.",
    "Qwen/Qwen-Image-Edit-2511": (
        "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central "
        "park square."
    ),
    "Tongyi-MAI/Z-Image-Turbo": (
        "Young Chinese woman in red Hanfu with intricate embroidery, impeccable makeup, an elaborate golden "
        "phoenix headdress, and a round folding fan, softly lit outdoors at night with a tiered pagoda behind her."
    ),
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": (
        "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ),
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": (
        "Summer beach vacation style: a white cat wearing sunglasses sits on a surfboard, gazing at the camera "
        "while sea breeze moves its fur against crystal-clear water and distant green hills."
    ),
    "nvidia/Cosmos3-Nano": "A small warehouse robot moves a blue box across a clean floor.",
}

_PUBLISHER_ADDITIONAL_INPUT_EXAMPLES = {
    ("Qwen/Qwen-Image-2512", "text2image"): {
        "negative_prompt": (
            "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，"
            "画面具有AI感。构图混乱。文字模糊，扭曲。"
        ),
    },
}

# Some official Modular repositories publish several workflows whose examples
# are intentionally different. A repository-wide prompt would erase that
# semantic distinction, so preserve the exact pinned Diffusers documentation
# example per workflow. These entries take precedence over the older
# repository-wide model-card table above.
_PUBLISHER_WORKFLOW_PROMPT_EXAMPLES = {
    ("Qwen/Qwen-Image-2512", "text2image"): {
        "prompt": (
            "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—"
            "expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is "
            "either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful "
            "freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, "
            "minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. "
            "Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone "
            "snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm."
        ),
        "label": "Qwen/Qwen-Image-2512 model card example",
        "url": (
            "https://huggingface.co/Qwen/Qwen-Image-2512/blob/"
            "25468b98e3276ca6700de15c6628e51b7de54a26/README.md"
        ),
    },
    ("MiniMaxAI/MiniMax-H3", "t2va"): {
        "prompt": "A red fox trotting through a snowy pine forest, snow crunching underfoot",
        "label": "Hugging Face Diffusers MiniMax-H3 text/video example",
        "url": (
            "https://github.com/huggingface/diffusers/blob/"
            "2f7e0154a9db246e95c9ede43edba7db5b130805/docs/source/en/api/pipelines/minimax_h3.md"
        ),
    },
    ("MiniMaxAI/MiniMax-H3", "fl2va"): {
        "prompt": "A red fox trotting through a snowy pine forest, snow crunching underfoot",
        "label": "Hugging Face Diffusers MiniMax-H3 keyframe example",
        "url": (
            "https://github.com/huggingface/diffusers/blob/"
            "2f7e0154a9db246e95c9ede43edba7db5b130805/docs/source/en/api/pipelines/minimax_h3.md"
        ),
    },
    ("MiniMaxAI/MiniMax-H3", "ref2va"): {
        "prompt": "The character speaks in time with the reference recording, natural lip movement",
        "label": "Hugging Face Diffusers MiniMax-H3 omni-reference example",
        "url": (
            "https://github.com/huggingface/diffusers/blob/"
            "2f7e0154a9db246e95c9ede43edba7db5b130805/docs/source/en/api/pipelines/minimax_h3.md"
        ),
    },
}

_TASK_STARTER_PROMPTS = {
    "text_to_image": "A cinematic photograph of a handcrafted object on a studio table, natural light, crisp detail.",
    "image_to_image": "Preserve the source composition while refining materials, lighting, and fine detail.",
    "edit_image": "Preserve the subject and composition while applying the requested visual edit naturally.",
    "inpaint": "Fill only the masked region so it matches the surrounding lighting, texture, and perspective.",
    "text_to_video": "A cinematic wide shot with clear subject motion, smooth camera movement, and continuous action.",
    "image_to_video": "Animate the source image with coherent subject motion, stable identity, and smooth camera movement.",
    "video_to_video": "Preserve the source timing and motion while applying the requested visual transformation.",
    "text_to_audio": "A polished original composition with a clear musical arc, detailed instrumentation, and a clean mix.",
}


def _suggested_inputs(definition: Mapping[str, Any]) -> dict[str, Any]:
    input_names = {str(field.get("name")) for field in definition.get("inputs", []) if isinstance(field, Mapping)}
    admissions = [item for item in definition.get("executionAdmissions", []) if isinstance(item, Mapping)]
    artifact = next(
        (
            item.get("artifact")
            for item in admissions
            if isinstance(item.get("artifact"), Mapping) and isinstance(item["artifact"].get("repo"), str)
        ),
        None,
    )
    repo = str(artifact["repo"]) if isinstance(artifact, Mapping) else ""
    revision = str(artifact.get("revision") or "") if isinstance(artifact, Mapping) else ""
    workflow_id = str(definition.get("workflowId") or "")
    workflow_example = _PUBLISHER_WORKFLOW_PROMPT_EXAMPLES.get((repo, workflow_id))
    publisher_prompt = (
        str(workflow_example["prompt"])
        if isinstance(workflow_example, Mapping)
        else _PUBLISHER_PROMPT_EXAMPLES.get(repo)
    )
    task_id = str(definition.get("taskId") or "")
    normalized_task_id = re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")
    normalized_task = next((name for name in _TASK_STARTER_PROMPTS if name in normalized_task_id), "")
    prompt = publisher_prompt or _TASK_STARTER_PROMPTS.get(normalized_task)
    values: dict[str, Any] = {}
    if prompt and "prompt" in input_names:
        values["prompt"] = prompt
    if publisher_prompt:
        values.update(
            {
                name: value
                for name, value in _PUBLISHER_ADDITIONAL_INPUT_EXAMPLES.get((repo, workflow_id), {}).items()
                if name in input_names
            }
        )
    source_kind = "publisher_example" if publisher_prompt else "modiff_task_starter"
    source_label = (
        str(workflow_example["label"])
        if isinstance(workflow_example, Mapping)
        else f"{repo} model card example" if publisher_prompt else "MoDiff task starter"
    )
    source_url = (
        str(workflow_example["url"])
        if isinstance(workflow_example, Mapping)
        else (
            f"https://huggingface.co/{repo}/blob/{revision}/README.md"
            if publisher_prompt and repo and re.fullmatch(r"[0-9a-f]{40}", revision)
            else ""
        )
    )
    return {
        "schemaVersion": 1,
        "values": values,
        "source": {"kind": source_kind, "label": source_label, "url": source_url},
    }


def _definition_with_suggestions(definition: Mapping[str, Any]) -> dict[str, Any]:
    enriched = {**_json_clone(definition), "schemaVersion": HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION}
    enriched["suggestedInputs"] = _suggested_inputs(enriched)
    body = {key: item for key, item in enriched.items() if key != "contentHash"}
    enriched["contentHash"] = _content_hash(body)
    return enriched


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _content_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _task_contract_id(task_id: str) -> str:
    return f"diffusers.task.{task_id}.v1"


def _definition_id(pipeline_class: str, workflow_id: str) -> str:
    return f"diffusers.modular:{pipeline_class}:{workflow_id}"


def _words(value: str) -> str:
    value = re.sub(r"ModularPipeline$", "", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return " ".join(value.split())


def _integration_status(pipeline_class: str, workflow_id: str) -> str:
    if equivalent_modular_targets(pipeline_class, workflow_id):
        return "equivalent_standard_route"
    if reviewed_whole_workflow_graph_adapter(pipeline_class, workflow_id) is not None:
        return "reviewed_modular_workflow_route"
    if pipeline_class in CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME:
        return "contract_only"
    if pipeline_class in PINNED_MODULAR_WORKFLOW_TRUTH:
        return "reviewed_modiff_contract"
    raise HuggingFaceNodeLibraryError(f"Unknown reviewed Modular pipeline {pipeline_class!r}.")


_EQUIVALENT_STANDARD_REQUIRED_INPUTS = {
    ("Flux2ModularPipeline", "text2image"): ("prompt",),
    ("Flux2ModularPipeline", "image_conditioned"): ("image", "prompt"),
    ("ErnieImageModularPipeline", "text2image"): ("prompt",),
    ("LTXModularPipeline", "text2video"): ("prompt",),
    ("LTXModularPipeline", "image2video"): ("image", "prompt"),
    ("Wan22ModularPipeline", "default"): ("prompt",),
    ("Wan22Image2VideoModularPipeline", "default"): ("image", "prompt"),
    ("LTX2ModularPipeline", "text2video"): ("prompt",),
    ("LTX2ModularPipeline", "image2video"): ("image", "prompt"),
    ("LTX2ModularPipeline", "condition"): ("conditions", "prompt"),
    ("LTX2ModularPipeline", "in_context"): ("num_frames", "prompt", "reference_conditions"),
}


def _graph_adapter_contracts(
    pipeline_class: str,
    workflow_id: str,
    workflow: Mapping[str, Any],
) -> list[dict[str, Any]]:
    whole_workflow = reviewed_whole_workflow_graph_adapter(pipeline_class, workflow_id)
    if whole_workflow is not None:
        return [
            {
                "schemaVersion": whole_workflow["schemaVersion"],
                "id": (
                    f"diffusers.modular-adapter:{pipeline_class}:{workflow_id}:workflow:"
                    f"{whole_workflow['adapterId']}"
                ),
                "source": "workflow",
                "adapterId": whole_workflow["adapterId"],
                "upstreamWorkflowId": workflow_id,
                "requiredInputs": whole_workflow["requiredInputs"],
                "actionSequence": whole_workflow["actionSequence"],
                "stateEdges": whole_workflow["stateEdges"],
                "upstreamBlockSequence": whole_workflow["upstreamBlockSequence"],
            }
        ]
    truth = PINNED_MODULAR_WORKFLOW_TRUTH.get(pipeline_class)
    if truth is None:
        required_inputs = _EQUIVALENT_STANDARD_REQUIRED_INPUTS.get((pipeline_class, workflow_id))
        if not equivalent_modular_targets(pipeline_class, workflow_id) or required_inputs is None:
            return []
        adapter_id = "equivalent_standard_route"
        return [
            {
                "schemaVersion": 1,
                "id": (
                    f"diffusers.modular-adapter:{pipeline_class}:{workflow_id}:mode:{adapter_id}"
                ),
                "source": "mode",
                "adapterId": adapter_id,
                "upstreamWorkflowId": workflow_id,
                "requiredInputs": list(required_inputs),
                # The standard executor remains one reviewed end-to-end action.
                # Exact upstream block hierarchy is retained separately in
                # blockPlacements and is never misrepresented as split-block
                # execution by this adapter.
                "actionSequence": ["full_pipeline"],
                "stateEdges": [],
                "upstreamBlockSequence": [str(step["path"]) for step in workflow["steps"]],
            }
        ]
    contracts: list[dict[str, Any]] = []
    for source, entries in (("mode", truth.modes), ("state_flow", truth.state_flows)):
        for adapter_id, contract in entries:
            upstream_workflow = contract.upstream_workflow
            if upstream_workflow is None and not truth.workflows:
                upstream_workflow = "default"
            if upstream_workflow != workflow_id:
                continue
            contracts.append(
                {
                    "schemaVersion": 1,
                    "id": (f"diffusers.modular-adapter:{pipeline_class}:{workflow_id}:{source}:{adapter_id}"),
                    "source": source,
                    "adapterId": adapter_id,
                    "upstreamWorkflowId": workflow_id,
                    "requiredInputs": sorted(contract.required_upstream_inputs),
                    "actionSequence": list(contract.action_sequence),
                    "stateEdges": [
                        {
                            "producerAction": edge.producer_action,
                            "producerOutput": edge.producer_output,
                            "consumerAction": edge.consumer_action,
                            "consumerInput": edge.consumer_input,
                        }
                        for edge in contract.state_edges
                    ],
                    "upstreamBlockSequence": list(contract.upstream_block_sequence),
                }
            )
    return sorted(contracts, key=lambda item: item["id"])


def _definition(
    contract: Mapping[str, Any],
    workflow: Mapping[str, Any],
    block_workflow: Mapping[str, Any],
    diffusers_revision: str,
) -> dict[str, Any]:
    pipeline_class = str(contract["pipelineClass"])
    workflow_id = str(workflow["id"])
    task_id = str(workflow["taskId"])
    body = {
        "schemaVersion": HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION,
        "id": _definition_id(pipeline_class, workflow_id),
        "provider": HUGGING_FACE_NODE_LIBRARY_PROVIDER,
        "publisher": HUGGING_FACE_NODE_LIBRARY_PUBLISHER,
        "surface": "diffusers_cluster_nodes",
        "definitionKind": "modular_pipeline_workflow",
        "ownership": "library",
        "mutable": False,
        "libraryRevision": diffusers_revision,
        "pipelineClass": pipeline_class,
        "blocksClass": str(contract["blocksClass"]),
        "pipelineKind": str(contract["kind"]),
        "workflowId": workflow_id,
        "workflowKind": str(workflow["kind"]),
        "taskId": task_id,
        "taskContractId": _task_contract_id(task_id),
        "label": f"{_words(pipeline_class)} — {str(workflow['label'])}",
        "description": str(contract["description"]),
        "integrationStatus": _integration_status(pipeline_class, workflow_id),
        # Runnability and Auto are intentionally joined from the execution and
        # resource registries in a later layer. Structural discovery alone is
        # never an execution claim.
        "executionClaim": "discovery_only",
        "executionAdmissions": [],
        "graphAdapterContracts": _graph_adapter_contracts(pipeline_class, workflow_id, workflow),
        "inputs": _json_clone(workflow["inputs"]),
        "outputs": _json_clone(workflow["outputs"]),
        "requiredInputs": _json_clone(workflow["requiredInputs"]),
        "requiredInputAlternatives": _json_clone(workflow.get("requiredInputAlternatives", [])),
        "stateKeys": _json_clone(workflow["stateKeys"]),
        "components": _json_clone(contract["components"]),
        "steps": _json_clone(workflow["steps"]),
        # These fields retain the exact upstream ``sub_blocks`` membership.
        # ``path`` is an array because an upstream top-level key may itself
        # contain a dot; splitting the legacy display path is ambiguous.
        "blockContractHash": str(block_workflow["contentHash"]),
        "rootBlockDefinitionId": str(block_workflow["rootBlockDefinitionId"]),
        "blockPlacements": _json_clone(block_workflow["placements"]),
    }
    return {**body, "contentHash": _content_hash(body)}


def build_huggingface_node_library(
    snapshot: Mapping[str, Any] | None = None,
    block_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable first-party library from reviewed snapshot data.

    Definitions retain their exact pipeline/workflow identities while sharing
    one generic task-contract reference with other families implementing the
    same task. No import or execution eligibility is inferred here.
    """

    normalized_snapshot = validate_modular_workflow_snapshot(
        snapshot if snapshot is not None else load_reviewed_modular_workflow_snapshot()
    )
    try:
        normalized_block_snapshot = validate_modular_block_snapshot(
            block_snapshot if block_snapshot is not None else reviewed_modular_block_snapshot(),
            workflow_snapshot=normalized_snapshot,
        )
    except ModularBlockContractError as error:
        raise HuggingFaceNodeLibraryError(f"Modular block contract is invalid: {error}") from error
    revision = normalized_snapshot["diffusersRevision"]
    if normalized_block_snapshot["diffusersRevision"] != revision:
        raise HuggingFaceNodeLibraryError("Workflow and block snapshots use different Diffusers revisions.")
    block_workflows = {
        (workflow["pipelineClass"], workflow["workflowId"]): workflow
        for workflow in normalized_block_snapshot["workflows"]
    }
    diffusers_definitions = [
        _definition(
            contract,
            workflow,
            block_workflows[(contract["pipelineClass"], workflow["id"])],
            revision,
        )
        for contract in normalized_snapshot["contracts"]
        for workflow in contract["workflows"]
    ]
    standard_diffusers_catalog = reviewed_diffusers_cluster_catalog(revision)
    all_diffusers_definitions = [*diffusers_definitions, *standard_diffusers_catalog["definitions"]]
    transformers_catalog = reviewed_transformers_cluster_catalog()
    definitions = [*all_diffusers_definitions, *transformers_catalog["definitions"]]
    definitions.sort(key=lambda item: item["id"])

    definitions_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for definition in all_diffusers_definitions:
        definitions_by_task[definition["taskId"]].append(definition)
    diffusers_task_contracts = [
        {
            "schemaVersion": HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION,
            "id": _task_contract_id(task_id),
            "provider": HUGGING_FACE_NODE_LIBRARY_PROVIDER,
            "taskId": task_id,
            "label": _words(task_id).title(),
            "definitionIds": sorted(definition["id"] for definition in task_definitions),
        }
        for task_id, task_definitions in sorted(definitions_by_task.items())
    ]
    task_contracts = sorted(
        [
            *diffusers_task_contracts,
            *[
                {**task_contract, "schemaVersion": HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION}
                for task_contract in transformers_catalog["taskContracts"]
            ],
        ],
        key=lambda item: item["id"],
    )
    library = {
        "schemaVersion": HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION,
        "diffusersRevision": revision,
        "providers": list(HUGGING_FACE_NODE_LIBRARY_PROVIDERS),
        "taskContracts": task_contracts,
        "definitions": definitions,
        "blockDefinitions": _json_clone(
            [
                *normalized_block_snapshot["blockDefinitions"],
                *standard_diffusers_catalog["blockDefinitions"],
                *transformers_catalog["blockDefinitions"],
            ]
        ),
        "blockRoleAdapters": build_reviewed_modular_block_role_adapters(
            normalized_block_snapshot,
            normalized_snapshot,
        ),
        "containerStateAdapters": build_reviewed_modular_container_state_adapters(
            normalized_block_snapshot,
        ),
    }
    from modiff.huggingface_cluster_admission import audit_reviewed_cluster_execution_candidates

    admissions_by_definition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for admission in audit_reviewed_cluster_execution_candidates(library):
        admissions_by_definition[admission["definitionId"]].append(admission)
    for definition in diffusers_definitions:
        definition["executionAdmissions"] = sorted(
            admissions_by_definition.get(definition["id"], []),
            key=lambda admission: admission["id"],
        )
        body = {key: item for key, item in definition.items() if key != "contentHash"}
        definition["contentHash"] = _content_hash(body)
    for index, definition in enumerate(definitions):
        definitions[index] = _definition_with_suggestions(definition)
    return validate_huggingface_node_library(library)


@lru_cache(maxsize=1)
def _cached_huggingface_node_library() -> dict[str, Any]:
    return build_huggingface_node_library()


def reviewed_huggingface_node_library() -> dict[str, Any]:
    """Return a detached copy of the reviewed first-party library."""

    return _json_clone(_cached_huggingface_node_library())


def validate_huggingface_node_library(value: Any) -> dict[str, Any]:
    """Validate and detach a schema-v6 first-party node-library payload."""

    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise HuggingFaceNodeLibraryError(f"Hugging Face node library is not finite JSON: {error}") from error
    if len(encoded.encode("utf-8")) > _MAX_LIBRARY_BYTES:
        raise HuggingFaceNodeLibraryError("Hugging Face node library exceeds the 8 MiB limit.")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schemaVersion",
        "diffusersRevision",
        "providers",
        "taskContracts",
        "definitions",
        "blockDefinitions",
        "blockRoleAdapters",
        "containerStateAdapters",
    }:
        raise HuggingFaceNodeLibraryError("Hugging Face node library has missing or unknown top-level fields.")
    if normalized.get("schemaVersion") != HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION:
        raise HuggingFaceNodeLibraryError(
            f"Hugging Face node library requires schemaVersion {HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION}."
        )
    if normalized.get("providers") != list(HUGGING_FACE_NODE_LIBRARY_PROVIDERS):
        raise HuggingFaceNodeLibraryError("Hugging Face node library has an unsupported provider set.")
    revision = normalized.get("diffusersRevision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise HuggingFaceNodeLibraryError("Hugging Face node library requires an immutable Diffusers revision.")

    task_contracts = normalized.get("taskContracts")
    definitions = normalized.get("definitions")
    block_definitions = normalized.get("blockDefinitions")
    block_role_adapters = normalized.get("blockRoleAdapters")
    container_state_adapters = normalized.get("containerStateAdapters")
    if not isinstance(task_contracts, list) or not 0 < len(task_contracts) <= _MAX_TASK_CONTRACTS:
        raise HuggingFaceNodeLibraryError("Hugging Face node library task contracts are malformed.")
    if not isinstance(definitions, list) or not 0 < len(definitions) <= _MAX_DEFINITIONS:
        raise HuggingFaceNodeLibraryError("Hugging Face node library definitions are malformed.")
    if not isinstance(block_definitions, list):
        raise HuggingFaceNodeLibraryError("Hugging Face node library block definitions are malformed.")
    if not isinstance(block_role_adapters, list):
        raise HuggingFaceNodeLibraryError("Hugging Face node library block-role adapters are malformed.")
    if not isinstance(container_state_adapters, list):
        raise HuggingFaceNodeLibraryError("Hugging Face node library container-state adapters are malformed.")

    diffusers_block_definitions = [
        definition
        for definition in block_definitions
        if isinstance(definition, dict) and str(definition.get("id") or "").startswith("diffusers.modular-block:")
    ]
    diffusers_composite_block_definitions = [
        definition
        for definition in block_definitions
        if isinstance(definition, dict)
        and str(definition.get("id") or "").startswith("diffusers.composite-block:")
    ]
    transformers_block_definitions = [
        definition
        for definition in block_definitions
        if isinstance(definition, dict)
        and str(definition.get("id") or "").startswith("transformers.composite-block:")
    ]
    if (
        len(diffusers_block_definitions)
        + len(diffusers_composite_block_definitions)
        + len(transformers_block_definitions)
        != len(block_definitions)
    ):
        raise HuggingFaceNodeLibraryError("Hugging Face node library contains an unknown block provider.")
    diffusers_composite_blocks_by_id: dict[str, dict[str, Any]] = {}
    transformer_blocks_by_id: dict[str, dict[str, Any]] = {}
    composite_block_keys = {
        "schemaVersion",
        "provider",
        "id",
        "className",
        "kind",
        "description",
        "inputs",
        "variadicInputs",
        "requiredInputs",
        "outputs",
        "components",
        "configs",
        "contentHash",
    }
    for provider, prefix, id_pattern, selected_blocks, blocks_by_id in (
        (
            "diffusers",
            "diffusers.composite-block",
            _DIFFUSERS_COMPOSITE_BLOCK_ID,
            diffusers_composite_block_definitions,
            diffusers_composite_blocks_by_id,
        ),
        (
            "transformers",
            "transformers.composite-block",
            _TRANSFORMERS_BLOCK_ID,
            transformers_block_definitions,
            transformer_blocks_by_id,
        ),
    ):
        for block in selected_blocks:
            if (
                set(block) != composite_block_keys
                or block.get("schemaVersion") != 1
                or block.get("provider") != provider
                or block.get("kind") not in {"auto", "sequential", "loop", "block"}
                or not isinstance(block.get("className"), str)
                or _IDENTIFIER.fullmatch(block["className"]) is None
                or not isinstance(block.get("id"), str)
                or id_pattern.fullmatch(block["id"]) is None
                or not isinstance(block.get("contentHash"), str)
                or _CONTENT_HASH.fullmatch(block["contentHash"]) is None
                or block["id"] != f"{prefix}:{block['className']}:{block['contentHash']}"
                or any(
                    not isinstance(block.get(key), list)
                    for key in ("inputs", "variadicInputs", "requiredInputs", "outputs", "components", "configs")
                )
            ):
                raise HuggingFaceNodeLibraryError(f"{provider.title()} composite block definition is malformed.")
            body = {key: item for key, item in block.items() if key not in {"id", "contentHash"}}
            if block["contentHash"] != _content_hash(body) or block["id"] in blocks_by_id:
                raise HuggingFaceNodeLibraryError(f"{provider.title()} composite block content hash is invalid.")
            blocks_by_id[block["id"]] = block

    standard_diffusers_catalog = reviewed_diffusers_cluster_catalog(revision)
    expected_diffusers_composite_definitions = {
        definition["id"]: _definition_with_suggestions(definition)
        for definition in standard_diffusers_catalog["definitions"]
    }
    if diffusers_composite_block_definitions != standard_diffusers_catalog["blockDefinitions"]:
        raise HuggingFaceNodeLibraryError("Standard Diffusers composite block catalog is not the reviewed contract.")

    transformer_catalog = reviewed_transformers_cluster_catalog()
    expected_transformer_definitions = {
        definition["id"]: _definition_with_suggestions(definition)
        for definition in transformer_catalog["definitions"]
    }
    if transformers_block_definitions != transformer_catalog["blockDefinitions"]:
        raise HuggingFaceNodeLibraryError("Transformers composite block catalog is not the reviewed contract.")

    definition_ids: set[str] = set()
    definitions_by_task: dict[tuple[str, str], list[str]] = defaultdict(list)
    definition_keys = {
        "schemaVersion",
        "id",
        "provider",
        "publisher",
        "surface",
        "definitionKind",
        "ownership",
        "mutable",
        "libraryRevision",
        "pipelineClass",
        "blocksClass",
        "pipelineKind",
        "workflowId",
        "workflowKind",
        "taskId",
        "taskContractId",
        "label",
        "description",
        "integrationStatus",
        "executionClaim",
        "executionAdmissions",
        "graphAdapterContracts",
        "inputs",
        "outputs",
        "requiredInputs",
        "requiredInputAlternatives",
        "stateKeys",
        "components",
        "steps",
        "blockContractHash",
        "rootBlockDefinitionId",
        "blockPlacements",
        "suggestedInputs",
        "contentHash",
    }
    reconstructed_block_workflows: list[dict[str, Any]] = []
    for definition in definitions:
        if not isinstance(definition, dict) or set(definition) != definition_keys:
            raise HuggingFaceNodeLibraryError("Hugging Face node definition is malformed.")
        definition_id = definition.get("id")
        if not isinstance(definition_id, str) or _DEFINITION_ID.fullmatch(definition_id) is None:
            raise HuggingFaceNodeLibraryError("Hugging Face node definition id is invalid.")
        if definition_id in definition_ids:
            raise HuggingFaceNodeLibraryError("Hugging Face node definition ids must be unique.")
        definition_ids.add(definition_id)
        suggested_inputs = definition.get("suggestedInputs")
        if (
            not isinstance(suggested_inputs, dict)
            or set(suggested_inputs) != {"schemaVersion", "values", "source"}
            or suggested_inputs.get("schemaVersion") != 1
            or not isinstance(suggested_inputs.get("values"), dict)
            or not isinstance(suggested_inputs.get("source"), dict)
            or set(suggested_inputs["source"]) != {"kind", "label", "url"}
            or suggested_inputs["source"].get("kind") not in {"publisher_example", "modiff_task_starter"}
            or not isinstance(suggested_inputs["source"].get("label"), str)
            or not suggested_inputs["source"]["label"]
            or len(suggested_inputs["source"]["label"]) > 512
            or not isinstance(suggested_inputs["source"].get("url"), str)
            or len(suggested_inputs["source"]["url"]) > 2048
        ):
            raise HuggingFaceNodeLibraryError("Hugging Face node definition suggested inputs are malformed.")
        input_names = {
            field.get("name")
            for field in definition.get("inputs", [])
            if isinstance(field, dict) and isinstance(field.get("name"), str)
        }
        if any(key not in input_names for key in suggested_inputs["values"]):
            raise HuggingFaceNodeLibraryError("Hugging Face suggested input does not name a declared input.")
        try:
            json.dumps(suggested_inputs["values"], allow_nan=False)
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise HuggingFaceNodeLibraryError(
                f"Hugging Face suggested inputs are not finite JSON: {error}"
            ) from error
        if suggested_inputs != _suggested_inputs(definition):
            raise HuggingFaceNodeLibraryError(
                "Hugging Face suggested inputs are not the reviewed publisher example or MoDiff task starter."
            )
        provider = definition.get("provider")
        if provider == "diffusers" and definition.get("definitionKind") == "studio_execution_composite":
            if definition != expected_diffusers_composite_definitions.get(definition_id):
                raise HuggingFaceNodeLibraryError(
                    "Standard Diffusers Cluster Node definition is not the exact reviewed composite contract."
                )
            content_hash = definition.get("contentHash")
            body = {key: item for key, item in definition.items() if key != "contentHash"}
            if (
                not isinstance(content_hash, str)
                or _CONTENT_HASH.fullmatch(content_hash) is None
                or content_hash != _content_hash(body)
            ):
                raise HuggingFaceNodeLibraryError("Standard Diffusers Cluster Node content hash is invalid.")
            definitions_by_task[("diffusers", definition["taskId"])].append(definition_id)
            continue
        if provider == "transformers":
            if definition != expected_transformer_definitions.get(definition_id):
                raise HuggingFaceNodeLibraryError(
                    "Transformers Cluster Node definition is not the exact reviewed composite contract."
                )
            content_hash = definition.get("contentHash")
            body = {key: item for key, item in definition.items() if key != "contentHash"}
            if (
                not isinstance(content_hash, str)
                or _CONTENT_HASH.fullmatch(content_hash) is None
                or content_hash != _content_hash(body)
            ):
                raise HuggingFaceNodeLibraryError("Transformers Cluster Node content hash is invalid.")
            definitions_by_task[("transformers", definition["taskId"])].append(definition_id)
            continue
        if (
            definition.get("schemaVersion") != HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION
            or definition.get("provider") != HUGGING_FACE_NODE_LIBRARY_PROVIDER
            or definition.get("publisher") != HUGGING_FACE_NODE_LIBRARY_PUBLISHER
            or definition.get("surface") != "diffusers_cluster_nodes"
            or definition.get("definitionKind") != "modular_pipeline_workflow"
            or definition.get("ownership") != "library"
            or definition.get("mutable") is not False
            or definition.get("libraryRevision") != revision
            or definition.get("executionClaim") != "discovery_only"
        ):
            raise HuggingFaceNodeLibraryError("Hugging Face node definition ownership or provenance is invalid.")
        for key in ("pipelineClass", "blocksClass", "workflowId", "taskId"):
            if not isinstance(definition.get(key), str) or _IDENTIFIER.fullmatch(definition[key]) is None:
                raise HuggingFaceNodeLibraryError(f"Hugging Face node definition {key} is invalid.")
        if definition.get("pipelineKind") not in {"auto", "sequential", "loop", "block"} or definition.get(
            "workflowKind"
        ) not in {"auto", "sequential", "loop", "block"}:
            raise HuggingFaceNodeLibraryError("Hugging Face node definition block kind is invalid.")
        task_contract_id = definition.get("taskContractId")
        if task_contract_id != _task_contract_id(definition["taskId"]):
            raise HuggingFaceNodeLibraryError("Hugging Face node definition task contract is invalid.")
        if not isinstance(task_contract_id, str) or _TASK_CONTRACT_ID.fullmatch(task_contract_id) is None:
            raise HuggingFaceNodeLibraryError("Hugging Face node definition task contract id is invalid.")
        if definition.get("integrationStatus") not in {
            "reviewed_modiff_contract",
            "reviewed_modular_workflow_route",
            "contract_only",
            "equivalent_standard_route",
        }:
            raise HuggingFaceNodeLibraryError("Hugging Face node definition integration status is invalid.")
        if not all(
            isinstance(definition.get(key), list)
            for key in (
                "inputs",
                "outputs",
                "requiredInputs",
                "requiredInputAlternatives",
                "stateKeys",
                "components",
                "steps",
                "blockPlacements",
                "graphAdapterContracts",
                "executionAdmissions",
            )
        ):
            raise HuggingFaceNodeLibraryError("Hugging Face node definition collections are malformed.")
        if definition["graphAdapterContracts"] != _graph_adapter_contracts(
            definition["pipelineClass"], definition["workflowId"], definition
        ):
            raise HuggingFaceNodeLibraryError("Hugging Face node definition graph adapter contracts are invalid.")
        block_contract_hash = definition.get("blockContractHash")
        if not isinstance(block_contract_hash, str) or _CONTENT_HASH.fullmatch(block_contract_hash) is None:
            raise HuggingFaceNodeLibraryError("Hugging Face node definition block contract hash is invalid.")
        reconstructed_block_workflows.append(
            {
                "schemaVersion": MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION,
                "pipelineClass": definition["pipelineClass"],
                "workflowId": definition["workflowId"],
                "workflowKind": definition["workflowKind"],
                "rootBlockDefinitionId": definition["rootBlockDefinitionId"],
                "placements": definition["blockPlacements"],
                "contentHash": block_contract_hash,
            }
        )
        content_hash = definition.get("contentHash")
        body = {key: item for key, item in definition.items() if key != "contentHash"}
        if not isinstance(content_hash, str) or _CONTENT_HASH.fullmatch(content_hash) is None:
            raise HuggingFaceNodeLibraryError("Hugging Face node definition content hash is invalid.")
        if content_hash != _content_hash(body):
            raise HuggingFaceNodeLibraryError("Hugging Face node definition content hash does not match its body.")
        definitions_by_task[("diffusers", definition["taskId"])].append(definition_id)

    task_ids: set[tuple[str, str]] = set()
    referenced_definition_ids: set[str] = set()
    task_contract_keys = {"schemaVersion", "id", "provider", "taskId", "label", "definitionIds"}
    for task_contract in task_contracts:
        if not isinstance(task_contract, dict) or set(task_contract) != task_contract_keys:
            raise HuggingFaceNodeLibraryError("Hugging Face task contract is malformed.")
        task_id = task_contract.get("taskId")
        provider = task_contract.get("provider")
        task_key = (str(provider), str(task_id))
        if (
            provider not in HUGGING_FACE_NODE_LIBRARY_PROVIDERS
            or not isinstance(task_id, str)
            or _IDENTIFIER.fullmatch(task_id) is None
            or task_key in task_ids
        ):
            raise HuggingFaceNodeLibraryError("Hugging Face task contract id is invalid or duplicated.")
        task_ids.add(task_key)
        if (
            task_contract.get("schemaVersion") != HUGGING_FACE_NODE_LIBRARY_SCHEMA_VERSION
            or task_contract.get("id") != f"{provider}.task.{task_id}.v1"
            or not isinstance(task_contract.get("label"), str)
            or not task_contract["label"]
        ):
            raise HuggingFaceNodeLibraryError("Hugging Face task contract provenance is invalid.")
        expected_ids = definitions_by_task.get(task_key, [])
        if task_contract.get("definitionIds") != expected_ids:
            raise HuggingFaceNodeLibraryError("Hugging Face task contract definition membership is invalid.")
        referenced_definition_ids.update(expected_ids)
    if set(definitions_by_task) != task_ids or referenced_definition_ids != definition_ids:
        raise HuggingFaceNodeLibraryError("Hugging Face task contracts do not cover every definition exactly once.")

    from modiff.huggingface_cluster_admission import audit_reviewed_cluster_execution_candidates

    expected_admissions_by_definition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for admission in audit_reviewed_cluster_execution_candidates(normalized):
        expected_admissions_by_definition[admission["definitionId"]].append(admission)
    for definition in definitions:
        if definition["provider"] != "diffusers" or definition["definitionKind"] != "modular_pipeline_workflow":
            continue
        expected_admissions = sorted(
            expected_admissions_by_definition.get(definition["id"], []),
            key=lambda admission: admission["id"],
        )
        if definition["executionAdmissions"] != expected_admissions:
            raise HuggingFaceNodeLibraryError("Hugging Face node definition execution admissions are invalid.")

    # Reuse the pinned companion-snapshot validator instead of maintaining a
    # second, weaker interpretation of block inputs, outputs, components,
    # configs, path segments, ordering, hashes, or workflow coverage here.
    reconstructed_block_snapshot = {
        "schemaVersion": MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION,
        "diffusersRevision": revision,
        "blockDefinitions": diffusers_block_definitions,
        "workflows": sorted(
            reconstructed_block_workflows,
            key=lambda workflow: (workflow["pipelineClass"], workflow["workflowId"]),
        ),
    }
    try:
        validated_block_snapshot = validate_modular_block_snapshot(reconstructed_block_snapshot)
    except ModularBlockContractError as error:
        raise HuggingFaceNodeLibraryError(f"Hugging Face block library is invalid: {error}") from error
    expected_block_role_adapters = build_reviewed_modular_block_role_adapters(validated_block_snapshot)
    if block_role_adapters != expected_block_role_adapters:
        raise HuggingFaceNodeLibraryError(
            "Hugging Face node library block-role adapters are not the exact reviewed contracts."
        )
    expected_container_state_adapters = build_reviewed_modular_container_state_adapters(
        validated_block_snapshot
    )
    if container_state_adapters != expected_container_state_adapters:
        raise HuggingFaceNodeLibraryError(
            "Hugging Face node library container-state adapters are not the exact reviewed contracts."
        )
    return normalized
