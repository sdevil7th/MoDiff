"""No-download structural admission research for the remaining Cosmos 3 routes.

The constants in this module are intentionally narrower than a runtime or
publication contract.  They record what MoDiff can honestly materialize from
the official Diffusers source pinned at :data:`PINNED_DIFFUSERS_REVISION` and
the checked-in Cosmos 3 artifact review.  Nothing here authorizes a model
download, license acceptance, optional-runtime installation, execution, Auto,
Gallery, or public promotion.

The first tranche contains the five non-action Nano routes.  The second
contains the two Distilled workflows for which the artifact review pins an
exact route-specific 4-step checkpoint.  Both tranches remain structural-only:
the mandatory guardrail, heavyweight snapshots, tensor-parallel resource
profile, execution, and publication gates stay closed.  Action workflows
remain closed until an opaque upstream ``CosmosActionCondition`` can be
composed from typed V2 inputs without changing the official workflow
interface.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


PINNED_DIFFUSERS_REVISION = "2f7e0154a9db246e95c9ede43edba7db5b130805"

COSMOS3_GUARDRAIL = {
    "id": "cosmos3-mandatory-safety-guardrail",
    "kind": "safety_checker",
    "repository": "nvidia/Cosmos-Guardrail1",
    "revision": "d6d4bfa899a71454a700907664f3e88f503950cf",
    "package": "cosmos-guardrail",
    "packageVersion": "0.3.1",
}

_COSMOS3_DISTILLED_COMMON_DIFFUSERS_FILES = (
    "README.md",
    "SAFETY.md",
    "modular_model_index.json",
    "scheduler/scheduler_config.json",
    "text_tokenizer/added_tokens.json",
    "text_tokenizer/chat_template.jinja",
    "text_tokenizer/merges.txt",
    "text_tokenizer/special_tokens_map.json",
    "text_tokenizer/tokenizer.json",
    "text_tokenizer/tokenizer_config.json",
    "text_tokenizer/vocab.json",
    "transformer/config.json",
    *tuple(
        f"transformer/diffusion_pytorch_model-{index:05d}-of-00027.safetensors"
        for index in range(1, 28)
    ),
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
COSMOS3_DISTILLED_TEXT2IMAGE_DIFFUSERS_FILES = (
    *_COSMOS3_DISTILLED_COMMON_DIFFUSERS_FILES[:4],
    "sound_tokenizer/config.json",
    "sound_tokenizer/diffusion_pytorch_model.safetensors",
    *_COSMOS3_DISTILLED_COMMON_DIFFUSERS_FILES[4:],
)
COSMOS3_DISTILLED_IMAGE2VIDEO_DIFFUSERS_FILES = _COSMOS3_DISTILLED_COMMON_DIFFUSERS_FILES

COSMOS3_ARTIFACTS = {
    "nano": {
        "pipelineClass": "Cosmos3OmniModularPipeline",
        "repository": "nvidia/Cosmos3-Nano",
        "revision": "7a312c868bcce8e40b3eb40861300a9d0ba3fde1",
        "blocksClass": "Cosmos3OmniBlocks",
        "snapshotByteSize": 34_986_890_561,
    },
    "distilled_text2image": {
        "pipelineClass": "Cosmos3DistilledModularPipeline",
        "repository": "nvidia/Cosmos3-Super-Text2Image-4Step",
        "revision": "aa0d5a57b7b045d68daa60fbacd84ec723c7cb7b",
        "blocksClass": "Cosmos3DistilledBlocks",
        "snapshotByteSize": 131_423_464_104,
        "weightByteSize": 131_391_926_304,
        "downloadFiles": COSMOS3_DISTILLED_TEXT2IMAGE_DIFFUSERS_FILES,
    },
    "distilled_image2video": {
        "pipelineClass": "Cosmos3DistilledModularPipeline",
        "repository": "nvidia/Cosmos3-Super-Image2Video-4Step",
        "revision": "cd55ce81bc5cea51a09c37cd7652144e7278f049",
        "blocksClass": "Cosmos3DistilledBlocks",
        "snapshotByteSize": 129_446_040_640,
        "weightByteSize": 129_405_417_712,
        "downloadFiles": COSMOS3_DISTILLED_IMAGE2VIDEO_DIFFUSERS_FILES,
    },
}

_OMNI_ACTIONS_WITHOUT_CONDITIONING = (
    "workflow_cosmos3_omni_text_encoder",
    "workflow_cosmos3_omni_denoise",
    "workflow_cosmos3_omni_decoder",
    "workflow_cosmos3_omni_after_decode",
)
_OMNI_ACTIONS_WITH_CONDITIONING = (
    "workflow_cosmos3_omni_text_encoder",
    "workflow_cosmos3_omni_vae_encoder",
    "workflow_cosmos3_omni_denoise",
    "workflow_cosmos3_omni_decoder",
    "workflow_cosmos3_omni_after_decode",
)
_DISTILLED_ACTIONS_WITHOUT_CONDITIONING = (
    "workflow_cosmos3_distilled_text_encoder",
    "workflow_cosmos3_distilled_denoise",
    "workflow_cosmos3_distilled_decoder",
)
_DISTILLED_ACTIONS_WITH_CONDITIONING = (
    "workflow_cosmos3_distilled_text_encoder",
    "workflow_cosmos3_distilled_vae_encoder",
    "workflow_cosmos3_distilled_denoise",
    "workflow_cosmos3_distilled_decoder",
)

_ACTION_TO_ROLE = {
    "workflow_cosmos3_distilled_text_encoder": "prompt",
    "workflow_cosmos3_distilled_vae_encoder": "imageEncode",
    "workflow_cosmos3_distilled_denoise": "denoise",
    "workflow_cosmos3_distilled_decoder": "decode",
    "workflow_cosmos3_omni_text_encoder": "prompt",
    "workflow_cosmos3_omni_vae_encoder": "imageEncode",
    "workflow_cosmos3_omni_denoise": "denoise",
    "workflow_cosmos3_omni_decoder": "decode",
    "workflow_cosmos3_omni_after_decode": "afterDecode",
}

_ACTION_TO_BLOCK = {
    "workflow_cosmos3_distilled_text_encoder": "text_encoder",
    "workflow_cosmos3_distilled_vae_encoder": "vae_encoder",
    "workflow_cosmos3_distilled_denoise": "denoise",
    "workflow_cosmos3_distilled_decoder": "decode",
    "workflow_cosmos3_omni_text_encoder": "text_encoder",
    "workflow_cosmos3_omni_vae_encoder": "vae_encoder",
    "workflow_cosmos3_omni_denoise": "denoise",
    "workflow_cosmos3_omni_decoder": "decode",
    "workflow_cosmos3_omni_after_decode": "after_decode",
}

_COMMON_STRUCTURAL_BLOCKERS = (
    "cosmos_guardrail_hub_access_and_license_acknowledgement",
    "cosmos_guardrail_0_3_1_optional_runtime_admission",
    "models_loader_guardrail_component_integration",
    "snapshot_installation_and_exact_resource_qualification",
    "visible_frontend_output_approval",
    "runtime_auto_gallery_and_publication_authority",
    "remote_heavy_hardware_execution",
    "physical_macos_execution",
)


def _state_edges(actions: tuple[str, ...]) -> tuple[tuple[str, str, str, str], ...]:
    """Project the official PipelineState chain onto reviewed Studio roles."""

    return tuple(
        (_ACTION_TO_ROLE[producer], "state_out", _ACTION_TO_ROLE[consumer], "state_in")
        for producer, consumer in zip(actions, actions[1:])
    )


def _action_roles(actions: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_ACTION_TO_ROLE[action] for action in actions)


def _blocks(actions: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_ACTION_TO_BLOCK[action] for action in actions)


def _route(
    *,
    pipeline_class: str,
    workflow_id: str,
    task_id: str,
    required_inputs: tuple[str, ...],
    actions: tuple[str, ...],
    semantic_outputs: tuple[str, ...],
    artifact_key: str | None,
    tranche: str,
    sealed_defaults: Mapping[str, Any],
    route_blockers: tuple[str, ...] = (),
) -> dict[str, Any]:
    artifact = COSMOS3_ARTIFACTS.get(artifact_key) if artifact_key is not None else None
    return {
        "pipelineClass": pipeline_class,
        "workflowId": workflow_id,
        "taskId": task_id,
        "requiredInputs": required_inputs,
        "actionSequence": actions,
        "actionRoles": _action_roles(actions),
        "upstreamBlockSequence": _blocks(actions),
        "stateEdges": _state_edges(actions),
        "semanticOutputs": semantic_outputs,
        "artifact": deepcopy(artifact),
        "tranche": tranche,
        "sealedDefaults": dict(sealed_defaults),
        "modelDependencies": (deepcopy(COSMOS3_GUARDRAIL),),
        "blockers": (*route_blockers, *_COMMON_STRUCTURAL_BLOCKERS),
        "claim": "structural_research_only",
    }


_OMNI_DEFAULTS = {
    "dtype": "bfloat16",
    "quantization": None,
    "offloadMode": "none",
    "autoOffload": False,
    "width": 1280,
    "height": 720,
    "numFrames": 189,
    "fps": 24.0,
    "numInferenceSteps": 35,
    "guidanceScale": 6.0,
    "seed": 0,
    "useSystemPrompt": True,
    "addResolutionTemplate": True,
    "addDurationTemplate": True,
    "outputType": "pil",
    "trustRemoteCode": False,
}

_DISTILLED_DEFAULTS = {
    "dtype": "bfloat16",
    "quantization": None,
    "offloadMode": "none",
    "autoOffload": False,
    "width": 1280,
    "height": 720,
    "numFrames": 189,
    "fps": 24.0,
    "numInferenceSteps": 4,
    "guidanceScale": 1.0,
    "seed": 0,
    "useSystemPrompt": True,
    "addResolutionTemplate": True,
    "addDurationTemplate": True,
    "outputType": "pil",
    "trustRemoteCode": False,
}

_ACTION_DEFAULTS = {
    **_OMNI_DEFAULTS,
    # These are the installed official block/node defaults, not a runtime or
    # quality recommendation for any action route.
    "numInferenceSteps": 50,
    "guidanceScale": 6.0,
    "actionChunkSize": 16,
    "actionDomain": "droid_lerobot",
    "actionResolutionTier": 480,
    "actionViewPoint": "ego_view",
}


COSMOS3_REMAINING_ROUTE_RESEARCH: dict[tuple[str, str], dict[str, Any]] = {
    # Coherent immediate tranche: same admitted Nano artifact and exact
    # package-owned workflow nodes as the existing text2image/text2video pair.
    ("Cosmos3OmniModularPipeline", "image2video"): _route(
        pipeline_class="Cosmos3OmniModularPipeline",
        workflow_id="image2video",
        task_id="image_to_video",
        required_inputs=("image", "num_inference_steps", "prompt"),
        actions=_OMNI_ACTIONS_WITH_CONDITIONING,
        semantic_outputs=("videos",),
        artifact_key="nano",
        tranche="nano_non_action_media",
        sealed_defaults=_OMNI_DEFAULTS,
    ),
    ("Cosmos3OmniModularPipeline", "video2video"): _route(
        pipeline_class="Cosmos3OmniModularPipeline",
        workflow_id="video2video",
        task_id="video_to_video",
        required_inputs=("num_inference_steps", "prompt", "video"),
        actions=_OMNI_ACTIONS_WITH_CONDITIONING,
        semantic_outputs=("videos",),
        artifact_key="nano",
        tranche="nano_non_action_media",
        sealed_defaults=_OMNI_DEFAULTS,
    ),
    ("Cosmos3OmniModularPipeline", "text2video_with_sound"): _route(
        pipeline_class="Cosmos3OmniModularPipeline",
        workflow_id="text2video_with_sound",
        task_id="text_to_video_with_audio",
        required_inputs=("num_inference_steps", "prompt"),
        actions=_OMNI_ACTIONS_WITHOUT_CONDITIONING,
        semantic_outputs=("videos", "sound", "sampling_rate"),
        artifact_key="nano",
        tranche="nano_non_action_media",
        sealed_defaults={**_OMNI_DEFAULTS, "enableSound": True},
    ),
    ("Cosmos3OmniModularPipeline", "image2video_with_sound"): _route(
        pipeline_class="Cosmos3OmniModularPipeline",
        workflow_id="image2video_with_sound",
        task_id="image_to_video_with_audio",
        required_inputs=("image", "num_inference_steps", "prompt"),
        actions=_OMNI_ACTIONS_WITH_CONDITIONING,
        semantic_outputs=("videos", "sound", "sampling_rate"),
        artifact_key="nano",
        tranche="nano_non_action_media",
        sealed_defaults={**_OMNI_DEFAULTS, "enableSound": True},
    ),
    ("Cosmos3OmniModularPipeline", "video2video_with_sound"): _route(
        pipeline_class="Cosmos3OmniModularPipeline",
        workflow_id="video2video_with_sound",
        task_id="video_to_video_with_audio",
        required_inputs=("num_inference_steps", "prompt", "video"),
        actions=_OMNI_ACTIONS_WITH_CONDITIONING,
        semantic_outputs=("videos", "sound", "sampling_rate"),
        artifact_key="nano",
        tranche="nano_non_action_media",
        sealed_defaults={**_OMNI_DEFAULTS, "enableSound": True},
    ),
    # These are the only two Distilled workflows with exact reviewed 4-step
    # checkpoints. Their route-specific repository selection and fixed
    # schedule are part of the structural admission; this still grants no
    # runtime or publication authority.
    ("Cosmos3DistilledModularPipeline", "text2image"): _route(
        pipeline_class="Cosmos3DistilledModularPipeline",
        workflow_id="text2image",
        task_id="text_to_image",
        required_inputs=("prompt",),
        actions=_DISTILLED_ACTIONS_WITHOUT_CONDITIONING,
        semantic_outputs=("image",),
        artifact_key="distilled_text2image",
        tranche="distilled_exact_artifact_structural_admission",
        sealed_defaults={**_DISTILLED_DEFAULTS, "numFrames": 1},
        route_blockers=("distilled_tensor_parallel_resource_qualification",),
    ),
    ("Cosmos3DistilledModularPipeline", "image2video"): _route(
        pipeline_class="Cosmos3DistilledModularPipeline",
        workflow_id="image2video",
        task_id="image_to_video",
        required_inputs=("image", "prompt"),
        actions=_DISTILLED_ACTIONS_WITH_CONDITIONING,
        semantic_outputs=("videos",),
        artifact_key="distilled_image2video",
        tranche="distilled_exact_artifact_structural_admission",
        sealed_defaults=_DISTILLED_DEFAULTS,
        route_blockers=("distilled_tensor_parallel_resource_qualification",),
    ),
    # No reviewed Distilled checkpoint maps to either workflow.  An Omni/Super
    # checkpoint cannot be relabeled as a Distilled blocks artifact.
    ("Cosmos3DistilledModularPipeline", "text2video"): _route(
        pipeline_class="Cosmos3DistilledModularPipeline",
        workflow_id="text2video",
        task_id="text_to_video",
        required_inputs=("prompt",),
        actions=_DISTILLED_ACTIONS_WITHOUT_CONDITIONING,
        semantic_outputs=("videos",),
        artifact_key=None,
        tranche="blocked_no_exact_artifact",
        sealed_defaults=_DISTILLED_DEFAULTS,
        route_blockers=("no_reviewed_distilled_text2video_checkpoint",),
    ),
    ("Cosmos3DistilledModularPipeline", "video2video"): _route(
        pipeline_class="Cosmos3DistilledModularPipeline",
        workflow_id="video2video",
        task_id="video_to_video",
        required_inputs=("prompt", "video"),
        actions=_DISTILLED_ACTIONS_WITH_CONDITIONING,
        semantic_outputs=("videos",),
        artifact_key=None,
        tranche="blocked_no_exact_artifact",
        sealed_defaults=_DISTILLED_DEFAULTS,
        route_blockers=("no_reviewed_distilled_video2video_checkpoint",),
    ),
    # The pinned upstream interface requires a CosmosActionCondition object.
    # The current MoDiff node can construct one internally, but V2 has no
    # reviewed projection proving that image/video/raw-action fields compose
    # the official opaque `action` input while preserving the public interface.
    ("Cosmos3OmniModularPipeline", "action_policy"): _route(
        pipeline_class="Cosmos3OmniModularPipeline",
        workflow_id="action_policy",
        task_id="action_policy",
        required_inputs=("action", "num_inference_steps", "prompt"),
        actions=_OMNI_ACTIONS_WITH_CONDITIONING,
        semantic_outputs=("videos", "action"),
        artifact_key="nano",
        tranche="blocked_typed_action_composition",
        sealed_defaults={**_ACTION_DEFAULTS, "actionMode": "policy"},
        route_blockers=(
            "typed_cosmos_action_condition_composition",
            "v2_effective_interface_projection_for_opaque_action",
            "dual_video_and_action_output_contract",
            "no_reviewed_publisher_policy_starter",
        ),
    ),
    ("Cosmos3OmniModularPipeline", "action_forward_dynamics"): _route(
        pipeline_class="Cosmos3OmniModularPipeline",
        workflow_id="action_forward_dynamics",
        task_id="action_forward_dynamics",
        required_inputs=("action", "num_inference_steps", "prompt"),
        actions=_OMNI_ACTIONS_WITH_CONDITIONING,
        semantic_outputs=("videos",),
        artifact_key="nano",
        tranche="blocked_typed_action_composition",
        sealed_defaults={**_ACTION_DEFAULTS, "actionMode": "forward_dynamics"},
        route_blockers=(
            "typed_cosmos_action_condition_composition",
            "v2_effective_interface_projection_for_opaque_action",
            "required_nonempty_finite_raw_action_matrix_and_domain_width_contract",
        ),
    ),
    ("Cosmos3OmniModularPipeline", "action_inverse_dynamics"): _route(
        pipeline_class="Cosmos3OmniModularPipeline",
        workflow_id="action_inverse_dynamics",
        task_id="action_inverse_dynamics",
        required_inputs=("action", "num_inference_steps", "prompt"),
        actions=_OMNI_ACTIONS_WITH_CONDITIONING,
        semantic_outputs=("action",),
        artifact_key="nano",
        tranche="blocked_typed_action_composition",
        sealed_defaults={**_ACTION_DEFAULTS, "actionMode": "inverse_dynamics"},
        route_blockers=(
            "typed_cosmos_action_condition_composition",
            "v2_effective_interface_projection_for_opaque_action",
            "action_only_output_sink_and_serialization_contract",
            "source_video_frame_count_must_equal_action_chunk_size_plus_one",
        ),
    ),
}


def cosmos3_remaining_route_research() -> dict[tuple[str, str], dict[str, Any]]:
    """Return a detached copy of all twelve remaining route contracts."""

    return deepcopy(COSMOS3_REMAINING_ROUTE_RESEARCH)


def cosmos3_structural_admission_tranches() -> dict[str, tuple[tuple[str, str], ...]]:
    """Group routes by the next honest structural-admission boundary."""

    grouped: dict[str, list[tuple[str, str]]] = {}
    for identity, contract in COSMOS3_REMAINING_ROUTE_RESEARCH.items():
        grouped.setdefault(contract["tranche"], []).append(identity)
    return {name: tuple(sorted(identities)) for name, identities in sorted(grouped.items())}


def _validate_contract() -> None:
    expected_workflows = {
        "Cosmos3DistilledModularPipeline": {
            "text2image",
            "text2video",
            "image2video",
            "video2video",
        },
        "Cosmos3OmniModularPipeline": {
            "image2video",
            "video2video",
            "text2video_with_sound",
            "image2video_with_sound",
            "video2video_with_sound",
            "action_policy",
            "action_forward_dynamics",
            "action_inverse_dynamics",
        },
    }
    actual_workflows = {
        pipeline_class: {
            workflow_id
            for route_pipeline, workflow_id in COSMOS3_REMAINING_ROUTE_RESEARCH
            if route_pipeline == pipeline_class
        }
        for pipeline_class in expected_workflows
    }
    if actual_workflows != expected_workflows or len(COSMOS3_REMAINING_ROUTE_RESEARCH) != 12:
        raise ValueError("Cosmos 3 remaining-route research no longer covers the exact twelve-route backlog.")

    for (pipeline_class, workflow_id), contract in COSMOS3_REMAINING_ROUTE_RESEARCH.items():
        actions = contract["actionSequence"]
        if (
            contract["pipelineClass"] != pipeline_class
            or contract["workflowId"] != workflow_id
            or not actions
            or len(actions) != len(contract["actionRoles"])
            or len(actions) != len(contract["upstreamBlockSequence"])
            or len(contract["stateEdges"]) != len(actions) - 1
            or contract["claim"] != "structural_research_only"
            or contract["sealedDefaults"].get("quantization", object()) is not None
            or contract["sealedDefaults"].get("trustRemoteCode") is not False
        ):
            raise ValueError(f"Invalid Cosmos 3 structural route research for {pipeline_class}/{workflow_id}.")
        for index, edge in enumerate(contract["stateEdges"]):
            if edge != (
                contract["actionRoles"][index],
                "state_out",
                contract["actionRoles"][index + 1],
                "state_in",
            ):
                raise ValueError(f"Invalid Cosmos 3 PipelineState edge for {pipeline_class}/{workflow_id}.")


_validate_contract()
