"""Reviewed package-owned Modular Diffusers workflow execution adapters.

These contracts describe workflows whose faithful MoDiff graph runs official
top-level ``ModularPipelineBlocks`` and transfers a sealed ``PipelineState``
between them.  This module is data-only: discovery may import it without
loading Diffusers, Transformers, Torch, or model weights.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS: dict[tuple[str, str], dict[str, Any]] = {
    ("AnimaModularPipeline", "text2image"): {
        "schemaVersion": 1,
        "adapterId": "official_top_level_blocks",
        "requiredInputs": ["prompt"],
        "actionSequence": [
            "workflow_text_encoder",
            "workflow_image_denoise",
            "workflow_image_decoder",
        ],
        "stateEdges": [
            {
                "producerAction": "workflow_text_encoder",
                "producerOutput": "state_out",
                "consumerAction": "workflow_image_denoise",
                "consumerInput": "state_in",
            },
            {
                "producerAction": "workflow_image_denoise",
                "producerOutput": "state_out",
                "consumerAction": "workflow_image_decoder",
                "consumerInput": "state_in",
            },
        ],
        "upstreamBlockSequence": ["text_encoder", "denoise", "decode"],
    },
    ("AnimaModularPipeline", "img2img"): {
        "schemaVersion": 1,
        "adapterId": "official_top_level_blocks",
        "requiredInputs": ["image", "prompt"],
        "actionSequence": [
            "workflow_text_encoder",
            "workflow_image_encoder",
            "workflow_image_denoise",
            "workflow_image_decoder",
        ],
        "stateEdges": [
            {
                "producerAction": "workflow_text_encoder",
                "producerOutput": "state_out",
                "consumerAction": "workflow_image_encoder",
                "consumerInput": "state_in",
            },
            {
                "producerAction": "workflow_image_encoder",
                "producerOutput": "state_out",
                "consumerAction": "workflow_image_denoise",
                "consumerInput": "state_in",
            },
            {
                "producerAction": "workflow_image_denoise",
                "producerOutput": "state_out",
                "consumerAction": "workflow_image_decoder",
                "consumerInput": "state_in",
            },
        ],
        "upstreamBlockSequence": ["text_encoder", "vae_encoder", "denoise", "decode"],
    },
    ("MiniMaxMusic3ModularPipeline", "default"): {
        "schemaVersion": 1,
        "adapterId": "official_top_level_blocks",
        "requiredInputs": ["lyrics", "prompt"],
        "actionSequence": [
            "semantic_generator",
            "workflow_denoise",
            "workflow_audio_decoder",
        ],
        "stateEdges": [
            {
                "producerAction": "semantic_generator",
                "producerOutput": "state_out",
                "consumerAction": "workflow_denoise",
                "consumerInput": "state_in",
            },
            {
                "producerAction": "workflow_denoise",
                "producerOutput": "state_out",
                "consumerAction": "workflow_audio_decoder",
                "consumerInput": "state_in",
            },
        ],
        "upstreamBlockSequence": ["semantic_generator", "denoise", "decode"],
    },
}


def _sequential_adapter(required_inputs: list[str], actions: list[str], blocks: list[str]) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "adapterId": "official_top_level_blocks",
        "requiredInputs": required_inputs,
        "actionSequence": actions,
        "stateEdges": [
            {
                "producerAction": producer,
                "producerOutput": "state_out",
                "consumerAction": consumer,
                "consumerInput": "state_in",
            }
            for producer, consumer in zip(actions, actions[1:])
        ],
        "upstreamBlockSequence": blocks,
    }


for _helios_pipeline in (
    "HeliosModularPipeline",
    "HeliosPyramidModularPipeline",
    "HeliosPyramidDistilledModularPipeline",
):
    PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS.update(
        {
            (_helios_pipeline, "text2video"): _sequential_adapter(
                ["prompt"],
                ["workflow_text_encoder", "workflow_video_denoise", "workflow_video_decoder"],
                ["text_encoder", "denoise", "decode"],
            ),
            (_helios_pipeline, "image2video"): _sequential_adapter(
                ["image", "prompt"],
                [
                    "workflow_text_encoder",
                    "workflow_video_image_encoder",
                    "workflow_video_denoise",
                    "workflow_video_decoder",
                ],
                ["text_encoder", "vae_encoder", "denoise", "decode"],
            ),
            (_helios_pipeline, "video2video"): _sequential_adapter(
                ["prompt", "video"],
                [
                    "workflow_text_encoder",
                    "workflow_video_encoder",
                    "workflow_video_denoise",
                    "workflow_video_decoder",
                ],
                ["text_encoder", "vae_encoder", "denoise", "decode"],
            ),
        }
    )


for _wan_animate_pipeline in (
    "WanAnimate2ModularPipeline",
    "WanAnimate2DistilledModularPipeline",
):
    PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS[(_wan_animate_pipeline, "default")] = _sequential_adapter(
        ["driving_video", "image", "prompt"],
        [
            "workflow_wan_animate_text_encoder",
            "workflow_wan_animate_image_encoder",
            "workflow_wan_animate_video_encoder",
            "workflow_wan_animate_vae_encoder",
            "workflow_wan_animate_denoise",
            "workflow_wan_animate_decoder",
        ],
        ["text_encoder", "image_encoder", "video_encoder", "vae_encoder", "denoise", "decode"],
    )


PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS.update(
    {
        ("HunyuanVideo15ModularPipeline", "text2video"): _sequential_adapter(
            ["prompt"],
            [
                "workflow_hunyuan_video15_text_encoder",
                "workflow_hunyuan_video15_denoise",
                "workflow_hunyuan_video15_decoder",
            ],
            ["text_encoder", "denoise", "decode"],
        ),
        ("HunyuanVideo15ModularPipeline", "image2video"): _sequential_adapter(
            ["image", "prompt"],
            [
                "workflow_hunyuan_video15_text_encoder",
                "workflow_hunyuan_video15_vae_encoder",
                "workflow_hunyuan_video15_image_encoder",
                "workflow_hunyuan_video15_denoise",
                "workflow_hunyuan_video15_decoder",
            ],
            ["text_encoder", "vae_encoder", "image_encoder", "denoise", "decode"],
        ),
        ("StableDiffusion3ModularPipeline", "text2image"): _sequential_adapter(
            ["prompt"],
            [
                "workflow_stable_diffusion3_text_encoder",
                "workflow_stable_diffusion3_denoise",
                "workflow_stable_diffusion3_decoder",
            ],
            ["text_encoder", "denoise", "decode"],
        ),
        ("StableDiffusion3ModularPipeline", "image2image"): _sequential_adapter(
            ["image", "prompt"],
            [
                "workflow_stable_diffusion3_text_encoder",
                "workflow_stable_diffusion3_vae_encoder",
                "workflow_stable_diffusion3_denoise",
                "workflow_stable_diffusion3_decoder",
            ],
            ["text_encoder", "vae_encoder", "denoise", "decode"],
        ),
        ("Krea2ModularPipeline", "text2image"): _sequential_adapter(
            ["prompt"],
            [
                "workflow_krea2_text_encoder",
                "workflow_krea2_denoise",
                "workflow_krea2_decoder",
            ],
            ["text_encoder", "denoise", "decode"],
        ),
        ("Krea2TurboModularPipeline", "text2image"): _sequential_adapter(
            ["prompt"],
            [
                "workflow_krea2_turbo_text_encoder",
                "workflow_krea2_turbo_denoise",
                "workflow_krea2_decoder",
            ],
            ["text_encoder", "denoise", "decode"],
        ),
        ("Ideogram4ModularPipeline", "text2image"): _sequential_adapter(
            ["prompt"],
            [
                "workflow_ideogram4_prompt_upsample",
                "workflow_ideogram4_text_encoder",
                "workflow_ideogram4_denoise",
                "workflow_ideogram4_decoder",
            ],
            ["prompt_upsample", "text_encoder", "denoise", "decode"],
        ),
    }
)


PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS.update(
    {
        ("LTX25ModularPipeline", "text2video"): _sequential_adapter(
            ["prompt"],
            [
                "workflow_ltx25_text_encoder",
                "workflow_ltx25_duration",
                "workflow_ltx25_denoise",
                "workflow_ltx25_decoder",
            ],
            ["text_encoder", "duration", "denoise", "decode"],
        ),
        ("LTX25ModularPipeline", "image2video"): _sequential_adapter(
            ["image", "prompt"],
            [
                "workflow_ltx25_text_encoder",
                "workflow_ltx25_duration",
                "workflow_ltx25_vae_encoder",
                "workflow_ltx25_denoise",
                "workflow_ltx25_decoder",
            ],
            ["text_encoder", "duration", "vae_encoder", "denoise", "decode"],
        ),
        ("LTX25ModularPipeline", "condition"): _sequential_adapter(
            ["conditions", "prompt"],
            [
                "workflow_ltx25_text_encoder",
                "workflow_ltx25_duration",
                "workflow_ltx25_condition_encoder",
                "workflow_ltx25_denoise",
                "workflow_ltx25_decoder",
            ],
            ["text_encoder", "duration", "condition_encoder", "denoise", "decode"],
        ),
        ("LTX25ModularPipeline", "in_context"): _sequential_adapter(
            ["num_frames", "prompt", "reference_conditions"],
            [
                "workflow_ltx25_text_encoder",
                "workflow_ltx25_condition_encoder",
                "workflow_ltx25_reference_encoder",
                "workflow_ltx25_denoise",
                "workflow_ltx25_decoder",
            ],
            ["text_encoder", "condition_encoder", "reference_encoder", "denoise", "decode"],
        ),
    }
)


for _cosmos3_distilled_workflow, _cosmos3_distilled_required_inputs in (
    ("text2image", ["prompt"]),
    ("text2video", ["prompt"]),
    ("image2video", ["image", "prompt"]),
    ("video2video", ["prompt", "video"]),
):
    _cosmos3_distilled_actions = ["workflow_cosmos3_distilled_text_encoder"]
    _cosmos3_distilled_blocks = ["text_encoder"]
    if _cosmos3_distilled_workflow in {"image2video", "video2video"}:
        _cosmos3_distilled_actions.append("workflow_cosmos3_distilled_vae_encoder")
        _cosmos3_distilled_blocks.append("vae_encoder")
    _cosmos3_distilled_actions.extend(["workflow_cosmos3_distilled_denoise", "workflow_cosmos3_distilled_decoder"])
    _cosmos3_distilled_blocks.extend(["denoise", "decode"])
    PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS[("Cosmos3DistilledModularPipeline", _cosmos3_distilled_workflow)] = (
        _sequential_adapter(
            _cosmos3_distilled_required_inputs,
            _cosmos3_distilled_actions,
            _cosmos3_distilled_blocks,
        )
    )


for _cosmos3_omni_workflow, _cosmos3_omni_required_inputs in (
    ("text2image", ["num_inference_steps", "prompt"]),
    ("text2video", ["num_inference_steps", "prompt"]),
    ("image2video", ["image", "num_inference_steps", "prompt"]),
    ("video2video", ["num_inference_steps", "prompt", "video"]),
    ("text2video_with_sound", ["num_inference_steps", "prompt"]),
    ("image2video_with_sound", ["image", "num_inference_steps", "prompt"]),
    ("video2video_with_sound", ["num_inference_steps", "prompt", "video"]),
    ("action_policy", ["action", "num_inference_steps", "prompt"]),
    ("action_forward_dynamics", ["action", "num_inference_steps", "prompt"]),
    ("action_inverse_dynamics", ["action", "num_inference_steps", "prompt"]),
):
    _cosmos3_omni_actions = ["workflow_cosmos3_omni_text_encoder"]
    _cosmos3_omni_blocks = ["text_encoder"]
    if _cosmos3_omni_workflow in {
        "image2video",
        "video2video",
        "image2video_with_sound",
        "video2video_with_sound",
        "action_policy",
        "action_forward_dynamics",
        "action_inverse_dynamics",
    }:
        _cosmos3_omni_actions.append("workflow_cosmos3_omni_vae_encoder")
        _cosmos3_omni_blocks.append("vae_encoder")
    _cosmos3_omni_actions.extend(
        [
            "workflow_cosmos3_omni_denoise",
            "workflow_cosmos3_omni_decoder",
            "workflow_cosmos3_omni_after_decode",
        ]
    )
    _cosmos3_omni_blocks.extend(["denoise", "decode", "after_decode"])
    PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS[("Cosmos3OmniModularPipeline", _cosmos3_omni_workflow)] = _sequential_adapter(
        _cosmos3_omni_required_inputs,
        _cosmos3_omni_actions,
        _cosmos3_omni_blocks,
    )


PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS.update(
    {
        ("MiniMaxH3ModularPipeline", "t2va"): _sequential_adapter(
            ["num_inference_steps", "prompt"],
            [
                "workflow_minimax_h3_text_encoder",
                "workflow_minimax_h3_denoise",
                "workflow_minimax_h3_decoder",
            ],
            ["text_encoder", "denoise", "decode"],
        ),
        ("MiniMaxH3ModularPipeline", "fl2va"): _sequential_adapter(
            ["num_inference_steps", "prompt"],
            [
                "workflow_minimax_h3_before_encode",
                "workflow_minimax_h3_text_encoder",
                "workflow_minimax_h3_vae_encoder",
                "workflow_minimax_h3_denoise",
                "workflow_minimax_h3_decoder",
            ],
            ["before_encode", "text_encoder", "vae_encoder", "denoise", "decode"],
        ),
        ("MiniMaxH3ModularPipeline", "ref2va"): _sequential_adapter(
            ["num_frames", "num_inference_steps", "prompt", "references"],
            [
                "workflow_minimax_h3_before_encode",
                "workflow_minimax_h3_text_encoder",
                "workflow_minimax_h3_vae_encoder",
                "workflow_minimax_h3_denoise",
                "workflow_minimax_h3_decoder",
            ],
            ["before_encode", "text_encoder", "vae_encoder", "denoise", "decode"],
        ),
    }
)


def _validate_contracts() -> None:
    for (pipeline_class, workflow_id), contract in PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS.items():
        actions = contract.get("actionSequence")
        state_edges = contract.get("stateEdges")
        if (
            not pipeline_class.endswith("ModularPipeline")
            or not workflow_id
            or contract.get("schemaVersion") != 1
            or not isinstance(contract.get("adapterId"), str)
            or not isinstance(actions, list)
            or len(actions) < 2
            or len(actions) != len(set(actions))
            or not isinstance(state_edges, list)
            or len(state_edges) != len(actions) - 1
            or not isinstance(contract.get("upstreamBlockSequence"), list)
            or len(contract["upstreamBlockSequence"]) != len(actions)
        ):
            raise ValueError("Invalid reviewed whole-workflow graph adapter contract.")
        action_set = set(actions)
        for edge in state_edges:
            if (
                not isinstance(edge, Mapping)
                or set(edge) != {"producerAction", "producerOutput", "consumerAction", "consumerInput"}
                or edge["producerAction"] not in action_set
                or edge["consumerAction"] not in action_set
                or not all(isinstance(value, str) and value for value in edge.values())
            ):
                raise ValueError("Invalid reviewed whole-workflow state edge.")


_validate_contracts()


def reviewed_whole_workflow_graph_adapter(
    pipeline_class: str,
    workflow_id: str,
) -> dict[str, Any] | None:
    contract = PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS.get((pipeline_class, workflow_id))
    return deepcopy(contract) if contract is not None else None


def reviewed_whole_workflow_model_types() -> frozenset[str]:
    return frozenset(pipeline_class for pipeline_class, _workflow_id in PINNED_WHOLE_WORKFLOW_GRAPH_ADAPTERS)
