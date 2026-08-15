"""Resolve Comfy research candidates against current MoDiff task boundaries.

The source research ledger deliberately does not infer runtime support.  This
second, still non-executable layer answers a narrower question for every
``new_contract_not_authored`` record: does MoDiff already have a same-family
task candidate, only a same-task boundary with another model, or no matching
task/output boundary at all?
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from modiff.comfy_template_research import PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION
from modiff.comfy_research_contracts import validate_comfy_research_contract_ledger
from modiff.template_authoring_specs import load_template_authoring_spec_ledger


COMFY_CONTRACT_RESOLUTION_SCHEMA_VERSION = 1
COMFY_CONTRACT_RESOLUTION_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "research" / "comfy-contract-resolution.v1.json"
)

_COMFY_CONTRACT_PATH = "data/research/comfy-research-contracts.v1.json"
_COMFY_CATALOG_PATH = "data/research/comfy-workflow-catalog.v1.json"
_WORKFLOW_MANIFEST_PATH = "data/workflow-library-manifest.json"
_UPSTREAM_COVERAGE_PATH = "data/upstream-coverage.v1.json"
_AUTHORING_SPEC_PATH = "data/template-authoring-specs.v1.json"
_LABEL_TO_FAMILIES = {
    "ace-step": {"ACE Audio"},
    "chroma": {"Chroma"},
    "ernie-image": {"ERNIE Image"},
    "flux": {"FLUX Image"},
    "flux.2 dev": {"FLUX Image"},
    "flux.2 klein": {"FLUX Image"},
    "kandinsky": {"Kandinsky"},
    "lightricks": {"LTX Video"},
    "ltx-0.9.5": {"LTX Video"},
    "ltx-2": {"LTX Video"},
    "ltx-2.3": {"LTX Video"},
    "ltx-2.5": {"LTX Video"},
    "omnigen": {"OmniGen"},
    "qwen image edit 2511": {"Qwen Image"},
    "qwen image 2512": {"Qwen Image"},
    "qwen-image": {"Qwen Image"},
    "qwen-image-edit": {"Qwen Image"},
    "qwen-image-layered": {"Qwen Image"},
    "sd1.5": {"Stable Diffusion 1.x"},
    "sdxl": {"Stable Diffusion XL"},
    "stable audio": {"Stable Audio"},
    "wan": {"Wan Video"},
    "wan2.1": {"Wan Video"},
    "wan2.1 infinitetalk": {"Wan Video"},
    "wan2.2": {"Wan Video"},
    "z-image": {"Z-Image"},
}
_MEDIA_KIND_ALIASES = {"text": "json", "three_d": "video"}
_PINNED_SOURCE_REVIEW_REVISION = "d9e66019b85da231b7c936ad9cb7ff08cec16557"
_PINNED_SOURCE_REVIEWS = {
    "comfy-research:template:audio_stable_audio_3_medium": {
        "assetPath": "templates/audio_stable_audio_3_medium.json",
        "assetSha256": "04e86880c18c2959a24c7fe6ad5d6d65dbf83bab68d823340f4976fd791f3b64",
        "gitBlobOid": "c9cc1983e174c38b53c625c616853dd9a1dc3a12",
        "artifactDependencies": [
            {"repository": "Comfy-Org/stable-audio-3", "artifact": "stable_audio_3_medium.safetensors"},
            {"repository": "Comfy-Org/stable-audio-3", "artifact": "t5gemma_b_b_ul2.safetensors"},
            {"repository": "Comfy-Org/Qwen3.5", "artifact": "qwen3.5_2b_bf16.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "StableAudioPipeline:text_to_audio",
            "currentRepository": "stabilityai/stable-audio-open-1.0",
            "currentRevision": "f21265c1e2710b3bd2386596943f0007f55f802e",
            "reasonCode": "stable_audio_3_is_not_stable_audio_open_1",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:audio_stable_audio_3_medium_base": {
        "assetPath": "templates/audio_stable_audio_3_medium_base.json",
        "assetSha256": "7bfc1d24c61424f137d40c394d4fe5f4225228c46d508352a2fc06d650a33378",
        "gitBlobOid": "245b29a3d7d804bcec79eb3d3bbd81ffda2fb30d",
        "artifactDependencies": [
            {"repository": "Comfy-Org/stable-audio-3", "artifact": "stable_audio_3_medium_base.safetensors"},
            {"repository": "Comfy-Org/stable-audio-3", "artifact": "t5gemma_b_b_ul2.safetensors"},
            {"repository": "Comfy-Org/Qwen3.5", "artifact": "qwen3.5_2b_bf16.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "StableAudioPipeline:text_to_audio",
            "currentRepository": "stabilityai/stable-audio-open-1.0",
            "currentRevision": "f21265c1e2710b3bd2386596943f0007f55f802e",
            "reasonCode": "stable_audio_3_is_not_stable_audio_open_1",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_chroma_text_to_image": {
        "assetPath": "templates/image_chroma_text_to_image.json",
        "assetSha256": "7fd2bbee72b1d32aa437433ed95d69d8676b41177d3db6880f8ece6f5421ccbc",
        "gitBlobOid": "1b9525f95e3b80c3e6b07835bd869854aba1d182",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Chroma1-HD_repackaged", "artifact": "Chroma1-HD-fp8mixed.safetensors"},
            {"repository": "Comfy-Org/Lumina_Image_2.0_Repackaged", "artifact": "ae.safetensors"},
            {"repository": "comfyanonymous/flux_text_encoders", "artifact": "t5xxl_fp8_e4m3fn_scaled.safetensors"},
        ],
        "comparison": {
            "state": "same_upstream_family_different_default_partition",
            "currentWorkflowId": "ChromaPipeline:text_to_image",
            "currentRepository": "lodestones/Chroma1-HD",
            "currentRevision": "0e0c60ece1e82b17cb7f77342d765ba5024c40c0",
            "reasonCode": "comfy_default_fp8_repack_is_not_the_admitted_diffusers_partition",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_exact_checkpoint_not_proven",
    },
    "comfy-research:template:image_chroma1_radiance_text_to_image": {
        "assetPath": "templates/image_chroma1_radiance_text_to_image.json",
        "assetSha256": "8141af643b4dcaf8edd21faa92be84cd472fa689d0f228b842339baf2ee5e880",
        "gitBlobOid": "23773eaea7e90edc29e0680d4335039eb3ba72fc",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Chroma1-Radiance_Repackaged",
                "artifact": "chroma-radiance-x0.safetensors",
            },
            {
                "repository": "comfyanonymous/flux_text_encoders",
                "artifact": "t5xxl_fp8_e4m3fn_scaled.safetensors",
            },
        ],
        "sourceRuntimeComponents": [{"component": "vae", "implementation": "pixel_space"}],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "ChromaPipeline:text_to_image",
            "currentRepository": "lodestones/Chroma1-HD",
            "currentRevision": "0e0c60ece1e82b17cb7f77342d765ba5024c40c0",
            "reasonCode": "chroma1_radiance_is_not_chroma1_hd",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_flux2_klein_image_edit_9b_base": {
        "assetPath": "templates/image_flux2_klein_image_edit_9b_base.json",
        "assetSha256": "b5b6e389b6892cf739dba45a76fdadb48611c38505c1c3829bae75d783df229b",
        "gitBlobOid": "82401997a8d24d174fe4a65d8392eba9f87d8024",
        "reviewedTaskMode": "edit_image",
        "artifactDependencies": [
            {
                "repository": "black-forest-labs/FLUX.2-klein-base-9b-fp8",
                "artifact": "flux-2-klein-base-9b-fp8.safetensors",
            },
            {
                "repository": "Comfy-Org/flux2-klein-9B",
                "artifact": "qwen_3_8b_fp8mixed.safetensors",
            },
            {
                "repository": "black-forest-labs/FLUX.2-small-decoder",
                "artifact": "full_encoder_small_decoder.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "Flux2KleinPipeline:edit_image",
            "currentRepository": "black-forest-labs/FLUX.2-klein-4B",
            "currentRevision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            "reasonCode": "flux2_klein_base_9b_is_not_flux2_klein_4b",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_flux2_klein_image_edit_9b_distilled": {
        "assetPath": "templates/image_flux2_klein_image_edit_9b_distilled.json",
        "assetSha256": "4e4285c1a5cba3df9cf022073425ec9585c418a8cb0b7570f936110105763599",
        "gitBlobOid": "5d74c3e059b73215ed2a2b2fb54a691ab286645b",
        "reviewedTaskMode": "edit_image",
        "artifactDependencies": [
            {
                "repository": "black-forest-labs/FLUX.2-klein-9b-fp8",
                "artifact": "flux-2-klein-9b-fp8.safetensors",
            },
            {
                "repository": "Comfy-Org/flux2-klein-9B",
                "artifact": "qwen_3_8b_fp8mixed.safetensors",
            },
            {
                "repository": "black-forest-labs/FLUX.2-small-decoder",
                "artifact": "full_encoder_small_decoder.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "Flux2KleinPipeline:edit_image",
            "currentRepository": "black-forest-labs/FLUX.2-klein-4B",
            "currentRevision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            "reasonCode": "flux2_klein_distilled_9b_is_not_flux2_klein_4b",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image-qwen_image_edit_2511_lora_inflation": {
        "assetPath": "templates/image-qwen_image_edit_2511_lora_inflation.json",
        "assetSha256": "08173d2a252b5491c370fb609fa2cccfcf46f837829535b0f7a121ce00ba2e75",
        "gitBlobOid": "44d4a14160d119df71716dbbefc2a16512a36835",
        "reviewedWorkflowId": "QwenImageEditPlusModularPipeline:edit_image",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_vae.safetensors"},
            {
                "repository": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
                "artifact": "qwen_image_edit_2511_bf16.safetensors",
            },
            {
                "repository": "Comfy-Org/HunyuanVideo_1.5_repackaged",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {
                "repository": "systms/SYSTMS-INFL8-LoRA-Qwen-Image-Edit-2511",
                "artifact": "SYSTMS_INFL8_LoRA_Qwen_Image_Edit_2511.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_requires_auxiliary_admission",
            "currentWorkflowId": "QwenImageEditPlusModularPipeline:edit_image",
            "currentRepository": "Qwen/Qwen-Image-Edit-2511",
            "currentRevision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
            "currentAuxiliaryRepositories": ["lightx2v/Qwen-Image-Edit-2511-Lightning"],
            "reasonCode": "qwen_edit_2511_infl8_lora_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_auxiliary_artifact_not_admitted_exact_checkpoint_not_proven",
    },
    "comfy-research:template:image_qwen_image": {
        "assetPath": "templates/image_qwen_image.json",
        "assetSha256": "0f1dfed53e4d37d47a3c5fe93516c342b3b5b5961b8fbc169c2fffd53ae59349",
        "gitBlobOid": "2a8e9aee5c43a30e95274b2a59dbbc10a218a083",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_fp8_e4m3fn.safetensors"},
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors"},
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_vae.safetensors"},
            {"repository": "lightx2v/Qwen-Image-Lightning", "artifact": "Qwen-Image-Lightning-8steps-V1.0.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "QwenImageModularPipeline:text_to_image",
            "currentRepository": "Qwen/Qwen-Image-2512",
            "currentRevision": "25468b98e3276ca6700de15c6628e51b7de54a26",
            "reasonCode": "qwen_image_original_plus_lightning_is_not_qwen_image_2512",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_qwen_image_edit_2509": {
        "assetPath": "templates/image_qwen_image_edit_2509.json",
        "assetSha256": "088309f628217cc176236b549585dab7f96c282c0a926b3900052de65752f787",
        "gitBlobOid": "522c66b253bc74333b8791e02296407a510c2295",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_vae.safetensors"},
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
                "artifact": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
            },
            {
                "repository": "lightx2v/Qwen-Image-Lightning",
                "artifact": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_different_partition_and_auxiliary",
            "currentWorkflowId": "QwenImageEditModularPipeline:edit_image",
            "currentRepository": "Qwen/Qwen-Image-Edit",
            "currentRevision": "ac7f9318f633fc4b5778c59367c8128225f1e3de",
            "reasonCode": "qwen_edit_2509_fp8_repack_and_lightning_not_exactly_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_partition_and_auxiliary_exact_checkpoint_not_proven",
    },
    "comfy-research:template:image_qwen_image_layered_control": {
        "assetPath": "templates/image_qwen_image_layered_control.json",
        "assetSha256": "87f5727f23e67c766364544ee31900736e12c2ab7d9fbca47c498c46526ecc89",
        "gitBlobOid": "351ec57da7d9a025e5d4104b8f5e786a37fc1c88",
        "artifactDependencies": [
            {
                "repository": "DiffSynth-Studio/Qwen-Image-Layered-Control",
                "artifact": "qwen_image_layered_control_bf16.safetensors",
            },
            {
                "repository": "Comfy-Org/HunyuanVideo_1.5_repackaged",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Qwen-Image-Layered_ComfyUI",
                "artifact": "qwen_image_layered_vae.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "QwenImageLayeredModularPipeline:layer_decomposition",
            "currentRepository": "Qwen/Qwen-Image-Layered",
            "currentRevision": "8f0ca708dfff6ba1dd5f2d85d78f8c108a040bcf",
            "reasonCode": "qwen_layered_control_is_not_qwen_image_layered",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_qwen_image_union_control_lora": {
        "assetPath": "templates/image_qwen_image_union_control_lora.json",
        "assetSha256": "bedd54cf982ec7ab3da0322ee95cb55444d6d936ad6b862096fa7385aa26d80f",
        "gitBlobOid": "483ab15cb0d913dfa5c8633bea0a57693f1b2192",
        "reviewedTaskMode": "control_image",
        "reviewedWorkflowId": "QwenImageModularPipeline:control_image",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_vae.safetensors"},
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_image_fp8_e4m3fn.safetensors",
            },
            {
                "repository": "Comfy-Org/Qwen-Image-DiffSynth-ControlNets",
                "artifact": "qwen_image_union_diffsynth_lora.safetensors",
            },
            {
                "repository": "lightx2v/Qwen-Image-Lightning",
                "artifact": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "QwenImageModularPipeline:control_image",
            "currentRepository": "Qwen/Qwen-Image-2512",
            "currentRevision": "25468b98e3276ca6700de15c6628e51b7de54a26",
            "currentAuxiliaryRepositories": ["InstantX/Qwen-Image-ControlNet-Union"],
            "reasonCode": "original_qwen_image_diffsynth_control_lora_is_not_qwen_2512_instantx_union",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:template_qwen_image_edit_2511_systms_action": {
        "assetPath": "templates/template_qwen_image_edit_2511_systms_action.json",
        "assetSha256": "1d1398ae40491957a30a1d117d0262aa703c4299c6116264967ae346263088ae",
        "gitBlobOid": "5150867d119e05bb2de5bf2695f6f0627507b702",
        "reviewedWorkflowId": "QwenImageEditPlusModularPipeline:edit_image",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_vae.safetensors"},
            {
                "repository": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
                "artifact": "qwen_image_edit_2511_bf16.safetensors",
            },
            {
                "repository": "Comfy-Org/HunyuanVideo_1.5_repackaged",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {
                "repository": "lightx2v/Qwen-Image-Edit-2511-Lightning",
                "artifact": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            },
            {
                "repository": "systms/SYSTMS-ACTION-LoRA-Qwen-Image-Edit-2511",
                "artifact": "QWEN_EDIT_ACTION_V1.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_requires_auxiliary_admission",
            "currentWorkflowId": "QwenImageEditPlusModularPipeline:edit_image",
            "currentRepository": "Qwen/Qwen-Image-Edit-2511",
            "currentRevision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
            "currentAuxiliaryRepositories": ["lightx2v/Qwen-Image-Edit-2511-Lightning"],
            "reasonCode": "qwen_edit_2511_action_lora_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_auxiliary_artifact_not_admitted_exact_checkpoint_not_proven",
    },
    "comfy-research:template:image_z_image": {
        "assetPath": "templates/image_z_image.json",
        "assetSha256": "f7b3ceb43a3688e17cc5bce67d1beafe833a674b4e4a257c626a6243b3ebedc2",
        "gitBlobOid": "97cfc42585f59bbe43139e2fea4c5a6530240592",
        "artifactDependencies": [
            {"repository": "Comfy-Org/z_image", "artifact": "z_image_bf16.safetensors"},
            {"repository": "Comfy-Org/z_image_turbo", "artifact": "qwen_3_4b.safetensors"},
            {"repository": "Comfy-Org/z_image_turbo", "artifact": "ae.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "ZImageModularPipeline:text_to_image",
            "currentRepository": "Tongyi-MAI/Z-Image-Turbo",
            "currentRevision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
            "reasonCode": "z_image_base_is_not_z_image_turbo",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_z_image_int8": {
        "assetPath": "templates/image_z_image_int8.json",
        "assetSha256": "7e80aff2a7aecc61602bcf1f8a83afa30a5c714559547295f9917444d3ea222c",
        "gitBlobOid": "2dd3f57d9d01e83b10caa16cddba37d356d50e23",
        "artifactDependencies": [
            {"repository": "Comfy-Org/z_image", "artifact": "z_image_int8_convrot.safetensors"},
            {"repository": "Comfy-Org/z_image_turbo", "artifact": "qwen_3_4b.safetensors"},
            {"repository": "Comfy-Org/z_image_turbo", "artifact": "ae.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "ZImageModularPipeline:text_to_image",
            "currentRepository": "Tongyi-MAI/Z-Image-Turbo",
            "currentRevision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
            "reasonCode": "z_image_base_int8_is_not_z_image_turbo",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
}
_BOUNDARY = {
    "researchOnly": True,
    "importsComfyGraphs": False,
    "executesComfyNodes": False,
    "copiesComfyNodes": False,
    "copiesComfyPrompts": False,
    "opensPinnedComfyGraphsForSourceReview": True,
    "downloadsModelsOrMedia": False,
    "claimsExactCatalogCheckpointCompatibility": False,
    "claimsMoDiffWorkflowSupportFromCatalogMetadata": False,
    "publishesTemplates": False,
    "generatesAssets": False,
    "maximumClaim": "pinned_source_dependency_and_semantic_task_resolution",
}


class ComfyContractResolutionError(RuntimeError):
    """Raised when resolution sources or the checked-in ledger drift."""


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
        raise ComfyContractResolutionError(f"{label} is unavailable or invalid: {relative_path}") from error
    if not isinstance(value, dict):
        raise ComfyContractResolutionError(f"{label} must be a JSON object.")
    return value, "sha256:" + _sha256_bytes(raw)


def _records(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ComfyContractResolutionError(f"{label} must be a list of objects.")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComfyContractResolutionError(f"{label} must be a non-empty string.")
    return value


def _base_workflow_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    workflows = _records(manifest.get("workflows"), label="canonical workflows")
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in workflows:
        model_type = _string(row.get("modelType"), label="canonical model type")
        mode = _string(row.get("mode"), label="canonical mode")
        workflow_id = _string(row.get("id"), label="canonical workflow id")
        for field in ("modelFamily", "mediaKind", "graphHash", "qualificationStatus"):
            _string(row.get(field), label=f"canonical workflow {workflow_id} {field}")
        key = (model_type, mode)
        current = by_pair.get(key)
        if current is None or workflow_id == f"{model_type}:{mode}":
            by_pair[key] = row
    return sorted(by_pair.values(), key=lambda item: item["id"])


def _public_templates_by_workflow(coverage: Mapping[str, Any]) -> dict[str, list[str]]:
    rows = _records(coverage.get("canonicalWorkflows"), label="coverage canonical workflows")
    result = {}
    for row in rows:
        workflow_id = _string(row.get("id"), label="coverage workflow id")
        values = row.get("publicTemplateIds")
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ComfyContractResolutionError(f"Coverage public-template ids are invalid for {workflow_id}.")
        result[workflow_id] = sorted(values)
    return result


def _catalog_families(labels: list[str]) -> set[str]:
    families = set()
    for label in labels:
        families.update(_LABEL_TO_FAMILIES.get(label.casefold(), set()))
    return families


def _representative_options(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        family = row["modelFamily"]
        current = by_family.get(family)
        if current is None or row["id"] < current["id"]:
            by_family[family] = row
    return [
        {
            "canonicalWorkflowId": row["id"],
            "modelType": row["modelType"],
            "modelFamily": row["modelFamily"],
            "qualificationStatus": row["qualificationStatus"],
        }
        for _, row in sorted(by_family.items())
    ]


def build_comfy_contract_resolution_ledger(root: Path) -> dict[str, Any]:
    """Build an exhaustive resolution for all unresolved Comfy proposals."""

    root = root.resolve(strict=True)
    if _PINNED_SOURCE_REVIEW_REVISION != PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION:
        raise ComfyContractResolutionError(
            "Pinned Comfy graph-source reviews must be repeated for the current catalog revision."
        )
    comfy, comfy_sha = _read_json(root, _COMFY_CONTRACT_PATH, label="Comfy research contracts")
    comfy_catalog, comfy_catalog_sha = _read_json(root, _COMFY_CATALOG_PATH, label="Comfy research catalog")
    manifest, manifest_sha = _read_json(root, _WORKFLOW_MANIFEST_PATH, label="workflow manifest")
    coverage, coverage_sha = _read_json(root, _UPSTREAM_COVERAGE_PATH, label="upstream coverage")
    authoring, authoring_sha = _read_json(root, _AUTHORING_SPEC_PATH, label="template authoring specs")
    if authoring != load_template_authoring_spec_ledger(root / _AUTHORING_SPEC_PATH, root=root):
        raise ComfyContractResolutionError("The template authoring specification ledger is invalid.")

    manifest_workflows = _records(manifest.get("workflows"), label="canonical workflows")
    supported_workflow_ids = {_string(row.get("id"), label="canonical workflow id") for row in manifest_workflows}
    validate_comfy_research_contract_ledger(
        comfy,
        catalog_ledger=comfy_catalog,
        catalog_ledger_sha256=comfy_catalog_sha.removeprefix("sha256:"),
        supported_workflow_ids=supported_workflow_ids,
    )
    workflows = _base_workflow_records(manifest)
    public_templates = _public_templates_by_workflow(coverage)
    authoring_by_workflow = {
        row["canonicalWorkflowId"]: row["id"]
        for row in _records(authoring.get("specifications"), label="template authoring specifications")
    }
    resolutions = []
    source_rows = [
        row
        for row in _records(comfy.get("contracts"), label="Comfy research contracts")
        if row.get("contractState") == "new_contract_not_authored"
    ]
    for row in source_rows:
        contract_id = _string(row.get("contractId"), label="Comfy research contract id")
        task_evidence = row.get("taskEvidence")
        media_evidence = row.get("mediaEvidence")
        model_evidence = row.get("modelEvidence")
        source_review = row.get("sourceReview")
        if any(
            not isinstance(value, Mapping) for value in (task_evidence, media_evidence, model_evidence, source_review)
        ):
            raise ComfyContractResolutionError(f"Comfy research evidence is incomplete for {contract_id}.")
        catalog_mode = _string(task_evidence.get("selectedCandidateMode"), label=f"{contract_id} selected mode")
        pinned_source_review = deepcopy(_PINNED_SOURCE_REVIEWS.get(contract_id))
        mode = catalog_mode
        if pinned_source_review is not None and pinned_source_review.get("reviewedTaskMode") is not None:
            mode = _string(
                pinned_source_review.get("reviewedTaskMode"),
                label=f"{contract_id} reviewed task mode",
            )
            pinned_source_review["catalogSelectedCandidateMode"] = catalog_mode
        output_kinds = media_evidence.get("candidateOutputMediaKinds")
        labels = model_evidence.get("catalogModelLabels")
        if not isinstance(output_kinds, list) or any(
            not isinstance(value, str) or not value for value in output_kinds
        ):
            raise ComfyContractResolutionError(f"Comfy output media evidence is invalid for {contract_id}.")
        if not isinstance(labels, list) or any(not isinstance(value, str) or not value for value in labels):
            raise ComfyContractResolutionError(f"Comfy model evidence is invalid for {contract_id}.")
        normalized_output_kinds = {_MEDIA_KIND_ALIASES.get(value, value) for value in output_kinds}
        task_options = [
            workflow
            for workflow in workflows
            if workflow["mode"] == mode and workflow["mediaKind"] in normalized_output_kinds
        ]
        catalog_families = _catalog_families(labels)
        family_options = [workflow for workflow in task_options if workflow["modelFamily"] in catalog_families]
        if family_options:
            resolution_state = "existing_family_workflow_candidate"
            mapping_meaning = "task_and_family_semantic_candidate_only_exact_checkpoint_not_proven"
            options = family_options
        elif task_options:
            resolution_state = "existing_task_boundary_model_admission_required"
            mapping_meaning = "task_boundary_reuse_candidate_only_catalog_model_unmatched"
            options = task_options
        else:
            resolution_state = "new_task_boundary_required"
            mapping_meaning = "no_current_mode_and_output_boundary"
            options = []
        representatives = _representative_options(options)
        if pinned_source_review is not None and pinned_source_review.get("reviewedWorkflowId") is not None:
            reviewed_workflow_id = _string(
                pinned_source_review.get("reviewedWorkflowId"),
                label=f"{contract_id} reviewed workflow id",
            )
            reviewed_workflow = next(
                (workflow for workflow in options if workflow["id"] == reviewed_workflow_id),
                None,
            )
            if reviewed_workflow is None:
                raise ComfyContractResolutionError(
                    f"Pinned Comfy source review workflow is not a current task/family option for {contract_id}."
                )
            catalog_recommended_id = representatives[0]["canonicalWorkflowId"] if representatives else None
            pinned_source_review["catalogRecommendedWorkflowId"] = catalog_recommended_id
            reviewed_representative = {
                "canonicalWorkflowId": reviewed_workflow["id"],
                "modelType": reviewed_workflow["modelType"],
                "modelFamily": reviewed_workflow["modelFamily"],
                "qualificationStatus": reviewed_workflow["qualificationStatus"],
            }
            representatives = [reviewed_representative] + [
                row for row in representatives if row["canonicalWorkflowId"] != reviewed_workflow_id
            ]
        recommended = representatives[0] if representatives else None
        recommended_id = recommended["canonicalWorkflowId"] if recommended else None
        if pinned_source_review is not None:
            comparison = pinned_source_review["comparison"]
            if comparison["currentWorkflowId"] != recommended_id:
                raise ComfyContractResolutionError(
                    f"Pinned Comfy source review recommendation drifted for {contract_id}."
                )
            pinned_source_review["state"] = "complete"
            pinned_source_review["sourceRevision"] = _PINNED_SOURCE_REVIEW_REVISION
            pinned_source_review["importsGraph"] = False
            pinned_source_review["copiesNodesOrPrompts"] = False
            pinned_source_review["executesGraph"] = False
            resolution_state = pinned_source_review.pop("resolutionState")
            mapping_meaning = pinned_source_review.pop("mappingMeaning")
            blockers = [
                "pinned_source_dependencies_reviewed_without_import_or_execution",
                "exact_catalog_checkpoint_and_component_compatibility_not_proven",
                f"source_review_{comparison['reasonCode']}",
                "model_artifact_and_input_output_rights_review_required",
                "execution_and_asset_quality_not_proven",
            ]
        else:
            blockers = [
                "catalog_metadata_is_semantic_evidence_only",
                "exact_catalog_checkpoint_and_component_compatibility_not_proven",
                "model_artifact_and_input_output_rights_review_required",
                "execution_and_asset_quality_not_proven",
            ]
            if source_review.get("state") == "required":
                blockers.insert(0, "catalog_entry_source_review_required")
        if resolution_state == "existing_task_boundary_model_admission_required":
            blockers.append("exact_catalog_model_or_variant_admission_required")
        elif resolution_state == "new_task_boundary_required":
            blockers.extend(
                [
                    "new_bounded_modiff_task_contract_required",
                    "new_original_modiff_workflow_required",
                ]
            )
        else:
            blockers.append("family_match_requires_exact_checkpoint_and_variant_review")
        resolutions.append(
            {
                "contractId": contract_id,
                "sourceKind": row.get("sourceKind"),
                "catalogId": row.get("catalogId"),
                "title": row.get("title"),
                "selectedCandidateMode": mode,
                "candidateOutputMediaKinds": deepcopy(output_kinds),
                "catalogModelLabels": deepcopy(labels),
                "recognizedMoDiffFamilies": sorted(catalog_families),
                "resolutionState": resolution_state,
                "mappingMeaning": mapping_meaning,
                **({"sourceReview": pinned_source_review} if pinned_source_review is not None else {}),
                "currentTaskBoundaryOptionCount": len(task_options),
                "currentFamilyOptionCount": len(family_options),
                "representativeWorkflowOptions": representatives,
                "recommendedWorkflow": (
                    {
                        **deepcopy(recommended),
                        "publicTemplateIds": public_templates.get(recommended_id, []),
                        "authoringSpecId": authoring_by_workflow.get(recommended_id),
                    }
                    if recommended is not None
                    else None
                ),
                "exactCatalogModelReproductionRequiresAdmission": True,
                "blockers": blockers,
                "claims": {
                    "comfyGraphImported": False,
                    "comfyPromptCopied": False,
                    "exactCatalogCheckpointSupported": False,
                    "recommendedWorkflowEquivalent": False,
                    "newMoDiffWorkflowAuthored": False,
                    "publicTemplateCreated": False,
                    "assetGenerated": False,
                    "assetApproved": False,
                },
            }
        )

    resolutions.sort(key=lambda item: item["contractId"])
    state_counts = Counter(item["resolutionState"] for item in resolutions)
    mode_counts = Counter(item["selectedCandidateMode"] for item in resolutions)
    ledger: dict[str, Any] = {
        "schemaVersion": COMFY_CONTRACT_RESOLUTION_SCHEMA_VERSION,
        "contractKind": "non_executable_comfy_contract_resolution_ledger",
        "boundary": deepcopy(_BOUNDARY),
        "sources": {
            "comfyResearchContracts": {
                "path": _COMFY_CONTRACT_PATH,
                "sha256": comfy_sha,
                "contentHash": comfy.get("contentHash"),
            },
            "comfyResearchCatalog": {
                "path": _COMFY_CATALOG_PATH,
                "sha256": comfy_catalog_sha,
                "contentHash": comfy_catalog.get("contentHash"),
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
            "sourceProposalCount": len(source_rows),
            "resolutionCount": len(resolutions),
            "resolutionStateCounts": {key: state_counts[key] for key in sorted(state_counts)},
            "modeCounts": {key: mode_counts[key] for key in sorted(mode_counts)},
            "recordsWithRecommendedWorkflow": sum(item["recommendedWorkflow"] is not None for item in resolutions),
            "recordsWithPublicTemplateOption": sum(
                bool(item["recommendedWorkflow"] and item["recommendedWorkflow"]["publicTemplateIds"])
                for item in resolutions
            ),
            "recordsWithHiddenAuthoringSpecOption": sum(
                bool(item["recommendedWorkflow"] and item["recommendedWorkflow"]["authoringSpecId"])
                for item in resolutions
            ),
            "exactCatalogCheckpointSupportClaims": 0,
            "newWorkflowClaims": 0,
            "assetClaims": 0,
            "pinnedSourceReviewCount": sum(item.get("sourceReview") is not None for item in resolutions),
            "pinnedSourceReviewDecisionCounts": dict(
                sorted(
                    Counter(
                        item["sourceReview"]["comparison"]["state"]
                        for item in resolutions
                        if item.get("sourceReview") is not None
                    ).items()
                )
            ),
        },
        "resolutions": resolutions,
    }
    ledger["contentHash"] = _content_hash(ledger)
    return ledger


def validate_comfy_contract_resolution_ledger(
    ledger: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    if ledger.get("schemaVersion") != COMFY_CONTRACT_RESOLUTION_SCHEMA_VERSION:
        raise ComfyContractResolutionError("The Comfy contract resolution schema is unsupported.")
    if ledger.get("contractKind") != "non_executable_comfy_contract_resolution_ledger":
        raise ComfyContractResolutionError("Comfy contract resolution must remain non-executable.")
    if ledger.get("contentHash") != _content_hash(ledger):
        raise ComfyContractResolutionError("The Comfy contract resolution content hash is invalid.")
    expected = build_comfy_contract_resolution_ledger(root)
    if dict(ledger) != expected:
        raise ComfyContractResolutionError(
            "The Comfy contract resolution ledger is stale, incomplete, or internally inconsistent."
        )
    return expected


def render_comfy_contract_resolution_ledger(ledger: Mapping[str, Any]) -> str:
    return json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_comfy_contract_resolution_ledger(
    path: Path = COMFY_CONTRACT_RESOLUTION_PATH,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComfyContractResolutionError(
            "The checked-in Comfy contract resolution ledger is unavailable or invalid."
        ) from error
    if not isinstance(ledger, dict):
        raise ComfyContractResolutionError("The Comfy contract resolution ledger must be an object.")
    return validate_comfy_contract_resolution_ledger(
        ledger,
        root=root if root is not None else Path(__file__).resolve().parents[1],
    )
