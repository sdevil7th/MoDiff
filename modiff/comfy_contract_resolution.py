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
    "comfy-research:template:audio_ace_step_1_m2m_editing": {
        "assetPath": "templates/audio_ace_step_1_m2m_editing.json",
        "assetSha256": "47044f579bf20ff6623615001482205c96250c169e3aec1eda6431823adb4f3d",
        "gitBlobOid": "1020bf48371ecba2d941ea381a9a0d8d85ce1b50",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/ACE-Step_ComfyUI_repackaged",
                "artifact": "ace_step_v1_3.5b.safetensors",
            }
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": "ACE-Step/acestep-v15-xl-turbo-diffusers",
            "currentRevision": "200ba991ae448051e14b0183157e35c2d27c9fb0",
            "reasonCode": "ace_step_v1_music_to_music_is_not_v15_text_to_audio",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_audio_edit_task_and_model_generation_required",
    },
    "comfy-research:template:audio_ace_step_1_t2a_instrumentals": {
        "assetPath": "templates/audio_ace_step_1_t2a_instrumentals.json",
        "assetSha256": "88acb4f61ca03e04fa9645b84413f91710c0bacd2e174a2faf9072af75187294",
        "gitBlobOid": "42ed75275069735fae4510219f4b9bc06235dbce",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/ACE-Step_ComfyUI_repackaged",
                "artifact": "ace_step_v1_3.5b.safetensors",
            }
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "AceStepAudioPipeline:text_to_audio",
            "currentRepository": "ACE-Step/acestep-v15-xl-turbo-diffusers",
            "currentRevision": "200ba991ae448051e14b0183157e35c2d27c9fb0",
            "reasonCode": "ace_step_v1_3_5b_is_not_acestep_v15_xl_turbo",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:audio_ace_step_1_t2a_song": {
        "assetPath": "templates/audio_ace_step_1_t2a_song.json",
        "assetSha256": "be4d2c2f9d07005af00905ce4ae7f376dbc7e904b4d4276951f0d5b4a389adcf",
        "gitBlobOid": "43626d5e5b17858b023717ae71a65f34f6e77815",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/ACE-Step_ComfyUI_repackaged",
                "artifact": "ace_step_v1_3.5b.safetensors",
            }
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "AceStepAudioPipeline:text_to_audio",
            "currentRepository": "ACE-Step/acestep-v15-xl-turbo-diffusers",
            "currentRevision": "200ba991ae448051e14b0183157e35c2d27c9fb0",
            "reasonCode": "ace_step_v1_3_5b_is_not_acestep_v15_xl_turbo",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
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
    "comfy-research:template:image_flux2_klein_9b_kv_image_edit": {
        "assetPath": "templates/image_flux2_klein_9b_kv_image_edit.json",
        "assetSha256": "8971a174ac2387a4cdda19ee1102ebc560e4cac75783f6162467c38eb07a499f",
        "gitBlobOid": "5a67a1cce4b06a69f97c20eaa56a800f9cf2cd18",
        "artifactDependencies": [
            {
                "repository": "black-forest-labs/FLUX.2-klein-9b-kv-fp8",
                "artifact": "flux-2-klein-9b-kv-fp8.safetensors",
            },
            {
                "repository": "Comfy-Org/flux2-klein-9B",
                "artifact": "qwen_3_8b_fp8mixed.safetensors",
            },
            {"repository": "Comfy-Org/flux2-dev", "artifact": "flux2-vae.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "Flux2KleinPipeline:edit_image",
            "currentRepository": "black-forest-labs/FLUX.2-klein-4B",
            "currentRevision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            "reasonCode": "flux2_klein_9b_kv_is_not_flux2_klein_4b",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_flux2_text_to_image": {
        "assetPath": "templates/image_flux2_text_to_image.json",
        "assetSha256": "440a33ed8079a602c420910afc0cf860e4d3ed4114f5df10399a80b9237b48fa",
        "gitBlobOid": "bb7d4e5be6f379834e7c6ee563dd58687fc78dad",
        "artifactDependencies": [
            {"repository": "Comfy-Org/flux2-dev", "artifact": "flux2_dev_fp8mixed.safetensors"},
            {
                "repository": "Comfy-Org/flux2-dev",
                "artifact": "mistral_3_small_flux2_bf16.safetensors",
            },
            {
                "repository": "black-forest-labs/FLUX.2-small-decoder",
                "artifact": "full_encoder_small_decoder.safetensors",
            },
            {"repository": "ByteZSzn/Flux.2-Turbo-ComfyUI", "artifact": "Flux_2-Turbo-LoRA_comfyui.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "Flux2KleinPipeline:text_to_image",
            "currentRepository": "black-forest-labs/FLUX.2-klein-4B",
            "currentRevision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            "reasonCode": "flux2_dev_turbo_lora_is_not_flux2_klein_4b",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_flux2_text_to_image_9b": {
        "assetPath": "templates/image_flux2_text_to_image_9b.json",
        "assetSha256": "81f534c8761fffb34030d8e09de51b96a3dd7d0ebf0505104444568cd0360c62",
        "gitBlobOid": "6700b1f5c1a1ff8169df6b8b40b6f441dbe43486",
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
            "currentWorkflowId": "Flux2KleinPipeline:text_to_image",
            "currentRepository": "black-forest-labs/FLUX.2-klein-4B",
            "currentRevision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            "reasonCode": "flux2_klein_base_9b_is_not_flux2_klein_4b",
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
    "comfy-research:template:image_qwen_image_2512_with_2steps_lora": {
        "assetPath": "templates/image_qwen_image_2512_with_2steps_lora.json",
        "assetSha256": "42e3926b69286c0ab57a6c9c3065e2b884e8890e8dce9003414de63f1a3dee5e",
        "gitBlobOid": "c4214781ab66852199b88bbfe98e5f564004b4fa",
        "reviewedWorkflowId": "QwenImageModularPipeline:text_to_image",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_image_2512_fp8_e4m3fn.safetensors",
            },
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_vae.safetensors"},
            {
                "repository": "Wuli-art/Qwen-Image-2512-Turbo-LoRA-2-Steps",
                "artifact": "Wuli-Qwen-Image-2512-Turbo-LoRA-2steps-V1.0-bf16.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_different_partition_and_auxiliary",
            "currentWorkflowId": "QwenImageModularPipeline:text_to_image",
            "currentRepository": "Qwen/Qwen-Image-2512",
            "currentRevision": "25468b98e3276ca6700de15c6628e51b7de54a26",
            "reasonCode": "qwen_2512_fp8_repack_and_two_step_lora_not_exactly_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_partition_and_auxiliary_exact_checkpoint_not_proven",
    },
    "comfy-research:template:template_qwen_Image_2512_360_lora": {
        "assetPath": "templates/template_qwen_Image_2512_360_lora.json",
        "assetSha256": "a2c3de32eeacd75b329b26caa86178b60b9285dd6dcf684318ba5ff761f75e6f",
        "gitBlobOid": "80e423a1f68544e703c6211a8b1796268b946fe4",
        "reviewedWorkflowId": "QwenImageModularPipeline:text_to_image",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_image_2512_fp8_e4m3fn.safetensors",
            },
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_vae.safetensors"},
            {
                "repository": "lightx2v/Qwen-Image-2512-Lightning",
                "artifact": "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
            },
            {
                "repository": "ProGamerGov/qwen-360-diffusion",
                "artifact": "qwen-360-diffusion-2512-int8-bf16-v2.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_different_partition_and_auxiliary",
            "currentWorkflowId": "QwenImageModularPipeline:text_to_image",
            "currentRepository": "Qwen/Qwen-Image-2512",
            "currentRevision": "25468b98e3276ca6700de15c6628e51b7de54a26",
            "reasonCode": "qwen_2512_fp8_repack_lightning_and_360_lora_not_exactly_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_partition_and_auxiliary_exact_checkpoint_not_proven",
    },
    "comfy-research:template:image_qwen_image_controlnet_patch": {
        "assetPath": "templates/image_qwen_image_controlnet_patch.json",
        "assetSha256": "99394439a187e9675749df2a975d9069051ca9264e137b28d944acbcd212277e",
        "gitBlobOid": "84f811d25740c0f979f655120b6b277ef7379662",
        "reviewedTaskMode": "control_image",
        "reviewedWorkflowId": "QwenImageModularPipeline:control_image",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_image_fp8_e4m3fn.safetensors",
            },
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_vae.safetensors"},
            {
                "repository": "lightx2v/Qwen-Image-Lightning",
                "artifact": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            },
            {
                "repository": "Comfy-Org/Qwen-Image-DiffSynth-ControlNets",
                "artifact": "qwen_image_canny_diffsynth_controlnet.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "QwenImageModularPipeline:control_image",
            "currentRepository": "Qwen/Qwen-Image-2512",
            "currentRevision": "25468b98e3276ca6700de15c6628e51b7de54a26",
            "currentAuxiliaryRepositories": ["InstantX/Qwen-Image-ControlNet-Union"],
            "reasonCode": "original_qwen_image_diffsynth_patch_is_not_qwen_2512_instantx_union",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_qwen_image_instantx_inpainting_controlnet": {
        "assetPath": "templates/image_qwen_image_instantx_inpainting_controlnet.json",
        "assetSha256": "7fdb06c966d36180feeeca406e618de254808c2948180837f1fe34e2f57abca7",
        "gitBlobOid": "128234800aee77e5659c47c811785b974b146b71",
        "reviewedWorkflowId": "QwenImageModularPipeline:inpaint",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_image_fp8_e4m3fn.safetensors",
            },
            {
                "repository": "Comfy-Org/Qwen-Image_ComfyUI",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {"repository": "Comfy-Org/Qwen-Image_ComfyUI", "artifact": "qwen_image_vae.safetensors"},
            {
                "repository": "Comfy-Org/Qwen-Image-InstantX-ControlNets",
                "artifact": "Qwen-Image-InstantX-ControlNet-Inpainting.safetensors",
            },
            {
                "repository": "lightx2v/Qwen-Image-Lightning",
                "artifact": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "QwenImageModularPipeline:inpaint",
            "currentRepository": "Qwen/Qwen-Image-2512",
            "currentRevision": "25468b98e3276ca6700de15c6628e51b7de54a26",
            "reasonCode": "original_qwen_image_inpainting_controlnet_is_not_qwen_2512_inpaint",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_qwen_image_edit_2509_relight": {
        "assetPath": "templates/image_qwen_image_edit_2509_relight.json",
        "assetSha256": "1aa6672afaf7a9e07abed0288b3053a24cc80ca0868e297f575105214b8167f3",
        "gitBlobOid": "66b9818073500e7cbf670d6e4ebfa12ccf8a2449",
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
            {
                "repository": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
                "artifact": "Qwen-Image-Edit-2509-Relight.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_different_partition_and_auxiliary",
            "currentWorkflowId": "QwenImageEditModularPipeline:edit_image",
            "currentRepository": "Qwen/Qwen-Image-Edit",
            "currentRevision": "ac7f9318f633fc4b5778c59367c8128225f1e3de",
            "reasonCode": "qwen_edit_2509_fp8_lightning_and_relight_not_exactly_admitted",
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
    "comfy-research:template:templates-1_click_multiple_character_angles-v1.0": {
        "assetPath": "templates/templates-1_click_multiple_character_angles-v1.0.json",
        "assetSha256": "d16ea80b6c7b28b66b84d0a3c4ce3e5c52ac61b89aad28a2bec9205b4aa89820",
        "gitBlobOid": "65342f0c439959c17c2a93035769214faa8689f9",
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
                "repository": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
                "artifact": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_requires_auxiliary_admission",
            "currentWorkflowId": "QwenImageEditPlusModularPipeline:edit_image",
            "currentRepository": "Qwen/Qwen-Image-Edit-2511",
            "currentRevision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
            "currentAuxiliaryRepositories": ["lightx2v/Qwen-Image-Edit-2511-Lightning"],
            "reasonCode": "qwen_edit_2511_multiple_angles_lora_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_auxiliary_artifact_not_admitted_exact_checkpoint_not_proven",
    },
    "comfy-research:template:templates-1_click_multiple_scene_angles-v1.0": {
        "assetPath": "templates/templates-1_click_multiple_scene_angles-v1.0.json",
        "assetSha256": "d9bf5d0920bd1780d981160d28de79ae5536e6b090c7f9f3992080ebc768b894",
        "gitBlobOid": "05f187fd77665e7ecd5f5a5b194e36759c90e8b6",
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
                "repository": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
                "artifact": "Qwen-Edit-2509-Multiple-angles.safetensors",
            },
            {
                "repository": "lightx2v/Qwen-Image-Lightning",
                "artifact": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_requires_auxiliary_admission",
            "currentWorkflowId": "QwenImageEditModularPipeline:edit_image",
            "currentRepository": "Qwen/Qwen-Image-Edit",
            "currentRevision": "ac7f9318f633fc4b5778c59367c8128225f1e3de",
            "reasonCode": "qwen_edit_2509_multiple_angles_lora_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_auxiliary_artifact_not_admitted_exact_checkpoint_not_proven",
    },
    "comfy-research:template:ltxv_image_to_video": {
        "assetPath": "templates/ltxv_image_to_video.json",
        "assetSha256": "96bf9ddbbf4591af29df2b707933c2ea25d9c3a406296ade0fbe1ae91d43b0d0",
        "gitBlobOid": "261c4a1720cef505ace7f6f1150ed55c5f04f2d4",
        "artifactDependencies": [
            {"repository": "Lightricks/LTX-Video", "artifact": "ltx-video-2b-v0.9.5.safetensors"},
            {"repository": "comfyanonymous/flux_text_encoders", "artifact": "t5xxl_fp16.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:image_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_video_0_9_5_is_not_ltx_2",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:ltxv_text_to_video": {
        "assetPath": "templates/ltxv_text_to_video.json",
        "assetSha256": "8bbd5219ec8351fc55da8e8c8cfadcd687e0e6b28edebc49473c9c26ec156392",
        "gitBlobOid": "f82d2e921c9af597e51df82b525313cdcbd54e98",
        "artifactDependencies": [
            {"repository": "Lightricks/LTX-Video", "artifact": "ltx-video-2b-v0.9.safetensors"},
            {"repository": "comfyanonymous/flux_text_encoders", "artifact": "t5xxl_fp16.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:text_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_video_0_9_is_not_ltx_2",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_ltx2_3_i2v": {
        "assetPath": "templates/video_ltx2_3_i2v.json",
        "assetSha256": "91dd8e44926fd37f6d9307789484370fa333582b14e53ed771d63ed805379ee4",
        "gitBlobOid": "549f9861e62fa21d931640bad4156a1cbce08a9c",
        "artifactDependencies": [
            {"repository": "Lightricks/LTX-2.3-fp8", "artifact": "ltx-2.3-22b-dev-fp8.safetensors"},
            {
                "repository": "Comfy-Org/ltx-2.3",
                "artifact": "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
            },
            {
                "repository": "Lightricks/LTX-2.3",
                "artifact": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            },
            {
                "repository": "Comfy-Org/ltx-2",
                "artifact": "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:image_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_2_3_22b_assembly_is_not_ltx_2",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_ltx2_3_t2v": {
        "assetPath": "templates/video_ltx2_3_t2v.json",
        "assetSha256": "75b10f3ee48c1fe00c7fb21b24c0c247b133e5ee34676144de4b652ac7dcbe7f",
        "gitBlobOid": "7e3895f18a7dc7ff5005f995469db9ccc2779077",
        "artifactDependencies": [
            {"repository": "Lightricks/LTX-2.3-fp8", "artifact": "ltx-2.3-22b-dev-fp8.safetensors"},
            {
                "repository": "Comfy-Org/ltx-2.3",
                "artifact": "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
            },
            {
                "repository": "Lightricks/LTX-2.3",
                "artifact": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            },
            {
                "repository": "Comfy-Org/ltx-2",
                "artifact": "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:text_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_2_3_22b_assembly_is_not_ltx_2",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_ltx2_5_i2v": {
        "assetPath": "templates/video_ltx2_5_i2v.json",
        "assetSha256": "bcd3239835e8e5bf287a664954c253c67cd31147a4a4193ef5975525e246a7a0",
        "gitBlobOid": "355653e7cb940543c3ad91b08ae32cf05c0d33ab",
        "artifactDependencies": [
            {"repository": "Comfy-Org/gemma-4", "artifact": "gemma4_e2b_it_bf16.safetensors"},
            {
                "repository": "Lightricks/LTX-2.5",
                "artifact": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            },
            {"repository": "Lightricks/LTX-2.5", "artifact": "ltx-2.5-video-vae-bf16.safetensors"},
            {"repository": "Lightricks/LTX-2.5", "artifact": "ltx-2.5-audio-vae-bf16.safetensors"},
            {
                "repository": "Lightricks/LTX-2.5",
                "artifact": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            },
            {
                "repository": "Lightricks/LTX-2.5",
                "artifact": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:image_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_2_5_22b_assembly_is_not_ltx_2",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_ltx2_5_t2v": {
        "assetPath": "templates/video_ltx2_5_t2v.json",
        "assetSha256": "b8ab11a3cb349bf6dccd9ad09307213e0088d833d1867270d23e1f794bab6a9d",
        "gitBlobOid": "60683c9f3cd9c708581e1fb2e2030d987d540634",
        "artifactDependencies": [
            {"repository": "Comfy-Org/gemma-4", "artifact": "gemma4_e2b_it_bf16.safetensors"},
            {
                "repository": "Lightricks/LTX-2.5",
                "artifact": "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            },
            {"repository": "Lightricks/LTX-2.5", "artifact": "ltx-2.5-video-vae-bf16.safetensors"},
            {"repository": "Lightricks/LTX-2.5", "artifact": "ltx-2.5-audio-vae-bf16.safetensors"},
            {
                "repository": "Lightricks/LTX-2.5",
                "artifact": "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            },
            {
                "repository": "Lightricks/LTX-2.5",
                "artifact": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:text_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_2_5_22b_assembly_is_not_ltx_2",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:template_ltx2_3_ic_lora_ingredients": {
        "assetPath": "templates/template_ltx2_3_ic_lora_ingredients.json",
        "assetSha256": "7d4dec964ab81ba4e0dd171b91842044ff486a0ca832d0a6bec0153a5c76fa74",
        "gitBlobOid": "3d7c046be2bc55ba54e53087c3289a711f70fc6f",
        "artifactDependencies": [
            {"repository": "Lightricks/LTX-2.3-fp8", "artifact": "ltx-2.3-22b-distilled-fp8.safetensors"},
            {
                "repository": "Comfy-Org/ltx-2.3",
                "artifact": "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
            },
            {
                "repository": "Comfy-Org/ltx-2",
                "artifact": "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:reference_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_2_3_reference_lora_assembly_is_not_ltx_2",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_ltx2_3_ia2v": {
        "assetPath": "templates/video_ltx2_3_ia2v.json",
        "assetSha256": "7823a703f472d9c5e6f82c462235ff89a0fa14752ec1fd947c4422cf53e47685",
        "gitBlobOid": "ed24f8c177844139466ee55d63e9aba4dae7749d",
        "reviewedTaskMode": "image_audio_to_video",
        "artifactDependencies": [
            {"repository": "Lightricks/LTX-2.3-fp8", "artifact": "ltx-2.3-22b-dev-fp8.safetensors"},
            {
                "repository": "Comfy-Org/ltx-2.3",
                "artifact": "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
            },
            {
                "repository": "Lightricks/LTX-2.3",
                "artifact": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            },
            {
                "repository": "Comfy-Org/ltx-2",
                "artifact": "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "ltx_2_3_image_audio_conditioning_requires_new_task_and_model",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_image_audio_video_task_and_model_generation_required",
    },
    "comfy-research:template:video_ltx2_i2v_distilled": {
        "assetPath": "templates/video_ltx2_i2v_distilled.json",
        "assetSha256": "054a4f1029f04fbe391eb6a66eb1c65baa29daf2bbf5930c7031b4ee0f642781",
        "gitBlobOid": "d7635abc038557428bd4e9c8ed4b0c5fa22d6047",
        "artifactDependencies": [
            {"repository": "Lightricks/LTX-2", "artifact": "ltx-2-19b-distilled.safetensors"},
            {
                "repository": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
                "artifact": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
            },
            {
                "repository": "Lightricks/LTX-2",
                "artifact": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_requires_auxiliary_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:image_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_2_camera_lora_and_latent_upscaler_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_auxiliary_artifact_not_admitted_exact_checkpoint_not_proven",
    },
    "comfy-research:template:video_ltx2_i2v_lora": {
        "assetPath": "templates/video_ltx2_i2v_lora.json",
        "assetSha256": "f53787bcd98d09f74a66ed42248e384986095d855d82509da59bf5e390453f69",
        "gitBlobOid": "0339c0e5959cf699c573f05481738d4bde95474f",
        "artifactDependencies": [
            {"repository": "Lightricks/LTX-2", "artifact": "ltx-2-19b-dev.safetensors"},
            {"repository": "Lightricks/LTX-2", "artifact": "ltx-2-19b-distilled-lora-384.safetensors"},
            {
                "repository": "Lightricks/LTX-2",
                "artifact": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            },
            {"repository": "Comfy-Org/ltx-2", "artifact": "ltx2-squish.safetensors"},
        ],
        "comparison": {
            "state": "same_upstream_generation_requires_auxiliary_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:image_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_2_distilled_squish_loras_and_latent_upscaler_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_auxiliary_artifact_not_admitted_exact_checkpoint_not_proven",
    },
    "comfy-research:template:video_ltx2_t2v_distilled": {
        "assetPath": "templates/video_ltx2_t2v_distilled.json",
        "assetSha256": "90d55eab58cb969f19f45d874304fc0d1b51cd79875bbdc21d5a6b1372114c7f",
        "gitBlobOid": "890dd2b04308f2db59be5b2e24946e14387273eb",
        "artifactDependencies": [
            {"repository": "Lightricks/LTX-2", "artifact": "ltx-2-19b-distilled.safetensors"},
            {
                "repository": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
                "artifact": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
            },
            {
                "repository": "Lightricks/LTX-2",
                "artifact": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_requires_auxiliary_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:text_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "ltx_2_camera_lora_and_latent_upscaler_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_auxiliary_artifact_not_admitted_exact_checkpoint_not_proven",
    },
    "comfy-research:template:image_to_video_wan": {
        "assetPath": "templates/image_to_video_wan.json",
        "assetSha256": "12930b58377d52e9581ad25b1c24bff93fddb693bd9066b9fdd93cafda59182f",
        "gitBlobOid": "acde9a84eb81fd560275032192595beb0292fd1c",
        "reviewedTaskMode": "image_to_video",
        "reviewedWorkflowId": "WanImageToVideoPipeline:image_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan2.1_i2v_480p_14B_fp16.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan_2.1_vae.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "WanImageToVideoPipeline:image_to_video",
            "currentRepository": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "currentRevision": "596658fd9ca6b7b71d5057529bbf319ecbc61d74",
            "reasonCode": "wan2_1_i2v_480p_14b_is_not_wan2_2_i2v_a14b",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_wan2.1_alpha_t2v_14B": {
        "assetPath": "templates/video_wan2.1_alpha_t2v_14B.json",
        "assetSha256": "57b22a6cb4577473bfcd527f8d01a47c86d357acb263759828d4886a878d95e0",
        "gitBlobOid": "ac6049b77add3cf6a43c542fb5153d85f330a1cd",
        "reviewedWorkflowId": "WanVideoPipeline:text_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan2.1_t2v_14B_fp8_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan_alpha_2.1_rgba_lora.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan_alpha_2.1_vae_rgb_channel.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan_alpha_2.1_vae_alpha_channel.safetensors",
            },
            {
                "repository": "Kijai/WanVideo_comfy",
                "artifact": "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "WanVideoPipeline:text_to_video",
            "currentRepository": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "currentRevision": "0fad780a534b6463e45facd96134c9f345acfa5b",
            "reasonCode": "wan2_1_alpha_14b_assembly_is_not_wan2_1_t2v_1_3b",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_wan2.1_fun_camera_v1.1_1.3B": {
        "assetPath": "templates/video_wan2.1_fun_camera_v1.1_1.3B.json",
        "assetSha256": "e8d590f10dd5fb80bca0e5dd83d9722a16bdc14aaa0af3119316b2d6fe7f5c8e",
        "gitBlobOid": "9a341101b06053db4e022b41809d9a56bfc69810",
        "reviewedTaskMode": "camera_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan_2.1_vae.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "wan2_1_fun_camera_requires_camera_motion_task_and_model",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_camera_video_task_and_model_generation_required",
    },
    "comfy-research:template:video_wan2.1_fun_camera_v1.1_14B": {
        "assetPath": "templates/video_wan2.1_fun_camera_v1.1_14B.json",
        "assetSha256": "e955740d40b1b5c5479b6338a27ea1a6d542cf0a8acbdb6d5332900c84bead54",
        "gitBlobOid": "386e500f9e6deb022c0da1281aef36983964161e",
        "reviewedTaskMode": "camera_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan2.1_fun_camera_v1.1_14B_bf16.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan_2.1_vae.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "wan2_1_fun_camera_14b_requires_camera_motion_task_and_model",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_camera_video_task_and_model_generation_required",
    },
    "comfy-research:template:video_wan2_2_14B_fun_camera": {
        "assetPath": "templates/video_wan2_2_14B_fun_camera.json",
        "assetSha256": "aa7b009804e4ba117a3bf597d7c0c1e0a21c43ff58b59c3630816b6fb00f58fe",
        "gitBlobOid": "a2aaea351b04b2ce6b4284ccfcba33f91e6a8cf9",
        "reviewedTaskMode": "camera_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan_2.1_vae.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "wan2_2_fun_camera_requires_camera_motion_task_and_model",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_camera_video_task_and_model_generation_required",
    },
    "comfy-research:template:video_wan2_2_14B_fun_control": {
        "assetPath": "templates/video_wan2_2_14B_fun_control.json",
        "assetSha256": "a0f68cfa069c425acf59587c73588d0bcbbaf703d0b0625357343a367ab35505",
        "gitBlobOid": "643e223e867ca74e808c700917d3272563051478",
        "reviewedTaskMode": "control_video_to_video",
        "reviewedWorkflowId": "AnimateDiffVideoToVideoControlNetPipeline:control_video_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan_2.1_vae.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "AnimateDiffVideoToVideoControlNetPipeline:control_video_to_video",
            "currentRepository": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "currentRevision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            "currentAuxiliaryRepositories": ["guoyww/animatediff-motion-adapter-v1-5-2"],
            "reasonCode": "wan2_2_fun_control_reuses_boundary_not_animatediff_model",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_causal_forcing_i2v": {
        "assetPath": "templates/video_causal_forcing_i2v.json",
        "assetSha256": "9fa0b00d61da61966b5149bda4fa6da295247680f1a00632e518faf591c03ca2",
        "gitBlobOid": "03ee6adfc18b1c49021b4bd81bb43ac40a3d8ff0",
        "reviewedWorkflowId": "WanImageToVideoPipeline:image_to_video",
        "artifactDependencies": [
            {
                "repository": "TalmajM/causal_forcing_framewise_ComfyUI_repackaged",
                "artifact": "causal_forcing-framewise.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan_2.1_vae.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "WanImageToVideoPipeline:image_to_video",
            "currentRepository": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "currentRevision": "596658fd9ca6b7b71d5057529bbf319ecbc61d74",
            "reasonCode": "causal_forcing_framewise_is_not_wan2_2_i2v",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_wan2_2_5B_fun_control": {
        "assetPath": "templates/video_wan2_2_5B_fun_control.json",
        "assetSha256": "2e80a254d3fb263f02d1a74e6abb0676d4dbb6aaf813575bf75ff4bda7bbb40d",
        "gitBlobOid": "92cd847090ae109ec9dec8182e021ac719243931",
        "reviewedTaskMode": "control_video_to_video",
        "reviewedWorkflowId": "AnimateDiffVideoToVideoControlNetPipeline:control_video_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_fun_control_5B_bf16.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            {"repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "artifact": "wan2.2_vae.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "AnimateDiffVideoToVideoControlNetPipeline:control_video_to_video",
            "currentRepository": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "currentRevision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            "currentAuxiliaryRepositories": ["guoyww/animatediff-motion-adapter-v1-5-2"],
            "reasonCode": "wan2_2_5b_fun_control_reuses_boundary_not_animatediff_model",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_wanmove_480p": {
        "assetPath": "templates/video_wanmove_480p.json",
        "assetSha256": "6492b8deabf5b3151402d003a73c8b79615edb34e2b23ac1b8617d70e407da85",
        "gitBlobOid": "294015f805fbb69564a662029d96122105b364a9",
        "reviewedTaskMode": "motion_track_to_video",
        "artifactDependencies": [
            {
                "repository": "Kijai/WanVideo_comfy_fp8_scaled",
                "artifact": "Wan21-WanMove_fp8_scaled_e4m3fn_KJ.safetensors",
            },
            {
                "repository": "Kijai/WanVideo_comfy",
                "artifact": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan_2.1_vae.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "wanmove_requires_motion_track_task_and_model",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_motion_track_video_task_and_model_generation_required",
    },
    "comfy-research:template:video_wan2_2_5B_fun_inpaint": {
        "assetPath": "templates/video_wan2_2_5B_fun_inpaint.json",
        "assetSha256": "47a669db3795ec2f06b99f00c579c3e214530cb16bb67a2a08b5428af784fb79",
        "gitBlobOid": "15161f3c41adfb40228a50da6c0238ab5c649b7b",
        "reviewedTaskMode": "first_last_frame_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                "artifact": "wan2.2_fun_inpaint_5B_bf16.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            {"repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "artifact": "wan2.2_vae.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "wan_fun_inpaint_graph_is_first_last_frame_generation_not_video_inpaint",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_first_last_frame_video_task_and_model_generation_required",
    },
    "comfy-research:template:video_wan_vace_14B_ref2v": {
        "assetPath": "templates/video_wan_vace_14B_ref2v.json",
        "assetSha256": "7c120c85409ca5b87ac5caad3245470c005b056adfe732392f892b8d0a9ea455",
        "gitBlobOid": "55cb2a0b5736a48b0128fb04ef64dd028907ad74",
        "reviewedTaskMode": "reference_to_video",
        "reviewedWorkflowId": "LTX2ConditionPipeline:reference_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan2.1_vace_14B_fp16.safetensors",
            },
            {
                "repository": "Kijai/WanVideo_comfy",
                "artifact": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp16.safetensors",
            },
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "wan_2.1_vae.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "LTX2ConditionPipeline:reference_to_video",
            "currentRepository": "Lightricks/LTX-2",
            "currentRevision": "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            "reasonCode": "wan_vace_14b_reference_generation_is_not_ltx2",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_wan_vace_14B_t2v": {
        "assetPath": "templates/video_wan_vace_14B_t2v.json",
        "assetSha256": "d022cdcf44060fb017b53188d419996aaa1bba09262a72c8ed2d3cd2f1443893",
        "gitBlobOid": "bbf56c17732e6db1a7182a721afdf497c6847245",
        "reviewedWorkflowId": "WanVACEPipeline:text_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan2.1_vace_14B_fp16.safetensors",
            },
            {
                "repository": "Kijai/WanVideo_comfy",
                "artifact": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp16.safetensors",
            },
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "wan_2.1_vae.safetensors"},
        ],
        "comparison": {
            "state": "same_upstream_generation_different_partition_and_auxiliary",
            "currentWorkflowId": "WanVACEPipeline:text_to_video",
            "currentRepository": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "currentRevision": "ec4d2cb062b548996b179d493fdd05340de702a1",
            "reasonCode": "wan_vace_14b_partition_and_causvid_lora_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_partition_and_auxiliary_exact_checkpoint_not_proven",
    },
    "comfy-research:template:video_wan_vace_14B_v2v": {
        "assetPath": "templates/video_wan_vace_14B_v2v.json",
        "assetSha256": "b1b79044f20eea5590aa419b4ca28a5fb455981c38ffd22db8a9fb6f641e00b9",
        "gitBlobOid": "716f54717cf4b80800f994d641b72be96d8a197f",
        "reviewedTaskMode": "control_to_video",
        "reviewedWorkflowId": "WanVACEPipeline:control_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan2.1_vace_14B_fp16.safetensors",
            },
            {
                "repository": "Kijai/WanVideo_comfy",
                "artifact": "Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp16.safetensors",
            },
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "wan_2.1_vae.safetensors"},
        ],
        "comparison": {
            "state": "same_upstream_generation_different_partition_and_auxiliary",
            "currentWorkflowId": "WanVACEPipeline:control_to_video",
            "currentRepository": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "currentRevision": "ec4d2cb062b548996b179d493fdd05340de702a1",
            "reasonCode": "wan_vace_14b_control_partition_and_causvid_lora_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_partition_and_auxiliary_exact_checkpoint_not_proven",
    },
    "comfy-research:template:wan2.1_fun_control": {
        "assetPath": "templates/wan2.1_fun_control.json",
        "assetSha256": "aff0f79b23f071f5aee6293ac0d6da6f254762e19df783243e151f61d6929fab",
        "gitBlobOid": "15b557afbee8c9b56270576316e285a4b3f1ad79",
        "reviewedTaskMode": "control_to_video",
        "reviewedWorkflowId": "WanVACEPipeline:control_to_video",
        "artifactDependencies": [
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "wan2.1_fun_control_1.3B_bf16.safetensors",
            },
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            },
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "wan_2.1_vae.safetensors"},
            {
                "repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "artifact": "clip_vision_h.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "WanVACEPipeline:control_to_video",
            "currentRepository": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "currentRevision": "ec4d2cb062b548996b179d493fdd05340de702a1",
            "reasonCode": "wan_fun_control_model_is_not_wan_vace",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_ernie_image": {
        "assetPath": "templates/image_ernie_image.json",
        "assetSha256": "d75ba44837ed432a5acf4f4eeb6a6f688b55827f8f2cfb7a464e09a8bf67be2c",
        "gitBlobOid": "4b841c64cd1742dced75614e4b51747ee13adcaf",
        "artifactDependencies": [
            {"repository": "Comfy-Org/ERNIE-Image", "artifact": "ernie-image.safetensors"},
            {"repository": "Comfy-Org/ERNIE-Image", "artifact": "ministral-3-3b.safetensors"},
            {"repository": "Comfy-Org/ERNIE-Image", "artifact": "flux2-vae.safetensors"},
            {
                "repository": "Comfy-Org/ERNIE-Image",
                "artifact": "ernie-image-prompt-enhancer.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "ErnieImagePipeline:text_to_image",
            "currentRepository": "baidu/ERNIE-Image-Turbo",
            "currentRevision": "bc68c81e2a1730a394d5fc9fae70713dee940140",
            "reasonCode": "ernie_image_is_not_ernie_image_turbo",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_kandinsky5_t2i": {
        "assetPath": "templates/image_kandinsky5_t2i.json",
        "assetSha256": "1cf174899cbe087c2dc4c187e938b877c4a8a1cac28305c2279f6d35cc1b4be4",
        "gitBlobOid": "f1f9521c422b35c991902d6e61335fecf024579f",
        "artifactDependencies": [
            {
                "repository": "kandinskylab/Kandinsky-5.0-T2I-Lite",
                "artifact": "kandinsky5lite_t2i.safetensors",
            },
            {
                "repository": "Comfy-Org/HunyuanVideo_1.5_repackaged",
                "artifact": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            },
            {"repository": "comfyanonymous/flux_text_encoders", "artifact": "clip_l.safetensors"},
            {"repository": "Comfy-Org/z_image_turbo", "artifact": "ae.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "Kandinsky3Pipeline:text_to_image",
            "currentRepository": "kandinsky-community/kandinsky-3",
            "currentRevision": "bf79e6c219da8a94abb50235fdc4567eb8fb4632",
            "reasonCode": "kandinsky_5_lite_is_not_kandinsky_3",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_lotus_depth_v1_1": {
        "assetPath": "templates/image_lotus_depth_v1_1.json",
        "assetSha256": "829014bad41d91b27783e2f5a1eec98e9ffefc6a04a3096f17c9d14219c22493",
        "gitBlobOid": "46b550b2a8f006edb5592465051b5ea88ff1d192",
        "reviewedTaskMode": "depth_estimation",
        "reviewedWorkflowId": "MarigoldDepthPipeline:depth_estimation",
        "artifactDependencies": [
            {"repository": "Comfy-Org/lotus", "artifact": "lotus-depth-d-v1-1.safetensors"},
            {
                "repository": "stabilityai/sd-vae-ft-mse-original",
                "artifact": "vae-ft-mse-840000-ema-pruned.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "MarigoldDepthPipeline:depth_estimation",
            "currentRepository": "prs-eth/marigold-depth-lcm-v1-0",
            "currentRevision": "04a73502f7fd8fc5e59947b9df3b2266d71d6849",
            "reasonCode": "lotus_depth_is_not_marigold_depth_lcm",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_netayume_lumina_t2i": {
        "assetPath": "templates/image_netayume_lumina_t2i.json",
        "assetSha256": "c9ed8360e179a3d68e110ced68b8b1db74e6f8a4261652b8eeda30c753bab676",
        "gitBlobOid": "8d7426f8ca3ada611df2b785ff1cac952a06aa1b",
        "reviewedWorkflowId": "Lumina2Pipeline:text_to_image",
        "artifactDependencies": [
            {
                "repository": "duongve/NetaYume-Lumina-Image-2.0",
                "artifact": "NetaYumev35_pretrained_all_in_one.safetensors",
            }
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "Lumina2Pipeline:text_to_image",
            "currentRepository": "Alpha-VLLM/Lumina-Image-2.0",
            "currentRevision": "53504abd8178b30685b6c4c7a4cd181ff78b73e9",
            "reasonCode": "netayume_lumina_finetune_is_not_lumina_image_2_base",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_omnigen2_t2i": {
        "assetPath": "templates/image_omnigen2_t2i.json",
        "assetSha256": "d0ccb8ea4d207288e0efa7ef929c3cedc24f8bb3bf7de1827e584d7d3b6d7bbc",
        "gitBlobOid": "4b4df6632653eef2f4e192e7e988e82d20ccee9f",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Omnigen2_ComfyUI_repackaged", "artifact": "omnigen2_fp16.safetensors"},
            {
                "repository": "Comfy-Org/Omnigen2_ComfyUI_repackaged",
                "artifact": "qwen_2.5_vl_fp16.safetensors",
            },
            {"repository": "Comfy-Org/Lumina_Image_2.0_Repackaged", "artifact": "ae.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "OmniGenPipeline:text_to_image",
            "currentRepository": "Shitao/OmniGen-v1-diffusers",
            "currentRevision": "016e2f61d12a98303f6bbdf122687694d7984268",
            "reasonCode": "omnigen2_is_not_omnigen_v1",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:image_omnigen2_image_edit": {
        "assetPath": "templates/image_omnigen2_image_edit.json",
        "assetSha256": "3a63f64bf3b58ad8fa761e4606d7d5ca1e6efd42fc1df57afe9e3e6d075ca593",
        "gitBlobOid": "c14f55f4797cf66a0980a5dedf51919f91865942",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Omnigen2_ComfyUI_repackaged", "artifact": "omnigen2_fp16.safetensors"},
            {
                "repository": "Comfy-Org/Omnigen2_ComfyUI_repackaged",
                "artifact": "qwen_2.5_vl_fp16.safetensors",
            },
            {"repository": "Comfy-Org/Lumina_Image_2.0_Repackaged", "artifact": "ae.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "OmniGenPipeline:edit_image",
            "currentRepository": "Shitao/OmniGen-v1-diffusers",
            "currentRevision": "016e2f61d12a98303f6bbdf122687694d7984268",
            "reasonCode": "omnigen2_is_not_omnigen_v1",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:sdxl_refiner_prompt_example": {
        "assetPath": "templates/sdxl_refiner_prompt_example.json",
        "assetSha256": "c376af45ad93aeb52326d51d73f2d218c85423e36eec05a498edda50b769e4fb",
        "gitBlobOid": "88dceaf20deea2bad07352daf72e0ac445300b84",
        "reviewedWorkflowId": "StableDiffusionXLPipeline:text_to_image",
        "artifactDependencies": [
            {
                "repository": "stabilityai/stable-diffusion-xl-base-1.0",
                "artifact": "sd_xl_base_1.0.safetensors",
            },
            {
                "repository": "stabilityai/stable-diffusion-xl-refiner-1.0",
                "artifact": "sd_xl_refiner_1.0.safetensors",
            },
        ],
        "comparison": {
            "state": "same_upstream_generation_requires_auxiliary_admission",
            "currentWorkflowId": "StableDiffusionXLPipeline:text_to_image",
            "currentRepository": "stabilityai/stable-diffusion-xl-base-1.0",
            "currentRevision": "462165984030d82259a11f4367a4eed129e94a7b",
            "reasonCode": "sdxl_refiner_stage_and_artifact_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_auxiliary_artifact_not_admitted_exact_checkpoint_not_proven",
    },
    "comfy-research:template:sdxl_revision_text_prompts": {
        "assetPath": "templates/sdxl_revision_text_prompts.json",
        "assetSha256": "4f48ce57fc864689e9de3d7575394cd4eed7973fb49dec43827dbf45cc5216b1",
        "gitBlobOid": "383ee14ed5cb7a468f0b4b44406594958cb24013",
        "reviewedTaskMode": "reference_to_image",
        "artifactDependencies": [
            {
                "repository": "stabilityai/stable-diffusion-xl-base-1.0",
                "artifact": "sd_xl_base_1.0.safetensors",
            },
            {"repository": "comfyanonymous/clip_vision_g", "artifact": "clip_vision_g.safetensors"},
        ],
        "comparison": {
            "state": "same_upstream_generation_and_new_task_auxiliary_required",
            "currentWorkflowId": None,
            "currentRepository": "stabilityai/stable-diffusion-xl-base-1.0",
            "currentRevision": "462165984030d82259a11f4367a4eed129e94a7b",
            "reasonCode": "sdxl_unclip_revision_requires_reference_image_task_and_clip_vision",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_reference_image_task_and_auxiliary_required",
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
    "comfy-research:template:utility_video_frame_interpolation": {
        "assetPath": "templates/utility_video_frame_interpolation.json",
        "assetSha256": "d79bbc9f662c4c40fef1095c10e861a9b25b5fb38a1e62a779d0ef81d7602fc9",
        "gitBlobOid": "7dfba76f14ce008309cf8940fad2eafcb9ad47dc",
        "reviewedWorkflowId": "BuiltinVideoOperation:frame_interpolation",
        "artifactDependencies": [
            {"repository": "Comfy-Org/frame_interpolation", "artifact": "film_net_fp16.safetensors"}
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "BuiltinVideoOperation:frame_interpolation",
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "film_network_is_not_the_builtin_frame_interpolation_implementation",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:utility_seedvr2_3b_int8_upscale_video": {
        "assetPath": "templates/utility_seedvr2_3b_int8_upscale_video.json",
        "assetSha256": "f3ef8ed12f44457f012ed174d2d421e9bfdde1fd5b76dd19d7c5a3d1437156d4",
        "gitBlobOid": "c1404a5cbb86992d2fb7c48b556c9781339674fb",
        "reviewedWorkflowId": "SpandrelVideoUpscale:video_upscale",
        "artifactDependencies": [
            {"repository": "Comfy-Org/SeedVR2", "artifact": "seedvr2_ema_vae_fp16.safetensors"},
            {"repository": "Comfy-Org/SeedVR2", "artifact": "seedvr2_3b_int8_convrot.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "SpandrelVideoUpscale:video_upscale",
            "currentRepository": "nateraw/real-esrgan",
            "currentRevision": "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094",
            "reasonCode": "seedvr2_video_upscaler_is_not_realesrgan_spandrel",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:utility_pid_latent_upscale_dit": {
        "assetPath": "templates/utility_pid_latent_upscale_dit.json",
        "assetSha256": "f5e2d5eb5e061e892ebc7491d33776e59aa29bd38506c5562ba06df4afd9513d",
        "gitBlobOid": "4d7a20ed232561945e4730f1a7da037ee53ce071",
        "reviewedTaskMode": "text_to_image",
        "reviewedWorkflowId": "ZImageModularPipeline:text_to_image",
        "artifactDependencies": [
            {"repository": "Comfy-Org/PixelDiT", "artifact": "gemma_2_2b_it_elm_fp8_scaled.safetensors"},
            {"repository": "Comfy-Org/PixelDiT", "artifact": "pid_flux1_1024_to_4096_4step_bf16.safetensors"},
            {"repository": "Comfy-Org/z_image_turbo", "artifact": "qwen_3_4b.safetensors"},
            {"repository": "Comfy-Org/z_image_turbo", "artifact": "ae.safetensors"},
            {"repository": "Comfy-Org/z_image_turbo", "artifact": "z_image_turbo_bf16.safetensors"},
        ],
        "comparison": {
            "state": "same_upstream_generation_requires_auxiliary_admission",
            "currentWorkflowId": "ZImageModularPipeline:text_to_image",
            "currentRepository": "Tongyi-MAI/Z-Image-Turbo",
            "currentRevision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
            "reasonCode": "pixeldit_latent_upscale_stage_and_artifacts_not_admitted",
        },
        "resolutionState": "existing_family_workflow_candidate",
        "mappingMeaning": "source_reviewed_family_candidate_auxiliary_artifact_not_admitted_exact_checkpoint_not_proven",
    },
    "comfy-research:template:utility_seedvr2_3b_int8_upscale_image": {
        "assetPath": "templates/utility_seedvr2_3b_int8_upscale_image.json",
        "assetSha256": "3f26fec421168072c44fe008db9654b846def5a2ee8a1b5baa2845709ee5e3a4",
        "gitBlobOid": "540d96107bbf469238af38d3c48800b110a83e31",
        "reviewedWorkflowId": "BuiltinImageOperation:image_upscale",
        "artifactDependencies": [
            {"repository": "Comfy-Org/SeedVR2", "artifact": "seedvr2_ema_vae_fp16.safetensors"},
            {"repository": "Comfy-Org/SeedVR2", "artifact": "seedvr2_3b_int8_convrot.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "BuiltinImageOperation:image_upscale",
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "seedvr2_3b_image_upscaler_is_not_builtin_interpolation",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:utility_seedvr2_7b_int8_upscale_image": {
        "assetPath": "templates/utility_seedvr2_7b_int8_upscale_image.json",
        "assetSha256": "36d7072d0a518034be5e3b159107af7d10a3424f5a1618f1f8991530623be309",
        "gitBlobOid": "4f8920827c0cdee42e6cc5e6a9f22759358dc20c",
        "reviewedWorkflowId": "BuiltinImageOperation:image_upscale",
        "artifactDependencies": [
            {"repository": "Comfy-Org/SeedVR2", "artifact": "seedvr2_ema_vae_fp16.safetensors"},
            {"repository": "Comfy-Org/SeedVR2", "artifact": "seedvr2_7b_int8_convrot.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "BuiltinImageOperation:image_upscale",
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "seedvr2_7b_image_upscaler_is_not_builtin_interpolation",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:utility_birefnet_remove_background": {
        "assetPath": "templates/utility_birefnet_remove_background.json",
        "assetSha256": "4b473eae9c4c034fd9526ae5511afb427791626a3500aa04dbd7701f423cb904",
        "gitBlobOid": "31c25593f87170ced962316ab8747463b70c1410",
        "artifactDependencies": [
            {"repository": "Comfy-Org/BiRefNet", "artifact": "birefnet.safetensors"}
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "birefnet_requires_remove_background_task_and_model",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_remove_background_task_and_model_generation_required",
    },
    "comfy-research:template:llm_gemma4_text_gen": {
        "assetPath": "templates/llm_gemma4_text_gen.json",
        "assetSha256": "a9673276612a885bc2898ef7cf2faea50fb78d75f5ee328620aa484b85b47469",
        "gitBlobOid": "27589bbb387986f1279ac306053b3064e44cfc0f",
        "reviewedTaskMode": "image_audio_to_text",
        "artifactDependencies": [
            {"repository": "Comfy-Org/gemma-4", "artifact": "gemma4_e4b_it_fp8_scaled.safetensors"}
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "gemma4_graph_requires_combined_image_audio_conditioned_text_task",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_image_audio_text_task_and_model_generation_required",
    },
    "comfy-research:template:llm_qwen3_5_text_gen": {
        "assetPath": "templates/llm_qwen3_5_text_gen.json",
        "assetSha256": "8555d9f50e335ce9d2b4ea69b7531be3249c76062dfa272aa9eb933d736664e3",
        "gitBlobOid": "4cd72528d24bfb698bfe47d0edaa48a01cd6ad7e",
        "reviewedTaskMode": "image_to_text",
        "reviewedWorkflowId": "HuggingFaceImageTextToTextModel:image_to_text",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Qwen3.5", "artifact": "qwen3.5_4b_bf16.safetensors"}
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "HuggingFaceImageTextToTextModel:image_to_text",
            "currentRepository": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "currentRevision": "7e3e67edbbed1bf9888184d9df282b700a323964",
            "reasonCode": "qwen3_5_vision_language_generation_is_not_smolvlm",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:llm_qwen3_text_gen": {
        "assetPath": "templates/llm_qwen3_text_gen.json",
        "assetSha256": "b09690ecd1504aa292d3ade8b49bf0052e49b6cab8430b8bfbe960a5a7f6d02f",
        "gitBlobOid": "afe00a0e4f7523ac23e5f31f390c6280bf2af843",
        "reviewedWorkflowId": "HuggingFaceTextGenerationModel:text_generation",
        "artifactDependencies": [
            {"repository": "Comfy-Org/flux2-klein", "artifact": "qwen_3_4b.safetensors"}
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "HuggingFaceTextGenerationModel:text_generation",
            "currentRepository": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "currentRevision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
            "reasonCode": "qwen3_4b_generation_is_not_smollm2",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:llm_qwen3vl_text_gen": {
        "assetPath": "templates/llm_qwen3vl_text_gen.json",
        "assetSha256": "bed74ae16694e3887532824dfd2c58b7228afa0e43d45f8d93326506ab82ad97",
        "gitBlobOid": "e175bf1e88bfdb54ca00d226d276a481999a0e39",
        "reviewedTaskMode": "image_to_text",
        "reviewedWorkflowId": "HuggingFaceImageTextToTextModel:image_to_text",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Qwen3-VL", "artifact": "qwen3vl_4b_fp8_scaled.safetensors"}
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "HuggingFaceImageTextToTextModel:image_to_text",
            "currentRepository": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "currentRevision": "7e3e67edbbed1bf9888184d9df282b700a323964",
            "reasonCode": "qwen3vl_generation_is_not_smolvlm",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:audio_minimax_music_3": {
        "assetPath": "templates/audio_minimax_music_3.json",
        "assetSha256": "0322153265b3e785961511b7849f6659f46a8fa7e8cb66976e5279ff1774b228",
        "gitBlobOid": "acd3f160d3171a832d064e364c43fcf704316e92",
        "reviewedWorkflowId": "AceStepAudioPipeline:text_to_audio",
        "artifactDependencies": [
            {"repository": "Comfy-Org/MiniMax-Music-3", "artifact": "minimax_music3_dit_fp16.safetensors"},
            {
                "repository": "Comfy-Org/MiniMax-Music-3",
                "artifact": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
            },
            {"repository": "Comfy-Org/MiniMax-Music-3", "artifact": "minimax_music3_dav.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "AceStepAudioPipeline:text_to_audio",
            "currentRepository": "ACE-Step/acestep-v15-xl-turbo-diffusers",
            "currentRevision": "200ba991ae448051e14b0183157e35c2d27c9fb0",
            "reasonCode": "minimax_music3_is_not_ace_step",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_minimax_h3_r2v": {
        "assetPath": "templates/video_minimax_h3_r2v.json",
        "assetSha256": "edd1e3e2ac0cee2ac28337b311cc9b7c6984cb83ac3558a2fda8f8cacbbf2da7",
        "gitBlobOid": "90d4012e1f83f7102f9ab0f30e28a435f5e03b18",
        "reviewedTaskMode": "reference_to_video_with_audio",
        "reviewedOutputMediaKinds": ["video", "audio"],
        "artifactDependencies": [
            {"repository": "Comfy-Org/MiniMax-H3", "artifact": "minimax_h3_video_vae_fp16.safetensors"},
            {"repository": "Comfy-Org/MiniMax-H3", "artifact": "minimax_h3_audio_vae_fp32.safetensors"},
            {
                "repository": "Comfy-Org/MiniMax-H3",
                "artifact": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            },
            {
                "repository": "Comfy-Org/MiniMax-H3",
                "artifact": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "minimax_h3_reference_generation_requires_synchronized_video_audio_task",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_reference_video_audio_task_and_model_generation_required",
    },
    "comfy-research:template:video_minimax_h3_i2v": {
        "assetPath": "templates/video_minimax_h3_i2v.json",
        "assetSha256": "bb71aecdd3c0b62e56eafe03acb14d1cfeabec7072eaed9cbdf473c2aaf73009",
        "gitBlobOid": "65b7b1cc627fd892d50468041430706260ad81a0",
        "reviewedTaskMode": "image_to_video_with_audio",
        "reviewedOutputMediaKinds": ["video", "audio"],
        "artifactDependencies": [
            {"repository": "Comfy-Org/MiniMax-H3", "artifact": "minimax_h3_video_vae_fp16.safetensors"},
            {"repository": "Comfy-Org/MiniMax-H3", "artifact": "minimax_h3_audio_vae_fp32.safetensors"},
            {
                "repository": "Comfy-Org/MiniMax-H3",
                "artifact": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            },
            {
                "repository": "Comfy-Org/MiniMax-H3",
                "artifact": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "minimax_h3_image_generation_requires_synchronized_video_audio_task",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_image_video_audio_task_and_model_generation_required",
    },
    "comfy-research:template:video_minimax_h3_t2v": {
        "assetPath": "templates/video_minimax_h3_t2v.json",
        "assetSha256": "31ab33fdb053a7834cc866bd7aa08b887518fc656e4a796c89779c6b5e1786e6",
        "gitBlobOid": "2502a910c45e08c55b37dd5d422efef6e1877304",
        "reviewedTaskMode": "text_to_video_with_audio",
        "reviewedOutputMediaKinds": ["video", "audio"],
        "artifactDependencies": [
            {"repository": "Comfy-Org/MiniMax-H3", "artifact": "minimax_h3_video_vae_fp16.safetensors"},
            {"repository": "Comfy-Org/MiniMax-H3", "artifact": "minimax_h3_audio_vae_fp32.safetensors"},
            {
                "repository": "Comfy-Org/MiniMax-H3",
                "artifact": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            },
            {
                "repository": "Comfy-Org/MiniMax-H3",
                "artifact": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            },
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "minimax_h3_text_generation_requires_synchronized_video_audio_task",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_text_video_audio_task_and_model_generation_required",
    },
    "comfy-research:template:video_wan2_1_infinitetalk": {
        "assetPath": "templates/video_wan2_1_infinitetalk.json",
        "assetSha256": "29daad103b235336af64ea3ec87aa2a5ef1a88ea13d2f24c24c27413b3ade293",
        "gitBlobOid": "add0693ba9cd40026c4e9842a01fd02646fc16e7",
        "artifactDependencies": [
            {"repository": "Kijai/WanVideo_comfy_fp8_scaled", "artifact": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors"},
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
            {"repository": "Kijai/WanVideo_comfy", "artifact": "Wan2_1_VAE_bf16.safetensors"},
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "wan2.1_infiniteTalk_multi_fp16.safetensors"},
            {"repository": "Kijai/wav2vec2_safetensors", "artifact": "wav2vec2-chinese-base_fp16.safetensors"},
            {"repository": "Kijai/WanVideo_comfy", "artifact": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "infinitetalk_requires_audio_conditioned_video_task_and_model_patch",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_audio_video_task_and_model_generation_required",
    },
    "comfy-research:template:video_wan2_2_14B_s2v": {
        "assetPath": "templates/video_wan2_2_14B_s2v.json",
        "assetSha256": "e5e8eda3c0ec86fdeca79c1a8d8648685ccbd200708ec87aa2087b75d98d0a45",
        "gitBlobOid": "e1106394a66ad51011394c48064899f46735cdb3",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "artifact": "wan2.2_s2v_14B_fp8_scaled.safetensors"},
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
            {"repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "artifact": "wan_2.1_vae.safetensors"},
            {"repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "artifact": "wav2vec2_large_english_fp16.safetensors"},
            {"repository": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "artifact": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "wan_s2v_requires_audio_conditioned_video_task_and_model",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_audio_video_task_and_model_generation_required",
    },
    "comfy-research:template:video_wan_dancer": {
        "assetPath": "templates/video_wan_dancer.json",
        "assetSha256": "8558abb7332977ca80d20580887711c0919be2f122350c68d5407150d547c311",
        "gitBlobOid": "26ad8546a395a39c725351f8d5687343f9649b7e",
        "artifactDependencies": [
            {"repository": "Comfy-Org/Wan-Dancer", "artifact": "wan2.2_dancer_14b_global_fp8_scaled.safetensors"},
            {"repository": "Comfy-Org/Wan-Dancer", "artifact": "wan2.2_dancer_14b_local_fp8_scaled.safetensors"},
            {"repository": "Kijai/WanVideo_comfy", "artifact": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"},
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "umt5_xxl_fp16.safetensors"},
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "clip_vision_h.safetensors"},
            {"repository": "Kijai/WanVideo_comfy", "artifact": "Wan2_1_VAE_bf16.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_and_new_task_required",
            "currentWorkflowId": None,
            "currentRepository": None,
            "currentRevision": None,
            "reasonCode": "wan_dancer_requires_audio_pose_video_task_and_dual_model_generation",
        },
        "resolutionState": "new_task_boundary_required",
        "mappingMeaning": "source_reviewed_new_audio_video_task_and_model_generation_required",
    },
    "comfy-research:template:video_wan21_scail2_character_replacement": {
        "assetPath": "templates/video_wan21_scail2_character_replacement.json",
        "assetSha256": "d85a06af10724e66e5f7857e5f7d9fc10cf28fc528abe19b86514b04ee84ccde",
        "gitBlobOid": "d442da88aab84e0b9673162dc77e5e7a479e16c4",
        "reviewedWorkflowId": "WanAnimatePipeline:character_replace",
        "artifactDependencies": [
            {"repository": "Comfy-Org/SCAIL-2", "artifact": "wan2.1_14B_SCAIL_2_fp16.safetensors"},
            {"repository": "Comfy-Org/SCAIL-2", "artifact": "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"},
            {"repository": "Comfy-Org/sam3.1", "artifact": "sam3.1_multiplex_fp16.safetensors"},
            {"repository": "Kijai/WanVideo_comfy", "artifact": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"},
            {"repository": "Kijai/WanVideo_comfy", "artifact": "Wan2_1_VAE_bf16.safetensors"},
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "clip_vision_h.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "WanAnimatePipeline:character_replace",
            "currentRepository": "Wan-AI/Wan2.2-Animate-14B-Diffusers",
            "currentRevision": "6f4df10861c758af86ac3c979aacc1bf5c03eff0",
            "reasonCode": "scail2_and_sam3_character_replacement_is_not_wan_animate",
        },
        "resolutionState": "existing_task_boundary_model_admission_required",
        "mappingMeaning": "source_reviewed_task_boundary_only_different_model_generation",
    },
    "comfy-research:template:video_wan21_scail2_character_replacement_int8": {
        "assetPath": "templates/video_wan21_scail2_character_replacement_int8.json",
        "assetSha256": "e1f89c0e5bc58697ebef14a8d55569643e067de57b70d560418ba927912c748d",
        "gitBlobOid": "2c404d05c00a455aa76def65bc6be13e3fc93c7e",
        "reviewedWorkflowId": "WanAnimatePipeline:character_replace",
        "artifactDependencies": [
            {"repository": "Comfy-Org/SCAIL-2", "artifact": "wan2.1_14B_SCAIL_2_int8_convrot.safetensors"},
            {"repository": "Comfy-Org/SCAIL-2", "artifact": "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"},
            {"repository": "Comfy-Org/sam3.1", "artifact": "sam3.1_multiplex_fp16.safetensors"},
            {"repository": "Kijai/WanVideo_comfy", "artifact": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"},
            {"repository": "Kijai/WanVideo_comfy", "artifact": "Wan2_1_VAE_bf16.safetensors"},
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
            {"repository": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "artifact": "clip_vision_h.safetensors"},
        ],
        "comparison": {
            "state": "different_model_generation_requires_admission",
            "currentWorkflowId": "WanAnimatePipeline:character_replace",
            "currentRepository": "Wan-AI/Wan2.2-Animate-14B-Diffusers",
            "currentRevision": "6f4df10861c758af86ac3c979aacc1bf5c03eff0",
            "reasonCode": "scail2_int8_and_sam3_character_replacement_is_not_wan_animate",
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
        if pinned_source_review is not None and pinned_source_review.get("reviewedOutputMediaKinds") is not None:
            reviewed_output_kinds = pinned_source_review.get("reviewedOutputMediaKinds")
            if not isinstance(reviewed_output_kinds, list) or not reviewed_output_kinds or any(
                not isinstance(value, str) or not value for value in reviewed_output_kinds
            ):
                raise ComfyContractResolutionError(
                    f"Pinned Comfy source review output media kinds are invalid for {contract_id}."
                )
            pinned_source_review["catalogOutputMediaKinds"] = deepcopy(output_kinds)
            output_kinds = reviewed_output_kinds
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
                (workflow for workflow in task_options if workflow["id"] == reviewed_workflow_id),
                None,
            )
            if reviewed_workflow is None:
                raise ComfyContractResolutionError(
                    f"Pinned Comfy source review workflow is not a current task option for {contract_id}."
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
        resolved_catalog_families = set(catalog_families)
        if pinned_source_review is not None and resolution_state == "existing_family_workflow_candidate":
            if recommended is None:
                raise ComfyContractResolutionError(
                    f"Pinned Comfy family review has no current workflow recommendation for {contract_id}."
                )
            reviewed_family = recommended["modelFamily"]
            if reviewed_family not in resolved_catalog_families:
                pinned_source_review["catalogRecognizedMoDiffFamilies"] = sorted(catalog_families)
                resolved_catalog_families.add(reviewed_family)
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
                "recognizedMoDiffFamilies": sorted(resolved_catalog_families),
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
