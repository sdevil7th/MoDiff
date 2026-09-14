import copy
import hashlib
import json
import subprocess
import sys
import unittest

from modiff.diffusers_profiles import execution_profiles_for_execution
from modiff.huggingface_node_library import (
    HuggingFaceNodeLibraryError,
    build_huggingface_node_library,
    validate_huggingface_node_library,
)
from modiff.modular_block_role_adapters import (
    PINNED_CUSTOM_CONTAINER_BLOCK_ROLE_WORKFLOWS,
    PINNED_STATICALLY_CLOSED_BLOCK_ROLE_WORKFLOWS,
)
from modiff.modular_container_state_adapters import PINNED_CONTAINER_STATE_ADAPTER_TRUTH
from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES


class HuggingFaceNodeLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = build_huggingface_node_library()

    def definition(self, pipeline_class, workflow_id):
        return next(
            definition
            for definition in self.library["definitions"]
            if definition["pipelineClass"] == pipeline_class and definition["workflowId"] == workflow_id
        )

    def test_cached_sound_generators_keep_exact_native_recipe_contracts(self):
        for pipeline, label, duration, steps, guidance, sample_source, sample_rate in (
            ("LongCatAudioDiTPipeline", "LongCat AudioDiT", 5, 16, 4.0, "sampleRate24000", 24000),
            ("AudioLDM2Pipeline", "AudioLDM2", 10, 200, 3.5, "sampleRate16000", 16000),
        ):
            with self.subTest(pipeline=pipeline):
                definition = self.definition(pipeline, "text_to_audio")
                self.assertEqual(definition["definitionKind"], "studio_execution_composite")
                self.assertEqual(definition["label"], f"{label} — Text To Audio")
                fields = {field["name"]: field for field in definition["inputs"]}
                self.assertNotIn("lyrics", fields)
                self.assertEqual(fields["audio_duration"]["default"], duration)
                self.assertEqual(fields["num_inference_steps"]["default"], steps)
                self.assertEqual(fields["guidance_scale"]["default"], guidance)
                admission = definition["executionAdmissions"][0]
                self.assertEqual(admission["sealedBindingValues"][sample_source], sample_rate)
                self.assertEqual(admission["sealedBindingValues"]["text2audio"], "text2audio")
                self.assertFalse(admission["publication"]["autoEligible"])
                if pipeline == "AudioLDM2Pipeline":
                    self.assertEqual(admission["sealedBindingValues"]["numWaveforms3"], 3)

    def test_current_snapshot_builds_immutable_first_party_definitions(self):
        self.assertEqual(self.library["schemaVersion"], 6)
        self.assertEqual(len(self.library["definitions"]), 135)
        self.assertEqual(len(self.library["blockDefinitions"]), 559)
        self.assertEqual(self.library["providers"], ["diffusers", "transformers"])
        diffusers_definitions = [item for item in self.library["definitions"] if item["provider"] == "diffusers"]
        transformers_definitions = [item for item in self.library["definitions"] if item["provider"] == "transformers"]
        self.assertEqual(len(diffusers_definitions), 127)
        self.assertEqual(len(transformers_definitions), 8)
        for definition in diffusers_definitions:
            self.assertEqual(definition["schemaVersion"], 6)
            self.assertEqual(definition["provider"], "diffusers")
            self.assertEqual(definition["publisher"], "huggingface")
            self.assertEqual(definition["surface"], "diffusers_cluster_nodes")
            self.assertEqual(definition["ownership"], "library")
            self.assertIs(definition["mutable"], False)
            self.assertEqual(definition["executionClaim"], "discovery_only")
            self.assertTrue(definition["contentHash"].startswith("sha256:"))
            self.assertTrue(definition["blockContractHash"].startswith("sha256:"))
            self.assertTrue(definition["blockPlacements"])
            self.assertEqual(definition["suggestedInputs"]["schemaVersion"], 1)
            self.assertTrue(
                definition["rootBlockDefinitionId"].startswith(
                    "diffusers.composite-block:"
                    if definition["definitionKind"] == "studio_execution_composite"
                    else "diffusers.modular-block:"
                )
            )
        for definition in transformers_definitions:
            self.assertEqual(definition["surface"], "transformers_cluster_nodes")
            self.assertEqual(definition["definitionKind"], "studio_execution_composite")
            self.assertEqual(definition["integrationStatus"], "reviewed_transformers_contract")
            self.assertTrue(definition["rootBlockDefinitionId"].startswith("transformers.composite-block:"))
            runtime = next(
                component for component in definition["components"] if component["name"] == "optional_runtime"
            )
            profile = execution_profiles_for_execution(definition["pipelineClass"], definition["workflowId"])[0]
            runtime_profile_ids = profile.optional_runtime_profile_ids_for_target()
            self.assertEqual(len(runtime_profile_ids), 1)
            runtime_profile = OPTIONAL_RUNTIME_PROFILES[runtime_profile_ids[0]]
            self.assertEqual(runtime["type"], f"{runtime_profile.id}@{runtime_profile.spec_digest}")

    def test_suggested_inputs_distinguish_publisher_examples_from_modiff_starters(self):
        flux = self.definition("FluxModularPipeline", "text2image")
        self.assertEqual(flux["suggestedInputs"]["values"]["prompt"], "A cat holding a sign that says hello world")
        self.assertEqual(flux["suggestedInputs"]["source"]["kind"], "publisher_example")
        self.assertEqual(
            flux["suggestedInputs"]["source"]["url"],
            "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/"
            "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/README.md",
        )

        anima = self.definition("AnimaModularPipeline", "text2image")
        self.assertTrue(anima["suggestedInputs"]["values"]["prompt"])
        self.assertEqual(anima["suggestedInputs"]["source"]["kind"], "modiff_task_starter")
        self.assertEqual(anima["suggestedInputs"]["source"]["label"], "MoDiff task starter")
        self.assertEqual(anima["suggestedInputs"]["source"]["url"], "")

        qwen = self.definition("QwenImageModularPipeline", "text2image")
        self.assertIn("surrounded by banners, posters, or stalls", qwen["suggestedInputs"]["values"]["prompt"])
        self.assertEqual(
            qwen["suggestedInputs"]["values"]["negative_prompt"],
            "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，"
            "画面具有AI感。构图混乱。文字模糊，扭曲。",
        )
        self.assertIn("25468b98e3276ca6700de15c6628e51b7de54a26", qwen["suggestedInputs"]["source"]["url"])

    def test_static_admission_is_attached_without_publishing_execution(self):
        admissions = [
            admission for definition in self.library["definitions"] for admission in definition["executionAdmissions"]
        ]

        self.assertEqual(len(admissions), 122)
        self.assertEqual(sum(admission["status"] == "admitted" for admission in admissions), 122)
        self.assertTrue(all(admission["executable"] is False for admission in admissions))
        self.assertTrue(
            all(definition["executionClaim"] == "discovery_only" for definition in self.library["definitions"])
        )

    def test_wan_22_5b_is_an_honest_standard_diffusers_composite(self):
        definition = self.definition("WanTI2VPipeline", "text_to_video")
        self.assertEqual(definition["id"], "diffusers.composite:WanTI2VPipeline:text_to_video")
        self.assertEqual(definition["definitionKind"], "studio_execution_composite")
        self.assertEqual(definition["integrationStatus"], "reviewed_diffusers_composite")
        self.assertTrue(definition["rootBlockDefinitionId"].startswith("diffusers.composite-block:"))
        self.assertFalse(any("ModularPipeline" in step["className"] for step in definition["steps"]))
        admission = definition["executionAdmissions"][0]
        self.assertEqual(admission["sealedBindingValues"]["pipelineClass"], "WanTI2VPipeline")
        self.assertEqual(admission["studioExecutionSpec"]["id"], "wan-22-ti2v-5b:text-to-video:v1")
        self.assertEqual(admission["artifact"]["revision"], "b8fff7315c768468a5333511427288870b2e9635")

    def test_whisper_admissions_seal_the_loader_execution_identity(self):
        definitions = [
            definition
            for definition in self.library["definitions"]
            if definition["pipelineClass"] == "HuggingFaceSpeechRecognitionModel"
        ]
        self.assertEqual(
            {definition["workflowId"] for definition in definitions},
            {"speech_to_text", "speech_translation"},
        )
        for definition in definitions:
            with self.subTest(workflow=definition["workflowId"]):
                admission = definition["executionAdmissions"][0]
                self.assertEqual(
                    admission["sealedBindingValues"]["pipelineClass"],
                    "AutoModelForSpeechSeq2Seq",
                )
                self.assertEqual(
                    admission["sealedBindingValues"]["executionProfileId"],
                    "whisper-tiny:direct",
                )
                self.assertNotIn("pipelineClass", admission["executionParameterSources"])
                self.assertNotIn("executionProfileId", admission["executionParameterSources"])

    def test_wav2vec2_ctc_cluster_omits_whisper_only_controls(self):
        definition = self.definition("HuggingFaceCTCSpeechRecognitionModel", "speech_to_text")
        inputs = {field["name"]: field for field in definition["inputs"]}
        self.assertEqual(inputs["timestamps"]["default"], "word")
        self.assertNotIn("language", inputs)
        admission = definition["executionAdmissions"][0]
        self.assertEqual(admission["sealedBindingValues"]["pipelineClass"], "AutoModelForCTC")
        self.assertEqual(
            admission["sealedBindingValues"]["executionProfileId"],
            "wav2vec2-base-960h:ctc-direct",
        )
        self.assertNotIn("speechLanguage", admission["bindingSources"])
        self.assertNotIn("transcribe", admission["bindingSources"])
        self.assertNotIn("translate", admission["bindingSources"])

    def test_exact_reviewed_graph_adapter_contracts_remain_discovery_only(self):
        adapter_definitions = [item for item in self.library["definitions"] if item["graphAdapterContracts"]]
        adapters = [adapter for definition in adapter_definitions for adapter in definition["graphAdapterContracts"]]

        self.assertEqual(len(adapter_definitions), 135)
        self.assertEqual(len(adapters), 139)
        self.assertTrue(all(definition["executionClaim"] == "discovery_only" for definition in adapter_definitions))
        flux = self.definition("FluxModularPipeline", "text2image")
        self.assertEqual(flux["graphAdapterContracts"][0]["actionSequence"], ["text_encoder", "denoise", "decoder"])
        self.assertEqual(
            flux["graphAdapterContracts"][0]["stateEdges"][0],
            {
                "producerAction": "text_encoder",
                "producerOutput": "embeddings",
                "consumerAction": "denoise",
                "consumerInput": "embeddings",
            },
        )
        flux2_text = self.definition("Flux2ModularPipeline", "text2image")
        flux2_image = self.definition("Flux2ModularPipeline", "image_conditioned")
        self.assertEqual(flux2_text["integrationStatus"], "reviewed_modiff_contract")
        self.assertEqual(flux2_image["integrationStatus"], "reviewed_modiff_contract")
        self.assertEqual(flux2_text["graphAdapterContracts"][0]["requiredInputs"], ["prompt"])
        self.assertEqual(flux2_image["graphAdapterContracts"][0]["requiredInputs"], ["image", "prompt"])
        helios_text = self.definition("HeliosModularPipeline", "text2video")
        helios_image = self.definition("HeliosPyramidModularPipeline", "image2video")
        helios_video = self.definition("HeliosPyramidDistilledModularPipeline", "video2video")
        self.assertEqual(
            helios_text["graphAdapterContracts"][0]["actionSequence"],
            ["workflow_text_encoder", "workflow_video_denoise", "workflow_video_decoder"],
        )
        self.assertEqual(helios_image["graphAdapterContracts"][0]["requiredInputs"], ["image", "prompt"])
        self.assertEqual(helios_video["graphAdapterContracts"][0]["requiredInputs"], ["prompt", "video"])
        hunyuan_text = self.definition("HunyuanVideo15ModularPipeline", "text2video")
        hunyuan_image = self.definition("HunyuanVideo15ModularPipeline", "image2video")
        self.assertEqual(hunyuan_text["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(hunyuan_image["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(
            hunyuan_text["graphAdapterContracts"][0]["actionSequence"],
            [
                "workflow_hunyuan_video15_text_encoder",
                "workflow_hunyuan_video15_denoise",
                "workflow_hunyuan_video15_decoder",
            ],
        )
        self.assertEqual(
            hunyuan_image["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["text_encoder", "vae_encoder", "image_encoder", "denoise", "decode"],
        )
        self.assertEqual(len(hunyuan_text["executionAdmissions"]), 1)
        self.assertEqual(len(hunyuan_image["executionAdmissions"]), 1)
        self.assertEqual(hunyuan_text["executionAdmissions"][0]["status"], "admitted")
        self.assertEqual(hunyuan_image["executionAdmissions"][0]["status"], "admitted")
        self.assertEqual(
            hunyuan_text["executionAdmissions"][0]["studioExecutionSpec"]["id"],
            "hunyuan-video15:modular-text-to-video:v1",
        )
        self.assertEqual(
            hunyuan_image["executionAdmissions"][0]["studioExecutionSpec"]["id"],
            "hunyuan-video15:modular-image-to-video:v1",
        )
        self.assertFalse(hunyuan_text["executionAdmissions"][0]["executable"])
        self.assertFalse(hunyuan_image["executionAdmissions"][0]["executable"])
        sd3_text = self.definition("StableDiffusion3ModularPipeline", "text2image")
        sd3_image = self.definition("StableDiffusion3ModularPipeline", "image2image")
        self.assertEqual(sd3_text["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(sd3_image["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(
            sd3_text["graphAdapterContracts"][0]["actionSequence"],
            [
                "workflow_stable_diffusion3_text_encoder",
                "workflow_stable_diffusion3_denoise",
                "workflow_stable_diffusion3_decoder",
            ],
        )
        self.assertEqual(
            sd3_image["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["text_encoder", "vae_encoder", "denoise", "decode"],
        )
        self.assertEqual(sd3_text["executionAdmissions"], [])
        self.assertEqual(sd3_image["executionAdmissions"], [])
        krea = self.definition("Krea2ModularPipeline", "text2image")
        krea_turbo = self.definition("Krea2TurboModularPipeline", "text2image")
        self.assertEqual(krea["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(krea_turbo["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(
            krea["graphAdapterContracts"][0]["actionSequence"],
            ["workflow_krea2_text_encoder", "workflow_krea2_denoise", "workflow_krea2_decoder"],
        )
        self.assertEqual(
            krea_turbo["graphAdapterContracts"][0]["actionSequence"],
            [
                "workflow_krea2_turbo_text_encoder",
                "workflow_krea2_turbo_denoise",
                "workflow_krea2_decoder",
            ],
        )
        self.assertEqual(krea["executionAdmissions"], [])
        self.assertEqual(krea_turbo["executionAdmissions"], [])
        ideogram = self.definition("Ideogram4ModularPipeline", "text2image")
        self.assertEqual(ideogram["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(
            ideogram["graphAdapterContracts"][0]["actionSequence"],
            [
                "workflow_ideogram4_prompt_upsample",
                "workflow_ideogram4_text_encoder",
                "workflow_ideogram4_denoise",
                "workflow_ideogram4_decoder",
            ],
        )
        self.assertEqual(ideogram["executionAdmissions"], [])
        cosmos_text = self.definition("Cosmos3DistilledModularPipeline", "text2video")
        cosmos_image = self.definition("Cosmos3DistilledModularPipeline", "image2video")
        cosmos_video = self.definition("Cosmos3DistilledModularPipeline", "video2video")
        self.assertEqual(cosmos_text["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(
            cosmos_text["graphAdapterContracts"][0]["actionSequence"],
            [
                "workflow_cosmos3_distilled_text_encoder",
                "workflow_cosmos3_distilled_denoise",
                "workflow_cosmos3_distilled_decoder",
            ],
        )
        self.assertEqual(
            cosmos_image["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["text_encoder", "vae_encoder", "denoise", "decode"],
        )
        self.assertEqual(cosmos_video["graphAdapterContracts"][0]["requiredInputs"], ["prompt", "video"])
        self.assertEqual(cosmos_text["executionAdmissions"], [])
        self.assertEqual(len(cosmos_image["executionAdmissions"]), 1)
        cosmos_image_admission = cosmos_image["executionAdmissions"][0]
        self.assertEqual(cosmos_image_admission["status"], "admitted")
        self.assertEqual(cosmos_image_admission["studioMode"], "image_to_video")
        self.assertEqual(
            cosmos_image_admission["artifact"],
            {
                "repo": "nvidia/Cosmos3-Super-Image2Video-4Step",
                "revision": "cd55ce81bc5cea51a09c37cd7652144e7278f049",
            },
        )
        self.assertIs(cosmos_image_admission["publication"]["insertable"], True)
        self.assertIs(cosmos_image_admission["publication"]["executable"], False)
        self.assertIs(cosmos_image_admission["publication"]["autoEligible"], False)
        self.assertEqual(cosmos_video["executionAdmissions"], [])
        cosmos_omni_image = self.definition("Cosmos3OmniModularPipeline", "text2image")
        cosmos_omni_text = self.definition("Cosmos3OmniModularPipeline", "text2video")
        cosmos_omni_sound = self.definition("Cosmos3OmniModularPipeline", "image2video_with_sound")
        cosmos_omni_action = self.definition("Cosmos3OmniModularPipeline", "action_policy")
        self.assertEqual(
            cosmos_omni_text["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["text_encoder", "denoise", "decode", "after_decode"],
        )
        self.assertEqual(
            cosmos_omni_sound["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["text_encoder", "vae_encoder", "denoise", "decode", "after_decode"],
        )
        self.assertEqual(
            cosmos_omni_action["graphAdapterContracts"][0]["requiredInputs"],
            ["action", "num_inference_steps", "prompt"],
        )
        for cosmos_definition, studio_mode in (
            (cosmos_omni_image, "text_to_image"),
            (cosmos_omni_text, "text_to_video"),
        ):
            self.assertEqual(len(cosmos_definition["executionAdmissions"]), 1)
            admission = cosmos_definition["executionAdmissions"][0]
            self.assertEqual(admission["status"], "admitted")
            self.assertEqual(admission["studioMode"], studio_mode)
            self.assertEqual(admission["artifact"]["repo"], "nvidia/Cosmos3-Nano")
            self.assertIs(admission["publication"]["insertable"], True)
            self.assertIs(admission["publication"]["executable"], False)
            self.assertEqual(
                cosmos_definition["suggestedInputs"],
                {
                    "schemaVersion": 1,
                    "values": {"prompt": "A small warehouse robot moves a blue box across a clean floor."},
                    "source": {
                        "kind": "publisher_example",
                        "label": "nvidia/Cosmos3-Nano model card example",
                        "url": (
                            "https://huggingface.co/nvidia/Cosmos3-Nano/blob/"
                            "7a312c868bcce8e40b3eb40861300a9d0ba3fde1/README.md"
                        ),
                    },
                },
            )
        self.assertEqual(len(cosmos_omni_sound["executionAdmissions"]), 1)
        cosmos_omni_sound_admission = cosmos_omni_sound["executionAdmissions"][0]
        self.assertEqual(cosmos_omni_sound_admission["status"], "admitted")
        self.assertEqual(cosmos_omni_sound_admission["studioMode"], "image_to_video_with_audio")
        self.assertEqual(cosmos_omni_sound_admission["artifact"]["repo"], "nvidia/Cosmos3-Nano")
        self.assertIs(cosmos_omni_sound_admission["publication"]["insertable"], True)
        self.assertIs(cosmos_omni_sound_admission["publication"]["executable"], False)
        self.assertIs(cosmos_omni_sound_admission["publication"]["autoEligible"], False)
        self.assertEqual(cosmos_omni_action["executionAdmissions"], [])
        minimax_text = self.definition("MiniMaxH3ModularPipeline", "t2va")
        minimax_keyframes = self.definition("MiniMaxH3ModularPipeline", "fl2va")
        minimax_references = self.definition("MiniMaxH3ModularPipeline", "ref2va")
        self.assertEqual(
            minimax_text["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["text_encoder", "denoise", "decode"],
        )
        self.assertEqual(
            minimax_keyframes["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["before_encode", "text_encoder", "vae_encoder", "denoise", "decode"],
        )
        self.assertEqual(
            minimax_references["graphAdapterContracts"][0]["requiredInputs"],
            ["num_frames", "num_inference_steps", "prompt", "references"],
        )
        for minimax_definition, studio_mode in (
            (minimax_text, "text_to_video_with_audio"),
            (minimax_keyframes, "first_last_frame_to_video_with_audio"),
            (minimax_references, "reference_to_video_with_audio"),
        ):
            self.assertEqual(len(minimax_definition["executionAdmissions"]), 1)
            admission = minimax_definition["executionAdmissions"][0]
            self.assertEqual(admission["status"], "admitted")
            self.assertEqual(admission["studioMode"], studio_mode)
            self.assertEqual(admission["artifact"]["repo"], "MiniMaxAI/MiniMax-H3")
            self.assertIs(admission["publication"]["insertable"], True)
            self.assertIs(admission["publication"]["executable"], False)
            self.assertIs(admission["publication"]["autoEligible"], False)
        ltx25_text = self.definition("LTX25ModularPipeline", "text2video")
        ltx25_image = self.definition("LTX25ModularPipeline", "image2video")
        ltx25_condition = self.definition("LTX25ModularPipeline", "condition")
        ltx25_context = self.definition("LTX25ModularPipeline", "in_context")
        self.assertEqual(
            ltx25_text["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["text_encoder", "duration", "denoise", "decode"],
        )
        self.assertEqual(
            ltx25_image["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["text_encoder", "duration", "vae_encoder", "denoise", "decode"],
        )
        self.assertEqual(
            ltx25_condition["graphAdapterContracts"][0]["requiredInputs"],
            ["conditions", "prompt"],
        )
        self.assertEqual(
            ltx25_context["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["text_encoder", "condition_encoder", "reference_encoder", "denoise", "decode"],
        )
        self.assertEqual(ltx25_text["executionAdmissions"], [])
        self.assertEqual(ltx25_image["executionAdmissions"], [])
        self.assertEqual(ltx25_condition["executionAdmissions"], [])
        self.assertEqual(ltx25_context["executionAdmissions"], [])
        wan_animate = self.definition("WanAnimate2ModularPipeline", "default")
        wan_animate_distilled = self.definition("WanAnimate2DistilledModularPipeline", "default")
        expected_wan_actions = [
            "workflow_wan_animate_text_encoder",
            "workflow_wan_animate_image_encoder",
            "workflow_wan_animate_video_encoder",
            "workflow_wan_animate_vae_encoder",
            "workflow_wan_animate_denoise",
            "workflow_wan_animate_decoder",
        ]
        self.assertEqual(wan_animate["graphAdapterContracts"][0]["actionSequence"], expected_wan_actions)
        self.assertEqual(wan_animate_distilled["graphAdapterContracts"][0]["actionSequence"], expected_wan_actions)
        self.assertEqual(
            wan_animate["graphAdapterContracts"][0]["requiredInputs"],
            ["driving_video", "image", "prompt"],
        )
        kontext = self.definition("FluxKontextModularPipeline", "text2image")
        self.assertEqual(kontext["graphAdapterContracts"][0]["adapterId"], "text_to_image")
        self.assertEqual(kontext["graphAdapterContracts"][0]["requiredInputs"], ["prompt"])
        klein_base = self.definition("Flux2KleinBaseModularPipeline", "text2image")
        self.assertEqual(klein_base["integrationStatus"], "reviewed_modiff_contract")
        self.assertEqual(klein_base["graphAdapterContracts"][0]["adapterId"], "text_to_image")
        self.assertEqual(klein_base["graphAdapterContracts"][0]["requiredInputs"], ["prompt"])
        equivalent = self.definition("LTXModularPipeline", "image2video")
        self.assertEqual(equivalent["integrationStatus"], "equivalent_standard_route")
        self.assertEqual(equivalent["graphAdapterContracts"][0]["actionSequence"], ["full_pipeline"])
        self.assertEqual(equivalent["graphAdapterContracts"][0]["stateEdges"], [])
        self.assertEqual(equivalent["graphAdapterContracts"][0]["requiredInputs"], ["image", "prompt"])
        anima_text = self.definition("AnimaModularPipeline", "text2image")
        anima_image = self.definition("AnimaModularPipeline", "img2img")
        self.assertEqual(
            anima_text["graphAdapterContracts"][0]["actionSequence"],
            ["workflow_text_encoder", "workflow_image_denoise", "workflow_image_decoder"],
        )
        self.assertEqual(
            anima_image["graphAdapterContracts"][0]["actionSequence"],
            [
                "workflow_text_encoder",
                "workflow_image_encoder",
                "workflow_image_denoise",
                "workflow_image_decoder",
            ],
        )
        self.assertEqual(anima_text["graphAdapterContracts"][0]["requiredInputs"], ["prompt"])
        self.assertEqual(anima_image["graphAdapterContracts"][0]["requiredInputs"], ["image", "prompt"])
        ltx2_text = self.definition("LTX2ModularPipeline", "text2video")
        ltx2_image = self.definition("LTX2ModularPipeline", "image2video")
        ltx2_condition = self.definition("LTX2ModularPipeline", "condition")
        ltx2_in_context = self.definition("LTX2ModularPipeline", "in_context")
        self.assertEqual(ltx2_text["integrationStatus"], "equivalent_standard_route")
        self.assertEqual(ltx2_image["integrationStatus"], "equivalent_standard_route")
        self.assertEqual(ltx2_condition["integrationStatus"], "equivalent_standard_route")
        self.assertEqual(ltx2_in_context["integrationStatus"], "equivalent_standard_route")
        self.assertEqual(ltx2_text["graphAdapterContracts"][0]["requiredInputs"], ["prompt"])
        self.assertEqual(ltx2_image["graphAdapterContracts"][0]["requiredInputs"], ["image", "prompt"])
        self.assertEqual(ltx2_condition["graphAdapterContracts"][0]["requiredInputs"], ["conditions", "prompt"])
        self.assertEqual(
            ltx2_in_context["graphAdapterContracts"][0]["requiredInputs"],
            ["num_frames", "prompt", "reference_conditions"],
        )
        minimax = self.definition("MiniMaxMusic3ModularPipeline", "default")
        self.assertEqual(minimax["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(
            minimax["graphAdapterContracts"][0]["actionSequence"],
            ["semantic_generator", "workflow_denoise", "workflow_audio_decoder"],
        )
        self.assertEqual(
            minimax["graphAdapterContracts"][0]["upstreamBlockSequence"],
            ["semantic_generator", "denoise", "decode"],
        )

    def test_statically_closed_contract_only_workflows_have_exact_reusable_block_roles(self):
        adapters = self.library["blockRoleAdapters"]
        adapters_by_block = {adapter["blockDefinitionId"]: adapter for adapter in adapters}

        self.assertEqual(len(adapters), 321)
        self.assertEqual(
            {adapter["role"] for adapter in adapters},
            {
                "workflow",
                "text_encoder",
                "vae_encoder",
                "image_encoder",
                "video_encoder",
                "semantic_generator",
                "prompt_transform",
                "duration",
                "condition_encoder",
                "reference_encoder",
                "before_encode",
                "denoise",
                "decoder",
                "after_decode",
            },
        )
        self.assertTrue(all(adapter["executionClaim"] == "structural_adapter_only" for adapter in adapters))

        promoted = {
            (definition["pipelineClass"], definition["workflowId"]): definition
            for definition in self.library["definitions"]
            if definition["provider"] == "diffusers"
            and (definition["pipelineClass"], definition["workflowId"])
            in PINNED_STATICALLY_CLOSED_BLOCK_ROLE_WORKFLOWS
        }
        self.assertEqual(set(promoted), PINNED_STATICALLY_CLOSED_BLOCK_ROLE_WORKFLOWS)
        for key, definition in promoted.items():
            with self.subTest(workflow=key):
                referenced = {
                    definition["rootBlockDefinitionId"],
                    *(placement["blockDefinitionId"] for placement in definition["blockPlacements"]),
                }
                self.assertTrue(referenced.issubset(adapters_by_block))
                self.assertEqual(adapters_by_block[definition["rootBlockDefinitionId"]]["role"], "workflow")

        equivalent_definitions = [
            definition
            for definition in self.library["definitions"]
            if definition["provider"] == "diffusers" and definition["integrationStatus"] == "equivalent_standard_route"
        ]
        self.assertEqual(len(equivalent_definitions), 9)
        for definition in equivalent_definitions:
            referenced = {
                definition["rootBlockDefinitionId"],
                *(placement["blockDefinitionId"] for placement in definition["blockPlacements"]),
            }
            self.assertTrue(referenced.issubset(adapters_by_block))

    def test_custom_container_workflows_have_roles_plus_exact_loop_state_adapters(self):
        adapted_block_ids = {adapter["blockDefinitionId"] for adapter in self.library["blockRoleAdapters"]}
        container_adapters = self.library["containerStateAdapters"]
        container_block_ids = {adapter["blockDefinitionId"] for adapter in container_adapters}
        definitions = {
            (definition["pipelineClass"], definition["workflowId"]): definition
            for definition in self.library["definitions"]
            if definition["provider"] == "diffusers"
        }
        self.assertEqual(len(PINNED_CUSTOM_CONTAINER_BLOCK_ROLE_WORKFLOWS), 12)
        self.assertEqual(len(container_adapters), 9)
        self.assertEqual(
            {adapter["className"] for adapter in container_adapters},
            {truth.class_name for truth in PINNED_CONTAINER_STATE_ADAPTER_TRUTH},
        )
        self.assertTrue(
            all(adapter["executionClaim"] == "container_state_adapter_only" for adapter in container_adapters)
        )
        for key in PINNED_CUSTOM_CONTAINER_BLOCK_ROLE_WORKFLOWS:
            with self.subTest(workflow=key):
                definition = definitions[key]
                referenced = {
                    definition["rootBlockDefinitionId"],
                    *(placement["blockDefinitionId"] for placement in definition["blockPlacements"]),
                }
                self.assertTrue(referenced.issubset(adapted_block_ids))
                loop_ids = {
                    placement["blockDefinitionId"]
                    for placement in definition["blockPlacements"]
                    if placement["blockDefinitionId"] in container_block_ids
                }
                self.assertEqual(len(loop_ids), 1)

    def test_exact_path_segments_distinguish_dotted_names_from_nesting(self):
        flux = self.definition("FluxModularPipeline", "text2image")
        placements = {placement["legacyPath"]: placement for placement in flux["blockPlacements"]}

        self.assertEqual(placements["denoise.input"]["path"], ["denoise.input"])
        self.assertEqual(placements["denoise.denoise"]["path"], ["denoise.denoise"])
        self.assertEqual(
            placements["denoise.denoise.denoiser"]["path"],
            ["denoise.denoise", "denoiser"],
        )

    def test_exact_block_contracts_include_variadic_inputs_components_and_configs(self):
        definitions = self.library["blockDefinitions"]

        self.assertTrue(any(definition["variadicInputs"] for definition in definitions))
        self.assertTrue(any(definition["components"] for definition in definitions))
        self.assertTrue(any(definition["configs"] for definition in definitions))
        self.assertTrue(
            any(
                field["kwargsType"] is not None
                for definition in definitions
                for field in (*definition["inputs"], *definition["outputs"])
            )
        )

    def test_block_definitions_are_reused_across_model_families(self):
        owners = {}
        for definition in self.library["definitions"]:
            owners.setdefault(definition["rootBlockDefinitionId"], set()).add(definition["pipelineClass"])
            for placement in definition["blockPlacements"]:
                owners.setdefault(placement["blockDefinitionId"], set()).add(definition["pipelineClass"])

        shared = [pipeline_classes for pipeline_classes in owners.values() if len(pipeline_classes) > 1]
        self.assertGreaterEqual(len(shared), 124)
        self.assertTrue(
            any(
                "WanImage2VideoModularPipeline" in pipeline_classes
                and "Wan22Image2VideoModularPipeline" in pipeline_classes
                for pipeline_classes in shared
            )
        )

    def test_text_to_image_families_share_one_generic_task_contract(self):
        flux = self.definition("FluxModularPipeline", "text2image")
        qwen = self.definition("QwenImageModularPipeline", "text2image")

        self.assertEqual(flux["taskContractId"], "diffusers.task.text_to_image.v1")
        self.assertEqual(qwen["taskContractId"], flux["taskContractId"])
        self.assertNotEqual(flux["pipelineClass"], qwen["pipelineClass"])
        self.assertNotEqual(flux["steps"], qwen["steps"])

        task_contract = next(
            contract
            for contract in self.library["taskContracts"]
            if contract["id"] == "diffusers.task.text_to_image.v1"
        )
        self.assertIn(flux["id"], task_contract["definitionIds"])
        self.assertIn(qwen["id"], task_contract["definitionIds"])

    def test_image_to_video_families_share_one_generic_task_contract(self):
        wan = self.definition("WanImage2VideoModularPipeline", "image2video")
        ltx = self.definition("LTXModularPipeline", "image2video")
        cosmos = self.definition("Cosmos3OmniModularPipeline", "image2video")

        self.assertEqual(wan["taskContractId"], "diffusers.task.image_to_video.v1")
        self.assertEqual(
            {wan["taskContractId"], ltx["taskContractId"], cosmos["taskContractId"]},
            {"diffusers.task.image_to_video.v1"},
        )
        self.assertNotEqual(wan["steps"], ltx["steps"])
        self.assertNotEqual(wan["components"], cosmos["components"])

    def test_integration_status_does_not_turn_discovery_into_execution_claim(self):
        flux = self.definition("FluxModularPipeline", "text2image")
        anima = self.definition("AnimaModularPipeline", "text2image")
        ltx = self.definition("LTXModularPipeline", "image2video")

        self.assertEqual(flux["integrationStatus"], "reviewed_modiff_contract")
        self.assertEqual(anima["integrationStatus"], "reviewed_modular_workflow_route")
        self.assertEqual(ltx["integrationStatus"], "equivalent_standard_route")
        self.assertEqual({flux["executionClaim"], anima["executionClaim"], ltx["executionClaim"]}, {"discovery_only"})

    def test_content_hash_rejects_definition_mutation(self):
        changed = copy.deepcopy(self.library)
        changed_definition = next(
            definition for definition in changed["definitions"] if definition["definitionKind"] == "modular_pipeline_workflow"
        )
        changed_definition["label"] = "Changed"
        with self.assertRaisesRegex(HuggingFaceNodeLibraryError, "content hash"):
            validate_huggingface_node_library(changed)

    def test_block_contract_tampering_is_rejected(self):
        changed = copy.deepcopy(self.library)
        changed["blockDefinitions"][0]["description"] = "Changed"
        with self.assertRaisesRegex(HuggingFaceNodeLibraryError, "block library"):
            validate_huggingface_node_library(changed)

        changed = copy.deepcopy(self.library)
        changed["blockRoleAdapters"][0]["role"] = "invented"
        with self.assertRaisesRegex(HuggingFaceNodeLibraryError, "block-role adapters"):
            validate_huggingface_node_library(changed)

        changed = copy.deepcopy(self.library)
        changed["containerStateAdapters"][0]["publishedState"] = ["invented"]
        with self.assertRaisesRegex(HuggingFaceNodeLibraryError, "container-state adapters"):
            validate_huggingface_node_library(changed)

        changed = copy.deepcopy(self.library)
        changed_definition = next(
            definition
            for definition in changed["definitions"]
            if definition["pipelineClass"] == "FluxModularPipeline" and definition["workflowId"] == "text2image"
        )
        changed_definition["graphAdapterContracts"] = []
        body = {key: value for key, value in changed_definition.items() if key != "contentHash"}
        encoded = json.dumps(body, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        changed_definition["contentHash"] = f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"
        with self.assertRaisesRegex(HuggingFaceNodeLibraryError, "graph adapter"):
            validate_huggingface_node_library(changed)

        changed = copy.deepcopy(self.library)
        admitted_definition = next(
            definition
            for definition in changed["definitions"]
            if definition["definitionKind"] == "modular_pipeline_workflow" and definition["executionAdmissions"]
        )
        admitted_definition["executionAdmissions"] = []
        body = {key: value for key, value in admitted_definition.items() if key != "contentHash"}
        encoded = json.dumps(body, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        admitted_definition["contentHash"] = f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"
        with self.assertRaisesRegex(HuggingFaceNodeLibraryError, "execution admissions"):
            validate_huggingface_node_library(changed)

        changed = copy.deepcopy(self.library)
        changed_definition = next(
            definition for definition in changed["definitions"] if definition["definitionKind"] == "modular_pipeline_workflow"
        )
        changed_definition["blockPlacements"][0]["path"] = ["invented"]
        body = {key: value for key, value in changed_definition.items() if key != "contentHash"}
        encoded = json.dumps(body, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        changed_definition["contentHash"] = f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"
        with self.assertRaisesRegex(HuggingFaceNodeLibraryError, "block library"):
            validate_huggingface_node_library(changed)

    def test_revision_and_response_size_are_bounded(self):
        changed = copy.deepcopy(self.library)
        changed["diffusersRevision"] = "main"
        with self.assertRaisesRegex(HuggingFaceNodeLibraryError, "immutable Diffusers revision"):
            validate_huggingface_node_library(changed)

        oversized = copy.deepcopy(self.library)
        oversized["definitions"][0]["description"] = "x" * (8 * 1024 * 1024)
        with self.assertRaisesRegex(HuggingFaceNodeLibraryError, "8 MiB"):
            validate_huggingface_node_library(oversized)

    def test_import_and_build_do_not_import_model_libraries(self):
        command = (
            "import sys; "
            "from modiff.huggingface_node_library import build_huggingface_node_library; "
            "build_huggingface_node_library(); "
            "assert 'diffusers' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'torch' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
