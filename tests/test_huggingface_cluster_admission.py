import copy
import subprocess
import sys
import unittest

from modiff.huggingface_cluster_admission import audit_reviewed_cluster_execution_candidates
from modiff.huggingface_node_library import build_huggingface_node_library
from modiff.auto_resource import auto_resource_pair_is_declared
from modiff.diffusers_profiles import optional_runtime_profile_ids_for_execution
from modiff.studio_execution_specs import LTX2_DIFFUSERS_FILES, STUDIO_EXECUTION_SPEC_DEFINITIONS


class HuggingFaceClusterAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = build_huggingface_node_library()

    def results(self, **kwargs):
        return {result["id"]: result for result in audit_reviewed_cluster_execution_candidates(self.library, **kwargs)}

    def result(self, pipeline_class, workflow_id, adapter_id, **kwargs):
        result_id = f"diffusers.cluster-admission:{pipeline_class}:{workflow_id}:mode:{adapter_id}"
        return self.results(**kwargs)[result_id]

    def state_flow_result(self, pipeline_class, workflow_id, adapter_id, **kwargs):
        result_id = f"diffusers.cluster-admission:{pipeline_class}:{workflow_id}:state_flow:{adapter_id}"
        return self.results(**kwargs)[result_id]

    def workflow_result(self, pipeline_class, workflow_id, adapter_id, **kwargs):
        result_id = f"diffusers.cluster-admission:{pipeline_class}:{workflow_id}:workflow:{adapter_id}"
        return self.results(**kwargs)[result_id]

    def test_minimax_music_uses_exact_official_block_and_state_route(self):
        result = self.workflow_result(
            "MiniMaxMusic3ModularPipeline",
            "default",
            "official_top_level_blocks",
        )
        self.assertEqual(result["status"], "admitted")
        self.assertEqual(
            result["artifact"],
            {
                "repo": "MiniMaxAI/MiniMax-Music3",
                "revision": "fbdf52fbaaca799592917417eb05f1899f1255ec",
            },
        )
        self.assertEqual(
            result["instanceInputBindings"],
            [
                {"bindingSource": "audioDuration", "input": "audio_duration"},
                {"bindingSource": "lyrics", "input": "lyrics"},
                {"bindingSource": "prompt", "input": "prompt"},
                {"bindingSource": "steps", "input": "num_inference_steps"},
            ],
        )
        self.assertEqual(
            result["sealedBindingValues"],
            {
                "artifact": "MiniMaxAI/MiniMax-Music3",
                "defaultRevision": "fbdf52fbaaca799592917417eb05f1899f1255ec",
                "defaultWorkflow": "default",
                "false": False,
                "pipelineClass": "MiniMaxMusic3ModularPipeline",
                "sampleRate44100": 44100,
                "semanticGeneratorBlock": "semantic_generator",
                "workflowDecodeBlock": "decode",
                "workflowDenoiseBlock": "denoise",
            },
        )
        self.assertIs(result["publication"]["autoEligible"], False)
        # Legacy v1 output approvals are historical after the promotion-v2
        # identity boundary. Static admission must not silently retain live
        # publication authority for the newly sealed BlockDefinitionV2.
        self.assertIs(result["publication"]["liveProof"], False)
        self.assertEqual(
            {reason["code"] for reason in result["publication"]["reasons"]},
            {"runtime_resource_admission_required", "live_output_review_pending"},
        )

    def test_flux_sdxl_qwen_candidates_and_wan_flf_pass_static_graph_contract_admission(self):
        results = self.results()
        admitted = [result for result in results.values() if result["status"] == "admitted"]

        self.assertEqual(len(results), 81)
        self.assertEqual(len(admitted), 81)
        for result in admitted:
            specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
            if any(
                role == "models" and node_key == "modules.ModularDiffusers.ModelsLoader"
                for role, node_key, _x, _y in specification["roles"]
            ):
                self.assertEqual(
                    result["sealedBindingValues"].get("defaultRevision"),
                    result["artifact"]["revision"],
                    result["id"],
                )
        self.assertEqual(
            {result["definitionId"] for result in admitted},
            {
                "diffusers.modular:Flux2ModularPipeline:text2image",
                "diffusers.modular:Flux2ModularPipeline:image_conditioned",
                "diffusers.modular:FluxModularPipeline:text2image",
                "diffusers.modular:FluxModularPipeline:image2image",
                "diffusers.modular:FluxKontextModularPipeline:text2image",
                "diffusers.modular:FluxKontextModularPipeline:image_conditioned",
                "diffusers.modular:Flux2KleinModularPipeline:text2image",
                "diffusers.modular:Flux2KleinModularPipeline:image_conditioned",
                "diffusers.modular:Flux2KleinBaseModularPipeline:text2image",
                "diffusers.modular:Flux2KleinBaseModularPipeline:image_conditioned",
                "diffusers.modular:StableDiffusionXLModularPipeline:text2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:image2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:inpainting",
                "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_text2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_image2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_inpainting",
                "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_union_text2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_union_image2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_union_inpainting",
                "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_text2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_image2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_inpainting",
                "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_controlnet_text2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_controlnet_image2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_controlnet_inpainting",
                "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_controlnet_union_text2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_controlnet_union_image2image",
                "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_controlnet_union_inpainting",
                "diffusers.modular:QwenImageEditModularPipeline:image_conditioned",
                "diffusers.modular:QwenImageEditModularPipeline:image_conditioned_inpainting",
                "diffusers.modular:QwenImageEditPlusModularPipeline:default",
                "diffusers.modular:QwenImageLayeredModularPipeline:default",
                "diffusers.modular:QwenImageModularPipeline:controlnet_text2image",
                "diffusers.modular:QwenImageModularPipeline:text2image",
                "diffusers.modular:QwenImageModularPipeline:image2image",
                "diffusers.modular:QwenImageModularPipeline:inpainting",
                "diffusers.modular:QwenImageModularPipeline:controlnet_image2image",
                "diffusers.modular:QwenImageModularPipeline:controlnet_inpainting",
                "diffusers.modular:WanModularPipeline:default",
                "diffusers.modular:WanImage2VideoModularPipeline:image2video",
                "diffusers.modular:WanImage2VideoModularPipeline:flf2v",
                "diffusers.modular:ZImageModularPipeline:text2image",
                "diffusers.modular:ZImageModularPipeline:image2image",
                "diffusers.modular:ErnieImageModularPipeline:text2image",
                "diffusers.modular:LTXModularPipeline:text2video",
                "diffusers.modular:LTXModularPipeline:image2video",
                "diffusers.modular:Wan22ModularPipeline:default",
                "diffusers.modular:Wan22Image2VideoModularPipeline:default",
                "diffusers.modular:LTX2ModularPipeline:text2video",
                "diffusers.modular:LTX2ModularPipeline:image2video",
                "diffusers.modular:LTX2ModularPipeline:condition",
                "diffusers.modular:LTX2ModularPipeline:in_context",
                "diffusers.modular:MiniMaxMusic3ModularPipeline:default",
                "diffusers.modular:AnimaModularPipeline:text2image",
                "diffusers.modular:AnimaModularPipeline:img2img",
                "diffusers.modular:HeliosModularPipeline:text2video",
                "diffusers.modular:HeliosModularPipeline:image2video",
                "diffusers.modular:HeliosModularPipeline:video2video",
                "diffusers.modular:HeliosPyramidModularPipeline:text2video",
                "diffusers.modular:HeliosPyramidModularPipeline:image2video",
                "diffusers.modular:HeliosPyramidModularPipeline:video2video",
                "diffusers.modular:HeliosPyramidDistilledModularPipeline:text2video",
                "diffusers.modular:HeliosPyramidDistilledModularPipeline:image2video",
                "diffusers.modular:HeliosPyramidDistilledModularPipeline:video2video",
                "diffusers.modular:HunyuanVideo15ModularPipeline:text2video",
                "diffusers.modular:HunyuanVideo15ModularPipeline:image2video",
                "diffusers.modular:Cosmos3DistilledModularPipeline:text2image",
                "diffusers.modular:Cosmos3DistilledModularPipeline:image2video",
                "diffusers.modular:MiniMaxH3ModularPipeline:t2va",
                "diffusers.modular:MiniMaxH3ModularPipeline:fl2va",
                "diffusers.modular:MiniMaxH3ModularPipeline:ref2va",
                "diffusers.modular:WanAnimate2ModularPipeline:default",
                "diffusers.modular:WanAnimate2DistilledModularPipeline:default",
                "diffusers.modular:Cosmos3OmniModularPipeline:text2image",
                "diffusers.modular:Cosmos3OmniModularPipeline:text2video",
                "diffusers.modular:Cosmos3OmniModularPipeline:image2video",
                "diffusers.modular:Cosmos3OmniModularPipeline:video2video",
                "diffusers.modular:Cosmos3OmniModularPipeline:text2video_with_sound",
                "diffusers.modular:Cosmos3OmniModularPipeline:image2video_with_sound",
                "diffusers.modular:Cosmos3OmniModularPipeline:video2video_with_sound",
            },
        )
        promoted_admission_ids = set()
        for result in admitted:
            self.assertEqual(result["schemaVersion"], 4)
            self.assertEqual(result["claim"], "static_graph_contract_compatible")
            self.assertIs(result["executable"], False)
            self.assertEqual(result["publication"]["readiness"], "graph_qualified")
            self.assertIs(result["publication"]["insertable"], True)
            self.assertIs(result["publication"]["executable"], False)
            self.assertIs(result["publication"]["autoEligible"], False)
            self.assertIs(result["publication"]["liveProof"], result["id"] in promoted_admission_ids)
            self.assertIn(
                "runtime_resource_admission_required",
                {reason["code"] for reason in result["publication"]["reasons"]},
            )
            self.assertRegex(result["artifact"]["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(result["studioExecutionSpec"]["contentHash"].startswith("studio-spec-v1-"))
            self.assertTrue(result["bindingSources"])
            self.assertNotIn("artifact", result["executionParameterSources"])
            self.assertNotIn("pipelineClass", result["executionParameterSources"])
            self.assertNotIn("mode", result["executionParameterSources"])

        edit = self.result("QwenImageEditModularPipeline", "image_conditioned", "edit_image")
        self.assertIn(
            {"bindingSource": "referenceImages", "input": "image"},
            edit["instanceInputBindings"],
        )
        self.assertIn("seed", edit["executionParameterSources"])
        self.assertIn("guidanceScale", edit["executionParameterSources"])
        layered = self.result("QwenImageLayeredModularPipeline", "default", "layer_decomposition")
        self.assertEqual(layered["sealedBindingValues"]["addAlpha"], "add alpha")
        self.assertNotIn("addAlpha", layered["executionParameterSources"])
        self.assertEqual(
            edit["dynamicFieldActions"],
            [
                {
                    "role": "models",
                    "field": "model_type",
                    "event": "onChange",
                    "valueSource": "pipelineClass",
                },
                {
                    "role": "prompt",
                    "field": "text_encoders",
                    "event": "onSignal",
                    "valueSource": "pipelineClass",
                },
                {
                    "role": "imageEncode",
                    "field": "vae",
                    "event": "onSignal",
                    "valueSource": "pipelineClass",
                },
                {
                    "role": "denoise",
                    "field": "unet",
                    "event": "onSignal",
                    "valueSource": "pipelineClass",
                },
                {
                    "role": "decode",
                    "field": "vae",
                    "event": "onSignal",
                    "valueSource": "pipelineClass",
                },
            ],
        )

        control = self.result("QwenImageModularPipeline", "controlnet_text2image", "control_image")
        self.assertIn(
            {
                "role": "controlnet",
                "field": "controlnet_bundle",
                "event": "onSignal",
                "valueSource": "pipelineClass",
            },
            control["dynamicFieldActions"],
        )
        self.assertEqual(
            control["modelDependencies"],
            [
                {
                    "id": "qwen-controlnet-union",
                    "kind": "controlnet",
                    "repo": "InstantX/Qwen-Image-ControlNet-Union",
                    "revision": "b13036f066d6dee7c20513e263d3d673055e9de8",
                }
            ],
        )
        self.assertEqual(control["sealedBindingValues"]["kind"], "controlnet")
        self.assertEqual(control["sealedBindingValues"]["repo"], "InstantX/Qwen-Image-ControlNet-Union")
        self.assertEqual(
            control["sealedBindingValues"]["revision"],
            "b13036f066d6dee7c20513e263d3d673055e9de8",
        )
        self.assertIn(
            {
                "role": "controlnetModel",
                "field": "model_type",
                "event": "onChange",
                "valueSource": "kind",
            },
            control["dynamicFieldActions"],
        )

    def test_sdxl_text_to_image_uses_the_exact_modular_graph_and_artifact(self):
        result = self.result("StableDiffusionXLModularPipeline", "text2image", "text_to_image")

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["definitionId"], "diffusers.modular:StableDiffusionXLModularPipeline:text2image")
        self.assertEqual(result["studioExecutionSpec"]["id"], "sdxl-base:modular-text-to-image:v1")
        self.assertEqual(result["studioExecutionSpec"]["executionProfileId"], "sdxl-base:modular")
        self.assertEqual(result["artifact"]["repo"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(result["artifact"]["revision"], "462165984030d82259a11f4367a4eed129e94a7b")
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertEqual(
            [(role, node_key) for role, node_key, _x, _y in specification["roles"]],
            [
                ("models", "modules.ModularDiffusers.ModelsLoader"),
                ("prompt", "modules.ModularDiffusers.EncodePrompt"),
                ("denoise", "modules.ModularDiffusers.Denoise"),
                ("decode", "modules.ModularDiffusers.DecodeLatents"),
                ("preview", "modules.Image.Preview"),
            ],
        )
        self.assertIn(("models", "vae_out", "denoise", "vae"), specification["edges"])
        self.assertIn(("prompt", "embeddings", "denoise", "embeddings"), specification["edges"])
        self.assertIn(("denoise", "latents", "decode", "latents"), specification["edges"])
        self.assertIn(("denoise", "route_state_out", "decode", "route_state_in"), specification["edges"])
        self.assertEqual(
            result["instanceInputBindings"],
            [
                {"bindingSource": "height", "input": "height"},
                {"bindingSource": "negativePrompt", "input": "negative_prompt"},
                {"bindingSource": "prompt", "input": "prompt"},
                {"bindingSource": "steps", "input": "num_inference_steps"},
                {"bindingSource": "width", "input": "width"},
            ],
        )
        self.assertIs(result["executable"], False)

    def test_hunyuan_video_15_preserves_distinct_t2v_and_vae_siglip_i2v_routes(self):
        text = self.workflow_result(
            "HunyuanVideo15ModularPipeline",
            "text2video",
            "official_top_level_blocks",
        )
        image = self.workflow_result(
            "HunyuanVideo15ModularPipeline",
            "image2video",
            "official_top_level_blocks",
        )
        self.assertEqual(text["status"], "admitted")
        self.assertEqual(image["status"], "admitted")
        self.assertEqual(text["reasons"], [])
        self.assertEqual(image["reasons"], [])
        self.assertEqual(
            text["artifact"],
            {
                "repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
                "revision": "286be7ce72277246578a3e3cc2487e95ddae5bcf",
            },
        )
        self.assertEqual(
            image["artifact"],
            {
                "repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
                "revision": "854c04a4c8a53d990b418c7478f0802c0fc8c726",
            },
        )
        text_spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[text["studioExecutionSpec"]["id"]]
        image_spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[image["studioExecutionSpec"]["id"]]
        self.assertEqual(
            [role for role, _node_key, _x, _y in text_spec["roles"]],
            ["models", "prompt", "denoise", "decode", "videoExport"],
        )
        self.assertEqual(
            [role for role, _node_key, _x, _y in image_spec["roles"]],
            [
                "models",
                "loadImage",
                "prompt",
                "imageEncode",
                "imageEmbeddings",
                "denoise",
                "decode",
                "videoExport",
            ],
        )
        self.assertIn(("imageEncode", "state_out", "imageEmbeddings", "state_in"), image_spec["edges"])
        self.assertIn(("imageEmbeddings", "state_out", "denoise", "state_in"), image_spec["edges"])
        self.assertIn(("denoise", "width", "width"), text_spec["bindings"])
        self.assertIn(("denoise", "height", "height"), text_spec["bindings"])
        self.assertNotIn(("denoise", "width", "width"), image_spec["bindings"])
        self.assertNotIn(("denoise", "height", "height"), image_spec["bindings"])
        self.assertIn(("imageEncode", "width", "optionalWidth"), image_spec["bindings"])
        self.assertIn(("imageEncode", "height", "optionalHeight"), image_spec["bindings"])
        self.assertEqual(text["sealedBindingValues"]["oneVideo"], 1)
        self.assertEqual(image["sealedBindingValues"]["workflowVaeEncoderBlock"], "vae_encoder")
        self.assertEqual(image["sealedBindingValues"]["workflowImageEmbeddingsBlock"], "image_encoder")
        self.assertIn(
            {"bindingSource": "optionalWidth", "input": "width"},
            image["instanceInputBindings"],
        )
        self.assertIn(
            {"bindingSource": "optionalHeight", "input": "height"},
            image["instanceInputBindings"],
        )
        for result in (text, image):
            self.assertIs(result["executable"], False)
            self.assertIs(result["publication"]["executable"], False)
            self.assertIs(result["publication"]["autoEligible"], False)
            self.assertIs(result["publication"]["liveProof"], False)

    def test_cosmos3_nano_non_action_routes_seal_guardrails_and_the_official_state_graph(self):
        expected = {
            "text2image": ("text_to_image", "preview", "oneFrame", 1, None, False),
            "text2video": ("text_to_video", "videoExport", "numFrames", None, None, False),
            "image2video": (
                "image_to_video",
                "videoExport",
                "numFrames",
                None,
                "loadImage",
                False,
            ),
            "video2video": (
                "video_to_video",
                "videoExport",
                "numFrames",
                None,
                "loadVideo",
                False,
            ),
            "text2video_with_sound": (
                "text_to_video_with_audio",
                "videoExport",
                "numFrames",
                None,
                None,
                True,
            ),
            "image2video_with_sound": (
                "image_to_video_with_audio",
                "videoExport",
                "numFrames",
                None,
                "loadImage",
                True,
            ),
            "video2video_with_sound": (
                "video_to_video_with_audio",
                "videoExport",
                "numFrames",
                None,
                "loadVideo",
                True,
            ),
        }
        for workflow_id, (mode, sink_role, frame_source, frame_value, load_role, with_audio) in expected.items():
            with self.subTest(workflow_id=workflow_id):
                result = self.workflow_result(
                    "Cosmos3OmniModularPipeline",
                    workflow_id,
                    "official_top_level_blocks",
                )
                self.assertEqual(result["status"], "admitted")
                self.assertEqual(result["reasons"], [])
                self.assertEqual(result["studioMode"], mode)
                self.assertEqual(
                    result["artifact"],
                    {
                        "repo": "nvidia/Cosmos3-Nano",
                        "revision": "7a312c868bcce8e40b3eb40861300a9d0ba3fde1",
                    },
                )
                self.assertEqual(
                    result["modelDependencies"],
                    [
                        {
                            "id": "cosmos3-mandatory-safety-guardrail",
                            "kind": "safety_checker",
                            "repo": "nvidia/Cosmos-Guardrail1",
                            "revision": "d6d4bfa899a71454a700907664f3e88f503950cf",
                        }
                    ],
                )
                self.assertEqual(result["sealedBindingValues"]["workflowAfterDecodeBlock"], "after_decode")
                self.assertIs(result["publication"]["insertable"], True)
                self.assertIs(result["publication"]["executable"], False)
                self.assertIs(result["publication"]["autoEligible"], False)
                specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
                roles = ["models"]
                if load_role:
                    roles.append(load_role)
                roles.append("prompt")
                if load_role:
                    roles.append("imageEncode")
                roles.extend(["denoise", "decode", "afterDecode", sink_role])
                self.assertEqual(
                    [role for role, _node_key, _x, _y in specification["roles"]],
                    roles,
                )
                if load_role:
                    self.assertNotIn(("prompt", "state_out", "denoise", "state_in"), specification["edges"])
                    self.assertIn(
                        ("prompt", "state_out", "imageEncode", "state_in"),
                        specification["edges"],
                    )
                    self.assertIn(
                        ("imageEncode", "state_out", "denoise", "state_in"),
                        specification["edges"],
                    )
                else:
                    self.assertIn(("prompt", "state_out", "denoise", "state_in"), specification["edges"])
                self.assertIn(("denoise", "state_out", "decode", "state_in"), specification["edges"])
                self.assertIn(("decode", "state_out", "afterDecode", "state_in"), specification["edges"])
                if with_audio:
                    self.assertIn(("decode", "audio", "videoExport", "audio"), specification["edges"])
                self.assertIn(("prompt", "num_frames", frame_source), specification["bindings"])
                self.assertIn(
                    {"bindingSource": "steps", "input": "num_inference_steps"},
                    result["instanceInputBindings"],
                )
                if frame_value is None:
                    self.assertIn(
                        {"bindingSource": "numFrames", "input": "num_frames"},
                        result["instanceInputBindings"],
                    )
                else:
                    self.assertEqual(result["sealedBindingValues"][frame_source], frame_value)
                    self.assertNotIn(frame_source, result["executionParameterSources"])

    def test_equivalent_standard_routes_keep_modular_structure_but_seal_full_pipeline_execution(self):
        expected = {
            ("Flux2ModularPipeline", "text2image"): (
                "text_to_image",
                "Flux2Pipeline",
                "modules.DiffusersImage.LoadPipeline",
                "black-forest-labs/FLUX.2-dev",
            ),
            ("Flux2ModularPipeline", "image_conditioned"): (
                "multi_image_reference_edit",
                "Flux2Pipeline",
                "modules.DiffusersImage.LoadPipeline",
                "black-forest-labs/FLUX.2-dev",
            ),
            ("ErnieImageModularPipeline", "text2image"): (
                "text_to_image",
                "ErnieImagePipeline",
                "modules.DiffusersImage.LoadPipeline",
                "baidu/ERNIE-Image-Turbo",
            ),
            ("LTXModularPipeline", "text2video"): (
                "text_to_video",
                "LTXConditionPipeline",
                "modules.DiffusersVideo.LoadPipeline",
                "Lightricks/LTX-Video-0.9.8-13B-distilled",
            ),
            ("LTXModularPipeline", "image2video"): (
                "image_to_video",
                "LTXConditionPipeline",
                "modules.DiffusersVideo.LoadPipeline",
                "Lightricks/LTX-Video-0.9.8-13B-distilled",
            ),
            ("Wan22ModularPipeline", "default"): (
                "text_to_video",
                "Wan22Pipeline",
                "modules.DiffusersVideo.LoadPipeline",
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            ),
            ("Wan22Image2VideoModularPipeline", "default"): (
                "image_to_video",
                "WanImageToVideoPipeline",
                "modules.DiffusersVideo.LoadPipeline",
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            ),
            ("LTX2ModularPipeline", "text2video"): (
                "text_to_video",
                "LTX2ConditionPipeline",
                "modules.DiffusersVideo.LoadPipeline",
                "Lightricks/LTX-2",
            ),
            ("LTX2ModularPipeline", "image2video"): (
                "image_to_video",
                "LTX2ConditionPipeline",
                "modules.DiffusersVideo.LoadPipeline",
                "Lightricks/LTX-2",
            ),
            ("LTX2ModularPipeline", "condition"): (
                "reference_to_video",
                "LTX2ConditionPipeline",
                "modules.DiffusersVideo.LoadPipeline",
                "Lightricks/LTX-2",
            ),
            ("LTX2ModularPipeline", "in_context"): (
                "in_context_to_video",
                "LTX2InContextPipeline",
                "modules.DiffusersVideo.LoadPipeline",
                "Lightricks/LTX-2",
            ),
        }

        for (pipeline_class, workflow_id), (mode, execution_class, loader, repo) in expected.items():
            with self.subTest(pipeline_class=pipeline_class, workflow_id=workflow_id):
                result = self.result(pipeline_class, workflow_id, "equivalent_standard_route")
                definition = next(
                    item
                    for item in self.library["definitions"]
                    if item["pipelineClass"] == pipeline_class and item["workflowId"] == workflow_id
                )
                adapter = definition["graphAdapterContracts"][0]
                specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
                profile = specification["profile"]

                self.assertEqual(result["status"], "admitted")
                self.assertEqual(result["reasons"], [])
                self.assertEqual(definition["integrationStatus"], "equivalent_standard_route")
                self.assertEqual(adapter["actionSequence"], ["full_pipeline"])
                self.assertEqual(adapter["stateEdges"], [])
                self.assertEqual(result["studioMode"], mode)
                self.assertEqual(result["artifact"]["repo"], repo)
                self.assertRegex(result["artifact"]["revision"], r"^[0-9a-f]{40}$")
                self.assertEqual(result["sealedBindingValues"]["pipelineClass"], execution_class)
                self.assertEqual(result["sealedBindingValues"]["executionProfileId"], profile["id"])
                self.assertEqual(result["sealedBindingValues"]["mode"], mode)
                self.assertIn(
                    ("execution_profile_id", "executionProfileId"),
                    {(field, source) for _role, field, source in specification["bindings"]},
                )
                self.assertIn(loader, {node_key for _role, node_key, _x, _y in specification["roles"]})
                self.assertEqual(profile["model_type"], pipeline_class)
                self.assertEqual(profile["pipeline_class"], execution_class)
                self.assertIn(profile["execution_path"], {"direct-diffusers-image", "direct-diffusers-video"})
                if pipeline_class == "LTX2ModularPipeline":
                    self.assertEqual(specification["capability"]["downloadFiles"], LTX2_DIFFUSERS_FILES)
                    self.assertEqual(
                        len(
                            [
                                path
                                for path in LTX2_DIFFUSERS_FILES
                                if path.startswith("text_encoder/model-")
                            ]
                        ),
                        11,
                    )
                    self.assertEqual(
                        len(
                            [
                                path
                                for path in LTX2_DIFFUSERS_FILES
                                if path.startswith("transformer/diffusion_pytorch_model-")
                            ]
                        ),
                        8,
                    )
                    self.assertFalse(any(path.startswith("ltx-2-19b-") for path in LTX2_DIFFUSERS_FILES))
                    if workflow_id == "condition":
                        self.assertIn(
                            {"bindingSource": "conditionImages", "input": "conditions"},
                            result["instanceInputBindings"],
                        )
                        self.assertIn(
                            {"bindingSource": "prompt", "input": "prompt"},
                            result["instanceInputBindings"],
                        )
                        self.assertNotIn("referenceImages", result["bindingSources"])
                self.assertIs(result["publication"]["executable"], False)
                self.assertIs(result["publication"]["autoEligible"], False)

    def test_flux_workflows_share_generic_roles_but_keep_exact_artifact_and_state(self):
        text = self.result("FluxModularPipeline", "text2image", "text_to_image")
        image = self.result("FluxModularPipeline", "image2image", "image_to_image")

        self.assertEqual(text["status"], "admitted")
        self.assertEqual(image["status"], "admitted")
        self.assertEqual(text["artifact"], image["artifact"])
        self.assertEqual(text["artifact"]["repo"], "black-forest-labs/FLUX.1-dev")
        self.assertEqual(text["artifact"]["revision"], "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
        self.assertEqual(text["studioExecutionSpec"]["id"], "flux-dev:modular-text-to-image:v1")
        self.assertEqual(image["studioExecutionSpec"]["id"], "flux-dev:modular-image-to-image:v1")
        self.assertEqual(text["studioExecutionSpec"]["executionProfileId"], "flux-dev:modular")
        self.assertEqual(image["studioExecutionSpec"]["executionProfileId"], "flux-dev:modular")
        image_spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[image["studioExecutionSpec"]["id"]]
        text_spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[text["studioExecutionSpec"]["id"]]
        for spec in (text_spec, image_spec):
            self.assertIn(("denoise", "out_width", "decode", "width"), spec["edges"])
            self.assertIn(("denoise", "out_height", "decode", "height"), spec["edges"])
        self.assertIn(("imageEncode", "width", "optionalWidth"), image_spec["bindings"])
        self.assertIn(("imageEncode", "height", "optionalHeight"), image_spec["bindings"])
        self.assertIn(("imageEncode", "image_latents", "denoise", "image_latents"), image_spec["edges"])
        self.assertNotIn(("models", "vae_out", "denoise", "vae"), image_spec["edges"])
        self.assertFalse(any("route_state" in handle for edge in image_spec["edges"] for handle in edge))
        self.assertIn({"bindingSource": "referenceImages", "input": "image"}, image["instanceInputBindings"])
        self.assertIn({"bindingSource": "strength", "input": "strength"}, image["instanceInputBindings"])
        self.assertIs(text["publication"]["autoEligible"], False)
        self.assertIs(image["publication"]["autoEligible"], False)

    def test_flux_kontext_and_flux2_klein_use_exact_auto_block_dispatch(self):
        expected = {
            "FluxKontextModularPipeline": (
                "black-forest-labs/FLUX.1-Kontext-dev",
                "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d",
                "flux-kontext:modular",
            ),
            "Flux2KleinModularPipeline": (
                "black-forest-labs/FLUX.2-klein-4B",
                "e7b7dc27f91deacad38e78976d1f2b499d76a294",
                "flux2-klein:modular",
            ),
        }
        for pipeline_class, (repo, revision, profile_id) in expected.items():
            with self.subTest(pipeline_class=pipeline_class):
                text = self.result(pipeline_class, "text2image", "text_to_image")
                image = self.result(pipeline_class, "image_conditioned", "edit_image")
                self.assertEqual(text["status"], "admitted")
                self.assertEqual(image["status"], "admitted")
                self.assertEqual(text["artifact"], {"repo": repo, "revision": revision})
                self.assertEqual(image["artifact"], text["artifact"])
                self.assertEqual(text["studioExecutionSpec"]["executionProfileId"], profile_id)
                self.assertEqual(image["studioExecutionSpec"]["executionProfileId"], profile_id)

                text_spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[text["studioExecutionSpec"]["id"]]
                image_spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[image["studioExecutionSpec"]["id"]]
                self.assertNotIn(("models", "vae_out", "denoise", "vae"), text_spec["edges"])
                self.assertNotIn(("models", "vae_out", "denoise", "vae"), image_spec["edges"])
                self.assertIn(("imageEncode", "image_latents", "denoise", "image_latents"), image_spec["edges"])
                self.assertFalse(any("route_state" in handle for edge in image_spec["edges"] for handle in edge))
                self.assertIn(("denoise", "width", "width"), text_spec["bindings"])
                self.assertIn(("denoise", "height", "height"), text_spec["bindings"])
                if pipeline_class == "FluxKontextModularPipeline":
                    for spec in (text_spec, image_spec):
                        self.assertIn(("denoise", "out_width", "decode", "width"), spec["edges"])
                        self.assertIn(("denoise", "out_height", "decode", "height"), spec["edges"])
                self.assertNotIn(("denoise", "strength", "strength"), image_spec["bindings"])
                self.assertIn(
                    {"bindingSource": "referenceImages", "input": "image"},
                    image["instanceInputBindings"],
                )
                self.assertIs(text["publication"]["executable"], False)
                self.assertIs(image["publication"]["autoEligible"], False)

    def test_qwen_image_conditioned_workflows_preserve_exact_route_state_and_dependencies(self):
        text = self.result("QwenImageModularPipeline", "text2image", "text_to_image")
        image = self.state_flow_result("QwenImageModularPipeline", "image2image", "image2image")
        inpaint = self.state_flow_result("QwenImageModularPipeline", "inpainting", "inpainting")
        control_text = self.result("QwenImageModularPipeline", "controlnet_text2image", "control_image")
        control = self.state_flow_result(
            "QwenImageModularPipeline", "controlnet_image2image", "controlnet_image2image"
        )
        control_inpaint = self.state_flow_result(
            "QwenImageModularPipeline", "controlnet_inpainting", "controlnet_inpainting"
        )

        for result in (text, image, inpaint, control_text, control, control_inpaint):
            self.assertEqual(result["status"], "admitted")
            self.assertEqual(result["artifact"]["repo"], "Qwen/Qwen-Image-2512")

        # This phase deliberately exposes one same-task family selector on the
        # Qwen text-to-image Cluster. Image-to-image, inpaint, and ControlNet
        # admissions remain separate tasks and must not silently inherit that
        # model control until their own compatible variant sets are reviewed.
        self.assertIn("modelVariant", text["executionParameterSources"])
        for result in (image, inpaint, control_text, control, control_inpaint):
            self.assertNotIn("modelVariant", result["executionParameterSources"])
            self.assertIs(result["publication"]["executable"], False)
            self.assertIs(result["publication"]["autoEligible"], False)
        text_spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[text["studioExecutionSpec"]["id"]]
        self.assertEqual(text["studioMode"], "modular_text_to_image")
        self.assertIn(("denoise", "route_state_out", "decode", "route_state_in"), text_spec["edges"])
        self.assertNotIn(("models", "vae_out", "denoise", "vae"), text_spec["edges"])
        self.assertEqual(
            text["artifact"]["revision"],
            "25468b98e3276ca6700de15c6628e51b7de54a26",
        )
        self.assertIn({"bindingSource": "paddingMaskCrop", "input": "padding_mask_crop"}, inpaint["instanceInputBindings"])
        for result in (text, image, inpaint, control_text, control, control_inpaint):
            self.assertIn(
                {"bindingSource": "maxSequenceLength", "input": "max_sequence_length"},
                result["instanceInputBindings"],
            )
        self.assertEqual(control_text["modelDependencies"], control["modelDependencies"])
        self.assertEqual(control["modelDependencies"], control_inpaint["modelDependencies"])
        self.assertEqual(control["modelDependencies"][0]["id"], "qwen-controlnet-union")
        for result in (control_text, control, control_inpaint):
            self.assertIn(
                {"bindingSource": "controlGuidanceStart", "input": "control_guidance_start"},
                result["instanceInputBindings"],
            )
            self.assertIn(
                {"bindingSource": "controlGuidanceEnd", "input": "control_guidance_end"},
                result["instanceInputBindings"],
            )
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[control["studioExecutionSpec"]["id"]]
        self.assertIn(("imageEncode", "route_state_out", "controlnet", "route_state_in"), specification["edges"])
        self.assertIn(("controlnet", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertNotIn(("imageEncode", "route_state_out", "denoise", "route_state_in"), specification["edges"])

    def test_sdxl_image_to_image_uses_the_exact_upstream_state_flow(self):
        result = self.result("StableDiffusionXLModularPipeline", "image2image", "image_to_image")

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["definitionId"], "diffusers.modular:StableDiffusionXLModularPipeline:image2image")
        self.assertEqual(result["studioExecutionSpec"]["id"], "sdxl-base:modular-image-to-image:v1")
        self.assertEqual(result["studioExecutionSpec"]["executionProfileId"], "sdxl-base:modular")
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertEqual(
            [(role, node_key) for role, node_key, _x, _y in specification["roles"]],
            [
                ("models", "modules.ModularDiffusers.ModelsLoader"),
                ("prompt", "modules.ModularDiffusers.EncodePrompt"),
                ("loadImage", "modules.Image.Load"),
                ("imageEncode", "modules.ModularDiffusers.ImageEncode"),
                ("denoise", "modules.ModularDiffusers.Denoise"),
                ("decode", "modules.ModularDiffusers.DecodeLatents"),
                ("preview", "modules.Image.Preview"),
            ],
        )
        self.assertIn(("models", "vae_out", "denoise", "vae"), specification["edges"])
        self.assertIn(("imageEncode", "image_latents", "denoise", "image_latents"), specification["edges"])
        self.assertIn(("imageEncode", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertNotIn(("loadImage", "image", "prompt", "image"), specification["edges"])
        self.assertIn(("denoise", "strength", "strength"), specification["bindings"])
        self.assertIn(
            {"bindingSource": "referenceImages", "input": "image"},
            result["instanceInputBindings"],
        )
        self.assertIn(
            {"bindingSource": "strength", "input": "strength"},
            result["instanceInputBindings"],
        )
        self.assertIn(
            {"bindingSource": "dtype", "input": "dtype"},
            result["instanceInputBindings"],
        )
        self.assertNotIn("strength", result["executionParameterSources"])
        self.assertNotIn("dtype", result["executionParameterSources"])
        self.assertIs(result["executable"], False)

    def test_sdxl_inpainting_uses_the_exact_mask_and_latent_state_flow(self):
        result = self.result("StableDiffusionXLModularPipeline", "inpainting", "inpaint")

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["definitionId"], "diffusers.modular:StableDiffusionXLModularPipeline:inpainting")
        self.assertEqual(result["studioExecutionSpec"]["id"], "sdxl-base:modular-inpainting:v1")
        self.assertEqual(result["studioExecutionSpec"]["executionProfileId"], "sdxl-base:modular")
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertEqual(
            [(role, node_key) for role, node_key, _x, _y in specification["roles"]],
            [
                ("models", "modules.ModularDiffusers.ModelsLoader"),
                ("prompt", "modules.ModularDiffusers.EncodePrompt"),
                ("loadImage", "modules.Image.Load"),
                ("loadMask", "modules.Image.Load"),
                ("imageEncode", "modules.ModularDiffusers.ImageEncode"),
                ("denoise", "modules.ModularDiffusers.Denoise"),
                ("decode", "modules.ModularDiffusers.DecodeLatents"),
                ("preview", "modules.Image.Preview"),
            ],
        )
        self.assertIn(("loadMask", "image", "imageEncode", "mask_image"), specification["edges"])
        self.assertIn(("imageEncode", "mask", "denoise", "mask"), specification["edges"])
        self.assertIn(
            ("imageEncode", "masked_image_latents", "denoise", "masked_image_latents"),
            specification["edges"],
        )
        self.assertIn(("imageEncode", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertIn(("loadMask", "file", "maskImage"), specification["bindings"])
        self.assertIn({"bindingSource": "maskImage", "input": "mask_image"}, result["instanceInputBindings"])
        self.assertIn({"bindingSource": "referenceImages", "input": "image"}, result["instanceInputBindings"])
        self.assertIn({"bindingSource": "strength", "input": "strength"}, result["instanceInputBindings"])
        self.assertNotIn("maskImage", result["executionParameterSources"])
        self.assertNotIn("strength", result["executionParameterSources"])
        self.assertIs(result["executable"], False)

    def test_sdxl_controlnet_text_to_image_uses_the_exact_ordinary_component_flow(self):
        result = self.result("StableDiffusionXLModularPipeline", "controlnet_text2image", "control_image")

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(
            result["definitionId"],
            "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_text2image",
        )
        self.assertEqual(
            result["studioExecutionSpec"]["id"],
            "sdxl-base:modular-controlnet-text-to-image:v1",
        )
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertEqual(
            [(role, node_key) for role, node_key, _x, _y in specification["roles"]],
            [
                ("models", "modules.ModularDiffusers.ModelsLoader"),
                ("prompt", "modules.ModularDiffusers.EncodePrompt"),
                ("loadImage", "modules.Image.Load"),
                ("controlnetModel", "modules.ModularDiffusers.AutoModelLoader"),
                ("controlnet", "modules.ModularDiffusers.Controlnet"),
                ("denoise", "modules.ModularDiffusers.Denoise"),
                ("decode", "modules.ModularDiffusers.DecodeLatents"),
                ("preview", "modules.Image.Preview"),
            ],
        )
        self.assertIn(("controlnetModel", "model", "controlnet", "controlnet"), specification["edges"])
        self.assertIn(("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"), specification["edges"])
        self.assertNotIn(("models", "vae_out", "controlnet", "vae"), specification["edges"])
        self.assertNotIn(("controlnet", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertEqual(result["sealedBindingValues"]["ordinary"], "ordinary")
        self.assertEqual(result["sealedBindingValues"]["fp16"], "fp16")
        self.assertEqual(
            result["modelDependencies"],
            [
                {
                    "id": "sdxl-controlnet-canny",
                    "kind": "controlnet",
                    "repo": "diffusers/controlnet-canny-sdxl-1.0",
                    "revision": "eb115a19a10d14909256db740ed109532ab1483c",
                }
            ],
        )
        self.assertIn({"bindingSource": "controlImage", "input": "control_image"}, result["instanceInputBindings"])
        self.assertIn(
            {"bindingSource": "conditioningScale", "input": "controlnet_conditioning_scale"},
            result["instanceInputBindings"],
        )
        self.assertIs(result["executable"], False)

    def test_sdxl_controlnet_image_to_image_joins_both_exact_state_flows(self):
        result = self.result(
            "StableDiffusionXLModularPipeline",
            "controlnet_image2image",
            "control_edit_image",
        )

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(
            result["definitionId"],
            "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_image2image",
        )
        self.assertEqual(
            result["studioExecutionSpec"]["id"],
            "sdxl-base:modular-controlnet-image-to-image:v1",
        )
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertEqual(
            [(role, node_key) for role, node_key, _x, _y in specification["roles"]],
            [
                ("models", "modules.ModularDiffusers.ModelsLoader"),
                ("prompt", "modules.ModularDiffusers.EncodePrompt"),
                ("loadImage", "modules.Image.Load"),
                ("imageEncode", "modules.ModularDiffusers.ImageEncode"),
                ("loadControlImage", "modules.Image.Load"),
                ("controlnetModel", "modules.ModularDiffusers.AutoModelLoader"),
                ("controlnet", "modules.ModularDiffusers.Controlnet"),
                ("denoise", "modules.ModularDiffusers.Denoise"),
                ("decode", "modules.ModularDiffusers.DecodeLatents"),
                ("preview", "modules.Image.Preview"),
            ],
        )
        self.assertIn(("imageEncode", "image_latents", "denoise", "image_latents"), specification["edges"])
        self.assertIn(("imageEncode", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertIn(("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"), specification["edges"])
        self.assertNotIn(("models", "vae_out", "controlnet", "vae"), specification["edges"])
        self.assertNotIn(("controlnet", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertIn(("loadImage", "file", "referenceImages"), specification["bindings"])
        self.assertIn(("loadControlImage", "file", "controlImage"), specification["bindings"])
        self.assertIn(("denoise", "strength", "strength"), specification["bindings"])
        self.assertEqual(result["sealedBindingValues"]["ordinary"], "ordinary")
        self.assertEqual(result["sealedBindingValues"]["fp16"], "fp16")
        self.assertIn({"bindingSource": "referenceImages", "input": "image"}, result["instanceInputBindings"])
        self.assertIn({"bindingSource": "controlImage", "input": "control_image"}, result["instanceInputBindings"])
        self.assertIn({"bindingSource": "strength", "input": "strength"}, result["instanceInputBindings"])
        self.assertNotIn("strength", result["executionParameterSources"])
        self.assertIs(result["executable"], False)

    def test_sdxl_controlnet_inpainting_joins_mask_latent_and_control_state(self):
        result = self.result(
            "StableDiffusionXLModularPipeline",
            "controlnet_inpainting",
            "control_inpaint",
        )

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(
            result["definitionId"],
            "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_inpainting",
        )
        self.assertEqual(
            result["studioExecutionSpec"]["id"],
            "sdxl-base:modular-controlnet-inpainting:v1",
        )
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertEqual(len(specification["roles"]), 11)
        self.assertIn(("imageEncode", "mask", "denoise", "mask"), specification["edges"])
        self.assertIn(
            ("imageEncode", "masked_image_latents", "denoise", "masked_image_latents"),
            specification["edges"],
        )
        self.assertIn(("imageEncode", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertIn(("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"), specification["edges"])
        self.assertNotIn(("models", "vae_out", "controlnet", "vae"), specification["edges"])
        self.assertNotIn(("controlnet", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertIn(("loadImage", "file", "referenceImages"), specification["bindings"])
        self.assertIn(("loadMask", "file", "maskImage"), specification["bindings"])
        self.assertIn(("loadControlImage", "file", "controlImage"), specification["bindings"])
        self.assertEqual(result["sealedBindingValues"]["ordinary"], "ordinary")
        self.assertEqual(result["sealedBindingValues"]["fp16"], "fp16")
        self.assertIn({"bindingSource": "referenceImages", "input": "image"}, result["instanceInputBindings"])
        self.assertIn({"bindingSource": "maskImage", "input": "mask_image"}, result["instanceInputBindings"])
        self.assertIn({"bindingSource": "controlImage", "input": "control_image"}, result["instanceInputBindings"])
        self.assertIs(result["executable"], False)

    def test_wan_flf_uses_the_exact_flf2v_state_flow_and_artifact(self):
        result_id = "diffusers.cluster-admission:WanImage2VideoModularPipeline:flf2v:state_flow:flf2v"
        result = self.results()[result_id]

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertIn(":flf2v:state_flow:flf2v", result["id"])
        self.assertEqual(result["artifact"]["repo"], "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers")
        self.assertIn(
            {"bindingSource": "maxSequenceLength", "input": "max_sequence_length"},
            result["instanceInputBindings"],
        )
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertIn(("prompt", "max_sequence_length", "maxSequenceLength"), specification["bindings"])
        self.assertIs(result["executable"], False)

    def test_qwen_edit_inpainting_preserves_mask_overlay_state_through_the_route(self):
        result_id = (
            "diffusers.cluster-admission:QwenImageEditModularPipeline:"
            "image_conditioned_inpainting:state_flow:image_conditioned_inpainting"
        )
        result = self.results()[result_id]

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["studioMode"], "modular_inpainting")
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertIn(("loadImage", "file", "referenceImages"), specification["bindings"])
        self.assertIn(("loadMask", "file", "maskImage"), specification["bindings"])
        self.assertIn(("imageEncode", "padding_mask_crop", "paddingMaskCrop"), specification["bindings"])
        self.assertIn(("loadImage", "image", "prompt", "image"), specification["edges"])
        self.assertNotIn(("denoise", "height", "height"), specification["bindings"])
        self.assertNotIn(("denoise", "width", "width"), specification["bindings"])
        self.assertIn(("imageEncode", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertIn(("denoise", "route_state_out", "decode", "route_state_in"), specification["edges"])
        self.assertNotIn(("models", "vae_out", "denoise", "vae"), specification["edges"])
        self.assertIs(result["executable"], False)

    def test_wan_text_to_video_uses_the_fixed_text_denoise_decode_sequence(self):
        result = self.result("WanModularPipeline", "default", "text_to_video")

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["artifact"]["repo"], "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(result["artifact"]["revision"], "0fad780a534b6463e45facd96134c9f345acfa5b")
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertEqual(
            [role for role, _node_key, _x, _y in specification["roles"]],
            ["models", "prompt", "denoise", "decode", "videoExport"],
        )
        self.assertNotIn(("models", "vae_out", "denoise", "vae"), specification["edges"])
        self.assertFalse(any("route_state" in handle for edge in specification["edges"] for handle in edge[1::2]))
        self.assertIn({"bindingSource": "prompt", "input": "prompt"}, result["instanceInputBindings"])
        self.assertIn({"bindingSource": "numFrames", "input": "num_frames"}, result["instanceInputBindings"])
        self.assertIn({"bindingSource": "width", "input": "width"}, result["instanceInputBindings"])
        self.assertIn({"bindingSource": "height", "input": "height"}, result["instanceInputBindings"])
        self.assertIn(
            {"bindingSource": "maxSequenceLength", "input": "max_sequence_length"},
            result["instanceInputBindings"],
        )
        self.assertIn(("denoise", "width", "width"), specification["bindings"])
        self.assertIn(("denoise", "height", "height"), specification["bindings"])
        self.assertIn(("prompt", "max_sequence_length", "maxSequenceLength"), specification["bindings"])
        self.assertIs(result["executable"], False)

    def test_wan_single_image_to_video_is_the_flf_graph_without_last_image(self):
        result = self.result("WanImage2VideoModularPipeline", "image2video", "image_to_video")

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["studioMode"], "single_image_to_video")
        self.assertEqual(result["artifact"]["repo"], "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
        self.assertEqual(result["artifact"]["revision"], "b184e23a8a16b20f108f727c902e769e873ffc73")
        specification = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        self.assertNotIn("loadLastImage", {role for role, _node_key, _x, _y in specification["roles"]})
        self.assertFalse(any(edge[0] == "loadLastImage" or edge[2] == "loadLastImage" for edge in specification["edges"]))
        self.assertIn(
            ("imageEmbeddings", "route_state_out", "imageEncode", "route_state_in"),
            specification["edges"],
        )
        self.assertIn(("imageEncode", "route_state_out", "denoise", "route_state_in"), specification["edges"])
        self.assertIn(("denoise", "route_state_out", "decode", "route_state_in"), specification["edges"])
        self.assertIn({"bindingSource": "referenceImages", "input": "image"}, result["instanceInputBindings"])
        self.assertIn(
            {"bindingSource": "maxSequenceLength", "input": "max_sequence_length"},
            result["instanceInputBindings"],
        )
        self.assertNotIn({"bindingSource": "lastImage", "input": "last_image"}, result["instanceInputBindings"])
        self.assertIs(result["executable"], False)

    def test_z_image_uses_exact_modular_text_and_image_workflows(self):
        result = self.result("ZImageModularPipeline", "text2image", "text_to_image")
        image = self.result("ZImageModularPipeline", "image2image", "image_to_image")

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(image["status"], "admitted")
        self.assertEqual(result["studioMode"], "modular_text_to_image")
        self.assertEqual(image["studioMode"], "modular_image_to_image")
        self.assertEqual(result["artifact"]["repo"], "Tongyi-MAI/Z-Image-Turbo")
        self.assertEqual(result["artifact"]["revision"], image["artifact"]["revision"])
        text_spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]
        image_spec = STUDIO_EXECUTION_SPEC_DEFINITIONS[image["studioExecutionSpec"]["id"]]
        self.assertNotIn(("models", "vae_out", "denoise", "vae"), text_spec["edges"])
        self.assertFalse(any("route_state" in handle for edge in text_spec["edges"] for handle in edge[1::2]))
        self.assertIn(("imageEncode", "image_latents", "denoise", "image_latents"), image_spec["edges"])
        self.assertIn(("imageEncode", "width", "optionalWidth"), image_spec["bindings"])
        self.assertIn(("imageEncode", "height", "optionalHeight"), image_spec["bindings"])
        self.assertFalse(any("route_state" in handle for edge in image_spec["edges"] for handle in edge[1::2]))
        self.assertNotIn(("prompt", "negative_prompt", "negativePrompt"), text_spec["bindings"])
        self.assertIs(result["executable"], False)
        self.assertIs(image["executable"], False)

    def test_resource_and_optional_runtime_gates_remain_separate_from_static_admission(self):
        admitted = [result for result in self.results().values() if result["status"] == "admitted"]
        pairs = {
            (STUDIO_EXECUTION_SPEC_DEFINITIONS[result["studioExecutionSpec"]["id"]]["modelType"], result["studioMode"])
            for result in admitted
        }

        self.assertEqual(len(pairs), 81)
        expert_only_model_types = {
            "StableDiffusionXLModularPipeline",
            "Flux2ModularPipeline",
            "WanImage2VideoModularPipeline",
            "ErnieImageModularPipeline",
            "LTXModularPipeline",
            "Wan22ModularPipeline",
            "Wan22Image2VideoModularPipeline",
            "LTX2ModularPipeline",
            "MiniMaxMusic3ModularPipeline",
            "AnimaModularPipeline",
            "HeliosModularPipeline",
            "HeliosPyramidModularPipeline",
            "HeliosPyramidDistilledModularPipeline",
            "HunyuanVideo15ModularPipeline",
            "Cosmos3DistilledModularPipeline",
            "MiniMaxH3ModularPipeline",
            "WanAnimate2ModularPipeline",
            "WanAnimate2DistilledModularPipeline",
            "Cosmos3OmniModularPipeline",
        }
        self.assertTrue(
            all(
                auto_resource_pair_is_declared(model_type, mode)
                for model_type, mode in pairs
                if model_type not in expert_only_model_types
            )
        )
        self.assertTrue(auto_resource_pair_is_declared("StableDiffusionXLModularPipeline", "text_to_image"))
        self.assertTrue(auto_resource_pair_is_declared("StableDiffusionXLModularPipeline", "edit_image"))
        self.assertTrue(auto_resource_pair_is_declared("StableDiffusionXLModularPipeline", "inpaint"))
        self.assertTrue(auto_resource_pair_is_declared("StableDiffusionXLModularPipeline", "control_image"))
        self.assertTrue(auto_resource_pair_is_declared("StableDiffusionXLModularPipeline", "control_edit_image"))
        self.assertTrue(auto_resource_pair_is_declared("StableDiffusionXLModularPipeline", "control_inpaint"))
        self.assertTrue(
            all(
                auto_resource_pair_is_declared(model_type, mode)
                for model_type, mode in pairs
                if model_type == "StableDiffusionXLModularPipeline"
            )
        )
        self.assertFalse(auto_resource_pair_is_declared("WanImage2VideoModularPipeline", "image_to_video"))
        self.assertTrue(
            auto_resource_pair_is_declared("WanImage2VideoModularPipeline", "single_image_to_video")
        )
        self.assertTrue(
            all(
                not auto_resource_pair_is_declared(model_type, mode)
                for model_type, mode in pairs
                if model_type in {
                    "ErnieImageModularPipeline",
                    "Flux2ModularPipeline",
                    "LTXModularPipeline",
                    "Wan22ModularPipeline",
                    "Wan22Image2VideoModularPipeline",
                    "LTX2ModularPipeline",
                }
            )
        )
        self.assertTrue(
            all(optional_runtime_profile_ids_for_execution(model_type, mode) for model_type, mode in pairs)
        )
        self.assertTrue(all(result["executable"] is False for result in admitted))

    def test_sdxl_union_ip_adapter_admission_seals_artifacts_and_persists_tuning(self):
        result_id = (
            "diffusers.cluster-admission:StableDiffusionXLModularPipeline:"
            "ip_adapter_controlnet_union_inpainting:state_flow:ip_adapter_controlnet_union_inpainting"
        )
        result = self.results()[result_id]

        self.assertEqual(result["status"], "admitted")
        self.assertEqual(result["reasons"], [])
        self.assertEqual(len(result["modelDependencies"]), 2)
        self.assertEqual(result["sealedBindingValues"]["controlnetRouteVariant"], "union")
        self.assertEqual(result["sealedBindingValues"]["controlnetLoadClass"], "ControlNetUnionModel")
        self.assertEqual(result["sealedBindingValues"]["controlnetWeightVariant"], "")
        self.assertEqual(
            result["sealedBindingValues"]["ipAdapterWeightName"],
            "sdxl_models/ip-adapter_sdxl.safetensors",
        )
        self.assertIn({"bindingSource": "controlMode", "input": "control_mode"}, result["instanceInputBindings"])
        self.assertIn(
            {"bindingSource": "ipAdapterImage", "input": "ip_adapter_image"},
            result["instanceInputBindings"],
        )
        self.assertIn("ipAdapterScale", result["executionParameterSources"])
        self.assertNotIn("controlnetLoadClass", result["executionParameterSources"])

    def test_live_profile_evidence_does_not_skip_runtime_resource_admission(self):
        definitions = copy.deepcopy(STUDIO_EXECUTION_SPEC_DEFINITIONS)
        definitions["qwen-image-edit:edit-image:v1"]["profile"]["live_proof"] = True
        result = self.result(
            "QwenImageEditModularPipeline",
            "image_conditioned",
            "edit_image",
            studio_definitions=definitions,
        )

        self.assertEqual(result["status"], "admitted")
        self.assertIs(result["publication"]["liveProof"], True)
        self.assertIs(result["publication"]["insertable"], True)
        self.assertIs(result["publication"]["executable"], False)
        self.assertEqual(
            {reason["code"] for reason in result["publication"]["reasons"]},
            {"runtime_resource_admission_required"},
        )

    def test_missing_state_edge_fails_closed(self):
        definitions = copy.deepcopy(STUDIO_EXECUTION_SPEC_DEFINITIONS)
        definition = definitions["qwen-image-edit:edit-image:v1"]
        definition["edges"] = tuple(
            edge
            for edge in definition["edges"]
            if edge != ("imageEncode", "image_latents", "denoise", "image_latents")
        )
        result = self.result(
            "QwenImageEditModularPipeline",
            "image_conditioned",
            "edit_image",
            studio_definitions=definitions,
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("state_edge_mismatch", {reason["code"] for reason in result["reasons"]})

    def test_wrong_action_node_fails_closed(self):
        definitions = copy.deepcopy(STUDIO_EXECUTION_SPEC_DEFINITIONS)
        definition = definitions["qwen-image-edit:edit-image:v1"]
        definition["roles"] = tuple(
            (role, "modules.ModularDiffusers.Controlnet", x, y) if role == "prompt" else item
            for item in definition["roles"]
            for role, _node_key, x, y in (item,)
        )
        result = self.result(
            "QwenImageEditModularPipeline",
            "image_conditioned",
            "edit_image",
            studio_definitions=definitions,
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("action_role_mismatch", {reason["code"] for reason in result["reasons"]})

    def test_audit_does_not_import_model_libraries(self):
        command = (
            "import sys; "
            "from modiff.huggingface_cluster_admission import audit_reviewed_cluster_execution_candidates; "
            "audit_reviewed_cluster_execution_candidates(); "
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
