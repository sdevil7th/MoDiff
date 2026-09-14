from copy import deepcopy
import unittest
from unittest.mock import patch

import modules as module_registry

from modiff.auto_resource import AUTO_MODEL_REQUIREMENTS
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.server import STUDIO_MODEL_CAPABILITIES, WebServer
from modiff.studio_execution_specs import (
    STUDIO_EXECUTION_SPEC_DEFINITIONS,
    _execution_spec_role_params,
    assert_studio_execution_graph,
    studio_execution_spec_for_pair,
    studio_model_dependencies_for_pair,
    studio_model_requirements_for_pair,
    validate_studio_execution_specs,
)


def executable_graph_for_spec(spec):
    nodes = {}
    node_ids = {}
    for index, (role, node_key, _x, _y) in enumerate(spec["roles"]):
        module, action = node_key.rsplit(".", 1)
        node_id = f"node-{index}"
        node_ids[role] = node_id
        definition = module_registry.MODULE_MAP[module][action]
        params = {
            key: {**deepcopy(value), "value": deepcopy(value.get("default"))}
            for key, value in _execution_spec_role_params(spec, node_key, definition).items()
        }
        nodes[node_id] = {"module": module, "action": action, "params": params}
    for source_role, source_handle, target_role, target_handle in spec["edges"]:
        nodes[node_ids[target_role]]["params"][target_handle].update(
            {"sourceId": node_ids[source_role], "sourceKey": source_handle}
        )
    graph = {"nodes": nodes, "paths": [list(node_ids.values())]}
    hints = {
        "modelType": spec["modelType"],
        "mode": spec["mode"],
        "studioExecutionSpec": {
            "schemaVersion": 1,
            "id": spec["id"],
            "contentHash": spec["contentHash"],
            "nodes": node_ids,
        },
    }
    return graph, hints


class StudioExecutionSpecTests(unittest.TestCase):
    def test_every_modular_models_loader_binds_the_exact_artifact_revision(self):
        modular_specs = [
            (spec_id, definition)
            for spec_id, definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.items()
            if any(
                role == "models" and node_key == "modules.ModularDiffusers.ModelsLoader"
                for role, node_key, _x, _y in definition.get("roles", ())
            )
        ]

        self.assertGreater(len(modular_specs), 0)
        for spec_id, definition in modular_specs:
            with self.subTest(spec_id=spec_id):
                self.assertIn(("models", "revision", "defaultRevision"), definition["bindings"])

    def test_modular_control_routes_inject_the_external_model_into_the_component_bundle(self):
        """Every reviewed ControlNet route must feed both graph stages.

        The control block consumes the model while constructing control inputs,
        and the selected Modular pipeline also requires that same component in
        the ModelsLoader bundle before ``init_pipeline()`` validates its
        pretrained components.
        """

        control_specs = [
            studio_execution_spec_for_pair("StableDiffusionXLModularPipeline", mode)
            for mode in ("control_image", "control_edit_image", "control_inpaint")
        ] + [
            studio_execution_spec_for_pair("QwenImageModularPipeline", mode)
            for mode in ("control_image", "control_edit_image", "control_inpaint")
        ]
        for spec in control_specs:
            self.assertIsNotNone(spec)
            with self.subTest(spec_id=spec["id"]):
                self.assertIn(("controlnetModel", "model", "models", "controlnet"), spec["edges"])
                self.assertIn(("controlnetModel", "model", "controlnet", "controlnet"), spec["edges"])
                models_definition = module_registry.MODULE_MAP["modules.ModularDiffusers"]["ModelsLoader"]
                models_params = _execution_spec_role_params(
                    spec,
                    "modules.ModularDiffusers.ModelsLoader",
                    models_definition,
                )
                self.assertEqual(models_params["controlnet"]["display"], "input")
                self.assertEqual(models_params["controlnet"]["type"], "diffusers_auto_model")

        text_to_image = studio_execution_spec_for_pair("QwenImageModularPipeline", "text_to_image")
        models_definition = module_registry.MODULE_MAP["modules.ModularDiffusers"]["ModelsLoader"]
        self.assertNotIn(
            "controlnet",
            _execution_spec_role_params(
                text_to_image,
                "modules.ModularDiffusers.ModelsLoader",
                models_definition,
            ),
        )

    def test_real_esrgan_image_upscale_uses_the_generic_spandrel_node(self):
        definition = STUDIO_EXECUTION_SPEC_DEFINITIONS["real-esrgan-x2-image-upscale:v1"]
        spec = studio_execution_spec_for_pair("SpandrelImageUpscale", "image_upscale")

        self.assertIsNotNone(spec)
        self.assertEqual(spec["loaderModule"], "modules.Spandrel")
        self.assertEqual(spec["loaderAction"], "Upscaler")
        self.assertEqual(spec["pipelineClass"], "SpandrelImageUpscaleV1")
        self.assertEqual(spec["defaultRepo"], "nateraw/real-esrgan")
        self.assertEqual(
            tuple((role, node_key) for role, node_key, _x, _y in spec["roles"]),
            (
                ("loadImage", "modules.Image.Load"),
                ("imageUpscaler", "modules.Spandrel.Upscaler"),
                ("preview", "modules.Image.Preview"),
            ),
        )
        self.assertNotIn("modules.ImageOperations.ProcessImage", {item[1] for item in spec["roles"]})
        self.assertEqual(definition["capability"]["artifactKind"], "spandrel_upscaler")
        self.assertEqual(definition["capability"]["downloadFiles"], ["RealESRGAN_x2plus.pth"])
        self.assertEqual(
            definition["capability"]["modeRequirements"]["image_upscale"]["requiredImages"],
            ["referenceImages"],
        )
        selection = module_registry.MODULE_MAP["modules.Spandrel"]["Upscaler"]["params"]["model_id"]["default"]
        self.assertEqual(selection["value"], "nateraw/real-esrgan/RealESRGAN_x2plus.pth")
        self.assertEqual(selection["revision"], "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094")
        self.assertEqual(selection["sha256"], "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb")

    def test_model_dependencies_are_pair_specific_immutable_artifact_receipts(self):
        qwen = studio_model_dependencies_for_pair(
            "QwenImageModularPipeline",
            "control_image",
        )
        qwen_direct = studio_model_dependencies_for_pair(
            "QwenImageControlNetPipeline",
            "control_image",
        )
        redux = studio_model_dependencies_for_pair("FluxReduxPipeline", "edit_image")
        redux_multi = studio_model_dependencies_for_pair("FluxReduxPipeline", "multi_image_reference_edit")
        sdxl_controlnet = studio_model_dependencies_for_pair(
            "StableDiffusionXLControlNetPipeline",
            "control_image",
        )
        sd15_controlnet = studio_model_dependencies_for_pair(
            "StableDiffusionPipeline",
            "control_image",
        )
        sdxl_adapter = studio_model_dependencies_for_pair(
            "StableDiffusionXLAdapterPipeline",
            "control_image",
        )
        animatediff_motion = studio_model_dependencies_for_pair(
            "AnimateDiffPipeline",
            "text_to_video",
        )
        cosmos_guardrail = [
            {
                "id": "cosmos3-mandatory-safety-guardrail",
                "kind": "safety_checker",
                "repo": "nvidia/Cosmos-Guardrail1",
                "revision": "d6d4bfa899a71454a700907664f3e88f503950cf",
            }
        ]

        self.assertEqual(
            qwen,
            [
                {
                    "id": "qwen-controlnet-union",
                    "kind": "controlnet",
                    "repo": "InstantX/Qwen-Image-ControlNet-Union",
                    "revision": "b13036f066d6dee7c20513e263d3d673055e9de8",
                }
            ],
        )
        self.assertEqual(qwen_direct, qwen)
        self.assertEqual(
            redux,
            [
                {
                    "id": "flux-redux-base",
                    "kind": "base",
                    "repo": "black-forest-labs/FLUX.1-dev",
                    "revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                }
            ],
        )
        self.assertEqual(redux_multi, redux)
        self.assertEqual(
            sdxl_controlnet,
            [
                {
                    "id": "sdxl-controlnet-canny",
                    "kind": "controlnet",
                    "repo": "diffusers/controlnet-canny-sdxl-1.0",
                    "revision": "eb115a19a10d14909256db740ed109532ab1483c",
                }
            ],
        )
        self.assertEqual(
            sdxl_adapter,
            [
                {
                    "id": "sdxl-t2i-adapter-canny",
                    "kind": "t2i_adapter",
                    "repo": "TencentARC/t2i-adapter-canny-sdxl-1.0",
                    "revision": "2d7244ba45ded9129cfbf8e96a4befb7f6094210",
                }
            ],
        )
        self.assertEqual(
            animatediff_motion,
            [
                {
                    "id": "animatediff-motion-adapter-v1-5-2",
                    "kind": "adapter",
                    "repo": "guoyww/animatediff-motion-adapter-v1-5-2",
                    "revision": "6167b88ffe39b4441fdf2113e77b99a6f56b7906",
                }
            ],
        )
        for mode in (
            "text_to_image",
            "text_to_video",
            "image_to_video",
            "video_to_video",
            "text_to_video_with_audio",
            "image_to_video_with_audio",
            "video_to_video_with_audio",
        ):
            self.assertEqual(
                studio_model_dependencies_for_pair("Cosmos3OmniModularPipeline", mode),
                cosmos_guardrail,
            )
            requirement = studio_model_requirements_for_pair("Cosmos3OmniModularPipeline", mode)[0]
            self.assertEqual(requirement["requiredForModes"], [mode])
            self.assertIn("Mandatory gated Cosmos", requirement["description"])
        for model_type, mode, expected_dependencies in (
            ("AnimateDiffPAGPipeline", "text_to_video", animatediff_motion),
            (
                "AnimateDiffVideoToVideoPipeline",
                "video_to_video",
                animatediff_motion,
            ),
            (
                "AnimateDiffControlNetPipeline",
                "control_to_video",
                animatediff_motion + sd15_controlnet,
            ),
            (
                "AnimateDiffVideoToVideoControlNetPipeline",
                "control_video_to_video",
                animatediff_motion + sd15_controlnet,
            ),
        ):
            with self.subTest(animatediff_dependency=(model_type, mode)):
                self.assertEqual(
                    studio_model_dependencies_for_pair(model_type, mode),
                    expected_dependencies,
                )
                self.assertTrue(
                    all(
                        requirement["requiredForModes"] == [mode]
                        for requirement in studio_model_requirements_for_pair(
                            model_type,
                            mode,
                        )
                    )
                )
        for model_type, mode, expected_dependency in (
            ("StableDiffusionPipeline", "control_edit_image", sd15_controlnet),
            ("StableDiffusionPipeline", "control_inpaint", sd15_controlnet),
            ("StableDiffusionPAGPipeline", "control_image", sd15_controlnet),
            ("StableDiffusionPAGPipeline", "control_inpaint", sd15_controlnet),
            ("StableDiffusionXLControlNetPipeline", "control_edit_image", sdxl_controlnet),
            ("StableDiffusionXLControlNetPipeline", "control_inpaint", sdxl_controlnet),
            ("StableDiffusionXLModularPipeline", "control_inpaint", sdxl_controlnet),
            ("StableDiffusionXLPAGPipeline", "control_image", sdxl_controlnet),
            ("StableDiffusionXLPAGPipeline", "control_edit_image", sdxl_controlnet),
        ):
            with self.subTest(model_type=model_type, mode=mode):
                self.assertEqual(
                    studio_model_dependencies_for_pair(model_type, mode),
                    expected_dependency,
                )
                requirements = studio_model_requirements_for_pair(model_type, mode)
                self.assertEqual(requirements[0]["requiredForModes"], [mode])
        self.assertEqual(
            studio_model_requirements_for_pair("FluxReduxPipeline", "edit_image")[0]["requiredForModes"],
            ["edit_image", "multi_image_reference_edit"],
        )
        self.assertEqual(studio_model_dependencies_for_pair("FluxReduxPipeline", "text_to_image"), [])

    def test_flux_registry_owns_profile_capability_and_auto_contracts(self):
        specs = validate_studio_execution_specs(module_registry.MODULE_MAP)
        self.assertEqual(
            [(item["modelType"], item["mode"]) for item in specs],
            [
                ("FluxSchnellPipeline", "text_to_image"),
                ("FluxDevPipeline", "text_to_image"),
                ("FluxKreaPipeline", "text_to_image"),
                ("FluxDepthPipeline", "control_image"),
                ("FluxCannyPipeline", "control_image"),
                ("FluxReduxPipeline", "edit_image"),
                ("FluxKontextPipeline", "edit_image"),
                ("FluxKontextPipeline", "multi_image_reference_edit"),
                ("FluxFillPipeline", "inpaint"),
                ("FluxFillPipeline", "outpaint"),
                ("Flux2KleinPipeline", "text_to_image"),
                ("Flux2KleinPipeline", "edit_image"),
                ("Flux2KleinPipeline", "multi_image_reference_edit"),
                ("WanImageToVideoPipeline", "image_to_video"),
                ("WanTI2VPipeline", "text_to_video"),
                ("WanVideoPipeline", "text_to_video"),
                ("WanVideoPipeline", "video_to_video"),
                ("WanVideoPipeline", "video_color_edit"),
                ("LTXVideoPipeline", "text_to_video"),
                ("LTXVideoPipeline", "image_to_video"),
                ("LTXVideoPipeline", "video_to_video"),
                ("LTXVideoPipeline", "reference_to_video"),
                ("AceStepAudioPipeline", "text_to_audio"),
                ("AceStepAudioPipeline", "audio_variation"),
                ("AceStepAudioPipeline", "audio_continuation"),
                ("AceStepAudioPipeline", "audio_repaint"),
                ("QwenImageEditModularPipeline", "inpaint"),
                ("WanVACEPipeline", "text_to_video"),
                ("WanVACEPipeline", "video_inpaint"),
                ("WanVACEPipeline", "video_outpaint"),
                ("WanVACEPipeline", "control_to_video"),
                ("QwenImageEditModularPipeline", "outpaint"),
                ("ZImageModularPipeline", "text_to_image"),
                ("ZImageModularPipeline", "edit_image"),
                ("QwenImageModularPipeline", "text_to_image"),
                ("QwenImageModularPipeline", "edit_image"),
                ("QwenImageModularPipeline", "inpaint"),
                ("QwenImageEditModularPipeline", "edit_image"),
                ("QwenImageEditModularPipeline", "modular_inpainting"),
                ("FluxModularPipeline", "text_to_image"),
                ("FluxModularPipeline", "image_to_image"),
                ("FluxKontextModularPipeline", "text_to_image"),
                ("FluxKontextModularPipeline", "edit_image"),
                ("Flux2KleinModularPipeline", "text_to_image"),
                ("Flux2KleinModularPipeline", "edit_image"),
                ("Flux2KleinBaseModularPipeline", "text_to_image"),
                ("Flux2KleinBaseModularPipeline", "edit_image"),
                ("ZImageModularPipeline", "modular_text_to_image"),
                ("ZImageModularPipeline", "modular_image_to_image"),
                ("StableDiffusionXLModularPipeline", "text_to_image"),
                ("StableDiffusionXLModularPipeline", "edit_image"),
                ("StableDiffusionXLModularPipeline", "inpaint"),
                ("StableDiffusionXLModularPipeline", "control_image"),
                ("StableDiffusionXLModularPipeline", "control_edit_image"),
                ("StableDiffusionXLModularPipeline", "control_inpaint"),
                ("QwenImageEditPlusModularPipeline", "edit_image"),
                ("QwenImageEditPlusModularPipeline", "multi_image_reference_edit"),
                ("QwenImageLayeredModularPipeline", "layer_decomposition"),
                ("QwenImageModularPipeline", "control_image"),
                ("QwenImageModularPipeline", "image_to_image"),
                ("QwenImageModularPipeline", "modular_text_to_image"),
                ("QwenImageModularPipeline", "inpainting"),
                ("QwenImageModularPipeline", "control_edit_image"),
                ("QwenImageModularPipeline", "control_inpaint"),
                ("QwenImageControlNetPipeline", "control_image"),
                ("QwenImageLayeredPipeline", "layer_decomposition"),
                ("StableAudioPipeline", "text_to_audio"),
                ("HeliosModularPipeline", "text_to_video"),
                ("HeliosModularPipeline", "image_to_video"),
                ("HeliosModularPipeline", "video_to_video"),
                ("HeliosPyramidModularPipeline", "text_to_video"),
                ("HeliosPyramidModularPipeline", "image_to_video"),
                ("HeliosPyramidModularPipeline", "video_to_video"),
                ("HeliosPyramidDistilledModularPipeline", "text_to_video"),
                ("HeliosPyramidDistilledModularPipeline", "image_to_video"),
                ("HeliosPyramidDistilledModularPipeline", "video_to_video"),
                ("HunyuanVideo15ModularPipeline", "text_to_video"),
                ("HunyuanVideo15ModularPipeline", "image_to_video"),
                ("MiniMaxH3ModularPipeline", "text_to_video_with_audio"),
                (
                    "MiniMaxH3ModularPipeline",
                    "first_last_frame_to_video_with_audio",
                ),
                ("MiniMaxH3ModularPipeline", "reference_to_video_with_audio"),
                ("WanAnimate2ModularPipeline", "character_animate"),
                ("WanAnimate2DistilledModularPipeline", "character_animate"),
                ("Cosmos3OmniModularPipeline", "text_to_image"),
                ("Cosmos3OmniModularPipeline", "text_to_video"),
                ("Cosmos3OmniModularPipeline", "image_to_video"),
                ("Cosmos3OmniModularPipeline", "video_to_video"),
                ("Cosmos3OmniModularPipeline", "text_to_video_with_audio"),
                ("Cosmos3OmniModularPipeline", "image_to_video_with_audio"),
                ("Cosmos3OmniModularPipeline", "video_to_video_with_audio"),
                ("Cosmos3DistilledModularPipeline", "text_to_image"),
                ("Cosmos3DistilledModularPipeline", "image_to_video"),
                ("AnimaModularPipeline", "text_to_image"),
                ("AnimaModularPipeline", "image_to_image"),
                ("MiniMaxMusic3ModularPipeline", "text_to_audio"),
                ("LongCatAudioDiTPipeline", "text_to_audio"),
                ("AudioLDM2Pipeline", "text_to_audio"),
                ("ShapEPipeline", "text_to_3d"),
                ("ShapEImg2ImgPipeline", "image_to_3d"),
                ("FluxDevPipeline", "edit_image"),
                ("FluxDevPipeline", "inpaint"),
                ("StableDiffusionXLPipeline", "text_to_image"),
                ("StableDiffusionXLPipeline", "edit_image"),
                ("StableDiffusionXLPipeline", "inpaint"),
                ("StableDiffusionXLModularPipeline", "control_union_image"),
                ("StableDiffusionXLModularPipeline", "control_union_edit_image"),
                ("StableDiffusionXLModularPipeline", "control_union_inpaint"),
                ("StableDiffusionXLModularPipeline", "ip_adapter_image"),
                ("StableDiffusionXLModularPipeline", "ip_adapter_edit_image"),
                ("StableDiffusionXLModularPipeline", "ip_adapter_inpaint"),
                ("StableDiffusionXLModularPipeline", "ip_adapter_control_image"),
                ("StableDiffusionXLModularPipeline", "ip_adapter_control_edit_image"),
                ("StableDiffusionXLModularPipeline", "ip_adapter_control_inpaint"),
                ("StableDiffusionXLModularPipeline", "ip_adapter_control_union_image"),
                ("StableDiffusionXLModularPipeline", "ip_adapter_control_union_edit_image"),
                ("StableDiffusionXLModularPipeline", "ip_adapter_control_union_inpaint"),
                ("Wan22Pipeline", "text_to_video"),
                ("WanAnimatePipeline", "character_animate"),
                ("WanAnimatePipeline", "character_replace"),
                ("LTXI2VLongMultiPromptPipeline", "image_to_video"),
                ("LTX2ConditionPipeline", "text_to_video"),
                ("LTX2ConditionPipeline", "image_to_video"),
                ("LTX2ConditionPipeline", "reference_to_video"),
                ("LTX2ConditionPipeline", "video_to_video"),
                ("LTX2InContextPipeline", "in_context_to_video"),
                ("HunyuanVideoFramepackPipeline", "image_to_video"),
                ("StableVideoDiffusionPipeline", "image_to_video"),
                ("AnimateDiffPipeline", "text_to_video"),
                ("AnimateDiffPAGPipeline", "text_to_video"),
                ("AnimateDiffVideoToVideoPipeline", "video_to_video"),
                ("AnimateDiffControlNetPipeline", "control_to_video"),
                (
                    "AnimateDiffVideoToVideoControlNetPipeline",
                    "control_video_to_video",
                ),
                ("AnimateLCMPipeline", "text_to_video"),
                ("CogVideoXPipeline", "text_to_video"),
                ("CogVideoXVideoToVideoPipeline", "video_to_video"),
                ("AllegroPipeline", "text_to_video"),
                ("LattePipeline", "text_to_video"),
                ("MochiPipeline", "text_to_video"),
                ("SanaVideoPipeline", "text_to_video"),
                ("SanaImageToVideoPipeline", "image_to_video"),
                ("WanModularPipeline", "text_to_video"),
                ("WanImage2VideoModularPipeline", "single_image_to_video"),
                ("WanImage2VideoModularPipeline", "image_to_video"),
                ("DDPMPipeline", "unconditional_image"),
                ("DDIMPipeline", "unconditional_image"),
                ("ConsistencyModelPipeline", "unconditional_image"),
                ("StableDiffusionPipeline", "text_to_image"),
                ("StableDiffusionPipeline", "edit_image"),
                ("StableDiffusionPipeline", "inpaint"),
                ("StableDiffusionPipeline", "control_image"),
                ("StableDiffusionPipeline", "control_edit_image"),
                ("StableDiffusionPipeline", "control_inpaint"),
                ("StableDiffusionXLTurboPipeline", "text_to_image"),
                ("StableDiffusionXLInstructPix2PixPipeline", "edit_image"),
                ("StableDiffusionXLControlNetPipeline", "control_image"),
                ("StableDiffusionXLControlNetPipeline", "control_edit_image"),
                ("StableDiffusionXLControlNetPipeline", "control_inpaint"),
                ("HunyuanDiTPipeline", "text_to_image"),
                ("HunyuanDiTPAGPipeline", "text_to_image"),
                ("HunyuanDiTControlNetPipeline", "control_image"),
                ("StableDiffusionXLAdapterPipeline", "control_image"),
                ("StableDiffusionXLPAGPipeline", "text_to_image"),
                ("StableDiffusionXLPAGPipeline", "edit_image"),
                ("StableDiffusionXLPAGPipeline", "inpaint"),
                ("StableDiffusionXLPAGPipeline", "control_image"),
                ("StableDiffusionXLPAGPipeline", "control_edit_image"),
                ("SanaPipeline", "text_to_image"),
                ("SanaPAGPipeline", "text_to_image"),
                ("SanaSprintPipeline", "text_to_image"),
                ("SanaSprintPipeline", "edit_image"),
                ("PixArtSigmaPipeline", "text_to_image"),
                ("PixArtSigmaPAGPipeline", "text_to_image"),
                ("Kandinsky3Pipeline", "text_to_image"),
                ("Kandinsky3Pipeline", "edit_image"),
                ("LongCatImagePipeline", "text_to_image"),
                ("LongCatImageEditPipeline", "edit_image"),
                ("LuminaPipeline", "text_to_image"),
                ("Lumina2Pipeline", "text_to_image"),
                ("OmniGenPipeline", "text_to_image"),
                ("OmniGenPipeline", "edit_image"),
                ("OmniGenPipeline", "multi_image_reference_edit"),
                ("OvisImagePipeline", "text_to_image"),
                ("PRXPipeline", "text_to_image"),
                ("NucleusMoEImagePipeline", "text_to_image"),
                ("AuraFlowPipeline", "text_to_image"),
                ("ChromaPipeline", "text_to_image"),
                ("CogView3PlusPipeline", "text_to_image"),
                ("CogView4Pipeline", "text_to_image"),
                ("ErnieImagePipeline", "text_to_image"),
                ("GlmImagePipeline", "text_to_image"),
                ("JoyImageEditPipeline", "text_to_image"),
                ("JoyImageEditPipeline", "edit_image"),
                ("JoyImageEditPlusPipeline", "edit_image"),
                ("JoyImageEditPlusPipeline", "multi_image_reference_edit"),
                ("DreamLitePipeline", "text_to_image"),
                ("DreamLitePipeline", "edit_image"),
                ("DreamLiteMobilePipeline", "text_to_image"),
                ("DreamLiteMobilePipeline", "edit_image"),
                ("LatentConsistencyModelPipeline", "text_to_image"),
                ("LatentConsistencyModelPipeline", "edit_image"),
                ("StableDiffusionPAGPipeline", "text_to_image"),
                ("StableDiffusionPAGPipeline", "edit_image"),
                ("StableDiffusionPAGPipeline", "inpaint"),
                ("StableDiffusionPAGPipeline", "control_image"),
                ("StableDiffusionPAGPipeline", "control_inpaint"),
                ("MarigoldDepthPipeline", "depth_estimation"),
                ("HuggingFaceTextGenerationModel", "text_generation"),
                ("HuggingFaceImageTextToTextModel", "image_to_text"),
                ("HuggingFaceAnyToAnyModel", "text_generation"),
                ("HuggingFaceAnyToAnyModel", "image_to_text"),
                ("HuggingFaceAnyToAnyModel", "text_to_image"),
                ("HuggingFaceSpeechRecognitionModel", "speech_to_text"),
                ("HuggingFaceSpeechRecognitionModel", "speech_translation"),
                ("HuggingFaceCTCSpeechRecognitionModel", "speech_to_text"),
                ("FluxReduxPipeline", "multi_image_reference_edit"),
                ("FluxDepthPipeline", "control_edit_image"),
                ("FluxDepthPipeline", "control_inpaint"),
                ("FluxCannyPipeline", "control_edit_image"),
                ("FluxCannyPipeline", "control_inpaint"),
                ("QwenImageEditPipeline", "edit_image"),
                ("QwenImageEditPlusPipeline", "edit_image"),
                ("QwenImageEditPlusPipeline", "multi_image_reference_edit"),
                ("ZImageInpaintPipeline", "inpaint"),
                ("ZImageInpaintPipeline", "outpaint"),
                ("FluxKontextInpaintPipeline", "inpaint"),
                ("FluxKontextInpaintPipeline", "outpaint"),
                ("Flux2KleinInpaintPipeline", "inpaint"),
                ("Flux2KleinInpaintPipeline", "outpaint"),
                ("ChromaImg2ImgPipeline", "edit_image"),
                ("ChromaInpaintPipeline", "inpaint"),
                ("ChromaInpaintPipeline", "outpaint"),
                ("LTX2Pipeline", "text_to_video"),
                ("BuiltinImageOperation", "image_adjustment"),
                ("BuiltinImageOperation", "image_filter"),
                ("BuiltinImageOperation", "image_crop"),
                ("BuiltinImageOperation", "image_upscale"),
                ("BuiltinImageOperation", "image_stitch"),
                ("BuiltinImageOperation", "image_tile"),
                ("BuiltinImageOperation", "image_channels"),
                ("BuiltinImageOperation", "mask_composite"),
                ("BuiltinAudioOperation", "audio_trim"),
                ("BuiltinAudioOperation", "audio_join"),
                ("BuiltinAudioOperation", "audio_loudness_match"),
                ("BuiltinDataOperation", "text_select"),
                ("BuiltinDataOperation", "data_conversion"),
                ("BuiltinDataOperation", "graph_utility"),
                ("BuiltinVideoOperation", "video_frame_extract"),
                ("BuiltinVideoOperation", "frame_interpolation"),
                ("BuiltinVideoOperation", "video_stitch"),
                ("BuiltinVideoOperation", "video_trim"),
                ("BuiltinVideoOperation", "video_reverse"),
                ("BuiltinVideoOperation", "video_tile"),
                ("SpandrelVideoUpscale", "video_upscale"),
                ("SpandrelImageUpscale", "image_upscale"),
                ("Flux2Pipeline", "text_to_image"),
                ("Flux2Pipeline", "multi_image_reference_edit"),
                ("Flux2ModularPipeline", "text_to_image"),
                ("Flux2ModularPipeline", "edit_image"),
                ("ErnieImageModularPipeline", "text_to_image"),
                ("LTXModularPipeline", "text_to_video"),
                ("LTXModularPipeline", "image_to_video"),
                ("Wan22ModularPipeline", "text_to_video"),
                ("Wan22Image2VideoModularPipeline", "image_to_video"),
                ("LTX2ModularPipeline", "text_to_video"),
                ("LTX2ModularPipeline", "image_to_video"),
                ("LTX2ModularPipeline", "reference_to_video"),
                ("LTX2ModularPipeline", "in_context_to_video"),
                ("FluxControlNetPipeline", "control_image"),
                ("FluxControlNetImg2ImgPipeline", "control_edit_image"),
                ("FluxControlNetInpaintPipeline", "control_inpaint"),
                ("Flux2KleinKVPipeline", "text_to_image"),
                ("Flux2KleinKVPipeline", "edit_image"),
                ("Flux2KleinKVPipeline", "multi_image_reference_edit"),
            ],
        )
        by_id = {item["id"]: item for item in specs}
        for spec_id, pipeline_class in (
            ("ddpm-cifar10:unconditional-image:v1", "DDPMPipeline"),
            ("ddim-cifar10:unconditional-image:v1", "DDIMPipeline"),
            ("consistency-imagenet64:unconditional-image:v1", "ConsistencyModelPipeline"),
        ):
            with self.subTest(unconditional_spec=spec_id):
                specification = by_id[spec_id]
                self.assertEqual(specification["pipelineClass"], pipeline_class)
                self.assertEqual(specification["mode"], "unconditional_image")
                self.assertIn(
                    ("diffusersImagePipeline", "pipeline", "diffusersUnconditionalGenerate", "pipeline"),
                    specification["edges"],
                )
                self.assertIn(
                    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
                    specification["edges"],
                )
                self.assertIn(
                    ("diffusersQuantization", "components", "quantizedComponents"),
                    specification["bindings"],
                )
                self.assertTrue(DIFFUSERS_EXECUTION_PROFILES[specification["executionProfileId"]].live_proof)
                self.assertNotIn("modules.DiffusersImage.Generate", [role[1] for role in specification["roles"]])
        marigold = by_id["marigold-depth-lcm-v1-0:depth-estimation:v1"]
        self.assertEqual(marigold["pipelineClass"], "MarigoldDepthPipeline")
        self.assertIn(
            ("diffusersImagePipeline", "pipeline", "diffusersPredictMap", "pipeline"),
            marigold["edges"],
        )
        self.assertIn(("loadImage", "image", "diffusersPredictMap", "image"), marigold["edges"])
        self.assertIn(("diffusersPredictMap", "preview_images", "preview", "image"), marigold["edges"])
        self.assertIn(("diffusersPredictMap", "processing_resolution", "processingResolution"), marigold["bindings"])
        self.assertIn(("diffusersPredictMap", "match_input_resolution", "matchInputResolution"), marigold["bindings"])
        smollm = by_id["smollm2-135m-instruct:text-generation:v1"]
        self.assertEqual(smollm["pipelineClass"], "AutoModelForCausalLM")
        self.assertIn(
            ("transformersTextModel", "model", "transformersTextGenerate", "model"),
            smollm["edges"],
        )
        self.assertIn(
            ("transformersTextGenerate", "result", "transformersTextPreview", "value"),
            smollm["edges"],
        )
        smolvlm = by_id["smolvlm-256m-instruct:image-to-text:v1"]
        self.assertEqual(smolvlm["pipelineClass"], "AutoModelForImageTextToText")
        self.assertIn(
            ("loadImage", "image", "transformersImageTextGenerate", "images"),
            smolvlm["edges"],
        )
        self.assertIn(("loadImage", "file", "referenceImages"), smolvlm["bindings"])
        whisper = by_id["whisper-tiny:speech-to-text:v1"]
        self.assertEqual(whisper["pipelineClass"], "AutoModelForSpeechSeq2Seq")
        self.assertIn(("speechModel", "model", "transcribeAudio", "model"), whisper["edges"])
        self.assertIn(("loadAudio", "audio", "transcribeAudio", "audio"), whisper["edges"])
        self.assertIn(("transcribeAudio", "transcript", "transcriptPreview", "value"), whisper["edges"])
        self.assertIn(("transcribeAudio", "task", "transcribe"), whisper["bindings"])
        translation = by_id["whisper-tiny:speech-translation:v1"]
        self.assertIn(("transcribeAudio", "task", "translate"), translation["bindings"])
        for spec_id, mode, pipeline_class in (
            ("sd15-base:text-to-image:v1", "text_to_image", "StableDiffusionPipeline"),
            ("sd15-base:edit-image:v1", "edit_image", "StableDiffusionImg2ImgPipeline"),
            ("sd15-base:inpaint:v1", "inpaint", "StableDiffusionInpaintPipeline"),
        ):
            with self.subTest(sd15_spec=spec_id):
                specification = by_id[spec_id]
                self.assertEqual(specification["modelType"], "StableDiffusionPipeline")
                self.assertEqual(specification["mode"], mode)
                self.assertEqual(specification["pipelineClass"], pipeline_class)
                self.assertEqual(specification["defaultRepo"], "stable-diffusion-v1-5/stable-diffusion-v1-5")
                self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), specification["bindings"])
        controlnet = by_id["sd15-controlnet-canny:control-image:v1"]
        self.assertEqual(controlnet["pipelineClass"], "StableDiffusionControlNetPipeline")
        self.assertIn(("diffusersImagePipeline", "conditioning_kind", "kind"), controlnet["bindings"])
        self.assertIn(("diffusersImagePipeline", "conditioning_model_id", "repo"), controlnet["bindings"])
        self.assertIn(("diffusersImagePipeline", "conditioning_revision", "revision"), controlnet["bindings"])
        self.assertIn(
            ("loadImage", "image", "controlPreprocessor", "image"),
            controlnet["edges"],
        )
        self.assertIn(
            ("controlPreprocessor", "output", "diffusersImageControl", "control_image"),
            controlnet["edges"],
        )
        self.assertIn(("controlPreprocessor", "low_threshold", "cannyLowThreshold"), controlnet["bindings"])
        self.assertIn(("controlPreprocessor", "high_threshold", "cannyHighThreshold"), controlnet["bindings"])
        self.assertIn(
            ("diffusersImageControl", "conditioning_scale", "conditioningScale"),
            controlnet["bindings"],
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[controlnet["executionProfileId"]].live_proof)
        turbo = by_id["sdxl-turbo:text-to-image:v1"]
        self.assertEqual(turbo["modelType"], "StableDiffusionXLTurboPipeline")
        self.assertEqual(turbo["pipelineClass"], "StableDiffusionXLTurboPipeline")
        self.assertEqual(turbo["defaultRepo"], "stabilityai/sdxl-turbo")
        self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), turbo["bindings"])
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[turbo["executionProfileId"]].live_proof)
        instruct = by_id["sdxl-instruct-pix2pix:edit-image:v1"]
        self.assertEqual(instruct["modelType"], "StableDiffusionXLInstructPix2PixPipeline")
        self.assertEqual(instruct["pipelineClass"], "StableDiffusionXLInstructPix2PixPipeline")
        self.assertEqual(instruct["defaultRepo"], "diffusers/sdxl-instructpix2pix-768")
        self.assertIn(("diffusersImageEdit", "image_guidance_scale", "conditioningScale"), instruct["bindings"])
        self.assertNotIn(("diffusersImageEdit", "reference_strength", "conditioningScale"), instruct["bindings"])
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[instruct["executionProfileId"]].live_proof)
        sdxl_controlnet = by_id["sdxl-controlnet-canny:control-image:v1"]
        self.assertEqual(sdxl_controlnet["modelType"], "StableDiffusionXLControlNetPipeline")
        self.assertEqual(sdxl_controlnet["pipelineClass"], "StableDiffusionXLControlNetPipeline")
        self.assertEqual(sdxl_controlnet["defaultRepo"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertIn(
            ("diffusersImagePipeline", "conditioning_model_id", "repo"),
            sdxl_controlnet["bindings"],
        )
        self.assertIn(
            ("controlPreprocessor", "output", "diffusersImageControl", "control_image"),
            sdxl_controlnet["edges"],
        )
        self.assertIn(
            ("diffusersImageControl", "conditioning_scale", "conditioningScale"),
            sdxl_controlnet["bindings"],
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[sdxl_controlnet["executionProfileId"]].live_proof)
        hunyuan_controlnet = by_id["hunyuan-dit-v1-2-controlnet-canny:control-image:v1"]
        self.assertEqual(hunyuan_controlnet["modelType"], "HunyuanDiTControlNetPipeline")
        self.assertEqual(hunyuan_controlnet["pipelineClass"], "HunyuanDiTControlNetPipeline")
        self.assertEqual(
            hunyuan_controlnet["defaultRepo"],
            "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        )
        self.assertIn(
            ("diffusersImagePipeline", "conditioning_model_id", "repo"),
            hunyuan_controlnet["bindings"],
        )
        self.assertIn(
            ("controlPreprocessor", "output", "diffusersImageControl", "control_image"),
            hunyuan_controlnet["edges"],
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[hunyuan_controlnet["executionProfileId"]].live_proof)
        hunyuan_pag = by_id["hunyuan-dit-v1-2-distilled-pag:text-to-image:v1"]
        self.assertEqual(hunyuan_pag["modelType"], "HunyuanDiTPAGPipeline")
        self.assertEqual(hunyuan_pag["pipelineClass"], "HunyuanDiTPAGPipeline")
        self.assertEqual(
            hunyuan_pag["defaultRepo"],
            "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        )
        self.assertIn(("diffusersImageGenerate", "pag_scale", "pagScale"), hunyuan_pag["bindings"])
        self.assertNotIn(
            ("diffusersImageGenerate", "max_sequence_length", "maxSequenceLength"),
            hunyuan_pag["bindings"],
        )
        self.assertNotIn(("diffusersImageGenerate", "strength", "strength"), hunyuan_pag["bindings"])
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[hunyuan_pag["executionProfileId"]].live_proof)
        sdxl_adapter = by_id["sdxl-t2i-adapter-canny:control-image:v1"]
        self.assertEqual(sdxl_adapter["modelType"], "StableDiffusionXLAdapterPipeline")
        self.assertEqual(sdxl_adapter["pipelineClass"], "StableDiffusionXLAdapterPipeline")
        self.assertEqual(sdxl_adapter["defaultRepo"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertIn(
            ("diffusersImagePipeline", "conditioning_model_id", "repo"),
            sdxl_adapter["bindings"],
        )
        self.assertIn(
            ("controlPreprocessor", "output", "diffusersImageControl", "control_image"),
            sdxl_adapter["edges"],
        )
        self.assertIn(
            ("diffusersImageControl", "conditioning_scale", "conditioningScale"),
            sdxl_adapter["bindings"],
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[sdxl_adapter["executionProfileId"]].live_proof)
        sdxl_pag = by_id["sdxl-pag:text-to-image:v1"]
        self.assertEqual(sdxl_pag["modelType"], "StableDiffusionXLPAGPipeline")
        self.assertEqual(sdxl_pag["pipelineClass"], "StableDiffusionXLPAGPipeline")
        self.assertEqual(sdxl_pag["defaultRepo"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertIn(("diffusersImageGenerate", "pag_scale", "pagScale"), sdxl_pag["bindings"])
        self.assertIn(
            ("diffusersImageGenerate", "pag_adaptive_scale", "pagAdaptiveScale"),
            sdxl_pag["bindings"],
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[sdxl_pag["executionProfileId"]].live_proof)
        sdxl_pag_edit = by_id["sdxl-pag:edit-image:v1"]
        self.assertEqual(sdxl_pag_edit["pipelineClass"], "StableDiffusionXLPAGImg2ImgPipeline")
        self.assertIn(("diffusersImageEdit", "pag_scale", "pagScale"), sdxl_pag_edit["bindings"])
        self.assertIn(
            ("diffusersImageEdit", "pag_adaptive_scale", "pagAdaptiveScale"),
            sdxl_pag_edit["bindings"],
        )
        self.assertIn(
            ("loadImage", "image", "diffusersImageEdit", "image"),
            sdxl_pag_edit["edges"],
        )
        sdxl_pag_inpaint = by_id["sdxl-pag:inpaint:v1"]
        self.assertEqual(sdxl_pag_inpaint["pipelineClass"], "StableDiffusionXLPAGInpaintPipeline")
        self.assertIn(
            ("diffusersImageInpaint", "pag_scale", "pagScale"),
            sdxl_pag_inpaint["bindings"],
        )
        self.assertIn(
            ("diffusersImageInpaint", "pag_adaptive_scale", "pagAdaptiveScale"),
            sdxl_pag_inpaint["bindings"],
        )
        sana = by_id["sana-600m:text-to-image:v1"]
        self.assertEqual(sana["modelType"], "SanaPipeline")
        self.assertEqual(sana["pipelineClass"], "SanaPipeline")
        self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), sana["bindings"])
        self.assertEqual(DIFFUSERS_EXECUTION_PROFILES[sana["executionProfileId"]].max_low_memory_steps, 20)
        sana_pag = by_id["sana-600m-pag:text-to-image:v1"]
        self.assertEqual(sana_pag["modelType"], "SanaPAGPipeline")
        self.assertEqual(sana_pag["pipelineClass"], "SanaPAGPipeline")
        self.assertIn(("diffusersImageGenerate", "pag_scale", "pagScale"), sana_pag["bindings"])
        self.assertIn(
            ("diffusersImageGenerate", "max_sequence_length", "maxSequenceLength"),
            sana_pag["bindings"],
        )
        self.assertNotIn(("diffusersImageGenerate", "strength", "strength"), sana_pag["bindings"])
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[sana_pag["executionProfileId"]].live_proof)
        sana_sprint = by_id["sana-sprint-600m:text-to-image:v1"]
        self.assertEqual(sana_sprint["pipelineClass"], "SanaSprintPipeline")
        self.assertEqual(
            DIFFUSERS_EXECUTION_PROFILES[sana_sprint["executionProfileId"]].max_low_memory_steps,
            4,
        )
        sana_sprint_edit = by_id["sana-sprint-600m:edit-image:v1"]
        self.assertEqual(sana_sprint_edit["modelType"], "SanaSprintPipeline")
        self.assertEqual(sana_sprint_edit["pipelineClass"], "SanaSprintImg2ImgPipeline")
        self.assertIn(("diffusersImageEdit", "strength", "strength"), sana_sprint_edit["bindings"])
        self.assertIn(("loadImage", "image", "diffusersImageEdit", "image"), sana_sprint_edit["edges"])
        pixart = by_id["pixart-sigma-1024:text-to-image:v1"]
        self.assertEqual(pixart["modelType"], "PixArtSigmaPipeline")
        self.assertEqual(pixart["pipelineClass"], "PixArtSigmaPipeline")
        self.assertEqual(
            pixart["defaultRepo"],
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        )
        self.assertEqual(
            DIFFUSERS_EXECUTION_PROFILES[pixart["executionProfileId"]].max_low_memory_steps,
            20,
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[pixart["executionProfileId"]].live_proof)
        pixart_pag = by_id["pixart-sigma-1024-pag:text-to-image:v1"]
        self.assertEqual(pixart_pag["modelType"], "PixArtSigmaPAGPipeline")
        self.assertEqual(pixart_pag["pipelineClass"], "PixArtSigmaPAGPipeline")
        self.assertEqual(pixart_pag["defaultRepo"], pixart["defaultRepo"])
        self.assertIn(("diffusersImageGenerate", "pag_scale", "pagScale"), pixart_pag["bindings"])
        self.assertIn(
            ("diffusersImageGenerate", "max_sequence_length", "maxSequenceLength"),
            pixart_pag["bindings"],
        )
        self.assertNotIn(("diffusersImageGenerate", "strength", "strength"), pixart_pag["bindings"])
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[pixart_pag["executionProfileId"]].live_proof)
        kandinsky3 = by_id["kandinsky3:text-to-image:v1"]
        self.assertEqual(kandinsky3["modelType"], "Kandinsky3Pipeline")
        self.assertEqual(kandinsky3["pipelineClass"], "Kandinsky3Pipeline")
        self.assertEqual(kandinsky3["defaultRepo"], "kandinsky-community/kandinsky-3")
        kandinsky3_profile = DIFFUSERS_EXECUTION_PROFILES[kandinsky3["executionProfileId"]]
        self.assertEqual(kandinsky3_profile.max_low_memory_side, 1024)
        self.assertEqual(kandinsky3_profile.max_low_memory_steps, 25)
        self.assertFalse(kandinsky3_profile.live_proof)
        kandinsky3_edit = by_id["kandinsky3:edit-image:v1"]
        self.assertEqual(kandinsky3_edit["pipelineClass"], "Kandinsky3Img2ImgPipeline")
        self.assertIn(("loadImage", "image", "diffusersImageEdit", "image"), kandinsky3_edit["edges"])
        self.assertIn(("diffusersImageEdit", "strength", "strength"), kandinsky3_edit["bindings"])
        auraflow = by_id["auraflow-v0.3:text-to-image:v1"]
        self.assertEqual(auraflow["modelType"], "AuraFlowPipeline")
        self.assertEqual(auraflow["pipelineClass"], "AuraFlowPipeline")
        self.assertEqual(auraflow["defaultRepo"], "fal/AuraFlow-v0.3")
        auraflow_profile = DIFFUSERS_EXECUTION_PROFILES[auraflow["executionProfileId"]]
        self.assertEqual(auraflow_profile.max_low_memory_side, 1536)
        self.assertEqual(auraflow_profile.max_low_memory_steps, 50)
        self.assertFalse(auraflow_profile.live_proof)
        chroma = by_id["chroma1-hd:text-to-image:v1"]
        self.assertEqual(chroma["modelType"], "ChromaPipeline")
        self.assertEqual(chroma["pipelineClass"], "ChromaPipeline")
        self.assertEqual(chroma["defaultRepo"], "lodestones/Chroma1-HD")
        chroma_profile = DIFFUSERS_EXECUTION_PROFILES[chroma["executionProfileId"]]
        self.assertEqual(chroma_profile.max_low_memory_side, 1024)
        self.assertEqual(chroma_profile.max_low_memory_steps, 40)
        self.assertFalse(chroma_profile.live_proof)
        cogview3 = by_id["cogview3-plus-3b:text-to-image:v1"]
        self.assertEqual(cogview3["modelType"], "CogView3PlusPipeline")
        self.assertEqual(cogview3["pipelineClass"], "CogView3PlusPipeline")
        self.assertEqual(cogview3["defaultRepo"], "zai-org/CogView3-Plus-3B")
        cogview3_profile = DIFFUSERS_EXECUTION_PROFILES[cogview3["executionProfileId"]]
        self.assertEqual(cogview3_profile.max_low_memory_side, 1024)
        self.assertEqual(cogview3_profile.max_low_memory_steps, 50)
        self.assertFalse(cogview3_profile.live_proof)
        cogview4 = by_id["cogview4-6b:text-to-image:v1"]
        self.assertEqual(cogview4["modelType"], "CogView4Pipeline")
        self.assertEqual(cogview4["pipelineClass"], "CogView4Pipeline")
        self.assertEqual(cogview4["defaultRepo"], "zai-org/CogView4-6B")
        cogview4_profile = DIFFUSERS_EXECUTION_PROFILES[cogview4["executionProfileId"]]
        self.assertEqual(cogview4_profile.max_low_memory_side, 1024)
        self.assertEqual(cogview4_profile.max_low_memory_steps, 50)
        self.assertFalse(cogview4_profile.live_proof)
        ernie = by_id["ernie-image-turbo:text-to-image:v1"]
        self.assertEqual(ernie["modelType"], "ErnieImagePipeline")
        self.assertEqual(ernie["pipelineClass"], "ErnieImagePipeline")
        self.assertEqual(ernie["defaultRepo"], "baidu/ERNIE-Image-Turbo")
        ernie_profile = DIFFUSERS_EXECUTION_PROFILES[ernie["executionProfileId"]]
        self.assertEqual(ernie_profile.max_low_memory_side, 1024)
        self.assertEqual(ernie_profile.max_low_memory_steps, 8)
        self.assertFalse(ernie_profile.live_proof)
        glm_image = by_id["glm-image:text-to-image:v1"]
        self.assertEqual(glm_image["modelType"], "GlmImagePipeline")
        self.assertEqual(glm_image["pipelineClass"], "GlmImagePipeline")
        self.assertEqual(glm_image["defaultRepo"], "zai-org/GLM-Image")
        glm_image_profile = DIFFUSERS_EXECUTION_PROFILES[glm_image["executionProfileId"]]
        self.assertEqual(glm_image_profile.max_low_memory_side, 1024)
        self.assertEqual(glm_image_profile.max_low_memory_steps, 50)
        self.assertFalse(glm_image_profile.live_proof)
        dreamlite = by_id["dreamlite-base:text-to-image:v1"]
        self.assertEqual(dreamlite["defaultRepo"], "carlofkl/DreamLite-base")
        self.assertEqual(dreamlite["pipelineClass"], "DreamLitePipeline")
        self.assertIn(
            ("diffusersImageGenerate", "guidance_scale", "guidanceScale"),
            dreamlite["bindings"],
        )
        dreamlite_edit = by_id["dreamlite-base:edit-image:v1"]
        self.assertIn(
            ("diffusersImageEdit", "image_guidance_scale", "conditioningScale"),
            dreamlite_edit["bindings"],
        )
        self.assertNotIn(
            ("diffusersImageEdit", "max_sequence_length", "maxSequenceLength"),
            dreamlite_edit["bindings"],
        )
        dreamlite_mobile = by_id["dreamlite-mobile:text-to-image:v1"]
        self.assertEqual(dreamlite_mobile["defaultRepo"], "carlofkl/DreamLite-mobile")
        self.assertEqual(dreamlite_mobile["pipelineClass"], "DreamLiteMobilePipeline")
        self.assertFalse(
            any(
                role == "diffusersImageGenerate" and parameter in {"guidance_scale", "negative_prompt"}
                for role, parameter, _source in dreamlite_mobile["bindings"]
            )
        )
        dreamlite_mobile_edit = by_id["dreamlite-mobile:edit-image:v1"]
        self.assertFalse(
            any(
                role == "diffusersImageEdit"
                and parameter
                in {
                    "guidance_scale",
                    "image_guidance_scale",
                    "negative_prompt",
                }
                for role, parameter, _source in dreamlite_mobile_edit["bindings"]
            )
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[dreamlite["executionProfileId"]].live_proof)
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[dreamlite_mobile["executionProfileId"]].live_proof)
        self.assertIn(
            ("loadMask", "image", "diffusersImageInpaint", "mask_image"),
            sdxl_pag_inpaint["edges"],
        )
        lcm = by_id["lcm-dreamshaper-v7:text-to-image:v1"]
        self.assertEqual(lcm["modelType"], "LatentConsistencyModelPipeline")
        self.assertEqual(lcm["pipelineClass"], "LatentConsistencyModelPipeline")
        self.assertEqual(lcm["defaultRepo"], "SimianLuo/LCM_Dreamshaper_v7")
        self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), lcm["bindings"])
        self.assertTrue(DIFFUSERS_EXECUTION_PROFILES[lcm["executionProfileId"]].live_proof)
        lcm_edit = by_id["lcm-dreamshaper-v7:edit-image:v1"]
        self.assertEqual(lcm_edit["modelType"], "LatentConsistencyModelPipeline")
        self.assertEqual(lcm_edit["pipelineClass"], "LatentConsistencyModelImg2ImgPipeline")
        self.assertIn(("loadImage", "image", "diffusersImageEdit", "image"), lcm_edit["edges"])
        self.assertIn(("diffusersImageEdit", "strength", "strength"), lcm_edit["bindings"])
        self.assertNotIn(("diffusersImageEdit", "negative_prompt", "negativePrompt"), lcm_edit["bindings"])
        self.assertNotIn(("diffusersImageEdit", "width", "width"), lcm_edit["bindings"])
        self.assertNotIn(("diffusersImageEdit", "height", "height"), lcm_edit["bindings"])
        self.assertNotIn(
            ("diffusersImageEdit", "max_sequence_length", "maxSequenceLength"),
            lcm_edit["bindings"],
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[lcm_edit["executionProfileId"]].live_proof)
        pag = by_id["sd15-pag:text-to-image:v1"]
        self.assertEqual(pag["modelType"], "StableDiffusionPAGPipeline")
        self.assertEqual(pag["pipelineClass"], "StableDiffusionPAGPipeline")
        self.assertEqual(pag["defaultRepo"], "stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.assertIn(("diffusersImageGenerate", "pag_scale", "pagScale"), pag["bindings"])
        self.assertIn(("diffusersImageGenerate", "pag_adaptive_scale", "pagAdaptiveScale"), pag["bindings"])
        self.assertTrue(DIFFUSERS_EXECUTION_PROFILES[pag["executionProfileId"]].live_proof)
        pag_edit = by_id["sd15-pag:edit-image:v1"]
        self.assertEqual(pag_edit["pipelineClass"], "StableDiffusionPAGImg2ImgPipeline")
        self.assertIn(("diffusersImageEdit", "pag_scale", "pagScale"), pag_edit["bindings"])
        self.assertNotIn(("diffusersImageEdit", "width", "width"), pag_edit["bindings"])
        self.assertNotIn(("diffusersImageEdit", "height", "height"), pag_edit["bindings"])
        self.assertNotIn(
            ("diffusersImageEdit", "max_sequence_length", "maxSequenceLength"),
            pag_edit["bindings"],
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[pag_edit["executionProfileId"]].live_proof)
        pag_inpaint = by_id["sd15-pag:inpaint:v1"]
        self.assertEqual(pag_inpaint["pipelineClass"], "StableDiffusionPAGInpaintPipeline")
        self.assertIn(("loadMask", "image", "diffusersImageInpaint", "mask_image"), pag_inpaint["edges"])
        self.assertIn(("diffusersImageInpaint", "pag_scale", "pagScale"), pag_inpaint["bindings"])
        self.assertNotIn(
            ("diffusersImageInpaint", "reference_strength", "conditioningScale"),
            pag_inpaint["bindings"],
        )
        self.assertNotIn(
            ("diffusersImageInpaint", "max_sequence_length", "maxSequenceLength"),
            pag_inpaint["bindings"],
        )
        self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[pag_inpaint["executionProfileId"]].live_proof)
        sdxl = by_id["sdxl-base:text-to-image:v1"]
        self.assertEqual(sdxl["id"], "sdxl-base:text-to-image:v1")
        self.assertEqual(sdxl["pipelineClass"], "StableDiffusionXLPipeline")
        self.assertEqual(sdxl["defaultRepo"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(sdxl["roles"], specs[0]["roles"])
        self.assertEqual(sdxl["edges"], specs[0]["edges"])
        self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), sdxl["bindings"])
        sdxl_edit = by_id["sdxl-base:edit-image:v1"]
        self.assertEqual(sdxl_edit["id"], "sdxl-base:edit-image:v1")
        self.assertEqual(sdxl_edit["pipelineClass"], "StableDiffusionXLImg2ImgPipeline")
        self.assertEqual(sdxl_edit["defaultRepo"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(sdxl_edit["roles"], specs[5]["roles"])
        self.assertEqual(sdxl_edit["edges"], specs[5]["edges"])
        self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), sdxl_edit["bindings"])
        sdxl_inpaint = by_id["sdxl-base:inpaint:v1"]
        self.assertEqual(sdxl_inpaint["id"], "sdxl-base:inpaint:v1")
        self.assertEqual(sdxl_inpaint["pipelineClass"], "StableDiffusionXLInpaintPipeline")
        self.assertEqual(sdxl_inpaint["defaultRepo"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(sdxl_inpaint["roles"], specs[8]["roles"])
        self.assertEqual(sdxl_inpaint["edges"], specs[8]["edges"])
        self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), sdxl_inpaint["bindings"])
        flux_dev_edit = by_id["flux-dev:edit-image:v1"]
        self.assertEqual(flux_dev_edit["id"], "flux-dev:edit-image:v1")
        self.assertEqual(flux_dev_edit["pipelineClass"], "FluxImg2ImgPipeline")
        self.assertEqual(flux_dev_edit["roles"], specs[5]["roles"])
        self.assertEqual(flux_dev_edit["edges"], specs[5]["edges"])
        flux_dev_inpaint = by_id["flux-dev:inpaint:v1"]
        self.assertEqual(flux_dev_inpaint["id"], "flux-dev:inpaint:v1")
        self.assertEqual(flux_dev_inpaint["pipelineClass"], "FluxInpaintPipeline")
        self.assertEqual(flux_dev_inpaint["roles"], specs[8]["roles"])
        self.assertEqual(flux_dev_inpaint["edges"], specs[8]["edges"])
        self.assertIn(
            ("diffusersImagePipeline", "revision", "defaultRevision"),
            flux_dev_inpaint["bindings"],
        )
        self.assertEqual(specs[0]["roles"], specs[1]["roles"])
        self.assertEqual(specs[0]["edges"], specs[1]["edges"])
        self.assertEqual(specs[0]["bindings"], specs[1]["bindings"])
        self.assertEqual(specs[0]["roles"], specs[2]["roles"])
        self.assertEqual(specs[0]["edges"], specs[2]["edges"])
        self.assertEqual(specs[0]["bindings"], specs[2]["bindings"])
        self.assertEqual(
            [item[0] for item in specs[3]["roles"]],
            [
                "diffusersQuantization",
                "diffusersRecipe",
                "diffusersImagePipeline",
                "loadImage",
                "diffusersImageControl",
                "preview",
            ],
        )
        self.assertIn(("loadImage", "file", "controlImage"), specs[3]["bindings"])
        self.assertIn(
            ("loadImage", "image", "diffusersImageControl", "control_image"),
            specs[3]["edges"],
        )
        self.assertEqual(specs[3]["roles"], specs[4]["roles"])
        self.assertEqual(specs[3]["edges"], specs[4]["edges"])
        self.assertEqual(specs[3]["bindings"], specs[4]["bindings"])
        self.assertEqual(
            [item[0] for item in specs[5]["roles"]],
            [
                "diffusersQuantization",
                "diffusersRecipe",
                "diffusersImagePipeline",
                "loadImage",
                "diffusersImageEdit",
                "preview",
            ],
        )
        self.assertIn(("loadImage", "file", "referenceImages"), specs[5]["bindings"])
        self.assertIn(("diffusersImageEdit", "reference_strength", "conditioningScale"), specs[5]["bindings"])
        self.assertEqual(specs[6]["roles"], specs[5]["roles"])
        self.assertEqual(specs[6]["edges"], specs[5]["edges"])
        self.assertEqual(specs[6]["bindings"], specs[5]["bindings"])
        self.assertEqual(specs[6]["pipelineClass"], "FluxKontextPipeline")
        self.assertEqual(specs[7]["roles"], specs[6]["roles"])
        self.assertEqual(specs[7]["edges"], specs[6]["edges"])
        self.assertEqual(specs[7]["bindings"], specs[6]["bindings"])
        self.assertEqual(specs[7]["pipelineClass"], "FluxKontextPipeline")
        self.assertEqual(
            [item[0] for item in specs[8]["roles"]],
            [
                "diffusersQuantization",
                "diffusersRecipe",
                "diffusersImagePipeline",
                "loadImage",
                "loadMask",
                "diffusersImageInpaint",
                "preview",
            ],
        )
        self.assertIn(("loadMask", "file", "maskImage"), specs[8]["bindings"])
        self.assertIn(("loadMask", "alpha_channel", "removeAlpha"), specs[8]["bindings"])
        self.assertIn(("diffusersImageInpaint", "reference_strength", "conditioningScale"), specs[8]["bindings"])
        self.assertIn(("loadMask", "image", "diffusersImageInpaint", "mask_image"), specs[8]["edges"])
        self.assertEqual(specs[8]["pipelineClass"], "FluxFillPipeline")
        self.assertEqual(specs[9]["roles"], specs[8]["roles"])
        self.assertEqual(specs[9]["edges"], specs[8]["edges"])
        self.assertEqual(specs[9]["bindings"], specs[8]["bindings"])
        self.assertEqual(specs[9]["pipelineClass"], "FluxFillPipeline")
        self.assertNotEqual(specs[9]["contentHash"], specs[8]["contentHash"])
        self.assertEqual(specs[10]["roles"], specs[0]["roles"])
        self.assertEqual(specs[10]["edges"], specs[0]["edges"])
        self.assertEqual(specs[10]["bindings"], specs[0]["bindings"])
        self.assertEqual(specs[10]["pipelineClass"], "Flux2KleinPipeline")
        self.assertEqual(specs[11]["roles"], specs[5]["roles"])
        self.assertEqual(specs[11]["edges"], specs[5]["edges"])
        self.assertEqual(specs[11]["bindings"], specs[5]["bindings"])
        self.assertEqual(specs[11]["pipelineClass"], "Flux2KleinPipeline")
        self.assertEqual(specs[12]["roles"], specs[11]["roles"])
        self.assertEqual(specs[12]["edges"], specs[11]["edges"])
        self.assertEqual(specs[12]["bindings"], specs[11]["bindings"])
        self.assertEqual(specs[12]["pipelineClass"], "Flux2KleinPipeline")
        self.assertEqual(
            [item[0] for item in specs[13]["roles"]],
            ["diffusersQuantization", "diffusersRecipe", "wanPipeline", "wanGenerate", "videoExport", "loadImage"],
        )
        self.assertIn(("diffusersQuantization", "components", "dualQuantizedComponents"), specs[13]["bindings"])
        self.assertIn(("loadImage", "file", "referenceImages"), specs[13]["bindings"])
        self.assertIn(("loadImage", "image", "wanGenerate", "reference_images"), specs[13]["edges"])
        self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), specs[13]["bindings"])
        self.assertEqual(
            DIFFUSERS_EXECUTION_PROFILES[specs[13]["executionProfileId"]].default_quantized_components,
            ("transformer", "transformer_2"),
        )
        self.assertEqual(
            [item[0] for item in specs[14]["roles"]],
            ["diffusersQuantization", "diffusersRecipe", "wanPipeline", "wanGenerate", "videoExport"],
        )
        self.assertIn(("wanGenerate", "scheduler_flow_shift", "shift"), specs[14]["bindings"])
        self.assertIn(("wanGenerate", "video_out", "videoExport", "video"), specs[14]["edges"])
        self.assertEqual(specs[15]["pipelineClass"], "WanPipeline")
        self.assertEqual(specs[15]["roles"], specs[14]["roles"])
        self.assertEqual(specs[15]["edges"], specs[14]["edges"])
        self.assertIn(("wanPipeline", "revision", "defaultRevision"), specs[14]["bindings"])
        self.assertIn(("wanPipeline", "execution_profile_id", "executionProfileId"), specs[14]["bindings"])
        self.assertIn(("wanPipeline", "revision", "empty"), specs[15]["bindings"])
        self.assertNotIn(("wanPipeline", "execution_profile_id", "executionProfileId"), specs[15]["bindings"])
        self.assertEqual(
            [item[0] for item in specs[16]["roles"]],
            [
                "diffusersQuantization",
                "diffusersRecipe",
                "wanPipeline",
                "wanGenerate",
                "videoExport",
                "loadVideo",
                "normalizeVideo",
            ],
        )
        self.assertIn(("loadVideo", "file", "sourceVideo"), specs[16]["bindings"])
        self.assertIn(("loadVideo", "video", "normalizeVideo", "video"), specs[16]["edges"])
        self.assertIn(("normalizeVideo", "output", "wanGenerate", "video"), specs[16]["edges"])
        self.assertEqual(specs[16]["pipelineClass"], "WanVideoToVideoPipeline")
        self.assertEqual(specs[17]["roles"], specs[16]["roles"])
        self.assertEqual(specs[17]["edges"], specs[16]["edges"])
        self.assertEqual(specs[17]["bindings"], specs[16]["bindings"])
        self.assertEqual(specs[17]["pipelineClass"], "WanVideoToVideoPipeline")
        self.assertEqual(specs[18]["roles"], specs[15]["roles"])
        self.assertEqual(specs[18]["edges"], specs[15]["edges"])
        self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), specs[18]["bindings"])
        self.assertIn(("diffusersRecipe", "attention_components", "empty"), specs[18]["bindings"])
        self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), specs[18]["bindings"])
        self.assertEqual(specs[18]["pipelineClass"], "LTXConditionPipeline")
        self.assertEqual(specs[19]["roles"], specs[13]["roles"])
        self.assertEqual(specs[19]["edges"], specs[13]["edges"])
        self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), specs[19]["bindings"])
        self.assertIn(("loadImage", "file", "referenceImages"), specs[19]["bindings"])
        self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), specs[19]["bindings"])
        self.assertEqual(specs[19]["pipelineClass"], "LTXConditionPipeline")
        self.assertEqual(specs[20]["roles"], specs[16]["roles"])
        self.assertEqual(specs[20]["edges"], specs[16]["edges"])
        self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), specs[20]["bindings"])
        self.assertIn(("wanGenerate", "strength", "conditioningScale"), specs[20]["bindings"])
        self.assertIn(("wanGenerate", "denoise_strength", "strength"), specs[20]["bindings"])
        self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), specs[20]["bindings"])
        self.assertEqual(specs[20]["pipelineClass"], "LTXConditionPipeline")
        self.assertEqual(specs[21]["roles"], specs[19]["roles"])
        self.assertEqual(specs[21]["edges"], specs[19]["edges"])
        self.assertEqual(specs[21]["bindings"], specs[19]["bindings"])
        self.assertEqual(specs[21]["pipelineClass"], "LTXConditionPipeline")
        self.assertEqual(
            [item[0] for item in specs[22]["roles"]],
            ["diffusersQuantization", "diffusersRecipe", "audioPipeline", "audioGenerate", "audioExport"],
        )
        self.assertIn(("audioGenerate", "task_type", "text2music"), specs[22]["bindings"])
        self.assertIn(("audioGenerate", "audio", "audioExport", "audio"), specs[22]["edges"])
        self.assertEqual(specs[22]["pipelineClass"], "AceStepPipeline")
        self.assertEqual(
            [item[0] for item in specs[23]["roles"]],
            ["loadAudio", "diffusersQuantization", "diffusersRecipe", "audioPipeline", "audioGenerate", "audioExport"],
        )
        self.assertIn(("loadAudio", "file", "sourceAudio"), specs[23]["bindings"])
        self.assertIn(("audioGenerate", "task_type", "cover"), specs[23]["bindings"])
        self.assertIn(("loadAudio", "audio", "audioGenerate", "source_audio"), specs[23]["edges"])
        self.assertEqual(specs[23]["pipelineClass"], "AceStepPipeline")
        self.assertNotEqual(specs[23]["contentHash"], specs[22]["contentHash"])
        self.assertEqual(
            [item[0] for item in specs[24]["roles"]],
            [
                "loadAudio",
                "diffusersQuantization",
                "diffusersRecipe",
                "audioPipeline",
                "audioGenerate",
                "audioExport",
                "audioLoudnessMatch",
                "audioJoin",
            ],
        )
        self.assertIn(("audioGenerate", "task_type", "continuation"), specs[24]["bindings"])
        self.assertIn(("audioGenerate", "return_continuation_tail", "true"), specs[24]["bindings"])
        self.assertIn(("audioGenerate", "audio", "audioLoudnessMatch", "audio"), specs[24]["edges"])
        self.assertIn(("audioJoin", "output", "audioExport", "audio"), specs[24]["edges"])
        self.assertNotEqual(specs[24]["contentHash"], specs[23]["contentHash"])
        self.assertEqual(specs[25]["roles"], specs[23]["roles"])
        self.assertEqual(specs[25]["edges"], specs[23]["edges"])
        self.assertIn(("audioGenerate", "task_type", "repaint"), specs[25]["bindings"])
        self.assertIn(("loadAudio", "file", "sourceAudio"), specs[25]["bindings"])
        self.assertNotEqual(specs[25]["contentHash"], specs[24]["contentHash"])
        self.assertEqual(
            [item[0] for item in specs[26]["roles"]],
            [
                "diffusersQuantization",
                "diffusersRecipe",
                "diffusersImagePipeline",
                "loadImage",
                "loadMask",
                "diffusersImageInpaint",
                "preview",
            ],
        )
        self.assertEqual(specs[26]["executionProfileId"], "qwen-edit:direct-inpaint")
        self.assertEqual(specs[26]["pipelineClass"], "QwenImageEditInpaintPipeline")
        self.assertIn(("loadImage", "file", "referenceImages"), specs[26]["bindings"])
        self.assertIn(("loadMask", "file", "maskImage"), specs[26]["bindings"])
        self.assertIn(("loadMask", "image", "diffusersImageInpaint", "mask_image"), specs[26]["edges"])
        self.assertEqual(specs[27]["roles"], specs[14]["roles"])
        self.assertEqual(specs[27]["edges"], specs[14]["edges"])
        self.assertEqual(specs[27]["executionProfileId"], "wan-vace:direct")
        self.assertEqual(specs[27]["pipelineClass"], "WanVACEPipeline")
        self.assertIn(("wanPipeline", "revision", "wanVaceRevision"), specs[27]["bindings"])
        self.assertIn(("wanGenerate", "mode", "mode"), specs[27]["bindings"])
        self.assertIn(("wanGenerate", "scheduler_flow_shift", "shift"), specs[27]["bindings"])
        self.assertEqual(
            [item[0] for item in specs[28]["roles"]],
            [
                "diffusersQuantization",
                "diffusersRecipe",
                "wanPipeline",
                "wanGenerate",
                "videoExport",
                "loadVideo",
                "normalizeVideo",
                "loadMaskVideo",
                "alignMaskVideo",
            ],
        )
        self.assertIn(("normalizeVideo", "output", "alignMaskVideo", "video"), specs[28]["edges"])
        self.assertIn(("loadMaskVideo", "video", "alignMaskVideo", "mask"), specs[28]["edges"])
        self.assertIn(("alignMaskVideo", "output", "wanGenerate", "mask"), specs[28]["edges"])
        self.assertIn(("loadMaskVideo", "file", "maskVideo"), specs[28]["bindings"])
        self.assertIn(("alignMaskVideo", "threshold", "maskThreshold127"), specs[28]["bindings"])
        self.assertIn(("alignMaskVideo", "grow_pixels", "inpaintMaskGrow96"), specs[28]["bindings"])
        self.assertEqual(specs[29]["roles"], specs[28]["roles"])
        self.assertEqual(specs[29]["edges"], specs[28]["edges"])
        self.assertIn(("alignMaskVideo", "grow_pixels", "outpaintMaskGrow0"), specs[29]["bindings"])
        self.assertNotIn(("alignMaskVideo", "grow_pixels", "inpaintMaskGrow96"), specs[29]["bindings"])
        self.assertNotEqual(specs[29]["contentHash"], specs[28]["contentHash"])
        self.assertEqual(
            [item[0] for item in specs[30]["roles"]],
            [
                "diffusersQuantization",
                "diffusersRecipe",
                "wanPipeline",
                "wanGenerate",
                "videoExport",
                "loadControlVideo",
                "normalizeVideo",
            ],
        )
        self.assertIn(("loadControlVideo", "video", "normalizeVideo", "video"), specs[30]["edges"])
        self.assertIn(("normalizeVideo", "output", "wanGenerate", "video"), specs[30]["edges"])
        self.assertIn(("loadControlVideo", "file", "controlVideo"), specs[30]["bindings"])
        self.assertIn(("normalizeVideo", "num_frames", "numFrames"), specs[30]["bindings"])
        self.assertNotIn(("loadVideo", "file", "sourceVideo"), specs[30]["bindings"])
        self.assertNotIn(("loadMaskVideo", "file", "maskVideo"), specs[30]["bindings"])
        self.assertEqual(
            [item[0] for item in specs[31]["roles"]],
            [
                "diffusersQuantization",
                "diffusersRecipe",
                "diffusersImagePipeline",
                "loadImage",
                "qwenOutpaintCanvas",
                "diffusersImageInpaint",
                "preview",
            ],
        )
        self.assertIn(("loadImage", "image", "qwenOutpaintCanvas", "image"), specs[31]["edges"])
        self.assertIn(("qwenOutpaintCanvas", "canvas", "diffusersImageInpaint", "image"), specs[31]["edges"])
        self.assertIn(
            ("qwenOutpaintCanvas", "mask_image", "diffusersImageInpaint", "mask_image"),
            specs[31]["edges"],
        )
        self.assertIn(("qwenOutpaintCanvas", "overlap", "outpaintOverlap"), specs[31]["bindings"])
        self.assertNotIn(("loadMask", "file", "maskImage"), specs[31]["bindings"])
        self.assertEqual(specs[0]["actions"], ())
        self.assertRegex(specs[0]["contentHash"], r"^studio-spec-v1-[0-9a-f]{8}$")
        self.assertEqual(specs, validate_studio_execution_specs(module_registry.MODULE_MAP))

        for spec in specs:
            definition = STUDIO_EXECUTION_SPEC_DEFINITIONS[spec["id"]]
            profile = DIFFUSERS_EXECUTION_PROFILES[spec["executionProfileId"]]
            self.assertEqual(profile.default_repo, definition["profile"]["default_repo"])
            self.assertEqual(profile.pipeline_class, spec["pipelineClass"])
            if "capability" in definition:
                self.assertEqual(STUDIO_MODEL_CAPABILITIES[spec["modelType"]], definition["capability"])
            if "autoRequirements" in definition:
                requirements_key = definition.get("autoRequirementKey", spec["modelType"])
                self.assertEqual(AUTO_MODEL_REQUIREMENTS[requirements_key], definition["autoRequirements"])

    def test_combined_control_specs_are_exact_generic_graphs_with_immutable_profiles(self):
        specs = {item["id"]: item for item in validate_studio_execution_specs(module_registry.MODULE_MAP)}
        expected = {
            "sd15-pag-controlnet-canny:control-image:v1": (
                "StableDiffusionPAGPipeline",
                "control_image",
                "StableDiffusionControlNetPAGPipeline",
                "sd15-pag-controlnet-canny:direct",
                True,
            ),
            "sdxl-pag-controlnet-canny:control-image:v1": (
                "StableDiffusionXLPAGPipeline",
                "control_image",
                "StableDiffusionXLControlNetPAGPipeline",
                "sdxl-pag-controlnet-canny:direct",
                True,
            ),
            "sd15-controlnet-canny:control-edit-image:v1": (
                "StableDiffusionPipeline",
                "control_edit_image",
                "StableDiffusionControlNetImg2ImgPipeline",
                "sd15-controlnet-canny:img2img-direct",
                True,
            ),
            "sdxl-controlnet-canny:control-edit-image:v1": (
                "StableDiffusionXLControlNetPipeline",
                "control_edit_image",
                "StableDiffusionXLControlNetImg2ImgPipeline",
                "sdxl-controlnet-canny:img2img-direct",
                True,
            ),
            "sdxl-pag-controlnet-canny:control-edit-image:v1": (
                "StableDiffusionXLPAGPipeline",
                "control_edit_image",
                "StableDiffusionXLControlNetPAGImg2ImgPipeline",
                "sdxl-pag-controlnet-canny:img2img-direct",
                True,
            ),
            "sd15-controlnet-canny:control-inpaint:v1": (
                "StableDiffusionPipeline",
                "control_inpaint",
                "StableDiffusionControlNetInpaintPipeline",
                "sd15-controlnet-canny:inpaint-direct",
                True,
            ),
            "sd15-pag-controlnet-canny:control-inpaint:v1": (
                "StableDiffusionPAGPipeline",
                "control_inpaint",
                "StableDiffusionControlNetPAGInpaintPipeline",
                "sd15-pag-controlnet-canny:inpaint-direct",
                True,
            ),
            "sdxl-controlnet-canny:control-inpaint:v1": (
                "StableDiffusionXLControlNetPipeline",
                "control_inpaint",
                "StableDiffusionXLControlNetInpaintPipeline",
                "sdxl-controlnet-canny:inpaint-direct",
                True,
            ),
            "flux-depth:control-edit-image:v1": (
                "FluxDepthPipeline",
                "control_edit_image",
                "FluxControlImg2ImgPipeline",
                "flux-depth:img2img-direct",
                False,
            ),
            "flux-depth:control-inpaint:v1": (
                "FluxDepthPipeline",
                "control_inpaint",
                "FluxControlInpaintPipeline",
                "flux-depth:inpaint-direct",
                False,
            ),
            "flux-canny:control-edit-image:v1": (
                "FluxCannyPipeline",
                "control_edit_image",
                "FluxControlImg2ImgPipeline",
                "flux-canny:img2img-direct",
                False,
            ),
            "flux-canny:control-inpaint:v1": (
                "FluxCannyPipeline",
                "control_inpaint",
                "FluxControlInpaintPipeline",
                "flux-canny:inpaint-direct",
                False,
            ),
        }
        action_contracts = {
            "control_image": (
                "diffusersImageControl",
                "modules.DiffusersImage.ControlGenerate",
                {"controlImage"},
            ),
            "control_edit_image": (
                "diffusersImageControlEdit",
                "modules.DiffusersImage.ControlEdit",
                {"referenceImages", "controlImage"},
            ),
            "control_inpaint": (
                "diffusersImageControlInpaint",
                "modules.DiffusersImage.ControlInpaint",
                {"referenceImages", "maskImage", "controlImage"},
            ),
        }
        for spec_id, (model_type, mode, pipeline_class, profile_id, auxiliary) in expected.items():
            with self.subTest(spec=spec_id):
                specification = specs[spec_id]
                self.assertEqual(
                    (
                        specification["modelType"],
                        specification["mode"],
                        specification["pipelineClass"],
                        specification["executionProfileId"],
                    ),
                    (model_type, mode, pipeline_class, profile_id),
                )
                role, node_key, required_media = action_contracts[mode]
                roles = {item[0]: item[1] for item in specification["roles"]}
                self.assertEqual(roles[role], node_key)
                binding_sources = {item[2] for item in specification["bindings"]}
                self.assertTrue(required_media.issubset(binding_sources))
                self.assertEqual(
                    studio_execution_spec_for_pair(model_type, mode)["id"],
                    spec_id,
                )
                self.assertFalse(DIFFUSERS_EXECUTION_PROFILES[profile_id].live_proof)
                if auxiliary:
                    self.assertIn("controlPreprocessor", roles)
                    self.assertIn(
                        ("diffusersImagePipeline", "conditioning_model_id", "repo"),
                        specification["bindings"],
                    )
                    self.assertIn(
                        ("controlPreprocessor", "output", role, "control_image"),
                        specification["edges"],
                    )
                else:
                    self.assertNotIn("controlPreprocessor", roles)
                    self.assertNotIn(
                        ("diffusersImagePipeline", "conditioning_model_id", "repo"),
                        specification["bindings"],
                    )
                if "pag-controlnet" in spec_id:
                    self.assertIn((role, "pag_scale", "pagScale"), specification["bindings"])
                    self.assertIn(
                        (role, "pag_adaptive_scale", "pagAdaptiveScale"),
                        specification["bindings"],
                    )

        for model_type, modes in (
            ("StableDiffusionPipeline", {"control_edit_image", "control_inpaint"}),
            ("StableDiffusionPAGPipeline", {"control_image", "control_inpaint"}),
            ("StableDiffusionXLControlNetPipeline", {"control_edit_image", "control_inpaint"}),
            ("StableDiffusionXLPAGPipeline", {"control_image", "control_edit_image"}),
            ("FluxDepthPipeline", {"control_edit_image", "control_inpaint"}),
            ("FluxCannyPipeline", {"control_edit_image", "control_inpaint"}),
        ):
            capability = STUDIO_MODEL_CAPABILITIES[model_type]
            self.assertTrue(modes.issubset(capability["modes"]))
            self.assertTrue(capability["supportsImageInput"])
            self.assertTrue(capability["supportsControlImage"])
            if "control_inpaint" in modes:
                self.assertTrue(capability["supportsMask"])

        for model_type in ("FluxDepthPipeline", "FluxCannyPipeline"):
            self.assertEqual(
                AUTO_MODEL_REQUIREMENTS[model_type]["supportedTasks"],
                ["control_image", "control_edit_image", "control_inpaint"],
            )

    def test_registry_validation_rejects_unknown_nodes_params_handles_and_dangling_edges(self):
        broken_modules = deepcopy(module_registry.MODULE_MAP)
        del broken_modules["modules.DiffusersImage"]["Generate"]
        with self.assertRaisesRegex(ValueError, "unknown node"):
            validate_studio_execution_specs(broken_modules)

        broken_modules = deepcopy(module_registry.MODULE_MAP)
        del broken_modules["modules.DiffusersImage"]["Generate"]["params"]["prompt"]
        with self.assertRaisesRegex(ValueError, "binding"):
            validate_studio_execution_specs(broken_modules)

        broken_modules = deepcopy(module_registry.MODULE_MAP)
        broken_modules["modules.DiffusersImage"]["Generate"]["params"]["pipeline"]["type"] = "audio"
        with self.assertRaisesRegex(ValueError, "incompatible handle"):
            validate_studio_execution_specs(broken_modules)

        import modiff.studio_execution_specs as specs_module

        with patch.object(
            specs_module,
            "_GRAPH_EDGES",
            (*specs_module._GRAPH_EDGES, ("missingRole", "output", "preview", "image")),
        ):
            with self.assertRaisesRegex(ValueError, "edge"):
                validate_studio_execution_specs(module_registry.MODULE_MAP)

        with patch.object(
            specs_module,
            "_GRAPH_BINDINGS",
            (*specs_module._GRAPH_BINDINGS, ("preview", "missing", "prompt")),
        ):
            with self.assertRaisesRegex(ValueError, "binding"):
                validate_studio_execution_specs(module_registry.MODULE_MAP)

        with patch.object(
            specs_module,
            "_GRAPH_BINDINGS",
            (*specs_module._GRAPH_BINDINGS, ("diffusersQuantization", "quantization_config", "empty")),
        ):
            with self.assertRaisesRegex(ValueError, "binding"):
                validate_studio_execution_specs(module_registry.MODULE_MAP)

        with patch.object(specs_module, "_GRAPH_EDGES", specs_module._GRAPH_EDGES[:1]):
            with self.assertRaisesRegex(ValueError, "disconnected"):
                validate_studio_execution_specs(module_registry.MODULE_MAP)

    def test_cosmos3_nano_specs_use_exact_official_nodes_and_remain_unqualified(self):
        expected = {
            "text_to_image": (
                "cosmos3-nano:modular-text-to-image:v1",
                None,
                "modules.Image.Preview",
                ("prompt", "num_frames", "oneFrame"),
            ),
            "text_to_video": (
                "cosmos3-nano:modular-text-to-video:v1",
                None,
                "modules.Video.Export",
                ("prompt", "num_frames", "numFrames"),
            ),
            "image_to_video": (
                "cosmos3-nano:modular-image-to-video:v1",
                "modules.Image.Load",
                "modules.Video.Export",
                ("prompt", "num_frames", "numFrames"),
            ),
            "video_to_video": (
                "cosmos3-nano:modular-video-to-video:v1",
                "modules.Video.Load",
                "modules.Video.Export",
                ("prompt", "num_frames", "numFrames"),
            ),
            "text_to_video_with_audio": (
                "cosmos3-nano:modular-text-to-video-with-audio:v1",
                None,
                "modules.Video.ExportWithAudio",
                ("prompt", "num_frames", "numFrames"),
            ),
            "image_to_video_with_audio": (
                "cosmos3-nano:modular-image-to-video-with-audio:v1",
                "modules.Image.Load",
                "modules.Video.ExportWithAudio",
                ("prompt", "num_frames", "numFrames"),
            ),
            "video_to_video_with_audio": (
                "cosmos3-nano:modular-video-to-video-with-audio:v1",
                "modules.Video.Load",
                "modules.Video.ExportWithAudio",
                ("prompt", "num_frames", "numFrames"),
            ),
        }
        for mode, (spec_id, source, sink, frame_binding) in expected.items():
            with self.subTest(mode=mode):
                spec = studio_execution_spec_for_pair("Cosmos3OmniModularPipeline", mode)
                self.assertIsNotNone(spec)
                self.assertEqual(spec["id"], spec_id)
                self.assertEqual(spec["executionProfileId"], "cosmos3-nano:official-modular-workflow")
                self.assertEqual(spec["defaultRepo"], "nvidia/Cosmos3-Nano")
                self.assertEqual(spec["auxiliaryTerminalRoles"], ("afterDecode",))
                node_keys = ["modules.ModularDiffusers.ModelsLoader"]
                if source:
                    node_keys.append(source)
                node_keys.append("modules.ModularDiffusers.WorkflowCosmos3OmniTextEncode")
                if source:
                    node_keys.append("modules.ModularDiffusers.WorkflowCosmos3OmniVaeEncode")
                node_keys.extend(
                    [
                        "modules.ModularDiffusers.WorkflowCosmos3OmniDenoise",
                        "modules.ModularDiffusers.WorkflowCosmos3OmniDecode",
                        "modules.ModularDiffusers.WorkflowCosmos3OmniAfterDecode",
                        sink,
                    ]
                )
                self.assertEqual([node_key for _role, node_key, _x, _y in spec["roles"]], node_keys)
                self.assertIn(frame_binding, spec["bindings"])
                self.assertIn(
                    ("decode", "state_out", "afterDecode", "state_in"),
                    spec["edges"],
                )
                if source:
                    self.assertIn(("prompt", "state_out", "imageEncode", "state_in"), spec["edges"])
                    self.assertIn(("imageEncode", "state_out", "denoise", "state_in"), spec["edges"])
                if mode.endswith("with_audio"):
                    self.assertIn(("decode", "audio", "videoExport", "audio"), spec["edges"])
                definition = STUDIO_EXECUTION_SPEC_DEFINITIONS[spec_id]
                self.assertEqual(definition["capability"]["qualifiedModes"], [])
                self.assertIs(definition["capability"]["autoEligible"], False)
                self.assertIs(definition["capability"]["templateEligible"], False)
                self.assertIs(definition["capability"]["galleryEligible"], False)
                self.assertEqual(definition["profile"]["supported_offload_modes"], ("none",))
                self.assertEqual(definition["profile"]["quantizable_components"], ())
                graph, hints = executable_graph_for_spec(spec)
                assert_studio_execution_graph(graph, hints)

        import modiff.studio_execution_specs as specs_module

        with patch.dict(
            specs_module.STUDIO_EXECUTION_SPEC_DEFINITIONS[
                "cosmos3-nano:modular-text-to-video:v1"
            ],
            {"auxiliaryTerminalRoles": ("denoise",)},
        ):
            with self.assertRaisesRegex(ValueError, "auxiliary terminal"):
                validate_studio_execution_specs(module_registry.MODULE_MAP)

    def test_runtime_receipt_binds_graph_profile_and_topology(self):
        spec = studio_execution_spec_for_pair("FluxKreaPipeline", "text_to_image")
        self.assertIsNotNone(spec)
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

        selected = {
            "executionProfileId": spec["executionProfileId"],
            "studioExecutionSpecContract": {
                "schemaVersion": spec["schemaVersion"],
                "id": spec["id"],
                "contentHash": spec["contentHash"],
                "executionProfileId": spec["executionProfileId"],
            },
        }
        hints["autoResourcePlan"] = selected
        assert_studio_execution_graph(graph, hints)

        missing_receipt = dict(hints)
        missing_receipt.pop("studioExecutionSpec")
        with self.assertRaisesRegex(RuntimeError, "receipt is required"):
            assert_studio_execution_graph(graph, missing_receipt)

        selected["studioExecutionSpecContract"] = {
            **selected["studioExecutionSpecContract"],
            "contentHash": "studio-spec-v1-00000000",
        }
        with self.assertRaisesRegex(RuntimeError, "Auto graph contract"):
            assert_studio_execution_graph(graph, hints)
        selected["studioExecutionSpecContract"]["contentHash"] = spec["contentHash"]

        selected["executionProfileId"] = "flux-dev:direct"
        with self.assertRaisesRegex(RuntimeError, "Auto profile"):
            assert_studio_execution_graph(graph, hints)

        selected["executionProfileId"] = spec["executionProfileId"]
        graph["nodes"][hints["studioExecutionSpec"]["nodes"]["preview"]]["params"]["image"].pop("sourceId")
        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

    def test_runtime_receipt_accepts_only_the_exact_reviewed_video_upscale_edge_replacement(self):
        spec = studio_execution_spec_for_pair("WanVideoPipeline", "text_to_video")
        self.assertIsNotNone(spec)
        graph, hints = executable_graph_for_spec(spec)
        receipt_nodes = hints["studioExecutionSpec"]["nodes"]
        source_role, source_handle, target_role, target_handle = next(
            edge for edge in spec["edges"] if edge[2] == "videoExport"
        )
        source_id = receipt_nodes[source_role]
        target = graph["nodes"][receipt_nodes[target_role]]
        upscaler_id = "controlled-video-upscaler"
        graph["nodes"][upscaler_id] = {
            "module": "modules.Spandrel",
            "action": "Upscaler",
            "params": {
                "image": {"sourceId": source_id, "sourceKey": source_handle},
                "output": {},
            },
        }
        graph["paths"][0].append(upscaler_id)
        target["params"][target_handle].update({"sourceId": upscaler_id, "sourceKey": "output"})

        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

        hints["controlledGraphContracts"] = ["upscale.video.v1"]
        assert_studio_execution_graph(graph, hints)

        graph["nodes"][upscaler_id]["params"]["image"]["sourceKey"] = "wrong_output"
        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

    def test_runtime_receipt_accepts_only_the_exact_reviewed_diffusers_image_lora_edge_replacement(self):
        spec = studio_execution_spec_for_pair("ZImageModularPipeline", "text_to_image")
        self.assertIsNotNone(spec)
        graph, hints = executable_graph_for_spec(spec)
        receipt_nodes = hints["studioExecutionSpec"]["nodes"]
        source_role, source_handle, target_role, target_handle = next(
            edge for edge in spec["edges"] if edge[2] == "diffusersImageGenerate"
        )
        source_id = receipt_nodes[source_role]
        target = graph["nodes"][receipt_nodes[target_role]]
        adapter_id = "controlled-image-lora"
        graph["nodes"][adapter_id] = {
            "module": "modules.DiffusersImage",
            "action": "LoadAdapter",
            "params": {
                "pipeline": {"sourceId": source_id, "sourceKey": source_handle},
                "output": {},
            },
        }
        graph["paths"][0].append(adapter_id)
        target["params"][target_handle].update({"sourceId": adapter_id, "sourceKey": "output"})

        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

        hints["controlledGraphContracts"] = ["lora.diffusers-image.v1"]
        assert_studio_execution_graph(graph, hints)

        target["params"][target_handle]["sourceKey"] = "wrong_output"
        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

    def test_runtime_receipt_accepts_only_the_exact_reviewed_diffusers_audio_lora_edge_replacement(self):
        spec = studio_execution_spec_for_pair("AceStepAudioPipeline", "text_to_audio")
        self.assertIsNotNone(spec)
        graph, hints = executable_graph_for_spec(spec)
        receipt_nodes = hints["studioExecutionSpec"]["nodes"]
        source_role, source_handle, target_role, target_handle = next(
            edge for edge in spec["edges"] if edge[2] == "audioGenerate"
        )
        source_id = receipt_nodes[source_role]
        target = graph["nodes"][receipt_nodes[target_role]]
        adapter_id = "controlled-audio-lora"
        graph["nodes"][adapter_id] = {
            "module": "modules.DiffusersAudio",
            "action": "LoadAdapter",
            "params": {
                "pipeline": {"sourceId": source_id, "sourceKey": source_handle},
                "output": {},
            },
        }
        graph["paths"][0].append(adapter_id)
        target["params"][target_handle].update({"sourceId": adapter_id, "sourceKey": "output"})

        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

        hints["controlledGraphContracts"] = ["lora.diffusers-audio.v1"]
        assert_studio_execution_graph(graph, hints)

        graph["nodes"][adapter_id]["params"]["pipeline"]["sourceKey"] = "wrong_output"
        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

    def test_runtime_receipt_accepts_an_exact_bounded_diffusers_image_lora_chain(self):
        spec = studio_execution_spec_for_pair("FluxDevPipeline", "text_to_image")
        self.assertIsNotNone(spec)
        graph, hints = executable_graph_for_spec(spec)
        receipt_nodes = hints["studioExecutionSpec"]["nodes"]
        source_role, source_handle, target_role, target_handle = next(
            edge for edge in spec["edges"] if edge[2] == "diffusersImageGenerate"
        )
        source_id = receipt_nodes[source_role]
        target = graph["nodes"][receipt_nodes[target_role]]
        for index in range(2):
            adapter_id = f"controlled-image-lora-{index}"
            graph["nodes"][adapter_id] = {
                "module": "modules.DiffusersImage",
                "action": "LoadAdapter",
                "params": {
                    "pipeline": {
                        "sourceId": source_id if index == 0 else f"controlled-image-lora-{index - 1}",
                        "sourceKey": source_handle if index == 0 else "output",
                    },
                    "output": {},
                },
            }
            graph["paths"][0].append(adapter_id)
        target["params"][target_handle].update({"sourceId": "controlled-image-lora-1", "sourceKey": "output"})
        hints["controlledGraphContracts"] = ["lora.diffusers-image.v1"]
        assert_studio_execution_graph(graph, hints)

        graph["nodes"]["controlled-image-lora-1"]["params"]["pipeline"]["sourceId"] = source_id
        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

    def test_z_image_auto_seals_the_exact_direct_image_route(self):
        spec = studio_execution_spec_for_pair("ZImageModularPipeline", "text_to_image")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["id"], "z-image:text-to-image:v1")
        self.assertEqual(spec["executionProfileId"], "z-image:auto")
        self.assertEqual(spec["loaderModule"], "modules.DiffusersImage")
        self.assertEqual(spec["loaderAction"], "LoadPipeline")
        self.assertEqual(spec["executionPath"], "direct-diffusers-image")
        self.assertEqual(spec["pipelineClass"], "ZImagePipeline")
        self.assertEqual(spec["defaultRepo"], "Tongyi-MAI/Z-Image-Turbo")
        self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), spec["bindings"])
        self.assertEqual(spec["roles"], validate_studio_execution_specs(module_registry.MODULE_MAP)[0]["roles"])
        self.assertEqual(spec["edges"], validate_studio_execution_specs(module_registry.MODULE_MAP)[0]["edges"])
        registry_bindings = validate_studio_execution_specs(module_registry.MODULE_MAP)[0]["bindings"]
        self.assertEqual(
            tuple(item for item in spec["bindings"] if item[:2] != ("diffusersRecipe", "attention_backend")),
            tuple(item for item in registry_bindings if item[:2] != ("diffusersRecipe", "attention_backend")),
        )
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

        graph["nodes"][hints["studioExecutionSpec"]["nodes"]["diffusersImagePipeline"]]["action"] = "Edit"
        hints["autoResourcePlan"] = {
            "executionProfileId": spec["executionProfileId"],
            "studioExecutionSpecContract": {
                "schemaVersion": spec["schemaVersion"],
                "id": spec["id"],
                "contentHash": spec["contentHash"],
                "executionProfileId": spec["executionProfileId"],
            },
        }
        with self.assertRaisesRegex(RuntimeError, "node identity"):
            assert_studio_execution_graph(graph, hints)

        edit = studio_execution_spec_for_pair("ZImageModularPipeline", "edit_image")
        self.assertIsNotNone(edit)
        self.assertEqual(edit["id"], "z-image:edit-image:v1")
        self.assertEqual(edit["executionProfileId"], "z-image:img2img-direct")
        self.assertEqual(edit["pipelineClass"], "ZImageImg2ImgPipeline")
        self.assertEqual(edit["defaultRepo"], "Tongyi-MAI/Z-Image-Turbo")
        self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), edit["bindings"])
        self.assertEqual(edit["roles"], validate_studio_execution_specs(module_registry.MODULE_MAP)[5]["roles"])
        self.assertEqual(edit["edges"], validate_studio_execution_specs(module_registry.MODULE_MAP)[5]["edges"])
        self.assertIn(("loadImage", "file", "referenceImages"), edit["bindings"])
        self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), edit["bindings"])

        inpaint = studio_execution_spec_for_pair("QwenImageModularPipeline", "inpaint")
        self.assertIsNotNone(inpaint)
        self.assertEqual(inpaint["id"], "qwen-image-2512:inpaint:v1")
        self.assertEqual(inpaint["executionProfileId"], "qwen-image:inpaint-direct")
        self.assertEqual(inpaint["pipelineClass"], "QwenImageInpaintPipeline")
        self.assertEqual(inpaint["roles"], validate_studio_execution_specs(module_registry.MODULE_MAP)[8]["roles"])
        self.assertEqual(inpaint["edges"], validate_studio_execution_specs(module_registry.MODULE_MAP)[8]["edges"])
        self.assertIn(("loadImage", "file", "referenceImages"), inpaint["bindings"])
        self.assertIn(("loadMask", "file", "maskImage"), inpaint["bindings"])
        self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), inpaint["bindings"])

    def test_qwen_image_text_to_image_seals_the_exact_direct_image_route(self):
        spec = studio_execution_spec_for_pair("QwenImageModularPipeline", "text_to_image")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["id"], "qwen-image-2512:text-to-image:v1")
        self.assertEqual(spec["executionProfileId"], "qwen-image:t2i-direct")
        self.assertEqual(spec["loaderModule"], "modules.DiffusersImage")
        self.assertEqual(spec["loaderAction"], "LoadPipeline")
        self.assertEqual(spec["executionPath"], "direct-diffusers-image")
        self.assertEqual(spec["pipelineClass"], "QwenImagePipeline")
        self.assertEqual(spec["defaultRepo"], "Qwen/Qwen-Image-2512")
        z_image = studio_execution_spec_for_pair("ZImageModularPipeline", "text_to_image")
        self.assertEqual(spec["roles"], z_image["roles"])
        self.assertEqual(spec["edges"], z_image["edges"])
        self.assertEqual(
            tuple(item for item in spec["bindings"] if item[:2] != ("diffusersRecipe", "attention_backend")),
            tuple(item for item in z_image["bindings"] if item[:2] != ("diffusersRecipe", "attention_backend")),
        )
        self.assertIn(("diffusersRecipe", "attention_backend", "attentionBackend"), spec["bindings"])
        self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), z_image["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

        edit = studio_execution_spec_for_pair("QwenImageModularPipeline", "edit_image")
        self.assertIsNotNone(edit)
        self.assertEqual(edit["id"], "qwen-image-2512:edit-image:v1")
        self.assertEqual(edit["executionProfileId"], "qwen-image:img2img-direct")
        self.assertEqual(edit["pipelineClass"], "QwenImageImg2ImgPipeline")
        self.assertEqual(edit["defaultRepo"], "Qwen/Qwen-Image-2512")
        self.assertEqual(edit["roles"], validate_studio_execution_specs(module_registry.MODULE_MAP)[5]["roles"])
        self.assertEqual(edit["edges"], validate_studio_execution_specs(module_registry.MODULE_MAP)[5]["edges"])
        self.assertIn(("loadImage", "file", "referenceImages"), edit["bindings"])
        self.assertIn(("diffusersImagePipeline", "revision", "defaultRevision"), edit["bindings"])

    def test_wan_modes_have_exact_receipts_and_v2v_modes_share_the_reviewed_recipe(self):
        text = studio_execution_spec_for_pair("WanVideoPipeline", "text_to_video")
        video = studio_execution_spec_for_pair("WanVideoPipeline", "video_to_video")
        color = studio_execution_spec_for_pair("WanVideoPipeline", "video_color_edit")
        self.assertIsNotNone(text)
        self.assertIsNotNone(video)
        self.assertIsNotNone(color)
        self.assertEqual(text["pipelineClass"], "WanPipeline")
        self.assertEqual(video["pipelineClass"], "WanVideoToVideoPipeline")
        self.assertEqual(color["pipelineClass"], "WanVideoToVideoPipeline")
        self.assertEqual(color["roles"], video["roles"])
        self.assertEqual(color["edges"], video["edges"])
        self.assertEqual(color["bindings"], video["bindings"])
        self.assertNotEqual(color["contentHash"], video["contentHash"])
        for spec in (text, video, color):
            graph, hints = executable_graph_for_spec(spec)
            assert_studio_execution_graph(graph, hints)

    def test_all_ltx_modes_seal_portable_attention_with_exact_receipts(self):
        for mode in ("text_to_video", "image_to_video", "video_to_video", "reference_to_video"):
            spec = studio_execution_spec_for_pair("LTXVideoPipeline", mode)
            self.assertIsNotNone(spec)
            self.assertEqual(spec["pipelineClass"], "LTXConditionPipeline")
            graph, hints = executable_graph_for_spec(spec)
            assert_studio_execution_graph(graph, hints)

    def test_video_specs_bind_only_controls_consumed_by_each_adapter(self):
        guidance_only_pairs = {
            ("LTXVideoPipeline", "text_to_video"),
            ("LTXVideoPipeline", "image_to_video"),
            ("LTXVideoPipeline", "video_to_video"),
            ("LTXVideoPipeline", "reference_to_video"),
            ("WanImageToVideoPipeline", "image_to_video"),
            ("WanTI2VPipeline", "text_to_video"),
            ("WanVACEPipeline", "text_to_video"),
            ("WanVACEPipeline", "video_inpaint"),
            ("WanVACEPipeline", "video_outpaint"),
            ("WanVACEPipeline", "control_to_video"),
            ("WanVideoPipeline", "text_to_video"),
            ("WanVideoPipeline", "video_to_video"),
            ("WanVideoPipeline", "video_color_edit"),
        }
        for pair in guidance_only_pairs:
            with self.subTest(pair=pair, control="true_cfg_scale"):
                spec = studio_execution_spec_for_pair(*pair)
                self.assertIn(("wanGenerate", "guidance_scale", "guidanceScale"), spec["bindings"])
                self.assertFalse(
                    any(
                        role == "wanGenerate" and param == "true_cfg_scale"
                        for role, param, _source in spec["bindings"]
                    )
                )

        framepack = studio_execution_spec_for_pair("HunyuanVideoFramepackPipeline", "image_to_video")
        self.assertIn(("wanGenerate", "true_cfg_scale", "trueCfgScale1"), framepack["bindings"])

        no_flow_shift_pairs = {
            ("LTXVideoPipeline", "text_to_video"),
            ("LTXVideoPipeline", "image_to_video"),
            ("LTXVideoPipeline", "video_to_video"),
            ("LTXVideoPipeline", "reference_to_video"),
            ("WanImageToVideoPipeline", "image_to_video"),
        }
        for pair in no_flow_shift_pairs:
            with self.subTest(pair=pair, control="scheduler_flow_shift"):
                spec = studio_execution_spec_for_pair(*pair)
                self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), spec["bindings"])
        wan_ti2v = studio_execution_spec_for_pair("WanTI2VPipeline", "text_to_video")
        self.assertIn(("wanGenerate", "scheduler_flow_shift", "shift"), wan_ti2v["bindings"])

        native_flash_pairs = guidance_only_pairs - {
            ("LTXVideoPipeline", "text_to_video"),
            ("LTXVideoPipeline", "image_to_video"),
            ("LTXVideoPipeline", "video_to_video"),
            ("LTXVideoPipeline", "reference_to_video"),
        }
        for pair in native_flash_pairs:
            with self.subTest(pair=pair, control="attention_backend"):
                spec = studio_execution_spec_for_pair(*pair)
                self.assertIn(("diffusersRecipe", "attention_backend", "nativeFlashAttention"), spec["bindings"])
        for mode in ("text_to_video", "image_to_video", "video_to_video", "reference_to_video"):
            ltx = studio_execution_spec_for_pair("LTXVideoPipeline", mode)
            self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), ltx["bindings"])

    def test_all_ace_modes_seal_exact_generic_audio_routes(self):
        text = studio_execution_spec_for_pair("AceStepAudioPipeline", "text_to_audio")
        variation = studio_execution_spec_for_pair("AceStepAudioPipeline", "audio_variation")
        continuation = studio_execution_spec_for_pair("AceStepAudioPipeline", "audio_continuation")
        repaint = studio_execution_spec_for_pair("AceStepAudioPipeline", "audio_repaint")
        self.assertIsNotNone(text)
        self.assertIsNotNone(variation)
        self.assertIsNotNone(continuation)
        self.assertIsNotNone(repaint)
        self.assertEqual(text["executionProfileId"], "ace-step-audio:direct")
        self.assertEqual(variation["executionProfileId"], "ace-step-audio:direct")
        self.assertEqual(text["pipelineClass"], "AceStepPipeline")
        self.assertEqual(variation["pipelineClass"], "AceStepPipeline")
        self.assertEqual(text["defaultRepo"], "ACE-Step/acestep-v15-xl-turbo-diffusers")
        self.assertIn(("audioGenerate", "task_type", "text2music"), text["bindings"])
        self.assertIn(("audioGenerate", "task_type", "cover"), variation["bindings"])
        self.assertIn(("loadAudio", "file", "sourceAudio"), variation["bindings"])
        self.assertIn(("loadAudio", "audio", "audioGenerate", "source_audio"), variation["edges"])
        self.assertIn(("audioGenerate", "task_type", "continuation"), continuation["bindings"])
        self.assertIn(
            ("audioLoudnessMatch", "reference_window_seconds", "referenceWindow15"), continuation["bindings"]
        )
        self.assertIn(("audioJoin", "boundary_fade_seconds", "boundaryFade001"), continuation["bindings"])
        self.assertIn(("loadAudio", "audio", "audioJoin", "source"), continuation["edges"])
        self.assertIn(("audioGenerate", "task_type", "repaint"), repaint["bindings"])
        self.assertIn(("loadAudio", "file", "sourceAudio"), repaint["bindings"])
        self.assertEqual(repaint["roles"], variation["roles"])
        self.assertEqual(repaint["edges"], variation["edges"])
        for spec in (text, variation, continuation, repaint):
            self.assertIn(("audioGenerate", "sample_rate", "sampleRate48000"), spec["bindings"])
            self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), spec["bindings"])
            graph, hints = executable_graph_for_spec(spec)
            assert_studio_execution_graph(graph, hints)

    def test_stable_audio_seals_exact_pinned_generic_audio_route(self):
        spec = studio_execution_spec_for_pair("StableAudioPipeline", "text_to_audio")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "stable-audio:direct")
        self.assertEqual(spec["pipelineClass"], "StableAudioPipeline")
        self.assertEqual(spec["defaultRepo"], "stabilityai/stable-audio-open-1.0")
        self.assertIn(("audioPipeline", "revision", "defaultRevision"), spec["bindings"])
        self.assertIn(("audioGenerate", "task_type", "text2audio"), spec["bindings"])
        self.assertIn(("audioGenerate", "stable_audio_steps", "steps"), spec["bindings"])
        self.assertIn(("audioGenerate", "stable_audio_guidance", "guidanceScale"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_minimax_music_seals_official_modular_workflow_route(self):
        spec = studio_execution_spec_for_pair("MiniMaxMusic3ModularPipeline", "text_to_audio")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "minimax-music3:official-modular-workflow")
        self.assertEqual(spec["pipelineClass"], "MiniMaxMusic3ModularPipeline")
        self.assertEqual(spec["defaultRepo"], "MiniMaxAI/MiniMax-Music3")
        self.assertEqual(
            [node_key for _role, node_key, _x, _y in spec["roles"]],
            [
                "modules.ModularDiffusers.ModelsLoader",
                "modules.ModularDiffusers.WorkflowSemanticGeneration",
                "modules.ModularDiffusers.WorkflowDenoise",
                "modules.ModularDiffusers.WorkflowDecodeAudio",
                "modules.Audio.Export",
            ],
        )
        self.assertIn(("prompt", "state_out", "denoise", "state_in"), spec["edges"])
        self.assertIn(("denoise", "state_out", "decode", "state_in"), spec["edges"])
        self.assertIn(("audioExport", "sample_rate", "sampleRate44100"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_anima_seals_exact_official_modular_text_and_image_workflows(self):
        text = studio_execution_spec_for_pair("AnimaModularPipeline", "text_to_image")
        image = studio_execution_spec_for_pair("AnimaModularPipeline", "image_to_image")
        self.assertIsNotNone(text)
        self.assertIsNotNone(image)
        for spec in (text, image):
            self.assertEqual(spec["executionProfileId"], "anima:official-modular-workflow")
            self.assertEqual(spec["pipelineClass"], "AnimaModularPipeline")
            self.assertEqual(spec["defaultRepo"], "circlestone-labs/Anima-Base-v1.0-Diffusers")
            self.assertIn(("denoise", "state_out", "decode", "state_in"), spec["edges"])
            graph, hints = executable_graph_for_spec(spec)
            assert_studio_execution_graph(graph, hints)
        self.assertIn(("prompt", "state_out", "denoise", "state_in"), text["edges"])
        self.assertEqual(
            [node_key for _role, node_key, _x, _y in text["roles"]],
            [
                "modules.ModularDiffusers.ModelsLoader",
                "modules.ModularDiffusers.WorkflowTextEncode",
                "modules.ModularDiffusers.WorkflowImageDenoise",
                "modules.ModularDiffusers.WorkflowDecodeImage",
                "modules.Image.Preview",
            ],
        )
        self.assertIn(("loadImage", "image", "imageEncode", "image"), image["edges"])
        self.assertIn(("prompt", "state_out", "imageEncode", "state_in"), image["edges"])
        self.assertIn(("imageEncode", "state_out", "denoise", "state_in"), image["edges"])
        self.assertIn(("loadImage", "file", "referenceImages"), image["bindings"])

    def test_longcat_and_audioldm2_seal_exact_pinned_generic_audio_routes(self):
        cases = (
            (
                "LongCatAudioDiTPipeline",
                "longcat-audio-dit-1b:direct",
                "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
                "sampleRate24000",
                None,
            ),
            (
                "AudioLDM2Pipeline",
                "audioldm2-base:direct",
                "cvssp/audioldm2",
                "sampleRate16000",
                "numWaveforms3",
            ),
        )
        for model_type, profile_id, repo, sample_rate, waveforms in cases:
            with self.subTest(model_type=model_type):
                spec = studio_execution_spec_for_pair(model_type, "text_to_audio")
                self.assertIsNotNone(spec)
                self.assertEqual(spec["executionProfileId"], profile_id)
                self.assertEqual(spec["pipelineClass"], model_type)
                self.assertEqual(spec["defaultRepo"], repo)
                self.assertIn(("audioPipeline", "revision", "defaultRevision"), spec["bindings"])
                self.assertIn(("audioGenerate", "task_type", "text2audio"), spec["bindings"])
                self.assertIn(("audioGenerate", "sample_rate", sample_rate), spec["bindings"])
                self.assertIn(("audioExport", "sample_rate", sample_rate), spec["bindings"])
                if model_type == "LongCatAudioDiTPipeline":
                    self.assertIn(("diffusersRecipe", "vae_slicing", "false"), spec["bindings"])
                    self.assertIn(("diffusersRecipe", "vae_tiling", "false"), spec["bindings"])
                    self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), spec["bindings"])
                else:
                    self.assertIn(("diffusersRecipe", "vae_slicing", "true"), spec["bindings"])
                    self.assertIn(("diffusersRecipe", "vae_tiling", "true"), spec["bindings"])
                if waveforms is None:
                    self.assertFalse(any(param == "num_waveforms" for _role, param, _source in spec["bindings"]))
                else:
                    self.assertIn(("audioGenerate", "num_waveforms", waveforms), spec["bindings"])
                graph, hints = executable_graph_for_spec(spec)
                assert_studio_execution_graph(graph, hints)

    def test_shap_e_seals_exact_safe_rendered_orbit_route(self):
        spec = studio_execution_spec_for_pair("ShapEPipeline", "text_to_3d")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "shap-e:direct")
        self.assertEqual(spec["executionPath"], "direct-diffusers-three-d")
        self.assertEqual(spec["defaultRepo"], "openai/shap-e")
        self.assertIn(("diffusersThreeDPipeline", "revision", "defaultRevision"), spec["bindings"])
        self.assertIn(("diffusersThreeDGenerate", "frame_size", "width"), spec["bindings"])
        self.assertIn(("diffusersThreeDGenerate", "video", "videoExport", "video"), spec["edges"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

        image_spec = studio_execution_spec_for_pair("ShapEImg2ImgPipeline", "image_to_3d")
        self.assertIsNotNone(image_spec)
        self.assertEqual(image_spec["executionProfileId"], "shap-e-img2img:direct")
        self.assertEqual(image_spec["defaultRepo"], "openai/shap-e-img2img")
        self.assertIn(
            ("loadImage", "image", "diffusersThreeDGenerate", "reference_images"),
            image_spec["edges"],
        )
        self.assertIn(("loadImage", "file", "referenceImages"), image_spec["bindings"])
        self.assertFalse(
            any(
                role == "diffusersThreeDGenerate" and param == "prompt"
                for role, param, _source in image_spec["bindings"]
            )
        )
        graph, hints = executable_graph_for_spec(image_spec)
        assert_studio_execution_graph(graph, hints)

    def test_stable_video_diffusion_seals_exact_safe_image_to_video_route(self):
        spec = studio_execution_spec_for_pair("StableVideoDiffusionPipeline", "image_to_video")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "stable-video-diffusion:direct")
        self.assertEqual(spec["executionPath"], "direct-diffusers-video")
        self.assertEqual(spec["defaultRepo"], "stabilityai/stable-video-diffusion-img2vid-xt-1-1")
        self.assertIn(("wanPipeline", "revision", "defaultRevision"), spec["bindings"])
        self.assertIn(("diffusersQuantization", "components", "empty"), spec["bindings"])
        self.assertIn(("diffusersRecipe", "attention_backend", "nativeMath"), spec["bindings"])
        self.assertIn(("wanGenerate", "prompt", "empty"), spec["bindings"])
        self.assertIn(("wanGenerate", "negative_prompt", "empty"), spec["bindings"])
        self.assertIn(("loadImage", "image", "wanGenerate", "reference_images"), spec["edges"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_cogvideox_seals_exact_safe_short_text_to_video_route(self):
        spec = studio_execution_spec_for_pair("CogVideoXPipeline", "text_to_video")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "cogvideox-2b:direct")
        self.assertEqual(spec["defaultRepo"], "zai-org/CogVideoX-2b")
        self.assertIn(("wanPipeline", "revision", "defaultRevision"), spec["bindings"])
        self.assertIn(("diffusersQuantization", "components", "empty"), spec["bindings"])
        self.assertIn(("diffusersRecipe", "vae_tiling", "false"), spec["bindings"])
        self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_extended_animatediff_and_cogvideox_routes_are_separate_expert_graphs(self):
        cases = (
            (
                "AnimateDiffPAGPipeline",
                "text_to_video",
                "animatediff-pag:text-to-video:v1",
                "animatediff-sd15-v2-pag:direct",
                (),
            ),
            (
                "AnimateDiffVideoToVideoPipeline",
                "video_to_video",
                "animatediff-video-to-video:video-to-video:v1",
                "animatediff-sd15-v2-video-to-video:direct",
                ("sourceVideo",),
            ),
            (
                "AnimateDiffControlNetPipeline",
                "control_to_video",
                "animatediff-controlnet:control-to-video:v1",
                "animatediff-sd15-v2-controlnet:direct",
                ("controlVideo",),
            ),
            (
                "AnimateDiffVideoToVideoControlNetPipeline",
                "control_video_to_video",
                "animatediff-controlnet-video-to-video:control-video-to-video:v1",
                "animatediff-sd15-v2-controlnet-video-to-video:direct",
                ("sourceVideo", "controlVideo"),
            ),
            (
                "CogVideoXVideoToVideoPipeline",
                "video_to_video",
                "cogvideox-2b-video-to-video:video-to-video:v1",
                "cogvideox-2b-video-to-video:direct",
                ("sourceVideo",),
            ),
        )
        for model_type, mode, spec_id, profile_id, required_videos in cases:
            with self.subTest(pair=(model_type, mode)):
                spec = studio_execution_spec_for_pair(model_type, mode)
                self.assertEqual(spec["id"], spec_id)
                self.assertEqual(spec["pipelineClass"], model_type)
                self.assertEqual(spec["executionProfileId"], profile_id)
                capability = STUDIO_MODEL_CAPABILITIES[model_type]
                self.assertEqual(
                    capability["modeRequirements"][mode].get(
                        "requiredVideos",
                        [],
                    ),
                    list(required_videos),
                )
                self.assertEqual(capability["executionStatus"], "expert_only")
                self.assertEqual(
                    capability["qualificationStatus"],
                    "graph-qualified-execution-pending",
                )
                self.assertEqual(capability["qualifiedModes"], [])
                self.assertFalse(capability["autoEligible"])
                self.assertFalse(capability["galleryEligible"])
                self.assertFalse(capability["liveProof"])
                graph, hints = executable_graph_for_spec(spec)
                assert_studio_execution_graph(graph, hints)

        pag = studio_execution_spec_for_pair(
            "AnimateDiffPAGPipeline",
            "text_to_video",
        )
        self.assertIn(("wanGenerate", "pag_scale", "pagScale"), pag["bindings"])
        self.assertIn(
            ("wanGenerate", "pag_adaptive_scale", "pagAdaptiveScale"),
            pag["bindings"],
        )
        control = studio_execution_spec_for_pair(
            "AnimateDiffControlNetPipeline",
            "control_to_video",
        )
        combined = studio_execution_spec_for_pair(
            "AnimateDiffVideoToVideoControlNetPipeline",
            "control_video_to_video",
        )
        for spec in (control, combined):
            self.assertIn(
                (
                    "controlPreprocessor",
                    "output",
                    "wanGenerate",
                    "control_video",
                ),
                spec["edges"],
            )
            self.assertIn(
                (
                    "controlPreprocessor",
                    "low_threshold",
                    "videoCannyLowThreshold100",
                ),
                spec["bindings"],
            )
            self.assertIn(
                (
                    "controlPreprocessor",
                    "high_threshold",
                    "videoCannyHighThreshold200",
                ),
                spec["bindings"],
            )
            self.assertIn(
                ("wanPipeline", "motion_adapter_id", "motionAdapterRepo"),
                spec["bindings"],
            )
            self.assertIn(
                (
                    "wanPipeline",
                    "motion_adapter_revision",
                    "motionAdapterRevision",
                ),
                spec["bindings"],
            )
        self.assertNotIn(
            ("normalizeVideo", "output", "wanGenerate", "video"),
            control["edges"],
        )
        self.assertIn(
            ("normalizeVideo", "output", "wanGenerate", "video"),
            combined["edges"],
        )

    def test_allegro_seals_exact_safe_native_text_to_video_route(self):
        spec = studio_execution_spec_for_pair("AllegroPipeline", "text_to_video")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "allegro:direct")
        self.assertEqual(spec["defaultRepo"], "rhymes-ai/Allegro")
        self.assertIn(("wanPipeline", "revision", "defaultRevision"), spec["bindings"])
        self.assertIn(("diffusersQuantization", "components", "empty"), spec["bindings"])
        self.assertIn(("diffusersRecipe", "vae_tiling", "false"), spec["bindings"])
        self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_latte_seals_exact_safe_native_text_to_video_route(self):
        spec = studio_execution_spec_for_pair("LattePipeline", "text_to_video")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "latte:direct")
        self.assertEqual(spec["defaultRepo"], "maxin-cn/Latte-1")
        self.assertIn(("wanPipeline", "revision", "defaultRevision"), spec["bindings"])
        self.assertIn(("diffusersQuantization", "components", "empty"), spec["bindings"])
        self.assertIn(("diffusersRecipe", "vae_tiling", "false"), spec["bindings"])
        self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_mochi_seals_exact_safe_native_text_to_video_route(self):
        spec = studio_execution_spec_for_pair("MochiPipeline", "text_to_video")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "mochi:direct")
        self.assertEqual(spec["defaultRepo"], "genmo/mochi-1-preview")
        self.assertIn(("wanPipeline", "revision", "defaultRevision"), spec["bindings"])
        self.assertIn(("diffusersQuantization", "components", "empty"), spec["bindings"])
        self.assertIn(("diffusersRecipe", "vae_tiling", "false"), spec["bindings"])
        self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_sana_video_seals_exact_safe_native_text_and_image_routes(self):
        expected = {
            ("SanaVideoPipeline", "text_to_video"): "sana-video-480p:direct",
            ("SanaImageToVideoPipeline", "image_to_video"): "sana-video-480p-i2v:direct",
        }
        for pair, profile_id in expected.items():
            with self.subTest(pair=pair):
                spec = studio_execution_spec_for_pair(*pair)
                self.assertIsNotNone(spec)
                self.assertEqual(spec["executionProfileId"], profile_id)
                self.assertEqual(spec["defaultRepo"], "Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
                self.assertIn(("wanPipeline", "revision", "defaultRevision"), spec["bindings"])
                self.assertIn(("diffusersQuantization", "components", "empty"), spec["bindings"])
                self.assertIn(("diffusersRecipe", "vae_tiling", "false"), spec["bindings"])
                self.assertNotIn(("wanGenerate", "scheduler_flow_shift", "shift"), spec["bindings"])
                graph, hints = executable_graph_for_spec(spec)
                assert_studio_execution_graph(graph, hints)

    def test_qwen_image_edit_inpaint_seals_the_exact_direct_mask_route(self):
        spec = studio_execution_spec_for_pair("QwenImageEditModularPipeline", "inpaint")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "qwen-edit:direct-inpaint")
        self.assertEqual(spec["pipelineClass"], "QwenImageEditInpaintPipeline")
        self.assertIn(("loadImage", "image", "diffusersImageInpaint", "image"), spec["edges"])
        self.assertIn(("loadMask", "image", "diffusersImageInpaint", "mask_image"), spec["edges"])
        self.assertIn(("diffusersImageInpaint", "strength", "strength"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_qwen_image_edit_seals_the_exact_dynamic_modular_route(self):
        spec = studio_execution_spec_for_pair("QwenImageEditModularPipeline", "edit_image")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "qwen-edit:modular")
        self.assertEqual(spec["executionPath"], "modular-diffusers")
        self.assertEqual(spec["pipelineClass"], "QwenImageEditModularPipeline")
        self.assertEqual(
            [item[0] for item in spec["roles"]],
            ["models", "prompt", "loadImage", "imageEncode", "denoise", "decode", "preview"],
        )
        self.assertIn(("loadImage", "image", "prompt", "image"), spec["edges"])
        self.assertIn(("imageEncode", "route_state_out", "denoise", "route_state_in"), spec["edges"])
        self.assertIn(("denoise", "route_state_out", "decode", "route_state_in"), spec["edges"])
        self.assertIn(("models", "model_type", "pipelineClass"), spec["bindings"])
        self.assertIn(("loadImage", "file", "referenceImages"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

        graph["nodes"][hints["studioExecutionSpec"]["nodes"]["denoise"]]["params"]["route_state_in"].pop("sourceId")
        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

    def test_qwen_image_edit_modular_inpaint_conditions_prompt_on_the_source_image(self):
        spec = studio_execution_spec_for_pair("QwenImageEditModularPipeline", "modular_inpainting")

        self.assertIsNotNone(spec)
        self.assertEqual(spec["id"], "qwen-image-edit:modular-inpainting:v1")
        self.assertIn(("loadImage", "image", "prompt", "image"), spec["edges"])
        self.assertNotIn(("denoise", "height", "height"), spec["bindings"])
        self.assertNotIn(("denoise", "width", "width"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_sdxl_modular_inpainting_seals_all_mask_and_latent_routes(self):
        spec = studio_execution_spec_for_pair("StableDiffusionXLModularPipeline", "inpaint")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["id"], "sdxl-base:modular-inpainting:v1")
        self.assertEqual(spec["executionProfileId"], "sdxl-base:modular")
        self.assertEqual(spec["executionPath"], "modular-diffusers")
        self.assertEqual(
            [item[0] for item in spec["roles"]],
            ["models", "prompt", "loadImage", "loadMask", "imageEncode", "denoise", "decode", "preview"],
        )
        self.assertIn(("loadMask", "image", "imageEncode", "mask_image"), spec["edges"])
        self.assertIn(("imageEncode", "image_latents", "denoise", "image_latents"), spec["edges"])
        self.assertIn(("imageEncode", "mask", "denoise", "mask"), spec["edges"])
        self.assertIn(
            ("imageEncode", "masked_image_latents", "denoise", "masked_image_latents"),
            spec["edges"],
        )
        self.assertIn(("loadMask", "file", "maskImage"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_sdxl_modular_controlnet_seals_the_ordinary_component_route(self):
        spec = studio_execution_spec_for_pair("StableDiffusionXLModularPipeline", "control_image")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["id"], "sdxl-base:modular-controlnet-text-to-image:v1")
        self.assertEqual(spec["executionProfileId"], "sdxl-base:modular")
        self.assertEqual(spec["executionPath"], "modular-diffusers")
        self.assertEqual(
            [item[0] for item in spec["roles"]],
            ["models", "prompt", "loadImage", "controlnetModel", "controlnet", "denoise", "decode", "preview"],
        )
        self.assertIn(("loadImage", "image", "controlnet", "control_image"), spec["edges"])
        self.assertIn(("controlnetModel", "model", "models", "controlnet"), spec["edges"])
        self.assertIn(("controlnetModel", "model", "controlnet", "controlnet"), spec["edges"])
        self.assertIn(("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"), spec["edges"])
        self.assertNotIn(("models", "vae_out", "controlnet", "vae"), spec["edges"])
        self.assertIn(("models", "vae_out", "denoise", "vae"), spec["edges"])
        self.assertIn(("controlnet", "controlnet_variant", "ordinary"), spec["bindings"])
        self.assertIn(("controlnetModel", "variant", "fp16"), spec["bindings"])
        self.assertEqual(
            studio_model_dependencies_for_pair("StableDiffusionXLModularPipeline", "control_image"),
            [
                {
                    "id": "sdxl-controlnet-canny",
                    "kind": "controlnet",
                    "repo": "diffusers/controlnet-canny-sdxl-1.0",
                    "revision": "eb115a19a10d14909256db740ed109532ab1483c",
                }
            ],
        )
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_sdxl_modular_controlnet_image_to_image_seals_both_input_routes(self):
        spec = studio_execution_spec_for_pair("StableDiffusionXLModularPipeline", "control_edit_image")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["id"], "sdxl-base:modular-controlnet-image-to-image:v1")
        self.assertEqual(spec["executionProfileId"], "sdxl-base:modular")
        self.assertEqual(spec["executionPath"], "modular-diffusers")
        self.assertEqual(
            [item[0] for item in spec["roles"]],
            [
                "models",
                "prompt",
                "loadImage",
                "imageEncode",
                "loadControlImage",
                "controlnetModel",
                "controlnet",
                "denoise",
                "decode",
                "preview",
            ],
        )
        self.assertIn(("loadImage", "image", "imageEncode", "image"), spec["edges"])
        self.assertIn(("loadControlImage", "image", "controlnet", "control_image"), spec["edges"])
        self.assertIn(("controlnetModel", "model", "models", "controlnet"), spec["edges"])
        self.assertIn(("imageEncode", "image_latents", "denoise", "image_latents"), spec["edges"])
        self.assertIn(("imageEncode", "route_state_out", "denoise", "route_state_in"), spec["edges"])
        self.assertIn(("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"), spec["edges"])
        self.assertNotIn(("models", "vae_out", "controlnet", "vae"), spec["edges"])
        self.assertIn(("controlnet", "controlnet_variant", "ordinary"), spec["bindings"])
        self.assertIn(("controlnetModel", "variant", "fp16"), spec["bindings"])
        self.assertIn(("denoise", "strength", "strength"), spec["bindings"])
        self.assertEqual(
            studio_model_dependencies_for_pair("StableDiffusionXLModularPipeline", "control_edit_image"),
            [
                {
                    "id": "sdxl-controlnet-canny",
                    "kind": "controlnet",
                    "repo": "diffusers/controlnet-canny-sdxl-1.0",
                    "revision": "eb115a19a10d14909256db740ed109532ab1483c",
                }
            ],
        )
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_qwen_image_edit_plus_modes_seal_the_exact_dynamic_modular_route(self):
        # Template materialization resolves defaultRevision from the capability
        # served to the browser, not from a separately compiled Cluster receipt.
        capability = STUDIO_MODEL_CAPABILITIES["QwenImageEditPlusModularPipeline"]
        self.assertEqual(
            capability["revisionCandidates"],
            ["6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"],
        )
        specs = [
            studio_execution_spec_for_pair("QwenImageEditPlusModularPipeline", mode)
            for mode in ("edit_image", "multi_image_reference_edit")
        ]
        for spec in specs:
            self.assertIsNotNone(spec)
            self.assertEqual(spec["executionProfileId"], "qwen-edit-plus:modular")
            self.assertEqual(spec["executionPath"], "modular-diffusers")
            self.assertEqual(spec["pipelineClass"], "QwenImageEditPlusModularPipeline")
            self.assertEqual(spec["roles"], specs[0]["roles"])
            self.assertEqual(spec["edges"], specs[0]["edges"])
            self.assertEqual(spec["bindings"], specs[0]["bindings"])
            self.assertIn(("models", "revision", "defaultRevision"), spec["bindings"])
            graph, hints = executable_graph_for_spec(spec)
            assert_studio_execution_graph(graph, hints)

        self.assertNotEqual(specs[0]["id"], specs[1]["id"])
        self.assertNotEqual(specs[0]["contentHash"], specs[1]["contentHash"])

    def test_qwen_image_layered_seals_the_exact_dynamic_modular_route(self):
        spec = studio_execution_spec_for_pair("QwenImageLayeredModularPipeline", "layer_decomposition")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "qwen-layered:modular")
        self.assertEqual(spec["executionPath"], "modular-diffusers")
        self.assertEqual(spec["pipelineClass"], "QwenImageLayeredModularPipeline")
        self.assertEqual(len(spec["roles"]), 7)
        self.assertEqual(len(spec["edges"]), 11)
        self.assertEqual(len(spec["bindings"]), 20)
        self.assertIn(("models", "revision", "defaultRevision"), spec["bindings"])
        self.assertNotIn("route_state_out", [item[1] for item in spec["edges"]])
        self.assertIn(("loadImage", "alpha_channel", "addAlpha"), spec["bindings"])
        self.assertIn(("prompt", "resolution", "resolution"), spec["bindings"])
        self.assertIn(("imageEncode", "resolution", "resolution"), spec["bindings"])
        self.assertIn(("denoise", "layers", "layers"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_qwen_control_seals_the_exact_auxiliary_loader_and_route_state_chain(self):
        spec = studio_execution_spec_for_pair("QwenImageModularPipeline", "control_image")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "qwen-image:modular")
        self.assertEqual(spec["executionPath"], "modular-diffusers")
        self.assertEqual(spec["pipelineClass"], "QwenImageModularPipeline")
        self.assertEqual(len(spec["roles"]), 8)
        self.assertEqual(len(spec["edges"]), 14)
        self.assertEqual(len(spec["bindings"]), 36)
        self.assertIn(("models", "revision", "defaultRevision"), spec["bindings"])
        self.assertIn(("controlnetModel", "model_id", "repo"), spec["bindings"])
        self.assertIn(("controlnetModel", "revision", "revision"), spec["bindings"])
        self.assertIn(("prompt", "max_sequence_length", "maxSequenceLength"), spec["bindings"])
        self.assertIn(("controlnet", "control_guidance_start", "controlGuidanceStart"), spec["bindings"])
        self.assertIn(("controlnet", "control_guidance_end", "controlGuidanceEnd"), spec["bindings"])
        self.assertIn(("controlnet", "route_state_out", "denoise", "route_state_in"), spec["edges"])
        self.assertIn(("controlnetModel", "model", "models", "controlnet"), spec["edges"])
        self.assertIn(("denoise", "route_state_out", "decode", "route_state_in"), spec["edges"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_standard_qwen_pairs_are_separate_expert_only_generic_graphs(self):
        modular_control = studio_execution_spec_for_pair("QwenImageModularPipeline", "control_image")
        modular_layered = studio_execution_spec_for_pair("QwenImageLayeredModularPipeline", "layer_decomposition")
        direct_control = studio_execution_spec_for_pair("QwenImageControlNetPipeline", "control_image")
        direct_layered = studio_execution_spec_for_pair("QwenImageLayeredPipeline", "layer_decomposition")

        self.assertEqual(modular_control["id"], "qwen-image-2512:control-image:v1")
        self.assertEqual(modular_layered["id"], "qwen-image-layered:layer-decomposition:v1")
        self.assertEqual(direct_control["id"], "qwen-image-controlnet-direct:control-image:v1")
        self.assertEqual(direct_layered["id"], "qwen-image-layered-direct:layer-decomposition:v1")
        self.assertEqual(direct_control["executionProfileId"], "qwen-image-controlnet:direct")
        self.assertEqual(direct_layered["executionProfileId"], "qwen-image-layered:direct")
        self.assertEqual(direct_control["pipelineClass"], "QwenImageControlNetPipeline")
        self.assertEqual(direct_layered["pipelineClass"], "QwenImageLayeredPipeline")
        self.assertNotEqual(direct_control["roles"], modular_control["roles"])
        self.assertNotEqual(direct_layered["roles"], modular_layered["roles"])
        self.assertIn(
            ("diffusersImageControl", "control_guidance_start", "controlGuidanceStart"),
            direct_control["bindings"],
        )
        self.assertIn(("diffusersImageLayerDecompose", "layers", "layers"), direct_layered["bindings"])

        for specification in (direct_control, direct_layered):
            with self.subTest(pair=(specification["modelType"], specification["mode"])):
                capability = STUDIO_MODEL_CAPABILITIES[specification["modelType"]]
                self.assertEqual(capability["executionStatus"], "expert_only")
                self.assertFalse(capability["autoEligible"])
                self.assertFalse(capability["galleryEligible"])
                self.assertFalse(capability["liveProof"])
                graph, hints = executable_graph_for_spec(specification)
                assert_studio_execution_graph(graph, hints)

    def test_qwen_image_edit_outpaint_seals_the_generated_canvas_and_mask_route(self):
        spec = studio_execution_spec_for_pair("QwenImageEditModularPipeline", "outpaint")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "qwen-edit:direct-inpaint")
        self.assertEqual(spec["pipelineClass"], "QwenImageEditInpaintPipeline")
        self.assertIn(("loadImage", "image", "qwenOutpaintCanvas", "image"), spec["edges"])
        self.assertIn(("qwenOutpaintCanvas", "canvas", "diffusersImageInpaint", "image"), spec["edges"])
        self.assertIn(
            ("qwenOutpaintCanvas", "mask_image", "diffusersImageInpaint", "mask_image"),
            spec["edges"],
        )
        self.assertIn(("qwenOutpaintCanvas", "fill_color", "outpaintFillColor"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_wan_vace_text_to_video_seals_the_exact_generic_video_route(self):
        spec = studio_execution_spec_for_pair("WanVACEPipeline", "text_to_video")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "wan-vace:direct")
        self.assertEqual(spec["executionPath"], "direct-wan-vace")
        self.assertEqual(spec["pipelineClass"], "WanVACEPipeline")
        self.assertIn(("wanPipeline", "revision", "wanVaceRevision"), spec["bindings"])
        self.assertIn(("wanPipeline", "pipeline", "wanGenerate", "pipeline"), spec["edges"])
        self.assertIn(("wanGenerate", "mode", "mode"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_wan_vace_video_inpaint_seals_source_and_mask_conditioning(self):
        spec = studio_execution_spec_for_pair("WanVACEPipeline", "video_inpaint")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "wan-vace:direct")
        self.assertIn(("loadVideo", "video", "normalizeVideo", "video"), spec["edges"])
        self.assertIn(("loadMaskVideo", "video", "alignMaskVideo", "mask"), spec["edges"])
        self.assertIn(("alignMaskVideo", "output", "wanGenerate", "mask"), spec["edges"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_wan_vace_video_outpaint_seals_boundary_mask_conditioning(self):
        spec = studio_execution_spec_for_pair("WanVACEPipeline", "video_outpaint")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "wan-vace:direct")
        self.assertIn(("loadVideo", "video", "normalizeVideo", "video"), spec["edges"])
        self.assertIn(("normalizeVideo", "output", "alignMaskVideo", "video"), spec["edges"])
        self.assertIn(("alignMaskVideo", "output", "wanGenerate", "mask"), spec["edges"])
        self.assertIn(("alignMaskVideo", "grow_pixels", "outpaintMaskGrow0"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_wan_vace_control_to_video_seals_control_normalization(self):
        spec = studio_execution_spec_for_pair("WanVACEPipeline", "control_to_video")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["executionProfileId"], "wan-vace:direct")
        self.assertIn(("loadControlVideo", "video", "normalizeVideo", "video"), spec["edges"])
        self.assertIn(("normalizeVideo", "output", "wanGenerate", "video"), spec["edges"])
        self.assertIn(("loadControlVideo", "file", "controlVideo"), spec["bindings"])
        self.assertIn(("normalizeVideo", "width", "width"), spec["bindings"])
        self.assertIn(("normalizeVideo", "height", "height"), spec["bindings"])
        self.assertIn(("normalizeVideo", "num_frames", "numFrames"), spec["bindings"])
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

    def test_flux_kontext_modes_have_distinct_exact_receipts_and_only_edit_has_auto_requirements(self):
        edit = studio_execution_spec_for_pair("FluxKontextPipeline", "edit_image")
        multi = studio_execution_spec_for_pair("FluxKontextPipeline", "multi_image_reference_edit")
        self.assertIsNotNone(edit)
        self.assertIsNotNone(multi)
        self.assertEqual(edit["pipelineClass"], "FluxKontextPipeline")
        self.assertEqual(multi["pipelineClass"], "FluxKontextPipeline")
        self.assertEqual(edit["executionProfileId"], multi["executionProfileId"])
        self.assertNotEqual(edit["contentHash"], multi["contentHash"])
        for spec in (edit, multi):
            graph, hints = executable_graph_for_spec(spec)
            assert_studio_execution_graph(graph, hints)
        self.assertEqual(AUTO_MODEL_REQUIREMENTS["FluxKontextPipeline"]["supportedTasks"], ["edit_image"])

    def test_flux_redux_modes_have_exact_receipts_and_multi_reference_stays_expert_only(self):
        edit = studio_execution_spec_for_pair("FluxReduxPipeline", "edit_image")
        multi = studio_execution_spec_for_pair("FluxReduxPipeline", "multi_image_reference_edit")
        self.assertIsNotNone(edit)
        self.assertIsNotNone(multi)
        self.assertEqual(edit["pipelineClass"], "FluxReduxPipeline")
        self.assertEqual(multi["pipelineClass"], "FluxReduxPipeline")
        self.assertEqual(edit["executionProfileId"], multi["executionProfileId"])
        self.assertNotEqual(edit["contentHash"], multi["contentHash"])
        for spec in (edit, multi):
            graph, hints = executable_graph_for_spec(spec)
            assert_studio_execution_graph(graph, hints)
        self.assertEqual(AUTO_MODEL_REQUIREMENTS["FluxReduxPipeline"]["supportedTasks"], ["edit_image"])

    def test_flux_fill_modes_have_distinct_exact_receipts_and_shared_auto_requirements(self):
        inpaint = studio_execution_spec_for_pair("FluxFillPipeline", "inpaint")
        outpaint = studio_execution_spec_for_pair("FluxFillPipeline", "outpaint")
        self.assertIsNotNone(inpaint)
        self.assertIsNotNone(outpaint)
        self.assertEqual(inpaint["pipelineClass"], "FluxFillPipeline")
        self.assertEqual(outpaint["pipelineClass"], "FluxFillPipeline")
        self.assertEqual(inpaint["executionProfileId"], outpaint["executionProfileId"])
        self.assertNotEqual(inpaint["contentHash"], outpaint["contentHash"])
        for spec in (inpaint, outpaint):
            graph, hints = executable_graph_for_spec(spec)
            assert_studio_execution_graph(graph, hints)
        self.assertEqual(AUTO_MODEL_REQUIREMENTS["FluxFillPipeline"]["supportedTasks"], ["inpaint", "outpaint"])

    def test_flux2_klein_modes_have_distinct_exact_receipts(self):
        text = studio_execution_spec_for_pair("Flux2KleinPipeline", "text_to_image")
        edit = studio_execution_spec_for_pair("Flux2KleinPipeline", "edit_image")
        multi = studio_execution_spec_for_pair("Flux2KleinPipeline", "multi_image_reference_edit")
        self.assertIsNotNone(text)
        self.assertIsNotNone(edit)
        self.assertIsNotNone(multi)
        self.assertEqual(text["pipelineClass"], "Flux2KleinPipeline")
        self.assertEqual(edit["pipelineClass"], "Flux2KleinPipeline")
        self.assertEqual(multi["pipelineClass"], "Flux2KleinPipeline")
        self.assertEqual(text["roles"], validate_studio_execution_specs(module_registry.MODULE_MAP)[0]["roles"])
        self.assertEqual(edit["roles"], studio_execution_spec_for_pair("FluxReduxPipeline", "edit_image")["roles"])
        self.assertEqual(multi["roles"], edit["roles"])
        self.assertNotEqual(text["contentHash"], edit["contentHash"])
        self.assertNotEqual(multi["contentHash"], edit["contentHash"])
        for spec in (text, edit, multi):
            graph, hints = executable_graph_for_spec(spec)
            assert_studio_execution_graph(graph, hints)
        self.assertEqual(
            AUTO_MODEL_REQUIREMENTS["Flux2KleinPipeline"]["supportedTasks"],
            ["text_to_image", "edit_image", "multi_image_reference_edit"],
        )

    def test_runtime_hint_parser_rejects_malformed_new_receipts_without_legacy_fallback(self):
        server = WebServer(module_registry.MODULE_MAP)
        spec = studio_execution_spec_for_pair("FluxDevPipeline", "text_to_image")
        graph, hints = executable_graph_for_spec(spec)
        parsed = server._coerce_runtime_hints(hints)
        self.assertEqual(parsed["studioExecutionSpec"], hints["studioExecutionSpec"])
        assert_studio_execution_graph(graph, parsed)

        for malformed in (
            {**hints["studioExecutionSpec"], "contentHash": "bad"},
            {**hints["studioExecutionSpec"], "nodes": {"x": "y"}},
            {**hints["studioExecutionSpec"], "extra": True},
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(RuntimeError, "Studio execution specification"):
                    server._coerce_runtime_hints({**hints, "studioExecutionSpec": malformed})

    def test_control_image_receipt_requires_the_exact_source_route(self):
        spec = studio_execution_spec_for_pair("FluxDepthPipeline", "control_image")
        self.assertIsNotNone(spec)
        graph, hints = executable_graph_for_spec(spec)
        assert_studio_execution_graph(graph, hints)

        control_id = hints["studioExecutionSpec"]["nodes"]["diffusersImageControl"]
        graph["nodes"][control_id]["params"]["control_image"].pop("sourceId")
        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)


if __name__ == "__main__":
    unittest.main()
