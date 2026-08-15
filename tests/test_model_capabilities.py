import json
import unittest

import modules as module_registry
from modiff.auto_resource import AUTO_MODEL_REQUIREMENTS
from modiff.diffusers_profiles import (
    CONTRACT_ONLY_DIFFUSERS_PIPELINES,
    DIFFUSERS_EXECUTION_PROFILES,
)
from modiff.model_artifact_catalog import catalog_revision
from modiff.modular_contract_only_registry import CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME
from modiff.server import WebServer
from modules.DiffusersAudio.main import AUDIO_PIPELINE_ADAPTERS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS
from modules.DiffusersVideo.main import VIDEO_PIPELINE_ADAPTERS


class FakeRequest:
    query = {}


class ModelCapabilitiesTests(unittest.IsolatedAsyncioTestCase):
    async def test_capabilities_publish_normalized_execution_contract(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        payload = json.loads(response.text)
        self.assertEqual(payload["schemaVersion"], 2)
        self.assertEqual(len(payload["experimentalCapabilities"]), 24)
        self.assertTrue(all(item["supportTier"] == "experimental" for item in payload["experimentalCapabilities"]))
        experimental = {item["modelType"]: item for item in payload["experimentalCapabilities"]}
        self.assertNotIn("DiffusionGemmaForBlockDiffusion", experimental)
        for model_type in CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME:
            with self.subTest(contract_only_modular=model_type):
                capability = experimental[model_type]
                self.assertEqual(capability["runnableModes"], [])
                self.assertEqual(capability["qualificationStatus"], "contract_only")
                self.assertTrue(capability["expertVisible"])
                self.assertFalse(capability["autoEligible"])
        minimax_music = experimental["MiniMaxMusic3ModularPipeline"]
        self.assertEqual(minimax_music["upstreamWorkflows"], ["text_to_audio"])
        self.assertEqual(minimax_music["runnableModes"], [])
        self.assertFalse(minimax_music["templateEligible"])
        self.assertFalse(minimax_music["galleryEligible"])
        # Official Hugging Face libraries may back generic task nodes, but the
        # removed library/model-specific driver must not return as a parallel path.
        self.assertNotIn("modules.TransformersMultimodal", module_registry.MODULE_MAP)
        self.assertEqual(
            experimental["Flux2KleinModularPipeline"]["runnableModes"],
            ["text_to_image", "edit_image", "multi_image_reference_edit"],
        )
        self.assertEqual(
            experimental["FluxModularPipeline"]["runnableModes"],
            ["text_to_image", "image_to_image"],
        )
        self.assertEqual(
            experimental["ZImageModularPipeline"]["backendPath"],
            "modules.ModularDiffusers.ModelsLoader",
        )
        self.assertEqual(
            experimental["ZImageModularPipeline"]["pipelineClasses"],
            ["ZImageModularPipeline"],
        )
        sdxl = experimental["StableDiffusionXLModularPipeline"]
        self.assertEqual(sdxl["qualificationStatus"], "contract_only")
        self.assertEqual(sdxl["runnableModes"], ["text_to_image", "image_to_image", "control_image", "inpaint"])
        self.assertEqual(sdxl["pipelineClasses"], ["StableDiffusionXLModularPipeline"])
        self.assertEqual(sdxl["backendPath"], "modules.ModularDiffusers.ModelsLoader")
        self.assertEqual(sdxl["executionProfiles"], [])
        self.assertFalse(sdxl["autoEligible"])
        self.assertFalse(sdxl["templateEligible"])
        self.assertFalse(sdxl["galleryEligible"])
        self.assertNotIn("StableDiffusionXLModularPipeline", AUTO_MODEL_REQUIREMENTS)
        for capability in payload["experimentalCapabilities"]:
            self.assertIn("executionProfiles", capability)
            self.assertIn("inputContracts", capability)
            self.assertIn("parameterAliases", capability)
            self.assertIn("defaults", capability)
            self.assertIn("artifactCandidates", capability)
            self.assertIn("revisionCandidates", capability)
            self.assertIn("quantizationSupport", capability)
        by_model = {item["modelType"]: item for item in payload["capabilities"]}
        planning_video = {
            "Wan22Pipeline": (
                ["text_to_video"],
                "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
            ),
            "WanAnimatePipeline": (
                ["character_animate", "character_replace"],
                "6f4df10861c758af86ac3c979aacc1bf5c03eff0",
            ),
            "LTXI2VLongMultiPromptPipeline": (
                ["image_to_video"],
                "7c64400e1861cc0d7b98d570a1926d5408ec60cd",
            ),
            "LTX2ConditionPipeline": (
                ["image_to_video", "reference_to_video", "text_to_video", "video_to_video"],
                "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            ),
            "LTX2Pipeline": (
                ["text_to_video"],
                "47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
            ),
            "HunyuanVideoFramepackPipeline": (
                ["image_to_video"],
                "86cef4396041b6002c957852daac4c91aaa47c79",
            ),
            "StableVideoDiffusionPipeline": (
                ["image_to_video"],
                "043843887ccd51926e3efed36270444a838e7861",
            ),
            "AnimateDiffPipeline": (
                ["text_to_video"],
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            ),
            "AnimateDiffPAGPipeline": (
                ["text_to_video"],
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            ),
            "AnimateDiffVideoToVideoPipeline": (
                ["video_to_video"],
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            ),
            "AnimateDiffControlNetPipeline": (
                ["control_to_video"],
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            ),
            "AnimateDiffVideoToVideoControlNetPipeline": (
                ["control_video_to_video"],
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            ),
            "AnimateLCMPipeline": (
                ["text_to_video"],
                "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            ),
            "CogVideoXPipeline": (
                ["text_to_video"],
                "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01",
            ),
            "CogVideoXVideoToVideoPipeline": (
                ["video_to_video"],
                "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01",
            ),
            "AllegroPipeline": (
                ["text_to_video"],
                "c1b9207bb5cb79e2aa08f3d139c17d26c0de55b6",
            ),
            "LattePipeline": (
                ["text_to_video"],
                "0653024365272f061fc44d1078134df22842b687",
            ),
            "MochiPipeline": (
                ["text_to_video"],
                "14be5fcea23095ed330cb214647916a451e38b6e",
            ),
            "SanaVideoPipeline": (
                ["text_to_video"],
                "db5f398b13ca086d09a50ce156c20527773841b1",
            ),
            "SanaImageToVideoPipeline": (
                ["image_to_video"],
                "db5f398b13ca086d09a50ce156c20527773841b1",
            ),
            "WanImage2VideoModularPipeline": (
                ["image_to_video"],
                "17c30769b1e0b5dcaa1799b117bf20a9c31f59d7",
            ),
        }
        for model_type, (modes, revision) in planning_video.items():
            with self.subTest(planning_video=model_type):
                capability = by_model[model_type]
                self.assertEqual(capability["runnableModes"], modes)
                self.assertEqual(capability["revisionCandidates"], [revision])
                self.assertEqual(capability["qualificationStatus"], "graph-qualified-execution-pending")
                self.assertEqual(capability["executionStatus"], "expert_only")
                self.assertEqual(capability["qualifiedModes"], [])
                self.assertNotIn(model_type, experimental)

        self.assertEqual(len(payload["studioExecutionSpecs"]), 184)
        for model_type in (
            "FluxSchnellPipeline",
            "FluxDevPipeline",
            "FluxKreaPipeline",
            "FluxDepthPipeline",
            "FluxCannyPipeline",
            "FluxReduxPipeline",
            "FluxKontextPipeline",
            "FluxFillPipeline",
            "Flux2KleinPipeline",
            "WanImageToVideoPipeline",
            "WanTI2VPipeline",
            "Wan22Pipeline",
            "WanAnimatePipeline",
            "WanImage2VideoModularPipeline",
            "WanVideoPipeline",
            "LTXVideoPipeline",
            "LTXI2VLongMultiPromptPipeline",
            "LTX2ConditionPipeline",
            "LTX2Pipeline",
            "HunyuanVideoFramepackPipeline",
            "StableVideoDiffusionPipeline",
            "AnimateDiffPipeline",
            "AnimateDiffPAGPipeline",
            "AnimateDiffVideoToVideoPipeline",
            "AnimateDiffControlNetPipeline",
            "AnimateDiffVideoToVideoControlNetPipeline",
            "AnimateLCMPipeline",
            "CogVideoXVideoToVideoPipeline",
            "AllegroPipeline",
            "LattePipeline",
            "MochiPipeline",
            "SanaVideoPipeline",
            "SanaImageToVideoPipeline",
            "AceStepAudioPipeline",
            "ZImageModularPipeline",
            "QwenImageModularPipeline",
            "QwenImageControlNetPipeline",
            "QwenImageLayeredPipeline",
            "QwenImageEditPipeline",
            "QwenImageEditPlusPipeline",
            "ZImageInpaintPipeline",
            "ChromaImg2ImgPipeline",
            "ChromaInpaintPipeline",
            "FluxKontextInpaintPipeline",
            "Flux2KleinInpaintPipeline",
            "StableDiffusionXLPipeline",
            "DDPMPipeline",
            "DDIMPipeline",
            "ConsistencyModelPipeline",
            "StableDiffusionPipeline",
            "StableDiffusionXLTurboPipeline",
            "StableDiffusionXLInstructPix2PixPipeline",
            "StableDiffusionXLControlNetPipeline",
            "HunyuanDiTPipeline",
            "HunyuanDiTPAGPipeline",
            "HunyuanDiTControlNetPipeline",
            "StableDiffusionXLAdapterPipeline",
            "StableDiffusionXLPAGPipeline",
            "SanaPipeline",
            "SanaPAGPipeline",
            "SanaSprintPipeline",
            "PixArtSigmaPipeline",
            "PixArtSigmaPAGPipeline",
            "Kandinsky3Pipeline",
            "OvisImagePipeline",
            "PRXPipeline",
            "NucleusMoEImagePipeline",
            "AuraFlowPipeline",
            "ChromaPipeline",
            "CogView3PlusPipeline",
            "CogView4Pipeline",
            "ErnieImagePipeline",
            "GlmImagePipeline",
            "JoyImageEditPipeline",
            "JoyImageEditPlusPipeline",
            "DreamLitePipeline",
            "DreamLiteMobilePipeline",
            "LatentConsistencyModelPipeline",
            "StableDiffusionPAGPipeline",
            "MarigoldDepthPipeline",
            "HuggingFaceSpeechRecognitionModel",
            "HuggingFaceAnyToAnyModel",
        ):
            self.assertEqual(
                by_model[model_type]["studioExecutionSpecs"],
                [item for item in payload["studioExecutionSpecs"] if item["modelType"] == model_type],
            )
        self.assertEqual(
            payload["studioExecutionSpecs"][0]["roles"],
            payload["studioExecutionSpecs"][1]["roles"],
        )
        self.assertEqual(
            payload["studioExecutionSpecs"][0]["edges"],
            payload["studioExecutionSpecs"][1]["edges"],
        )
        self.assertEqual(
            payload["studioExecutionSpecs"][0]["roles"],
            payload["studioExecutionSpecs"][2]["roles"],
        )
        self.assertEqual(
            payload["studioExecutionSpecs"][0]["edges"],
            payload["studioExecutionSpecs"][2]["edges"],
        )
        sdxl = by_model["StableDiffusionXLPipeline"]
        self.assertEqual(sdxl["executionProfiles"][0]["id"], "sdxl-base:direct")
        self.assertEqual(
            sdxl["revisionCandidates"],
            ["462165984030d82259a11f4367a4eed129e94a7b"],
        )
        self.assertTrue(sdxl["templateEligible"])
        self.assertFalse(sdxl["autoEligible"])
        self.assertFalse(sdxl["galleryEligible"])
        self.assertNotIn("StableDiffusionXLPipeline", experimental)
        sdxl_edit = next(item for item in sdxl["studioExecutionSpecs"] if item["mode"] == "edit_image")
        self.assertEqual(sdxl["studioExecutionSpecModes"], ["edit_image", "inpaint", "text_to_image"])
        self.assertEqual(
            sdxl["pipelineClasses"],
            [
                "StableDiffusionXLImg2ImgPipeline",
                "StableDiffusionXLInpaintPipeline",
                "StableDiffusionXLPipeline",
            ],
        )
        self.assertEqual(sdxl_edit["pipelineClass"], "StableDiffusionXLImg2ImgPipeline")
        self.assertEqual(
            next(profile for profile in sdxl["executionProfiles"] if profile["id"] == "sdxl-base:img2img-direct")[
                "modes"
            ],
            ["edit_image"],
        )
        self.assertEqual(
            sdxl["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )
        turbo = by_model["StableDiffusionXLTurboPipeline"]
        self.assertEqual(turbo["revisionCandidates"], ["71153311d3dbb46851df1931d3ca6e939de83304"])
        self.assertEqual(turbo["recommendedSteps"], 1)
        self.assertEqual(turbo["recommendedGuidance"], 0.0)
        self.assertFalse(turbo["autoEligible"])
        self.assertFalse(turbo["galleryEligible"])
        self.assertNotIn("StableDiffusionXLTurboPipeline", experimental)
        instruct = by_model["StableDiffusionXLInstructPix2PixPipeline"]
        self.assertEqual(instruct["revisionCandidates"], ["06653d47f8d22f2c2205a5884d6a24c5e76d2ca7"])
        self.assertEqual(instruct["recommendedSteps"], 30)
        self.assertEqual(instruct["recommendedGuidance"], 3.0)
        self.assertEqual(instruct["conditioningScale"], 1.5)
        self.assertEqual(instruct["modeRequirements"]["edit_image"]["requiredImages"], ["referenceImages"])
        self.assertFalse(instruct["autoEligible"])
        self.assertFalse(instruct["galleryEligible"])
        self.assertNotIn("StableDiffusionXLInstructPix2PixPipeline", experimental)
        sdxl_controlnet = by_model["StableDiffusionXLControlNetPipeline"]
        self.assertEqual(
            sdxl_controlnet["revisionCandidates"],
            ["462165984030d82259a11f4367a4eed129e94a7b"],
        )
        self.assertEqual(sdxl_controlnet["recommendedSteps"], 50)
        self.assertEqual(sdxl_controlnet["recommendedGuidance"], 5.0)
        self.assertEqual(sdxl_controlnet["conditioningScale"], 0.5)
        self.assertEqual(
            sdxl_controlnet["modeRequirements"]["control_image"]["modelRequirements"][0]["revision"],
            "eb115a19a10d14909256db740ed109532ab1483c",
        )
        self.assertEqual(
            sdxl_controlnet["modeRequirements"]["control_image"]["requiredImages"],
            ["controlImage"],
        )
        self.assertEqual(
            sdxl_controlnet["modes"],
            ["control_image", "control_edit_image", "control_inpaint"],
        )
        self.assertEqual(
            sdxl_controlnet["pipelineClasses"],
            [
                "StableDiffusionXLControlNetImg2ImgPipeline",
                "StableDiffusionXLControlNetInpaintPipeline",
                "StableDiffusionXLControlNetPipeline",
            ],
        )
        self.assertFalse(sdxl_controlnet["autoEligible"])
        self.assertFalse(sdxl_controlnet["galleryEligible"])
        self.assertNotIn("StableDiffusionXLControlNetPipeline", experimental)
        sdxl_adapter = by_model["StableDiffusionXLAdapterPipeline"]
        self.assertEqual(
            sdxl_adapter["revisionCandidates"],
            ["462165984030d82259a11f4367a4eed129e94a7b"],
        )
        self.assertEqual(sdxl_adapter["recommendedSteps"], 30)
        self.assertEqual(sdxl_adapter["recommendedGuidance"], 7.5)
        self.assertEqual(sdxl_adapter["conditioningScale"], 0.8)
        self.assertEqual(
            sdxl_adapter["modeRequirements"]["control_image"]["modelRequirements"][0]["revision"],
            "2d7244ba45ded9129cfbf8e96a4befb7f6094210",
        )
        self.assertEqual(
            sdxl_adapter["modeRequirements"]["control_image"]["requiredImages"],
            ["controlImage"],
        )
        self.assertFalse(sdxl_adapter["autoEligible"])
        self.assertFalse(sdxl_adapter["galleryEligible"])
        self.assertNotIn("StableDiffusionXLAdapterPipeline", experimental)
        sdxl_pag = by_model["StableDiffusionXLPAGPipeline"]
        self.assertEqual(
            sdxl_pag["revisionCandidates"],
            ["462165984030d82259a11f4367a4eed129e94a7b"],
        )
        self.assertEqual(sdxl_pag["recommendedSteps"], 50)
        self.assertEqual(sdxl_pag["recommendedGuidance"], 5.0)
        self.assertEqual(sdxl_pag["recommendedPagScale"], 3.0)
        self.assertEqual(sdxl_pag["recommendedPagAdaptiveScale"], 0.0)
        self.assertEqual(
            sdxl_pag["modes"],
            ["text_to_image", "edit_image", "inpaint", "control_image", "control_edit_image"],
        )
        self.assertEqual(
            sdxl_pag["pipelineClasses"],
            [
                "StableDiffusionXLControlNetPAGImg2ImgPipeline",
                "StableDiffusionXLControlNetPAGPipeline",
                "StableDiffusionXLPAGImg2ImgPipeline",
                "StableDiffusionXLPAGInpaintPipeline",
                "StableDiffusionXLPAGPipeline",
            ],
        )
        self.assertEqual(
            sdxl_pag["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )
        self.assertEqual(
            sdxl_pag["modeRequirements"]["inpaint"]["requiredImages"],
            ["referenceImages", "maskImage"],
        )
        self.assertFalse(sdxl_pag["autoEligible"])
        self.assertFalse(sdxl_pag["galleryEligible"])
        self.assertNotIn("StableDiffusionXLPAGPipeline", experimental)
        hunyuan_pag = by_model["HunyuanDiTPAGPipeline"]
        self.assertEqual(
            hunyuan_pag["revisionCandidates"],
            ["ba991d1546d8c50936c4c16398ed0a87b9b99fb1"],
        )
        self.assertEqual(hunyuan_pag["recommendedSteps"], 25)
        self.assertEqual(hunyuan_pag["recommendedGuidance"], 5.0)
        self.assertEqual(hunyuan_pag["recommendedPagScale"], 3.0)
        self.assertEqual(hunyuan_pag["recommendedPagAdaptiveScale"], 0.0)
        self.assertNotIn("recommendedMaxSequenceLength", hunyuan_pag)
        self.assertEqual(hunyuan_pag["modes"], ["text_to_image"])
        self.assertEqual(hunyuan_pag["pipelineClasses"], ["HunyuanDiTPAGPipeline"])
        self.assertFalse(hunyuan_pag["autoEligible"])
        self.assertFalse(hunyuan_pag["galleryEligible"])
        self.assertNotIn("HunyuanDiTPAGPipeline", experimental)
        sana = by_model["SanaPipeline"]
        self.assertEqual(sana["revisionCandidates"], ["28f3af7689de15f3883d5863059a2fca0aa9b829"])
        self.assertEqual(sana["recommendedSteps"], 20)
        self.assertEqual(sana["recommendedGuidance"], 4.5)
        self.assertEqual(sana["recommendedMaxSequenceLength"], 300)
        self.assertEqual(sana["defaultDtype"], "float16")
        self.assertEqual(sana["pipelineClasses"], ["SanaPipeline"])
        self.assertFalse(sana["autoEligible"])
        self.assertFalse(sana["galleryEligible"])
        sana_pag = by_model["SanaPAGPipeline"]
        self.assertEqual(
            sana_pag["revisionCandidates"],
            ["28f3af7689de15f3883d5863059a2fca0aa9b829"],
        )
        self.assertEqual(sana_pag["recommendedSteps"], 20)
        self.assertEqual(sana_pag["recommendedGuidance"], 4.5)
        self.assertEqual(sana_pag["recommendedMaxSequenceLength"], 300)
        self.assertEqual(sana_pag["recommendedPagScale"], 3.0)
        self.assertEqual(sana_pag["recommendedPagAdaptiveScale"], 0.0)
        self.assertEqual(sana_pag["pipelineClasses"], ["SanaPAGPipeline"])
        self.assertFalse(sana_pag["autoEligible"])
        self.assertFalse(sana_pag["galleryEligible"])
        self.assertNotIn("SanaPAGPipeline", experimental)
        sana_sprint = by_model["SanaSprintPipeline"]
        self.assertEqual(
            sana_sprint["revisionCandidates"],
            ["aa76e7f4f4928f378716b6716a2130fba3caf5b1"],
        )
        self.assertEqual(sana_sprint["recommendedSteps"], 2)
        self.assertEqual(sana_sprint["recommendedStrength"], 0.5)
        self.assertEqual(sana_sprint["recommendedMaxSequenceLength"], 300)
        self.assertEqual(sana_sprint["modes"], ["text_to_image", "edit_image"])
        self.assertEqual(
            sana_sprint["pipelineClasses"],
            ["SanaSprintImg2ImgPipeline", "SanaSprintPipeline"],
        )
        self.assertEqual(
            sana_sprint["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )
        self.assertFalse(sana_sprint["autoEligible"])
        self.assertFalse(sana_sprint["galleryEligible"])
        self.assertNotIn("SanaPipeline", experimental)
        self.assertNotIn("SanaSprintPipeline", experimental)
        pixart = by_model["PixArtSigmaPipeline"]
        self.assertEqual(
            pixart["revisionCandidates"],
            ["e102b3591cc82e97071b8b4cb90d834d0c487207"],
        )
        self.assertEqual(pixart["recommendedSteps"], 20)
        self.assertEqual(pixart["recommendedGuidance"], 4.5)
        self.assertEqual(pixart["recommendedMaxSequenceLength"], 300)
        self.assertEqual(pixart["modes"], ["text_to_image"])
        self.assertEqual(pixart["pipelineClasses"], ["PixArtSigmaPipeline"])
        self.assertFalse(pixart["autoEligible"])
        self.assertFalse(pixart["galleryEligible"])
        self.assertNotIn("PixArtSigmaPipeline", experimental)
        pixart_pag = by_model["PixArtSigmaPAGPipeline"]
        self.assertEqual(
            pixart_pag["revisionCandidates"],
            ["e102b3591cc82e97071b8b4cb90d834d0c487207"],
        )
        self.assertEqual(pixart_pag["recommendedSteps"], 20)
        self.assertEqual(pixart_pag["recommendedGuidance"], 4.5)
        self.assertEqual(pixart_pag["recommendedMaxSequenceLength"], 300)
        self.assertEqual(pixart_pag["recommendedPagScale"], 3.0)
        self.assertEqual(pixart_pag["recommendedPagAdaptiveScale"], 0.0)
        self.assertEqual(pixart_pag["pipelineClasses"], ["PixArtSigmaPAGPipeline"])
        self.assertFalse(pixart_pag["autoEligible"])
        self.assertFalse(pixart_pag["galleryEligible"])
        self.assertNotIn("PixArtSigmaPAGPipeline", experimental)
        kandinsky3 = by_model["Kandinsky3Pipeline"]
        self.assertEqual(
            kandinsky3["revisionCandidates"],
            ["bf79e6c219da8a94abb50235fdc4567eb8fb4632"],
        )
        self.assertEqual(kandinsky3["recommendedSteps"], 25)
        self.assertEqual(kandinsky3["recommendedGuidance"], 3.0)
        self.assertEqual(kandinsky3["recommendedMaxSequenceLength"], 128)
        self.assertEqual(kandinsky3["modes"], ["text_to_image", "edit_image"])
        self.assertEqual(
            kandinsky3["pipelineClasses"],
            ["Kandinsky3Img2ImgPipeline", "Kandinsky3Pipeline"],
        )
        self.assertEqual(
            kandinsky3["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )
        self.assertFalse(kandinsky3["autoEligible"])
        self.assertFalse(kandinsky3["galleryEligible"])
        self.assertNotIn("Kandinsky3Pipeline", experimental)
        auraflow = by_model["AuraFlowPipeline"]
        self.assertEqual(
            auraflow["revisionCandidates"],
            ["2cd8588f04c886002be4571697d84654a50e3af3"],
        )
        self.assertEqual(auraflow["defaultSize"], {"width": 1536, "height": 768, "aspectRatio": "custom"})
        self.assertEqual(auraflow["recommendedSteps"], 50)
        self.assertEqual(auraflow["recommendedGuidance"], 3.5)
        self.assertEqual(auraflow["recommendedMaxSequenceLength"], 256)
        self.assertEqual(auraflow["modes"], ["text_to_image"])
        self.assertEqual(auraflow["pipelineClasses"], ["AuraFlowPipeline"])
        self.assertFalse(auraflow["autoEligible"])
        self.assertFalse(auraflow["galleryEligible"])
        self.assertNotIn("AuraFlowPipeline", experimental)
        chroma = by_model["ChromaPipeline"]
        self.assertEqual(
            chroma["revisionCandidates"],
            ["0e0c60ece1e82b17cb7f77342d765ba5024c40c0"],
        )
        self.assertEqual(chroma["defaultSize"], {"width": 1024, "height": 1024, "aspectRatio": "1:1"})
        self.assertEqual(chroma["recommendedSteps"], 40)
        self.assertEqual(chroma["recommendedGuidance"], 3.0)
        self.assertEqual(chroma["recommendedMaxSequenceLength"], 512)
        self.assertEqual(chroma["modes"], ["text_to_image"])
        self.assertEqual(chroma["pipelineClasses"], ["ChromaPipeline"])
        self.assertFalse(chroma["autoEligible"])
        self.assertFalse(chroma["galleryEligible"])
        self.assertNotIn("ChromaPipeline", experimental)
        cogview3 = by_model["CogView3PlusPipeline"]
        self.assertEqual(
            cogview3["revisionCandidates"],
            ["5d70e40732ac0efac98524c51a7fa9c82707f1e5"],
        )
        self.assertEqual(cogview3["defaultSize"], {"width": 1024, "height": 1024, "aspectRatio": "1:1"})
        self.assertEqual(cogview3["recommendedSteps"], 50)
        self.assertEqual(cogview3["recommendedGuidance"], 7.0)
        self.assertEqual(cogview3["recommendedMaxSequenceLength"], 224)
        self.assertEqual(cogview3["modes"], ["text_to_image"])
        self.assertEqual(cogview3["pipelineClasses"], ["CogView3PlusPipeline"])
        self.assertFalse(cogview3["autoEligible"])
        self.assertFalse(cogview3["galleryEligible"])
        self.assertNotIn("CogView3PlusPipeline", experimental)
        cogview4 = by_model["CogView4Pipeline"]
        self.assertEqual(
            cogview4["revisionCandidates"],
            ["63a52b7f6dace7033380cd6da14d0915eab3e6b5"],
        )
        self.assertEqual(cogview4["defaultSize"], {"width": 1024, "height": 1024, "aspectRatio": "1:1"})
        self.assertEqual(cogview4["recommendedSteps"], 50)
        self.assertEqual(cogview4["recommendedGuidance"], 3.5)
        self.assertEqual(cogview4["recommendedMaxSequenceLength"], 1024)
        self.assertEqual(cogview4["modes"], ["text_to_image"])
        self.assertEqual(cogview4["pipelineClasses"], ["CogView4Pipeline"])
        self.assertFalse(cogview4["autoEligible"])
        self.assertFalse(cogview4["galleryEligible"])
        self.assertNotIn("CogView4Pipeline", experimental)
        ernie = by_model["ErnieImagePipeline"]
        self.assertEqual(
            ernie["revisionCandidates"],
            ["bc68c81e2a1730a394d5fc9fae70713dee940140"],
        )
        self.assertEqual(ernie["defaultSize"], {"width": 1024, "height": 1024, "aspectRatio": "1:1"})
        self.assertEqual(ernie["recommendedSteps"], 8)
        self.assertEqual(ernie["recommendedGuidance"], 1.0)
        self.assertEqual(ernie["recommendedMaxSequenceLength"], 2048)
        self.assertEqual(ernie["modes"], ["text_to_image"])
        self.assertEqual(ernie["pipelineClasses"], ["ErnieImagePipeline"])
        self.assertFalse(ernie["supportsNegativePrompt"])
        self.assertFalse(ernie["autoEligible"])
        self.assertFalse(ernie["galleryEligible"])
        self.assertNotIn("ErnieImagePipeline", experimental)
        glm_image = by_model["GlmImagePipeline"]
        self.assertEqual(
            glm_image["revisionCandidates"],
            ["2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4"],
        )
        self.assertEqual(glm_image["defaultSize"], {"width": 1024, "height": 1024, "aspectRatio": "1:1"})
        self.assertEqual(glm_image["recommendedSteps"], 50)
        self.assertEqual(glm_image["recommendedGuidance"], 1.5)
        self.assertEqual(glm_image["recommendedMaxSequenceLength"], 2048)
        self.assertEqual(glm_image["modes"], ["text_to_image"])
        self.assertEqual(glm_image["pipelineClasses"], ["GlmImagePipeline"])
        self.assertFalse(glm_image["supportsNegativePrompt"])
        self.assertFalse(glm_image["supportsLora"])
        self.assertFalse(glm_image["autoEligible"])
        self.assertFalse(glm_image["galleryEligible"])
        self.assertNotIn("GlmImagePipeline", experimental)
        dreamlite = by_model["DreamLitePipeline"]
        self.assertEqual(
            dreamlite["revisionCandidates"],
            ["751cb8dbb9072a8c8ffd8684e0f254b50f20531b"],
        )
        self.assertEqual(dreamlite["recommendedSteps"], 28)
        self.assertEqual(dreamlite["recommendedGuidance"], 3.5)
        self.assertEqual(dreamlite["conditioningScale"], 1.5)
        self.assertEqual(dreamlite["recommendedMaxSequenceLength"], 200)
        self.assertEqual(dreamlite["modes"], ["text_to_image", "edit_image"])
        self.assertEqual(dreamlite["pipelineClasses"], ["DreamLitePipeline"])
        self.assertEqual(
            dreamlite["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )
        self.assertFalse(dreamlite["autoEligible"])
        self.assertFalse(dreamlite["galleryEligible"])
        self.assertNotIn("DreamLitePipeline", experimental)
        dreamlite_mobile = by_model["DreamLiteMobilePipeline"]
        self.assertEqual(
            dreamlite_mobile["revisionCandidates"],
            ["6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4"],
        )
        self.assertEqual(dreamlite_mobile["recommendedSteps"], 4)
        self.assertEqual(dreamlite_mobile["recommendedGuidance"], 0.0)
        self.assertEqual(dreamlite_mobile["conditioningScale"], 0.0)
        self.assertEqual(dreamlite_mobile["recommendedMaxSequenceLength"], 200)
        self.assertEqual(dreamlite_mobile["modes"], ["text_to_image", "edit_image"])
        self.assertEqual(
            dreamlite_mobile["pipelineClasses"],
            ["DreamLiteMobilePipeline"],
        )
        self.assertFalse(dreamlite_mobile["supportsNegativePrompt"])
        self.assertFalse(dreamlite_mobile["autoEligible"])
        self.assertFalse(dreamlite_mobile["galleryEligible"])
        self.assertNotIn("DreamLiteMobilePipeline", experimental)
        sdxl_inpaint = next(item for item in sdxl["studioExecutionSpecs"] if item["mode"] == "inpaint")
        self.assertEqual(sdxl_inpaint["pipelineClass"], "StableDiffusionXLInpaintPipeline")
        self.assertEqual(
            next(profile for profile in sdxl["executionProfiles"] if profile["id"] == "sdxl-base:inpaint-direct")[
                "modes"
            ],
            ["inpaint"],
        )
        self.assertEqual(
            sdxl["modeRequirements"]["inpaint"]["requiredImages"],
            ["referenceImages", "maskImage"],
        )
        sd15 = by_model["StableDiffusionPipeline"]
        self.assertEqual(
            sd15["studioExecutionSpecModes"],
            [
                "control_edit_image",
                "control_image",
                "control_inpaint",
                "edit_image",
                "inpaint",
                "text_to_image",
            ],
        )
        self.assertTrue(sd15["supportsControlImage"])
        self.assertFalse(sd15["autoEligible"])
        self.assertFalse(sd15["galleryEligible"])
        self.assertEqual(
            sd15["modeRequirements"]["control_image"]["modelRequirements"][0]["revision"],
            "115a470d547982438f70198e353a921996e2e819",
        )
        lcm = by_model["LatentConsistencyModelPipeline"]
        self.assertEqual(lcm["modes"], ["text_to_image", "edit_image"])
        self.assertEqual(lcm["studioExecutionSpecModes"], ["edit_image", "text_to_image"])
        self.assertEqual(
            lcm["pipelineClasses"],
            ["LatentConsistencyModelImg2ImgPipeline", "LatentConsistencyModelPipeline"],
        )
        self.assertEqual(
            lcm["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )
        self.assertTrue(lcm["supportsImageInput"])
        self.assertNotIn("LatentConsistencyModelImg2ImgPipeline", experimental)
        sd15_pag = by_model["StableDiffusionPAGPipeline"]
        self.assertEqual(
            sd15_pag["modes"],
            ["text_to_image", "edit_image", "inpaint", "control_image", "control_inpaint"],
        )
        self.assertEqual(
            sd15_pag["studioExecutionSpecModes"],
            ["control_image", "control_inpaint", "edit_image", "inpaint", "text_to_image"],
        )
        self.assertEqual(
            sd15_pag["pipelineClasses"],
            [
                "StableDiffusionControlNetPAGInpaintPipeline",
                "StableDiffusionControlNetPAGPipeline",
                "StableDiffusionPAGImg2ImgPipeline",
                "StableDiffusionPAGInpaintPipeline",
                "StableDiffusionPAGPipeline",
            ],
        )
        self.assertEqual(
            sd15_pag["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )
        self.assertEqual(
            sd15_pag["modeRequirements"]["inpaint"]["requiredImages"],
            ["referenceImages", "maskImage"],
        )
        self.assertTrue(sd15_pag["supportsImageInput"])
        self.assertTrue(sd15_pag["supportsMask"])
        self.assertNotIn("StableDiffusionPAGImg2ImgPipeline", experimental)
        self.assertNotIn("StableDiffusionPAGInpaintPipeline", experimental)
        flux_dev = by_model["FluxDevPipeline"]
        self.assertEqual(flux_dev["studioExecutionSpecModes"], ["edit_image", "inpaint", "text_to_image"])
        self.assertEqual(
            flux_dev["pipelineClasses"],
            ["FluxImg2ImgPipeline", "FluxInpaintPipeline", "FluxPipeline"],
        )
        flux_dev_edit = next(item for item in flux_dev["studioExecutionSpecs"] if item["mode"] == "edit_image")
        self.assertEqual(flux_dev_edit["pipelineClass"], "FluxImg2ImgPipeline")
        self.assertEqual(
            flux_dev["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )
        flux_dev_inpaint = next(item for item in flux_dev["studioExecutionSpecs"] if item["mode"] == "inpaint")
        self.assertEqual(flux_dev_inpaint["pipelineClass"], "FluxInpaintPipeline")
        self.assertEqual(
            flux_dev["modeRequirements"]["inpaint"]["requiredImages"],
            ["referenceImages", "maskImage"],
        )
        depth_spec = by_model["FluxDepthPipeline"]["studioExecutionSpecs"][0]
        self.assertEqual(
            by_model["FluxDepthPipeline"]["modes"],
            ["control_image", "control_edit_image", "control_inpaint"],
        )
        self.assertEqual(depth_spec["mode"], "control_image")
        self.assertIn("diffusersImageControl", [item[0] for item in depth_spec["roles"]])
        self.assertIn("loadImage", [item[0] for item in depth_spec["roles"]])

        canny_spec = by_model["FluxCannyPipeline"]["studioExecutionSpecs"][0]
        self.assertEqual(
            by_model["FluxCannyPipeline"]["modes"],
            ["control_image", "control_edit_image", "control_inpaint"],
        )
        self.assertEqual(canny_spec["mode"], "control_image")
        self.assertEqual(canny_spec["roles"], depth_spec["roles"])
        self.assertEqual(canny_spec["edges"], depth_spec["edges"])

        redux_specs = by_model["FluxReduxPipeline"]["studioExecutionSpecs"]
        redux_spec = next(item for item in redux_specs if item["mode"] == "edit_image")
        redux_multi_spec = next(item for item in redux_specs if item["mode"] == "multi_image_reference_edit")
        self.assertEqual(
            by_model["FluxReduxPipeline"]["modes"],
            ["edit_image", "multi_image_reference_edit"],
        )
        self.assertEqual(redux_spec["mode"], "edit_image")
        self.assertEqual(redux_multi_spec["pipelineClass"], "FluxReduxPipeline")
        self.assertNotEqual(redux_multi_spec["contentHash"], redux_spec["contentHash"])
        self.assertIn("diffusersImageEdit", [item[0] for item in redux_spec["roles"]])
        redux_requirement = by_model["FluxReduxPipeline"]["modeRequirements"]["edit_image"]["modelRequirements"][0]
        self.assertEqual(redux_requirement["repo"], "black-forest-labs/FLUX.1-dev")
        self.assertEqual(
            redux_requirement["revision"],
            "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        )
        self.assertEqual(
            by_model["FluxReduxPipeline"]["additionalRequirements"][0],
            redux_requirement,
        )
        self.assertEqual(
            by_model["FluxReduxPipeline"]["modeRequirements"]["multi_image_reference_edit"]["modelRequirements"][0],
            redux_requirement,
        )

        kontext_specs = by_model["FluxKontextPipeline"]["studioExecutionSpecs"]
        kontext_spec = next(item for item in kontext_specs if item["mode"] == "edit_image")
        self.assertEqual(by_model["FluxKontextPipeline"]["modes"], ["edit_image", "multi_image_reference_edit"])
        self.assertEqual(kontext_spec["mode"], "edit_image")
        self.assertEqual(kontext_spec["pipelineClass"], "FluxKontextPipeline")
        self.assertIn("diffusersImageEdit", [item[0] for item in kontext_spec["roles"]])
        kontext_multi_spec = next(item for item in kontext_specs if item["mode"] == "multi_image_reference_edit")
        self.assertEqual(kontext_multi_spec["pipelineClass"], "FluxKontextPipeline")
        self.assertNotEqual(kontext_multi_spec["contentHash"], kontext_spec["contentHash"])

        fill_specs = by_model["FluxFillPipeline"]["studioExecutionSpecs"]
        fill_spec = next(item for item in fill_specs if item["mode"] == "inpaint")
        fill_outpaint_spec = next(item for item in fill_specs if item["mode"] == "outpaint")
        self.assertEqual(by_model["FluxFillPipeline"]["modes"], ["inpaint", "outpaint"])
        self.assertEqual(by_model["FluxFillPipeline"]["studioExecutionSpecModes"], ["inpaint", "outpaint"])
        self.assertEqual(fill_spec["pipelineClass"], "FluxFillPipeline")
        self.assertEqual(fill_outpaint_spec["pipelineClass"], "FluxFillPipeline")
        self.assertNotEqual(fill_outpaint_spec["contentHash"], fill_spec["contentHash"])
        self.assertIn("diffusersImageInpaint", [item[0] for item in fill_spec["roles"]])
        self.assertIn("loadMask", [item[0] for item in fill_spec["roles"]])

        klein_specs = by_model["Flux2KleinPipeline"]["studioExecutionSpecs"]
        klein_text = next(item for item in klein_specs if item["mode"] == "text_to_image")
        klein_edit = next(item for item in klein_specs if item["mode"] == "edit_image")
        klein_multi = next(item for item in klein_specs if item["mode"] == "multi_image_reference_edit")
        self.assertEqual(klein_text["id"], "flux2-klein:text-to-image:v1")
        self.assertEqual(klein_edit["id"], "flux2-klein:edit-image:v1")
        self.assertEqual(klein_multi["id"], "flux2-klein:multi-image-reference-edit:v1")
        self.assertEqual(klein_text["pipelineClass"], "Flux2KleinPipeline")
        self.assertEqual(klein_edit["pipelineClass"], "Flux2KleinPipeline")
        self.assertEqual(klein_multi["pipelineClass"], "Flux2KleinPipeline")
        self.assertNotEqual(klein_text["contentHash"], klein_edit["contentHash"])
        self.assertNotEqual(klein_multi["contentHash"], klein_edit["contentHash"])
        self.assertEqual(
            by_model["Flux2KleinPipeline"]["studioExecutionSpecModes"],
            ["edit_image", "multi_image_reference_edit", "text_to_image"],
        )

        i2v_spec = by_model["WanImageToVideoPipeline"]["studioExecutionSpecs"][0]
        self.assertEqual(i2v_spec["mode"], "image_to_video")
        self.assertEqual(i2v_spec["pipelineClass"], "WanImageToVideoPipeline")
        self.assertIn("loadImage", [item[0] for item in i2v_spec["roles"]])

        ti2v_spec = by_model["WanTI2VPipeline"]["studioExecutionSpecs"][0]
        self.assertEqual(ti2v_spec["mode"], "text_to_video")
        self.assertEqual(ti2v_spec["pipelineClass"], "WanTI2VPipeline")
        self.assertIn("wanGenerate", [item[0] for item in ti2v_spec["roles"]])
        for model_type in (
            "FluxSchnellPipeline",
            "FluxDevPipeline",
            "FluxKreaPipeline",
            "FluxDepthPipeline",
            "FluxCannyPipeline",
            "FluxReduxPipeline",
            "FluxKontextPipeline",
            "FluxFillPipeline",
            "Flux2KleinPipeline",
            "WanImageToVideoPipeline",
            "WanTI2VPipeline",
            "WanVideoPipeline",
            "LTXVideoPipeline",
            "AceStepAudioPipeline",
            "StableAudioPipeline",
            "StableDiffusionXLPipeline",
        ):
            capability = by_model[model_type]
            self.assertEqual(
                capability["studioExecutionSpecModes"],
                sorted(item["mode"] for item in capability["studioExecutionSpecs"]),
            )

        z_image = by_model["ZImageModularPipeline"]
        self.assertEqual(z_image["pipelineClasses"], ["ZImageImg2ImgPipeline", "ZImagePipeline"])
        self.assertEqual(
            z_image["executionProfiles"][0]["backend_path"],
            "modules.DiffusersImage.LoadPipeline",
        )
        self.assertEqual(z_image["executionProfiles"][0]["execution_path"], "direct-diffusers-image")
        self.assertNotIn("expert_quantization_modes", z_image["executionProfiles"][0])
        self.assertEqual(z_image["studioExecutionSpecModes"], ["edit_image", "text_to_image"])
        z_image_text = next(item for item in z_image["studioExecutionSpecs"] if item["mode"] == "text_to_image")
        z_image_edit = next(item for item in z_image["studioExecutionSpecs"] if item["mode"] == "edit_image")
        self.assertEqual(z_image_text["id"], "z-image:text-to-image:v1")
        self.assertEqual(z_image_text["pipelineClass"], "ZImagePipeline")
        self.assertEqual(z_image_edit["id"], "z-image:edit-image:v1")
        self.assertEqual(z_image_edit["pipelineClass"], "ZImageImg2ImgPipeline")
        self.assertEqual(
            z_image["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )

        qwen_image = by_model["QwenImageModularPipeline"]
        self.assertTrue(
            all(profile["expert_quantization_modes"] == ["bnb_4bit"] for profile in qwen_image["executionProfiles"])
        )
        self.assertEqual(
            by_model["FluxSchnellPipeline"]["executionProfiles"][0]["expert_quantization_modes"],
            ["bnb_4bit", "bnb_8bit", "quanto_float8", "torchao_float8"],
        )
        self.assertEqual(
            qwen_image["studioExecutionSpecModes"],
            ["control_image", "edit_image", "inpaint", "text_to_image"],
        )
        qwen_text_spec = next(item for item in qwen_image["studioExecutionSpecs"] if item["mode"] == "text_to_image")
        qwen_edit_spec = next(item for item in qwen_image["studioExecutionSpecs"] if item["mode"] == "edit_image")
        qwen_inpaint_spec = next(item for item in qwen_image["studioExecutionSpecs"] if item["mode"] == "inpaint")
        qwen_control_spec = next(
            item for item in qwen_image["studioExecutionSpecs"] if item["mode"] == "control_image"
        )
        self.assertEqual(qwen_text_spec["id"], "qwen-image-2512:text-to-image:v1")
        self.assertEqual(qwen_text_spec["pipelineClass"], "QwenImagePipeline")
        self.assertEqual(qwen_edit_spec["id"], "qwen-image-2512:edit-image:v1")
        self.assertEqual(qwen_edit_spec["pipelineClass"], "QwenImageImg2ImgPipeline")
        self.assertEqual(qwen_inpaint_spec["id"], "qwen-image-2512:inpaint:v1")
        self.assertEqual(qwen_inpaint_spec["pipelineClass"], "QwenImageInpaintPipeline")
        self.assertEqual(
            qwen_image["modeRequirements"]["edit_image"]["requiredImages"],
            ["referenceImages"],
        )
        self.assertEqual(
            qwen_image["modeRequirements"]["inpaint"]["requiredImages"],
            ["referenceImages", "maskImage"],
        )
        self.assertEqual(qwen_control_spec["executionProfileId"], "qwen-image:modular")
        self.assertEqual(qwen_control_spec["pipelineClass"], "QwenImageModularPipeline")
        self.assertIn(["controlnetModel", "revision", "revision"], qwen_control_spec["bindings"])
        self.assertIn(
            ["controlnet", "route_state_out", "denoise", "route_state_in"],
            qwen_control_spec["edges"],
        )

        qwen_direct_control = by_model["QwenImageControlNetPipeline"]
        qwen_direct_layered = by_model["QwenImageLayeredPipeline"]
        self.assertEqual(qwen_direct_control["runnableModes"], ["control_image"])
        self.assertEqual(qwen_direct_layered["runnableModes"], ["layer_decomposition"])
        for capability in (qwen_direct_control, qwen_direct_layered):
            with self.subTest(model_type=capability["modelType"]):
                self.assertEqual(capability["executionStatus"], "expert_only")
                self.assertEqual(capability["qualificationStatus"], "graph-qualified-execution-pending")
                self.assertEqual(capability["qualifiedModes"], [])
                self.assertFalse(capability["autoEligible"])
                self.assertFalse(capability["galleryEligible"])
                self.assertFalse(capability["liveProof"])

        self.assertEqual(qwen_direct_control["pipelineClasses"], ["QwenImageControlNetPipeline"])
        self.assertEqual(qwen_direct_layered["pipelineClasses"], ["QwenImageLayeredPipeline"])
        self.assertEqual(
            qwen_direct_control["modeRequirements"]["control_image"]["requiredImages"],
            ["controlImage"],
        )
        self.assertEqual(
            qwen_direct_layered["modeRequirements"]["layer_decomposition"]["requiredImages"],
            ["referenceImages"],
        )
        self.assertEqual(qwen_direct_layered["layerCount"], {"default": 4, "min": 1, "max": 10})
        self.assertEqual(qwen_direct_layered["layerResolutions"], [640, 1024])

        wan = by_model["WanVACEPipeline"]
        self.assertEqual(wan["mediaKind"], "video")
        self.assertEqual(wan["pipelineClasses"], ["WanVACEPipeline"])
        self.assertNotIn("video_to_video", wan["runnableModes"])
        self.assertEqual(wan["executionProfiles"][0]["backend_path"], "modules.DiffusersVideo.LoadPipeline")
        self.assertEqual(wan["qualificationStatus"], "qualified")
        self.assertEqual(
            wan["qualifiedModes"],
            ["text_to_video", "video_inpaint", "video_outpaint", "control_to_video"],
        )
        self.assertEqual(
            wan["runnableModes"],
            ["control_to_video", "text_to_video", "video_inpaint", "video_outpaint"],
        )
        self.assertIn("Wan-AI/Wan2.1-VACE-1.3B-diffusers", wan["artifactCandidates"])
        wan_video = by_model["WanVideoPipeline"]
        self.assertEqual(wan_video["pipelineClasses"], ["WanPipeline", "WanVideoToVideoPipeline"])
        self.assertEqual(wan_video["artifactCandidates"], ["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"])
        wan_v2v = next(
            profile for profile in wan_video["executionProfiles"] if profile["id"] == "wan-video-to-video:direct"
        )
        self.assertEqual(wan_v2v["modes"], ["video_to_video", "video_color_edit"])
        wan_t2v = next(
            profile for profile in wan_video["executionProfiles"] if profile["id"] == "wan-text-to-video:direct"
        )
        self.assertEqual(wan_t2v["modes"], ["text_to_video"])
        self.assertEqual(
            wan_video["studioExecutionSpecModes"],
            ["text_to_video", "video_color_edit", "video_to_video"],
        )
        wan_text_spec = next(item for item in wan_video["studioExecutionSpecs"] if item["mode"] == "text_to_video")
        wan_video_spec = next(item for item in wan_video["studioExecutionSpecs"] if item["mode"] == "video_to_video")
        wan_color_spec = next(item for item in wan_video["studioExecutionSpecs"] if item["mode"] == "video_color_edit")
        self.assertEqual(wan_text_spec["pipelineClass"], "WanPipeline")
        self.assertEqual(wan_video_spec["pipelineClass"], "WanVideoToVideoPipeline")
        self.assertEqual(wan_color_spec["pipelineClass"], "WanVideoToVideoPipeline")
        self.assertIn("loadVideo", [item[0] for item in wan_video_spec["roles"]])
        self.assertIn("normalizeVideo", [item[0] for item in wan_video_spec["roles"]])
        self.assertEqual(wan_color_spec["roles"], wan_video_spec["roles"])
        self.assertEqual(wan_color_spec["edges"], wan_video_spec["edges"])
        self.assertEqual(wan_color_spec["bindings"], wan_video_spec["bindings"])
        ltx = by_model["LTXVideoPipeline"]
        self.assertEqual(
            ltx["studioExecutionSpecModes"],
            ["image_to_video", "reference_to_video", "text_to_video", "video_to_video"],
        )
        for ltx_spec in ltx["studioExecutionSpecs"]:
            self.assertEqual(ltx_spec["pipelineClass"], "LTXConditionPipeline")
            self.assertIn(["diffusersRecipe", "attention_backend", "nativeMath"], ltx_spec["bindings"])
            self.assertNotIn(["wanGenerate", "scheduler_flow_shift", "shift"], ltx_spec["bindings"])
        ltx_image_spec = next(item for item in ltx["studioExecutionSpecs"] if item["mode"] == "image_to_video")
        self.assertIn("loadImage", [item[0] for item in ltx_image_spec["roles"]])
        self.assertIn(["loadImage", "image", "wanGenerate", "reference_images"], ltx_image_spec["edges"])
        ltx_video_spec = next(item for item in ltx["studioExecutionSpecs"] if item["mode"] == "video_to_video")
        self.assertIn("loadVideo", [item[0] for item in ltx_video_spec["roles"]])
        self.assertIn(["normalizeVideo", "output", "wanGenerate", "video"], ltx_video_spec["edges"])
        self.assertIn(["wanGenerate", "strength", "conditioningScale"], ltx_video_spec["bindings"])
        self.assertIn(["wanGenerate", "denoise_strength", "strength"], ltx_video_spec["bindings"])
        ltx_reference_spec = next(item for item in ltx["studioExecutionSpecs"] if item["mode"] == "reference_to_video")
        self.assertEqual(ltx_reference_spec["roles"], ltx_image_spec["roles"])
        self.assertEqual(ltx_reference_spec["edges"], ltx_image_spec["edges"])
        self.assertEqual(ltx_reference_spec["bindings"], ltx_image_spec["bindings"])
        ace = by_model["AceStepAudioPipeline"]
        self.assertEqual(
            ace["studioExecutionSpecModes"],
            ["audio_continuation", "audio_repaint", "audio_variation", "text_to_audio"],
        )
        ace_text_spec = next(item for item in ace["studioExecutionSpecs"] if item["mode"] == "text_to_audio")
        ace_variation_spec = next(item for item in ace["studioExecutionSpecs"] if item["mode"] == "audio_variation")
        ace_continuation_spec = next(
            item for item in ace["studioExecutionSpecs"] if item["mode"] == "audio_continuation"
        )
        ace_repaint_spec = next(item for item in ace["studioExecutionSpecs"] if item["mode"] == "audio_repaint")
        self.assertEqual(ace_text_spec["pipelineClass"], "AceStepPipeline")
        self.assertEqual(
            [item[0] for item in ace_text_spec["roles"]],
            ["diffusersQuantization", "diffusersRecipe", "audioPipeline", "audioGenerate", "audioExport"],
        )
        self.assertIn(["audioGenerate", "task_type", "text2music"], ace_text_spec["bindings"])
        self.assertIn(["audioGenerate", "audio", "audioExport", "audio"], ace_text_spec["edges"])
        self.assertEqual(ace_variation_spec["pipelineClass"], "AceStepPipeline")
        self.assertIn("loadAudio", [item[0] for item in ace_variation_spec["roles"]])
        self.assertIn(["loadAudio", "file", "sourceAudio"], ace_variation_spec["bindings"])
        self.assertIn(["audioGenerate", "task_type", "cover"], ace_variation_spec["bindings"])
        self.assertIn(
            ["loadAudio", "audio", "audioGenerate", "source_audio"],
            ace_variation_spec["edges"],
        )
        self.assertNotEqual(ace_variation_spec["contentHash"], ace_text_spec["contentHash"])
        self.assertEqual(ace_continuation_spec["pipelineClass"], "AceStepPipeline")
        self.assertIn("audioLoudnessMatch", [item[0] for item in ace_continuation_spec["roles"]])
        self.assertIn("audioJoin", [item[0] for item in ace_continuation_spec["roles"]])
        self.assertIn(
            ["audioGenerate", "task_type", "continuation"],
            ace_continuation_spec["bindings"],
        )
        self.assertIn(
            ["audioJoin", "output", "audioExport", "audio"],
            ace_continuation_spec["edges"],
        )
        self.assertNotEqual(ace_continuation_spec["contentHash"], ace_variation_spec["contentHash"])
        self.assertEqual(ace_repaint_spec["roles"], ace_variation_spec["roles"])
        self.assertEqual(ace_repaint_spec["edges"], ace_variation_spec["edges"])
        self.assertIn(["audioGenerate", "task_type", "repaint"], ace_repaint_spec["bindings"])
        self.assertNotEqual(ace_repaint_spec["contentHash"], ace_continuation_spec["contentHash"])
        self.assertEqual(
            ace["runnableModes"],
            ["audio_continuation", "audio_repaint", "audio_variation", "text_to_audio"],
        )
        stable_audio = by_model["StableAudioPipeline"]
        self.assertEqual(stable_audio["runnableModes"], ["text_to_audio"])
        self.assertEqual(stable_audio["studioExecutionSpecModes"], ["text_to_audio"])
        self.assertEqual(stable_audio["studioExecutionSpecs"][0]["pipelineClass"], "StableAudioPipeline")
        self.assertEqual(stable_audio["revisionCandidates"], ["f21265c1e2710b3bd2386596943f0007f55f802e"])
        longcat = by_model["LongCatAudioDiTPipeline"]
        self.assertEqual(longcat["runnableModes"], ["text_to_audio"])
        self.assertEqual(longcat["recommendedSampleRate"], 24000)
        self.assertEqual(longcat["recommendedDuration"], 5)
        self.assertEqual(longcat["revisionCandidates"], ["f4c063ea37f262ba5e6129ebd80095a6d6a9de4d"])
        audioldm2 = by_model["AudioLDM2Pipeline"]
        self.assertEqual(audioldm2["runnableModes"], ["text_to_audio"])
        self.assertEqual(audioldm2["recommendedSampleRate"], 16000)
        self.assertEqual(audioldm2["recommendedDuration"], 10)
        self.assertEqual(audioldm2["revisionCandidates"], ["c8e7e189d324425c05c4c2f81214041ef4107983"])
        shap_e = by_model["ShapEPipeline"]
        self.assertEqual(shap_e["runnableModes"], ["text_to_3d"])
        self.assertEqual(shap_e["outputKind"], "video")
        self.assertEqual(shap_e["recommendedFrames"], 20)
        self.assertEqual(shap_e["revisionCandidates"], ["7bd337afdea1c17842e1c3cc45c4e268356dba40"])
        self.assertEqual(shap_e["studioExecutionSpecs"][0]["executionPath"], "direct-diffusers-three-d")
        stable_video = by_model["StableVideoDiffusionPipeline"]
        self.assertEqual(stable_video["runnableModes"], ["image_to_video"])
        self.assertEqual(stable_video["defaultDtype"], "float16")
        self.assertEqual(stable_video["defaultSize"], {"width": 1024, "height": 576, "aspectRatio": "16:9"})
        self.assertEqual(stable_video["recommendedSteps"], 25)
        self.assertEqual(stable_video["recommendedFrames"], 25)
        self.assertEqual(stable_video["recommendedFps"], 7)
        self.assertEqual(
            stable_video["revisionCandidates"],
            ["043843887ccd51926e3efed36270444a838e7861"],
        )
        self.assertFalse(stable_video["autoEligible"])
        self.assertFalse(stable_video["galleryEligible"])
        self.assertEqual(
            by_model["FluxKontextPipeline"]["studioExecutionSpecModes"],
            ["edit_image", "multi_image_reference_edit"],
        )
        qwen_edit = by_model["QwenImageEditModularPipeline"]
        self.assertEqual(qwen_edit["studioExecutionSpecModes"], ["edit_image", "inpaint", "outpaint"])
        qwen_inpaint_spec, qwen_outpaint_spec, qwen_edit_spec = qwen_edit["studioExecutionSpecs"]
        self.assertEqual(qwen_inpaint_spec["executionProfileId"], "qwen-edit:direct-inpaint")
        self.assertEqual(qwen_inpaint_spec["pipelineClass"], "QwenImageEditInpaintPipeline")
        self.assertIn("loadMask", [item[0] for item in qwen_inpaint_spec["roles"]])
        self.assertIn(
            ["loadMask", "image", "diffusersImageInpaint", "mask_image"],
            qwen_inpaint_spec["edges"],
        )
        self.assertEqual(qwen_outpaint_spec["executionProfileId"], "qwen-edit:direct-inpaint")
        self.assertIn("qwenOutpaintCanvas", [item[0] for item in qwen_outpaint_spec["roles"]])
        self.assertIn(
            ["qwenOutpaintCanvas", "mask_image", "diffusersImageInpaint", "mask_image"],
            qwen_outpaint_spec["edges"],
        )
        self.assertEqual(qwen_edit_spec["executionProfileId"], "qwen-edit:modular")
        self.assertEqual(qwen_edit_spec["executionPath"], "modular-diffusers")
        self.assertEqual(qwen_edit_spec["pipelineClass"], "QwenImageEditModularPipeline")
        self.assertIn("models", [item[0] for item in qwen_edit_spec["roles"]])
        self.assertIn(["imageEncode", "image_latents", "denoise", "image_latents"], qwen_edit_spec["edges"])
        qwen_layered = by_model["QwenImageLayeredModularPipeline"]
        self.assertEqual(qwen_layered["studioExecutionSpecModes"], ["layer_decomposition"])
        qwen_layered_spec = qwen_layered["studioExecutionSpecs"][0]
        self.assertEqual(qwen_layered_spec["executionProfileId"], "qwen-layered:modular")
        self.assertEqual(qwen_layered_spec["executionPath"], "modular-diffusers")
        self.assertEqual(qwen_layered_spec["pipelineClass"], "QwenImageLayeredModularPipeline")
        self.assertIn(["denoise", "layers", "layers"], qwen_layered_spec["bindings"])
        wan_vace = by_model["WanVACEPipeline"]
        self.assertEqual(
            wan_vace["studioExecutionSpecModes"],
            ["control_to_video", "text_to_video", "video_inpaint", "video_outpaint"],
        )
        wan_vace_text_spec, wan_vace_inpaint_spec, wan_vace_outpaint_spec, wan_vace_control_spec = wan_vace[
            "studioExecutionSpecs"
        ]
        self.assertEqual(wan_vace_control_spec["mode"], "control_to_video")
        self.assertIn(["loadControlVideo", "file", "controlVideo"], wan_vace_control_spec["bindings"])
        self.assertIn(
            ["normalizeVideo", "output", "wanGenerate", "video"],
            wan_vace_control_spec["edges"],
        )
        self.assertEqual(wan_vace_text_spec["executionProfileId"], "wan-vace:direct")
        self.assertEqual(wan_vace_text_spec["executionPath"], "direct-wan-vace")
        self.assertEqual(wan_vace_text_spec["pipelineClass"], "WanVACEPipeline")
        self.assertIn(["wanGenerate", "mode", "mode"], wan_vace_text_spec["bindings"])
        self.assertEqual(wan_vace_inpaint_spec["mode"], "video_inpaint")
        self.assertIn(["loadMaskVideo", "file", "maskVideo"], wan_vace_inpaint_spec["bindings"])
        self.assertIn(
            ["alignMaskVideo", "output", "wanGenerate", "mask"],
            wan_vace_inpaint_spec["edges"],
        )
        self.assertEqual(wan_vace_outpaint_spec["mode"], "video_outpaint")
        self.assertIn(
            ["alignMaskVideo", "grow_pixels", "outpaintMaskGrow0"],
            wan_vace_outpaint_spec["bindings"],
        )
        self.assertNotEqual(wan_vace_outpaint_spec["contentHash"], wan_vace_inpaint_spec["contentHash"])

        ltx = by_model["LTXVideoPipeline"]
        self.assertEqual(ltx["mediaKind"], "video")
        self.assertEqual(ltx["supportTier"], "supported")
        self.assertEqual(ltx["qualificationStatus"], "qualified")
        self.assertEqual(
            ltx["qualifiedModes"],
            ["text_to_video", "image_to_video", "video_to_video", "reference_to_video"],
        )
        self.assertEqual(ltx["pipelineClasses"], ["LTXConditionPipeline"])
        self.assertEqual(
            ltx["runnableModes"],
            ["image_to_video", "reference_to_video", "text_to_video", "video_to_video"],
        )
        self.assertEqual(ltx["executionProfiles"][0]["backend_path"], "modules.DiffusersVideo.LoadPipeline")
        self.assertEqual(ltx["maxPromptTokens"], 128)
        self.assertEqual(ltx["defaultRepo"], "Lightricks/LTX-Video-0.9.8-13B-distilled")
        self.assertEqual(
            ltx["artifactCandidates"],
            ["Lightricks/LTX-Video-0.9.8-13B-distilled"],
        )
        self.assertEqual(len(ltx["downloadFiles"]), 22)
        self.assertNotIn("ltxv-13b-0.9.8-dev.safetensors", ltx["downloadFiles"])

        canny = by_model["FluxCannyPipeline"]
        self.assertEqual(
            canny["artifactCandidates"][:2],
            [
                "black-forest-labs/FLUX.1-Canny-dev",
                "fuliucansheng/FLUX.1-Canny-dev-diffusers",
            ],
        )
        self.assertEqual(
            canny["verifiedRepairSources"][0]["repo"],
            "fuliucansheng/FLUX.1-Canny-dev-diffusers",
        )
        self.assertIn("guidance_scale", wan["parameterAliases"]["guidanceScale"])
        self.assertIn("num_inference_steps", wan["parameterAliases"]["steps"])
        self.assertEqual(wan["defaults"]["dtype"], "bfloat16")

        qwen_inpaint = by_model["QwenImageEditModularPipeline"]
        self.assertEqual(qwen_inpaint["inpaintContract"]["source"], "modules.DiffusersImage.Inpaint")
        self.assertIn("QwenImageEditInpaintPipeline", qwen_inpaint["pipelineClasses"])
        self.assertEqual(qwen_inpaint["modes"], ["edit_image", "inpaint", "outpaint"])
        self.assertEqual(qwen_inpaint["runnableModes"], ["edit_image", "inpaint", "outpaint"])

        qwen_control = by_model["QwenImageModularPipeline"]
        control_requirement = qwen_control["modeRequirements"]["control_image"]["modelRequirements"][0]
        self.assertEqual(control_requirement["repo"], "InstantX/Qwen-Image-ControlNet-Union")
        self.assertEqual(control_requirement["revision"], "b13036f066d6dee7c20513e263d3d673055e9de8")
        self.assertEqual(qwen_control["additionalRequirements"][0], control_requirement)

        blocked = by_model["QwenImageEditPlusModularPipeline"]
        self.assertEqual(blocked["modes"], ["edit_image", "multi_image_reference_edit"])
        self.assertEqual(blocked["runnableModes"], ["edit_image", "multi_image_reference_edit"])
        self.assertNotIn("inpaint", blocked["runnableModes"])
        self.assertFalse(blocked["inpaintContract"]["available"])
        self.assertEqual(blocked["inpaintContract"]["status"], "blocked")
        self.assertEqual(blocked["studioExecutionSpecModes"], ["edit_image", "multi_image_reference_edit"])
        self.assertEqual(
            [item["executionProfileId"] for item in blocked["studioExecutionSpecs"]],
            ["qwen-edit-plus:modular", "qwen-edit-plus:modular"],
        )
        self.assertTrue(all(item["executionPath"] == "modular-diffusers" for item in blocked["studioExecutionSpecs"]))

    async def test_contract_only_capabilities_close_registered_unprofiled_adapters(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        payload = json.loads(response.text)
        published = {
            item["modelType"]: item
            for item in payload["experimentalCapabilities"]
            if item.get("qualificationStatus") == "contract_only" and item.get("executionKind") == "standard"
        }

        profiled_classes = {profile.pipeline_class for profile in DIFFUSERS_EXECUTION_PROFILES.values()}
        adapters_by_media = {
            "image": IMAGE_PIPELINE_ADAPTERS,
            "video": VIDEO_PIPELINE_ADAPTERS,
            "audio": AUDIO_PIPELINE_ADAPTERS,
        }
        expected_classes = {
            pipeline_class
            for adapters in adapters_by_media.values()
            for pipeline_class in adapters
            if pipeline_class not in profiled_classes
        }
        declared_classes = {
            pipeline_class for pipeline_class, _media_kind, _repo, _modes in CONTRACT_ONLY_DIFFUSERS_PIPELINES
        }
        self.assertEqual(declared_classes, expected_classes)
        self.assertEqual(set(published), expected_classes)
        self.assertTrue(expected_classes.isdisjoint(AUTO_MODEL_REQUIREMENTS))
        self.assertTrue(expected_classes.isdisjoint(capability["modelType"] for capability in payload["capabilities"]))

        for pipeline_class, media_kind, repository, declared_modes in CONTRACT_ONLY_DIFFUSERS_PIPELINES:
            with self.subTest(pipeline_class=pipeline_class):
                adapter = adapters_by_media[media_kind][pipeline_class]
                adapter_modes = adapter.mode_options if media_kind == "image" else adapter.modes
                capability = published[pipeline_class]
                self.assertEqual(tuple(declared_modes), tuple(adapter_modes))
                self.assertEqual(repository, adapter.default_repo)
                self.assertEqual(capability["pipelineClasses"], [pipeline_class])
                self.assertEqual(capability["runnableModes"], list(adapter_modes))
                self.assertEqual(capability["mediaKind"], media_kind)
                self.assertEqual(
                    capability["backendPath"],
                    f"modules.Diffusers{media_kind.title()}.LoadPipeline",
                )
                self.assertEqual(capability["artifactCandidates"], [adapter.default_repo])
                self.assertEqual(
                    capability["revisionCandidates"],
                    [catalog_revision(adapter.default_repo)],
                )
                self.assertFalse(capability["autoEligible"])
                self.assertFalse(capability["templateEligible"])
                self.assertFalse(capability["galleryEligible"])
                self.assertEqual(capability["executionProfiles"], [])
                self.assertNotIn("optionalRuntimeRequirement", capability)


if __name__ == "__main__":
    unittest.main()
