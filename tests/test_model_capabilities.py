import json
import unittest

import modules as module_registry
from modiff.server import WebServer


class FakeRequest:
    query = {}


class ModelCapabilitiesTests(unittest.IsolatedAsyncioTestCase):
    async def test_capabilities_publish_normalized_execution_contract(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        payload = json.loads(response.text)
        self.assertEqual(payload["schemaVersion"], 2)
        self.assertEqual(len(payload["experimentalCapabilities"]), 6)
        self.assertTrue(all(item["supportTier"] == "experimental" for item in payload["experimentalCapabilities"]))
        experimental = {item["modelType"]: item for item in payload["experimentalCapabilities"]}
        self.assertNotIn("DiffusionGemmaForBlockDiffusion", experimental)
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
        for capability in payload["experimentalCapabilities"]:
            self.assertIn("executionProfiles", capability)
            self.assertIn("inputContracts", capability)
            self.assertIn("parameterAliases", capability)
            self.assertIn("defaults", capability)
            self.assertIn("artifactCandidates", capability)
            self.assertIn("revisionCandidates", capability)
            self.assertIn("quantizationSupport", capability)
        by_model = {item["modelType"]: item for item in payload["capabilities"]}

        z_image = by_model["ZImageModularPipeline"]
        self.assertEqual(z_image["pipelineClasses"], ["ZImagePipeline"])
        self.assertEqual(
            z_image["executionProfiles"][0]["backend_path"],
            "modules.DiffusersImage.LoadPipeline",
        )
        self.assertEqual(z_image["executionProfiles"][0]["execution_path"], "direct-diffusers-image")

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
        wan_t2v = next(profile for profile in wan_video["executionProfiles"] if profile["id"] == "wan-text-to-video:direct")
        self.assertEqual(wan_t2v["modes"], ["text_to_video"])

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
        self.assertIn("Lightricks/LTX-Video", ltx["artifactCandidates"])
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


if __name__ == "__main__":
    unittest.main()
