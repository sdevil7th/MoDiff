import json
import unittest

import modules as module_registry
from modiff.auto_resource import AUTO_MODEL_REQUIREMENTS
from modiff.diffusers_profiles import (
    CONTRACT_ONLY_DIFFUSERS_PIPELINES,
    DIFFUSERS_EXECUTION_PROFILES,
)
from modiff.model_artifact_catalog import catalog_revision
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
        self.assertEqual(len(payload["experimentalCapabilities"]), 25)
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

        self.assertEqual(len(payload["studioExecutionSpecs"]), 7)
        for model_type in (
            "FluxSchnellPipeline",
            "FluxDevPipeline",
            "FluxKreaPipeline",
            "FluxDepthPipeline",
            "FluxCannyPipeline",
            "FluxReduxPipeline",
            "WanTI2VPipeline",
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
        depth_spec = by_model["FluxDepthPipeline"]["studioExecutionSpecs"][0]
        self.assertEqual(by_model["FluxDepthPipeline"]["modes"], ["control_image"])
        self.assertEqual(depth_spec["mode"], "control_image")
        self.assertIn("diffusersImageControl", [item[0] for item in depth_spec["roles"]])
        self.assertIn("loadImage", [item[0] for item in depth_spec["roles"]])

        canny_spec = by_model["FluxCannyPipeline"]["studioExecutionSpecs"][0]
        self.assertEqual(by_model["FluxCannyPipeline"]["modes"], ["control_image"])
        self.assertEqual(canny_spec["mode"], "control_image")
        self.assertEqual(canny_spec["roles"], depth_spec["roles"])
        self.assertEqual(canny_spec["edges"], depth_spec["edges"])

        redux_spec = by_model["FluxReduxPipeline"]["studioExecutionSpecs"][0]
        self.assertEqual(by_model["FluxReduxPipeline"]["modes"], ["edit_image"])
        self.assertEqual(redux_spec["mode"], "edit_image")
        self.assertIn("diffusersImageEdit", [item[0] for item in redux_spec["roles"]])

        ti2v_spec = by_model["WanTI2VPipeline"]["studioExecutionSpecs"][0]
        self.assertEqual(ti2v_spec["mode"], "text_to_video")
        self.assertEqual(ti2v_spec["pipelineClass"], "WanTI2VPipeline")
        self.assertIn("wanGenerate", [item[0] for item in ti2v_spec["roles"]])

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

    async def test_contract_only_capabilities_close_registered_unprofiled_adapters(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        payload = json.loads(response.text)
        published = {
            item["modelType"]: item
            for item in payload["experimentalCapabilities"]
            if item.get("qualificationStatus") == "contract_only"
        }

        profiled_classes = {
            profile.pipeline_class for profile in DIFFUSERS_EXECUTION_PROFILES.values()
        }
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
            pipeline_class
            for pipeline_class, _media_kind, _repo, _modes in CONTRACT_ONLY_DIFFUSERS_PIPELINES
        }
        self.assertEqual(declared_classes, expected_classes)
        self.assertEqual(set(published), expected_classes)
        self.assertTrue(expected_classes.isdisjoint(AUTO_MODEL_REQUIREMENTS))
        self.assertTrue(
            expected_classes.isdisjoint(
                capability["modelType"] for capability in payload["capabilities"]
            )
        )

        for pipeline_class, media_kind, repository, declared_modes in CONTRACT_ONLY_DIFFUSERS_PIPELINES:
            with self.subTest(pipeline_class=pipeline_class):
                adapter = adapters_by_media[media_kind][pipeline_class]
                adapter_modes = (
                    adapter.mode_options
                    if media_kind == "image"
                    else adapter.modes
                )
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
