import json
import unittest
from copy import deepcopy
from pathlib import Path

import modules as module_registry

from modiff.server import WebServer
from modiff.task_template_contracts import (
    TaskTemplateContractError,
    contracts_by_pair,
    validate_task_template_graph,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
GRAPH_ROOT = ROOT / "data" / "graphs"


class FakeRequest:
    query = {}


class TaskTemplateContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        response = await WebServer(module_registry.MODULE_MAP).model_capabilities(FakeRequest())
        self.payload = json.loads(response.text)
        self.contracts = self.payload["taskTemplateContracts"]
        self.contract_by_pair = contracts_by_pair(self.contracts)
        self.spec_by_pair = {
            (specification["modelType"], specification["mode"]): specification
            for specification in self.payload["studioExecutionSpecs"]
        }

    async def test_every_execution_spec_has_one_exact_stable_task_contract(self):
        self.assertEqual(self.payload["taskTemplateContractSchemaVersion"], 1)
        self.assertEqual(len(self.contracts), 130)
        self.assertEqual(set(self.contract_by_pair), set(self.spec_by_pair))
        self.assertEqual(self.contracts, sorted(self.contracts, key=lambda item: item["id"]))
        self.assertEqual(self.contracts, json.loads(json.dumps(self.contracts)))

        capabilities = {item["modelType"]: item for item in self.payload["capabilities"]}
        for contract in self.contracts:
            with self.subTest(contract=contract["id"]):
                pair = (contract["modelType"], contract["mode"])
                specification = self.spec_by_pair[pair]
                capability = capabilities[contract["modelType"]]
                profiles = [
                    profile
                    for profile in capability["executionProfiles"]
                    if profile["id"] == contract["executionProfileId"]
                ]
                self.assertEqual(len(profiles), 1)
                profile = profiles[0]
                self.assertEqual(contract["id"], f"task-template:{specification['id']}")
                self.assertRegex(contract["contentHash"], r"^task-template-v1-[0-9a-f]{8}$")
                self.assertEqual(contract["executionSpecContentHash"], specification["contentHash"])
                self.assertEqual(contract["loaderModule"], profile["loader_module"])
                self.assertEqual(contract["loaderAction"], profile["loader_action"])
                self.assertEqual(contract["pipelineClass"], profile["pipeline_class"])
                self.assertEqual(contract["defaultRepo"], profile["default_repo"])
                self.assertEqual(contract["loaderRepositories"][0], profile["default_repo"])
                self.assertIn(
                    [
                        contract["loaderRole"],
                        f"{contract['loaderModule']}.{contract['loaderAction']}",
                        next(
                            role[2]
                            for role in specification["roles"]
                            if role[0] == contract["loaderRole"]
                        ),
                        next(
                            role[3]
                            for role in specification["roles"]
                            if role[0] == contract["loaderRole"]
                        ),
                    ],
                    specification["roles"],
                )
                self.assertFalse(contract["galleryEligible"])

    async def test_required_media_and_terminal_outputs_are_generic_and_exact(self):
        expected = {
            ("StableDiffusionXLPipeline", "text_to_image"): [],
            ("StableDiffusionXLTurboPipeline", "text_to_image"): [],
            ("StableDiffusionXLInstructPix2PixPipeline", "edit_image"): [
                ("image", "referenceImages"),
            ],
            ("StableDiffusionXLControlNetPipeline", "control_image"): [
                ("image", "controlImage"),
            ],
            ("HunyuanDiTPipeline", "text_to_image"): [],
            ("HunyuanDiTPAGPipeline", "text_to_image"): [],
            ("HunyuanDiTControlNetPipeline", "control_image"): [
                ("image", "controlImage"),
            ],
            ("StableDiffusionXLAdapterPipeline", "control_image"): [
                ("image", "controlImage"),
            ],
            ("StableDiffusionXLPAGPipeline", "text_to_image"): [],
            ("StableDiffusionXLPAGPipeline", "edit_image"): [
                ("image", "referenceImages"),
            ],
            ("StableDiffusionXLPAGPipeline", "inpaint"): [
                ("image", "referenceImages"),
                ("image", "maskImage"),
            ],
            ("SanaPipeline", "text_to_image"): [],
            ("SanaPAGPipeline", "text_to_image"): [],
            ("SanaSprintPipeline", "text_to_image"): [],
            ("SanaSprintPipeline", "edit_image"): [("image", "referenceImages")],
            ("PixArtSigmaPipeline", "text_to_image"): [],
            ("PixArtSigmaPAGPipeline", "text_to_image"): [],
            ("Kandinsky3Pipeline", "text_to_image"): [],
            ("Kandinsky3Pipeline", "edit_image"): [("image", "referenceImages")],
            ("OmniGenPipeline", "text_to_image"): [],
            ("OmniGenPipeline", "edit_image"): [("image", "referenceImages")],
            ("OmniGenPipeline", "multi_image_reference_edit"): [("image", "referenceImages")],
            ("OvisImagePipeline", "text_to_image"): [],
            ("PRXPipeline", "text_to_image"): [],
            ("NucleusMoEImagePipeline", "text_to_image"): [],
            ("AuraFlowPipeline", "text_to_image"): [],
            ("ChromaPipeline", "text_to_image"): [],
            ("CogView3PlusPipeline", "text_to_image"): [],
            ("CogView4Pipeline", "text_to_image"): [],
            ("ErnieImagePipeline", "text_to_image"): [],
            ("GlmImagePipeline", "text_to_image"): [],
            ("DreamLitePipeline", "text_to_image"): [],
            ("DreamLitePipeline", "edit_image"): [("image", "referenceImages")],
            ("DreamLiteMobilePipeline", "text_to_image"): [],
            ("DreamLiteMobilePipeline", "edit_image"): [("image", "referenceImages")],
            ("ShapEPipeline", "text_to_3d"): [],
            ("StableDiffusionXLPipeline", "edit_image"): [
                ("image", "referenceImages"),
            ],
            ("StableDiffusionXLPipeline", "inpaint"): [
                ("image", "referenceImages"),
                ("image", "maskImage"),
            ],
            ("StableDiffusionPipeline", "control_image"): [("image", "controlImage")],
            ("LatentConsistencyModelPipeline", "edit_image"): [
                ("image", "referenceImages"),
            ],
            ("StableDiffusionPAGPipeline", "edit_image"): [
                ("image", "referenceImages"),
            ],
            ("StableDiffusionPAGPipeline", "inpaint"): [
                ("image", "referenceImages"),
                ("image", "maskImage"),
            ],
            ("FluxDevPipeline", "edit_image"): [("image", "referenceImages")],
            ("FluxDevPipeline", "inpaint"): [
                ("image", "referenceImages"),
                ("image", "maskImage"),
            ],
            ("FluxReduxPipeline", "edit_image"): [("image", "referenceImages")],
            ("FluxReduxPipeline", "multi_image_reference_edit"): [
                ("image", "referenceImages"),
            ],
            ("ZImageModularPipeline", "edit_image"): [("image", "referenceImages")],
            ("QwenImageModularPipeline", "edit_image"): [("image", "referenceImages")],
            ("QwenImageModularPipeline", "inpaint"): [
                ("image", "referenceImages"),
                ("image", "maskImage"),
            ],
            ("FluxDepthPipeline", "control_image"): [("image", "controlImage")],
            ("FluxFillPipeline", "inpaint"): [
                ("image", "referenceImages"),
                ("image", "maskImage"),
            ],
            ("WanVACEPipeline", "video_inpaint"): [
                ("video", "sourceVideo"),
                ("video", "maskVideo"),
            ],
            ("Wan22Pipeline", "text_to_video"): [],
            ("WanAnimatePipeline", "character_animate"): [
                ("image", "referenceImages"),
                ("video", "poseVideo"),
                ("video", "faceVideo"),
            ],
            ("WanAnimatePipeline", "character_replace"): [
                ("image", "referenceImages"),
                ("video", "poseVideo"),
                ("video", "faceVideo"),
                ("video", "backgroundVideo"),
                ("video", "maskVideo"),
            ],
            ("WanImage2VideoModularPipeline", "image_to_video"): [
                ("image", "referenceImages"),
                ("image", "lastImage"),
            ],
            ("LTXI2VLongMultiPromptPipeline", "image_to_video"): [("image", "referenceImages")],
            ("LTX2ConditionPipeline", "text_to_video"): [],
            ("LTX2ConditionPipeline", "image_to_video"): [("image", "referenceImages")],
            ("LTX2ConditionPipeline", "reference_to_video"): [("image", "referenceImages")],
            ("LTX2ConditionPipeline", "video_to_video"): [("video", "sourceVideo")],
            ("HunyuanVideoFramepackPipeline", "image_to_video"): [("image", "referenceImages")],
            ("StableVideoDiffusionPipeline", "image_to_video"): [("image", "referenceImages")],
            ("CogVideoXPipeline", "text_to_video"): [],
            ("AllegroPipeline", "text_to_video"): [],
            ("LattePipeline", "text_to_video"): [],
            ("MochiPipeline", "text_to_video"): [],
            ("SanaVideoPipeline", "text_to_video"): [],
            ("SanaImageToVideoPipeline", "image_to_video"): [("image", "referenceImages")],
            ("AceStepAudioPipeline", "audio_continuation"): [("audio", "sourceAudio")],
            ("MarigoldDepthPipeline", "depth_estimation"): [("image", "referenceImages")],
            ("HuggingFaceTextGenerationModel", "text_generation"): [],
            ("HuggingFaceImageTextToTextModel", "image_to_text"): [
                ("image", "referenceImages")
            ],
            ("HuggingFaceSpeechRecognitionModel", "speech_to_text"): [("audio", "sourceAudio")],
            ("HuggingFaceSpeechRecognitionModel", "speech_translation"): [("audio", "sourceAudio")],
        }
        for pair, required in expected.items():
            with self.subTest(pair=pair):
                contract = self.contract_by_pair[pair]
                self.assertEqual(
                    [(item["kind"], item["field"]) for item in contract["requiredMedia"]],
                    required,
                )
                self.assertTrue(all(item["minimumCount"] == 1 for item in contract["requiredMedia"]))

        output_nodes = {
            "image": {"modules.Image.Preview"},
            "video": {"modules.Video.Export", "modules.Video.ExportWithAudio"},
            "audio": {"modules.Audio.Export"},
            "json": {"modules.Primitive.DataViewer"},
        }
        for contract in self.contracts:
            with self.subTest(output=contract["id"]):
                self.assertIn(contract["output"]["nodeKey"], output_nodes[contract["mediaKind"]])
        for mode in ("text_to_video", "image_to_video", "reference_to_video", "video_to_video"):
            output = self.contract_by_pair[("LTX2ConditionPipeline", mode)]["output"]
            self.assertEqual(output["nodeKey"], "modules.Video.ExportWithAudio")
            self.assertEqual(output["role"], "videoExport")
            self.assertEqual(output["inputHandle"], "video")

    async def test_image_video_and_audio_graphs_round_trip_against_the_generic_contract(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        workflows = [*manifest["workflows"], *manifest["experimentalWorkflows"]]
        representatives = {
            ("StableDiffusionXLPipeline", "text_to_image"),
            ("StableDiffusionXLPipeline", "edit_image"),
            ("StableDiffusionXLPipeline", "inpaint"),
            ("StableDiffusionXLTurboPipeline", "text_to_image"),
            ("StableDiffusionXLInstructPix2PixPipeline", "edit_image"),
            ("StableDiffusionXLControlNetPipeline", "control_image"),
            ("HunyuanDiTPAGPipeline", "text_to_image"),
            ("HunyuanDiTControlNetPipeline", "control_image"),
            ("StableDiffusionXLAdapterPipeline", "control_image"),
            ("StableDiffusionXLPAGPipeline", "text_to_image"),
            ("StableDiffusionXLPAGPipeline", "edit_image"),
            ("StableDiffusionXLPAGPipeline", "inpaint"),
            ("SanaPipeline", "text_to_image"),
            ("SanaPAGPipeline", "text_to_image"),
            ("SanaSprintPipeline", "text_to_image"),
            ("SanaSprintPipeline", "edit_image"),
            ("PixArtSigmaPipeline", "text_to_image"),
            ("PixArtSigmaPAGPipeline", "text_to_image"),
            ("Kandinsky3Pipeline", "text_to_image"),
            ("Kandinsky3Pipeline", "edit_image"),
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
            ("DreamLitePipeline", "text_to_image"),
            ("DreamLitePipeline", "edit_image"),
            ("DreamLiteMobilePipeline", "text_to_image"),
            ("DreamLiteMobilePipeline", "edit_image"),
            ("StableDiffusionPipeline", "text_to_image"),
            ("StableDiffusionPipeline", "edit_image"),
            ("StableDiffusionPipeline", "inpaint"),
            ("LatentConsistencyModelPipeline", "text_to_image"),
            ("LatentConsistencyModelPipeline", "edit_image"),
            ("StableDiffusionPAGPipeline", "text_to_image"),
            ("StableDiffusionPAGPipeline", "edit_image"),
            ("StableDiffusionPAGPipeline", "inpaint"),
            ("FluxDevPipeline", "edit_image"),
            ("FluxDevPipeline", "inpaint"),
            ("FluxReduxPipeline", "edit_image"),
            ("FluxReduxPipeline", "multi_image_reference_edit"),
            ("ZImageModularPipeline", "edit_image"),
            ("QwenImageModularPipeline", "edit_image"),
            ("QwenImageModularPipeline", "inpaint"),
            ("FluxFillPipeline", "inpaint"),
            ("WanVACEPipeline", "video_inpaint"),
            ("Wan22Pipeline", "text_to_video"),
            ("WanAnimatePipeline", "character_animate"),
            ("WanAnimatePipeline", "character_replace"),
            ("LTXI2VLongMultiPromptPipeline", "image_to_video"),
            ("LTX2ConditionPipeline", "text_to_video"),
            ("LTX2ConditionPipeline", "image_to_video"),
            ("LTX2ConditionPipeline", "reference_to_video"),
            ("LTX2ConditionPipeline", "video_to_video"),
            ("HunyuanVideoFramepackPipeline", "image_to_video"),
            ("StableVideoDiffusionPipeline", "image_to_video"),
            ("WanImage2VideoModularPipeline", "image_to_video"),
            ("AceStepAudioPipeline", "audio_continuation"),
            ("StableAudioPipeline", "text_to_audio"),
            ("LongCatAudioDiTPipeline", "text_to_audio"),
            ("AudioLDM2Pipeline", "text_to_audio"),
        }
        selected = [
            workflow
            for workflow in workflows
            if (workflow["modelType"], workflow["mode"]) in representatives and "variant" not in workflow
        ]
        self.assertEqual({(item["modelType"], item["mode"]) for item in selected}, representatives)
        for workflow in selected:
            pair = (workflow["modelType"], workflow["mode"])
            with self.subTest(graph=workflow["graphPath"]):
                graph = json.loads((GRAPH_ROOT / workflow["graphPath"]).read_text(encoding="utf-8"))
                round_trip = json.loads(json.dumps(graph))
                self.assertEqual(round_trip, graph)
                validate_task_template_graph(
                    round_trip,
                    workflow,
                    self.contract_by_pair[pair],
                    self.spec_by_pair[pair],
                )

    async def test_loader_output_and_required_media_tampering_fail_closed(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        workflow = next(
            item
            for item in manifest["workflows"]
            if item["modelType"] == "FluxFillPipeline" and item["mode"] == "inpaint"
        )
        pair = (workflow["modelType"], workflow["mode"])
        contract = self.contract_by_pair[pair]
        specification = self.spec_by_pair[pair]
        graph = json.loads((GRAPH_ROOT / workflow["graphPath"]).read_text(encoding="utf-8"))

        loader_tamper = deepcopy(graph)
        loader = next(node for node in loader_tamper["nodes"] if node["data"].get("studioRole") == "diffusersImagePipeline")
        loader["data"]["params"]["model_id"]["value"] = {
            "source": "hub",
            "value": "attacker/repository",
        }
        with self.assertRaisesRegex(TaskTemplateContractError, "loader identity"):
            validate_task_template_graph(loader_tamper, workflow, contract, specification)

        output_tamper = deepcopy(graph)
        output = next(node for node in output_tamper["nodes"] if node["data"].get("studioRole") == "preview")
        output["data"]["action"] = "DataViewer"
        with self.assertRaisesRegex(TaskTemplateContractError, "node identity|output identity"):
            validate_task_template_graph(output_tamper, workflow, contract, specification)

        media_tamper = deepcopy(graph)
        media_tamper["nodes"] = [
            node for node in media_tamper["nodes"] if node["data"].get("studioRole") != "loadMask"
        ]
        with self.assertRaisesRegex(TaskTemplateContractError, "node identity|required media"):
            validate_task_template_graph(media_tamper, workflow, contract, specification)


if __name__ == "__main__":
    unittest.main()
