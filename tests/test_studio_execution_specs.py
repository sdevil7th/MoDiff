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
    def test_model_dependencies_are_pair_specific_immutable_artifact_receipts(self):
        qwen = studio_model_dependencies_for_pair(
            "QwenImageModularPipeline",
            "control_image",
        )
        redux = studio_model_dependencies_for_pair("FluxReduxPipeline", "edit_image")

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
        self.assertEqual(
            studio_model_requirements_for_pair("FluxReduxPipeline", "edit_image")[0]["requiredForModes"],
            ["edit_image"],
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
                ("QwenImageEditPlusModularPipeline", "edit_image"),
                ("QwenImageEditPlusModularPipeline", "multi_image_reference_edit"),
                ("QwenImageLayeredModularPipeline", "layer_decomposition"),
                ("QwenImageModularPipeline", "control_image"),
                ("StableAudioPipeline", "text_to_audio"),
                ("FluxDevPipeline", "edit_image"),
                ("FluxDevPipeline", "inpaint"),
                ("StableDiffusionXLPipeline", "text_to_image"),
                ("StableDiffusionXLPipeline", "edit_image"),
                ("StableDiffusionXLPipeline", "inpaint"),
                ("Wan22Pipeline", "text_to_video"),
                ("WanAnimatePipeline", "character_animate"),
                ("WanAnimatePipeline", "character_replace"),
                ("LTXI2VLongMultiPromptPipeline", "image_to_video"),
                ("LTX2ConditionPipeline", "text_to_video"),
                ("LTX2ConditionPipeline", "image_to_video"),
                ("LTX2ConditionPipeline", "reference_to_video"),
                ("LTX2ConditionPipeline", "video_to_video"),
                ("HunyuanVideoFramepackPipeline", "image_to_video"),
                ("WanImage2VideoModularPipeline", "image_to_video"),
            ],
        )
        by_id = {item["id"]: item for item in specs}
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
        self.assertEqual(specs[15]["bindings"], specs[14]["bindings"])
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
        graph["nodes"][hints["studioExecutionSpec"]["nodes"]["preview"]]["params"]["image"].pop(
            "sourceId"
        )
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
        self.assertEqual(spec["roles"], validate_studio_execution_specs(module_registry.MODULE_MAP)[0]["roles"])
        self.assertEqual(spec["edges"], validate_studio_execution_specs(module_registry.MODULE_MAP)[0]["edges"])
        self.assertEqual(spec["bindings"], validate_studio_execution_specs(module_registry.MODULE_MAP)[0]["bindings"])
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
        self.assertEqual(spec["bindings"], z_image["bindings"])
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
        self.assertIn(("audioLoudnessMatch", "reference_window_seconds", "referenceWindow15"), continuation["bindings"])
        self.assertIn(("audioJoin", "boundary_fade_seconds", "boundaryFade001"), continuation["bindings"])
        self.assertIn(("loadAudio", "audio", "audioJoin", "source"), continuation["edges"])
        self.assertIn(("audioGenerate", "task_type", "repaint"), repaint["bindings"])
        self.assertIn(("loadAudio", "file", "sourceAudio"), repaint["bindings"])
        self.assertEqual(repaint["roles"], variation["roles"])
        self.assertEqual(repaint["edges"], variation["edges"])
        for spec in (text, variation, continuation, repaint):
            self.assertIn(("audioGenerate", "sample_rate", "sampleRate48000"), spec["bindings"])
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

        graph["nodes"][hints["studioExecutionSpec"]["nodes"]["denoise"]]["params"]["route_state_in"].pop(
            "sourceId"
        )
        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

    def test_qwen_image_edit_plus_modes_seal_the_exact_dynamic_modular_route(self):
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
        self.assertEqual(len(spec["bindings"]), 17)
        self.assertNotIn("route_state_out", [item[1] for item in spec["edges"]])
        self.assertIn(("loadImage", "alpha_channel", "addAlpha"), spec["bindings"])
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
        self.assertEqual(len(spec["edges"]), 13)
        self.assertEqual(len(spec["bindings"]), 32)
        self.assertIn(("controlnetModel", "model_id", "repo"), spec["bindings"])
        self.assertIn(("controlnetModel", "revision", "revision"), spec["bindings"])
        self.assertIn(("controlnet", "route_state_out", "denoise", "route_state_in"), spec["edges"])
        self.assertIn(("denoise", "route_state_out", "decode", "route_state_in"), spec["edges"])
        graph, hints = executable_graph_for_spec(spec)
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
