from copy import deepcopy
import unittest
from unittest.mock import patch

import modules as module_registry

from modiff.auto_resource import AUTO_MODEL_REQUIREMENTS
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.server import STUDIO_MODEL_CAPABILITIES, WebServer
from modiff.studio_execution_specs import (
    STUDIO_EXECUTION_SPEC_DEFINITIONS,
    assert_studio_execution_graph,
    studio_execution_spec_for_pair,
    validate_studio_execution_specs,
)


def executable_graph_for_spec(spec):
    nodes = {}
    node_ids = {}
    for index, (role, node_key, _x, _y) in enumerate(spec["roles"]):
        module, action = node_key.rsplit(".", 1)
        node_id = f"node-{index}"
        node_ids[role] = node_id
        params = {
            key: {**deepcopy(value), "value": deepcopy(value.get("default"))}
            for key, value in module_registry.MODULE_MAP[module][action]["params"].items()
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
            ],
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
        }
        hints["autoResourcePlan"] = selected
        assert_studio_execution_graph(graph, hints)

        selected["executionProfileId"] = "flux-dev:direct"
        with self.assertRaisesRegex(RuntimeError, "Auto profile"):
            assert_studio_execution_graph(graph, hints)

        selected["executionProfileId"] = spec["executionProfileId"]
        graph["nodes"][hints["studioExecutionSpec"]["nodes"]["preview"]]["params"]["image"].pop(
            "sourceId"
        )
        with self.assertRaisesRegex(RuntimeError, "edge"):
            assert_studio_execution_graph(graph, hints)

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
