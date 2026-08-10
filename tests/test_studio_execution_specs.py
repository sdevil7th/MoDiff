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
                ("WanTI2VPipeline", "text_to_video"),
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
        self.assertEqual(
            [item[0] for item in specs[6]["roles"]],
            ["diffusersQuantization", "diffusersRecipe", "wanPipeline", "wanGenerate", "videoExport"],
        )
        self.assertIn(("wanGenerate", "scheduler_flow_shift", "shift"), specs[6]["bindings"])
        self.assertIn(("wanGenerate", "video_out", "videoExport", "video"), specs[6]["edges"])
        self.assertEqual(specs[0]["actions"], ())
        self.assertRegex(specs[0]["contentHash"], r"^studio-spec-v1-[0-9a-f]{8}$")
        self.assertEqual(specs, validate_studio_execution_specs(module_registry.MODULE_MAP))

        for spec in specs:
            definition = STUDIO_EXECUTION_SPEC_DEFINITIONS[spec["id"]]
            profile = DIFFUSERS_EXECUTION_PROFILES[spec["executionProfileId"]]
            capability = STUDIO_MODEL_CAPABILITIES[spec["modelType"]]
            requirements = AUTO_MODEL_REQUIREMENTS[spec["modelType"]]
            self.assertEqual(profile.default_repo, definition["profile"]["default_repo"])
            self.assertEqual(profile.pipeline_class, spec["pipelineClass"])
            self.assertEqual(capability, definition["capability"])
            self.assertEqual(requirements, definition["autoRequirements"])

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
