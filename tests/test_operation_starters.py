"""Ordinary stage drafts must use the reviewed runtime's actual wires."""

import unittest
from unittest.mock import patch

from modules import MODULE_MAP
from modiff.operation_catalog import build_operation_catalog


class OperationStarterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})

    def resolve(self, pipeline, task):
        from modiff.operation_starters import resolve_operation_starter

        return resolve_operation_starter(MODULE_MAP, self.contracts, {"pipelineClass": pipeline, "task": task})

    def test_four_stages_are_an_ordinary_draft_with_exact_state_and_component_wires(self):
        with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed node")):
            result = self.resolve("AnimaModularPipeline", "text_to_image")
        self.assertEqual(len(result["nodes"]), 4)
        self.assertEqual(len(result["edges"]), 5)
        self.assertNotIn("receipt", result)
        self.assertTrue(all(n["operation"]["task"] == "text_to_image" for n in result["nodes"]))
        self.assertTrue(
            any(e["sourceHandle"] == "state_out" and e["targetHandle"] == "state_in" for e in result["edges"])
        )

    def test_task_switch_uses_upstream_rules_and_exposes_required_image(self):
        text = self.resolve("FluxModularPipeline", "text_to_image")
        image = self.resolve("FluxModularPipeline", "image_to_image")
        edit = self.resolve("FluxKontextModularPipeline", "edit_image")
        self.assertEqual(len(text["nodes"]), 4)
        self.assertEqual(len(image["nodes"]), 5)
        self.assertEqual(image["workflowId"], "image2image")
        self.assertEqual(edit["workflowId"], "image_conditioned")
        self.assertTrue(any(i["field"] == "image" for i in image["requiredInputs"]))
        self.assertTrue(any(e["sourceHandle"] == "out_width" and e["targetHandle"] == "width" for e in image["edges"]))
        with self.assertRaises(ValueError):
            self.resolve("FluxModularPipeline", "edit_image")

    def test_required_component_contract_completes_denoise_vae_wiring(self):
        # The SDXL runtime requires a managed VAE even for text-to-image.
        # Admission's minimal component edges alone do not describe this input.
        result = self.resolve("StableDiffusionXLModularPipeline", "text_to_image")
        edge = {
            "source": "diffusion.load_models",
            "sourceHandle": "vae_out",
            "target": "diffusion.denoise",
            "targetHandle": "vae",
        }
        self.assertIn(edge, result["edges"])
        image = self.resolve("StableDiffusionXLModularPipeline", "image_to_image")
        self.assertEqual(
            image["sharedInputs"],
            [
                {
                    "name": "seed",
                    "members": [
                        {"operationId": "diffusion.encode_image", "field": "seed"},
                        {"operationId": "diffusion.denoise", "field": "seed"},
                    ],
                }
            ],
        )
        self.assertEqual(result["sharedInputs"], [])
        # These adapters do not declare that denoiser component dependency.
        for pipeline in ("FluxModularPipeline", "QwenImageModularPipeline"):
            self.assertNotIn(edge, self.resolve(pipeline, "text_to_image")["edges"])

    def test_standard_fallback_is_a_load_and_call_and_keeps_audio_contract(self):
        result = self.resolve("StableAudioPipeline", "text_to_audio")
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["edges"][0]["targetHandle"], "pipeline")

    def test_all_task_drafts_have_existing_visible_endpoints_and_one_writer(self):
        with (
            patch("socket.socket.connect", side_effect=AssertionError("Network during authoring")),
            patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed node")),
        ):
            for pipeline, task in sorted({(c["pipelineClass"], c["task"]) for c in self.contracts if c["task"]}):
                with self.subTest(pipeline=pipeline, task=task):
                    result = self.resolve(pipeline, task)
                    nodes = {n["operation"]["operationId"]: n for n in result["nodes"]}
                    targets = set()
                    for edge in result["edges"]:
                        self.assertIn(edge["sourceHandle"], nodes[edge["source"]]["params"])
                        self.assertIn(edge["targetHandle"], nodes[edge["target"]]["params"])
                        target = edge["target"], edge["targetHandle"]
                        self.assertNotIn(target, targets)
                        targets.add(target)

    def test_invalid_selection_is_rejected_without_mutation(self):
        from modiff.operation_starters import resolve_operation_starter

        for selection in (
            None,
            {},
            {"pipelineClass": "__proto__", "task": "text_to_image"},
            {"pipelineClass": "FluxModularPipeline", "task": None},
            {"pipelineClass": "FluxModularPipeline", "task": "text_to_image", "code": "x"},
        ):
            with self.assertRaises(ValueError):
                resolve_operation_starter(MODULE_MAP, self.contracts, selection)
