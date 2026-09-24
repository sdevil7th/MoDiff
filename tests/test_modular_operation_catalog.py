"""Canonical stage bindings must retain the existing Modular execution boundary."""

import unittest
from unittest.mock import patch

from modules import MODULE_MAP


class ModularOperationCatalogTests(unittest.TestCase):
    def test_specialized_stages_have_generic_identities_and_exact_state_contracts(self):
        from modules.ModularDiffusers.operation_contracts import get_modular_task_operation_contracts

        with patch("socket.socket.connect", side_effect=AssertionError("Network during discovery")):
            contracts = get_modular_task_operation_contracts(MODULE_MAP)
        stages = {
            c["operationId"]: c
            for c in contracts
            if c["pipelineClass"] == "AnimaModularPipeline" and c["task"] == "text_to_image"
        }
        self.assertEqual(stages["diffusion.encode_prompt"]["nodeKey"], "modules.ModularDiffusers.WorkflowTextEncode")
        denoise = stages["diffusion.denoise"]
        self.assertEqual(denoise["binding"]["values"]["pipeline_class"], "AnimaModularPipeline")
        self.assertEqual(denoise["binding"]["values"]["workflow_id"], "text2image")
        self.assertEqual(denoise["binding"]["values"]["block_path"], "denoise")
        state = next(p for p in denoise["ports"] if p["name"] == "state_in")
        self.assertEqual(state["semantics"]["kind"], "state")
        self.assertEqual(state["semantics"]["state"], "text_encoder")
        self.assertEqual(state["semantics"]["scope"], "AnimaModularPipeline:text2image")
        self.assertEqual(state["semantics"]["owner"], "same_loader")

    def test_loaders_preserve_component_bundle_and_non_decomposed_models_are_not_invented(self):
        from modules.ModularDiffusers.operation_contracts import get_modular_task_operation_contracts

        contracts = get_modular_task_operation_contracts(MODULE_MAP)
        loader = next(
            c
            for c in contracts
            if c["pipelineClass"] == "QwenImageModularPipeline"
            and c["task"] == "text_to_image"
            and c["operationId"] == "diffusion.load_models"
        )
        self.assertEqual(loader["nodeKey"], "modules.ModularDiffusers.ModelsLoader")
        self.assertEqual(loader["binding"]["values"]["model_type"], "QwenImageModularPipeline")
        bundle = next(p for p in loader["ports"] if p["name"] == "pipeline_components")
        self.assertEqual(bundle["roles"], ["component"])
        self.assertTrue(bundle["semantics"]["members"])
        ernie = [c for c in contracts if c["pipelineClass"] == "ErnieImageModularPipeline"]
        self.assertEqual(
            [contract["operationId"] for contract in ernie],
            [
                "diffusion.load_models",
                "diffusion.rewrite_prompt",
                "diffusion.encode_prompt",
                "diffusion.denoise",
                "diffusion.decode_latents",
            ],
        )
        self.assertEqual(ernie[1]["nodeKey"], "modules.ModularDiffusers.WorkflowErniePromptEnhance")
        self.assertEqual(ernie[-1]["binding"]["values"]["block_path"], "decode")
        self.assertEqual(get_modular_task_operation_contracts({}), [])

    def test_loader_hides_absent_components_and_missing_actions_do_not_advertise_partial_routes(self):
        from copy import deepcopy
        from modules.ModularDiffusers.operation_contracts import get_modular_task_operation_contracts

        contracts = get_modular_task_operation_contracts(MODULE_MAP)
        qwen = next(
            c
            for c in contracts
            if c["pipelineClass"] == "QwenImageModularPipeline"
            and c["task"] == "text_to_image"
            and c["decomposition"] == "loader"
        )
        self.assertTrue(next(p for p in qwen["ports"] if p["name"] == "image_encoder")["hidden"])
        self.assertEqual(
            next(p for p in qwen["ports"] if p["name"] == "unet_out")["semantics"]["members"][0]["name"], "transformer"
        )
        missing = deepcopy(MODULE_MAP)
        missing["modules.ModularDiffusers"].pop("WorkflowImageDenoise")
        self.assertFalse(
            any(c["pipelineClass"] == "AnimaModularPipeline" for c in get_modular_task_operation_contracts(missing))
        )

    def test_numeric_latent_controls_do_not_claim_tensor_ownership(self):
        from modiff.operation_contracts import with_operation_semantics

        record = {
            "pipelineClass": "Pipeline",
            "ports": [
                {"types": ["int"], "semanticName": "num_latents", "roles": ["value"]},
                {"types": ["tensor"], "semanticName": "latents", "roles": ["value"]},
            ],
        }
        ports = with_operation_semantics(record)["ports"]
        self.assertEqual(ports[0]["semantics"]["kind"], "value")
        self.assertIsNone(ports[0]["semantics"]["scope"])
        self.assertEqual(ports[1]["semantics"]["kind"], "latents")
        self.assertEqual(ports[1]["semantics"]["owner"], "same_loader")

    def test_media_preparation_and_typed_reference_helpers_retain_separate_operations(self):
        from modules.ModularDiffusers.operation_contracts import get_modular_task_operation_contracts

        contracts = get_modular_task_operation_contracts(MODULE_MAP)
        media = {
            c["operationId"]: c
            for c in contracts
            if c["pipelineClass"] == "MiniMaxH3ModularPipeline" and c["task"] == "reference_to_video_with_audio"
        }
        self.assertIn("diffusion.prepare_media", media)
        self.assertTrue(
            any(
                c["pipelineClass"] == "Cosmos3OmniModularPipeline"
                and c["operationId"] == "diffusion.postprocess_media"
                for c in contracts
            )
        )
        helper = media["diffusion.assemble_references"]
        self.assertEqual(helper["nodeKey"], "modules.ModularDiffusers.WorkflowMiniMaxH3ReferenceAssembler")
        self.assertEqual(helper["decomposition"], "bundle")
        self.assertIsNone(helper["blockName"])
        self.assertEqual(next(p for p in helper["ports"] if p["name"] == "references")["semantics"]["kind"], "opaque")
