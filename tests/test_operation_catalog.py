import unittest
from unittest.mock import patch

from modules import MODULE_MAP


class OperationCatalogTests(unittest.TestCase):
    def test_shared_standard_pipeline_starters_retain_the_selected_public_profile(self):
        from modiff.diffusers_profiles import public_execution_profiles, resolve_execution_profiles_for_loader
        from modiff.operation_catalog import build_operation_catalog
        from modiff.operation_starters import resolve_operation_starter

        contracts, _ = build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})
        for pipeline, expected in (
            ("WanImageToVideoPipeline", "wan-22-image-to-video:direct"),
            ("Wan22Image2VideoModularPipeline", "wan22-i2v:equivalent-standard"),
        ):
            with self.subTest(pipeline=pipeline):
                starter = resolve_operation_starter(
                    MODULE_MAP, contracts, {"pipelineClass": pipeline, "task": "image_to_video"}
                )
                loader = next(node for node in starter["nodes"] if node["action"] == "LoadPipeline")
                self.assertEqual(loader["values"].get("execution_profile_id"), expected)
                profiles, reason = resolve_execution_profiles_for_loader(
                    loader["module"], loader["action"], loader["values"]
                )
                self.assertIsNone(reason)
                self.assertEqual([profile.id for profile in profiles], [expected])

    def test_shared_pipeline_binding_does_not_guess_between_profiles_for_one_identity(self):
        from dataclasses import replace
        from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, public_execution_profiles
        from modiff.operation_catalog import build_operation_catalog, resolve_operation

        duplicate = replace(DIFFUSERS_EXECUTION_PROFILES["wan-22-image-to-video:direct"], id="another-reviewed-route")
        with patch.dict(DIFFUSERS_EXECUTION_PROFILES, {duplicate.id: duplicate}):
            contracts, _ = build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})
            loader = resolve_operation(MODULE_MAP, contracts, {
                "pipelineClass": "WanImageToVideoPipeline", "task": "image_to_video",
                "operationId": "diffusion.load_models",
            })
        self.assertNotIn("execution_profile_id", loader["values"])

    def test_support_separates_adapters_dependencies_and_editable_stages(self):
        from modiff.operation_catalog import build_operation_catalog
        from modiff.diffusers_profiles import public_execution_profiles

        profiles = public_execution_profiles()
        for profile in profiles:
            requirement = profile["optionalRuntimeRequirement"]
            if requirement["requiredNow"]:
                requirement.update(state="missing", reason="optional_runtime_missing")
        contracts, support = build_operation_catalog(MODULE_MAP, profiles, catalog_resolver=lambda: {})
        by_class = {p["pipelineClass"]: p for p in support}
        self.assertGreaterEqual(len(by_class), 330)
        qwen = next(t for t in by_class["QwenImageModularPipeline"]["tasks"] if t["task"] == "text_to_image")
        self.assertEqual(qwen["execution"], "adapter")
        self.assertEqual(qwen["decomposition"], "stages")
        self.assertEqual(qwen["dependencies"], "blocked")
        krea = next(t for t in by_class["Krea2ModularPipeline"]["tasks"] if t["task"] == "text_to_image")
        self.assertEqual(krea["execution"], "declared")
        self.assertEqual(krea["dependencies"], "unknown")
        self.assertTrue(all("template" not in str(c["binding"]).lower() for c in contracts))
        self.assertFalse(
            any(c["pipelineClass"] == "LTXModularPipeline" and c["task"] == "video_to_video" for c in contracts)
        )
        for call in (c for c in contracts if c["decomposition"] == "pipeline"):
            self.assertTrue(
                any(
                    c["decomposition"] == "loader"
                    and c["pipelineClass"] == call["pipelineClass"]
                    and c["task"] == call["task"]
                    and c["binding"]["pipelineClass"] == call["binding"]["pipelineClass"]
                    and c["nodeKey"].rsplit(".", 1)[0] == call["nodeKey"].rsplit(".", 1)[0]
                    for c in contracts
                )
            )
        fallback = next(
            c
            for c in contracts
            if c["pipelineClass"] == "LTXModularPipeline"
            and c["task"] == "text_to_video"
            and c["decomposition"] == "pipeline"
        )
        self.assertEqual(fallback["binding"]["pipelineClass"], "LTXConditionPipeline")
        self.assertTrue(fallback["nodeKey"].startswith("modules.DiffusersVideo."))

    def test_semantic_connections_do_not_equate_different_state_stages_or_model_domains(self):
        from modiff.operation_catalog import operation_port_compatibility
        from modiff.operation_catalog import build_operation_catalog

        contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})

        def stage(pipeline, operation):
            return next(
                c
                for c in contracts
                if c["pipelineClass"] == pipeline and c["task"] == "text_to_image" and c["operationId"] == operation
            )

        def port(contract, name):
            return next(p for p in contract["ports"] if p["name"] == name)

        prompt = stage("AnimaModularPipeline", "diffusion.encode_prompt")
        denoise = stage("AnimaModularPipeline", "diffusion.denoise")
        decode = stage("AnimaModularPipeline", "diffusion.decode_latents")
        self.assertEqual(
            operation_port_compatibility(port(prompt, "state_out"), port(denoise, "state_in")), "runtime_validation"
        )
        self.assertEqual(
            operation_port_compatibility(port(prompt, "state_out"), port(decode, "state_in")), "incompatible"
        )
        other = stage("Krea2ModularPipeline", "diffusion.denoise")
        self.assertEqual(
            operation_port_compatibility(port(prompt, "state_out"), port(other, "state_in")), "incompatible"
        )

    def test_every_binding_resolves_without_models_or_mutating_registry(self):
        from copy import deepcopy
        from modiff.operation_catalog import build_operation_catalog, resolve_operation
        from modiff.diffusers_profiles import public_execution_profiles

        contracts, _ = build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})
        original = deepcopy(MODULE_MAP)
        with (
            patch("socket.socket.connect", side_effect=AssertionError("Network during resolution")),
            patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed node")),
        ):
            for contract in contracts:
                with self.subTest(
                    pipeline=contract["pipelineClass"], task=contract["task"], op=contract["operationId"]
                ):
                    resolved = resolve_operation(
                        MODULE_MAP, contracts, {k: contract[k] for k in ("pipelineClass", "task", "operationId")}
                    )
                    self.assertEqual(resolved["module"] + "." + resolved["action"], contract["nodeKey"])
                    for key, value in resolved["values"].items():
                        self.assertEqual(resolved["params"][key]["value"], value)
                    if resolved["action"] == "ModelsLoader":
                        from modules.ModularDiffusers.modular_utils import get_model_type_metadata
                        from modules.ModularDiffusers.loaders import MODELS_LOADER_IDENTITY_OUTPUTS

                        metadata = get_model_type_metadata(contract["pipelineClass"])
                        expected = (
                            "" if metadata.get("execution_status") == "contract_only" else contract["pipelineClass"]
                        )
                        for output in MODELS_LOADER_IDENTITY_OUTPUTS:
                            self.assertEqual(resolved["params"][output]["signal"]["value"], expected)
                    if contract["decomposition"] == "loader" and resolved["action"] == "LoadPipeline":
                        self.assertEqual(
                            resolved["params"]["pipeline"]["signal"]["value"]["pipelineClass"],
                            contract["binding"]["pipelineClass"],
                        )
        self.assertEqual(MODULE_MAP, original)

    def test_binding_resolution_never_constructs_a_node_and_returns_existing_fields(self):
        from modiff.operation_catalog import build_operation_catalog, resolve_operation

        contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
        with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed node")):
            result = resolve_operation(
                MODULE_MAP,
                contracts,
                {
                    "pipelineClass": "AnimaModularPipeline",
                    "task": "text_to_image",
                    "operationId": "diffusion.denoise",
                },
            )
        self.assertEqual(result["action"], "WorkflowImageDenoise")
        self.assertEqual(result["values"]["pipeline_class"], "AnimaModularPipeline")
        self.assertIn("state_in", result["params"])
        with self.assertRaises(ValueError):
            resolve_operation(
                MODULE_MAP,
                contracts,
                {"pipelineClass": "constructor", "task": None, "operationId": "diffusion.denoise"},
            )
