import unittest

from modiff.cosmos3_v2_route_contract import (
    COSMOS3_ARTIFACTS,
    COSMOS3_DISTILLED_IMAGE2VIDEO_DIFFUSERS_FILES,
    COSMOS3_DISTILLED_TEXT2IMAGE_DIFFUSERS_FILES,
    COSMOS3_GUARDRAIL,
    PINNED_DIFFUSERS_REVISION,
    cosmos3_remaining_route_research,
    cosmos3_structural_admission_tranches,
)
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter


class Cosmos3V2RouteContractTests(unittest.TestCase):
    def test_exact_twelve_route_backlog_is_partitioned_without_execution_claims(self):
        routes = cosmos3_remaining_route_research()
        self.assertEqual(len(routes), 12)
        self.assertEqual(
            cosmos3_structural_admission_tranches(),
            {
                "blocked_no_exact_artifact": (
                    ("Cosmos3DistilledModularPipeline", "text2video"),
                    ("Cosmos3DistilledModularPipeline", "video2video"),
                ),
                "blocked_typed_action_composition": (
                    ("Cosmos3OmniModularPipeline", "action_forward_dynamics"),
                    ("Cosmos3OmniModularPipeline", "action_inverse_dynamics"),
                    ("Cosmos3OmniModularPipeline", "action_policy"),
                ),
                "distilled_exact_artifact_structural_admission": (
                    ("Cosmos3DistilledModularPipeline", "image2video"),
                    ("Cosmos3DistilledModularPipeline", "text2image"),
                ),
                "nano_non_action_media": (
                    ("Cosmos3OmniModularPipeline", "image2video"),
                    ("Cosmos3OmniModularPipeline", "image2video_with_sound"),
                    ("Cosmos3OmniModularPipeline", "text2video_with_sound"),
                    ("Cosmos3OmniModularPipeline", "video2video"),
                    ("Cosmos3OmniModularPipeline", "video2video_with_sound"),
                ),
            },
        )
        for contract in routes.values():
            self.assertEqual(contract["claim"], "structural_research_only")
            self.assertIsNone(contract["sealedDefaults"]["quantization"])
            self.assertIs(contract["sealedDefaults"]["trustRemoteCode"], False)
            self.assertIn("runtime_auto_gallery_and_publication_authority", contract["blockers"])

    def test_every_route_reuses_the_exact_checked_in_official_workflow_adapter(self):
        for identity, contract in cosmos3_remaining_route_research().items():
            adapter = reviewed_whole_workflow_graph_adapter(*identity)
            self.assertIsNotNone(adapter)
            self.assertEqual(tuple(adapter["requiredInputs"]), contract["requiredInputs"])
            self.assertEqual(tuple(adapter["actionSequence"]), contract["actionSequence"])
            self.assertEqual(tuple(adapter["upstreamBlockSequence"]), contract["upstreamBlockSequence"])
            self.assertEqual(
                tuple(
                    (
                        contract["actionRoles"][index],
                        edge["producerOutput"],
                        contract["actionRoles"][index + 1],
                        edge["consumerInput"],
                    )
                    for index, edge in enumerate(adapter["stateEdges"])
                ),
                contract["stateEdges"],
            )

    def test_artifact_and_guardrail_identities_are_immutable_and_route_specific(self):
        self.assertEqual(PINNED_DIFFUSERS_REVISION, "2f7e0154a9db246e95c9ede43edba7db5b130805")
        self.assertEqual(
            (COSMOS3_GUARDRAIL["repository"], COSMOS3_GUARDRAIL["revision"]),
            ("nvidia/Cosmos-Guardrail1", "d6d4bfa899a71454a700907664f3e88f503950cf"),
        )
        self.assertEqual(
            {name: (artifact["repository"], artifact["revision"]) for name, artifact in COSMOS3_ARTIFACTS.items()},
            {
                "nano": ("nvidia/Cosmos3-Nano", "7a312c868bcce8e40b3eb40861300a9d0ba3fde1"),
                "distilled_text2image": (
                    "nvidia/Cosmos3-Super-Text2Image-4Step",
                    "aa0d5a57b7b045d68daa60fbacd84ec723c7cb7b",
                ),
                "distilled_image2video": (
                    "nvidia/Cosmos3-Super-Image2Video-4Step",
                    "cd55ce81bc5cea51a09c37cd7652144e7278f049",
                ),
            },
        )

    def test_distilled_fixed_schedule_is_sealed_and_unmatched_workflows_have_no_artifact(self):
        routes = cosmos3_remaining_route_research()
        for workflow_id in ("text2image", "image2video"):
            contract = routes[("Cosmos3DistilledModularPipeline", workflow_id)]
            self.assertEqual(contract["sealedDefaults"]["numInferenceSteps"], 4)
            self.assertEqual(contract["sealedDefaults"]["guidanceScale"], 1.0)
            self.assertIsNotNone(contract["artifact"])
            self.assertNotIn("distilled_pipeline_registry_registration", contract["blockers"])
            self.assertNotIn("distilled_route_specific_artifact_catalog_pins", contract["blockers"])
            self.assertNotIn("distilled_four_step_and_guidance_one_sealed_bindings", contract["blockers"])
            self.assertIn("distilled_tensor_parallel_resource_qualification", contract["blockers"])
        for workflow_id in ("text2video", "video2video"):
            contract = routes[("Cosmos3DistilledModularPipeline", workflow_id)]
            self.assertIsNone(contract["artifact"])
            self.assertEqual(contract["tranche"], "blocked_no_exact_artifact")

    def test_distilled_selective_manifests_are_exact_and_route_specific(self):
        self.assertEqual(len(COSMOS3_DISTILLED_TEXT2IMAGE_DIFFUSERS_FILES), 44)
        self.assertEqual(len(COSMOS3_DISTILLED_IMAGE2VIDEO_DIFFUSERS_FILES), 42)
        self.assertEqual(len(set(COSMOS3_DISTILLED_TEXT2IMAGE_DIFFUSERS_FILES)), 44)
        self.assertEqual(len(set(COSMOS3_DISTILLED_IMAGE2VIDEO_DIFFUSERS_FILES)), 42)
        self.assertEqual(
            set(COSMOS3_DISTILLED_TEXT2IMAGE_DIFFUSERS_FILES)
            - set(COSMOS3_DISTILLED_IMAGE2VIDEO_DIFFUSERS_FILES),
            {
                "sound_tokenizer/config.json",
                "sound_tokenizer/diffusion_pytorch_model.safetensors",
            },
        )
        for files in (
            COSMOS3_DISTILLED_TEXT2IMAGE_DIFFUSERS_FILES,
            COSMOS3_DISTILLED_IMAGE2VIDEO_DIFFUSERS_FILES,
        ):
            self.assertEqual(
                len([path for path in files if path.startswith("transformer/diffusion_pytorch_model-")]),
                27,
            )
            self.assertIn("modular_model_index.json", files)
            self.assertIn("scheduler/scheduler_config.json", files)
            self.assertIn("vae/diffusion_pytorch_model.safetensors", files)
        self.assertEqual(COSMOS3_ARTIFACTS["distilled_text2image"]["weightByteSize"], 131_391_926_304)
        self.assertEqual(COSMOS3_ARTIFACTS["distilled_image2video"]["weightByteSize"], 129_405_417_712)

    def test_sound_and_action_outputs_preserve_upstream_route_semantics(self):
        routes = cosmos3_remaining_route_research()
        for workflow_id in (
            "text2video_with_sound",
            "image2video_with_sound",
            "video2video_with_sound",
        ):
            contract = routes[("Cosmos3OmniModularPipeline", workflow_id)]
            self.assertEqual(contract["semanticOutputs"], ("videos", "sound", "sampling_rate"))
            self.assertIs(contract["sealedDefaults"]["enableSound"], True)
        self.assertEqual(
            routes[("Cosmos3OmniModularPipeline", "action_policy")]["semanticOutputs"],
            ("videos", "action"),
        )
        self.assertEqual(
            routes[("Cosmos3OmniModularPipeline", "action_forward_dynamics")]["semanticOutputs"],
            ("videos",),
        )
        self.assertEqual(
            routes[("Cosmos3OmniModularPipeline", "action_inverse_dynamics")]["semanticOutputs"],
            ("action",),
        )


if __name__ == "__main__":
    unittest.main()
