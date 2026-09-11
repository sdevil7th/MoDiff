import json
import re
import unittest
from pathlib import Path

from modiff.huggingface_cluster_admission import audit_reviewed_cluster_execution_candidates
from modiff.model_artifact_catalog import catalog_model, catalog_repository_pin
from modiff.modular_workflow_contracts import PINNED_MODULAR_REPOSITORY_VARIANTS
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modules.ModularDiffusers.loaders import ModelsLoader
from modules.ModularDiffusers.modular_utils import COSMOS3_NANO_PIPELINE_CONFIG


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "cosmos3-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Cosmos3ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_nine_exact_routes_are_graph_qualified_while_runtime_and_safety_stay_closed(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "partially_graph_qualified")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])
        self.assertNotIn("immutable_component_descriptors", admission["unresolvedGates"])
        self.assertNotIn("standard_index_class_resolution", admission["unresolvedGates"])
        self.assertIn(
            "cosmos_guardrail_hub_access_and_license_acknowledgement",
            admission["unresolvedGates"],
        )
        self.assertEqual(
            {
                (route["pipelineClass"], route["workflowId"], route["studioMode"])
                for route in admission["graphQualifiedRoutes"]
            },
            {
                ("Cosmos3OmniModularPipeline", "text2image", "text_to_image"),
                ("Cosmos3OmniModularPipeline", "text2video", "text_to_video"),
                ("Cosmos3OmniModularPipeline", "image2video", "image_to_video"),
                ("Cosmos3OmniModularPipeline", "video2video", "video_to_video"),
                (
                    "Cosmos3OmniModularPipeline",
                    "text2video_with_sound",
                    "text_to_video_with_audio",
                ),
                (
                    "Cosmos3OmniModularPipeline",
                    "image2video_with_sound",
                    "image_to_video_with_audio",
                ),
                (
                    "Cosmos3OmniModularPipeline",
                    "video2video_with_sound",
                    "video_to_video_with_audio",
                ),
                ("Cosmos3DistilledModularPipeline", "text2image", "text_to_image"),
                ("Cosmos3DistilledModularPipeline", "image2video", "image_to_video"),
            },
        )
        self.assertEqual(len(admission["graphQualifiedRoutes"]), 9)
        self.assertIn(
            "distilled_tensor_parallel_resource_and_output_qualification",
            admission["unresolvedGates"],
        )
        self.assertNotIn(
            "distilled_artifact_workflow_mapping_and_tensor_parallel_profile",
            admission["unresolvedGates"],
        )
        self.assertTrue(
            admission["componentRevisionResolution"]["nullSameRepositoryDescriptorsBoundToTopLevelRevision"]
        )
        self.assertTrue(
            admission["componentRevisionResolution"]["standardModelIndexIgnoredWhenModularIndexPresent"]
        )

        repositories = {item["role"]: item for item in self.review["repositories"]}
        for role in ("nano", "super"):
            self.assertEqual(repositories[role]["componentDescriptorRevisionState"], "all_null")
            self.assertFalse(repositories[role]["modelIndex"]["classExportedByPinnedDiffusers"])
        for role in ("superTextToImage4Step", "superImageToVideo4Step"):
            self.assertEqual(
                repositories[role]["componentDescriptorRevisionState"],
                "revision_key_absent",
            )
            self.assertFalse(repositories[role]["modelIndex"]["safetyCheckerConfigured"])

    def test_four_public_inventory_receipts_are_exact_and_structural_routes_are_cataloged(self):
        expected = {
            "nano": (10, 34894818144, 34986890561, 36, 4096, 64),
            "super": (30, 132624780208, 132710200448, 64, 5120, 64),
            "superTextToImage4Step": (29, 131391926304, 131423464104, 64, 5120, 32),
            "superImageToVideo4Step": (28, 129405417712, 129446040640, 64, 5120, 32),
        }
        for repository in self.review["repositories"]:
            component_count = sum(component["fileCount"] for component in repository["weightComponents"].values())
            component_bytes = sum(component["byteSize"] for component in repository["weightComponents"].values())
            transformer = repository["transformer"]
            self.assertEqual(component_count, repository["weightFileCount"])
            self.assertEqual(component_bytes, repository["weightBytes"])
            self.assertEqual(
                (
                    repository["weightFileCount"],
                    repository["weightBytes"],
                    repository["snapshotByteSize"],
                    transformer["hiddenLayers"],
                    transformer["hiddenSize"],
                    transformer["actionDimension"],
                ),
                expected[repository["role"]],
            )
            self.assertRegex(repository["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(repository["weightInventorySha256"], SHA256)
            self.assertRegex(transformer["configSha256"], SHA256)
            self.assertRegex(repository["modelIndex"]["sha256"], SHA256)
            self.assertRegex(repository["modularIndex"]["sha256"], SHA256)
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            catalog_pin = catalog_repository_pin(repository["repository"])
            if repository["role"] == "nano":
                self.assertEqual(catalog_pin["revision"], repository["revision"])
                catalog_entry = catalog_model("Cosmos3OmniModularPipeline")
                self.assertEqual(catalog_entry["snapshotByteSize"], repository["snapshotByteSize"])
                self.assertEqual(catalog_entry["weightByteSize"], repository["weightBytes"])
            elif repository["role"] == "superTextToImage4Step":
                self.assertEqual(catalog_pin["revision"], repository["revision"])
                catalog_entry = catalog_model("Cosmos3DistilledModularPipeline")
                self.assertEqual(catalog_entry["snapshotByteSize"], repository["snapshotByteSize"])
                self.assertEqual(catalog_entry["weightByteSize"], repository["weightBytes"])
            elif repository["role"] == "superImageToVideo4Step":
                self.assertEqual(catalog_pin["revision"], repository["revision"])
                self.assertEqual(catalog_pin["snapshotByteSize"], repository["snapshotByteSize"])
                self.assertEqual(catalog_pin["weightByteSize"], repository["weightBytes"])
            else:
                self.assertIsNone(catalog_pin)

    def test_existing_modular_workflows_match_the_sealed_contracts(self):
        for reviewed in self.review["modularContracts"]:
            contract = reviewed_modular_workflow_contract(reviewed["pipelineClass"])
            self.assertEqual(contract["blocksClass"], reviewed["blocksClass"])
            workflows = {workflow["id"]: workflow for workflow in contract["workflows"]}
            self.assertEqual(
                {
                    name: {
                        "taskId": workflow["taskId"],
                        "requiredInputs": workflow["requiredInputs"],
                    }
                    for name, workflow in workflows.items()
                },
                reviewed["workflows"],
            )
        self.assertEqual(len(self.review["modularContracts"][0]["workflows"]), 10)
        self.assertEqual(len(self.review["modularContracts"][1]["workflows"]), 4)

    def test_distilled_workflows_have_exact_sealed_top_level_graph_adapters(self):
        expected = {
            "text2image": ["text_encoder", "denoise", "decode"],
            "text2video": ["text_encoder", "denoise", "decode"],
            "image2video": ["text_encoder", "vae_encoder", "denoise", "decode"],
            "video2video": ["text_encoder", "vae_encoder", "denoise", "decode"],
        }
        for workflow_id, sequence in expected.items():
            adapter = reviewed_whole_workflow_graph_adapter(
                "Cosmos3DistilledModularPipeline",
                workflow_id,
            )
            self.assertEqual(adapter["adapterId"], "official_top_level_blocks")
            self.assertEqual(adapter["upstreamBlockSequence"], sequence)
            self.assertEqual(len(adapter["stateEdges"]), len(sequence) - 1)

        # Structural review does not create an artifact, runtime, or legal
        # admission claim; the artifact receipt above keeps those gates closed.
        self.assertFalse(self.review["admission"]["runtimeCatalogExposed"])
        self.assertFalse(self.review["admission"]["downloadCatalogExposed"])

    def test_omni_workflows_preserve_modality_branches_and_after_decode(self):
        conditioned = {
            "image2video",
            "video2video",
            "image2video_with_sound",
            "video2video_with_sound",
            "action_policy",
            "action_forward_dynamics",
            "action_inverse_dynamics",
        }
        for workflow_id in (
            "text2image",
            "text2video",
            "image2video",
            "video2video",
            "text2video_with_sound",
            "image2video_with_sound",
            "video2video_with_sound",
            "action_policy",
            "action_forward_dynamics",
            "action_inverse_dynamics",
        ):
            adapter = reviewed_whole_workflow_graph_adapter("Cosmos3OmniModularPipeline", workflow_id)
            expected = ["text_encoder"]
            if workflow_id in conditioned:
                expected.append("vae_encoder")
            expected.extend(["denoise", "decode", "after_decode"])
            self.assertEqual(adapter["upstreamBlockSequence"], expected)
            self.assertEqual(len(adapter["stateEdges"]), len(expected) - 1)

    def test_nano_admissions_bind_exact_guardrail_and_never_claim_execution(self):
        results = {
            result["definitionId"]: result
            for result in audit_reviewed_cluster_execution_candidates()
            if result["definitionId"].startswith("diffusers.modular:Cosmos3OmniModularPipeline:")
        }
        self.assertEqual(
            set(results),
            {
                "diffusers.modular:Cosmos3OmniModularPipeline:text2image",
                "diffusers.modular:Cosmos3OmniModularPipeline:text2video",
                "diffusers.modular:Cosmos3OmniModularPipeline:image2video",
                "diffusers.modular:Cosmos3OmniModularPipeline:video2video",
                "diffusers.modular:Cosmos3OmniModularPipeline:text2video_with_sound",
                "diffusers.modular:Cosmos3OmniModularPipeline:image2video_with_sound",
                "diffusers.modular:Cosmos3OmniModularPipeline:video2video_with_sound",
            },
        )
        for result in results.values():
            self.assertEqual(result["status"], "admitted")
            self.assertFalse(result["executable"])
            self.assertFalse(result["publication"]["executable"])
            self.assertFalse(result["publication"]["autoEligible"])
            self.assertEqual(
                result["modelDependencies"],
                [
                    {
                        "id": "cosmos3-mandatory-safety-guardrail",
                        "kind": "safety_checker",
                        "repo": "nvidia/Cosmos-Guardrail1",
                        "revision": "d6d4bfa899a71454a700907664f3e88f503950cf",
                    }
                ],
            )

        guardrail = self.review["guardrail"]
        self.assertEqual(guardrail["access"], "gated_auto")
        self.assertFalse(guardrail["acknowledgedByUser"])
        self.assertEqual(guardrail["exactPackageVersion"], "0.3.1")
        self.assertEqual(guardrail["snapshotByteSize"], 7171449905)
        self.assertEqual(guardrail["weightByteSize"], 3627263067)
        self.assertEqual(
            catalog_repository_pin("nvidia/Cosmos-Guardrail1")["revision"],
            guardrail["revision"],
        )

    def test_models_loader_closes_cosmos_to_the_single_reviewed_nano_artifact(self):
        model_type = "Cosmos3OmniModularPipeline"
        repository = "nvidia/Cosmos3-Nano"
        revision = "7a312c868bcce8e40b3eb40861300a9d0ba3fde1"
        self.assertEqual(COSMOS3_NANO_PIPELINE_CONFIG.default_repo, repository)
        self.assertNotIn(model_type, PINNED_MODULAR_REPOSITORY_VARIANTS)
        self.assertEqual(
            ModelsLoader._reviewed_builtin_selection(
                model_type=model_type,
                repo_id={"source": "hub", "value": repository},
                revision=None,
            ),
            ("hub", repository, revision),
        )
        with self.assertRaisesRegex(ValueError, "requires reviewed repository selection"):
            ModelsLoader._reviewed_builtin_selection(
                model_type=model_type,
                repo_id={"source": "hub", "value": "nvidia/Cosmos3-Super"},
                revision=None,
            )
        with self.assertRaisesRegex(ValueError, "reviewed immutable Hub artifact"):
            ModelsLoader._reviewed_builtin_selection(
                model_type=model_type,
                repo_id={"source": "local", "value": "/tmp/cosmos"},
                revision=None,
            )

    def test_recipes_preserve_distilled_and_safety_invariants_without_live_claims(self):
        recipes = self.review["reviewedRecipes"]
        omni = recipes["omni"]
        self.assertEqual(
            (
                omni["width"],
                omni["height"],
                omni["numFrames"],
                omni["numInferenceSteps"],
                omni["guidanceScale"],
                omni["fps"],
                omni["flowShift"],
            ),
            (1280, 720, 189, 35, 6.0, 24.0, 10.0),
        )
        distilled = recipes["distilled"]
        self.assertEqual(distilled["numInferenceSteps"], 4)
        self.assertEqual(distilled["guidanceScale"], 1.0)
        self.assertTrue(distilled["negativePromptIgnored"])
        self.assertEqual(
            distilled["textToImage"]["publicOutputProjection"],
            {
                "catalogOutputName": "videos",
                "executionField": "image",
                "executionRole": "decode",
                "publicMediaType": "image",
                "publicPortId": "images",
                "reason": (
                    "The pinned Cosmos3Distilled text2image catalog keeps the shared semantic output "
                    "name videos, while its exact decoder publishes one PIL image on decode.image; the "
                    "registered Block aliases only that reviewed terminal to an image socket."
                ),
            },
        )
        safety = recipes["safety"]
        self.assertTrue(safety["taskPipelineEnableSafetyCheckDefault"])
        self.assertTrue(safety["modularPipelineRequiresExplicitEnableSafetyChecker"])
        self.assertTrue(safety["cosmosGuardrailDependencyRequired"])
        self.assertFalse(recipes["promptUpsampling"]["allowedDuringDiscovery"])
        self.assertTrue(
            all(
                envelope["status"] == "estimate_only_qualification_pending"
                for envelope in self.review["remoteResourceEnvelopes"].values()
            )
        )

    def test_linked_license_is_recorded_but_not_misrepresented_as_embedded(self):
        license_contract = self.review["license"]
        self.assertFalse(license_contract["embeddedLicenseFilePresent"])
        self.assertTrue(license_contract["modelCardsLinkMutableLicenseUrl"])
        linked = license_contract["linkedLicenseSnapshot"]
        self.assertTrue(linked["copyAndOriginNoticesRequiredOnDistribution"])
        self.assertTrue(linked["patentOrCopyrightLitigationTermination"])
        self.assertFalse(linked["modelOutputUseRestricted"])
        self.assertRegex(linked["sha256"], SHA256)

    def test_current_diffusers_and_model_card_sources_are_hash_bound(self):
        source = self.review["sourceContract"]
        self.assertEqual(source["diffusersRevision"], "2f7e0154a9db246e95c9ede43edba7db5b130805")
        self.assertRegex(source["documentation"]["sha256"], SHA256)
        self.assertIn(source["diffusersRevision"], source["documentation"]["url"])
        self.assertEqual(source["modelCard"]["revision"], "7a312c868bcce8e40b3eb40861300a9d0ba3fde1")
        self.assertRegex(source["modelCard"]["readmeSha256"], SHA256)
        self.assertRegex(source["modelCard"]["safetySha256"], SHA256)
        for receipt in source["sources"].values():
            self.assertGreater(receipt["byteSize"], 0)
            self.assertRegex(receipt["sha256"], SHA256)


if __name__ == "__main__":
    unittest.main()
