import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "cosmos3-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Cosmos3ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_mutable_descriptors_and_safety_keep_runtime_closed(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "contract_only")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])
        self.assertIn("immutable_component_descriptors", admission["unresolvedGates"])
        self.assertIn("cosmos_guardrail_qualification", admission["unresolvedGates"])

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

    def test_four_public_inventory_receipts_are_exact_and_uncataloged(self):
        expected = {
            "nano": (10, 34894818144, 36, 4096, 64),
            "super": (30, 132624780208, 64, 5120, 64),
            "superTextToImage4Step": (29, 131391926304, 64, 5120, 32),
            "superImageToVideo4Step": (28, 129405417712, 64, 5120, 32),
        }
        for repository in self.review["repositories"]:
            component_count = sum(
                component["fileCount"] for component in repository["weightComponents"].values()
            )
            component_bytes = sum(
                component["byteSize"] for component in repository["weightComponents"].values()
            )
            transformer = repository["transformer"]
            self.assertEqual(component_count, repository["weightFileCount"])
            self.assertEqual(component_bytes, repository["weightBytes"])
            self.assertEqual(
                (
                    repository["weightFileCount"],
                    repository["weightBytes"],
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
            self.assertIsNone(catalog_repository_pin(repository["repository"]))

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


if __name__ == "__main__":
    unittest.main()
