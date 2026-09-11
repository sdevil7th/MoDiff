import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modiff.studio_execution_specs import ANIMA_DIFFUSERS_FILES, ANIMA_REPO, studio_capability_definitions


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "anima-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AnimaArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_public_snapshot_and_selected_component_closure_are_sealed(self):
        repository = self.review["repository"]
        self.assertEqual(repository["repository"], ANIMA_REPO)
        self.assertEqual(repository["revision"], "073c3a9db359c31ad0e8aa268d15775473c2176c")
        self.assertFalse(repository["private"])
        self.assertFalse(repository["gated"])
        self.assertEqual(repository["selectedFiles"], ANIMA_DIFFUSERS_FILES)
        self.assertEqual(repository["selectedFileCount"], len(ANIMA_DIFFUSERS_FILES))
        self.assertEqual(repository["fullRepositoryBytes"] - repository["selectedBytes"], 1757)
        self.assertEqual(
            catalog_repository_pin(ANIMA_REPO, model_type="AnimaModularPipeline"),
            {
                "repo": ANIMA_REPO,
                "revision": repository["revision"],
                "license": "circlestone-labs-non-commercial-license-v1.0",
                "kind": "base",
                "modelType": "AnimaModularPipeline",
            },
        )

    def test_weight_inventory_and_modular_index_are_exact(self):
        weights = self.review["weightFiles"]
        self.assertEqual(weights, sorted(weights, key=lambda item: item["path"]))
        self.assertEqual(len(weights), self.review["weightFileCount"])
        self.assertEqual(sum(item["byteSize"] for item in weights), self.review["weightBytes"])
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in weights))
        self.assertEqual(self.review["weightBytes"], 5_628_156_702)

        index = self.review["modularIndex"]
        self.assertEqual(index["pipelineClass"], "AnimaModularPipeline")
        self.assertEqual(index["blocksClass"], "AnimaAutoBlocks")
        self.assertEqual(
            index["components"],
            ["scheduler", "t5_tokenizer", "text_conditioner", "text_encoder", "tokenizer", "transformer", "vae"],
        )
        self.assertRegex(index["sha256"], SHA256)
        self.assertTrue(index["allRemoteComponentsSelfReferenceRepository"])
        self.assertTrue(index["componentRevisionsAreNull"])

    def test_official_workflows_map_to_the_reviewed_runtime_actions(self):
        contract = reviewed_modular_workflow_contract("AnimaModularPipeline")
        by_id = {workflow["id"]: workflow for workflow in contract["workflows"]}
        for review in self.review["reviewedWorkflows"]:
            with self.subTest(workflow=review["workflowId"]):
                workflow = by_id[review["workflowId"]]
                self.assertEqual(workflow["requiredInputs"], review["requiredInputs"])
                self.assertEqual(workflow["taskId"], review["taskId"])
                adapter = reviewed_whole_workflow_graph_adapter(
                    "AnimaModularPipeline", review["workflowId"]
                )
                self.assertIsNotNone(adapter)
                self.assertEqual(adapter["actionSequence"], review["runtimeActionSequence"])
                self.assertEqual(adapter["upstreamBlockSequence"], review["upstreamBlockSequence"])

    def test_license_and_publication_gates_are_not_overstated(self):
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["id"], "CircleStone Labs Non-Commercial License v1.0")
        self.assertIsNone(license_review["modelCardLicenseTag"])
        self.assertRegex(license_review["licenseFileSha256"], SHA256)
        self.assertTrue(license_review["revisionBoundAcknowledgementRequired"])
        self.assertEqual(license_review["modelAndDerivativeUseScope"], "noncommercial_only")
        self.assertTrue(license_review["commercialOutputUseAllowedSubjectToRestrictions"])

        capability = studio_capability_definitions()["AnimaModularPipeline"]
        self.assertEqual(capability["qualificationStatus"], "graph-qualified-execution-pending")
        self.assertFalse(capability["autoEligible"])
        self.assertFalse(capability["galleryEligible"])
        self.assertFalse(capability["liveProof"])
        admission = self.review["admission"]
        self.assertFalse(admission["liveQualified"])
        self.assertIn("generated_output_review", admission["unresolvedGates"])


if __name__ == "__main__":
    unittest.main()
