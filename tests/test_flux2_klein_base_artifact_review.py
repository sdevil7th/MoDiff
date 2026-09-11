import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import (
    FLUX2_KLEIN_BASE_DIFFUSERS_FILES,
    FLUX2_KLEIN_BASE_REPO,
    studio_capability_definitions,
)


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "flux2-klein-base-4b-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Flux2KleinBaseArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_public_apache_artifact_is_pinned_without_single_file_duplicate(self):
        review = self.review
        self.assertEqual(review["schemaVersion"], 1)
        self.assertEqual(review["repository"], FLUX2_KLEIN_BASE_REPO)
        self.assertFalse(review["hub"]["private"])
        self.assertFalse(review["hub"]["gated"])
        self.assertEqual(review["hub"]["license"], "apache-2.0")
        self.assertRegex(review["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(
            catalog_repository_pin(FLUX2_KLEIN_BASE_REPO, model_type="Flux2KleinBaseModularPipeline")["revision"],
            review["revision"],
        )
        self.assertIn("flux-2-klein-base-4b.safetensors", review["artifactInventory"]["excludedFiles"])

    def test_selected_component_closure_and_weight_digests_are_exact(self):
        inventory = self.review["artifactInventory"]
        self.assertEqual(inventory["selectedFiles"], FLUX2_KLEIN_BASE_DIFFUSERS_FILES)
        self.assertEqual(inventory["selectedFileCount"], len(FLUX2_KLEIN_BASE_DIFFUSERS_FILES))
        self.assertEqual(
            inventory["selectedWeightBytes"],
            sum(item["byteSize"] for item in inventory["weightFiles"]),
        )
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in inventory["weightFiles"]))
        self.assertTrue(
            {item["path"] for item in inventory["weightFiles"]}.issubset(FLUX2_KLEIN_BASE_DIFFUSERS_FILES)
        )
        self.assertFalse(any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in inventory["selectedFiles"]))

    def test_base_modular_contract_remains_distinct_and_unpublished(self):
        source = self.review["sourceReview"]
        self.assertEqual(source["standardPipelineClass"], "Flux2KleinPipeline")
        self.assertEqual(source["modularPipelineClass"], "Flux2KleinBaseModularPipeline")
        self.assertEqual(source["blocksClass"], "Flux2KleinBaseAutoBlocks")
        self.assertFalse(source["transformerGuidanceEmbeds"])
        self.assertEqual((source["defaultInferenceSteps"], source["defaultGuidanceScale"]), (50, 4.0))

        capability = studio_capability_definitions()["Flux2KleinBaseModularPipeline"]
        self.assertEqual(capability["defaultRepo"], FLUX2_KLEIN_BASE_REPO)
        self.assertEqual(capability["downloadFiles"], FLUX2_KLEIN_BASE_DIFFUSERS_FILES)
        self.assertEqual(capability["qualificationStatus"], "graph-qualified-execution-pending")
        self.assertFalse(capability["autoEligible"])
        self.assertFalse(capability["galleryEligible"])
        self.assertFalse(capability["liveProof"])

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "live_qualified_output_approval_pending")
        self.assertTrue(admission["liveQualified"])
        self.assertEqual(admission["unresolvedGates"], ["generated_output_review"])
        self.assertFalse(admission["qualificationEvidence"]["showcaseApproved"])


if __name__ == "__main__":
    unittest.main()
