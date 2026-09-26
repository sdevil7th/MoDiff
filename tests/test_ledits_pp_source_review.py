import json
from pathlib import Path
import re
import unittest

from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "ledits-pp-source-review.json"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "model-artifact-catalog.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LEditsPPSourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        self.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_only_existing_exact_base_snapshots_are_reused(self):
        managed_pairs = {
            (model["baseRepo"], model["baseRevision"], model["baseLicense"])
            for model in self.catalog["models"]
            if "baseRepo" in model
        }
        managed_pairs.update(
            (pin["repo"], pin["revision"], pin["license"])
            for pin in self.catalog["repositoryPins"]
        )
        for base in self.review["baseRepositories"]:
            self.assertRegex(base["revision"], SHA1)
            self.assertTrue(base["alreadyManagedByMoDiff"])
            self.assertFalse(base["newModelDownloadRequired"])
            self.assertIn(
                (base["repository"], base["revision"], base["license"]),
                managed_pairs,
            )

    def test_pinned_package_sources_and_two_phase_contract_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertRegex(pinned["revision"], SHA1)
        self.assertTrue(
            all(
                SHA256.fullmatch(value)
                for key, value in pinned.items()
                if key.endswith("Sha256")
            )
        )
        contract = self.review["pipelineContract"]
        self.assertEqual(contract["mode"], "stateful_two_phase_image_edit")
        self.assertEqual(contract["phases"], ["invert_input_image", "apply_one_or_more_text_edits"])
        self.assertEqual(contract["dimensionMultiple"], 32)
        self.assertEqual(contract["stableDiffusionDefaults"]["numInversionSteps"], 30)
        self.assertEqual(contract["stableDiffusionXLDefaults"]["numInversionSteps"], 50)
        self.assertTrue(all(value is None for value in contract["packageUpperBounds"].values()))

    def test_inversion_prevents_full_job_cancellation(self):
        cancellation = self.review["cancellationReview"]
        self.assertTrue(cancellation["editPhaseCallbackOnStepEndAvailable"])
        self.assertFalse(cancellation["inversionPhaseCallbackAvailable"])
        self.assertFalse(cancellation["interruptFlagAvailable"])
        self.assertFalse(cancellation["fullJobCooperativelyCancellable"])

    def test_state_and_safety_gaps_remain_explicit(self):
        state = self.review["stateAndSafetyReview"]
        self.assertTrue(state["inversionStateStoredOnPipelineInstance"])
        self.assertEqual(set(state["storedFields"]), {"inversion_steps", "init_latents", "zs"})
        self.assertTrue(state["requestIsolationContractRequired"])
        self.assertTrue(state["stableDiffusionSafetyCheckerAvailable"])
        self.assertFalse(state["stableDiffusionXLSafetyCheckerAvailable"])
        self.assertFalse(state["perfectInversionCurrentlyGuaranteedByPackage"])

    def test_review_adds_no_download_or_user_facing_surface(self):
        download = self.review["downloadReview"]
        self.assertFalse(download["newRepositoryRequired"])
        self.assertEqual(download["newWeightBytesRequired"], 0)
        self.assertFalse(download["appDownloadSubmittedForThisFamily"])
        self.assertFalse(download["directDownloadPerformed"])

        admission = self.review["admission"]
        self.assertEqual(
            admission["status"],
            "review_complete_blocked_full_job_cancellation_and_facade_contract",
        )
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertNotIn("LEditsPPPipelineStableDiffusion", IMAGE_PIPELINE_ADAPTERS)
        self.assertNotIn("LEditsPPPipelineStableDiffusionXL", IMAGE_PIPELINE_ADAPTERS)
        self.assertFalse(
            any(
                definition["modelType"].startswith("LEditsPP")
                for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
