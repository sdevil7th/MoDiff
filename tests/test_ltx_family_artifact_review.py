import json
from pathlib import Path
import re
import unittest

from modiff.diffusers_profiles import execution_profiles_for_execution
from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "ltx-family-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LTXFamilyArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_existing_graph_surfaces_are_preserved_while_modular_stays_contract_only(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "existing_graph_surfaces_modular_contract_only")
        self.assertEqual(admission["ltxModularStatus"], "contract_only")
        self.assertEqual(admission["ltx2ModularStatus"], "contract_only")
        self.assertFalse(admission["runtimeCatalogWidened"])
        self.assertFalse(admission["newDownloadCatalogEntryAdded"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(len(admission["existingStandardGraphSurfacesPreserved"]), 8)

        for repository in self.review["repositories"]:
            pin = catalog_repository_pin(repository["repository"])
            self.assertIsNotNone(pin)
            self.assertEqual(pin["revision"], repository["revision"])

    def test_ltx_13b_profiles_no_longer_accept_the_2b_family_index_as_fallback(self):
        correction = self.review["fallbackCorrection"]
        self.assertTrue(correction["removedFromLtx13BExecutionProfiles"])
        self.assertTrue(correction["keptInArtifactCatalogForExplicitReview"])
        self.assertTrue(correction["removedFromPublishedExecutionCandidates"])
        for mode in ("text_to_video", "image_to_video", "video_to_video", "reference_to_video"):
            profiles = execution_profiles_for_execution("LTXVideoPipeline", mode)
            self.assertEqual(len(profiles), 1)
            self.assertIsNone(profiles[0].fallback_repo)
            self.assertEqual(
                profiles[0].default_repo,
                "Lightricks/LTX-Video-0.9.8-13B-distilled",
            )

    def test_ltx_modular_contract_matches_the_two_exact_upstream_workflows(self):
        reviewed = self.review["ltxModularContract"]
        contract = reviewed_modular_workflow_contract(reviewed["pipelineClass"])
        self.assertEqual(contract["blocksClass"], reviewed["blocksClass"])
        workflows = {workflow["id"]: workflow for workflow in contract["workflows"]}
        self.assertEqual(set(workflows), set(reviewed["workflows"]))
        for name, expected in reviewed["workflows"].items():
            workflow = workflows[name]
            self.assertEqual(workflow["taskId"], expected["taskId"])
            self.assertEqual(workflow["requiredInputs"], expected["requiredInputs"])
            defaults = {item["name"]: item["default"] for item in workflow["inputs"]}
            self.assertEqual(
                (
                    defaults["height"],
                    defaults["width"],
                    defaults["num_frames"],
                    defaults["num_inference_steps"],
                ),
                (
                    expected["defaultHeight"],
                    expected["defaultWidth"],
                    expected["defaultNumFrames"],
                    expected["defaultNumInferenceSteps"],
                ),
            )
            self.assertIn("videos", {item["name"] for item in workflow["outputs"]})

    def test_inventory_receipts_separate_exact_selected_and_excluded_partitions(self):
        expected = {
            "classic13B": (21, 92762951244, 11, 47628403428),
            "classicFamilyCandidate": (27, 253809013320, 7, 28419691124),
            "ltx2": (44, 314290794056, 23, 92034380210),
        }
        for repository in self.review["repositories"]:
            selected_count = sum(
                component["weightFileCount"]
                for component in repository["selectedComponents"].values()
            )
            selected_bytes = sum(
                component["weightBytes"] for component in repository["selectedComponents"].values()
            )
            excluded_count = sum(item["fileCount"] for item in repository["excludedWeightGroups"])
            excluded_bytes = sum(item["byteSize"] for item in repository["excludedWeightGroups"])
            self.assertEqual(selected_count, repository["selectedWeightFileCount"])
            self.assertEqual(selected_bytes, repository["selectedWeightBytes"])
            self.assertEqual(selected_count + excluded_count, repository["fullWeightFileCount"])
            self.assertEqual(selected_bytes + excluded_bytes, repository["fullWeightBytes"])
            self.assertEqual(
                (
                    repository["fullWeightFileCount"],
                    repository["fullWeightBytes"],
                    repository["selectedWeightFileCount"],
                    repository["selectedWeightBytes"],
                ),
                expected[repository["role"]],
            )
            self.assertRegex(repository["fullWeightInventorySha256"], SHA256)
            self.assertRegex(repository["selectedWeightInventorySha256"], SHA256)
            for component in repository["selectedComponents"].values():
                digest = component.get("indexSha256") or component.get("sha256")
                self.assertRegex(digest, SHA256)

    def test_ltx2_license_and_two_stage_recipe_are_explicit_without_live_claims(self):
        license_contract = self.review["licenses"]["ltx2"]
        self.assertEqual(license_contract["commercialUseAgreementThresholdUsdAnnualRevenue"], 10000000)
        self.assertTrue(license_contract["entityRevenueAggregatesAffiliates"])
        self.assertTrue(license_contract["derivativeTransferObligations"])
        self.assertRegex(license_contract["sha256"], SHA256)
        self.assertFalse(
            self.review["licenses"]["ltxVideo"]["immutableRepositoryLicenseTextReviewed"]
        )

        recipe = self.review["reviewedRecipes"]["ltx2TwoStage"]
        self.assertEqual((recipe["width"], recipe["height"], recipe["numFrames"]), (768, 512, 121))
        self.assertEqual(
            (recipe["stage1"]["numInferenceSteps"], recipe["stage1"]["guidanceScale"]),
            (40, 4.0),
        )
        self.assertEqual(
            (recipe["stage2"]["numInferenceSteps"], recipe["stage2"]["guidanceScale"]),
            (3, 1.0),
        )
        self.assertTrue(recipe["stage2"]["reusesStage1AudioLatents"])
        self.assertFalse(
            self.review["reviewedRecipes"]["classic13BDistilled"][
                "exactDiffusersRecipePresentInPinnedModelCard"
            ]
        )
        self.assertTrue(
            all(
                value["status"] == "estimate_only_qualification_pending"
                for value in self.review["remoteResourceEnvelopes"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
