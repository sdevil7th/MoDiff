import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "hunyuanvideo-1.5-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HunyuanVideo15ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_territory_restricted_family_remains_contract_only_and_uncataloged(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "contract_only")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])
        for repository in self.review["repositories"]:
            self.assertRegex(repository["revision"], r"^[0-9a-f]{40}$")
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            self.assertIsNone(catalog_repository_pin(repository["repository"]))

        license_contract = self.review["license"]
        self.assertEqual(
            set(license_contract["preambleExcludedTerritories"]),
            {"European Union", "South Korea", "United Kingdom"},
        )
        self.assertEqual(license_contract["definedTerritoryExcludes"], ["European Union"])
        self.assertTrue(license_contract["territoryTextRequiresLegalReview"])
        self.assertEqual(license_contract["commercialLicenseThresholdMonthlyActiveUsers"], 100000000)
        self.assertTrue(license_contract["generatedContentDisclosureRequired"])

    def test_modular_contract_matches_reviewed_text_and_image_workflows(self):
        contract = reviewed_modular_workflow_contract("HunyuanVideo15ModularPipeline")
        reviewed = self.review["modularContract"]
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
        for workflow in workflows.values():
            defaults = {item["name"]: item["default"] for item in workflow["inputs"]}
            self.assertEqual(defaults["num_frames"], 121)
            self.assertEqual(defaults["num_inference_steps"], 50)
            self.assertIn("videos", {item["name"] for item in workflow["outputs"]})
        self.assertFalse(reviewed["promptRewriteDuringDiscoveryAllowed"])

    def test_exact_safe_weight_receipts_cover_upstream_and_two_diffusers_candidates(self):
        repositories = {item["role"]: item for item in self.review["repositories"]}
        self.assertEqual(
            {role: (item["weightFileCount"], item["weightBytes"]) for role, item in repositories.items()},
            {
                "upstreamFamily": (14, 371759988572),
                "diffusers480pTextToVideo": (13, 53367753676),
                "diffusers480pImageToVideoStepDistilled": (8, 34620593582),
            },
        )
        for repository in repositories.values():
            files = repository["weightFiles"]
            self.assertEqual(repository["weightFileCount"], len(files))
            self.assertEqual(repository["weightBytes"], sum(item["byteSize"] for item in files))
            self.assertEqual(len(files), len({item["path"] for item in files}))
            self.assertTrue(all(item["path"].endswith(".safetensors") for item in files))
            self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))

    def test_candidate_recipes_bind_full_t2v_and_step_distilled_i2v_without_live_claims(self):
        recipes = self.review["candidateRecipes"]
        text = recipes["textToVideo480p"]
        self.assertEqual(text["pipelineClass"], "HunyuanVideo15Pipeline")
        self.assertEqual((text["numFrames"], text["numInferenceSteps"], text["guidanceScale"]), (121, 50, 6.0))
        self.assertEqual(text["schedulerShift"], 5.0)
        self.assertTrue(text["modelCpuOffload"])
        self.assertTrue(text["vaeTiling"])

        image = recipes["imageToVideo480pStepDistilled"]
        self.assertEqual(image["pipelineClass"], "HunyuanVideo15ImageToVideoPipeline")
        self.assertEqual(image["recommendedNumInferenceSteps"], [8, 12])
        self.assertEqual(image["reviewedDefaultNumInferenceSteps"], 12)
        self.assertEqual(image["guidanceScale"], 1.0)
        self.assertEqual(image["schedulerShift"], 7.0)
        self.assertTrue(image["useMeanflow"])
        self.assertEqual(
            self.review["remoteResourceEnvelope"]["status"],
            "estimate_only_qualification_pending",
        )


if __name__ == "__main__":
    unittest.main()
