import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "skyreels-v2-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SkyReelsV2ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_license_and_heavy_runtime_gates_remain_closed(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertEqual(self.review["format"], "safetensors")
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "contract_only")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])
        self.assertIn("license_scope_legal_review", admission["unresolvedGates"])

        license_contract = self.review["license"]
        self.assertFalse(license_contract["scopeAppliesUnambiguouslyToSkyReelsV2"])
        self.assertTrue(license_contract["scopeRequiresLegalReview"])
        linked = license_contract["linkedCommunityLicense"]
        self.assertEqual(linked["definitionNamesModel"], "Skywork-13B")
        self.assertFalse(linked["repositoryNoticePinsLinkedCommit"])
        self.assertTrue(linked["revocable"])
        self.assertTrue(linked["distributionIncludesHostedServices"])
        self.assertRegex(linked["immutableSourceCommit"], r"^[0-9a-f]{40}$")
        self.assertRegex(linked["sha256"], SHA256)

    def test_all_eight_official_diffusers_inventories_are_exact_and_uncataloged(self):
        expected = {
            "textToVideo14B540p": (18, 80385341396, 16),
            "textToVideo14B720p": (18, 80385341396, 16),
            "imageToVideo1.3B540p": (9, 32017449724, 36),
            "imageToVideo14B540p": (21, 91339283940, 36),
            "imageToVideo14B720p": (21, 91339283940, 36),
            "diffusionForcing1.3B540p": (8, 28973450372, 16),
            "diffusionForcing14B540p": (18, 80385341396, 16),
            "diffusionForcing14B720p": (18, 80385341396, 16),
        }
        repositories = self.review["repositories"]
        self.assertEqual(len(repositories), 8)
        for repository in repositories:
            component_count = sum(
                component["fileCount"] for component in repository["weightComponents"].values()
            )
            component_bytes = sum(
                component["byteSize"] for component in repository["weightComponents"].values()
            )
            self.assertEqual(component_count, repository["weightFileCount"])
            self.assertEqual(component_bytes, repository["weightBytes"])
            self.assertEqual(
                (
                    repository["weightFileCount"],
                    repository["weightBytes"],
                    repository["transformerInputChannels"],
                ),
                expected[repository["role"]],
            )
            self.assertRegex(repository["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(repository["weightInventorySha256"], SHA256)
            self.assertRegex(repository["transformerConfigSha256"], SHA256)
            self.assertRegex(repository["modelIndex"]["sha256"], SHA256)
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_five_pipeline_contracts_keep_required_media_inputs(self):
        contracts = {
            contract["pipelineClass"]: contract for contract in self.review["pipelineContracts"]
        }
        self.assertEqual(
            set(contracts),
            {
                "SkyReelsV2Pipeline",
                "SkyReelsV2ImageToVideoPipeline",
                "SkyReelsV2DiffusionForcingPipeline",
                "SkyReelsV2DiffusionForcingImageToVideoPipeline",
                "SkyReelsV2DiffusionForcingVideoToVideoPipeline",
            },
        )
        self.assertEqual(contracts["SkyReelsV2Pipeline"]["requiredInputs"], ["prompt"])
        self.assertEqual(
            contracts["SkyReelsV2ImageToVideoPipeline"]["requiredInputs"], ["image"]
        )
        self.assertEqual(
            contracts["SkyReelsV2DiffusionForcingPipeline"]["requiredInputs"], ["prompt"]
        )
        self.assertEqual(
            contracts["SkyReelsV2DiffusionForcingImageToVideoPipeline"]["requiredInputs"],
            ["image"],
        )
        self.assertEqual(
            contracts["SkyReelsV2DiffusionForcingVideoToVideoPipeline"]["requiredInputs"],
            ["video"],
        )

    def test_source_only_long_form_recipe_preserves_known_defects(self):
        recipes = self.review["reviewedRecipes"]
        diffusion_forcing = recipes["diffusionForcingTextToVideo"]
        self.assertEqual(
            (
                diffusion_forcing["numFrames"],
                diffusion_forcing["baseNumFrames"],
                diffusion_forcing["numInferenceStepsPerBlock"],
                diffusion_forcing["arStep"],
                diffusion_forcing["causalBlockSize"],
                diffusion_forcing["totalAsynchronousSchedulerRows"],
            ),
            (97, 97, 30, 5, 5, 50),
        )
        long_form = recipes["longFormWindowContract"]
        self.assertEqual(long_form["generatedNumFrames"], 257)
        self.assertEqual(long_form["overlapHistory"], 17)
        self.assertEqual(long_form["generatedWindows"], [[1, 97], [81, 177], [161, 257]])
        self.assertEqual(long_form["qualificationStatus"], "source_only")

        defects = recipes["unacceptedLongFormExamples"]
        self.assertFalse(defects["firstLastFrameRepositoryResolvable"])
        self.assertFalse(defects["videoToVideoRepositoryResolvable"])
        self.assertFalse(defects["videoToVideoPipelineSourceExamplePassesRequiredVideo"])

        self.assertTrue(
            all(
                envelope["status"] == "estimate_only_qualification_pending"
                for envelope in self.review["remoteResourceEnvelopes"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
