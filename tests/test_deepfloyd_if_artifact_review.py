import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "deepfloyd-if-artifact-review.json"
)
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DeepFloydIfArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_gated_noncommercial_family_remains_uncataloged(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        admission = self.review["admission"]
        self.assertEqual(
            admission["status"],
            "reviewed_not_admitted_gated_noncommercial_remote_heavy",
        )
        self.assertTrue(admission["remoteOnly"])
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])

        license_review = self.review["licenseReview"]["deepfloydStages"]
        self.assertEqual(license_review["id"], "deepfloyd-if-license")
        self.assertTrue(license_review["acceptanceRequired"])
        self.assertFalse(license_review["commercialUseAllowed"])
        self.assertFalse(license_review["productionUseAllowed"])
        self.assertTrue(license_review["restrictedOutputData"])

        repositories = self.review["repositories"]
        self.assertTrue(repositories[0]["gated"])
        self.assertTrue(repositories[1]["gated"])
        self.assertFalse(repositories[2]["gated"])
        for repository in repositories:
            self.assertRegex(repository["revision"], SHA1)
            self.assertFalse(repository["private"])
            self.assertFalse(repository["pythonFilesPresent"])
            self.assertFalse(repository["trustRemoteCodeRequired"])
            self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_selected_safe_partitions_are_exact(self):
        repositories = {item["role"]: item for item in self.review["repositories"]}
        expected = {
            "stage_one_64px": ("stageOne", 5, 20755397515),
            "stage_two_256px": ("stageTwo", 3, 3094664803),
            "stage_three_1024px": ("stageThree", 3, 3476599143),
        }
        for role, (files_key, count, byte_size) in expected.items():
            files = sorted(
                self.review["selectedWeightFiles"][files_key],
                key=lambda item: item["path"],
            )
            repository = repositories[role]
            self.assertEqual(repository["selectedWeightFileCount"], count)
            self.assertEqual(repository["selectedWeightBytes"], byte_size)
            self.assertEqual(len(files), count)
            self.assertEqual(sum(item["byteSize"] for item in files), byte_size)
            canonical = json.dumps(
                files,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.assertEqual(
                repository["selectedWeightInventorySha256"],
                hashlib.sha256(canonical).hexdigest(),
            )
            self.assertGreater(repository["fullWeightFileCount"], count)
            self.assertGreater(repository["fullWeightBytes"], byte_size)
            self.assertRegex(repository["fullWeightInventorySha256"], SHA256)
            for item in files:
                self.assertTrue(item["path"].endswith(".safetensors"))
                self.assertRegex(item["sha256"], SHA256)

        envelope = self.review["resourceEnvelope"]
        self.assertEqual(envelope["aggregateRepositoryBytes"], 256489971745)
        self.assertEqual(envelope["aggregateSafetensorsBytes"], 122691875683)
        self.assertEqual(envelope["repositoryScopedSelectedWeightBytes"], 27326661461)
        self.assertEqual(envelope["deduplicatedSelectedWeightBytes"], 26718655226)
        self.assertEqual(envelope["status"], "estimate_only_qualification_pending")

    def test_three_stage_recipe_and_package_contracts_are_sealed(self):
        contracts = self.review["pipelineContracts"]
        text_to_image = contracts["textToImage"]
        self.assertEqual(text_to_image["stageOne"]["class"], "IFPipeline")
        self.assertEqual(
            (
                text_to_image["stageOne"]["nativeResolution"],
                text_to_image["stageTwo"]["nativeResolution"],
                text_to_image["stageThree"]["nativeResolution"],
            ),
            (64, 256, 1024),
        )
        self.assertEqual(
            (
                text_to_image["stageOne"]["defaultNumInferenceSteps"],
                text_to_image["stageTwo"]["defaultNumInferenceSteps"],
                text_to_image["stageThree"]["defaultNumInferenceSteps"],
            ),
            (100, 50, 75),
        )
        self.assertEqual(text_to_image["stageOne"]["promptTokenLimit"], 77)
        self.assertEqual(text_to_image["stageTwo"]["promptTokenLimit"], 77)
        self.assertTrue(contracts["sharedSafety"]["safetyCheckerRequired"])
        self.assertTrue(contracts["sharedSafety"]["watermarkerRequired"])

        recipe = self.review["reviewedRecipe"]
        self.assertTrue(recipe["stageTwo"]["reusePromptEmbeddings"])
        self.assertFalse(recipe["stageTwo"]["textEncoderLoaded"])
        self.assertTrue(recipe["stageThree"]["reusedStageOneSafetyModules"])
        self.assertEqual(recipe["stageThree"]["noiseLevel"], 100)

    def test_source_and_gated_metadata_evidence_do_not_overclaim(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertFalse(pinned["deprecated"])
        self.assertTrue(pinned["legacyPerStepCallbackAvailable"])
        self.assertEqual(
            pinned["revision"],
            "bb56997d4b7e87f0743f26a612f49ec4e7ce7213",
        )
        self.assertEqual(len(pinned["packageOwnedClasses"]), 7)
        self.assertTrue(
            all(SHA256.fullmatch(digest) for digest in pinned["sourceSha256"].values())
        )

        metadata_access = self.review["metadataAccess"]
        self.assertEqual(metadata_access["gatedConfigHttpStatus"], 401)
        self.assertFalse(metadata_access["gatedConfigPayloadsReviewed"])
        self.assertTrue(metadata_access["stageThreeConfigPayloadsReviewed"])
        for repository in self.review["repositories"][:2]:
            self.assertTrue(
                all(
                    SHA1.fullmatch(digest)
                    for digest in repository["metadataBlobSha1"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
