import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "stable-cascade-artifact-review.json"
)
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StableCascadeArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_deprecated_noncommercial_family_remains_uncataloged(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        admission = self.review["admission"]
        self.assertEqual(
            admission["status"],
            "reviewed_not_admitted_deprecated_noncommercial",
        )
        self.assertTrue(admission["remoteOnly"])
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])

        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "stable-cascade-nc-community")
        self.assertFalse(license_review["commercialUseAllowed"])
        self.assertFalse(license_review["productionUseAllowed"])
        self.assertFalse(license_review["hostedServiceAllowed"])
        self.assertRegex(license_review["licenseSha256"], SHA256)

        for repository in self.review["repositories"]:
            self.assertRegex(repository["revision"], SHA1)
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["private"])
            self.assertFalse(repository["pythonFilesPresent"])
            self.assertFalse(repository["trustRemoteCodeRequired"])
            self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_selected_bf16_partitions_are_exact_and_exclude_repository_alternatives(
        self,
    ):
        repositories = {item["role"]: item for item in self.review["repositories"]}
        expected = {
            "prior_stage_c": (3, 9175740992),
            "decoder_stages_b_a": (3, 4552279604),
        }
        for role, files_key in (
            ("prior_stage_c", "prior"),
            ("decoder_stages_b_a", "decoder"),
        ):
            files = sorted(
                self.review["selectedWeightFiles"][files_key],
                key=lambda item: item["path"],
            )
            repository = repositories[role]
            self.assertEqual(
                (
                    repository["selectedWeightFileCount"],
                    repository["selectedWeightBytes"],
                ),
                expected[role],
            )
            self.assertEqual(repository["selectedWeightFileCount"], len(files))
            self.assertEqual(
                repository["selectedWeightBytes"],
                sum(item["byteSize"] for item in files),
            )
            canonical = json.dumps(
                files,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.assertEqual(
                repository["selectedWeightInventorySha256"],
                hashlib.sha256(canonical).hexdigest(),
            )
            self.assertGreater(repository["fullWeightFileCount"], len(files))
            self.assertGreater(
                repository["fullWeightBytes"],
                repository["selectedWeightBytes"],
            )
            self.assertRegex(repository["fullWeightInventorySha256"], SHA256)
            for item in files:
                self.assertTrue(item["path"].endswith(".safetensors"))
                self.assertRegex(item["sha256"], SHA256)

        self.assertEqual(
            self.review["resourceEnvelope"]["combinedSelectedWeightBytes"],
            13728020596,
        )
        self.assertEqual(
            sum(item["fullRepositoryBytes"] for item in repositories.values()),
            121051240157,
        )

    def test_two_stage_recipe_preserves_distinct_pipeline_contracts(self):
        contracts = self.review["pipelineContracts"]
        self.assertEqual(contracts["prior"]["class"], "StableCascadePriorPipeline")
        self.assertEqual(contracts["prior"]["requiredOneOf"], ["prompt", "prompt_embeds"])
        self.assertEqual(
            (
                contracts["prior"]["defaultHeight"],
                contracts["prior"]["defaultWidth"],
                contracts["prior"]["defaultNumInferenceSteps"],
                contracts["prior"]["defaultGuidanceScale"],
            ),
            (1024, 1024, 20, 4.0),
        )
        self.assertEqual(contracts["decoder"]["requiredInputs"], ["image_embeddings"])
        self.assertEqual(
            (
                contracts["decoder"]["defaultNumInferenceSteps"],
                contracts["decoder"]["defaultGuidanceScale"],
            ),
            (10, 0.0),
        )
        self.assertTrue(contracts["prior"]["callbackOnStepEnd"])
        self.assertTrue(contracts["decoder"]["callbackOnStepEnd"])

    def test_current_pin_deprecation_and_connected_revision_hazard_are_explicit(
        self,
    ):
        pinned = self.review["pinnedDiffusers"]
        self.assertTrue(pinned["deprecated"])
        self.assertEqual(pinned["lastSupportedVersion"], "0.35.2")
        self.assertEqual(
            pinned["revision"],
            "bb56997d4b7e87f0743f26a612f49ec4e7ce7213",
        )
        self.assertTrue(
            all(SHA256.fullmatch(digest) for digest in pinned["sourceSha256"].values())
        )
        combined = self.review["unsafeCombinedLoaderReason"]
        self.assertFalse(combined["connectedRepositoryRevisionEmbedded"])
        self.assertNotEqual(combined["decoderRevision"], combined["priorRevision"])
        self.assertFalse(
            self.review["pipelineContracts"]["combined"]["safeForImmutableAdmission"]
        )
        self.assertEqual(
            self.review["reviewedRecipe"]["route"],
            "separate_prior_then_decoder",
        )
        self.assertEqual(
            self.review["resourceEnvelope"]["status"],
            "estimate_only_qualification_pending",
        )


if __name__ == "__main__":
    unittest.main()
