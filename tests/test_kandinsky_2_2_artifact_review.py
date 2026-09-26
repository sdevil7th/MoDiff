import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "kandinsky-2-2-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Kandinsky22ArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_all_five_official_versioned_snapshots_are_exact_and_uncataloged(self):
        expected = {
            "decoder": ("kandinsky-community/kandinsky-2-2-decoder", "9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3"),
            "prior": ("kandinsky-community/kandinsky-2-2-prior", "9fc51ad5732afc5d031724219d22e6c42179c5a8"),
            "inpaintDecoder": ("kandinsky-community/kandinsky-2-2-decoder-inpaint", "db790ad5cbcabed886f069ef2710774657621702"),
            "controlnetDepth": ("kandinsky-community/kandinsky-2-2-controlnet-depth", "4ecd717e8c9086cf4a16ca28b64894f70a42cd08"),
            "decoderRefiner": ("kandinsky-community/kandinsky-2-2-decoder-refiner", "9ae654a9d10d0a61c238fd4dd377d2f7a1b16386"),
        }
        discovery = self.review["officialRepositoryDiscovery"]
        self.assertEqual(discovery["exactVersion22RepositoryCount"], 5)
        self.assertEqual(set(self.review["repositories"]), set(expected))
        for role, identity in expected.items():
            with self.subTest(role=role):
                repository = self.review["repositories"][role]
                self.assertEqual((repository["repository"], repository["revision"]), identity)
                self.assertFalse(repository["private"])
                self.assertFalse(repository["gated"])
                self.assertFalse(repository["pythonFilesPresent"])
                self.assertFalse(repository["trustRemoteCodeRequired"])
                self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_safe_main_inventories_are_exact_and_only_headers_were_fetched(self):
        expected = {
            "decoder": (2, 5283689948, "a45c29fd2d223a898e4ecd3a037da02604d850a9e2eded9b8c840cf9b57694f7"),
            "prior": (3, 10573556608, "82183e03182b586858a9ea315a7b908f6ccc8911a2a355b3d653ec4d18288fe0"),
            "inpaintDecoder": (2, 5283759068, "7ba254f7f300d5ce8bf0c819fe5852be33309f363fccb5b36b1c9592b44e9554"),
        }
        unique_headers = {}
        for role, (count, byte_size, digest) in expected.items():
            repository = self.review["repositories"][role]
            weights = repository["weightFiles"]
            entries = [{field: item[field] for field in ("path", "byteSize", "sha256")} for item in weights]
            canonical = json.dumps(
                sorted(entries, key=lambda item: item["path"]),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.assertEqual((len(weights), sum(item["byteSize"] for item in weights)), (count, byte_size))
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)
            self.assertEqual(repository["selectedWeightInventorySha256"], digest)
            self.assertTrue(all(item["path"].endswith(".safetensors") for item in weights))
            self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in weights))
            for item in weights:
                unique_headers.setdefault(item["sha256"], item["headerBytesFetched"])

        evidence = self.review["headerEvidence"]
        self.assertEqual(len(unique_headers), evidence["uniqueSafetensorsArtifactsInspected"])
        self.assertEqual(sum(unique_headers.values()), evidence["uniqueSafetensorsHeaderBytes"])
        self.assertEqual((evidence["rangeRequests"], evidence["rangeBytesFetched"]), (7, 458000))
        self.assertFalse(evidence["fullWeightFilesDownloaded"])
        self.assertEqual(evidence["remoteWeightBytesFetched"], 0)

    def test_controlnet_and_refiner_fail_safe_serialization_gate(self):
        expected = {
            "controlnetDepth": (5285488364, "dd30812ab98bd0eb7e56fb6b80b48e832004caa71e1cb22cf5a4eea96d0daf03"),
            "decoderRefiner": (5283982252, "dfe02df39ea99107659a9aa227a340ba5b5089bb19c32c6520a6e9202d97ff5f"),
        }
        for role, (byte_size, digest) in expected.items():
            repository = self.review["repositories"][role]
            weights = repository["weightFiles"]
            entries = [{field: item[field] for field in ("path", "byteSize", "sha256")} for item in weights]
            canonical = json.dumps(
                sorted(entries, key=lambda item: item["path"]),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.assertIsNone(repository["selectedSerialization"])
            self.assertEqual(repository["safetensorsWeightFileCount"], 0)
            self.assertEqual(repository["legacyWeightFileCount"], 2)
            self.assertEqual(sum(item["byteSize"] for item in weights), byte_size)
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), digest)
            self.assertEqual(repository["legacyWeightInventorySha256"], digest)
            self.assertTrue(all(item["path"].endswith(".bin") for item in weights))
        refiner = self.review["repositories"]["decoderRefiner"]
        self.assertEqual(refiner["modelIndexPipelineClass"], "KandinskyPipeline")
        self.assertFalse(refiner["modelIndexClassMatchesVersion22Package"])
        self.assertIsNone(refiner["declaredLicense"])
        self.assertFalse(refiner["readmePresent"])

    def test_combined_callbacks_cover_both_stages_but_loader_remains_inexact(self):
        contracts = self.review["pipelineContracts"]
        modes = contracts["combinedModes"]
        self.assertEqual(modes["textToImage"]["requiredInputs"], ["prompt"])
        self.assertEqual(modes["imageToImage"]["requiredInputs"], ["prompt", "image"])
        self.assertEqual(modes["inpaint"]["requiredInputs"], ["prompt", "image", "mask_image"])
        self.assertEqual(
            (
                contracts["defaultNumInferenceSteps"],
                contracts["defaultPriorNumInferenceSteps"],
                contracts["defaultGuidanceScale"],
                contracts["defaultHeight"],
                contracts["defaultWidth"],
            ),
            (100, 25, 4.0, 512, 512),
        )
        self.assertTrue(contracts["priorModernStepEndCallbackAvailable"])
        self.assertTrue(contracts["decoderModernStepEndCallbackAvailable"])
        self.assertTrue(contracts["combinedCallbacksCoverBothDenoisingStages"])
        self.assertTrue(contracts["callbackExceptionCanAbortExecution"])
        self.assertFalse(contracts["safetyCheckerAvailable"])
        self.assertFalse(contracts["backendOwnedUpperBoundsPresent"])

        revision = self.review["connectedPriorRevisionReview"]
        self.assertTrue(revision["combinedPipelinesLoadConnectedPipes"])
        self.assertFalse(revision["connectedPriorRevisionEmbedded"])
        self.assertTrue(revision["fromPretrainedCopiesPrimaryRevisionToConnectedPrior"])
        self.assertFalse(revision["primaryRevisionsExistOnPriorRepository"])
        self.assertFalse(revision["immutableCombinedFromPretrainedLoadIsValid"])
        self.assertTrue(revision["downloadPathUsesMovingConnectedPrior"])
        self.assertFalse(revision["backendCompositeRevisionBindingImplemented"])

    def test_package_sources_license_and_composite_sizes_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["revision"], "bb56997d4b7e87f0743f26a612f49ec4e7ce7213")
        self.assertTrue(all(SHA256.fullmatch(digest) for digest in pinned["sourceSha256"].values()))
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["decoderPriorInpaintAndControlDeclaredLicense"], "apache-2.0")
        self.assertIsNone(license_review["refinerDeclaredLicense"])
        self.assertFalse(license_review["modelSnapshotLicenseFilePresent"])
        self.assertRegex(license_review["upstreamLicenseSha256"], SHA256)
        surfaces = self.review["compositeArtifactSurfaces"]
        self.assertEqual(surfaces["textAndImageGeneration"]["selectedWeightBytes"], 15857246556)
        self.assertEqual(surfaces["inpaint"]["selectedWeightBytes"], 15857315676)
        self.assertFalse(surfaces["controlnetDepth"]["safeCompositeAvailable"])
        self.assertFalse(surfaces["decoderRefiner"]["safeCompositeAvailable"])

    def test_admission_remains_closed_on_every_user_facing_surface(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_blocked")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertFalse(admission["liveQualified"])
        self.assertIn("exact_connected_prior_revision_binding", admission["unresolvedGates"])
        self.assertIn("official_controlnet_and_refiner_safetensors", admission["unresolvedGates"])
        self.assertIn("physical_macos_execution", admission["unresolvedGates"])
        self.assertFalse(
            any(
                definition.get("modelType", "").startswith("kandinsky-2-2")
                for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
