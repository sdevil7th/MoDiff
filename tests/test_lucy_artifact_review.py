import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "lucy-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LucyArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_public_snapshot_is_reviewed_but_every_surface_remains_closed(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        repository = self.review["repository"]
        self.assertEqual(repository["repository"], "decart-ai/Lucy-Edit-Dev")
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["private"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertIsNone(catalog_repository_pin(repository["repository"]))

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_not_admitted_noncommercial_nonproduction")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertEqual(admission["executableModes"], [])

    def test_five_file_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 5)
        self.assertEqual(repository["weightBytes"], 34182223896)
        self.assertEqual(len(files), 5)
        self.assertEqual(sum(item["byteSize"] for item in files), 34182223896)
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

    def test_noncommercial_nonproduction_license_keeps_surfaces_closed(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "lucy-edit-5b-model-community-license")
        self.assertFalse(license_review["licenseFilePresent"])
        self.assertTrue(license_review["mutableExternalTerms"])
        self.assertFalse(license_review["commercialUseAllowed"])
        self.assertFalse(license_review["productionUseAllowed"])
        self.assertFalse(license_review["outputCommercialUseAllowed"])
        self.assertTrue(license_review["distributionIncludesHostedRemoteAccess"])
        self.assertRegex(license_review["externalLicensePdfSha256"], SHA256)
        self.assertIn("noncommercial_nonproduction_license", self.review["admission"]["unresolvedGates"])

    def test_package_owned_edit_contract_and_sources_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["revision"], "bb56997d4b7e87f0743f26a612f49ec4e7ce7213")
        self.assertEqual(pinned["pipelineClass"], "LucyEditPipeline")
        self.assertEqual(pinned["transformerClass"], "WanTransformer3DModel")
        self.assertEqual(pinned["vaeClass"], "AutoencoderKLWan")
        self.assertEqual(pinned["outputClass"], "LucyPipelineOutput")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        for field in ("pipelineSha256", "transformerSha256", "vaeSha256", "outputSha256"):
            self.assertRegex(pinned[field], SHA256)

        contract = self.review["pipelineContract"]
        self.assertEqual(contract["defaultHeight"], 480)
        self.assertEqual(contract["defaultWidth"], 832)
        self.assertEqual(contract["defaultNumFrames"], 81)
        self.assertEqual(contract["defaultNumInferenceSteps"], 50)
        self.assertEqual(contract["defaultGuidanceScale"], 5.0)
        self.assertEqual(contract["documentedFramesPerSecond"], 24)
        self.assertEqual(contract["numFramesCongruence"], "num_frames = 4n + 1")
        self.assertFalse(contract["packageSafetyCheckerPresent"])
        self.assertIn(
            "num_frames is normalized but is not used to select or validate the input video's frame count",
            contract["packageValidationGaps"],
        )

    def test_resource_values_remain_estimates_and_platform_proof_is_pending(self):
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "estimate_only_license_and_remote_qualification_pending")
        self.assertEqual(envelope["minimumAcceleratorMemoryBytesEstimate"], 51539607552)
        self.assertEqual(envelope["minimumSelectiveDiskBytes"], 42949672960)
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])


if __name__ == "__main__":
    unittest.main()
