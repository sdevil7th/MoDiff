import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "consisid-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ConsisIDArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_generator_snapshot_is_reviewed_but_every_surface_remains_closed(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        repository = self.review["repositories"][0]
        self.assertEqual(repository["repository"], "BestWishYsh/ConsisID-preview")
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["private"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        self.assertIsNone(catalog_repository_pin(repository["repository"]))

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_not_admitted_unsafe_face_stack")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertEqual(admission["executableModes"], [])

    def test_five_file_generator_safetensors_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repositories"][0]
        self.assertEqual(repository["weightFileCount"], 5)
        self.assertEqual(repository["weightBytes"], 22821396692)
        self.assertEqual(sum(item["byteSize"] for item in files), 22821396692)
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

    def test_required_face_stack_is_mixed_format_cuda_only_and_not_safe(self):
        stack = self.review["faceIdentityStack"]
        self.assertEqual(stack["requiredArtifactCount"], 8)
        self.assertEqual(stack["requiredArtifactBytes"], 1446798634)
        self.assertEqual(sum(item["byteSize"] for item in stack["requiredArtifacts"]), 1446798634)
        self.assertEqual(stack["unsafeRequiredArtifactCount"], 3)
        self.assertEqual(stack["hardcodedOnnxProviders"], ["CUDAExecutionProvider"])
        self.assertEqual(
            {item["format"] for item in stack["requiredArtifacts"]},
            {"onnx", "unsafe-pytorch-pt", "unsafe-pytorch-pth"},
        )
        for item in stack["requiredArtifacts"]:
            self.assertRegex(item["sha256"], SHA256)
        self.assertIn("safe_serialized_face_identity_stack", self.review["admission"]["unresolvedGates"])
        self.assertIn("package_owned_cross_platform_face_preprocessing", self.review["admission"]["unresolvedGates"])

    def test_identity_inputs_are_semantically_required_but_not_validated(self):
        contract = self.review["pipelineContract"]
        self.assertEqual(contract["defaultHeight"], 480)
        self.assertEqual(contract["defaultWidth"], 720)
        self.assertEqual(contract["defaultNumFrames"], 49)
        self.assertEqual(contract["defaultNumInferenceSteps"], 50)
        self.assertEqual(contract["defaultGuidanceScale"], 6.0)
        self.assertEqual(contract["maxSequenceLength"], 226)
        self.assertIn(
            "identity tensors are documented as crucial but not required by input validation",
            contract["packageValidationGaps"],
        )
        self.assertIn("biometric_privacy_consent_and_misuse_controls", self.review["admission"]["unresolvedGates"])

    def test_package_owned_pipeline_and_face_utility_hashes_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "ConsisIDPipeline")
        self.assertEqual(pinned["transformerClass"], "ConsisIDTransformer3DModel")
        self.assertEqual(pinned["outputClass"], "ConsisIDPipelineOutput")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        for field in ("pipelineSha256", "transformerSha256", "outputSha256", "faceUtilitySha256"):
            self.assertRegex(pinned[field], SHA256)

    def test_second_documented_official_checkpoint_is_not_invented(self):
        missing = self.review["repositories"][1]
        self.assertEqual(missing["repository"], "BestWishYsh/ConsisID-1.5")
        self.assertEqual(missing["availability"], "not_found_unauthenticated_2026-08-13")
        self.assertFalse(missing["runtimeEligible"])

    def test_license_and_resource_evidence_remain_qualified(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "apache-2.0")
        self.assertFalse(license_review["licenseFilePresent"])
        envelope = self.review["remoteResourceEnvelope"]
        self.assertEqual(envelope["status"], "upstream_measured_generator_only_face_stack_pending")
        self.assertEqual(envelope["upstreamReportedGpuMemoryGB"]["baselineReserved"], 44)
        self.assertEqual(envelope["upstreamReportedGpuMemoryGB"]["sequentialOffloadSlicingAndTilingReserved"], 7)
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])


if __name__ == "__main__":
    unittest.main()
