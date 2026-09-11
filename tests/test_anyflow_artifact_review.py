import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "anyflow-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AnyFlowArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_all_four_official_snapshots_are_reviewed_but_not_admitted(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_not_admitted_noncommercial")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])
        self.assertEqual(admission["executableModes"], [])

        repositories = self.review["repositories"]
        self.assertEqual(
            {repository["repository"] for repository in repositories},
            {
                "nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
                "nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers",
                "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers",
                "nvidia/AnyFlow-FAR-Wan2.1-14B-Diffusers",
            },
        )
        for repository in repositories:
            with self.subTest(repository=repository["repository"]):
                self.assertRegex(repository["revision"], SHA1)
                self.assertFalse(repository["gated"])
                self.assertFalse(repository["private"])
                self.assertFalse(repository["pythonFilesPresent"])
                self.assertFalse(repository["trustRemoteCodeRequired"])
                self.assertTrue(repository["licenseFilePresent"])
                self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_each_safe_weight_inventory_is_exact(self):
        shared = self.review["sharedWeightFiles"]
        self.assertEqual(len(shared), 6)
        self.assertEqual(sum(item["byteSize"] for item in shared), 23231263636)

        expected = {
            "bidirectional-1.3b": (7, 26074853036),
            "far-1.3b": (7, 26075642740),
            "bidirectional-14b": (9, 51863430540),
            "far-14b": (9, 51866062428),
        }
        for repository in self.review["repositories"]:
            with self.subTest(variant=repository["variant"]):
                files = sorted(shared + repository["variantWeightFiles"], key=lambda item: item["path"])
                count, byte_size = expected[repository["variant"]]
                self.assertEqual(repository["weightFileCount"], count)
                self.assertEqual(repository["weightBytes"], byte_size)
                self.assertEqual(len(files), count)
                self.assertEqual(sum(item["byteSize"] for item in files), byte_size)
                canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
                self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
                for item in files:
                    self.assertTrue(item["path"].endswith(".safetensors"))
                    self.assertRegex(item["sha256"], SHA256)

    def test_package_owned_bidirectional_and_far_contracts_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["revision"], "bb56997d4b7e87f0743f26a612f49ec4e7ce7213")
        self.assertEqual(pinned["bidirectionalPipelineClass"], "AnyFlowPipeline")
        self.assertEqual(pinned["farPipelineClass"], "AnyFlowFARPipeline")
        self.assertEqual(pinned["outputClass"], "AnyFlowPipelineOutput")
        self.assertEqual(pinned["schedulerClass"], "FlowMapEulerDiscreteScheduler")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        for field in (
            "bidirectionalPipelineSha256",
            "farPipelineSha256",
            "bidirectionalTransformerSha256",
            "farTransformerSha256",
            "outputSha256",
            "schedulerSha256",
        ):
            self.assertRegex(pinned[field], SHA256)

        contract = self.review["pipelineContract"]
        self.assertEqual(contract["bidirectionalTasks"], ["text-to-video"])
        self.assertEqual(
            contract["farTasks"],
            ["text-to-video", "image-to-video", "video-to-video"],
        )
        self.assertEqual(contract["farChunkPartition"], [1, 3, 3, 3, 3, 3, 3, 2])
        self.assertEqual(contract["numFramesCongruence"], "num_frames = 4n + 1")
        self.assertEqual(contract["documentedNumInferenceSteps"], 4)
        self.assertEqual(contract["defaultGuidanceScale"], 1.0)

    def test_noncommercial_license_keeps_every_surface_closed(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "nvidia-one-way-noncommercial-license-nsclv1")
        self.assertTrue(license_review["licenseFilePresent"])
        self.assertRegex(license_review["licenseFileSha256"], SHA256)
        self.assertFalse(license_review["commercialUseAllowed"])
        self.assertTrue(license_review["derivativeWorksRetainUseLimitation"])
        self.assertIn("noncommercial_research_only_license", self.review["admission"]["unresolvedGates"])

    def test_api_mismatch_bounds_and_resource_estimates_are_not_hidden(self):
        mismatch = self.review["modelCardPackageApiMismatch"]
        self.assertEqual(mismatch["currentPackageConditioningArgument"], "video")
        self.assertEqual(mismatch["immutableFarCardsConditioningArgument"], "context_sequence")
        self.assertTrue(mismatch["immutableCardsImportCustomUpstreamPipeline"])

        gaps = self.review["pipelineContract"]["packageValidationGaps"]
        self.assertIn("no maximum resolution or cumulative output-pixel bound", gaps)
        self.assertIn("no maximum output-frame bound", gaps)
        envelopes = self.review["remoteResourceEnvelopes"]
        self.assertEqual(envelopes["1.3b"]["status"], "estimate_only_remote_qualification_pending")
        self.assertEqual(envelopes["14b"]["status"], "estimate_only_remote_qualification_pending")
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])


if __name__ == "__main__":
    unittest.main()
