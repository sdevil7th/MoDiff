import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "sd3-controlnet-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StableDiffusion3ControlNetArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_repositories_are_reviewed_but_not_cataloged(self):
        repositories = self.review["repositories"]
        self.assertEqual(
            repositories["base"]["revision"],
            "ea42f8cef0f178587cf766dc8129abd379c90671",
        )
        self.assertEqual(
            repositories["canny"]["revision"],
            "e42ecba9741dee3de8c54039eba8a879c0070ced",
        )
        self.assertEqual(
            repositories["tile"]["revision"],
            "48005f2448d5ea7aad3d383c2bb8b017723244d6",
        )
        self.assertEqual(
            repositories["inpainting"]["revision"],
            "c6763363fb1e9e9d7ce3e608b3999016e07805da",
        )
        for repository in repositories.values():
            self.assertFalse(repository["private"])
            self.assertFalse(repository["pythonFilesPresent"])
            self.assertFalse(repository["trustRemoteCodeRequired"])
            self.assertIsNone(catalog_repository_pin(repository["repository"]))

    def test_base_partition_is_exact_but_authenticated_configs_remain_unreviewed(self):
        base = self.review["repositories"]["base"]
        weights = self.review["selectedBaseWeightFiles"]
        self.assertEqual(base["gated"], "auto")
        self.assertEqual(base["selectedWeightFileCount"], 6)
        self.assertEqual(sum(item["byteSize"] for item in weights), 15499002486)
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in weights))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in weights))
        self.assertTrue(base["licenseFileAccessibleWithoutAuthentication"])
        self.assertFalse(base["modelIndexAccessibleWithoutAuthentication"])
        self.assertFalse(base["componentConfigsAccessibleWithoutAuthentication"])
        self.assertEqual(base["gatedConfigHttpStatus"], 401)

    def test_candidate_assembly_digests_and_bounds_are_reproducible(self):
        repositories = self.review["repositories"]
        for name in ("canny", "tile", "inpainting"):
            with self.subTest(name=name):
                assembly = self.review["reviewedAssemblies"][name]
                inventory = [
                    *self.review["selectedBaseWeightFiles"],
                    {
                        "path": "controlnet/diffusion_pytorch_model.safetensors",
                        "sha256": repositories[name]["weightSha256"],
                        "byteSize": repositories[name]["weightBytes"],
                    },
                ]
                canonical = json.dumps(
                    sorted(inventory, key=lambda item: item["path"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                self.assertEqual(assembly["weightFileCount"], 7)
                self.assertEqual(sum(item["byteSize"] for item in inventory), assembly["weightBytes"])
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), assembly["weightInventorySha256"])
                self.assertEqual(assembly["recommendedResolution"], [1024, 1024])
                self.assertEqual(assembly["numInferenceSteps"], 28)
                self.assertEqual(assembly["guidanceScale"], 7.0)

    def test_auxiliary_headers_are_safe_and_bounded_without_full_weight_downloads(self):
        repositories = self.review["repositories"]
        self.assertEqual(repositories["canny"]["safetensorsDtypeParameterCounts"], {"F16": 595428864})
        self.assertEqual(repositories["tile"]["safetensorsDtypeParameterCounts"], {"F16": 595428864})
        self.assertEqual(
            repositories["inpainting"]["safetensorsDtypeParameterCounts"],
            {"F16": 2080241664},
        )
        self.assertEqual(
            sum(item["remoteSafetensorsHeaderBytesFetched"] for item in repositories.values() if "remoteSafetensorsHeaderBytesFetched" in item),
            self.review["admission"]["remoteSafetensorsHeaderBytesFetched"],
        )
        self.assertFalse(self.review["admission"]["fullWeightFilesDownloaded"])

    def test_package_api_is_official_bounded_and_has_no_safety_checker(self):
        pinned = self.review["pinnedDiffusers"]
        for field in (
            "controlPipelineSha256",
            "inpaintingPipelineSha256",
            "controlNetSha256",
            "transformerSha256",
            "autoencoderSha256",
            "schedulerSha256",
            "outputSha256",
        ):
            self.assertRegex(pinned[field], SHA256)
        self.assertTrue(pinned["transformersOptionalRuntimeRequired"])
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertEqual(pinned["maxSequenceLengthDefault"], 256)
        self.assertEqual(pinned["maxSequenceLengthPackageUpperBound"], 512)
        self.assertEqual(
            pinned["modelCpuOffloadSequence"],
            "text_encoder->text_encoder_2->text_encoder_3->image_encoder->transformer->vae",
        )
        self.assertIn(
            "the package pipelines have no safety checker or equivalent output guardrail",
            self.review["admission"]["blockingReasons"],
        )

    def test_license_and_admission_fail_closed_without_overstating_rights(self):
        licenses = self.review["licenseReview"]
        self.assertTrue(licenses["base"]["nonCommercialOnly"])
        self.assertTrue(licenses["base"]["productionUseProhibitedWithoutSeparateLicense"])
        self.assertTrue(licenses["base"]["hostedServiceOrApiUseProhibitedWithoutSeparateLicense"])
        self.assertEqual(licenses["cannyAndTile"]["rightsStatus"], "undetermined")
        self.assertFalse(licenses["inpainting"]["derivativeWeightGrantUnambiguous"])
        self.assertFalse(licenses["inpainting"]["compatibleWithExactBaseLicenseUnambiguous"])
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_blocked")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertIn("physical_macos_execution", admission["unresolvedGates"])


if __name__ == "__main__":
    unittest.main()
