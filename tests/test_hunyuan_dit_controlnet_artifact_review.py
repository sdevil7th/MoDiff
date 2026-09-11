import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "hunyuan-dit-controlnet-artifact-review.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HunyuanDiTControlNetArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_selected_assembly_is_exact_safe_and_reproducible(self):
        selected = self.review["selectedAssembly"]
        weights = self.review["weightFiles"]
        canonical = json.dumps(
            sorted(weights, key=lambda entry: entry["path"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(selected["weightFileCount"], 6)
        self.assertEqual(selected["weightBytes"], 17399623404)
        self.assertEqual(sum(entry["byteSize"] for entry in weights), selected["weightBytes"])
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), selected["weightInventorySha256"])
        self.assertTrue(all(SHA256.fullmatch(entry["sha256"]) for entry in weights))
        self.assertEqual(selected["runtimeDtype"], "float16")
        self.assertFalse(selected["fullWeightFilesDownloaded"])
        self.assertLess(selected["remoteSafetensorsHeaderBytesFetched"], 400000)

    def test_repository_identities_and_parameter_counts_are_sealed(self):
        base = self.review["repositories"]["base"]
        canny = self.review["repositories"]["canny"]
        self.assertEqual(base["revision"], "ba991d1546d8c50936c4c16398ed0a87b9b99fb1")
        self.assertEqual(canny["revision"], "b2d21391ebcf78939344cfec84891932f9d53aa0")
        self.assertEqual(base["safetensorsDtypeParameterCounts"], {"F32": 3605600391})
        self.assertEqual(canny["safetensorsDtypeParameterCounts"], {"F32": 744223296})
        for repository in (base, canny):
            self.assertFalse(repository["private"])
            self.assertFalse(repository["gated"])
            self.assertFalse(repository["pythonFilesPresent"])
            self.assertFalse(repository["trustRemoteCodeRequired"])
            self.assertFalse(repository["licenseFilePresent"])
            pin = catalog_repository_pin(repository["repository"])
            self.assertIsNotNone(pin)
            self.assertEqual(pin["revision"], repository["revision"])
            self.assertEqual(pin["license"], "tencent-hunyuan-community")

    def test_depth_and_pose_are_reviewed_catalog_substitutions_not_canonical_workflows(self):
        expected = {
            "depth": (
                "Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Depth",
                "c09e53a3f84294f1c64bbd6fbc1a13b889bf9f37",
                "14f688b71d9f311943ecb2ef395dea77be76a2f7b19a5db33c062cd2c8d9021e",
            ),
            "pose": (
                "Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Pose",
                "62a1747f7f0c999c10b0eb1da2d0d50e6fa80e50",
                "84dca4d3e8076fe10ef314be112a8916944bcecad48611ec08840670c23eafc7",
            ),
        }
        for variant in self.review["optionalReviewedVariants"]:
            with self.subTest(kind=variant["kind"]):
                repo, revision, digest = expected[variant["kind"]]
                self.assertEqual(variant["repository"], repo)
                self.assertEqual(variant["revision"], revision)
                self.assertEqual(variant["weightSha256"], digest)
                self.assertFalse(variant["canonicalWorkflow"])
                self.assertTrue(variant["catalogedExpertSubstitution"])
                pin = catalog_repository_pin(repo)
                self.assertEqual(pin["revision"], revision)
                self.assertEqual(pin["sha256"], digest)

    def test_immutable_license_terms_and_obligations_are_not_overstated(self):
        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "tencent-hunyuan-community")
        self.assertFalse(license_review["sourceRepositoriesContainLicenseFile"])
        self.assertTrue(license_review["sourceCardsLinkMutableMainLicense"])
        self.assertEqual(
            license_review["sealedLicenseRevision"],
            "b47a590cac7a3e1a973036700e45b3fe457e2239",
        )
        self.assertRegex(license_review["sealedLicenseSha256"], SHA256)
        self.assertTrue(license_review["acceptanceTriggeredByUse"])
        self.assertTrue(license_review["hostedServiceAddressed"])
        self.assertEqual(license_review["commercialLicenseThresholdMonthlyActiveUsers"], 100000000)
        self.assertTrue(license_review["machineGeneratedContentDisclosureRequiredForPublicUse"])
        self.assertTrue(license_review["agreementAndUseRestrictionNoticeRequiredForDistribution"])
        self.assertTrue(license_review["immutableTermsAcknowledgementAdded"])

    def test_package_sources_and_bounded_recipe_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        for field in (
            "pipelineSha256",
            "basePipelineSha256",
            "transformerSha256",
            "controlNetSha256",
            "autoencoderSha256",
            "schedulerSha256",
            "outputSha256",
        ):
            self.assertRegex(pinned[field], SHA256)
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        contract = self.review["pipelineContract"]
        self.assertEqual(
            {field: contract[field] for field in (
                "width",
                "height",
                "numInferenceSteps",
                "guidanceScale",
                "controlNetConditioningScale",
                "t5TokenLimit",
            )},
            {
                "width": 1024,
                "height": 1024,
                "numInferenceSteps": 50,
                "guidanceScale": 6.0,
                "controlNetConditioningScale": 1.0,
                "t5TokenLimit": 256,
            },
        )
        self.assertEqual(contract["preprocessor"], "canny")
        self.assertTrue(contract["baseDeclaresSafetyCheckerRequired"])
        self.assertFalse(contract["baseShipsSafetyChecker"])
        self.assertEqual(self.review["remoteResourceEnvelope"]["status"], "estimate_only_modiff_qualification_pending")
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])

    def test_portable_graph_is_exact_and_runtime_unqualified(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        entry = next(
            workflow
            for workflow in manifest["workflows"]
            if workflow["id"] == "HunyuanDiTControlNetPipeline:control_image"
        )
        self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")
        self.assertEqual(
            entry["requiredArtifacts"],
            [
                self.review["repositories"]["canny"]["repository"],
                self.review["repositories"]["base"]["repository"],
            ],
        )
        graph = json.loads((ROOT / "data" / "graphs" / entry["graphPath"]).read_text(encoding="utf-8"))
        nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
        loader = nodes["diffusersImagePipeline"]["data"]["params"]
        generate = nodes["diffusersImageControl"]["data"]["params"]
        preprocessor = nodes["controlPreprocessor"]["data"]["params"]
        self.assertEqual(loader["pipeline_class"]["value"], "HunyuanDiTControlNetPipeline")
        self.assertEqual(loader["revision"]["value"], self.review["repositories"]["base"]["revision"])
        self.assertEqual(
            loader["conditioning_revision"]["value"],
            self.review["repositories"]["canny"]["revision"],
        )
        self.assertEqual(loader["dtype"]["value"], "float16")
        self.assertEqual(generate["width"]["value"], 1024)
        self.assertEqual(generate["height"]["value"], 1024)
        self.assertEqual(generate["num_inference_steps"]["value"], 50)
        self.assertEqual(generate["guidance_scale"]["value"], 6)
        self.assertEqual(generate["conditioning_scale"]["value"], 1)
        self.assertEqual(preprocessor["low_threshold"]["value"], 0.1)
        self.assertEqual(preprocessor["high_threshold"]["value"], 0.2)


if __name__ == "__main__":
    unittest.main()
