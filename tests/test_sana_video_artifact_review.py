import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data" / "sana-video-artifact-review.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SanaVideoArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_safe_repository_is_admitted_remote_only(self):
        admission = self.review["admission"]
        self.assertEqual(admission["status"], "source_complete_execution_pending")
        self.assertTrue(admission["runtimeCatalogExposed"])
        self.assertTrue(admission["downloadCatalogExposed"])
        self.assertTrue(admission["remoteOnly"])
        self.assertFalse(admission["liveQualified"])
        self.assertFalse(admission["autoEligible"])
        self.assertFalse(admission["galleryEligible"])

        repository = self.review["repository"]
        self.assertRegex(repository["revision"], SHA1)
        self.assertFalse(repository["gated"])
        self.assertFalse(repository["private"])
        self.assertFalse(repository["pythonFilesPresent"])
        self.assertFalse(repository["trustRemoteCodeRequired"])
        pin = catalog_repository_pin(repository["repository"])
        self.assertIsNotNone(pin)
        self.assertEqual(pin["revision"], repository["revision"])
        self.assertEqual(pin["license"], "apache-2.0")

    def test_five_file_mixed_precision_inventory_is_exact(self):
        files = sorted(self.review["weightFiles"], key=lambda item: item["path"])
        repository = self.review["repository"]
        self.assertEqual(repository["weightFileCount"], 5)
        self.assertEqual(repository["weightBytes"], 13963813420)
        self.assertEqual(sum(item["byteSize"] for item in files), repository["weightBytes"])
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(repository["weightInventorySha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(repository["safetensorsDtypeParameterCounts"], {"BF16": 2614341888, "F32": 2183754675})
        for item in files:
            self.assertTrue(item["path"].endswith(".safetensors"))
            self.assertRegex(item["sha256"], SHA256)

    def test_other_family_variants_and_unsafe_originals_are_explicitly_excluded(self):
        excluded = {item["id"]: item for item in self.review["excludedFamilyRepositories"]}
        self.assertEqual(
            set(excluded),
            {
                "original-480p-unsafe-pytorch",
                "original-720p-unsafe-pytorch",
                "diffusers-720p-unreviewed-heavy-variant",
                "original-longlive-unsafe-pytorch",
                "diffusers-longlive-duplicate-partitions",
            },
        )
        for item in excluded.values():
            self.assertRegex(item["revision"], SHA1)
        for key in ("original-480p-unsafe-pytorch", "original-720p-unsafe-pytorch", "original-longlive-unsafe-pytorch"):
            files = excluded[key]["unsafeWeightFiles"]
            self.assertEqual(excluded[key]["unsafeWeightBytes"], sum(item["byteSize"] for item in files))
            self.assertTrue(all(item["path"].endswith(".pth") for item in files))
            self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))

    def test_native_text_and_image_contracts_are_bounded(self):
        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_video", "image_to_video"])
        self.assertEqual(
            (
                contract["admittedWidth"],
                contract["admittedHeight"],
                contract["admittedNumFrames"],
                contract["admittedNumInferenceSteps"],
                contract["admittedGuidanceScale"],
                contract["admittedMotionScore"],
                contract["maxSequenceLength"],
                contract["documentedFramesPerSecond"],
            ),
            (832, 480, 81, 50, 6.0, 30, 300, 16),
        )
        self.assertEqual(contract["numFramesCongruence"], "num_frames = 4n + 1")
        self.assertTrue(contract["imageToVideoExactlyOneOpeningImage"])
        self.assertTrue(contract["vaeTilingRequired"])
        self.assertFalse(contract["resolutionBinning"])
        self.assertFalse(contract["safetyCheckerPresent"])

    def test_package_sources_license_and_resources_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["textToVideoPipelineClass"], "SanaVideoPipeline")
        self.assertEqual(pinned["imageToVideoPipelineClass"], "SanaImageToVideoPipeline")
        self.assertEqual(pinned["autoencoderClass"], "AutoencoderKLWan")
        self.assertEqual(pinned["transformerClass"], "SanaVideoTransformer3DModel")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["decodeOomWarningMayLeaveVideoUndefined"])
        self.assertTrue(pinned["mpsRotaryFrequencyDtypeWorkaroundPresent"])
        for field in (
            "textToVideoPipelineSha256",
            "imageToVideoPipelineSha256",
            "pipelineOutputSha256",
            "transformerSha256",
            "autoencoderSha256",
            "schedulerSha256",
        ):
            self.assertRegex(pinned[field], SHA256)
        self.assertTrue(all(SHA256.fullmatch(digest) for digest in self.review["metadataSha256"].values()))

        license_review = self.review["license"]
        self.assertEqual(license_review["id"], "apache-2.0")
        self.assertTrue(license_review["licenseFilePresent"])
        self.assertRegex(license_review["licenseSha256"], SHA256)
        self.assertEqual(self.review["remoteResourceEnvelope"]["status"], "estimate_only_modiff_qualification_pending")
        self.assertIn("physical_macos_execution", self.review["admission"]["unresolvedGates"])

    def test_portable_graphs_do_not_overstate_runtime_qualification(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = {
            "SanaVideoPipeline:text_to_video": False,
            "SanaImageToVideoPipeline:image_to_video": True,
        }
        for workflow_id, image_conditioned in expected.items():
            with self.subTest(workflow=workflow_id):
                entry = next(workflow for workflow in manifest["workflows"] if workflow["id"] == workflow_id)
                self.assertEqual(entry["requiredArtifacts"], [self.review["repository"]["repository"]])
                self.assertEqual(entry["runtimeQualificationStatus"], "unqualified")
                graph = json.loads((ROOT / "data" / "graphs" / entry["graphPath"]).read_text(encoding="utf-8"))
                nodes = {node["data"].get("studioRole"): node for node in graph["nodes"]}
                loader = nodes["wanPipeline"]["data"]["params"]
                generate = nodes["wanGenerate"]["data"]["params"]
                recipe = nodes["diffusersRecipe"]["data"]["params"]
                expected_class = "SanaImageToVideoPipeline" if image_conditioned else "SanaVideoPipeline"
                self.assertEqual(loader["pipeline_class"]["value"], expected_class)
                self.assertEqual(loader["revision"]["value"], self.review["repository"]["revision"])
                self.assertEqual(loader["dtype"]["value"], "bfloat16")
                self.assertEqual(recipe["offload_mode"]["value"], "sequential_cpu")
                self.assertFalse(recipe["vae_tiling"]["value"])
                self.assertEqual(
                    {field: generate[field]["value"] for field in (
                        "width", "height", "num_frames", "num_inference_steps", "guidance_scale", "max_sequence_length"
                    )},
                    {
                        "width": 832,
                        "height": 480,
                        "num_frames": 81,
                        "num_inference_steps": 50,
                        "guidance_scale": 6,
                        "max_sequence_length": 300,
                    },
                )
                roles = {node["data"].get("studioRole") for node in graph["nodes"]}
                self.assertEqual("loadImage" in roles, image_conditioned)


if __name__ == "__main__":
    unittest.main()
