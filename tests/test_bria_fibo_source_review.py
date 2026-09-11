import hashlib
import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "bria-fibo-source-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BriaFiboSourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_gated_noncommercial_family_remains_uncataloged(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        admission = self.review["admission"]
        self.assertEqual(
            admission["status"],
            "reviewed_not_admitted_gated_noncommercial_remote_code_prompt_path",
        )
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["executableModes"], [])

        license_review = self.review["licenseReview"]
        self.assertTrue(license_review["acceptanceRequired"])
        self.assertFalse(license_review["commercialUseAllowedWithoutSeparateAgreement"])
        self.assertFalse(license_review["exactGatedRepositoryLicensePayloadsReviewed"])
        self.assertEqual(license_review["linkedLicenseDeed"], "cc-by-nc-4.0")
        for repository in self.review["repositories"]:
            self.assertTrue(repository["gated"])
            self.assertFalse(repository["private"])
            self.assertRegex(repository["revision"], SHA1)
            self.assertIsNone(catalog_repository_pin(repository["repository"]))
        self.assertNotIn("BriaFiboPipeline", IMAGE_PIPELINE_ADAPTERS)
        self.assertNotIn("BriaFiboEditPipeline", IMAGE_PIPELINE_ADAPTERS)

    def test_selected_model_and_vlm_inventories_are_exact(self):
        repositories = {item["role"]: item for item in self.review["repositories"]}
        vlms = {item["role"]: item for item in self.review["nestedPromptVlmRepositories"]}
        expected = {
            "structured_json_text_to_image": ("generation", 5, 25540786720),
            "structured_json_image_edit_and_inpaint": ("edit", 5, 24131409512),
            "generation_refine_inspire_vlm": ("generationPromptVlm", 2, 8875719328),
            "image_edit_vlm": ("editPromptVlm", 2, 8875719328),
        }
        for role, (files_key, count, byte_size) in expected.items():
            files = sorted(
                self.review["selectedWeightFiles"][files_key],
                key=lambda item: item["path"],
            )
            repository = repositories.get(role) or vlms[role]
            count_key = "selectedWeightFileCount" if role.startswith("structured_") else "weightFileCount"
            bytes_key = "selectedWeightBytes" if role.startswith("structured_") else "weightBytes"
            digest_key = (
                "selectedWeightInventorySha256"
                if role.startswith("structured_")
                else "weightInventorySha256"
            )
            self.assertEqual(repository[count_key], count)
            self.assertEqual(repository[bytes_key], byte_size)
            self.assertEqual(sum(item["byteSize"] for item in files), byte_size)
            canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.assertEqual(repository[digest_key], hashlib.sha256(canonical).hexdigest())
            for item in files:
                self.assertTrue(item["path"].endswith(".safetensors"))
                self.assertRegex(item["sha256"], SHA256)

        edit = repositories["structured_json_image_edit_and_inpaint"]
        self.assertEqual(edit["fullWeightFileCount"], 7)
        self.assertEqual(edit["fullWeightBytes"], 40703183416)
        self.assertEqual(edit["archivedTransformerWeightFileCount"], 2)
        self.assertEqual(edit["archivedTransformerWeightBytes"], 16571773904)
        archived = self.review["archivedWeightFiles"]["editTransformer"]
        archived_canonical = json.dumps(
            archived,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            edit["archivedTransformerWeightInventorySha256"],
            hashlib.sha256(archived_canonical).hexdigest(),
        )
        full = sorted(
            [*self.review["selectedWeightFiles"]["edit"], *archived],
            key=lambda item: item["path"],
        )
        full_canonical = json.dumps(full, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(
            edit["fullWeightInventorySha256"],
            hashlib.sha256(full_canonical).hexdigest(),
        )

    def test_package_owned_structured_image_contracts_are_recorded(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(
            pinned["revision"],
            "bb56997d4b7e87f0743f26a612f49ec4e7ce7213",
        )
        self.assertEqual(len(pinned["packageOwnedClasses"]), 3)
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(all(SHA256.fullmatch(value) for value in pinned["sourceSha256"].values()))

        generation = self.review["pipelineContracts"]["generation"]
        self.assertTrue(generation["structuredJsonRequiredForReviewedQuality"])
        self.assertFalse(generation["structuredJsonValidatedByPackage"])
        self.assertEqual(generation["maximumPackageSequenceLength"], 3000)
        self.assertEqual(generation["documentedRecommendedNumInferenceSteps"], 50)
        edit = self.review["pipelineContracts"]["edit"]
        self.assertTrue(edit["promptMustBeJsonWithEditInstruction"])
        self.assertTrue(edit["maskOptional"])
        self.assertTrue(edit["autoResizeDefault"])
        for contract in (generation, edit):
            self.assertIn("num_inference_steps", contract["unboundedByPackage"])
            self.assertIn("height", contract["unboundedByPackage"])
            self.assertIn("width", contract["unboundedByPackage"])

    def test_custom_promptifiers_are_revision_unbound_and_cuda_only(self):
        self.assertEqual(len(self.review["promptifierReviews"]), 2)
        for promptifier in self.review["promptifierReviews"]:
            self.assertRegex(promptifier["revision"], SHA1)
            self.assertRegex(promptifier["codeSha256"], SHA256)
            self.assertRegex(promptifier["modularConfigSha256"], SHA256)
            self.assertTrue(promptifier["trustRemoteCodeRequired"])
            self.assertEqual(promptifier["hardCodedDevice"], "cuda")
            self.assertFalse(promptifier["nestedModelRevisionPassedToFromPretrained"])
            self.assertFalse(promptifier["samplingBoundsOwnedByCode"])
            self.assertEqual(promptifier["defaultSamplingMaxTokens"], 4096)

        envelope = self.review["resourceEnvelope"]
        self.assertEqual(
            envelope["generationSelectedModelAndPromptVlmBytes"],
            25540786720 + 8875719328,
        )
        self.assertEqual(
            envelope["editSelectedModelAndPromptVlmBytes"],
            24131409512 + 8875719328,
        )
        self.assertIsNone(envelope["minimumAcceleratorMemoryBytesEstimate"])


if __name__ == "__main__":
    unittest.main()
