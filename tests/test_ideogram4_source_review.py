import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "ideogram4-source-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Ideogram4SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_gated_noncommercial_family_remains_uncataloged(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        admission = self.review["admission"]
        self.assertEqual(
            admission["status"],
            "reviewed_not_admitted_gated_noncommercial_unresolved_artifacts",
        )
        self.assertFalse(admission["gateAcceptancePerformed"])
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["liveQualified"])
        self.assertEqual(admission["weightBytesDownloaded"], 0)
        self.assertEqual(admission["executableModes"], [])

        license_review = self.review["licenseReview"]
        self.assertTrue(license_review["acceptanceRequired"])
        self.assertFalse(license_review["acceptancePerformed"])
        self.assertFalse(license_review["commercialUseAllowedWithoutSeparateAgreement"])
        self.assertTrue(license_review["distributionIncludesHostedServicesAndApis"])
        self.assertRegex(license_review["licenseFileSha256"], SHA256)

        for repository in self.review["repositories"]:
            self.assertTrue(repository["gated"])
            self.assertFalse(repository["private"])
            self.assertFalse(repository["pythonFilesPresent"])
            self.assertRegex(repository["revision"], SHA1)
            self.assertIsNone(catalog_repository_pin(repository["repository"]))
        self.assertNotIn("Ideogram4Pipeline", IMAGE_PIPELINE_ADAPTERS)

    def test_visible_sizes_are_not_misrepresented_as_exact_artifact_identities(self):
        metadata = self.review["metadataAccess"]
        self.assertTrue(metadata["artifactIdentityFieldsMaskedBeforeAcceptance"])
        self.assertFalse(metadata["exactArtifactIdentitiesAvailable"])
        self.assertEqual(metadata["modelIndexHttpStatus"], 401)
        self.assertEqual(metadata["componentConfigHttpStatus"], 401)

        repositories = {item["role"]: item for item in self.review["repositories"]}
        expected = {
            "preferred_package_owned_nf4_candidate": ("packageNf4", 16095406068),
            "official_nf4_snapshot": ("officialNf4", 16095321720),
            "official_fp8_snapshot_not_supported_by_reviewed_diffusers_recipe": (
                "officialFp8",
                27526985054,
            ),
        }
        for role, (files_key, byte_size) in expected.items():
            repository = repositories[role]
            files = self.review["visibleWeightFiles"][files_key]
            self.assertEqual(repository["visibleWeightFileCount"], 4)
            self.assertEqual(repository["visibleWeightBytes"], byte_size)
            self.assertEqual(sum(item["byteSize"] for item in files), byte_size)
            for item in files:
                self.assertTrue(item["path"].endswith(".safetensors"))
                self.assertNotIn("sha256", item)

        envelope = self.review["resourceEnvelope"]
        self.assertEqual(
            envelope["status"],
            "not_estimated_until_authenticated_artifact_and_nf4_runtime_review",
        )
        self.assertIsNone(envelope["minimumAcceleratorMemoryBytesEstimate"])

    def test_package_contract_and_safety_gap_are_sealed(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(
            pinned["revision"],
            "bb56997d4b7e87f0743f26a612f49ec4e7ce7213",
        )
        self.assertEqual(pinned["packageOwnedClasses"][0], "Ideogram4Pipeline")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["interruptAvailable"])
        self.assertTrue(pinned["loraLoaderMixinPresent"])
        self.assertFalse(pinned["safetyCheckerPresent"])
        self.assertTrue(all(SHA256.fullmatch(value) for value in pinned["sourceSha256"].values()))

        contract = self.review["pipelineContract"]
        self.assertEqual(contract["defaultHeight"], 2048)
        self.assertEqual(contract["defaultWidth"], 2048)
        self.assertEqual(contract["defaultNumInferenceSteps"], 48)
        self.assertEqual(contract["defaultMaxSequenceLength"], 2048)
        self.assertFalse(contract["defaultPromptUpsampling"])
        self.assertFalse(contract["negativePromptSupported"])
        self.assertTrue(contract["guidanceScaleAndScheduleMutuallyExclusive"])
        self.assertTrue(contract["guidanceScheduleLengthMustMatchInferenceSteps"])
        self.assertTrue(contract["promptUpsamplingExternalApiForbiddenByMoDiff"])
        self.assertEqual(contract["documentedResolutionContract"]["multiple"], 16)

    def test_optional_prompt_enhancer_is_not_silently_admitted(self):
        enhancer = self.review["optionalPromptEnhancer"]
        self.assertFalse(enhancer["admitted"])
        self.assertFalse(enhancer["hostedIdeogramApiAllowed"])
        self.assertFalse(enhancer["pipelineDefaultEnabled"])
        candidate = enhancer["localHeadCandidate"]
        self.assertRegex(candidate["revision"], SHA1)
        self.assertFalse(candidate["gated"])

    def test_modular_graph_adapter_keeps_prompt_upsampling_as_a_separate_stage(self):
        adapter = reviewed_whole_workflow_graph_adapter("Ideogram4ModularPipeline", "text2image")
        self.assertEqual(adapter["requiredInputs"], ["prompt"])
        self.assertEqual(
            adapter["upstreamBlockSequence"],
            ["prompt_upsample", "text_encoder", "denoise", "decode"],
        )
        self.assertEqual(
            adapter["actionSequence"],
            [
                "workflow_ideogram4_prompt_upsample",
                "workflow_ideogram4_text_encoder",
                "workflow_ideogram4_denoise",
                "workflow_ideogram4_decoder",
            ],
        )


if __name__ == "__main__":
    unittest.main()
