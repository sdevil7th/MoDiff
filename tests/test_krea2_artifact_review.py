import hashlib
import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "krea2-artifact-review.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Krea2ArtifactReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_exact_gated_repositories_and_safe_candidate_partitions_are_sealed(self):
        candidate_files = self.review["candidateDiffusersFiles"]
        self.assertEqual(len(candidate_files), 17)
        self.assertTrue(all(not path.endswith(".py") for path in candidate_files))
        self.assertNotIn("raw.safetensors", candidate_files)
        self.assertNotIn("turbo.safetensors", candidate_files)

        for name, repository in self.review["repositories"].items():
            with self.subTest(name=name):
                self.assertRegex(repository["revision"], SHA1)
                self.assertFalse(repository["private"])
                self.assertEqual(repository["gated"], "auto")
                self.assertFalse(repository["pythonFilesPresent"])
                self.assertFalse(repository["trustRemoteCodeRequired"])
                self.assertEqual(repository["modelIndexClassFromRepositoryTag"], "Krea2Pipeline")
                self.assertFalse(repository["modelIndexAccessibleWithoutAcceptedGate"])
                self.assertEqual(repository["gatedArtifactHttpStatus"], 401)
                self.assertEqual(repository["selectedDiffusersFileCount"], len(candidate_files))
                self.assertEqual(repository["excludedPublisherMediaFileCount"], 37)
                self.assertRegex(repository["excludedRootNativeCheckpoint"]["sha256"], SHA256)

    def test_candidate_weight_inventories_are_exact_without_fetching_weight_bytes(self):
        for name, files in self.review["weightFiles"].items():
            with self.subTest(name=name):
                repository = self.review["repositories"][name]
                ordered = sorted(files, key=lambda item: item["path"])
                canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
                self.assertEqual(len(ordered), repository["selectedWeightFileCount"])
                self.assertEqual(sum(item["byteSize"] for item in ordered), repository["selectedWeightBytes"])
                self.assertEqual(
                    hashlib.sha256(canonical).hexdigest(),
                    repository["selectedWeightInventorySha256"],
                )
                self.assertTrue(all(item["path"].endswith(".safetensors") for item in ordered))
                self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in ordered))
        self.assertEqual(
            self.review["weightFiles"]["raw"][0],
            self.review["weightFiles"]["turbo"][0],
        )
        self.assertEqual(
            self.review["weightFiles"]["raw"][-1],
            self.review["weightFiles"]["turbo"][-1],
        )

    def test_standard_package_contract_and_existing_modular_contracts_are_distinct(self):
        pinned = self.review["pinnedDiffusers"]
        self.assertEqual(pinned["pipelineClass"], "Krea2Pipeline")
        self.assertEqual(pinned["transformerClass"], "Krea2Transformer2DModel")
        self.assertEqual(pinned["textEncoderClass"], "Qwen3VLModel")
        self.assertTrue(pinned["callbackOnStepEndAvailable"])
        self.assertTrue(pinned["interruptFlagAvailable"])
        self.assertTrue(pinned["interruptFlagReadByDenoisingLoop"])
        self.assertFalse(pinned["safetyCheckerAvailable"])
        self.assertTrue(all(SHA256.fullmatch(value) for value in pinned["sourceSha256"].values()))

        contract = self.review["pipelineContract"]
        self.assertEqual(contract["modes"], ["text_to_image"])
        self.assertEqual((contract["defaultWidth"], contract["defaultHeight"]), (1024, 1024))
        self.assertEqual(contract["dimensionMultiple"], 16)
        self.assertEqual(contract["base"]["numInferenceSteps"], 28)
        self.assertEqual(contract["base"]["guidanceScale"], 4.5)
        self.assertTrue(contract["base"]["negativePromptAvailable"])
        self.assertEqual(contract["turbo"]["numInferenceSteps"], 8)
        self.assertEqual(contract["turbo"]["guidanceScale"], 0.0)
        self.assertFalse(contract["turbo"]["negativePromptAvailable"])
        self.assertIsNone(contract["packageUpperBounds"]["numInferenceSteps"])
        self.assertIsNone(contract["packageUpperBounds"]["outputPixels"])

        base_modular = reviewed_modular_workflow_contract("Krea2ModularPipeline")
        turbo_modular = reviewed_modular_workflow_contract("Krea2TurboModularPipeline")
        self.assertEqual(base_modular["blocksClass"], "Krea2AutoBlocks")
        self.assertEqual(turbo_modular["blocksClass"], "Krea2TurboAutoBlocks")

    def test_base_and_turbo_graph_adapters_share_stages_but_preserve_distinct_actions(self):
        base = reviewed_whole_workflow_graph_adapter("Krea2ModularPipeline", "text2image")
        turbo = reviewed_whole_workflow_graph_adapter("Krea2TurboModularPipeline", "text2image")
        self.assertEqual(base["requiredInputs"], ["prompt"])
        self.assertEqual(turbo["requiredInputs"], ["prompt"])
        self.assertEqual(base["upstreamBlockSequence"], ["text_encoder", "denoise", "decode"])
        self.assertEqual(turbo["upstreamBlockSequence"], ["text_encoder", "denoise", "decode"])
        self.assertEqual(
            base["actionSequence"],
            ["workflow_krea2_text_encoder", "workflow_krea2_denoise", "workflow_krea2_decoder"],
        )
        self.assertEqual(
            turbo["actionSequence"],
            [
                "workflow_krea2_turbo_text_encoder",
                "workflow_krea2_turbo_denoise",
                "workflow_krea2_decoder",
            ],
        )

    def test_custom_terms_and_moving_use_policy_keep_standard_family_blocked(self):
        license_review = self.review["licenseReview"]
        self.assertEqual(license_review["modelCardLicenseTag"], "other")
        self.assertTrue(license_review["gateAcceptanceRequired"])
        self.assertFalse(license_review["taskScopedProductAcceptanceObtained"])
        self.assertRegex(license_review["licenseLinkCommit"], SHA1)
        self.assertRegex(license_review["licensePdfSha256"], SHA256)
        self.assertTrue(license_review["acceptanceTriggeredByDownloadAccessCopyUseOrDistribution"])
        self.assertEqual(license_review["commercialUseAnnualRevenueThresholdUsdExclusive"], 1000000)
        self.assertTrue(license_review["enterpriseLicenseRequiredAtOrAboveThreshold"])
        self.assertTrue(license_review["distributionRequiresRecipientAcceptanceLicenseCopyKreaNamePrefixAndNotice"])
        self.assertTrue(license_review["contentFiltersRequiredForDeployment"])
        self.assertTrue(license_review["acceptableUsePolicyIncorporatedByReference"])
        self.assertFalse(license_review["acceptableUsePolicyUrlImmutable"])
        self.assertFalse(license_review["legalAndProductApprovalComplete"])

    def test_app_preflight_and_admission_fail_closed_without_deletes_or_submission(self):
        preflight = self.review["appDownloadPreflight"]
        self.assertEqual(preflight["transportPolicy"], "app_only")
        self.assertTrue(preflight["planRequestedThroughApp"])
        self.assertFalse(preflight["submissionAttempted"])
        self.assertFalse(preflight["termsAcceptedByMoDiff"])
        self.assertFalse(preflight["directWeightDownloadPerformed"])
        self.assertFalse(preflight["olderModelsDeleted"])
        self.assertEqual(
            preflight["preSnapshotHeadroomAfterReserveBytes"],
            preflight["preflightFreeBytes"]
            - preflight["preflightExistingQueuedReservationBytes"]
            - preflight["reserveBytes"],
        )
        for name in ("raw", "turbo"):
            plan = preflight[name]
            self.assertTrue(plan["standaloneFits"])
            self.assertFalse(plan["fitsWithQueue"])
            self.assertEqual(
                plan["additionalBytesRequired"],
                plan["remainingBytes"] - preflight["preSnapshotHeadroomAfterReserveBytes"],
            )

        admission = self.review["admission"]
        self.assertEqual(admission["status"], "review_complete_blocked_custom_model_license_and_capacity")
        self.assertFalse(admission["runtimeCatalogExposed"])
        self.assertFalse(admission["downloadCatalogExposed"])
        self.assertFalse(admission["capabilityExposed"])
        self.assertEqual(admission["canonicalWorkflowsAdded"], 0)
        self.assertFalse(admission["clientSurfaceAdded"])
        self.assertFalse(admission["modelWeightsDownloaded"])
        self.assertEqual(admission["remoteWeightBytesFetched"], 0)
        for repository in self.review["repositories"].values():
            self.assertIsNone(catalog_repository_pin(repository["repository"]))
        self.assertNotIn("Krea2Pipeline", IMAGE_PIPELINE_ADAPTERS)
        self.assertFalse(
            any(
                definition["modelType"] == "Krea2Pipeline" for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
