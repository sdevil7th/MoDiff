import json
from pathlib import Path
import re
import unittest

from modiff.model_artifact_catalog import catalog_repository_pin


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "minimax-h3-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MiniMaxH3ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_review_is_immutable_safe_and_not_exposed_as_a_downloadable_runtime_artifact(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertRegex(self.review["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(self.review["format"], "safetensors")
        self.assertFalse(self.review["hub"]["gated"])
        self.assertEqual(self.review["admission"]["status"], "contract_only")
        self.assertFalse(self.review["admission"]["runtimeCatalogExposed"])
        self.assertFalse(self.review["admission"]["downloadCatalogExposed"])
        self.assertEqual(self.review["admission"]["executableModes"], [])
        self.assertIsNone(catalog_repository_pin(self.review["repository"]))

    def test_exact_partition_receipt_selects_one_transformer_per_workflow(self):
        selection = self.review["artifactSelection"]
        partitions = selection["partitions"]
        files = [item for partition in partitions.values() for item in partition["weightFiles"]]
        self.assertEqual(len(files), 46)
        self.assertEqual(len({item["path"] for item in files}), 46)
        self.assertTrue(all(item["path"].endswith(".safetensors") for item in files))
        self.assertTrue(all(SHA256.fullmatch(item["sha256"]) for item in files))
        for partition in partitions.values():
            self.assertEqual(
                partition["weightByteSize"],
                sum(item["byteSize"] for item in partition["weightFiles"]),
            )
        self.assertEqual(partitions["shared"]["weightByteSize"], 77735901100)
        self.assertEqual(partitions["transformer"]["weightByteSize"], 66280504216)
        self.assertEqual(partitions["transformer_ref"]["weightByteSize"], 66280504216)
        self.assertNotEqual(
            {item["sha256"] for item in partitions["transformer"]["weightFiles"]},
            {item["sha256"] for item in partitions["transformer_ref"]["weightFiles"]},
        )
        self.assertEqual(selection["singleWorkflowWeightBytes"], 144016405316)
        self.assertEqual(selection["allRootModularWeightBytes"], 210296909532)
        self.assertEqual(
            {name: value["transformerPartition"] for name, value in self.review["workflowPartitions"].items()},
            {"t2va": "transformer", "fl2va": "transformer", "ref2va": "transformer_ref"},
        )

    def test_conditioner_scheduler_reference_and_license_boundaries_are_explicit(self):
        contracts = self.review["componentContracts"]
        self.assertEqual(contracts["textEncoder"]["class"], "Qwen3VLForConditionalGeneration")
        self.assertEqual(contracts["textEncoder"]["hiddenStateLayer"], 50)
        self.assertEqual(contracts["videoScheduler"]["shift"], 12.0)
        self.assertEqual(contracts["audioScheduler"]["shift"], 3.0)
        self.assertEqual(contracts["video"]["frameAlignment"], "17*n+5")
        bounds = self.review["referenceBounds"]
        self.assertEqual(
            (bounds["maximumImages"], bounds["maximumVideos"], bounds["maximumAudios"], bounds["maximumTotal"]),
            (9, 3, 3, 12),
        )
        self.assertFalse(bounds["audioOnlyAllowed"])
        self.assertEqual(
            set(self.review["license"]["excludedTerritories"]),
            {"European Union", "Republic of Korea", "United Kingdom", "United States of America"},
        )
        self.assertEqual(self.review["remoteResourceEnvelope"]["status"], "estimate_only_qualification_pending")


if __name__ == "__main__":
    unittest.main()
