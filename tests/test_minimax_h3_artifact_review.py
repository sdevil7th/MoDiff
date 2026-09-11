import json
import re
import unittest
from pathlib import Path

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter


REVIEW_PATH = Path(__file__).resolve().parents[1] / "data" / "minimax-h3-artifact-review.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MiniMaxH3ArtifactReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_review_is_immutable_safe_and_remains_closed_for_download_and_execution(self):
        self.assertEqual(self.review["schemaVersion"], 1)
        self.assertRegex(self.review["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(self.review["format"], "safetensors")
        self.assertFalse(self.review["hub"]["gated"])
        self.assertEqual(self.review["admission"]["status"], "graph_qualified")
        self.assertFalse(self.review["admission"]["runtimeCatalogExposed"])
        self.assertFalse(self.review["admission"]["downloadCatalogExposed"])
        self.assertEqual(self.review["admission"]["executableModes"], [])
        self.assertEqual(
            {route["workflowId"] for route in self.review["admission"]["graphQualifiedRoutes"]},
            {"t2va", "fl2va", "ref2va"},
        )
        self.assertTrue(self.review["admission"]["unresolvedGates"])
        # Structural V2 admission requires an exact backend-owned artifact
        # identity, but does not itself publish Download or Run authority.
        self.assertEqual(
            catalog_repository_pin(self.review["repository"])["revision"],
            self.review["revision"],
        )

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

    def test_three_workflows_have_exact_top_level_modular_graph_adapters(self):
        expected = {
            "t2va": ["text_encoder", "denoise", "decode"],
            "fl2va": ["before_encode", "text_encoder", "vae_encoder", "denoise", "decode"],
            "ref2va": ["before_encode", "text_encoder", "vae_encoder", "denoise", "decode"],
        }
        for workflow_id, sequence in expected.items():
            adapter = reviewed_whole_workflow_graph_adapter("MiniMaxH3ModularPipeline", workflow_id)
            self.assertEqual(adapter["adapterId"], "official_top_level_blocks")
            self.assertEqual(adapter["upstreamBlockSequence"], sequence)
            self.assertEqual(len(adapter["stateEdges"]), len(sequence) - 1)
        self.assertFalse(self.review["admission"]["runtimeCatalogExposed"])
        self.assertFalse(self.review["admission"]["downloadCatalogExposed"])


if __name__ == "__main__":
    unittest.main()
