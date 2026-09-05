import copy
import json
from pathlib import PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from modiff.composite_migration_recovery_audit import scan_registered_cluster_recovery_audit
from modiff.huggingface_node_library import build_huggingface_node_library
from modiff.legacy_cluster_archived_definitions import (
    LegacyClusterArchivedDefinitionError,
    _hash_omitting,
    reviewed_archived_cluster_definition_evidence,
    reviewed_archived_cluster_definitions,
    validate_archived_cluster_definition_ledger,
)


EXPECTED_IDENTITIES = {
    (
        "diffusers.composite:WanTI2VPipeline:text_to_video",
        "sha256:69d22ed2d15598c7c888eb29a9f70423d91ee3e7bbf17d40fcc06c0b042bd915",
    ),
    (
        "diffusers.modular:HunyuanVideo15ModularPipeline:text2video",
        "sha256:d9990497f5cea8f3a2d30e3c4fc37f76524b59bfb29ccefc965129054e642d43",
    ),
    (
        "diffusers.modular:MiniMaxMusic3ModularPipeline:default",
        "sha256:915030cdaed8e5261bc64570c4a2eab17e5c45fcc281cd2143a1d48a8c705cad",
    ),
    (
        "diffusers.modular:QwenImageModularPipeline:text2image",
        "sha256:83a144c1f8681d16598ab9089d1f7140c3cdc28ab1196acabc86fa7d497b2944",
    ),
    (
        "diffusers.modular:StableDiffusionXLModularPipeline:controlnet_union_image2image",
        "sha256:4315adb71728a6d21578d8f217deac7ba4bba85bef12faa3e4da290566a2470c",
    ),
    (
        "diffusers.modular:StableDiffusionXLModularPipeline:ip_adapter_controlnet_union_inpainting",
        "sha256:32bc1deb00b369c02e7c3f2ad8b3d5f5ef44e5c2befa1d8be9fe2fa97e753009",
    ),
    (
        "transformers.composite:HuggingFaceCTCSpeechRecognitionModel:speech_to_text",
        "sha256:638b3821bc7dc004fe5e29c7e034a30a76d13007d9fc084f66e831fdce9b07a2",
    ),
}


class LegacyClusterArchivedDefinitionTests(unittest.TestCase):
    def test_checked_in_ledger_preserves_seven_exact_bodies_but_registers_only_six(self):
        ledger = reviewed_archived_cluster_definition_evidence()
        registered = reviewed_archived_cluster_definitions()

        self.assertEqual(ledger["schemaVersion"], 1)
        self.assertEqual(len(ledger["records"]), 7)
        self.assertEqual(len(registered), 6)
        self.assertEqual(
            {
                (record["definition"]["id"], record["definition"]["contentHash"])
                for record in ledger["records"]
            },
            EXPECTED_IDENTITIES,
        )
        self.assertEqual(ledger["contentHash"], _hash_omitting(ledger, "contentHash"))
        self.assertTrue(ledger["boundary"]["doesNotAuthorizeConversion"])
        self.assertTrue(ledger["boundary"]["doesNotAuthorizeWorkflowMutation"])
        self.assertTrue(ledger["boundary"]["doesNotAuthorizeExecution"])

        body_only = [
            record
            for record in ledger["records"]
            if record["registration"]["status"] == "archived_body_only"
        ]
        self.assertEqual(len(body_only), 1)
        self.assertEqual(
            body_only[0]["definition"]["id"],
            "diffusers.modular:HunyuanVideo15ModularPipeline:text2video",
        )
        self.assertEqual(
            body_only[0]["registration"]["reason"],
            "historical_execution_admission_absent",
        )
        self.assertEqual(body_only[0]["definition"]["executionAdmissions"], [])
        self.assertNotIn(body_only[0]["definition"]["id"], {item["id"] for item in registered})

    def test_every_body_and_source_receipt_is_collision_resistant_and_value_free(self):
        ledger = reviewed_archived_cluster_definition_evidence()

        for record in ledger["records"]:
            definition = record["definition"]
            source = record["source"]
            self.assertEqual(
                definition["contentHash"], _hash_omitting(definition, "contentHash")
            )
            self.assertEqual(record["recordHash"], _hash_omitting(record, "recordHash"))
            self.assertFalse(PurePosixPath(source["transcriptPath"]).is_absolute())
            self.assertNotIn("..", PurePosixPath(source["transcriptPath"]).parts)
            self.assertTrue(source["fetchRequestRecordSha256"].startswith("sha256:"))
            self.assertTrue(source["definitionResponseRecordSha256"].startswith("sha256:"))

        serialized = json.dumps(ledger)
        self.assertNotIn("/home/", serialized)
        self.assertNotIn("workflowPath", serialized)
        self.assertNotIn("instanceId", serialized)
        self.assertNotIn("20-year-old", serialized)

    def test_all_archived_block_references_remain_content_addressed_in_the_library(self):
        library = build_huggingface_node_library()
        block_ids = {item["id"] for item in library["blockDefinitions"]}

        for record in reviewed_archived_cluster_definition_evidence()["records"]:
            definition = record["definition"]
            referenced = {
                definition["rootBlockDefinitionId"],
                *(item["blockDefinitionId"] for item in definition["blockPlacements"]),
            }
            self.assertEqual(referenced - block_ids, set(), definition["id"])

    def test_validation_fails_closed_on_body_source_registration_and_order_drift(self):
        original = reviewed_archived_cluster_definition_evidence()

        body_drift = copy.deepcopy(original)
        body_drift["records"][0]["definition"]["label"] += " drift"
        with self.assertRaisesRegex(
            LegacyClusterArchivedDefinitionError, "complete historical body"
        ):
            validate_archived_cluster_definition_ledger(body_drift)

        source_drift = copy.deepcopy(original)
        source_drift["records"][0]["source"]["definitionResponseRecordBytes"] += 1
        with self.assertRaisesRegex(LegacyClusterArchivedDefinitionError, "recordHash"):
            validate_archived_cluster_definition_ledger(source_drift)

        registration_drift = copy.deepcopy(original)
        registration_drift["records"][0]["registration"] = {
            "status": "archived_body_only",
            "reason": "historical_execution_admission_absent",
        }
        with self.assertRaisesRegex(
            LegacyClusterArchivedDefinitionError, "recovered execution admissions"
        ):
            validate_archived_cluster_definition_ledger(registration_drift)

        order_drift = copy.deepcopy(original)
        order_drift["records"][0], order_drift["records"][1] = (
            order_drift["records"][1],
            order_drift["records"][0],
        )
        with self.assertRaisesRegex(LegacyClusterArchivedDefinitionError, "canonical identity order"):
            validate_archived_cluster_definition_ledger(order_drift)

    def test_returned_values_are_detached(self):
        ledger = reviewed_archived_cluster_definition_evidence()
        definitions = reviewed_archived_cluster_definitions()
        ledger["records"][0]["definition"]["label"] = "mutated"
        definitions[0]["label"] = "mutated"

        fresh = reviewed_archived_cluster_definition_evidence()
        self.assertNotEqual(fresh["records"][0]["definition"]["label"], "mutated")

    def test_default_recovery_scan_registers_the_checked_in_execution_complete_bodies(self):
        definition = reviewed_archived_cluster_definitions()[0]
        admission = definition["executionAdmissions"][0]
        inventory = {
            "workflows": [
                {
                    "composites": [
                        {
                            "classification": "legacy_cluster_root",
                            "derivedChildIds": [],
                            "sourceRefs": {
                                "legacyCluster": {
                                    "definition": {
                                        "id": definition["id"],
                                        "libraryRevision": definition["libraryRevision"],
                                        "contentHash": definition["contentHash"],
                                    },
                                    "admissionId": admission["id"],
                                    "studioExecutionSpec": admission["studioExecutionSpec"],
                                }
                            },
                        }
                    ]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as data_dir, patch(
            "modiff.composite_migration_inventory.scan_composite_migration_inventory",
            return_value=inventory,
        ):
            audit = scan_registered_cluster_recovery_audit(
                data_dir,
                library={"definitions": []},
            )

        self.assertEqual(audit["summary"]["archivedManifestExactInstanceCount"], 1)
        self.assertEqual(audit["summary"]["archivedManifestExactIdentityCount"], 1)
        self.assertEqual(audit["localEvidence"]["registeredArchivedDefinitionCount"], 6)
        self.assertTrue(audit["boundary"]["readOnly"])


if __name__ == "__main__":
    unittest.main()
