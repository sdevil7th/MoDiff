import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from modiff.composite_migration_recovery_audit import (
    _migration_storage_evidence,
    build_registered_cluster_recovery_audit,
)
from modiff.legacy_cluster_archived_definitions import reviewed_archived_cluster_definitions
from modiff.legacy_cluster_compiler_mappings import reviewed_historical_compiler_mappings
from modiff.legacy_cluster_semantic_equivalence import canonical_content_hash
from modiff.legacy_cluster_studio_spec_evidence import (
    reviewed_studio_spec_evidence,
    reviewed_studio_spec_partial_reviews,
)
from tests.test_legacy_cluster_semantic_equivalence import (
    ledger as equivalence_ledger,
    semantic_equivalence_receipt,
)


def studio_spec(label="fixture"):
    return {
        "id": f"studio:{label}:v1",
        "contentHash": "studio-spec-v1-"
        + hashlib.sha256(label.encode()).hexdigest()[:8],
        "executionProfileId": f"profile:{label}",
    }


def definition(definition_id, admission_id, *, execution_spec=None):
    execution_spec = execution_spec or studio_spec(admission_id)
    body = {
        "id": definition_id,
        "libraryRevision": "a" * 40,
        "executionAdmissions": [{"id": admission_id, "studioExecutionSpec": execution_spec}],
        "provider": "diffusers",
        "publisher": "huggingface",
        "surface": "diffusers_cluster_nodes",
        "ownership": "library",
        "mutable": False,
        "pipelineClass": "FixtureModularPipeline",
        "workflowId": "default",
        "graphAdapterContracts": [],
        "inputs": [],
        "outputs": [],
        "components": [],
        "steps": [],
        "blockContractHash": "sha256:" + "f" * 64,
        "rootBlockDefinitionId": "fixture-root",
        "blockPlacements": [],
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "contentHash": "sha256:" + hashlib.sha256(encoded).hexdigest()}


def cluster(definition_id, manifest_hash, admission_id, *, execution_spec=None, derived_child_ids=None):
    return {
        "classification": "legacy_cluster_root",
        "sourceRefs": {
            "legacyCluster": {
                "definition": {
                    "id": definition_id,
                    "libraryRevision": "a" * 40,
                    "contentHash": manifest_hash,
                },
                "admissionId": admission_id,
                "studioExecutionSpec": (
                    execution_spec or studio_spec(admission_id)
                    if admission_id is not None
                    else None
                ),
            }
        },
        "derivedChildIds": list(derived_child_ids or []),
    }


class CompositeMigrationRecoveryAuditTests(unittest.TestCase):
    def test_current_archived_missing_receipt_and_historical_states_are_separate(self):
        current = definition("diffusers.modular:Current:default", "admission:current")
        archive = definition("diffusers.modular:Archive:default", "admission:archive")
        current_hash = current["contentHash"]
        archive_hash = archive["contentHash"]
        historical_hash = "sha256:" + "3" * 64
        missing_hash = "sha256:" + "4" * 64
        inventory = {
            "workflows": [
                {
                    "composites": [
                        cluster(current["id"], current_hash, "admission:current"),
                        cluster(current["id"], current_hash, "admission:current"),
                        cluster(archive["id"], archive_hash, "admission:archive"),
                        cluster("diffusers.modular:Historical:default", historical_hash, "admission:old"),
                        cluster("diffusers.modular:Missing:default", missing_hash, None),
                    ]
                }
            ]
        }

        audit = build_registered_cluster_recovery_audit(
            inventory,
            {"definitions": [current]},
            archived_definitions=[archive],
        )

        self.assertEqual(audit["schemaVersion"], 4)
        self.assertEqual(
            audit["summary"],
            {
                "legacyClusterInstanceCount": 5,
                "definitionIdentityCount": 4,
                "currentManifestExactInstanceCount": 2,
                "currentManifestExactIdentityCount": 1,
                "currentExecutionTupleExactInstanceCount": 2,
                "currentExecutionTupleExactIdentityCount": 1,
                "archivedManifestExactInstanceCount": 1,
                "archivedManifestExactIdentityCount": 1,
                "semanticEquivalenceReviewedInstanceCount": 0,
                "semanticEquivalenceReviewedIdentityCount": 0,
                "compilerMappingEligibleInstanceCount": 0,
                "compilerMappingEligibleIdentityCount": 0,
                "remainingBlockedHistoricalInstanceCount": 3,
                "remainingBlockedHistoricalIdentityCount": 3,
                "historicalEvidenceRequiredInstanceCount": 2,
                "historicalEvidenceRequiredIdentityCount": 2,
                "completeExecutionReceiptInstanceCount": 4,
                "missingExecutionReceiptInstanceCount": 1,
                "missingExecutionReceiptIdentityCount": 1,
                "incompleteExecutionReceiptInstanceCount": 0,
                "incompleteExecutionReceiptIdentityCount": 0,
                "identityReferenceOnlyInstanceCount": 5,
                "embeddedCompleteDefinitionInstanceCount": 0,
                "presentationProjectionInstanceCount": 0,
                "presentationProjectionNodeCount": 0,
                "malformedIdentityInstanceCount": 0,
                "malformedIdentityCount": 0,
                "recoveredStudioSpecBodyCount": 0,
                "recoveredStudioSpecManifestIdentityCount": 0,
                "recoveredStudioSpecExecutionTupleCount": 0,
                "recoveredStudioSpecInstanceCount": 0,
                "partialReviewVerifiedExecutionTupleCount": 0,
                "partialReviewRejectedExecutionTupleCount": 0,
            },
        )
        self.assertEqual(
            {item["disposition"] for item in audit["identities"]},
            {
                "current_manifest_exact",
                "archived_manifest_exact",
                "historical_manifest_evidence_required",
                "execution_receipt_missing",
            },
        )
        self.assertNotIn("sourcePath", str(audit))
        self.assertNotIn("instanceId", str(audit))
        self.assertTrue(audit["contentHash"].startswith("sha256:"))
        self.assertTrue(
            audit["boundary"]["recoveredStudioSpecEvidenceDoesNotAuthorizeConversion"]
        )
        self.assertTrue(
            audit["boundary"]["compilerMappingPresenceAloneDoesNotAuthorizeConversion"]
        )
        historical = next(
            item for item in audit["identities"] if item["disposition"] == "historical_manifest_evidence_required"
        )
        self.assertEqual(
            [item["code"] for item in historical["missingEvidence"]],
            ["archived_definition_body_not_embedded", "historical_manifest_authority_not_registered"],
        )
        self.assertEqual(
            historical["sourceEvidence"]["currentCatalog"]["status"],
            "definition_id_absent",
        )
        self.assertEqual(
            [item["code"] for item in historical["safeNextActions"]],
            ["recover_canonical_archived_definition", "review_exact_semantic_equivalence"],
        )

    def test_recovered_studio_body_is_visible_but_disposition_stays_fail_closed(self):
        evidence = reviewed_studio_spec_evidence()
        reviews = reviewed_studio_spec_partial_reviews()
        entry = evidence["specifications"][0]
        spec = entry["identity"]
        item = cluster(
            "diffusers.modular:RecoveredHistorical:default",
            "sha256:" + "7" * 64,
            "admission:recovered",
            execution_spec=copy.deepcopy(spec),
        )

        audit = build_registered_cluster_recovery_audit(
            {"workflows": [{"composites": [copy.deepcopy(item), copy.deepcopy(item)]}]},
            {"definitions": []},
            studio_spec_evidence_ledger=evidence,
            studio_spec_partial_review_ledger=reviews,
        )

        identity = audit["identities"][0]
        self.assertEqual(identity["disposition"], "historical_manifest_evidence_required")
        self.assertEqual(audit["summary"]["historicalEvidenceRequiredInstanceCount"], 2)
        self.assertEqual(audit["summary"]["recoveredStudioSpecBodyCount"], 1)
        self.assertEqual(audit["summary"]["recoveredStudioSpecManifestIdentityCount"], 1)
        self.assertEqual(audit["summary"]["recoveredStudioSpecExecutionTupleCount"], 1)
        self.assertEqual(audit["summary"]["recoveredStudioSpecInstanceCount"], 2)
        self.assertEqual(
            identity["sourceEvidence"]["recoveredStudioSpecs"]["status"],
            "partial_evidence_available",
        )
        self.assertFalse(
            identity["sourceEvidence"]["recoveredStudioSpecs"][
                "isManifestDefinitionOrConversionAuthority"
            ]
        )
        self.assertEqual(
            identity["executionTuples"][0]["recoveredStudioSpec"]["canonicalBodySha256"],
            entry["canonicalBodySha256"],
        )
        self.assertEqual(identity["executionTuples"][0]["manualReview"]["status"], "unreviewed")
        self.assertFalse(identity["executionTuples"][0]["manualReview"]["authorizesConversion"])
        gap_codes = {gap["code"] for gap in identity["missingEvidence"]}
        self.assertTrue(
            {
                "historical_manifest_definition_body_missing",
                "historical_public_interface_missing",
                "historical_component_and_block_hierarchy_missing",
                "historical_artifact_authority_missing",
                "historical_compiler_mapping_missing",
                "semantic_equivalence_not_reviewed",
            }.issubset(gap_codes)
        )
        self.assertIn(
            "review_recovered_studio_spec_partial_evidence",
            [action["code"] for action in identity["safeNextActions"]],
        )
        self.assertEqual(
            audit["partialStudioSpecEvidence"]["specifications"][0]["specification"],
            entry["specification"],
        )
        self.assertFalse(
            audit["partialStudioSpecEvidence"]["specifications"][0]["authorizesConversion"]
        )
        self.assertFalse(audit["partialStudioSpecEvidence"]["authorizesConversion"])

    def test_reviewed_equivalence_is_counted_without_workflow_or_instance_data(self):
        receipt = semantic_equivalence_receipt()
        historical = receipt["historical"]
        inventory = {
            "workflows": [
                {
                    "sourcePath": "user-workflows/private-name.json",
                    "composites": [
                        cluster(
                            historical["manifestDefinitionId"],
                            historical["manifestContentHash"],
                            historical["executionAdmissionId"],
                            execution_spec=historical["studioExecutionSpec"],
                        )
                    ],
                }
            ]
        }
        # The inventory helper defaults to the current test revision. Bind the
        # checked-in-style receipt to that exact historical tuple.
        receipt["historical"]["libraryRevision"] = "a" * 40
        receipt["receiptHash"] = canonical_content_hash(receipt, omit="receiptHash")

        audit = build_registered_cluster_recovery_audit(
            inventory,
            {"definitions": []},
            semantic_equivalence_ledger=equivalence_ledger(receipt),
        )

        self.assertEqual(audit["identities"][0]["disposition"], "semantic_equivalence_reviewed")
        self.assertEqual(audit["summary"]["semanticEquivalenceReviewedInstanceCount"], 1)
        self.assertEqual(audit["summary"]["historicalEvidenceRequiredInstanceCount"], 0)
        self.assertEqual(audit["summary"]["remainingBlockedHistoricalInstanceCount"], 0)
        serialized = json.dumps(audit)
        self.assertNotIn("private-name", serialized)
        self.assertNotIn("instanceId", serialized)

    def test_old_hash_without_complete_archived_definition_stays_blocked(self):
        manifest_hash = "sha256:" + "5" * 64
        item = cluster("diffusers.modular:Historical:default", manifest_hash, "admission:old")
        inventory = {"workflows": [{"composites": [item]}]}

        audit = build_registered_cluster_recovery_audit(inventory, {"definitions": []})
        self.assertEqual(audit["summary"]["historicalEvidenceRequiredInstanceCount"], 1)
        self.assertEqual(audit["identities"][0]["disposition"], "historical_manifest_evidence_required")

        malformed_archive = copy.deepcopy(item["sourceRefs"]["legacyCluster"]["definition"])
        malformed_archive["executionAdmissions"] = []
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_registered_cluster_recovery_audit(
                inventory,
                {"definitions": []},
                archived_definitions=[malformed_archive],
            )

    def test_matching_manifest_with_wrong_execution_tuple_stays_historical(self):
        current = definition("diffusers.modular:Current:default", "admission:current")
        item = cluster(
            current["id"],
            current["contentHash"],
            "admission:current",
            execution_spec=studio_spec("different-runtime"),
            derived_child_ids=["presentation-only"],
        )

        audit = build_registered_cluster_recovery_audit(
            {"workflows": [{"composites": [item]}]},
            {"definitions": [current]},
        )

        identity = audit["identities"][0]
        self.assertEqual(identity["disposition"], "historical_manifest_evidence_required")
        self.assertEqual(audit["summary"]["currentManifestExactInstanceCount"], 1)
        self.assertEqual(audit["summary"]["currentExecutionTupleExactInstanceCount"], 0)
        self.assertEqual(
            identity["sourceEvidence"]["currentCatalog"]["status"],
            "exact_manifest_execution_tuple_missing_or_mismatched",
        )
        self.assertEqual(identity["sourceEvidence"]["presentationProjection"]["nodeCount"], 1)
        self.assertIn(
            "presentation_projection_is_not_definition_authority",
            [item["code"] for item in identity["missingEvidence"]],
        )

    def test_complete_embedded_body_is_reported_but_never_auto_registered(self):
        archive = definition("diffusers.modular:Embedded:default", "admission:embedded")
        item = cluster(archive["id"], archive["contentHash"], "admission:embedded")
        item["sourceRefs"]["legacyCluster"]["definition"] = copy.deepcopy(archive)

        audit = build_registered_cluster_recovery_audit(
            {"workflows": [{"composites": [item]}]},
            {"definitions": []},
        )

        identity = audit["identities"][0]
        self.assertEqual(identity["disposition"], "historical_manifest_evidence_required")
        self.assertEqual(
            identity["sourceEvidence"]["embeddedManifest"]["completeDefinitionBodyInstanceCount"],
            1,
        )
        self.assertEqual(
            identity["safeNextActions"][0]["code"],
            "review_and_register_embedded_archive",
        )

    def test_partial_execution_identity_is_distinguished_from_fully_missing(self):
        item = cluster(
            "diffusers.modular:Partial:default",
            "sha256:" + "6" * 64,
            "admission:partial",
        )
        item["sourceRefs"]["legacyCluster"]["studioExecutionSpec"] = None

        audit = build_registered_cluster_recovery_audit(
            {"workflows": [{"composites": [item]}]},
            {"definitions": []},
        )

        self.assertEqual(audit["identities"][0]["disposition"], "execution_receipt_incomplete")
        self.assertEqual(audit["summary"]["incompleteExecutionReceiptInstanceCount"], 1)
        self.assertEqual(audit["summary"]["missingExecutionReceiptInstanceCount"], 0)

    def test_redacted_live_inventory_shape_preserves_manifest_and_receipt_axes(self):
        current_definitions = [
            definition(f"diffusers.modular:Current{index}:default", f"admission:current:{index}")
            for index in range(2)
        ]
        composites = []
        for current, count in zip(current_definitions, (10, 11), strict=True):
            composites.extend(
                cluster(current["id"], current["contentHash"], current["executionAdmissions"][0]["id"])
                for _ in range(count)
            )
        # Redacted synthetic identities preserve the exact post-reseal 2026-09-02 local
        # inventory cardinalities without checking in workflow names, IDs,
        # prompts, parameters, or user paths.
        for index in range(97):
            count = 4 if index < 52 else 3
            composites.extend(
                cluster(
                    f"diffusers.modular:Historical{index}:default",
                    "sha256:" + f"{index + 1:064x}",
                    f"admission:historical:{index}",
                )
                for _ in range(count)
            )
        for index, count in enumerate((5, 3, 3)):
            composites.extend(
                cluster(
                    f"diffusers.modular:MissingReceipt{index}:default",
                    "sha256:" + f"{index + 1000:064x}",
                    None,
                )
                for _ in range(count)
            )

        audit = build_registered_cluster_recovery_audit(
            {"workflows": [{"composites": composites}]},
            {"definitions": current_definitions},
        )

        summary = audit["summary"]
        self.assertEqual(summary["legacyClusterInstanceCount"], 375)
        self.assertEqual(summary["definitionIdentityCount"], 102)
        self.assertEqual(summary["currentManifestExactInstanceCount"], 21)
        self.assertEqual(summary["currentManifestExactIdentityCount"], 2)
        self.assertEqual(summary["currentExecutionTupleExactInstanceCount"], 21)
        self.assertEqual(summary["currentExecutionTupleExactIdentityCount"], 2)
        self.assertEqual(summary["compilerMappingEligibleInstanceCount"], 0)
        self.assertEqual(summary["compilerMappingEligibleIdentityCount"], 0)
        self.assertEqual(summary["remainingBlockedHistoricalInstanceCount"], 354)
        self.assertEqual(summary["remainingBlockedHistoricalIdentityCount"], 100)
        self.assertEqual(summary["historicalEvidenceRequiredInstanceCount"], 354)
        self.assertEqual(summary["historicalEvidenceRequiredIdentityCount"], 100)
        self.assertEqual(summary["completeExecutionReceiptInstanceCount"], 364)
        self.assertEqual(summary["missingExecutionReceiptInstanceCount"], 11)
        self.assertEqual(summary["missingExecutionReceiptIdentityCount"], 3)
        self.assertEqual(summary["identityReferenceOnlyInstanceCount"], 375)
        self.assertEqual(summary["embeddedCompleteDefinitionInstanceCount"], 0)

    def test_checked_in_compiler_mappings_report_only_redacted_aggregate_progress(self):
        mappings = reviewed_historical_compiler_mappings()
        expected_instances = {
            "legacy-cluster-compiler-mapping:minimax-music3:2026-09-02": 7,
            "legacy-cluster-compiler-mapping:transformers-ctc-stt:2026-09-02": 5,
            "legacy-cluster-compiler-mapping:wan-22-ti2v-5b:2026-09-02": 3,
        }
        composites = []
        for mapping in mappings["mappings"]:
            historical = mapping["historical"]
            for _ in range(expected_instances[mapping["id"]]):
                item = cluster(
                    historical["manifestDefinitionId"],
                    historical["manifestContentHash"],
                    historical["executionAdmissionId"],
                    execution_spec=historical["studioExecutionSpec"],
                )
                item["sourceRefs"]["legacyCluster"]["definition"]["libraryRevision"] = historical[
                    "libraryRevision"
                ]
                composites.append(item)

        audit = build_registered_cluster_recovery_audit(
            {"workflows": [{"sourcePath": "private-workflow.json", "composites": composites}]},
            {"definitions": []},
            archived_definitions=reviewed_archived_cluster_definitions(),
            historical_compiler_mapping_ledger=mappings,
        )

        self.assertEqual(audit["summary"]["legacyClusterInstanceCount"], 15)
        self.assertEqual(audit["summary"]["compilerMappingEligibleInstanceCount"], 15)
        self.assertEqual(audit["summary"]["compilerMappingEligibleIdentityCount"], 3)
        self.assertEqual(audit["summary"]["remainingBlockedHistoricalInstanceCount"], 0)
        self.assertEqual(audit["summary"]["remainingBlockedHistoricalIdentityCount"], 0)
        self.assertEqual(audit["localEvidence"]["checkedInReviewedCompilerMappingCount"], 3)
        self.assertEqual(
            audit["localEvidence"]["checkedInReviewedCompilerMappingLedgerHash"],
            mappings["contentHash"],
        )
        serialized = json.dumps(audit)
        self.assertNotIn("private-workflow", serialized)
        self.assertNotIn("legacy-cluster-compiler-mapping:", serialized)

    def test_archives_cannot_shadow_a_current_manifest(self):
        current = definition("diffusers.modular:Current:default", "admission:current")
        with self.assertRaisesRegex(ValueError, "must not duplicate"):
            build_registered_cluster_recovery_audit(
                {"workflows": []},
                {"definitions": [current]},
                archived_definitions=[current],
            )

    def test_archived_definition_rejects_duplicate_execution_admissions(self):
        archive = definition("diffusers.modular:Archive:default", "admission:archive")
        archive["executionAdmissions"].append(
            {
                "id": "admission:archive",
                "studioExecutionSpec": studio_spec("conflicting-runtime"),
            }
        )
        body = {key: value for key, value in archive.items() if key != "contentHash"}
        archive["contentHash"] = "sha256:" + hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        with self.assertRaisesRegex(ValueError, "repeats an execution admission"):
            build_registered_cluster_recovery_audit(
                {"workflows": []},
                {"definitions": []},
                archived_definitions=[archive],
            )

    def test_malformed_identity_values_are_not_echoed_by_redacted_audit(self):
        private_values = {
            "definition": "private/workflow.json",
            "revision": "secret prompt from workflow",
            "manifest": "private-output.png",
            "admission": "the private prompt text",
            "studio": "/home/user/private.wav",
            "studioHash": "another private prompt",
            "profile": "workflow/path",
        }
        item = {
            "classification": "legacy_cluster_root",
            "sourceRefs": {
                "legacyCluster": {
                    "definition": {
                        "id": private_values["definition"],
                        "libraryRevision": private_values["revision"],
                        "contentHash": private_values["manifest"],
                    },
                    "admissionId": private_values["admission"],
                    "studioExecutionSpec": {
                        "id": private_values["studio"],
                        "contentHash": private_values["studioHash"],
                        "executionProfileId": private_values["profile"],
                    },
                }
            },
            "derivedChildIds": [],
        }

        audit = build_registered_cluster_recovery_audit(
            {"workflows": [{"sourcePath": "also-private.json", "composites": [item]}]},
            {"definitions": []},
        )

        identity = audit["identities"][0]
        self.assertEqual(identity["disposition"], "identity_malformed")
        self.assertIsNone(identity["definitionId"])
        self.assertIsNone(identity["libraryRevision"])
        self.assertIsNone(identity["manifestContentHash"])
        self.assertEqual(identity["admissionIds"], [])
        serialized = json.dumps(audit)
        for value in (*private_values.values(), "also-private.json"):
            self.assertNotIn(value, serialized)

    def test_migration_journal_evidence_is_counted_without_exposing_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "studio" / "composite-migrations" / "private-migration"
            (root / "backups" / "user-workflows").mkdir(parents=True)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "backups" / "user-workflows" / "private-workflow.json").write_text(
                "{}", encoding="utf-8"
            )

            evidence = _migration_storage_evidence(temporary)

        self.assertEqual(
            evidence,
            {"status": "available", "journalManifestCount": 1, "backupFileCount": 1},
        )
        self.assertNotIn("private", json.dumps(evidence))


if __name__ == "__main__":
    unittest.main()
