import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modiff.block_definition_v2 import (
    block_definition_canonical_sha256_v2,
    block_interface_hash_v2,
)
from modiff.composite_migration import (
    APPLY_CONFIRMATION,
    ROLLBACK_CONFIRMATION,
    CompositeMigrationAuthorizationError,
    apply_composite_migration,
    build_composite_migration_preview,
    rollback_composite_migration,
    scan_composite_migration_preview,
)
from modiff.legacy_cluster_archived_definitions import (
    reviewed_archived_cluster_definition_evidence,
)
from modiff.legacy_cluster_compiler_mappings import (
    canonical_content_hash,
    validate_historical_compiler_mapping,
)
from tests.test_composite_migration import (
    FAKE_REGISTERED_ADMISSION,
    registered_cluster_compiler_fixture,
    registered_cluster_workflow,
)


def archived_mapping_fixture(workflow, supplement):
    record = next(
        item
        for item in reviewed_archived_cluster_definition_evidence()["records"]
        if item["definition"]["id"]
        == "diffusers.modular:QwenImageModularPipeline:text2image"
    )
    historical_definition = record["definition"]
    historical_admission = historical_definition["executionAdmissions"][0]
    root = next(node for node in workflow["snapshot"]["nodes"] if node["id"] == "cluster-root")
    legacy = root["data"]["huggingFaceClusterInstance"]
    legacy["definition"] = {
        "id": historical_definition["id"],
        "libraryRevision": historical_definition["libraryRevision"],
        "contentHash": historical_definition["contentHash"],
    }
    legacy["execution"]["admissionId"] = historical_admission["id"]
    legacy["execution"]["studioExecutionSpec"] = copy.deepcopy(
        historical_admission["studioExecutionSpec"]
    )
    conversion = supplement["compilerOutputs"][0]["conversions"][0]
    destination_definition = conversion["blockInstanceV2"]["definitionSnapshot"]
    destination_source = destination_definition["source"]
    mapping = {
        "id": "legacy-cluster-compiler-mapping:qwen:text2image:fixture",
        "historical": {
            "manifestDefinitionId": historical_definition["id"],
            "libraryRevision": historical_definition["libraryRevision"],
            "manifestContentHash": historical_definition["contentHash"],
            "executionAdmissionId": historical_admission["id"],
            "studioExecutionSpec": copy.deepcopy(historical_admission["studioExecutionSpec"]),
            "archivedDefinitionRecordHash": record["recordHash"],
            "blockContractHash": historical_definition["blockContractHash"],
            "rootBlockDefinitionId": historical_definition["rootBlockDefinitionId"],
        },
        "destination": {
            "manifestDefinitionId": destination_source["manifestDefinitionId"],
            "libraryRevision": destination_source["libraryRevision"],
            "manifestContentHash": destination_source["manifestContentHash"],
            "executionAdmissionId": destination_source["executionAdmissionId"],
            "blockDefinitionId": destination_definition["definitionId"],
            "blockDefinitionContentHash": destination_definition["contentHash"],
            "blockDefinitionCanonicalSha256": block_definition_canonical_sha256_v2(
                destination_definition
            ),
            "executionGraphHash": destination_definition["graph"]["graphHash"],
            "interfaceHash": block_interface_hash_v2(destination_definition),
        },
        "review": {
            "decision": "historical_compiler_mapping_reviewed",
            "issuer": "workspace_owner:test-reviewer",
            "reviewedAt": "2026-09-02T18:00:00+05:30",
            "notes": "Fixture review of one exact archived body and exact compiler destination.",
        },
    }
    mapping["mappingHash"] = canonical_content_hash(mapping, omit="mappingHash")
    mapping = validate_historical_compiler_mapping(mapping)
    conversion["historicalCompilerMapping"] = {
        "mappingId": mapping["id"],
        "mappingHash": mapping["mappingHash"],
    }
    return mapping


def mapping_patches(mapping, pin):
    def by_reference(mapping_id, mapping_hash):
        if (mapping_id, mapping_hash) == (mapping["id"], mapping["mappingHash"]):
            return copy.deepcopy(mapping)
        return None

    return (
        patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ),
        patch(
            "modiff.composite_migration._historical_compiler_mapping_for_historical",
            return_value=copy.deepcopy(mapping),
        ),
        patch(
            "modiff.composite_migration._historical_compiler_mapping_by_reference",
            side_effect=by_reference,
        ),
    )


class CompositeMigrationArchivedMappingTests(unittest.TestCase):
    def _fixture(self):
        workflow = registered_cluster_workflow()
        source_preview = build_composite_migration_preview(
            {"user-workflows/mixed-workflow.json": workflow}, {}
        )
        source_sha = next(
            target["beforeSha256"]
            for target in source_preview["targets"]
            if target["sourcePath"] == "user-workflows/mixed-workflow.json"
        )
        supplement, pin = registered_cluster_compiler_fixture(workflow, source_sha)
        mapping = archived_mapping_fixture(workflow, supplement)
        # Changing the historical receipt changes the exact workflow bytes and
        # composite hash after the initial current-route fixture was compiled.
        refreshed = build_composite_migration_preview(
            {"user-workflows/mixed-workflow.json": workflow}, {}
        )
        candidate = next(item for item in refreshed["blocked"] if item["id"] == "cluster-root")
        supplement["compilerOutputs"][0]["sourceSha256"] = candidate["sourceSha256"]
        supplement["compilerOutputs"][0]["conversions"][0]["legacyCompositeHash"] = candidate[
            "legacyCompositeHash"
        ]
        return workflow, supplement, pin, mapping

    def test_preview_uses_only_exact_mapping_reference_and_rejects_tampering(self):
        workflow, supplement, pin, mapping = self._fixture()
        workflows = {"user-workflows/mixed-workflow.json": workflow}
        pin_patch, available_patch, reference_patch = mapping_patches(mapping, pin)
        with pin_patch, available_patch, reference_patch:
            preview = build_composite_migration_preview(
                workflows, {}, compiler_supplement=supplement
            )
        candidate = next(item for item in preview["candidates"] if item["id"] == "cluster-root")
        self.assertEqual(candidate["status"], "convertible")
        authority = candidate["compilerReceipt"]["historicalCompilerMappingAuthority"]
        self.assertEqual(authority["mappingHash"], mapping["mappingHash"])
        self.assertEqual(authority["historical"], mapping["historical"])
        self.assertNotIn("semanticEquivalenceAuthority", candidate["compilerReceipt"])

        tampered = copy.deepcopy(supplement)
        tampered["compilerOutputs"][0]["conversions"][0]["historicalCompilerMapping"][
            "mappingHash"
        ] = "sha256:" + "0" * 64
        pin_patch, available_patch, reference_patch = mapping_patches(mapping, pin)
        with pin_patch, available_patch, reference_patch:
            rejected = build_composite_migration_preview(
                workflows, {}, compiler_supplement=tampered
            )
        blocked = next(item for item in rejected["blocked"] if item["id"] == "cluster-root")
        self.assertIn("not present in the checked-in review ledger", blocked["reason"])

        conflicting = copy.deepcopy(supplement)
        conflicting["compilerOutputs"][0]["conversions"][0]["semanticEquivalenceReceipt"] = {
            "receiptId": "conflicting:receipt",
            "receiptHash": "sha256:" + "1" * 64,
        }
        with self.assertRaisesRegex(ValueError, "must not combine"):
            build_composite_migration_preview(
                workflows, {}, compiler_supplement=conflicting
            )

        graph_rewrite = copy.deepcopy(supplement)
        graph_rewrite["compilerOutputs"][0]["conversions"][0]["valueMappings"][0].update(
            {
                "targetKind": "graph_param",
                "targetNodeId": "generate",
                "targetFieldId": "steps",
            }
        )
        graph_rewrite["compilerOutputs"][0]["conversions"][0]["valueMappings"][0].pop(
            "targetValueId"
        )
        pin_patch, available_patch, reference_patch = mapping_patches(mapping, pin)
        with pin_patch, available_patch, reference_patch:
            rejected = build_composite_migration_preview(
                workflows, {}, compiler_supplement=graph_rewrite
            )
        blocked = next(item for item in rejected["blocked"] if item["id"] == "cluster-root")
        self.assertIn("only through BlockInstanceV2 values", blocked["reason"])

    def test_apply_refresh_and_rollback_preserve_exact_bytes(self):
        workflow, supplement, pin, mapping = self._fixture()
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            workflow_path = data_dir / "user-workflows" / "mixed-workflow.json"
            workflow_path.parent.mkdir(parents=True)
            workflow_path.write_text(
                json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            original = workflow_path.read_bytes()
            unrelated_path = data_dir / "user-workflows" / "unrelated.json"
            unrelated = {
                "id": "unrelated",
                "name": "Unrelated workflow",
                "snapshot": {"nodes": [], "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}},
            }
            unrelated_path.write_text(
                json.dumps(unrelated, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            unrelated_original = unrelated_path.read_bytes()
            # Rebind byte-level source receipt to the exact persisted encoding.
            default = scan_composite_migration_preview(str(data_dir))
            candidate = next(item for item in default["blocked"] if item["id"] == "cluster-root")
            supplement["compilerOutputs"][0]["sourceSha256"] = candidate["sourceSha256"]
            supplement["compilerOutputs"][0]["conversions"][0]["legacyCompositeHash"] = candidate[
                "legacyCompositeHash"
            ]
            pin_patch, available_patch, reference_patch = mapping_patches(mapping, pin)
            with pin_patch, available_patch, reference_patch:
                preview = scan_composite_migration_preview(
                    str(data_dir), compiler_supplement=supplement
                )
                with self.assertRaises(CompositeMigrationAuthorizationError):
                    apply_composite_migration(
                        str(data_dir),
                        migration_id=preview["migrationId"],
                        plan_hash=preview["planHash"],
                        confirmation="yes",
                        compiler_supplement=supplement,
                    )
                applied = apply_composite_migration(
                    str(data_dir),
                    migration_id=preview["migrationId"],
                    plan_hash=preview["planHash"],
                    confirmation=APPLY_CONFIRMATION,
                    compiler_supplement=supplement,
                )
            self.assertEqual(applied["status"]["effectiveState"], "applied")
            self.assertEqual(unrelated_path.read_bytes(), unrelated_original)
            refreshed = json.loads(workflow_path.read_text(encoding="utf-8"))
            migrated_root = next(
                node for node in refreshed["snapshot"]["nodes"] if node["id"] == "cluster-root"
            )
            self.assertEqual(migrated_root["type"], "block")
            migrated_instance = migrated_root["data"]["blockInstanceV2"]
            compiled_instance = supplement["compilerOutputs"][0]["conversions"][0][
                "blockInstanceV2"
            ]
            self.assertEqual(
                migrated_instance["definitionSnapshot"],
                compiled_instance["definitionSnapshot"],
            )
            self.assertEqual(
                migrated_instance["effectiveGraph"],
                compiled_instance["definitionSnapshot"]["graph"],
            )
            self.assertEqual(migrated_instance["values"], compiled_instance["values"])
            manifest_path = (
                data_dir
                / "studio"
                / "composite-migrations"
                / preview["migrationId"]
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["compilerReceipts"][0]["historicalCompilerMappingAuthority"][
                    "mappingHash"
                ],
                mapping["mappingHash"],
            )
            rolled_back = rollback_composite_migration(
                str(data_dir),
                migration_id=preview["migrationId"],
                confirmation=ROLLBACK_CONFIRMATION,
            )
            self.assertEqual(rolled_back["status"]["effectiveState"], "rolled_back")
            self.assertEqual(workflow_path.read_bytes(), original)
            self.assertEqual(unrelated_path.read_bytes(), unrelated_original)


if __name__ == "__main__":
    unittest.main()
