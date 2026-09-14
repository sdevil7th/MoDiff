import copy
import json
import unittest

from modiff.legacy_cluster_archived_definitions import (
    reviewed_archived_cluster_definition_evidence,
)
from modiff.legacy_cluster_compiler_mappings import (
    HISTORICAL_COMPILER_MAPPINGS_PATH,
    LegacyClusterCompilerMappingError,
    canonical_content_hash,
    validate_historical_compiler_mapping,
    validate_historical_compiler_mappings,
)


def compiler_mapping(record=None):
    record = record or next(
        item
        for item in reviewed_archived_cluster_definition_evidence()["records"]
        if item["registration"]["status"] == "recovery_audit_registered"
    )
    definition = record["definition"]
    admission = definition["executionAdmissions"][0]
    mapping = {
        "id": "legacy-cluster-compiler-mapping:fixture:reviewed",
        "historical": {
            "manifestDefinitionId": definition["id"],
            "libraryRevision": definition["libraryRevision"],
            "manifestContentHash": definition["contentHash"],
            "executionAdmissionId": admission["id"],
            "studioExecutionSpec": copy.deepcopy(admission["studioExecutionSpec"]),
            "archivedDefinitionRecordHash": record["recordHash"],
            "blockContractHash": definition["blockContractHash"],
            "rootBlockDefinitionId": definition["rootBlockDefinitionId"],
        },
        "destination": {
            "manifestDefinitionId": definition["id"],
            "libraryRevision": "a" * 40,
            "manifestContentHash": "sha256:" + "b" * 64,
            "executionAdmissionId": "diffusers.cluster-admission:fixture:current",
            "blockDefinitionId": "fixture:block:v2",
            "blockDefinitionContentHash": "block-definition-v2-1234abcd",
            "blockDefinitionCanonicalSha256": "sha256:" + "c" * 64,
            "executionGraphHash": "block-graph-v2-1234abcd",
            "interfaceHash": "block-interface-v2-1234abcd",
        },
        "review": {
            "decision": "historical_compiler_mapping_reviewed",
            "issuer": "workspace_owner:fixture-reviewer",
            "reviewedAt": "2026-09-02T18:00:00+05:30",
            "notes": "Reviewed the exact archived manifest and exact compiler destination pins.",
        },
    }
    mapping["mappingHash"] = canonical_content_hash(mapping, omit="mappingHash")
    return mapping


def mapping_ledger(*mappings):
    value = {
        "schemaVersion": 1,
        "kind": "legacy_cluster_compiler_mappings",
        "boundary": {
            "checkedInCompilerMappingAuthority": True,
            "requiresExactArchivedDefinition": True,
            "doesNotAssertSemanticEquivalence": True,
            "doesNotInferFromLabelsOrModels": True,
            "requiresPerInstancePreservationReceipts": True,
            "doesNotAuthorizeWorkflowMutation": True,
        },
        "mappings": list(mappings),
    }
    value["contentHash"] = canonical_content_hash(value)
    return value


class LegacyClusterCompilerMappingTests(unittest.TestCase):
    def test_mapping_binds_exact_archive_record_admission_and_destination(self):
        mapping = compiler_mapping()
        self.assertEqual(validate_historical_compiler_mapping(mapping), mapping)
        self.assertEqual(mapping["historical"]["archivedDefinitionRecordHash"][:7], "sha256:")
        self.assertEqual(mapping["review"]["decision"], "historical_compiler_mapping_reviewed")

    def test_source_destination_and_hash_tampering_fail_closed(self):
        for path, value, message in (
            (("historical", "blockContractHash"), "sha256:" + "0" * 64, "exact archived"),
            (
                ("historical", "studioExecutionSpec", "contentHash"),
                "studio-spec-v1-deadbeef",
                "exact archived",
            ),
            (
                ("destination", "blockDefinitionContentHash"),
                "block-definition-v2-deadbeef",
                "mapping hash",
            ),
        ):
            mapping = compiler_mapping()
            target = mapping
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            if path[0] == "historical":
                mapping["mappingHash"] = canonical_content_hash(mapping, omit="mappingHash")
            with self.assertRaisesRegex(LegacyClusterCompilerMappingError, message):
                validate_historical_compiler_mapping(mapping)

    def test_body_only_hunyuan_cannot_be_registered(self):
        record = next(
            item
            for item in reviewed_archived_cluster_definition_evidence()["records"]
            if item["registration"]["status"] == "archived_body_only"
        )
        definition = record["definition"]
        mapping = compiler_mapping()
        mapping["historical"].update(
            {
                "manifestDefinitionId": definition["id"],
                "libraryRevision": definition["libraryRevision"],
                "manifestContentHash": definition["contentHash"],
                "executionAdmissionId": "missing:historical:admission",
                "studioExecutionSpec": {
                    "id": "missing:historical:spec",
                    "contentHash": "studio-spec-v1-deadbeef",
                    "executionProfileId": "missing:historical:profile",
                },
                "archivedDefinitionRecordHash": record["recordHash"],
                "blockContractHash": definition["blockContractHash"],
                "rootBlockDefinitionId": definition["rootBlockDefinitionId"],
            }
        )
        mapping["mappingHash"] = canonical_content_hash(mapping, omit="mappingHash")
        with self.assertRaisesRegex(LegacyClusterCompilerMappingError, "registered exact archived"):
            validate_historical_compiler_mapping(mapping)

    def test_ledger_rejects_duplicate_authority_and_checked_in_ledger_has_only_exact_compiler_proofs(self):
        mapping = compiler_mapping()
        duplicate = copy.deepcopy(mapping)
        duplicate["id"] = "legacy-cluster-compiler-mapping:fixture:duplicate"
        duplicate["mappingHash"] = canonical_content_hash(duplicate, omit="mappingHash")
        mappings = sorted((mapping, duplicate), key=lambda item: item["id"])
        with self.assertRaisesRegex(LegacyClusterCompilerMappingError, "duplicate authority"):
            validate_historical_compiler_mappings(mapping_ledger(*mappings))

        checked_in = json.loads(HISTORICAL_COMPILER_MAPPINGS_PATH.read_text(encoding="utf-8"))
        validated = validate_historical_compiler_mappings(checked_in)
        self.assertEqual(
            validated["contentHash"],
            "sha256:40c74120e05409e98b2eaeb8bc6bd05e91bd336712a9ee2bacc49ba7fa601975",
        )
        self.assertEqual(
            [item["id"] for item in validated["mappings"]],
            [
                "legacy-cluster-compiler-mapping:minimax-music3:2026-09-02",
                "legacy-cluster-compiler-mapping:transformers-ctc-stt:2026-09-02",
                "legacy-cluster-compiler-mapping:wan-22-ti2v-5b:2026-09-02",
            ],
        )
        by_id = {item["id"]: item for item in validated["mappings"]}
        minimax = by_id["legacy-cluster-compiler-mapping:minimax-music3:2026-09-02"]
        self.assertEqual(
            minimax["destination"],
            {
                "manifestDefinitionId": "diffusers.modular:MiniMaxMusic3ModularPipeline:default",
                "libraryRevision": "2f7e0154a9db246e95c9ede43edba7db5b130805",
                "manifestContentHash": (
                    "sha256:b427e0863ca2fbbeccfd9acef02e5110daecd0bf84e7a8a02cf8c40484f207f7"
                ),
                "executionAdmissionId": (
                    "diffusers.cluster-admission:MiniMaxMusic3ModularPipeline:default:"
                    "workflow:official_top_level_blocks"
                ),
                "blockDefinitionId": (
                    "diffusers.cluster-admission:MiniMaxMusic3ModularPipeline:default:"
                    "workflow:official_top_level_blocks"
                ),
                "blockDefinitionContentHash": "block-definition-v2-5f610dae",
                "blockDefinitionCanonicalSha256": (
                    "sha256:b5f8b4618df9680126056c82203e6f867529f572e114ee874761f5ca689d2e7d"
                ),
                "executionGraphHash": "block-graph-v2-f35deb93",
                "interfaceHash": "block-interface-v2-35294a7f",
            },
        )
        self.assertEqual(
            minimax["mappingHash"],
            "sha256:87e92312604aba3495c6bd70ced9c68006ce96bcd8fa4aedd96dfdb46f591a3b",
        )
        self.assertEqual(
            by_id["legacy-cluster-compiler-mapping:transformers-ctc-stt:2026-09-02"][
                "mappingHash"
            ],
            "sha256:3d049153434a808b94c83752b7c729e8b1cfe7acdfdeb605b08ac6f9e408aec3",
        )
        self.assertEqual(
            by_id["legacy-cluster-compiler-mapping:wan-22-ti2v-5b:2026-09-02"][
                "mappingHash"
            ],
            "sha256:12d4d8eae96d868c28f1498f60662371db71d096ca25434b0858b83767bdcb24",
        )


if __name__ == "__main__":
    unittest.main()
