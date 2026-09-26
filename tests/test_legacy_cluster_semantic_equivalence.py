import copy
import json
import unittest

from modiff.legacy_cluster_semantic_equivalence import (
    SEMANTIC_EQUIVALENCE_RECEIPTS_PATH,
    canonical_content_hash,
    validate_semantic_equivalence_receipt,
    validate_semantic_equivalence_receipts,
)


def semantic_equivalence_receipt():
    receipt = {
        "id": "legacy-cluster-equivalence:qwen-image:text2image:2026-09-02",
        "historical": {
            "manifestDefinitionId": "diffusers.modular:QwenImageModularPipeline:text2image",
            "libraryRevision": "1" * 40,
            "manifestContentHash": "sha256:" + "2" * 64,
            "executionAdmissionId": "diffusers.cluster-admission:qwen:historical",
            "studioExecutionSpec": {
                "id": "qwen:historical:v1",
                "contentHash": "studio-spec-v1-12345678",
                "executionProfileId": "qwen:historical",
            },
            "executionGraphHash": "sha256:" + "3" * 64,
            "interfaceHash": "sha256:" + "4" * 64,
        },
        "destination": {
            "manifestDefinitionId": "diffusers.modular:QwenImageModularPipeline:text2image",
            "libraryRevision": "a" * 40,
            "manifestContentHash": "sha256:" + "b" * 64,
            "executionAdmissionId": "diffusers.cluster-admission:qwen:current",
            "blockDefinitionId": "qwen:text-to-image:v2",
            "blockDefinitionContentHash": "block-definition-v2-1234abcd",
            "blockDefinitionCanonicalSha256": "sha256:" + "5" * 64,
            "executionGraphHash": "block-graph-v2-1234abcd",
            "interfaceHash": "block-interface-v2-1234abcd",
        },
        "review": {
            "decision": "semantic_equivalent",
            "issuer": "workspace_owner:fixture-reviewer",
            "reviewedAt": "2026-09-02T12:00:00+05:30",
            "notes": "Reviewed every historical block, port, persisted value, and execution edge.",
        },
    }
    receipt["receiptHash"] = canonical_content_hash(receipt, omit="receiptHash")
    return receipt


def ledger(*receipts):
    value = {
        "schemaVersion": 1,
        "kind": "legacy_cluster_semantic_equivalence_receipts",
        "boundary": {
            "checkedInReviewAuthority": True,
            "doesNotInferFromLabelsOrModels": True,
            "requiresExactCompilerMapping": True,
            "doesNotAuthorizeWorkflowMutation": True,
        },
        "receipts": list(receipts),
    }
    value["contentHash"] = canonical_content_hash(value)
    return value


class LegacyClusterSemanticEquivalenceTests(unittest.TestCase):
    def test_strict_receipt_pins_both_contracts_and_review_metadata(self):
        receipt = semantic_equivalence_receipt()
        parsed = validate_semantic_equivalence_receipt(receipt)

        self.assertEqual(parsed, receipt)
        self.assertEqual(parsed["historical"]["executionGraphHash"], "sha256:" + "3" * 64)
        self.assertEqual(parsed["destination"]["executionGraphHash"], "block-graph-v2-1234abcd")
        self.assertEqual(parsed["destination"]["interfaceHash"], "block-interface-v2-1234abcd")
        self.assertEqual(parsed["review"]["issuer"], "workspace_owner:fixture-reviewer")

        tampered = copy.deepcopy(receipt)
        tampered["destination"]["blockDefinitionContentHash"] = "block-definition-v2-deadbeef"
        with self.assertRaisesRegex(ValueError, "receipt hash"):
            validate_semantic_equivalence_receipt(tampered)

    def test_review_timestamp_requires_offset_and_unknown_fields_fail_closed(self):
        no_offset = semantic_equivalence_receipt()
        no_offset["review"]["reviewedAt"] = "2026-09-02T12:00:00"
        no_offset["receiptHash"] = canonical_content_hash(no_offset, omit="receiptHash")
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            validate_semantic_equivalence_receipt(no_offset)

        unknown = semantic_equivalence_receipt()
        unknown["approvalShortcut"] = True
        with self.assertRaisesRegex(ValueError, "missing or unknown"):
            validate_semantic_equivalence_receipt(unknown)

    def test_ledger_rejects_duplicate_historical_authority_and_body_tampering(self):
        receipt = semantic_equivalence_receipt()
        duplicate = copy.deepcopy(receipt)
        duplicate["id"] = "legacy-cluster-equivalence:qwen-image:text2image:duplicate"
        duplicate["receiptHash"] = canonical_content_hash(duplicate, omit="receiptHash")
        value = ledger(receipt, duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate authority"):
            validate_semantic_equivalence_receipts(value)

        value = ledger(receipt)
        value["boundary"]["doesNotAuthorizeWorkflowMutation"] = False
        with self.assertRaisesRegex(ValueError, "boundary"):
            validate_semantic_equivalence_receipts(value)

    def test_checked_in_ledger_is_valid_and_contains_no_fabricated_receipts(self):
        value = json.loads(SEMANTIC_EQUIVALENCE_RECEIPTS_PATH.read_text(encoding="utf-8"))
        parsed = validate_semantic_equivalence_receipts(value)
        self.assertEqual(parsed["receipts"], [])


if __name__ == "__main__":
    unittest.main()
