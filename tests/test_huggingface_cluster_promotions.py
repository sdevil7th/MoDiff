import copy
import hashlib
import unittest
from unittest.mock import patch

from modiff.huggingface_cluster_promotions import (
    ClusterPromotionReceiptError,
    build_cluster_promotion_ledger_candidate,
    build_cluster_promotion_receipt_candidate,
    build_post_promotion_node_library_candidate,
    canonical_content_hash,
    promotion_receipt_for_admission,
    reviewed_cluster_promotion_receipts,
    validate_cluster_promotion_receipts,
)
from modiff.huggingface_cluster_runtime import REGISTERED_BLOCK_V2_DEFINITION_PINS
from modiff.huggingface_node_library import reviewed_huggingface_node_library


QWEN_ADMISSION_ID = (
    "diffusers.cluster-admission:QwenImageModularPipeline:text2image:mode:text_to_image"
)
MINIMAX_ADMISSION_ID = (
    "diffusers.cluster-admission:MiniMaxMusic3ModularPipeline:default:"
    "workflow:official_top_level_blocks"
)


class HuggingFaceClusterPromotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = reviewed_huggingface_node_library()
        cls.admissions = {
            admission["id"]: admission
            for definition in cls.library["definitions"]
            for admission in definition.get("executionAdmissions", ())
        }

    def current_receipt_ledger(self, admission_id):
        ledger = reviewed_cluster_promotion_receipts()
        admission = self.admissions[admission_id]
        historical = next(
            record["receipt"]
            for record in ledger["historicalReceipts"]
            if record["receipt"]["admissionId"] == admission_id
        )
        receipt = copy.deepcopy(historical)
        receipt["id"] = f"{receipt['id']}:schema-v2-test"
        receipt["evidence"]["taskId"] = f"schema-v2-{receipt['evidence']['taskId']}"
        pin = REGISTERED_BLOCK_V2_DEFINITION_PINS[admission_id]
        receipt["blockDefinition"] = {
            "definitionId": admission_id,
            "contentHash": pin[0],
            "canonicalSha256": pin[1],
        }
        receipt["definitionId"] = admission["definitionId"]
        receipt["artifact"] = {
            "repository": admission["artifact"]["repo"],
            "revision": admission["artifact"]["revision"],
        }
        receipt["studioExecutionSpec"] = copy.deepcopy(admission["studioExecutionSpec"])
        ledger["receipts"] = [receipt]
        ledger["contentHash"] = canonical_content_hash(ledger)
        return ledger, admission

    def current_frontend_provenance(self, admission_id, media_bytes=b"reviewed-output"):
        admission = self.admissions[admission_id]
        pin = REGISTERED_BLOCK_V2_DEFINITION_PINS[admission_id]
        media_sha256 = hashlib.sha256(media_bytes).hexdigest()
        return {
            "schemaVersion": 2,
            "format": "modiff.live-proof.provenance.v2",
            "taskId": "frontend-proof-task-1",
            "backendReportedRuntimeFingerprint": "sha256:" + "1" * 64,
            "routeBinding": {
                "schemaVersion": 1,
                "admissionId": admission_id,
                "blockDefinition": {
                    "definitionId": admission_id,
                    "contentHash": pin[0],
                    "canonicalSha256": pin[1],
                },
                "studioExecutionSpec": copy.deepcopy(admission["studioExecutionSpec"]),
                "artifact": {
                    "repository": admission["artifact"]["repo"],
                    "revision": admission["artifact"]["revision"],
                },
                "modelDependencies": [],
            },
            "output": {
                "items": [
                    {
                        "index": 0,
                        "mediaType": "image",
                        "byteSize": len(media_bytes),
                        "encodedSha256": f"sha256:encoded:{media_sha256}",
                    }
                ]
            },
            "blockers": [],
        }

    @staticmethod
    def lifecycle_confirmations():
        return {
            "insertedThroughFrontend": True,
            "expandedOfficialBlocks": True,
            "editedParameters": True,
            "savedAndRefreshed": True,
            "valuesRestoredExactly": True,
            "collapsedExpandedGraphEquivalent": True,
            "generatedThroughFrontend": True,
        }

    @staticmethod
    def qwen_post_promotion_identity():
        return {
            "post_promotion_block_definition": {
                "definitionId": QWEN_ADMISSION_ID,
                "contentHash": "block-definition-v2-74cfc64c",
                "canonicalSha256": (
                    "sha256:c0661cfc7442e5abb25c960b16773166d716fb9144ea9e99ca11939ebb08bef7"
                ),
            },
            "post_promotion_manifest_content_hash": (
                "sha256:98dddb4c03eea94a0a8e9b1d318ecf5aab9fe62113b829263d8d605772ed6033"
            ),
        }

    def test_checked_in_v1_approvals_are_preserved_but_non_authorizing(self):
        ledger = reviewed_cluster_promotion_receipts()

        self.assertEqual(ledger["schemaVersion"], 2)
        self.assertEqual(ledger["contentHash"], canonical_content_hash(ledger))
        self.assertEqual(ledger["receipts"], [])
        self.assertEqual(
            [record["receipt"]["admissionId"] for record in ledger["historicalReceipts"]],
            [MINIMAX_ADMISSION_ID, QWEN_ADMISSION_ID],
        )
        self.assertTrue(
            all(
                record["nonAuthorizingReason"]
                == "legacy_schema_v1_missing_exact_block_definition_identity"
                for record in ledger["historicalReceipts"]
            )
        )

    def test_historical_approvals_do_not_publish_live_proof(self):
        qwen = self.admissions[QWEN_ADMISSION_ID]
        minimax = self.admissions[MINIMAX_ADMISSION_ID]

        for admission in (qwen, minimax):
            with self.subTest(admission=admission["id"]):
                self.assertFalse(admission["publication"]["liveProof"])
                self.assertFalse(admission["publication"]["executable"])
                self.assertFalse(admission["publication"]["autoEligible"])
                self.assertIn(
                    "live_output_review_pending",
                    {reason["code"] for reason in admission["publication"]["reasons"]},
                )

        other_qwen = [
            admission
            for admission_id, admission in self.admissions.items()
            if admission_id.startswith("diffusers.cluster-admission:QwenImageModularPipeline:")
            and admission_id != QWEN_ADMISSION_ID
        ]
        self.assertTrue(other_qwen)
        self.assertTrue(all(not admission["publication"]["liveProof"] for admission in other_qwen))
        self.assertTrue(
            all(
                "live_output_review_pending"
                in {reason["code"] for reason in admission["publication"]["reasons"]}
                for admission in other_qwen
            )
        )

    def test_schema_v2_receipt_requires_current_block_definition_identity(self):
        ledger, admission = self.current_receipt_ledger(QWEN_ADMISSION_ID)
        validate_cluster_promotion_receipts(ledger)

        with patch(
            "modiff.huggingface_cluster_promotions._reviewed_cluster_promotion_receipts",
            return_value=ledger,
        ):
            receipt = promotion_receipt_for_admission(
                QWEN_ADMISSION_ID,
                definition_id=admission["definitionId"],
                artifact=admission["artifact"],
                studio_execution_spec=admission["studioExecutionSpec"],
            )
        self.assertEqual(
            receipt["blockDefinition"]["contentHash"],
            REGISTERED_BLOCK_V2_DEFINITION_PINS[QWEN_ADMISSION_ID][0],
        )

        stale = copy.deepcopy(ledger)
        stale["receipts"][0]["blockDefinition"]["contentHash"] = "block-definition-v2-00000000"
        stale["contentHash"] = canonical_content_hash(stale)
        with patch(
            "modiff.huggingface_cluster_promotions._reviewed_cluster_promotion_receipts",
            return_value=stale,
        ), self.assertRaisesRegex(ClusterPromotionReceiptError, "stale"):
            promotion_receipt_for_admission(
                QWEN_ADMISSION_ID,
                definition_id=admission["definitionId"],
                artifact=admission["artifact"],
                studio_execution_spec=admission["studioExecutionSpec"],
            )

    def test_stale_artifact_or_execution_binding_fails_closed(self):
        ledger, admission = self.current_receipt_ledger(QWEN_ADMISSION_ID)
        with self.assertRaisesRegex(ClusterPromotionReceiptError, "stale"):
            with patch(
                "modiff.huggingface_cluster_promotions._reviewed_cluster_promotion_receipts",
                return_value=ledger,
            ):
                promotion_receipt_for_admission(
                    QWEN_ADMISSION_ID,
                    definition_id=admission["definitionId"],
                    artifact={**admission["artifact"], "revision": "0" * 40},
                    studio_execution_spec=admission["studioExecutionSpec"],
                )

    def test_receipt_cannot_promote_runtime_or_auto_eligibility(self):
        ledger, _admission = self.current_receipt_ledger(QWEN_ADMISSION_ID)
        mutated = copy.deepcopy(ledger)
        mutated["receipts"][0]["publication"]["executable"] = True
        mutated["contentHash"] = canonical_content_hash(mutated)

        with self.assertRaisesRegex(ClusterPromotionReceiptError, "runtime publication boundary"):
            validate_cluster_promotion_receipts(mutated)

    def test_explicit_generator_binds_current_frontend_route_and_media(self):
        media = b"reviewed-output"
        receipt = build_cluster_promotion_receipt_candidate(
            admission_id=QWEN_ADMISSION_ID,
            provenance=self.current_frontend_provenance(QWEN_ADMISSION_ID, media),
            media_sha256=hashlib.sha256(media).hexdigest(),
            media_byte_size=len(media),
            receipt_id="cluster-promotion:qwen-image-2512:text2image:fixture-v2",
            reviewed_at="2026-09-02T20:00:00+05:30",
            review_comment="Approved this exact visible frontend output.",
            lifecycle_confirmations=self.lifecycle_confirmations(),
            workspace_owner_approved=True,
            **self.qwen_post_promotion_identity(),
        )
        ledger = build_cluster_promotion_ledger_candidate(receipt)
        repeated = build_cluster_promotion_receipt_candidate(
            admission_id=QWEN_ADMISSION_ID,
            provenance=self.current_frontend_provenance(QWEN_ADMISSION_ID, media),
            media_sha256=hashlib.sha256(media).hexdigest(),
            media_byte_size=len(media),
            receipt_id="cluster-promotion:qwen-image-2512:text2image:fixture-v2",
            reviewed_at="2026-09-02T20:00:00+05:30",
            review_comment="Approved this exact visible frontend output.",
            lifecycle_confirmations=self.lifecycle_confirmations(),
            workspace_owner_approved=True,
            **self.qwen_post_promotion_identity(),
        )

        self.assertEqual(repeated, receipt)
        self.assertEqual(receipt["blockDefinition"]["contentHash"], "block-definition-v2-74cfc64c")
        self.assertEqual(receipt["evidence"]["mediaSha256"], hashlib.sha256(media).hexdigest())
        self.assertEqual(ledger["receipts"], [receipt])
        self.assertEqual(ledger["contentHash"], canonical_content_hash(ledger))

    def test_generator_never_infers_human_or_lifecycle_approval(self):
        media = b"reviewed-output"
        arguments = {
            "admission_id": QWEN_ADMISSION_ID,
            "provenance": self.current_frontend_provenance(QWEN_ADMISSION_ID, media),
            "media_sha256": hashlib.sha256(media).hexdigest(),
            "media_byte_size": len(media),
            "receipt_id": "cluster-promotion:qwen-image-2512:text2image:fixture-v2",
            "reviewed_at": "2026-09-02T20:00:00+05:30",
            "review_comment": "Approved this exact visible frontend output.",
            "lifecycle_confirmations": self.lifecycle_confirmations(),
            "workspace_owner_approved": False,
            **self.qwen_post_promotion_identity(),
        }
        with self.assertRaisesRegex(ClusterPromotionReceiptError, "workspace-owner approval"):
            build_cluster_promotion_receipt_candidate(**arguments)

        arguments["workspace_owner_approved"] = True
        arguments["lifecycle_confirmations"] = {
            **self.lifecycle_confirmations(),
            "savedAndRefreshed": False,
        }
        with self.assertRaisesRegex(ClusterPromotionReceiptError, "every lifecycle confirmation"):
            build_cluster_promotion_receipt_candidate(**arguments)

    def test_generator_rejects_stale_route_or_media_bytes(self):
        media = b"reviewed-output"
        provenance = self.current_frontend_provenance(QWEN_ADMISSION_ID, media)
        provenance["routeBinding"]["blockDefinition"]["contentHash"] = (
            "block-definition-v2-00000000"
        )
        arguments = {
            "admission_id": QWEN_ADMISSION_ID,
            "provenance": provenance,
            "media_sha256": hashlib.sha256(media).hexdigest(),
            "media_byte_size": len(media),
            "receipt_id": "cluster-promotion:qwen-image-2512:text2image:fixture-v2",
            "reviewed_at": "2026-09-02T20:00:00+05:30",
            "review_comment": "Approved this exact visible frontend output.",
            "lifecycle_confirmations": self.lifecycle_confirmations(),
            "workspace_owner_approved": True,
            **self.qwen_post_promotion_identity(),
        }
        with self.assertRaisesRegex(ClusterPromotionReceiptError, "stale"):
            build_cluster_promotion_receipt_candidate(**arguments)

        arguments["provenance"] = self.current_frontend_provenance(QWEN_ADMISSION_ID, media)
        arguments["media_sha256"] = "0" * 64
        with self.assertRaisesRegex(ClusterPromotionReceiptError, "media bytes"):
            build_cluster_promotion_receipt_candidate(**arguments)

    def test_post_promotion_projection_is_exact_and_non_mutating(self):
        current = reviewed_huggingface_node_library()
        candidate = build_post_promotion_node_library_candidate(QWEN_ADMISSION_ID)

        def route(library):
            return next(
                (definition, admission)
                for definition in library["definitions"]
                for admission in definition.get("executionAdmissions", ())
                if admission["id"] == QWEN_ADMISSION_ID
            )

        current_definition, current_admission = route(current)
        candidate_definition, candidate_admission = route(candidate)
        self.assertFalse(current_admission["publication"]["liveProof"])
        self.assertEqual(
            current_definition["contentHash"],
            "sha256:9cbb38204acb409888b94e880a6e343e9bd67ea5703455560bc5c6d713437f68",
        )
        self.assertTrue(candidate_admission["publication"]["liveProof"])
        self.assertEqual(
            candidate_definition["contentHash"],
            "sha256:98dddb4c03eea94a0a8e9b1d318ecf5aab9fe62113b829263d8d605772ed6033",
        )
        self.assertEqual(reviewed_cluster_promotion_receipts()["receipts"], [])


if __name__ == "__main__":
    unittest.main()
