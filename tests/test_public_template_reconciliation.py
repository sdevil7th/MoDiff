import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from modiff.public_template_reconciliation import (
    BUCKET_CURRENT,
    BUCKET_NEVER,
    BUCKET_STALE,
    PublicTemplateReconciliationError,
    build_public_template_reconciliation,
    classify_public_template,
    write_public_template_reconciliation,
)


ROOT = Path(__file__).resolve().parents[1]


def _template(
    template_id,
    *,
    workflow="FluxPipeline:text_to_image",
    historical=None,
    last_run=None,
    missing=None,
    eligible=False,
):
    return {
        "id": template_id,
        "label": template_id.replace("_", " "),
        "canonicalWorkflowId": workflow,
        "modelType": "FluxPipeline",
        "mode": "text_to_image",
        "historicalEvidence": historical,
        "lastSuccessfulRealRun": last_run,
        "missingReleaseEvidence": missing or [],
        "releaseEligible": eligible,
        "reviewedExampleQualityReviewStatus": None if historical is None else "approved_reviewed",
        "reviewedExampleVerificationStatus": None if historical is None else "reviewed",
    }


class PublicTemplateReconciliationTests(unittest.TestCase):
    def test_classifies_current_stale_and_never_had_example(self):
        current = classify_public_template(
            _template(
                "keep_me",
                historical={"status": "legacy_node_contract_match", "reasons": [], "missingBackendNodes": []},
                last_run={"capturedAt": "2026-07-28T00:00:00Z", "verificationStatus": "reviewed"},
            )
        )
        stale = classify_public_template(
            _template(
                "stale_me",
                historical={
                    "status": "stale",
                    "reasons": ["canonical_workflow_node_contract_mismatch"],
                    "missingBackendNodes": [],
                },
            )
        )
        missing = classify_public_template(_template("never_me", historical=None, missing=["approved_gallery_example"]))
        self.assertEqual(current["bucket"], BUCKET_CURRENT)
        self.assertEqual(current["action"], "keep_historical_example")
        self.assertEqual(stale["bucket"], BUCKET_STALE)
        self.assertEqual(stale["action"], "plan_canary_rerun_on_qualification_host")
        self.assertEqual(missing["bucket"], BUCKET_NEVER)
        self.assertEqual(missing["action"], "plan_first_example_canary_on_qualification_host")
        self.assertTrue(current["doNotRerunHere"])

    def test_current_without_a_real_run_is_rejected(self):
        with self.assertRaisesRegex(PublicTemplateReconciliationError, "without lastSuccessfulRealRun"):
            classify_public_template(
                _template("broken", historical={"status": "legacy_node_contract_match", "reasons": []})
            )

    def test_ledger_counts_and_refuses_to_rerun(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {
                "coverage": {
                    "currentHistoricalEvidenceCount": 1,
                    "historicalEvidenceCount": 2,
                },
                "templates": [
                    _template(
                        "keep_me",
                        historical={
                            "status": "legacy_node_contract_match",
                            "reasons": [],
                            "missingBackendNodes": [],
                        },
                        last_run={"capturedAt": "2026-07-28T00:00:00Z", "verificationStatus": "reviewed"},
                    ),
                    _template(
                        "stale_me",
                        workflow="AceStepAudioPipeline:audio_repaint",
                        historical={
                            "status": "stale",
                            "reasons": ["canonical_workflow_node_contract_mismatch"],
                            "missingBackendNodes": [],
                        },
                    ),
                    _template("never_me", workflow="AceStepAudioPipeline:text_to_audio", historical=None),
                ],
            }
            ledger = build_public_template_reconciliation(root, release_contract=contract)
            self.assertEqual(ledger["kind"], "public_template_reconciliation")
            self.assertTrue(ledger["boundary"]["blanketRerunForbidden"])
            self.assertFalse(ledger["boundary"]["graphsSubmitted"])
            self.assertEqual(
                ledger["summary"],
                {
                    "templateCount": 3,
                    "currentKeepCount": 1,
                    "staleCanaryCount": 1,
                    "neverHadExampleCount": 1,
                    "staleReasonCounts": {"canonical_workflow_node_contract_mismatch": 1},
                    "localTechnicalReceiptOverlapCount": 0,
                },
            )

    def test_write_emits_json_and_html_without_submitting_graphs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            contract = {
                "coverage": {"currentHistoricalEvidenceCount": 0, "historicalEvidenceCount": 0},
                "templates": [_template("never_me", historical=None)],
            }
            (root / "data" / "release-contract.v1.json").write_text(json.dumps(contract), encoding="utf-8")
            ledger = write_public_template_reconciliation(root)
            pending = root / "review-pending" / "public-templates"
            self.assertTrue((pending / "reconciliation.v1.json").is_file())
            html = (pending / "index.html").read_text(encoding="utf-8")
            self.assertIn("Never had an example", html)
            self.assertIn("never_me", html)
            self.assertEqual(ledger["summary"]["neverHadExampleCount"], 1)

    def test_overlap_with_local_receipts_is_fail_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "data" / "qualification" / "local-review"
            receipts.mkdir(parents=True)
            (receipts / "technical-candidates.v1.json").write_text(
                json.dumps({"receipts": [{"workflowId": "FluxPipeline:text_to_image"}]}),
                encoding="utf-8",
            )
            contract = {
                "coverage": {"currentHistoricalEvidenceCount": 0, "historicalEvidenceCount": 0},
                "templates": [_template("never_me", historical=None)],
            }
            with self.assertRaisesRegex(PublicTemplateReconciliationError, "overlap public templates"):
                build_public_template_reconciliation(root, release_contract=contract)


class PublicTemplateReconciliationLiveLedgerTests(unittest.TestCase):
    def test_checked_in_release_contract_is_26_keep_44_stale_7_never(self):
        path = ROOT / "data" / "release-contract.v1.json"
        if not path.is_file():
            self.skipTest("Local release-contract ledger is not present.")
        ledger = build_public_template_reconciliation(ROOT)
        self.assertEqual(ledger["summary"]["templateCount"], 77)
        self.assertEqual(ledger["summary"]["currentKeepCount"], 26)
        self.assertEqual(ledger["summary"]["staleCanaryCount"], 44)
        self.assertEqual(ledger["summary"]["neverHadExampleCount"], 7)
        self.assertEqual(ledger["summary"]["localTechnicalReceiptOverlapCount"], 0)
        self.assertTrue(ledger["policy"]["blanketRerunForbidden"])
        keep = {item["templateId"] for item in ledger["items"] if item["bucket"] == BUCKET_CURRENT}
        never = {item["templateId"] for item in ledger["items"] if item["bucket"] == BUCKET_NEVER}
        self.assertIn("fast_lora", keep)
        self.assertIn("ace_step_custom_lora", never)
        self.assertIn("wan_video_long_showcase", never)


if __name__ == "__main__":
    unittest.main()
