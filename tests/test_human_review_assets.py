import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from modiff.human_review_assets import (
    HumanReviewAssetError,
    approve_review_asset,
    reject_review_asset,
    stage_review_assets,
)
from modiff.local_review_receipts import build_local_review_ledger, parse_record


class HumanReviewAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        data = self.root / "data"
        (data / "images").mkdir(parents=True)
        image = Image.new("RGB", (8, 8), (12, 64, 128))
        image.save(data / "images" / "candidate.png")
        contracts = {
            "contracts": [
                {
                    "canonicalWorkflowId": "BuiltinImageOperation:image_crop",
                    "contractId": "template-candidate:BuiltinImageOperation:image_crop",
                    "graphHash": "a" * 64,
                    "graphPath": "studio/builtin-image-operation/image-crop.json",
                    "mediaKind": "image",
                }
            ]
        }
        (data / "template-candidate-contracts.v1.json").write_text(json.dumps(contracts), encoding="utf-8")
        self.ledger_path = Path("data/qualification/local-review/fixture.json")
        (self.root / self.ledger_path).parent.mkdir(parents=True, exist_ok=True)
        document = build_local_review_ledger(
            root=self.root,
            records=[parse_record("BuiltinImageOperation:image_crop|task-image|images/candidate.png")],
            fetch_run=self._run,
        )
        (self.root / self.ledger_path).write_text(json.dumps(document), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _run(task_id):
        return {
            "task": {
                "task_id": task_id,
                "status": "completed",
                "sid": f"sid-{task_id}",
                "client_run_id": f"run-{task_id}",
                "completed_at": 123.5,
                "workflow_title": "Review",
                "runtimeMeasurement": {"elapsedSeconds": 0.25, "processRssBytes": 1024},
                "runtimeFingerprint": {
                    "fingerprint": "sha256:" + ("c" * 64),
                    "resourceFingerprint": "sha256:" + ("d" * 64),
                    "packages": {"python": "3.12", "torch": "2.8.0"},
                },
            }
        }

    def test_stage_copies_bytes_without_registering_gallery(self):
        document = stage_review_assets(self.root, ledger_paths=(self.ledger_path,))
        self.assertEqual(document["summary"]["pendingCount"], 1)
        self.assertEqual(document["summary"]["builtinOperationCount"], 1)
        self.assertEqual(document["summary"]["modelGenerationCount"], 0)
        self.assertFalse(document["boundary"]["galleryRegistered"])
        self.assertFalse(document["boundary"]["candidateLedgerMutated"])
        staged = self.root / "review-pending" / "BuiltinImageOperation__image_crop" / "candidate.png"
        self.assertTrue(staged.is_file())
        self.assertTrue((self.root / "review-pending" / "index.html").is_file())
        item = document["items"][0]
        self.assertEqual(item["reviewState"], "pending_human_review")
        self.assertEqual(item["reviewKind"], "builtin_operation")
        self.assertFalse(item["galleryRegistered"])
        html = (self.root / "review-pending" / "index.html").read_text(encoding="utf-8")
        self.assertIn("Builtin crop / stitch / trim canaries", html)
        self.assertIn("Diffusers / Transformers graph outputs", html)

    def test_approve_requires_quality_and_rights_and_does_not_publish(self):
        stage_review_assets(self.root, ledger_paths=(self.ledger_path,))
        with self.assertRaisesRegex(HumanReviewAssetError, "both --approve-quality and --approve-rights"):
            approve_review_asset(
                self.root,
                workflow_id="BuiltinImageOperation:image_crop",
                reviewer="Sourav",
                approve_quality=True,
                approve_rights=False,
            )
        binding = approve_review_asset(
            self.root,
            workflow_id="BuiltinImageOperation:image_crop",
            reviewer="Sourav",
            approve_quality=True,
            approve_rights=True,
        )
        self.assertEqual(binding["stitchState"], "bound_to_candidate_awaiting_public_template")
        self.assertFalse(binding["galleryRegistered"])
        self.assertFalse(binding["datasetPublished"])
        approved = self.root / "review-approved" / "BuiltinImageOperation__image_crop" / "candidate.png"
        self.assertTrue(approved.is_file())
        stitch = json.loads(
            (self.root / "review-approved" / "BuiltinImageOperation__image_crop" / "stitch.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stitch["candidateContractId"], "template-candidate:BuiltinImageOperation:image_crop")
        with self.assertRaisesRegex(HumanReviewAssetError, "already bound"):
            approve_review_asset(
                self.root,
                workflow_id="BuiltinImageOperation:image_crop",
                reviewer="Sourav",
                approve_quality=True,
                approve_rights=True,
            )

    def test_reject_does_not_create_a_binding(self):
        stage_review_assets(self.root, ledger_paths=(self.ledger_path,))
        item = reject_review_asset(
            self.root,
            workflow_id="BuiltinImageOperation:image_crop",
            reviewer="Sourav",
            reason="Composition is too small for a template card.",
        )
        self.assertEqual(item["reviewState"], "rejected")
        self.assertFalse((self.root / "review-approved" / "bindings.v1.json").exists())
        with self.assertRaisesRegex(HumanReviewAssetError, "Rejected assets cannot be approved"):
            approve_review_asset(
                self.root,
                workflow_id="BuiltinImageOperation:image_crop",
                reviewer="Sourav",
                approve_quality=True,
                approve_rights=True,
            )


if __name__ == "__main__":
    unittest.main()
