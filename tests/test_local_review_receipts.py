import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from modiff.local_review_receipts import (
    build_local_review_ledger,
    canonical_content_hash,
    loopback_base_url,
    merge_local_review_ledger,
    parse_record,
    validate_local_review_ledger,
)


class LocalReviewReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        data = self.root / "data"
        (data / "images").mkdir(parents=True)
        (data / "exports").mkdir(parents=True)
        (data / "images" / "candidate.png").write_bytes(b"review-image")
        (data / "exports" / "candidate.json").write_text('{"ready":true}\n', encoding="utf-8")
        contracts = {
            "contracts": [
                {
                    "canonicalWorkflowId": "BuiltinImageOperation:crop",
                    "contractId": "template-candidate:BuiltinImageOperation:crop",
                    "graphHash": "a" * 64,
                    "graphPath": "studio/builtin-image-operation/crop.json",
                    "mediaKind": "image",
                },
                {
                    "canonicalWorkflowId": "BuiltinDataOperation:data_conversion",
                    "contractId": "template-candidate:BuiltinDataOperation:data_conversion",
                    "graphHash": "b" * 64,
                    "graphPath": "studio/builtin-data-operation/data-conversion.json",
                    "mediaKind": "json",
                },
            ]
        }
        (data / "template-candidate-contracts.v1.json").write_text(json.dumps(contracts), encoding="utf-8")

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

    def test_capture_binds_completed_tasks_outputs_and_hidden_candidate_graphs(self):
        records = [
            parse_record("BuiltinImageOperation:crop|task-image|images/candidate.png"),
            parse_record("BuiltinDataOperation:data_conversion|task-data|exports/candidate.json"),
        ]
        document = build_local_review_ledger(root=self.root, records=records, fetch_run=self._run)

        self.assertEqual(document["summary"]["receiptCount"], 2)
        self.assertEqual(document["summary"]["outputKindCounts"], {"image": 1, "json": 1})
        self.assertTrue(document["boundary"]["humanReviewRequired"])
        self.assertFalse(document["boundary"]["galleryApprovalClaimed"])
        self.assertEqual(validate_local_review_ledger(document, root=self.root), document)
        image = next(item for item in document["receipts"] if item["workflowId"].endswith(":crop"))
        self.assertEqual(image["task"]["taskId"], "task-image")
        self.assertEqual(image["outputs"][0]["byteSize"], len(b"review-image"))
        self.assertEqual(image["reviewState"], "pending_human_review")
        self.assertFalse(image["claims"]["releaseEligible"])

    def test_capture_rejects_wrong_media_kind_incomplete_tasks_and_duplicate_identity(self):
        wrong_kind = [
            parse_record("BuiltinImageOperation:crop|task-image|exports/candidate.json"),
        ]
        with self.assertRaisesRegex(ValueError, "does not match workflow kind"):
            build_local_review_ledger(root=self.root, records=wrong_kind, fetch_run=self._run)

        incomplete = [
            parse_record("BuiltinImageOperation:crop|task-image|images/candidate.png"),
        ]
        with self.assertRaisesRegex(ValueError, "is not completed"):
            build_local_review_ledger(
                root=self.root,
                records=incomplete,
                fetch_run=lambda task_id: {"task": {"task_id": task_id, "status": "failed"}},
            )

        duplicate = [
            parse_record("BuiltinImageOperation:crop|task-image|images/candidate.png"),
            parse_record("BuiltinImageOperation:crop|task-other|images/candidate.png"),
        ]
        with self.assertRaisesRegex(ValueError, "workflow and task identities must be unique"):
            build_local_review_ledger(root=self.root, records=duplicate, fetch_run=self._run)

    def test_validation_rejects_tampering_and_stale_graph_binding(self):
        document = build_local_review_ledger(
            root=self.root,
            records=[parse_record("BuiltinImageOperation:crop|task-image|images/candidate.png")],
            fetch_run=self._run,
        )
        document["receipts"][0]["claims"]["galleryApproved"] = True
        with self.assertRaisesRegex(ValueError, "content hash"):
            validate_local_review_ledger(document, root=self.root)

        document["receipts"][0]["claims"]["galleryApproved"] = False
        document["contentHash"] = canonical_content_hash(document)
        contracts_path = self.root / "data" / "template-candidate-contracts.v1.json"
        contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
        contracts["contracts"][0]["graphHash"] = "e" * 64
        contracts_path.write_text(json.dumps(contracts), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "graph binding is stale"):
            validate_local_review_ledger(document, root=self.root)

    def test_validation_rehashes_local_output_bytes(self):
        document = build_local_review_ledger(
            root=self.root,
            records=[parse_record("BuiltinImageOperation:crop|task-image|images/candidate.png")],
            fetch_run=self._run,
        )
        (self.root / "data" / "images" / "candidate.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "output bytes drifted"):
            validate_local_review_ledger(document, root=self.root)

    def test_merge_validates_retained_receipts_and_fetches_only_new_tasks(self):
        existing = build_local_review_ledger(
            root=self.root,
            records=[parse_record("BuiltinImageOperation:crop|task-image|images/candidate.png")],
            fetch_run=self._run,
        )
        fetched = []

        def fetch(task_id):
            fetched.append(task_id)
            return self._run(task_id)

        merged = merge_local_review_ledger(
            root=self.root,
            existing=existing,
            records=[parse_record("BuiltinDataOperation:data_conversion|task-data|exports/candidate.json")],
            fetch_run=fetch,
        )

        self.assertEqual(fetched, ["task-data"])
        self.assertEqual(merged["summary"]["receiptCount"], 2)
        self.assertEqual(merged["summary"]["outputKindCounts"], {"image": 1, "json": 1})
        self.assertEqual(validate_local_review_ledger(merged, root=self.root), merged)

        with self.assertRaisesRegex(ValueError, "identities must be unique"):
            merge_local_review_ledger(
                root=self.root,
                existing=existing,
                records=[parse_record("BuiltinImageOperation:crop|task-other|images/candidate.png")],
                fetch_run=fetch,
            )

    def test_record_and_server_boundaries_fail_closed(self):
        self.assertEqual(loopback_base_url("http://127.0.0.1:8088/"), "http://127.0.0.1:8088")
        self.assertEqual(loopback_base_url("http://localhost:8088"), "http://localhost:8088")
        for server in ("https://example.com", "file:///tmp/socket", "http://127.0.0.1:8088/path"):
            with self.subTest(server=server), self.assertRaises(ValueError):
                loopback_base_url(server)
        for record in (
            "BuiltinImageOperation:crop|task-image|../candidate.png",
            "BuiltinImageOperation:crop|task-image|/images/candidate.png",
            "BuiltinImageOperation:crop|bad task|images/candidate.png",
            "bad-workflow|task-image|images/candidate.png",
            "BuiltinImageOperation:crop|task-image|models/candidate.png",
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                parse_record(record)


if __name__ == "__main__":
    unittest.main()
