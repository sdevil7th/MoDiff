import json
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from modiff.human_review_assets import (
    HumanReviewAssetError,
    _campaign_diffusers_model_artifact,
    _campaign_model_revision,
    _load_generation_research,
    approve_review_asset,
    load_pending_index,
    record_removed_review_assets,
    record_quality_review,
    reconcile_campaign_review_assets,
    reject_review_asset,
    stage_review_assets,
)
from modiff.local_review_receipts import build_local_review_ledger, canonical_content_hash, parse_record


class HumanReviewAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
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

    def test_generic_diffusers_loader_preserves_exact_structured_model_artifact(self):
        output = {
            "graphSnapshot": {
                "nodes": [
                    {
                        "data": {
                            "action": "LoadPipeline",
                            "params": {
                                "model_id": {
                                    "value": {
                                        "source": "hub",
                                        "value": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                                    }
                                },
                                "revision": {"value": "aa76e7f4f4928f378716b6716a2130fba3caf5b1"},
                            },
                        }
                    }
                ]
            }
        }

        artifact = _campaign_diffusers_model_artifact(output)

        self.assertEqual(
            artifact,
            {
                "source": "hub",
                "value": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                "revision": "aa76e7f4f4928f378716b6716a2130fba3caf5b1",
            },
        )
        self.assertEqual(
            _campaign_model_revision(output),
            (
                "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
                "aa76e7f4f4928f378716b6716a2130fba3caf5b1",
            ),
        )

    def test_generic_diffusers_artifact_fails_closed_when_multiple_loaders_match(self):
        node = {
            "data": {
                "action": "LoadPipeline",
                "params": {
                    "model_id": {"value": {"source": "hub", "value": "org/model"}},
                    "revision": {"value": "1" * 40},
                },
            }
        }
        output = {"graphSnapshot": {"nodes": [node, node]}}

        self.assertIsNone(_campaign_diffusers_model_artifact(output))
        self.assertEqual(_campaign_model_revision(output), (None, None))

    def test_campaign_research_requires_exact_recipe_and_official_findings(self):
        research_path = self.root / "review-pending" / "research" / "unit.v1.json"
        research_path.parent.mkdir(parents=True)
        research = {
            "schemaVersion": 1,
            "kind": "generation_recipe_research",
            "workflowId": "UnitPipeline:text_to_image",
            "officialSources": [
                {"url": "https://huggingface.co/org/model", "finding": "Official recipe uses 20 steps."}
            ],
            "recipe": {"prompt": "one object", "steps": 20},
        }
        research_path.write_text(json.dumps(research), encoding="utf-8")

        evidence = _load_generation_research(
            self.root,
            "UnitPipeline:text_to_image",
            {"promptSource": "review-pending/research/unit.v1.json"},
        )

        self.assertEqual(evidence["path"], "review-pending/research/unit.v1.json")
        self.assertEqual(evidence["recipeContentHash"], canonical_content_hash(research["recipe"]))
        self.assertEqual(evidence["officialSources"], research["officialSources"])

        research["officialSources"] = []
        research_path.write_text(json.dumps(research), encoding="utf-8")
        with self.assertRaisesRegex(HumanReviewAssetError, "official source findings"):
            _load_generation_research(
                self.root,
                "UnitPipeline:text_to_image",
                {"promptSource": "review-pending/research/unit.v1.json"},
            )

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

    def test_quality_only_review_preserves_rights_and_publication_boundaries(self):
        stage_review_assets(self.root, ledger_paths=(self.ledger_path,))

        item = record_quality_review(
            self.root,
            workflow_id="BuiltinImageOperation:image_crop",
            reviewer="workspace-owner",
            approved=True,
        )

        self.assertEqual(item["reviewState"], "quality_approved_rights_pending")
        self.assertEqual(item["qualityReview"]["outcome"], "approved")
        self.assertFalse(item["qualityReview"]["rightsApproved"])
        self.assertFalse(item["qualityReview"]["galleryRegistered"])
        self.assertFalse(item["qualityReview"]["datasetPublished"])
        self.assertFalse((self.root / "review-approved").exists())
        index = json.loads((self.root / "review-pending" / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["items"][0]["reviewState"], "quality_approved_rights_pending")
        self.assertEqual(index["summary"]["awaitingHumanReviewCount"], 0)
        self.assertEqual(index["summary"]["qualityApprovedRightsPendingCount"], 1)
        self.assertEqual(index["summary"]["rejectedCount"], 0)

    def test_rejected_asset_requires_a_new_staged_copy_before_quality_approval(self):
        stage_review_assets(self.root, ledger_paths=(self.ledger_path,))
        record_quality_review(
            self.root,
            workflow_id="BuiltinImageOperation:image_crop",
            reviewer="workspace-owner",
            approved=False,
            reason="Too small for a Gallery card.",
        )

        with self.assertRaisesRegex(HumanReviewAssetError, "new staged copy"):
            record_quality_review(
                self.root,
                workflow_id="BuiltinImageOperation:image_crop",
                reviewer="workspace-owner",
                approved=True,
            )
        index = json.loads((self.root / "review-pending" / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["summary"]["awaitingHumanReviewCount"], 0)
        self.assertEqual(index["summary"]["qualityApprovedRightsPendingCount"], 0)
        self.assertEqual(index["summary"]["rejectedCount"], 1)

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

    def test_removed_review_bytes_are_reconciled_without_losing_audit_metadata(self):
        stage_review_assets(self.root, ledger_paths=(self.ledger_path,))
        staged = self.root / "review-pending" / "BuiltinImageOperation__image_crop" / "candidate.png"
        recovery = self.root / "trash" / "candidate.png"
        recovery.parent.mkdir(parents=True)
        staged.replace(recovery)

        index = record_removed_review_assets(
            self.root,
            reason="Workspace owner rejected the old campaign asset.",
            recovery_path=str(recovery.parent),
        )

        self.assertEqual(index["items"], [])
        self.assertEqual(index["summary"]["removedReviewItemCount"], 1)
        removed = index["removedReviewItems"][0]
        self.assertEqual(removed["workflowId"], "BuiltinImageOperation:image_crop")
        self.assertEqual(removed["reviewState"], "removed_after_review")
        self.assertEqual(removed["missingAssets"], ["candidate.png"])
        self.assertEqual(removed["recoveryPath"], str(recovery.parent))
        self.assertEqual(load_pending_index(self.root)["removedReviewItems"], index["removedReviewItems"])

    def test_campaign_reconciliation_requires_exact_bytes_and_preserves_open_gates(self):
        contracts_path = self.root / "data" / "template-candidate-contracts.v1.json"
        contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
        contracts["contracts"].append(
            {
                "canonicalWorkflowId": "AuraFlowPipeline:text_to_image",
                "graphHash": "b" * 64,
                "graphPath": "studio/aura-flow-pipeline/text-to-image.json",
                "mediaKind": "image",
            }
        )
        contracts_path.write_text(json.dumps(contracts), encoding="utf-8")
        (self.root / "data" / "template-authoring-specs.v1.json").write_text(
            json.dumps(
                {
                    "specifications": [
                        {
                            "canonicalWorkflowId": "AuraFlowPipeline:text_to_image",
                            "authoringState": "draft_complete_execution_pending",
                            "artifactPlan": {"requiredArtifacts": ["fal/AuraFlow-v0.3"]},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        index = stage_review_assets(self.root, ledger_paths=(self.ledger_path,))
        self.assertEqual(index["blockedGeneration"][0]["workflowId"], "AuraFlowPipeline:text_to_image")

        slug = "AuraFlowPipeline__text_to_image"
        asset_dir = self.root / "review-pending" / slug
        asset_dir.mkdir(parents=True)
        asset_path = asset_dir / "campaign-aura.png"
        Image.new("RGB", (16, 8), (10, 30, 90)).save(asset_path, format="WEBP")
        asset_sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        graph = {
            "nodes": [
                {
                    "id": "load",
                    "type": "custom",
                    "data": {
                        "module": "modules.DiffusersImage",
                        "action": "LoadPipeline",
                        "studioRole": "diffusersImagePipeline",
                        "params": {
                            "model_id": {"value": {"source": "hub", "value": "fal/AuraFlow-v0.3"}},
                            "revision": {"value": "1" * 40},
                        },
                    },
                }
            ],
            "edges": [],
        }
        outputs_path = self.root / "data" / "studio" / "outputs.json"
        outputs_path.parent.mkdir(parents=True)
        outputs_path.write_text(
            json.dumps(
                {
                    "outputs": [
                        {
                            "id": "run-output-aura",
                            "taskId": "task-aura-123",
                            "clientRunId": "client-aura",
                            "runInputHash": "run_aura",
                            "createdAt": 100,
                            "modelType": "AuraFlowPipeline",
                            "mode": "text_to_image",
                            "repo": "fal/AuraFlow-v0.3",
                            "prompt": "A blue-hour tram stop",
                            "seed": 42,
                            "width": 16,
                            "height": 8,
                            "steps": 28,
                            "guidanceScale": 3.5,
                            "graphSnapshot": graph,
                            "mediaItems": [
                                {
                                    "mediaHash": f"sha256:bytes:{asset_sha256}",
                                    "byteSize": asset_path.stat().st_size,
                                    "contentType": "image/webp",
                                    "backendPath": "@data/studio/outputs/aura.webp",
                                }
                            ],
                        },
                        {
                            "id": "wrong-workflow",
                            "taskId": "task-wrong-123",
                            "createdAt": 1,
                            "modelType": "PRXPipeline",
                            "mode": "text_to_image",
                            "mediaItems": [
                                {
                                    "mediaHash": f"sha256:bytes:{asset_sha256}",
                                    "byteSize": asset_path.stat().st_size,
                                    "contentType": "image/webp",
                                }
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        review_path = self.root / "review-pending" / "user-quality-review.v1.json"
        review_path.write_text(
            json.dumps(
                {
                    "reviewer": "workspace-owner",
                    "reviewedAt": "2026-08-21",
                    "boundary": {"rightsApproved": False, "datasetPublished": False},
                    "results": [
                        {
                            "workflowId": "AuraFlowPipeline:text_to_image",
                            "outcome": "quality_approved_rights_pending",
                            "mediaPath": f"{slug}/{asset_path.name}",
                            "reason": "Visual quality approved.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        frozen_root = self.root / "frozen-campaign"
        frozen_receipt = frozen_root / slug / "campaign-aura.quality-acceptance.json"
        frozen_receipt.parent.mkdir(parents=True)
        frozen_receipt.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "generation_quality_acceptance",
                    "workflowId": "AuraFlowPipeline:text_to_image",
                    "taskId": "task-aura-123",
                    "sourceCommit": "1" * 40,
                    "clientCommit": "2" * 40,
                    "executionSpecId": "auraflow-v0.3:text-to-image:v1",
                    "executionSpecHash": "studio-spec-v1-test",
                    "executionProfileId": "auraflow-v0.3:direct",
                    "runtimeFingerprint": "sha256:" + ("d" * 64),
                    "mediaSha256": asset_sha256,
                    "mime": "image/webp",
                    "byteSize": asset_path.stat().st_size,
                    "runtimeMeasurement": {
                        "elapsedSeconds": 12.5,
                        "peakReservedBytes": 2048,
                        "processRssBytes": 4096,
                    },
                }
            ),
            encoding="utf-8",
        )

        result = reconcile_campaign_review_assets(
            self.root,
            studio_outputs_path=outputs_path,
            user_review_path=review_path,
            frozen_receipts_root=frozen_root,
        )

        self.assertEqual(result["reconciled"][0]["taskId"], "task-aura-123")
        campaign_item = next(
            item for item in result["index"]["items"] if item["workflowId"] == "AuraFlowPipeline:text_to_image"
        )
        self.assertEqual(campaign_item["reviewState"], "quality_approved_rights_pending")
        self.assertEqual(campaign_item["elapsedSeconds"], 12.5)
        self.assertEqual(campaign_item["processRssBytes"], 4096)
        self.assertFalse(campaign_item["assets"][0]["filenameExtensionMatchesMime"])
        self.assertNotIn(
            "AuraFlowPipeline:text_to_image",
            {item["workflowId"] for item in result["index"]["blockedGeneration"]},
        )
        receipt_path = asset_dir / "campaign-aura.generation-review-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["execution"]["modelRevision"], "1" * 40)
        self.assertEqual(receipt["execution"]["modelRepo"], "fal/AuraFlow-v0.3")
        self.assertFalse(receipt["rightsApproved"])
        self.assertFalse(receipt["turnoverEligible"])
        self.assertIn("filename_extension_mime_mismatch", receipt["evidenceGaps"])
        self.assertNotIn("task_runtime_measurement_missing", receipt["evidenceGaps"])
        self.assertNotIn("runtime_fingerprint_missing", receipt["evidenceGaps"])
        self.assertEqual(receipt["execution"]["runtimeMeasurement"]["elapsedSeconds"], 12.5)
        self.assertEqual(
            receipt["frozenCampaignAcceptance"]["path"],
            f"{slug}/campaign-aura.quality-acceptance.json",
        )


if __name__ == "__main__":
    unittest.main()
