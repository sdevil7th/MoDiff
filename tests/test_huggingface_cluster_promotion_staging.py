import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modiff.huggingface_cluster_promotions import (
    ClusterPromotionReceiptError,
    canonical_content_hash,
    reviewed_cluster_promotion_receipts,
)
from modiff.huggingface_cluster_runtime import REGISTERED_BLOCK_V2_DEFINITION_PINS
from modiff.huggingface_node_library import reviewed_huggingface_node_library
from scripts.stage_huggingface_cluster_promotion_receipt import (
    _historical_mapping_candidate,
    main,
)


QWEN = "diffusers.cluster-admission:QwenImageModularPipeline:text2image:mode:text_to_image"
MINIMAX = (
    "diffusers.cluster-admission:MiniMaxMusic3ModularPipeline:default:"
    "workflow:official_top_level_blocks"
)


class HuggingFaceClusterPromotionStagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = reviewed_huggingface_node_library()
        cls.qwen_definition, cls.qwen_admission = next(
            (definition, admission)
            for definition in cls.library["definitions"]
            for admission in definition.get("executionAdmissions", ())
            if admission["id"] == QWEN
        )

    def fixture_files(self, root):
        media = b"exact-reviewed-output"
        media_hash = hashlib.sha256(media).hexdigest()
        current_pin = REGISTERED_BLOCK_V2_DEFINITION_PINS[QWEN]
        provenance = {
            "schemaVersion": 2,
            "format": "modiff.live-proof.provenance.v2",
            "taskId": "frontend-stage-task-1",
            "backendReportedRuntimeFingerprint": "sha256:" + "1" * 64,
            "routeBinding": {
                "schemaVersion": 1,
                "admissionId": QWEN,
                "blockDefinition": {
                    "definitionId": QWEN,
                    "contentHash": current_pin[0],
                    "canonicalSha256": current_pin[1],
                },
                "studioExecutionSpec": self.qwen_admission["studioExecutionSpec"],
                "artifact": {
                    "repository": self.qwen_admission["artifact"]["repo"],
                    "revision": self.qwen_admission["artifact"]["revision"],
                },
                "modelDependencies": [],
            },
            "output": {
                "items": [
                    {
                        "index": 0,
                        "mediaType": "image",
                        "byteSize": len(media),
                        "encodedSha256": f"sha256:encoded:{media_hash}",
                    }
                ]
            },
            "blockers": [],
        }
        report = {
            "schemaVersion": 1,
            "format": "modiff.registered-block-v2-route-candidates.v1",
            "postPromotionAdmissionId": QWEN,
            "counts": {"existing": 90, "candidateSuccesses": 0, "blocked": 0, "total": 90},
            "existingPins": [
                {
                    "definitionId": self.qwen_definition["id"],
                    "definitionContentHash": (
                        "sha256:98dddb4c03eea94a0a8e9b1d318ecf5aab9fe62113b829263d8d605772ed6033"
                    ),
                    "admissionId": QWEN,
                    "previousCompiledDefinitionContentHash": current_pin[0],
                    "previousCompiledDefinitionCanonicalSha256": current_pin[1],
                    "compiledDefinitionContentHash": "block-definition-v2-74cfc64c",
                    "compiledDefinitionCanonicalSha256": (
                        "sha256:c0661cfc7442e5abb25c960b16773166d716fb9144ea9e99ca11939ebb08bef7"
                    ),
                    "executionGraphHash": "block-graph-v2-572aeab6",
                    "interfaceHash": "block-interface-v2-1e11c498",
                }
            ],
            "candidateSuccesses": [],
            "blocked": [],
        }
        provenance_path = root / "run-provenance.json"
        media_path = root / "output.webp"
        report_path = root / "post-promotion-route.json"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        media_path.write_bytes(media)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return provenance_path, media_path, report_path

    @staticmethod
    def arguments(provenance, media, report):
        return [
            "--admission-id",
            QWEN,
            "--provenance",
            str(provenance),
            "--media",
            str(media),
            "--post-promotion-candidate-report",
            str(report),
            "--receipt-id",
            "cluster-promotion:qwen-image-2512:text2image:staging-fixture",
            "--reviewed-at",
            "2026-09-02T20:00:00+05:30",
            "--review-comment",
            "Approved this exact frontend output.",
            "--approve-as-workspace-owner",
            "--confirm-inserted-through-frontend",
            "--confirm-expanded-official-blocks",
            "--confirm-edited-parameters",
            "--confirm-saved-and-refreshed",
            "--confirm-values-restored-exactly",
            "--confirm-collapsed-expanded-graph-equivalent",
            "--confirm-generated-through-frontend",
            "--atomic-bundle",
        ]

    def test_atomic_bundle_is_deterministic_and_non_installing(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture_files(Path(directory))
            first = io.StringIO()
            with patch("sys.stdout", first):
                self.assertEqual(main(self.arguments(*paths)), 0)
            second = io.StringIO()
            with patch("sys.stdout", second):
                self.assertEqual(main(self.arguments(*paths)), 0)

        self.assertEqual(first.getvalue(), second.getvalue())
        bundle = json.loads(first.getvalue())
        self.assertEqual(bundle["contentHash"], canonical_content_hash(bundle))
        self.assertEqual(
            bundle["evidenceRouteBinding"]["blockDefinition"]["contentHash"],
            REGISTERED_BLOCK_V2_DEFINITION_PINS[QWEN][0],
        )
        self.assertEqual(
            bundle["postPromotionRoute"]["blockDefinition"]["contentHash"],
            "block-definition-v2-74cfc64c",
        )
        self.assertEqual(
            bundle["promotionReceipt"]["blockDefinition"],
            bundle["postPromotionRoute"]["blockDefinition"],
        )
        self.assertIsNone(bundle["historicalCompilerMappingLedgerCandidate"])
        self.assertEqual(reviewed_cluster_promotion_receipts()["receipts"], [])

    def test_approval_is_required_before_bundle_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture_files(Path(directory))
            arguments = self.arguments(*paths)
            arguments.remove("--approve-as-workspace-owner")
            with self.assertRaisesRegex(ClusterPromotionReceiptError, "workspace-owner approval"):
                main(arguments)

    def test_minimax_bundle_rehashes_only_its_dependent_mapping(self):
        post_route = {
            "definitionId": "diffusers.modular:MiniMaxMusic3ModularPipeline:default",
            "manifestContentHash": (
                "sha256:8ab57813aa77c1d3bee7a63789fb94ac2f6c3884ff1c3a7e6da615629253cf60"
            ),
            "admissionId": MINIMAX,
            "blockDefinition": {
                "definitionId": MINIMAX,
                "contentHash": "block-definition-v2-0192c4b4",
                "canonicalSha256": (
                    "sha256:b503347bfb75e8379ab25b4c02b9097198064d1d149e848d70576078912e0b87"
                ),
            },
            "executionGraphHash": "block-graph-v2-f35deb93",
            "interfaceHash": "block-interface-v2-35294a7f",
        }
        ledger = _historical_mapping_candidate(post_route)
        by_id = {mapping["id"]: mapping for mapping in ledger["mappings"]}

        self.assertEqual(
            ledger["contentHash"],
            "sha256:8ffbacb22138d5927e07027b88c92ba5168acfb5a50b55d726dfff639a6ef001",
        )
        self.assertEqual(
            by_id["legacy-cluster-compiler-mapping:minimax-music3:2026-09-02"]["mappingHash"],
            "sha256:4895d1f3df1485e3c5b94cae97bf617a7d41e0762e3feec4d24579a243dd4d69",
        )
        self.assertEqual(
            by_id["legacy-cluster-compiler-mapping:transformers-ctc-stt:2026-09-02"]["mappingHash"],
            "sha256:3d049153434a808b94c83752b7c729e8b1cfe7acdfdeb605b08ac6f9e408aec3",
        )
        self.assertEqual(
            by_id["legacy-cluster-compiler-mapping:wan-22-ti2v-5b:2026-09-02"]["mappingHash"],
            "sha256:12d4d8eae96d868c28f1498f60662371db71d096ca25434b0858b83767bdcb24",
        )


if __name__ == "__main__":
    unittest.main()
