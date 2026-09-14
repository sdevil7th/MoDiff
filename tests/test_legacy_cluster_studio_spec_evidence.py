import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from modiff.legacy_cluster_studio_spec_evidence import (
    LegacyStudioSpecEvidenceError,
    extract_studio_spec_evidence,
    reviewed_studio_spec_evidence,
    reviewed_studio_spec_partial_reviews,
    studio_spec_content_hash_v1,
    validate_historical_studio_spec,
    validate_studio_spec_evidence_ledger,
    validate_studio_spec_partial_review_ledger,
)


class LegacyClusterStudioSpecEvidenceTests(unittest.TestCase):
    def test_checked_in_evidence_is_bounded_redacted_and_non_authorizing(self):
        evidence = reviewed_studio_spec_evidence()
        reviews = reviewed_studio_spec_partial_reviews()

        self.assertEqual(len(evidence["sources"]), 2)
        self.assertEqual(len(evidence["specifications"]), 18)
        self.assertEqual(reviews["reviews"], [])
        self.assertEqual(
            {
                entry["identity"]["id"]
                for entry in evidence["specifications"]
            }
            & {
                "whisper-tiny:speech-to-text:v1",
                "wan-flf:image-to-video:v1",
                "smollm2-135m-instruct:text-generation:v1",
            },
            {
                "whisper-tiny:speech-to-text:v1",
                "wan-flf:image-to-video:v1",
                "smollm2-135m-instruct:text-generation:v1",
            },
        )
        self.assertFalse(evidence["boundary"]["authorizesConversion"])
        self.assertFalse(evidence["boundary"]["authorizesCompilerSupplement"])
        self.assertFalse(evidence["boundary"]["authorizesSemanticEquivalence"])
        self.assertFalse(reviews["boundary"]["authorizesConversion"])
        serialized = json.dumps(evidence)
        for private_field in ("sourcePath", "instanceId", "workflowId", "promptValue"):
            self.assertNotIn(private_field, serialized)
        for source in evidence["sources"]:
            self.assertFalse(Path(source["artifactLabel"]).is_absolute())

    def test_evidence_body_self_hash_is_revalidated(self):
        evidence = reviewed_studio_spec_evidence()
        evidence["specifications"][0]["specification"]["defaultRepo"] = "tampered/repository"

        with self.assertRaisesRegex(LegacyStudioSpecEvidenceError, "self-hash"):
            validate_studio_spec_evidence_ledger(evidence)

    def test_evidence_boundary_cannot_be_upgraded_to_authority(self):
        evidence = reviewed_studio_spec_evidence()
        evidence["boundary"]["authorizesConversion"] = True

        with self.assertRaisesRegex(LegacyStudioSpecEvidenceError, "boundary"):
            validate_studio_spec_evidence_ledger(evidence)

    def test_historical_spec_rejects_value_bearing_actions(self):
        specification = copy.deepcopy(
            reviewed_studio_spec_evidence()["specifications"][0]["specification"]
        )
        specification["actions"] = [{"promptValue": "private prompt"}]
        specification["contentHash"] = studio_spec_content_hash_v1(specification)

        with self.assertRaisesRegex(LegacyStudioSpecEvidenceError, "actions must be empty"):
            validate_historical_studio_spec(specification)

    def test_partial_review_cannot_claim_semantic_equivalence(self):
        evidence = reviewed_studio_spec_evidence()
        reviews = reviewed_studio_spec_partial_reviews()
        entry = evidence["specifications"][0]
        reviews["reviews"] = [
            {
                "id": "legacy-studio-spec-partial-review:forbidden-decision",
                "historical": {
                    "manifestDefinitionId": "diffusers.modular:Historical:default",
                    "libraryRevision": "a" * 40,
                    "manifestContentHash": "sha256:" + "b" * 64,
                    "executionAdmissionId": "historical:admission",
                    "studioExecutionSpec": copy.deepcopy(entry["identity"]),
                },
                "evidence": {"canonicalBodySha256": entry["canonicalBodySha256"]},
                "decision": "semantic_equivalent",
                "reviewer": "reviewer",
                "reviewedAt": "2026-09-02T00:00:00Z",
                "notes": "This decision is outside the partial-evidence boundary.",
                "reviewHash": "sha256:" + "c" * 64,
            }
        ]

        with self.assertRaisesRegex(LegacyStudioSpecEvidenceError, "cannot authorize equivalence"):
            validate_studio_spec_partial_review_ledger(reviews, evidence_ledger=evidence)

    def test_extractor_reads_only_pinned_capability_specs_and_exact_historical_identity(self):
        checked_in = reviewed_studio_spec_evidence()
        specification = copy.deepcopy(checked_in["specifications"][0]["specification"])
        capture = {
            "nodes": {
                "studioModelCapabilities": [
                    {"studioExecutionSpecs": [specification]},
                ]
            },
            "workflows": [
                {
                    "sourcePath": "private/workflow.json",
                    "instanceId": "private-instance",
                    "promptValue": "private prompt",
                }
            ],
        }
        raw = json.dumps(capture, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(raw, mtime=0)
        source_id = "synthetic-reviewed-capture"
        expectations = {
            source_id: {
                "artifactLabel": "ignored-test-review/synthetic.json.gz",
                "compressedSha256": "sha256:" + hashlib.sha256(compressed).hexdigest(),
                "compressedBytes": len(compressed),
                "uncompressedSha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "uncompressedBytes": len(raw),
                "selector": "/nodes/studioModelCapabilities/*/studioExecutionSpecs/*",
            }
        }
        inventory = {
            "workflows": [
                {
                    "sourcePath": "private/workflow.json",
                    "composites": [
                        {
                            "classification": "legacy_cluster_root",
                            "sourceRefs": {
                                "legacyCluster": {
                                    "definition": {
                                        "id": "diffusers.modular:Historical:default",
                                        "libraryRevision": "a" * 40,
                                        "contentHash": "sha256:" + "b" * 64,
                                    },
                                    "admissionId": "historical:admission",
                                    "studioExecutionSpec": {
                                        "id": specification["id"],
                                        "contentHash": specification["contentHash"],
                                        "executionProfileId": specification["executionProfileId"],
                                    },
                                }
                            },
                        }
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json.gz"
            path.write_bytes(compressed)
            result = extract_studio_spec_evidence(
                {source_id: path},
                inventory,
                {"definitions": []},
                source_expectations=expectations,
            )

        self.assertEqual(len(result["specifications"]), 1)
        self.assertEqual(result["specifications"][0]["specification"], specification)
        serialized = json.dumps(result)
        self.assertNotIn("private/workflow", serialized)
        self.assertNotIn("private prompt", serialized)
        self.assertFalse(result["boundary"]["authorizesConversion"])

    def test_extractor_rejects_a_capture_with_the_wrong_byte_receipt(self):
        expectations = {
            "synthetic-reviewed-capture": {
                "artifactLabel": "ignored-test-review/synthetic.json.gz",
                "compressedSha256": "sha256:" + "0" * 64,
                "compressedBytes": 1,
                "uncompressedSha256": "sha256:" + "1" * 64,
                "uncompressedBytes": 1,
                "selector": "/nodes/studioModelCapabilities/*/studioExecutionSpecs/*",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json.gz"
            path.write_bytes(b"x")
            with self.assertRaisesRegex(LegacyStudioSpecEvidenceError, "compressed SHA-256"):
                extract_studio_spec_evidence(
                    {"synthetic-reviewed-capture": path},
                    {"workflows": []},
                    {"definitions": []},
                    source_expectations=expectations,
                )

    def test_extractor_rejects_a_symlinked_capture(self):
        expectations = {
            "synthetic-reviewed-capture": {
                "artifactLabel": "ignored-test-review/synthetic.json.gz",
                "compressedSha256": "sha256:" + "0" * 64,
                "compressedBytes": 1,
                "uncompressedSha256": "sha256:" + "1" * 64,
                "uncompressedBytes": 1,
                "selector": "/nodes/studioModelCapabilities/*/studioExecutionSpecs/*",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "capture-target.json.gz"
            target.write_bytes(b"x")
            link = Path(temporary) / "capture-link.json.gz"
            link.symlink_to(target)
            with self.assertRaisesRegex(LegacyStudioSpecEvidenceError, "safe regular file"):
                extract_studio_spec_evidence(
                    {"synthetic-reviewed-capture": link},
                    {"workflows": []},
                    {"definitions": []},
                    source_expectations=expectations,
                )


if __name__ == "__main__":
    unittest.main()
