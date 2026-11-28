import copy
import hashlib
import json
from pathlib import Path
import unittest

from modiff.huggingface_catalog_only_gates import (
    HuggingFaceCatalogOnlyGateError,
    audit_huggingface_catalog_only_definitions,
    reviewed_huggingface_catalog_only_gates,
    validate_huggingface_catalog_only_gates,
)
from modiff.huggingface_node_library import build_huggingface_node_library
from modiff.modular_contract_only_registry import CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES


class HuggingFaceCatalogOnlyGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = build_huggingface_node_library()

    def test_hash_bound_evidence_uses_repository_lf_bytes(self):
        # Hashing a CRLF working copy can pass locally while Git's eol=lf
        # checkout fails on every clean machine. Keep strict byte hashes and
        # enforce the repository's existing text policy before binding them.
        root = Path(__file__).resolve().parents[1]
        ledger = json.loads((root / "data/huggingface-cluster-catalog-gates.v1.json").read_bytes())
        for family in ledger["families"]:
            with self.subTest(family=family["id"]):
                raw = (root / family["evidencePath"]).read_bytes()
                self.assertNotIn(b"\r\n", raw, "Evidence bytes must match the LF Git checkout.")
                self.assertEqual(hashlib.sha256(raw).hexdigest(), family["evidenceSha256"])

    def test_every_catalog_only_definition_has_one_hash_bound_gate(self):
        audit = audit_huggingface_catalog_only_definitions(self.library)

        self.assertEqual(audit["schemaVersion"], 1)
        self.assertEqual(audit["kind"], "huggingface_cluster_catalog_only_audit")
        self.assertEqual(audit["familyCount"], 7)
        self.assertEqual(audit["definitionCount"], 21)
        self.assertEqual(audit["routeEligibleCount"], 0)
        self.assertTrue(audit["contentHash"].startswith("sha256:"))
        self.assertTrue(
            all(item["status"] == "catalog_only_evidence_blocked" for item in audit["definitions"])
        )
        catalog_only_classes = {item["pipelineClass"] for item in audit["definitions"]}
        expected_catalog_only_classes = {
            item["pipelineClass"] for item in self.library["definitions"] if not item["executionAdmissions"]
        }
        fully_contract_only_classes = {
            item.class_name for item in CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES
        }
        self.assertEqual(catalog_only_classes, expected_catalog_only_classes)
        self.assertTrue(fully_contract_only_classes.issubset(catalog_only_classes))
        self.assertIn("Cosmos3OmniModularPipeline", catalog_only_classes - fully_contract_only_classes)

    def test_gate_counts_preserve_the_exact_current_family_split(self):
        audit = audit_huggingface_catalog_only_definitions(self.library)
        by_family = {
            family_id: sum(item["familyId"] == family_id for item in audit["definitions"])
            for family_id in {item["familyId"] for item in audit["definitions"]}
        }

        self.assertEqual(
            by_family,
            {
                "cosmos-3": 5,
                "flux-3": 4,
                "ideogram-4": 1,
                "krea-2": 2,
                "ltx-2.5": 4,
                "pika-h3-flex": 3,
                "stable-diffusion-3": 2,
            },
        )

    def test_flux3_family_is_hash_bound_inactive_and_retains_every_artifact_blocker(self):
        root = Path(__file__).resolve().parents[1]
        family = next(item for item in reviewed_huggingface_catalog_only_gates()["families"]
                      if item["id"] == "flux-3")
        evidence_path = root / family["evidencePath"]
        self.assertEqual(hashlib.sha256(evidence_path.read_bytes()).hexdigest(), family["evidenceSha256"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        artifact = root / evidence["artifactReview"]["path"]
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), evidence["artifactReview"]["sha256"])
        reviewed = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(family["pipelineClasses"], ["Flux3DistilledModularPipeline", "Flux3ModularPipeline"])
        self.assertEqual(evidence["pipelineClasses"], family["pipelineClasses"])
        self.assertEqual(evidence["catalogSourceRevision"], reviewed["catalogSourceRevision"])
        self.assertEqual(family["blockers"], reviewed["admission"]["unresolvedGates"])
        self.assertEqual(evidence["admission"]["unresolvedGates"], family["blockers"])
        self.assertIs(evidence["admission"]["runtimeCatalogExposed"], False)
        self.assertIs(evidence["admission"]["downloadCatalogExposed"], False)
        for field in ("public", "autoEligible", "templateEligible", "liveProof"):
            self.assertIs(reviewed["admission"][field], False)
        audit = audit_huggingface_catalog_only_definitions(self.library)
        selected = [item for item in audit["definitions"] if item["familyId"] == "flux-3"]
        self.assertEqual(len(selected), 4)
        self.assertEqual(audit["routeEligibleCount"], 0)
        self.assertTrue(all(item["status"] == "catalog_only_evidence_blocked" for item in selected))

    def test_pika_supplement_stays_source_bound_and_catalog_only(self):
        root = Path(__file__).resolve().parents[1]
        family = next(row for row in reviewed_huggingface_catalog_only_gates()['families']
                      if row['id'] == 'pika-h3-flex')
        evidence = json.loads((root / family['evidencePath']).read_bytes())
        source = json.loads((root / 'data/pika-h3-flex-workflow-contract.v1.json').read_bytes())
        self.assertEqual(evidence['source'], source['source'])
        artifact = evidence['artifactReview']
        self.assertEqual(hashlib.sha256((root / artifact['path']).read_bytes()).hexdigest(), artifact['sha256'])
        self.assertEqual(evidence['admission']['unresolvedGates'], family['blockers'])
        routes = [row for row in audit_huggingface_catalog_only_definitions(self.library)['definitions']
                  if row['familyId'] == 'pika-h3-flex']
        self.assertEqual({row['workflowId'] for row in routes}, {'t2va', 'fl2va', 'ref2va'})
        self.assertTrue(all(row['status'] == 'catalog_only_evidence_blocked' for row in routes))

    def test_reviewed_ledger_is_detached_and_rejects_drift(self):
        first = reviewed_huggingface_catalog_only_gates()
        first["families"][0]["blockers"].append("caller_mutation")
        second = reviewed_huggingface_catalog_only_gates()
        self.assertNotIn("caller_mutation", second["families"][0]["blockers"])

        malformed = copy.deepcopy(second)
        malformed["families"][0]["blockers"].append("unknown_unreviewed_gate")
        with self.assertRaisesRegex(HuggingFaceCatalogOnlyGateError, "hash"):
            validate_huggingface_catalog_only_gates(malformed)

    def test_audit_fails_closed_for_an_unregistered_catalog_only_class(self):
        stale = copy.deepcopy(self.library)
        definition = next(item for item in stale["definitions"] if not item["executionAdmissions"])
        definition["pipelineClass"] = "UnknownModularPipeline"
        with self.assertRaisesRegex(HuggingFaceCatalogOnlyGateError, "no reviewed execution-gate family"):
            audit_huggingface_catalog_only_definitions(stale)


if __name__ == "__main__":
    unittest.main()
