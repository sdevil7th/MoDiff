import copy
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

    def test_every_catalog_only_definition_has_one_hash_bound_gate(self):
        audit = audit_huggingface_catalog_only_definitions(self.library)

        self.assertEqual(audit["schemaVersion"], 1)
        self.assertEqual(audit["kind"], "huggingface_cluster_catalog_only_audit")
        self.assertEqual(audit["familyCount"], 5)
        self.assertEqual(audit["definitionCount"], 14)
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
                "ideogram-4": 1,
                "krea-2": 2,
                "ltx-2.5": 4,
                "stable-diffusion-3": 2,
            },
        )

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
