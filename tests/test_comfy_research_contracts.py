import copy
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from modiff.comfy_research_contracts import (
    COMFY_RESEARCH_CATALOG_LEDGER_PATH,
    COMFY_RESEARCH_CONTRACT_SCHEMA_VERSION,
    ComfyResearchContractError,
    validate_comfy_research_contract_ledger,
)
from modiff.comfy_template_research import PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / COMFY_RESEARCH_CATALOG_LEDGER_PATH
CONTRACT_PATH = ROOT / "data" / "research" / "comfy-research-contracts.v1.json"
WORKFLOW_MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class ComfyResearchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog_payload = CATALOG_PATH.read_bytes()
        cls.catalog_sha256 = hashlib.sha256(cls.catalog_payload).hexdigest()
        cls.catalog = json.loads(cls.catalog_payload)
        cls.ledger = _load(CONTRACT_PATH)
        cls.workflow_ids = {item["id"] for item in _load(WORKFLOW_MANIFEST_PATH)["workflows"]}

    def test_checked_in_contracts_validate_against_exact_pinned_catalog(self):
        validated = validate_comfy_research_contract_ledger(
            self.ledger,
            catalog_ledger=self.catalog,
            catalog_ledger_sha256=self.catalog_sha256,
            supported_workflow_ids=self.workflow_ids,
        )
        self.assertEqual(validated, self.ledger)
        self.assertEqual(self.ledger["schemaVersion"], COMFY_RESEARCH_CONTRACT_SCHEMA_VERSION)
        self.assertEqual(self.ledger["source"]["revision"], PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION)
        self.assertEqual(self.ledger["source"]["catalogLedgerSha256"], self.catalog_sha256)

        self.assertEqual(
            self.ledger["summary"],
            {
                "contractCount": 308,
                "templateContractCount": 217,
                "blueprintContractCount": 91,
                "exactMappingCount": 54,
                "sourceKindCounts": {"blueprint": 91, "template": 217},
                "catalogDispositionCounts": {
                    "research_candidate": 164,
                    "source_review_required": 144,
                },
                "catalogMappingStatusCounts": {
                    "ambiguous_task_evidence": 40,
                    "ambiguous_task_or_model_evidence": 76,
                    "existing_contract_candidate": 54,
                    "research_candidate": 138,
                },
                "contractStateCounts": {
                    "contract_undetermined": 116,
                    "existing_modiff_contract_candidate": 54,
                    "new_contract_not_authored": 138,
                },
                "decisionStateCounts": {
                    "contract_authoring_required": 99,
                    "evidence_resolution_required": 116,
                    "semantic_comparison_pending": 30,
                    "source_review_before_contract_authoring": 39,
                    "source_review_before_semantic_comparison": 24,
                },
            },
        )

    def test_every_non_ineligible_template_and_blueprint_has_one_stable_contract(self):
        expected = {
            (source_kind, item["id"])
            for source_kind, records in (
                ("template", self.catalog["templates"]),
                ("blueprint", self.catalog["blueprints"]),
            )
            for item in records
            if item["disposition"] != "ineligible"
        }
        actual = {(item["sourceKind"], item["catalogId"]) for item in self.ledger["contracts"]}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(self.ledger["contracts"]))

        research_template_ids = {
            item["id"]
            for item in self.catalog["templates"]
            if item["mappingStatus"] == "research_candidate"
        }
        covered_research_template_ids = {
            item["catalogId"]
            for item in self.ledger["contracts"]
            if item["sourceKind"] == "template"
            and item["catalogMappingStatus"] == "research_candidate"
        }
        self.assertEqual(len(research_template_ids), 138)
        self.assertEqual(covered_research_template_ids, research_template_ids)

    def test_contracts_expose_evidence_decisions_blockers_rights_and_authoring_work(self):
        contracts = {item["contractId"]: item for item in self.ledger["contracts"]}

        new_contract = contracts["comfy-research:template:3d_moge_panorama_to_mesh"]
        self.assertEqual(new_contract["contractState"], "new_contract_not_authored")
        self.assertEqual(new_contract["decisionState"], "contract_authoring_required")
        self.assertEqual(new_contract["taskEvidence"]["selectedCandidateMode"], "image_to_3d")
        self.assertEqual(new_contract["mediaEvidence"]["candidateOutputMediaKinds"], ["three_d"])
        self.assertEqual(new_contract["modelEvidence"]["catalogModelLabels"], ["MoGe"])
        self.assertIn("author_bounded_modiff_contract", new_contract["authoringRequirements"])
        self.assertIn("original_modiff_workflow_not_authored", new_contract["blockers"])
        self.assertNotIn("exactMapping", new_contract)

        ambiguous = contracts["comfy-research:blueprint:brightness_and_contrast"]
        self.assertEqual(ambiguous["contractState"], "contract_undetermined")
        self.assertEqual(ambiguous["decisionState"], "evidence_resolution_required")
        self.assertEqual(ambiguous["taskEvidence"]["state"], "ambiguous_or_missing")
        self.assertEqual(ambiguous["mediaEvidence"]["catalogCategoryMediaHints"], ["image"])
        self.assertEqual(ambiguous["modelEvidence"]["state"], "missing")
        self.assertEqual(ambiguous["sourceReview"]["state"], "required")
        self.assertIn("resolve_task_media_and_model_evidence", ambiguous["authoringRequirements"])

        exact = contracts["comfy-research:template:flux_dev_full_text_to_image"]
        self.assertEqual(exact["contractState"], "existing_modiff_contract_candidate")
        self.assertEqual(exact["decisionState"], "semantic_comparison_pending")
        self.assertEqual(exact["exactMapping"]["moDiffWorkflowId"], "FluxDevPipeline:text_to_image")
        self.assertIn(exact["exactMapping"]["moDiffWorkflowId"], self.workflow_ids)
        self.assertIn("semantic_equivalence_not_reviewed", exact["blockers"])

        source_review = contracts["comfy-research:template:3d_hunyuan3d-v2.1"]
        self.assertEqual(source_review["decisionState"], "source_review_before_contract_authoring")
        self.assertEqual(source_review["sourceReview"]["state"], "required")
        self.assertIn("catalog_entry_source_review_required", source_review["blockers"])

    def test_every_record_is_explicitly_non_executable_and_has_no_asset_claim(self):
        self.assertEqual(
            self.ledger["boundary"],
            {
                "catalogMetadataOnly": True,
                "importsComfyGraphs": False,
                "copiesComfyNodesOrPackages": False,
                "downloadsModelsOrMedia": False,
                "executesComfyWorkflows": False,
                "publishesMoDiffTemplates": False,
                "generatesOrApprovesAssets": False,
                "executionClaim": "none",
                "assetClaim": "none",
            },
        )
        expected_claims = {
            "comfyGraphImported": False,
            "comfyWorkflowExecutable": False,
            "moDiffNodeSupportClaimedByRecord": False,
            "moDiffWorkflowSupportClaimedByRecord": False,
            "publicTemplateClaimedByRecord": False,
            "assetGenerated": False,
            "assetApproved": False,
        }
        for contract in self.ledger["contracts"]:
            with self.subTest(contract=contract["contractId"]):
                self.assertEqual(contract["claims"], expected_claims)
                self.assertTrue(contract["blockers"])
                self.assertTrue(contract["authoringRequirements"])
                self.assertEqual(contract["rightsReview"]["state"], "independent_review_required")

        forbidden_payload_keys = {
            "edges",
            "graph",
            "links",
            "nodes",
            "packages",
            "prompt",
            "workflowPayload",
        }

        def visit(value):
            if isinstance(value, dict):
                self.assertTrue(forbidden_payload_keys.isdisjoint(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.ledger)

    def test_exact_mappings_are_present_only_for_curated_existing_contract_candidates(self):
        for contract in self.ledger["contracts"]:
            with self.subTest(contract=contract["contractId"]):
                if contract["catalogMappingStatus"] == "existing_contract_candidate":
                    self.assertIn("exactMapping", contract)
                    self.assertIn(contract["exactMapping"]["moDiffWorkflowId"], self.workflow_ids)
                else:
                    self.assertNotIn("exactMapping", contract)

    def test_validator_rejects_boundary_decision_and_catalog_hash_mutations(self):
        for mutate, message in (
            (
                lambda value: value["boundary"].__setitem__("executionClaim", "qualified"),
                "execution or asset boundary",
            ),
            (
                lambda value: value["contracts"][0].__setitem__("decisionState", "approved"),
                "stale or internally inconsistent",
            ),
            (
                lambda value: value["source"].__setitem__("catalogLedgerSha256", "0" * 64),
                "provenance",
            ),
        ):
            with self.subTest(message=message):
                mutated = copy.deepcopy(self.ledger)
                mutate(mutated)
                with self.assertRaisesRegex(ComfyResearchContractError, message):
                    validate_comfy_research_contract_ledger(
                        mutated,
                        catalog_ledger=self.catalog,
                        catalog_ledger_sha256=self.catalog_sha256,
                        supported_workflow_ids=self.workflow_ids,
                    )

    def test_cli_check_and_validate_are_offline_and_deterministic(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_comfy_research_contracts.py",
                "--check",
                "--validate",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
