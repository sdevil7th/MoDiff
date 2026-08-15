import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from modiff.template_candidate_contracts import (
    TemplateCandidateContractError,
    build_template_candidate_contract_ledger,
    render_template_candidate_contract_ledger,
    validate_template_candidate_contract_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "template-candidate-contracts.v1.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
COVERAGE_PATH = ROOT / "data" / "upstream-coverage.v1.json"
COMFY_PATH = ROOT / "data" / "research" / "comfy-workflow-catalog.v1.json"
MANIFEST_BINDING_FIELDS = (
    "pipelineClasses",
    "requiredArtifacts",
    "requiredInputs",
    "qualificationStatus",
    "graphQualificationStatus",
    "runtimeQualificationStatus",
    "optimizationQualificationStatus",
    "qualifiedRuntimeProfiles",
    "qualificationScope",
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value):
    semantic = dict(value)
    semantic.pop("contentHash", None)
    encoded = json.dumps(
        semantic,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    value["contentHash"] = "sha256:" + hashlib.sha256(encoded).hexdigest()


def _needs_input_examples(required_inputs):
    if isinstance(required_inputs, list):
        return bool(required_inputs)
    return any(required_inputs.get(field) for field in ("requiredImages", "requiredVideos", "requiredAudio"))


class TemplateCandidateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = _load(LEDGER_PATH)
        cls.manifest = _load(MANIFEST_PATH)
        cls.coverage = _load(COVERAGE_PATH)
        cls.comfy = _load(COMFY_PATH)

    def test_checked_in_ledger_is_deterministic_and_valid(self):
        validated = validate_template_candidate_contract_ledger(self.ledger, root=ROOT)
        self.assertEqual(validated, self.ledger)
        self.assertEqual(build_template_candidate_contract_ledger(ROOT), self.ledger)
        self.assertEqual(render_template_candidate_contract_ledger(self.ledger), LEDGER_PATH.read_text())
        self.assertEqual(
            self.ledger["summary"],
            {
                "canonicalWorkflowCount": 197,
                "publicTemplateCount": 77,
                "canonicalWorkflowsWithPublicTemplates": 51,
                "candidateContractCount": 146,
                "mediaKindCounts": {"audio": 6, "image": 98, "json": 9, "video": 33},
                "contractsRequiringInputExamples": 91,
                "contractsWithComfyResearchRecords": 11,
                "comfyResearchRecordCount": 14,
                "comfyMappingStatusCounts": {"existing_contract_candidate": 14},
            },
        )

    def test_newly_admitted_workflows_are_exact_hidden_zero_asset_contracts(self):
        expected = {
            "HuggingFaceImageTextToTextModel:image_to_text",
            "HuggingFaceTextGenerationModel:text_generation",
            "HunyuanDiTPAGPipeline:text_to_image",
            "LatentConsistencyModelPipeline:edit_image",
            "PixArtSigmaPAGPipeline:text_to_image",
            "SanaPAGPipeline:text_to_image",
            "StableDiffusionPAGPipeline:edit_image",
            "StableDiffusionPAGPipeline:inpaint",
            "SpandrelVideoUpscale:video_upscale",
            "BuiltinAudioOperation:audio_trim",
            "BuiltinAudioOperation:audio_join",
            "BuiltinAudioOperation:audio_loudness_match",
            "BuiltinDataOperation:text_select",
            "BuiltinDataOperation:data_conversion",
            "BuiltinDataOperation:graph_utility",
            "BuiltinVideoOperation:frame_interpolation",
        }
        contracts_by_id = {row["canonicalWorkflowId"]: row for row in self.ledger["contracts"]}

        self.assertTrue(expected <= contracts_by_id.keys())
        for workflow_id in expected:
            contract = contracts_by_id[workflow_id]
            self.assertEqual(contract["publicationState"], "hidden_candidate")
            self.assertEqual(contract["assetState"], "not_generated")
            self.assertEqual(contract["assets"], [])
            self.assertEqual(contract["evidenceState"]["maximumClaim"], "canonical_graph_binding_only")

    def test_candidates_are_the_exact_complement_of_public_template_workflows(self):
        manifest_by_id = {row["id"]: row for row in self.manifest["workflows"]}
        public_templates = self.coverage["publicTemplates"]
        public_workflow_ids = {row["canonicalWorkflowId"] for row in public_templates}
        coverage_public_workflow_ids = {
            row["id"] for row in self.coverage["canonicalWorkflows"] if row["publicTemplateIds"]
        }
        contracts_by_id = {row["canonicalWorkflowId"]: row for row in self.ledger["contracts"]}

        self.assertEqual(len(public_templates), 77)
        self.assertEqual(len(public_workflow_ids), 51)
        self.assertEqual(public_workflow_ids, coverage_public_workflow_ids)
        self.assertEqual(len(contracts_by_id), len(self.ledger["contracts"]))
        self.assertEqual(set(contracts_by_id), set(manifest_by_id) - public_workflow_ids)
        self.assertFalse(set(contracts_by_id) & public_workflow_ids)

        for workflow_id, contract in contracts_by_id.items():
            manifest = manifest_by_id[workflow_id]
            for field in ("modelType", "mode", "mediaKind", "graphPath", "graphHash", *MANIFEST_BINDING_FIELDS):
                self.assertEqual(contract[field], manifest[field], f"{workflow_id} field {field}")

    def test_authoring_assets_publication_and_evidence_remain_pending(self):
        self.assertEqual(
            self.ledger["boundary"],
            {
                "studioVisible": False,
                "publicTemplate": False,
                "importsComfyGraphs": False,
                "executesComfyNodes": False,
                "downloadsModelsOrMedia": False,
                "generatesMedia": False,
                "allowsZeroAssets": True,
                "comfyMappingMeaning": "semantic_comparison_candidate_only",
                "qualificationMetadataMeaning": "exact_manifest_binding_not_candidate_evidence",
                "maximumEvidenceClaim": "canonical_graph_binding_only",
            },
        )
        forbidden_claim_fields = {
            "verificationStatus",
            "readinessPolicy",
            "qualityReviewStatus",
            "executionEvidence",
            "runtimeEvidence",
            "qualityEvidence",
            "publicTemplateIds",
        }
        manifest_by_id = {row["id"]: row for row in self.manifest["workflows"]}
        for contract in self.ledger["contracts"]:
            self.assertTrue(forbidden_claim_fields.isdisjoint(contract))
            self.assertEqual(contract["publicationState"], "hidden_candidate")
            self.assertEqual(contract["assetState"], "not_generated")
            self.assertEqual(contract["assets"], [])
            self.assertEqual(
                contract["evidenceState"],
                {
                    "maximumClaim": "canonical_graph_binding_only",
                    "execution": "not_claimed",
                    "runtime": "not_claimed",
                    "quality": "not_claimed",
                },
            )
            authoring = contract["authoringState"]
            self.assertEqual(authoring["status"], "authoring_required")
            self.assertEqual(authoring["prompt"], "required_if_applicable")
            self.assertEqual(authoring["defaults"], "required")
            expected_input_state = (
                "required"
                if _needs_input_examples(manifest_by_id[contract["canonicalWorkflowId"]]["requiredInputs"])
                else "not_applicable"
            )
            self.assertEqual(authoring["inputExamples"], expected_input_state)

        # The real checked-in ledger is the zero-asset case: validation must not
        # require fabricated thumbnails, examples, hashes, or media paths.
        self.assertEqual(validate_template_candidate_contract_ledger(self.ledger, root=ROOT), self.ledger)

    def test_every_applicable_comfy_mapping_is_preserved_without_graph_payloads(self):
        candidate_ids = {row["canonicalWorkflowId"] for row in self.ledger["contracts"]}
        expected = {workflow_id: [] for workflow_id in candidate_ids}
        for record_type, field in (("template", "templates"), ("blueprint", "blueprints")):
            for row in self.comfy[field]:
                target = row.get("candidateMapping", {}).get("moDiffWorkflowId")
                if target in candidate_ids:
                    expected[target].append(
                        {
                            "recordType": record_type,
                            "id": row["id"],
                            "mappingStatus": row["mappingStatus"],
                        }
                    )
        for rows in expected.values():
            rows.sort(key=lambda item: (item["recordType"], item["id"]))
        actual = {row["canonicalWorkflowId"]: row["comfyResearchRecords"] for row in self.ledger["contracts"]}
        self.assertEqual(actual, expected)
        self.assertEqual(sum(bool(rows) for rows in actual.values()), 11)
        self.assertEqual(sum(len(rows) for rows in actual.values()), 14)

        forbidden_keys = {"edges", "graph", "links", "nodes", "workflow"}

        def visit(value):
            if isinstance(value, dict):
                self.assertTrue(forbidden_keys.isdisjoint(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.ledger)

    def test_source_provenance_is_bound_to_exact_checked_in_bytes(self):
        source_paths = {
            "workflowManifest": MANIFEST_PATH,
            "upstreamCoverage": COVERAGE_PATH,
            "comfyResearchCatalog": COMFY_PATH,
        }
        for source_name, path in source_paths.items():
            expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(self.ledger["sources"][source_name]["sha256"], expected)

        self.assertEqual(
            self.ledger["sources"]["upstreamCoverage"]["contentHash"],
            self.coverage["contentHash"],
        )
        self.assertEqual(
            self.ledger["sources"]["comfyResearchCatalog"]["revision"],
            self.comfy["source"]["revision"],
        )

    def test_self_consistent_tampering_still_fails_against_current_sources(self):
        mutations = []

        publication = copy.deepcopy(self.ledger)
        publication["contracts"][0]["publicationState"] = "public"
        mutations.append(publication)

        graph = copy.deepcopy(self.ledger)
        graph["contracts"][0]["graphHash"] = "0" * 64
        mutations.append(graph)

        public_overlap = copy.deepcopy(self.ledger)
        public_overlap["contracts"][0]["canonicalWorkflowId"] = self.coverage["publicTemplates"][0][
            "canonicalWorkflowId"
        ]
        mutations.append(public_overlap)

        comfy_omission = copy.deepcopy(self.ledger)
        mapped = next(row for row in comfy_omission["contracts"] if row["comfyResearchRecords"])
        mapped["comfyResearchRecords"].pop()
        mutations.append(comfy_omission)

        source_hash = copy.deepcopy(self.ledger)
        source_hash["sources"]["workflowManifest"]["sha256"] = "sha256:" + "0" * 64
        mutations.append(source_hash)

        for mutated in mutations:
            _reseal(mutated)
            with self.subTest(workflow=mutated["contracts"][0]["canonicalWorkflowId"]):
                with self.assertRaises(TemplateCandidateContractError):
                    validate_template_candidate_contract_ledger(mutated, root=ROOT)

    def test_cli_checks_and_validates_without_external_sources(self):
        for option in ("--check", "--validate"):
            result = subprocess.run(
                [sys.executable, "scripts/generate_template_candidate_contracts.py", option],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
