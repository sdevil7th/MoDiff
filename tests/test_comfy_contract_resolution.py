import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from modiff.comfy_contract_resolution import (
    ComfyContractResolutionError,
    build_comfy_contract_resolution_ledger,
    render_comfy_contract_resolution_ledger,
    validate_comfy_contract_resolution_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "research" / "comfy-contract-resolution.v1.json"
COMFY_PATH = ROOT / "data" / "research" / "comfy-research-contracts.v1.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"
COVERAGE_PATH = ROOT / "data" / "upstream-coverage.v1.json"
AUTHORING_PATH = ROOT / "data" / "template-authoring-specs.v1.json"


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


class ComfyContractResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = _load(LEDGER_PATH)
        cls.comfy = _load(COMFY_PATH)
        cls.manifest = _load(MANIFEST_PATH)
        cls.coverage = _load(COVERAGE_PATH)
        cls.authoring = _load(AUTHORING_PATH)

    def test_checked_in_ledger_is_deterministic_and_exhaustive(self):
        validated = validate_comfy_contract_resolution_ledger(self.ledger, root=ROOT)
        self.assertEqual(validated, self.ledger)
        self.assertEqual(build_comfy_contract_resolution_ledger(ROOT), self.ledger)
        self.assertEqual(render_comfy_contract_resolution_ledger(self.ledger), LEDGER_PATH.read_text())
        self.assertEqual(
            self.ledger["summary"]["resolutionStateCounts"],
            {
                "existing_family_workflow_candidate": 57,
                "existing_task_boundary_model_admission_required": 59,
                "new_task_boundary_required": 22,
            },
        )
        self.assertEqual(self.ledger["summary"]["sourceProposalCount"], 138)
        self.assertEqual(self.ledger["summary"]["resolutionCount"], 138)
        self.assertEqual(self.ledger["summary"]["recordsWithRecommendedWorkflow"], 116)
        self.assertEqual(self.ledger["summary"]["recordsWithPublicTemplateOption"], 25)
        self.assertEqual(self.ledger["summary"]["recordsWithHiddenAuthoringSpecOption"], 91)

    def test_resolutions_are_exactly_the_unwritten_comfy_proposals(self):
        expected = {
            row["contractId"] for row in self.comfy["contracts"] if row["contractState"] == "new_contract_not_authored"
        }
        actual = {row["contractId"] for row in self.ledger["resolutions"]}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(self.ledger["resolutions"]))

    def test_new_task_boundaries_are_finite_and_have_no_recommendation(self):
        expected_mode_counts = {
            "audio_to_video": 3,
            "edit_audio": 1,
            "edit_video": 2,
            "first_last_frame_to_video": 7,
            "frame_interpolation": 1,
            "image_to_3d": 7,
            "remove_background": 1,
        }
        actual = {}
        for resolution in self.ledger["resolutions"]:
            if resolution["resolutionState"] != "new_task_boundary_required":
                continue
            mode = resolution["selectedCandidateMode"]
            actual[mode] = actual.get(mode, 0) + 1
            self.assertEqual(resolution["currentTaskBoundaryOptionCount"], 0)
            self.assertEqual(resolution["representativeWorkflowOptions"], [])
            self.assertIsNone(resolution["recommendedWorkflow"])
            self.assertIn("new_bounded_modiff_task_contract_required", resolution["blockers"])
        self.assertEqual(actual, expected_mode_counts)

    def test_image_upscale_proposals_reuse_the_bounded_install_free_task(self):
        rows = [
            row
            for row in self.ledger["resolutions"]
            if row["selectedCandidateMode"] == "image_upscale"
        ]
        self.assertEqual(len(rows), 4)
        for row in rows:
            with self.subTest(contract=row["contractId"]):
                self.assertEqual(
                    row["resolutionState"],
                    "existing_task_boundary_model_admission_required",
                )
                self.assertEqual(row["currentTaskBoundaryOptionCount"], 1)
                self.assertEqual(
                    row["recommendedWorkflow"]["canonicalWorkflowId"],
                    "BuiltinImageOperation:image_upscale",
                )

    def test_seedvr_video_upscale_reuses_only_the_task_boundary(self):
        row = next(
            row
            for row in self.ledger["resolutions"]
            if row["catalogId"] == "utility_seedvr2_3b_int8_upscale_video"
        )
        self.assertEqual(row["selectedCandidateMode"], "video_upscale")
        self.assertEqual(row["resolutionState"], "existing_task_boundary_model_admission_required")
        self.assertEqual(row["currentTaskBoundaryOptionCount"], 1)
        self.assertEqual(
            row["recommendedWorkflow"]["canonicalWorkflowId"],
            "SpandrelVideoUpscale:video_upscale",
        )
        self.assertFalse(row["claims"]["exactCatalogCheckpointSupported"])
        self.assertFalse(row["claims"]["recommendedWorkflowEquivalent"])
        self.assertTrue(row["exactCatalogModelReproductionRequiresAdmission"])

    def test_text_outputs_reuse_the_bounded_json_generation_boundary(self):
        rows = [row for row in self.ledger["resolutions"] if row["selectedCandidateMode"] == "text_generation"]
        self.assertEqual(len(rows), 4)
        for row in rows:
            with self.subTest(contract=row["contractId"]):
                self.assertEqual(
                    row["resolutionState"],
                    "existing_task_boundary_model_admission_required",
                )
                self.assertEqual(row["candidateOutputMediaKinds"], ["text"])
                self.assertEqual(row["currentTaskBoundaryOptionCount"], 2)
                self.assertIsNotNone(row["recommendedWorkflow"])

    def test_every_recommendation_is_a_current_workflow_with_public_or_hidden_authoring_state(self):
        workflow_by_id = {row["id"]: row for row in self.manifest["workflows"]}
        public_by_id = {row["id"]: row["publicTemplateIds"] for row in self.coverage["canonicalWorkflows"]}
        authoring_ids = {row["id"]: row["canonicalWorkflowId"] for row in self.authoring["specifications"]}
        for resolution in self.ledger["resolutions"]:
            recommendation = resolution["recommendedWorkflow"]
            if recommendation is None:
                continue
            workflow_id = recommendation["canonicalWorkflowId"]
            workflow = workflow_by_id[workflow_id]
            with self.subTest(contract=resolution["contractId"]):
                self.assertEqual(workflow["mode"], resolution["selectedCandidateMode"])
                self.assertEqual(recommendation["modelType"], workflow["modelType"])
                self.assertEqual(recommendation["modelFamily"], workflow["modelFamily"])
                self.assertEqual(recommendation["publicTemplateIds"], public_by_id[workflow_id])
                authoring_spec_id = recommendation["authoringSpecId"]
                if authoring_spec_id is None:
                    self.assertTrue(public_by_id[workflow_id])
                else:
                    self.assertEqual(authoring_ids[authoring_spec_id], workflow_id)

    def test_family_candidates_are_semantic_only_and_never_exact_checkpoint_claims(self):
        for resolution in self.ledger["resolutions"]:
            if resolution["resolutionState"] != "existing_family_workflow_candidate":
                continue
            recommendation = resolution["recommendedWorkflow"]
            self.assertIsNotNone(recommendation)
            self.assertIn(recommendation["modelFamily"], resolution["recognizedMoDiffFamilies"])
            self.assertTrue(resolution["exactCatalogModelReproductionRequiresAdmission"])
            self.assertIn("exact_checkpoint_not_proven", resolution["mappingMeaning"])
            self.assertFalse(resolution["claims"]["exactCatalogCheckpointSupported"])
            self.assertFalse(resolution["claims"]["recommendedWorkflowEquivalent"])

    def test_execution_publication_asset_and_comfy_copy_boundaries_remain_closed(self):
        self.assertEqual(
            self.ledger["boundary"],
            {
                "researchOnly": True,
                "importsComfyGraphs": False,
                "executesComfyNodes": False,
                "copiesComfyPrompts": False,
                "downloadsModelsOrMedia": False,
                "claimsExactCatalogCheckpointCompatibility": False,
                "claimsMoDiffWorkflowSupportFromCatalogMetadata": False,
                "publishesTemplates": False,
                "generatesAssets": False,
                "maximumClaim": "semantic_task_boundary_resolution",
            },
        )
        forbidden_keys = {"edges", "graph", "links", "nodes", "prompt", "workflow"}

        def visit(value):
            if isinstance(value, dict):
                self.assertTrue(forbidden_keys.isdisjoint(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.ledger)
        for resolution in self.ledger["resolutions"]:
            self.assertTrue(all(value is False for value in resolution["claims"].values()))

    def test_source_hashes_bind_all_current_ledgers(self):
        paths = {
            "comfyResearchContracts": COMFY_PATH,
            "workflowManifest": MANIFEST_PATH,
            "upstreamCoverage": COVERAGE_PATH,
            "templateAuthoringSpecs": AUTHORING_PATH,
        }
        for source, path in paths.items():
            self.assertEqual(
                self.ledger["sources"][source]["sha256"],
                "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_self_consistent_tampering_fails_against_sources(self):
        mutations = []

        state = copy.deepcopy(self.ledger)
        state["resolutions"][0]["resolutionState"] = "existing_family_workflow_candidate"
        mutations.append(state)

        exact = copy.deepcopy(self.ledger)
        exact["resolutions"][0]["claims"]["exactCatalogCheckpointSupported"] = True
        mutations.append(exact)

        recommendation = copy.deepcopy(self.ledger)
        recommendation["resolutions"][0]["recommendedWorkflow"] = {"canonicalWorkflowId": "unknown"}
        mutations.append(recommendation)

        for mutated in mutations:
            _reseal(mutated)
            with self.assertRaises(ComfyContractResolutionError):
                validate_comfy_contract_resolution_ledger(mutated, root=ROOT)

    def test_cli_checks_and_validates_offline(self):
        for option in ("--check", "--validate"):
            result = subprocess.run(
                [sys.executable, "scripts/generate_comfy_contract_resolution.py", option],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
