import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from modiff.comfy_evidence_resolution import (
    ComfyEvidenceResolutionError,
    build_comfy_evidence_resolution_ledger,
    render_comfy_evidence_resolution_ledger,
    validate_comfy_evidence_resolution_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "research" / "comfy-evidence-resolution.v1.json"
COMFY_PATH = ROOT / "data" / "research" / "comfy-research-contracts.v1.json"
CATALOG_PATH = ROOT / "data" / "research" / "comfy-workflow-catalog.v1.json"
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


class ComfyEvidenceResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = _load(LEDGER_PATH)
        cls.comfy = _load(COMFY_PATH)
        cls.manifest = _load(MANIFEST_PATH)
        cls.coverage = _load(COVERAGE_PATH)
        cls.authoring = _load(AUTHORING_PATH)

    def test_checked_in_ledger_is_deterministic_and_exhaustive(self):
        validated = validate_comfy_evidence_resolution_ledger(self.ledger, root=ROOT)
        self.assertEqual(validated, self.ledger)
        self.assertEqual(build_comfy_evidence_resolution_ledger(ROOT), self.ledger)
        self.assertEqual(render_comfy_evidence_resolution_ledger(self.ledger), LEDGER_PATH.read_text())
        self.assertEqual(
            self.ledger["summary"],
            {
                "assetClaims": 0,
                "exactModelSupportClaims": 0,
                "inferredModeCounts": {
                    "character_animate": 1,
                    "character_replace": 2,
                    "control_image": 12,
                    "control_to_video": 5,
                    "data_conversion": 1,
                    "depth_estimation": 6,
                    "edit_image": 7,
                    "edit_video": 1,
                    "face_detection": 2,
                    "first_last_frame_to_video": 2,
                    "frame_interpolation": 1,
                    "graph_utility": 1,
                    "image_adjustment": 7,
                    "image_channels": 1,
                    "image_crop": 2,
                    "image_filter": 7,
                    "image_segmentation": 2,
                    "image_tile": 1,
                    "image_to_3d": 3,
                    "image_to_pose": 4,
                    "image_to_video": 2,
                    "image_upscale": 1,
                    "inpaint": 1,
                    "mask_composite": 1,
                    "multi_image_reference_edit": 3,
                    "outpaint": 1,
                    "reference_to_video": 2,
                    "remove_background": 1,
                    "text_generation": 1,
                    "text_select": 1,
                    "text_to_audio": 2,
                    "text_to_image": 14,
                    "text_to_video": 1,
                    "unresolved": 1,
                    "video_depth_estimation": 3,
                    "video_face_detection": 1,
                    "video_frame_extract": 1,
                    "video_inpaint": 2,
                    "video_segmentation": 2,
                    "video_stitch": 3,
                    "video_to_pose": 3,
                    "video_upscale": 1,
                },
                "newWorkflowClaims": 0,
                "recordsStillUndetermined": 1,
                "recordsWithHiddenAuthoringSpecOption": 35,
                "recordsWithInferredTask": 115,
                "recordsWithPublicTemplateOption": 26,
                "recordsWithRecommendedWorkflow": 61,
                "resolutionCount": 116,
                "resolutionStateCounts": {
                    "explicit_new_task_candidate": 54,
                    "explicit_task_and_family_candidate": 25,
                    "explicit_task_candidate": 36,
                    "source_review_required": 1,
                },
                "sourceUndeterminedCount": 116,
                "taskBoundaryStateCounts": {
                    "existing_family_workflow_candidate": 25,
                    "existing_task_boundary_model_admission_required": 36,
                    "new_task_boundary_required": 54,
                    "task_undetermined": 1,
                },
            },
        )

    def test_resolutions_are_exactly_the_undetermined_comfy_records(self):
        expected = {
            row["contractId"] for row in self.comfy["contracts"] if row["contractState"] == "contract_undetermined"
        }
        actual = {row["contractId"] for row in self.ledger["resolutions"]}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(self.ledger["resolutions"]))

    def test_only_wan_ati_remains_semantically_undetermined(self):
        unresolved = [row for row in self.ledger["resolutions"] if row["inferredMode"] is None]
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["catalogId"], "video_wan_ati")
        self.assertEqual(unresolved[0]["resolutionState"], "source_review_required")
        self.assertEqual(unresolved[0]["taskBoundaryState"], "task_undetermined")
        self.assertIsNone(unresolved[0]["recommendedWorkflow"])
        self.assertIn("task_and_model_evidence_still_insufficient", unresolved[0]["blockers"])

    def test_explicit_image_tools_are_finite_new_task_candidates(self):
        expected_modes = {
            "image_adjustment": 6,
            "image_channels": 1,
            "image_crop": 2,
            "image_filter": 7,
            "image_tile": 1,
            "remove_background": 1,
        }
        counts = {}
        for row in self.ledger["resolutions"]:
            if row["categoryId"] != "blueprints:default:image-tools":
                continue
            counts[row["inferredMode"]] = counts.get(row["inferredMode"], 0) + 1
            self.assertEqual(row["resolutionState"], "explicit_new_task_candidate")
            self.assertEqual(row["taskBoundaryState"], "new_task_boundary_required")
            self.assertEqual(row["representativeWorkflowOptions"], [])
            self.assertIsNone(row["recommendedWorkflow"])
        self.assertEqual(counts, expected_modes)
        showcase = next(
            row for row in self.ledger["resolutions"] if row["catalogId"] == "basic_image_color_adjustment"
        )
        self.assertEqual(showcase["inferredMode"], "image_adjustment")

    def test_every_recommendation_is_a_current_public_or_hidden_workflow(self):
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
                self.assertEqual(workflow["mode"], resolution["inferredMode"])
                self.assertEqual(recommendation["modelType"], workflow["modelType"])
                self.assertEqual(recommendation["modelFamily"], workflow["modelFamily"])
                self.assertEqual(recommendation["publicTemplateIds"], public_by_id[workflow_id])
                authoring_spec_id = recommendation["authoringSpecId"]
                if authoring_spec_id is None:
                    self.assertTrue(public_by_id[workflow_id])
                else:
                    self.assertEqual(authoring_ids[authoring_spec_id], workflow_id)

    def test_metadata_inference_never_becomes_an_exact_or_public_claim(self):
        self.assertEqual(
            self.ledger["boundary"],
            {
                "researchOnly": True,
                "sourceFields": [
                    "catalogId",
                    "title",
                    "categoryId",
                    "presentationMediaHints",
                ],
                "opensComfyGraphs": False,
                "importsComfyGraphs": False,
                "copiesComfyPrompts": False,
                "downloadsModelsOrMedia": False,
                "claimsExactModelOrWorkflowCompatibility": False,
                "publishesTemplates": False,
                "generatesAssets": False,
                "maximumClaim": "explicit_catalog_metadata_task_candidate",
            },
        )
        forbidden_keys = {"edges", "graph", "links", "nodes", "prompt"}

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
            "comfyResearchCatalog": CATALOG_PATH,
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

        mode = copy.deepcopy(self.ledger)
        mode["resolutions"][0]["inferredMode"] = "text_to_image"
        mutations.append(mode)

        exact = copy.deepcopy(self.ledger)
        exact["resolutions"][0]["claims"]["exactModelSupported"] = True
        mutations.append(exact)

        recommendation = copy.deepcopy(self.ledger)
        recommendation["resolutions"][0]["recommendedWorkflow"] = {"canonicalWorkflowId": "unknown"}
        mutations.append(recommendation)

        for mutated in mutations:
            _reseal(mutated)
            with self.assertRaises(ComfyEvidenceResolutionError):
                validate_comfy_evidence_resolution_ledger(mutated, root=ROOT)

    def test_cli_checks_and_validates_offline(self):
        for option in ("--check", "--validate"):
            result = subprocess.run(
                [sys.executable, "scripts/generate_comfy_evidence_resolution.py", option],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
